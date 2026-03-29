"""
Instruction IR and VLIW scheduling for perf_takehome.KernelBuilder.

Schedules within regions separated by flow.pause (harness expects machine.run() segments).

``SCHEDULE_MODE``:
  - ``"identity"`` — emit bundles unchanged (same cycle count as an unscheduled build).
  - ``"list"`` — topological list schedule with greedy per-cycle packing under ``SLOT_LIMITS``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import hashlib
import pickle
from problem import SLOT_LIMITS, VLEN, Instruction

Engine = str


def _rng(addr: int, length: int) -> tuple[int, int]:
    return (addr, length)


def _addrs_in_range(r: tuple[int, int]) -> range:
    a, ln = r
    return range(a, a + ln)


def ranges_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    a0, al = a
    b0, bl = b
    return not (a0 + al <= b0 or b0 + bl <= a0)


def slot_scratch_rw(
    engine: Engine, slot: tuple
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """
    Return (read_ranges, write_ranges) for one slot.
    Ranges are [start, start+length) half-open in scratch words.
    """
    reads: list[tuple[int, int]] = []
    writes: list[tuple[int, int]] = []

    if engine == "alu":
        op, dest, a1, a2 = slot
        reads.extend([_rng(a1, 1), _rng(a2, 1)])
        writes.append(_rng(dest, 1))
    elif engine == "valu":
        match slot:
            case ("vbroadcast", dest, src):
                reads.append(_rng(src, 1))
                writes.append(_rng(dest, VLEN))
            case ("multiply_add", dest, a, b, c):
                reads.extend([_rng(a, VLEN), _rng(b, VLEN), _rng(c, VLEN)])
                writes.append(_rng(dest, VLEN))
            case (_, dest, a1, a2):
                reads.extend([_rng(a1, VLEN), _rng(a2, VLEN)])
                writes.append(_rng(dest, VLEN))
    elif engine == "load":
        match slot:
            case ("const", dest, _val):
                writes.append(_rng(dest, 1))
            case ("load", dest, addr):
                reads.append(_rng(addr, 1))
                writes.append(_rng(dest, 1))
            case ("vload", dest, addr):
                reads.append(_rng(addr, 1))
                writes.append(_rng(dest, VLEN))
            case ("load_offset", dest, addr, offset):
                reads.append(_rng(addr + offset, 1))
                writes.append(_rng(dest + offset, 1))
    elif engine == "store":
        match slot:
            case ("store", addr, src):
                reads.extend([_rng(addr, 1), _rng(src, 1)])
            case ("vstore", addr, src):
                reads.append(_rng(addr, 1))
                reads.append(_rng(src, VLEN))
    elif engine == "flow":
        match slot:
            case ("select", dest, cond, a, b):
                reads.extend([_rng(cond, 1), _rng(a, 1), _rng(b, 1)])
                writes.append(_rng(dest, 1))
            case ("add_imm", dest, a, _imm):
                reads.append(_rng(a, 1))
                writes.append(_rng(dest, 1))
            case ("vselect", dest, cond, a, b):
                reads.extend([_rng(cond, VLEN), _rng(a, VLEN), _rng(b, VLEN)])
                writes.append(_rng(dest, VLEN))
            case ("trace_write", val):
                reads.append(_rng(val, 1))
            case ("cond_jump", cond, addr):
                reads.extend([_rng(cond, 1), _rng(addr, 1)])
            case ("cond_jump_rel", cond, _offset):
                reads.append(_rng(cond, 1))
            case ("jump_indirect", addr):
                reads.append(_rng(addr, 1))
            case ("coreid", dest):
                writes.append(_rng(dest, 1))
            case ("pause",) | ("halt",) | ("jump", _):
                pass
    elif engine == "debug":
        match slot:
            case ("compare", loc, _key):
                reads.append(_rng(loc, 1))
            case ("vcompare", loc, _keys):
                reads.append(_rng(loc, VLEN))
            case ("comment", _):
                pass

    return reads, writes


@dataclass
class IROp:
    engine: Engine
    slot: tuple
    seq: int
    reads: list[tuple[int, int]] = field(default_factory=list)
    writes: list[tuple[int, int]] = field(default_factory=list)


def instruction_to_op_list(instr: Instruction) -> list[tuple[Engine, tuple]]:
    """Flatten in the same order as Machine.step iterates instr.items()."""
    out: list[tuple[Engine, tuple]] = []
    for eng, slots in instr.items():
        for slot in slots:
            out.append((eng, slot))
    return out


def make_irop(seq: int, engine: Engine, slot: tuple) -> IROp:
    reads, writes = slot_scratch_rw(engine, slot)
    return IROp(
        engine=engine,
        slot=slot,
        seq=seq,
        reads=reads,
        writes=writes,
    )


def is_pause_instruction(instr: Instruction) -> bool:
    if "flow" not in instr:
        return False
    return any(s == ("pause",) for s in instr["flow"])


def split_pause_regions(instrs: list[Instruction]) -> list[list[Instruction]]:
    """Each region ends with a pause instruction (same as machine.run segments)."""
    regions: list[list[Instruction]] = []
    cur: list[Instruction] = []
    for instr in instrs:
        cur.append(instr)
        if is_pause_instruction(instr):
            regions.append(cur)
            cur = []
    if cur:
        regions.append(cur)
    return regions


def pack_ops_to_instruction(ops: list[tuple[Engine, tuple]]) -> Instruction:
    d: dict[str, list] = defaultdict(list)
    for eng, slot in ops:
        d[eng].append(slot)
    return dict(d)


def emit_instruction_list(instrs: list[Instruction]) -> list[Instruction]:
    return instrs


def instrs_fingerprint(instrs: list[Instruction]) -> str:
    return hashlib.sha256(pickle.dumps(instrs, protocol=4)).hexdigest()


def build_dependency_edges(ops: list[IROp]) -> list[set[int]]:
    """
    For each op index i, preds[i] = set of j that must execute in a strictly earlier cycle.
    Rules: RAW/WAW/WAR on scratch words (memory ordering follows scratch pointer deps).
    """
    n = len(ops)
    preds: list[set[int]] = [set() for _ in range(n)]

    last_write: dict[int, int] = {}
    last_read: dict[int, int] = {}

    def add_pred(i: int, j: int) -> None:
        if j != i:
            preds[i].add(j)

    for i, op in enumerate(ops):
        for r in op.reads:
            for addr in _addrs_in_range(r):
                if addr in last_write:
                    add_pred(i, last_write[addr])
                last_read[addr] = i

        for w in op.writes:
            for addr in _addrs_in_range(w):
                if addr in last_write:
                    add_pred(i, last_write[addr])
                if addr in last_read:
                    add_pred(i, last_read[addr])

        for w in op.writes:
            for addr in _addrs_in_range(w):
                last_write[addr] = i
                if addr in last_read:
                    del last_read[addr]

    return preds


def check_acyclic(preds: list[set[int]], n: int) -> None:
    """Raise if cyclic (should not happen for valid kernels)."""
    indeg = [len(preds[i]) for i in range(n)]
    from collections import deque

    q = deque([i for i in range(n) if indeg[i] == 0])
    seen = 0
    while q:
        i = q.popleft()
        seen += 1
        for j in range(n):
            if i in preds[j]:
                indeg[j] -= 1
                if indeg[j] == 0:
                    q.append(j)
    if seen != n:
        raise RuntimeError("scheduler: dependency cycle detected")


# Default: list scheduling. Set to "identity" to skip reordering (for regression checks).
SCHEDULE_MODE = "list"


def schedule_region_identity(region: list[Instruction]) -> list[Instruction]:
    """Layer 1/2: preserve exact bundles."""
    return list(region)


def _writes_conflict(
    writes_a: list[tuple[int, int]], writes_b: list[tuple[int, int]]
) -> bool:
    for wa in writes_a:
        for wb in writes_b:
            if ranges_overlap(wa, wb):
                return True
    return False


def _can_pack_together(
    bucket: list[IROp],
    op: IROp,
    op_idx: int,
    preds: list[set[int]],
) -> bool:
    counts: dict[str, int] = defaultdict(int)
    for o in bucket:
        counts[o.engine] += 1
    ne = op.engine
    if counts[ne] + 1 > SLOT_LIMITS[ne]:
        return False
    for o in bucket:
        j = o.seq
        i = op_idx
        # Same bundle: all reads at start, writes at end — cannot mix pred/succ on scratch.
        if j in preds[i] or i in preds[j]:
            return False
        if _writes_conflict(o.writes, op.writes):
            return False
    return True


def schedule_region_list(region: list[Instruction]) -> list[Instruction]:
    """
    List schedule ops in region (except final pause is pinned last).
    """
    if not region:
        return region

    if is_pause_instruction(region[-1]):
        head, tail = region[:-1], region[-1:]
    else:
        head, tail = region, []

    flat: list[tuple[Engine, tuple]] = []
    for instr in head:
        flat.extend(instruction_to_op_list(instr))

    ops = [make_irop(i, e, s) for i, (e, s) in enumerate(flat)]
    preds = build_dependency_edges(ops)
    n = len(ops)
    if n > 0:
        check_acyclic(preds, n)

    succs: list[list[int]] = [[] for _ in range(n)]
    for i in range(n):
        for j in preds[i]:
            succs[j].append(i)

    # Critical-path lengths for tie-breaking (longer path first).
    outdeg = [len(succs[i]) for i in range(n)]
    rev_topo: list[int] = []
    from collections import deque

    q = deque([i for i in range(n) if outdeg[i] == 0])
    while q:
        i = q.popleft()
        rev_topo.append(i)
        for p in preds[i]:
            outdeg[p] -= 1
            if outdeg[p] == 0:
                q.append(p)
    cplen = [1] * n
    for i in rev_topo:
        if succs[i]:
            cplen[i] = 1 + max(cplen[s] for s in succs[i])

    # Lower = schedule sooner. Prefer filling bottleneck engines (few slots per cycle)
    # before ALU, so startup does not spend cycles with idle load/valu while ALU packs.
    ENGINE_SCHED_ORDER = {
        "load": 0,
        "valu": 1,
        "flow": 2,
        "store": 3,
        "alu": 4,
        "debug": 5,
    }
    indeg = [len(preds[i]) for i in range(n)]
    ready: set[int] = {i for i in range(n) if indeg[i] == 0}
    unscheduled: set[int] = set(range(n))
    bundles: list[Instruction] = []

    def _pick_best(cand: list[int]) -> int:
        return min(
            cand,
            key=lambda i: (
                ops[i].seq,
                ENGINE_SCHED_ORDER.get(ops[i].engine, 99),
                -cplen[i],
            ),
        )

    while unscheduled:
        bundle_idxs: list[int] = []
        bundle_ops: list[IROp] = []

        while True:
            candidates = [
                i for i in ready if _can_pack_together(bundle_ops, ops[i], i, preds)
            ]
            if not candidates:
                break
            # Among ready ops: prefer load/valu (tight slot limits), then longest critical path.
            best = _pick_best(candidates)
            bundle_idxs.append(best)
            bundle_ops.append(ops[best])
            ready.remove(best)

        if not bundle_idxs:
            # Fallback: schedule one ready op to guarantee progress.
            if not ready:
                raise RuntimeError("scheduler: no ready op but work remains")
            best = _pick_best(list(ready))
            bundle_idxs = [best]
            bundle_ops = [ops[best]]
            ready.remove(best)

        bundles.append(
            pack_ops_to_instruction([(ops[i].engine, ops[i].slot) for i in bundle_idxs])
        )

        newly_ready: list[int] = []
        for i in bundle_idxs:
            unscheduled.remove(i)
            for s in succs[i]:
                indeg[s] -= 1
                if indeg[s] == 0 and s in unscheduled:
                    newly_ready.append(s)
        for i in newly_ready:
            ready.add(i)

    return bundles + tail


def apply_schedule(instrs: list[Instruction], mode: str) -> list[Instruction]:
    """
    mode: 'identity' | 'list'
    """
    regions = split_pause_regions(instrs)
    out: list[Instruction] = []
    for region in regions:
        if mode == "identity":
            out.extend(schedule_region_identity(region))
        elif mode == "list":
            out.extend(schedule_region_list(region))
        else:
            raise ValueError(f"unknown schedule mode {mode}")
    return out


def verify_slot_limits(instrs: list[Instruction]) -> None:
    for pc, instr in enumerate(instrs):
        for name, slots in instr.items():
            if name == "debug":
                continue
            lim = SLOT_LIMITS.get(name, 0)
            if len(slots) > lim:
                raise AssertionError(
                    f"pc={pc} engine={name} slots={len(slots)} > limit {lim}"
                )
