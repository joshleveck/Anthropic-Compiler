"""
Compile each `compiler/test/t*.c` with the Rust compiler and run the resulting
program on `problem.Machine`, then assert final memory matches a Python reference.

Usage (from repo root):
    python compiler_harness.py
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any, Callable

from problem import Machine, VLEN, myhash
from load_from_rust import load_kernel_from_rust

REPO_ROOT = Path(__file__).resolve().parent
COMPILER_DIR = REPO_ROOT / "compiler"
TEST_DIR = COMPILER_DIR / "test"
OUT_JSON = COMPILER_DIR / "output" / "harness_program.json"

# Header pointers (same convention as build_mem_image in problem.py)
P_ROUNDS = 0
P_INP_VALUES_P = 6
P_INP_INDICES_P = 5
P_FOREST_VALUES_P = 4


def u32(x: int) -> int:
    return x % (2**32)


_COMPILER_EXE: Path | None = None


def compiler_exe() -> Path:
    """Path to the built `compiler` binary (runs `cargo build -q` once if missing)."""
    global _COMPILER_EXE
    if _COMPILER_EXE is not None:
        return _COMPILER_EXE
    name = "compiler.exe" if sys.platform == "win32" else "compiler"
    exe = COMPILER_DIR / "target" / "debug" / name
    if not exe.is_file():
        subprocess.run(
            ["cargo", "build", "-q"],
            cwd=COMPILER_DIR,
            check=True,
        )
    _COMPILER_EXE = exe
    return exe


def compile_c_to_json(c_path: Path, json_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [str(compiler_exe()), str(c_path), str(json_path)],
        cwd=COMPILER_DIR,
        check=True,
    )


def run_program(
    mem: list[int], program_json: Path, max_cycles: int = 500_000
) -> list[int]:
    """Run the kernel; return final memory.

    `problem.Machine` copies `mem_dump`, so writes are only visible on the returned list.
    """
    kernel = load_kernel_from_rust(program_json)
    m = Machine(
        mem_dump=list(mem),
        program=kernel.instrs,
        debug_info=kernel.debug_info(),
        trace=False,
    )
    m.run(max_cycles=max_cycles)
    return m.mem


def max_engine_slots(
    instrs: list[dict[str, list[tuple[Any, ...]]]], engine: str
) -> int:
    return max((len(b.get(engine, [])) for b in instrs), default=0)


def print_bundle_metrics(c_file_name: str) -> None:
    c_file = TEST_DIR / c_file_name
    compile_c_to_json(c_file, OUT_JSON)
    kernel = load_kernel_from_rust(OUT_JSON)
    instrs = kernel.instrs
    packed = sum(1 for b in instrs if sum(len(v) for v in b.values()) > 1)
    print(
        f"[metrics] {c_file_name}: bundles={len(instrs)} packed_bundles={packed} "
        f"max_alu={max_engine_slots(instrs, 'alu')} max_load={max_engine_slots(instrs, 'load')} "
        f"max_store={max_engine_slots(instrs, 'store')} max_valu={max_engine_slots(instrs, 'valu')}"
    )


# --- Reference simulations -------------------------------------------------


def vhash_lane_vec(v: list[int]) -> list[int]:
    return [myhash(x) for x in v]


def simulate_t09(
    mem: list[int],
    *,
    forest_values_p: int,
    idx_p: int,
    val_p: int,
    rounds: int = 2,
    forest_height: int = 3,
) -> tuple[list[int], list[int]]:
    idx = [mem[idx_p + i] for i in range(VLEN)]
    val = [mem[val_p + i] for i in range(VLEN)]
    root = mem[forest_values_p + 0]
    vroot = [u32(root)] * VLEN

    for r in range(rounds):
        if r == 0:
            node = vroot
        else:
            node = [mem[forest_values_p + idx[i]] for i in range(VLEN)]
        val = vhash_lane_vec([u32(val[i] ^ node[i]) for i in range(VLEN)])
        idx = [u32(2 * idx[i] + 1) for i in range(VLEN)]
        idx = [u32(idx[i] + (val[i] % 2)) for i in range(VLEN)]
        if r == forest_height:
            idx = [0] * VLEN
    return idx, val


def assert_mem_slice(
    mem: list[int], start: int, expected: list[int], label: str
) -> None:
    got = mem[start : start + len(expected)]
    assert got == expected, f"{label}: got {got!r} expected {expected!r}"


def case_t01() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    initial = 100
    mem[data_ptr] = initial
    snapshot = list(mem)
    final = run_program(mem, OUT_JSON)
    assert final[:8] == snapshot[:8]  # header untouched
    assert final[data_ptr] == u32(initial + 7), (final[data_ptr], u32(initial + 7))


def case_t02() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    init = [u32(1000 + i) for i in range(VLEN)]
    mem[data_ptr : data_ptr + VLEN] = init
    final = run_program(mem, OUT_JSON)
    assert_mem_slice(final, data_ptr, init, "t02 vector round-trip (simulator mem)")


def case_t03() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    init = [u32(50 + i * 11) for i in range(VLEN)]
    mem[data_ptr : data_ptr + VLEN] = init
    final = run_program(mem, OUT_JSON)
    exp = [u32(u32(v + 123) ^ 123) for v in init]
    assert_mem_slice(final, data_ptr, exp, "t03 vbroadcast/valu")


def case_t04() -> None:
    idx_ptr = 120
    val_ptr = 200
    mem = [0] * 2048
    mem[P_INP_INDICES_P] = idx_ptr
    mem[P_INP_VALUES_P] = val_ptr
    idx = [0, 1, 2, 3, 4, 5, 6, 7]
    val = [u32(300 + i) for i in range(VLEN)]
    mem[idx_ptr : idx_ptr + VLEN] = idx
    mem[val_ptr : val_ptr + VLEN] = val
    final = run_program(mem, OUT_JSON)
    exp = [1 if (idx[i] % 2) != 0 else val[i] for i in range(VLEN)]
    assert_mem_slice(final, val_ptr, exp, "t04 vselect")


def case_t05() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    init = [u32(999 + i * 100_000) for i in range(VLEN)]
    mem[data_ptr : data_ptr + VLEN] = init
    final = run_program(mem, OUT_JSON)
    exp = [myhash(v) for v in init]
    assert_mem_slice(final, data_ptr, exp, "t05 vhash")


def case_t06() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    init = [u32(i + 1) for i in range(VLEN)]
    mem[data_ptr : data_ptr + VLEN] = init
    final = run_program(mem, OUT_JSON)
    # r=0:+10, r=1:+20, r=2,3:+30 each
    exp = [u32(v + 10 + 20 + 30 + 30) for v in init]
    assert_mem_slice(final, data_ptr, exp, "t06 compile-time if / unroll")


def case_t07() -> None:
    data_ptr = 200
    for rounds, add in [(0, 111), (3, 222)]:
        mem = [0] * 2048
        mem[P_ROUNDS] = rounds
        mem[P_INP_VALUES_P] = data_ptr
        mem[data_ptr] = 55
        final = run_program(mem, OUT_JSON)
        assert final[data_ptr] == u32(55 + add), (rounds, final[data_ptr])


def case_t08() -> None:
    idx_ptr = 120
    val_ptr = 200
    mem = [0] * 2048
    mem[P_INP_INDICES_P] = idx_ptr
    mem[P_INP_VALUES_P] = val_ptr
    idx0 = [u32(10 + i) for i in range(VLEN)]
    val0 = [u32(1000 + i) for i in range(VLEN)]
    mem[idx_ptr : idx_ptr + VLEN] = idx0
    mem[val_ptr : val_ptr + VLEN] = val0
    final = run_program(mem, OUT_JSON)
    exp_idx = [u32(idx0[i] + val0[i]) for i in range(VLEN)]
    assert_mem_slice(final, idx_ptr, exp_idx, "t08 idx after add")
    assert final[val_ptr] == exp_idx[0], (final[val_ptr], exp_idx[0])


def case_t09() -> None:
    header = 7
    n_nodes = 16
    batch = VLEN
    forest_values_p = header
    inp_indices_p = forest_values_p + n_nodes
    inp_values_p = inp_indices_p + batch
    extra_room = inp_values_p + batch
    size = extra_room + 64
    mem = [0] * size
    mem[P_FOREST_VALUES_P] = forest_values_p
    mem[P_INP_INDICES_P] = inp_indices_p
    mem[P_INP_VALUES_P] = inp_values_p
    # Tree node values (only first few used by idx)
    for i in range(n_nodes):
        mem[forest_values_p + i] = u32(0x1000 + i * 17)
    indices = [1, 2, 3, 4, 5, 6, 7, 8]
    values = [u32(0x2000 + i) for i in range(batch)]
    mem[inp_indices_p : inp_indices_p + batch] = indices
    mem[inp_values_p : inp_values_p + batch] = values

    exp_idx, exp_val = simulate_t09(
        mem,
        forest_values_p=forest_values_p,
        idx_p=inp_indices_p,
        val_p=inp_values_p,
        rounds=2,
        forest_height=3,
    )
    final = run_program(mem, OUT_JSON)
    assert_mem_slice(final, inp_indices_p, exp_idx, "t09 idx")
    assert_mem_slice(final, inp_values_p, exp_val, "t09 val")


def case_t10() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    init = [u32(100 + i * 7) for i in range(VLEN)]
    mem[data_ptr : data_ptr + VLEN] = init
    final = run_program(mem, OUT_JSON)
    exp = [u32(v % 2) for v in init]
    assert_mem_slice(final, data_ptr, exp, "t10 vector mod2")


def case_t11() -> None:
    idx_ptr = 120
    val_ptr = 200
    mem = [0] * 2048
    mem[P_INP_INDICES_P] = idx_ptr
    mem[P_INP_VALUES_P] = val_ptr
    idx0 = [u32(3 + i) for i in range(VLEN)]
    val0 = [u32(50 + i * 11) for i in range(VLEN)]
    mem[idx_ptr : idx_ptr + VLEN] = idx0
    mem[val_ptr : val_ptr + VLEN] = val0
    final = run_program(mem, OUT_JSON)
    exp_idx = [u32(u32(2 * idx0[i] + 1) + u32(val0[i] % 2)) for i in range(VLEN)]
    assert_mem_slice(final, idx_ptr, exp_idx, "t11 idx walk")
    assert_mem_slice(final, val_ptr, val0, "t11 val unchanged")


def case_t12() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    init = [u32(5 + i) for i in range(VLEN)]
    mem[data_ptr : data_ptr + VLEN] = init
    final = run_program(mem, OUT_JSON)
    # r=0:+10, r=1:+100, r=2:+1000
    exp = [u32(v + 10 + 100 + 1000) for v in init]
    assert_mem_slice(final, data_ptr, exp, "t12 else-if rounds")


def case_t13() -> None:
    idx_ptr = 100
    val_ptr = 200
    mem = [0] * 2048
    mem[P_INP_INDICES_P] = idx_ptr
    mem[P_INP_VALUES_P] = val_ptr
    idx_a = [u32(1000 + i) for i in range(VLEN)]
    idx_b = [u32(2000 + i) for i in range(VLEN)]
    val_a = [u32(10 + i) for i in range(VLEN)]
    val_b = [u32(20 + i) for i in range(VLEN)]
    mem[idx_ptr : idx_ptr + VLEN] = idx_a
    mem[idx_ptr + 8 : idx_ptr + 16] = idx_b
    mem[val_ptr : val_ptr + VLEN] = val_a
    mem[val_ptr + 8 : val_ptr + 16] = val_b
    final = run_program(mem, OUT_JSON)
    exp_val_a = [u32(v + 1) for v in val_a]
    exp_val_b = [u32(v + 1) for v in val_b]
    assert_mem_slice(final, idx_ptr, idx_a, "t13 idx batch0")
    assert_mem_slice(final, idx_ptr + 8, idx_b, "t13 idx batch1")
    assert_mem_slice(final, val_ptr, exp_val_a, "t13 val batch0")
    assert_mem_slice(final, val_ptr + 8, exp_val_b, "t13 val batch1")


def case_t14() -> None:
    idx_ptr = 120
    val_ptr = 200
    mem = [0] * 2048
    mem[P_INP_INDICES_P] = idx_ptr
    mem[P_INP_VALUES_P] = val_ptr
    idx0 = [u32(1 + i) for i in range(VLEN)]
    val0 = [u32(100 + i) for i in range(VLEN)]
    mem[idx_ptr : idx_ptr + VLEN] = idx0
    mem[val_ptr : val_ptr + VLEN] = val0
    final = run_program(mem, OUT_JSON)
    exp_val = [u32(v + 3 + 3) for v in val0]
    assert_mem_slice(final, idx_ptr, idx0, "t14 idx")
    assert_mem_slice(final, val_ptr, exp_val, "t14 val +6")


def case_t15() -> None:
    idx_ptr = 120
    val_ptr = 200
    mem = [0] * 2048
    mem[P_INP_INDICES_P] = idx_ptr
    mem[P_INP_VALUES_P] = val_ptr
    idx0 = [u32(100 + i) for i in range(VLEN)]
    val0 = [u32(1000 + i) for i in range(VLEN)]
    mem[idx_ptr : idx_ptr + VLEN] = idx0
    mem[val_ptr : val_ptr + VLEN] = val0
    final = run_program(mem, OUT_JSON)
    # r=0,1,2: +1 each; r=3 FOREST_HEIGHT: vbroadcast(0)
    assert_mem_slice(final, idx_ptr, [0] * VLEN, "t15 idx wrap to 0")
    assert_mem_slice(final, val_ptr, val0, "t15 val unchanged")


def case_t16() -> None:
    forest_base = 60
    idx_ptr = 120
    val_ptr = 200
    mem = [0] * 2048
    mem[P_FOREST_VALUES_P] = forest_base
    mem[P_INP_INDICES_P] = idx_ptr
    mem[P_INP_VALUES_P] = val_ptr
    mem[forest_base + 1] = u32(0x1111)
    mem[forest_base + 2] = u32(0x2222)
    idx0 = [u32(1 + i) for i in range(VLEN)]
    mem[idx_ptr : idx_ptr + VLEN] = idx0
    mem[val_ptr : val_ptr + VLEN] = [0] * VLEN
    final = run_program(mem, OUT_JSON)
    exp = []
    for i in range(VLEN):
        t = u32(idx0[i] - 1)
        exp.append(u32(0x2222) if t != 0 else u32(0x1111))
    assert_mem_slice(final, val_ptr, exp, "t16 vselect idx-1")


def case_t18() -> None:
    data_ptr = 200
    for rounds, add in [(0, 111), (3, 222)]:
        mem = [0] * 2048
        mem[P_ROUNDS] = rounds
        mem[P_INP_VALUES_P] = data_ptr
        mem[data_ptr] = 55
        final = run_program(mem, OUT_JSON)
        assert final[data_ptr] == u32(55 + add), (rounds, final[data_ptr])


def case_t19() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    mem[data_ptr] = 7
    mem[data_ptr + 1] = 11
    kernel = load_kernel_from_rust(OUT_JSON)
    assert max_engine_slots(kernel.instrs, "alu") >= 2, "t19 expected ALU slot packing"
    final = run_program(mem, OUT_JSON)
    assert final[data_ptr] == u32(17), final[data_ptr]
    assert final[data_ptr + 1] == u32(31), final[data_ptr + 1]


def case_t20() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    mem[data_ptr] = 77
    final = run_program(mem, OUT_JSON)
    assert final[data_ptr] == u32(82), final[data_ptr]
    assert final[data_ptr + 1] == u32(82), final[data_ptr + 1]


def case_t21() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    mem[data_ptr] = 41
    kernel = load_kernel_from_rust(OUT_JSON)
    m = Machine(
        mem_dump=list(mem),
        program=kernel.instrs,
        debug_info=kernel.debug_info(),
        trace=False,
        value_trace={(0, 0, "val"): u32(42)},
    )
    m.run(max_cycles=100_000)
    assert m.mem[data_ptr] == u32(42), m.mem[data_ptr]


def case_t22() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    mem[data_ptr] = 10
    mem[data_ptr + 1] = 20
    kernel = load_kernel_from_rust(OUT_JSON)
    # sync() should materialize as flow sync slot.
    assert any(
        any(slot[0] == "sync" for slot in b.get("flow", [])) for b in kernel.instrs
    ), "t22 expected flow sync in emitted program"
    final = run_program(mem, OUT_JSON)
    assert final[data_ptr] == u32(11), final[data_ptr]
    assert final[data_ptr + 1] == u32(22), final[data_ptr + 1]


def case_t23() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    init = [u32(i + 1) for i in range(VLEN)]
    mem[data_ptr : data_ptr + VLEN] = init
    kernel = load_kernel_from_rust(OUT_JSON)
    assert any(
        any(slot[0] == "multiply_add" for slot in b.get("valu", []))
        for b in kernel.instrs
    ), "t23 expected fused multiply_add in emitted program"
    final = run_program(mem, OUT_JSON)
    exp = [u32((i + 1) * 2 + 10) for i in range(VLEN)]
    assert_mem_slice(final, data_ptr, exp, "t23 a*b+c fused to multiply_add")


def case_t24() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    init = [u32(i + 1) for i in range(VLEN)]
    mem[data_ptr : data_ptr + VLEN] = init
    kernel = load_kernel_from_rust(OUT_JSON)
    assert any(
        any(slot[0] == "multiply_add" for slot in b.get("valu", []))
        for b in kernel.instrs
    ), "t24 expected fused multiply_add in emitted program"
    final = run_program(mem, OUT_JSON)
    exp = [u32((i + 1) * 2 + 10 + 20) for i in range(VLEN)]
    assert_mem_slice(final, data_ptr, exp, "t24 a*b+c+d fused to multiply_add")


def case_t25() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    base = 300
    for i in range(VLEN):
        mem[data_ptr + i] = u32(base + i)
        mem[base + i] = u32(100 + i)
    kernel = load_kernel_from_rust(OUT_JSON)
    assert any(
        any(slot[0] == "load_offset" for slot in b.get("load", []))
        for b in kernel.instrs
    ), "t25 expected load_offset vector gather fusion"
    final = run_program(mem, OUT_JSON)
    exp = [u32(100 + i) for i in range(VLEN)]
    assert_mem_slice(final, data_ptr, exp, "t25 dst[lane]=load(addr[lane])")


def case_t26() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    final = run_program(mem, OUT_JSON)
    exp: list[int] = []
    for chunk in range(4):
        exp.extend([u32(chunk)] * VLEN)
    assert_mem_slice(final, data_ptr, exp, "t26 spawn 2x2 worker vbroadcast(i)")


def case_t26_compare() -> None:
    data_ptr = 200
    mem = [0] * 2048
    mem[P_INP_VALUES_P] = data_ptr
    final = load_kernel_from_rust(OUT_JSON)
    OUT_JSON_COMPARE = COMPILER_DIR / "output" / "harness_program_compare.json"
    compile_c_to_json(TEST_DIR / "t26_spawn.c", OUT_JSON_COMPARE)
    final_compare = load_kernel_from_rust(OUT_JSON_COMPARE)
    if len(final.instrs) != len(final_compare.instrs):
        print(
            f"t26_spawn_compare expected same program length, got {len(final.instrs)} and {len(final_compare.instrs)}"
        )
        assert False


CASES: dict[str, Callable[[], None]] = {
    "t01_scalar_load_store": case_t01,
    "t02_vector_load_store": case_t02,
    "t03_vbroadcast_and_valu": case_t03,
    "t04_vselect": case_t04,
    "t05_vhash": case_t05,
    "t06_compile_time_if": case_t06,
    "t07_runtime_if": case_t07,
    "t08_lane_read_write": case_t08,
    "t09_small_end_to_end": case_t09,
    "t10_vector_mod2": case_t10,
    "t11_idx_val_walk": case_t11,
    "t12_else_if_r": case_t12,
    "t13_two_batches": case_t13,
    "t14_nested_batch_round": case_t14,
    "t15_wrap_at_forest_height": case_t15,
    "t16_vselect_idx_minus_one": case_t16,
    "t18_opposite_runtime_if": case_t18,
    "t19_scheduler_pack_independent_alu": case_t19,
    "t20_scheduler_store_load_order": case_t20,
    "t21_scheduler_debug_after_producer": case_t21,
    "t22_sync_barrier": case_t22,
    "t23_vector_multiply_add": case_t23,
    "t24_vector_multiply_add_add": case_t24,
    "t25_load_offset_gather": case_t25,
    "t26_spawn": case_t26,
    "t26_spawn_compare": case_t26_compare,
}


def main() -> int:
    failures: list[str] = []
    for name, fn in sorted(CASES.items()):
        c_file = TEST_DIR / f"{name}.c"
        if not c_file.is_file():
            failures.append(f"missing {c_file}")
            continue
        print(f"--- {name} ({c_file.name}) ---")
        try:
            compile_c_to_json(c_file, OUT_JSON)
            fn()
            print("ok")
        except Exception as e:
            print(f"FAIL: {e!r}")
            failures.append(f"{name}: {e}")

    if failures:
        print("\nFailed:", file=sys.stderr)
        for f in failures:
            print(f"  {f}", file=sys.stderr)
        return 1
    print_bundle_metrics("t09_small_end_to_end.c")
    # print_bundle_metrics("sample_scalar_reference.c")
    print("\nAll compiler harness tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
