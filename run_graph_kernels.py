from __future__ import annotations

import argparse
import random
import subprocess
from pathlib import Path

from load_from_rust import load_kernel_from_rust
from problem import Machine, myhash, SCRATCH_SIZE

REPO_ROOT = Path(__file__).resolve().parent
COMPILER_DIR = REPO_ROOT / "compiler"


def u32(x: int) -> int:
    return x % (2**32)


N = 256
DEG = 8
STEPS = 16
Q = 16
MAX = 65535


def clamp(x: int, lo: int, hi: int) -> int:
    return lo if x < lo else hi if x > hi else x


def pwl_activate(x: int) -> int:
    # Must match the C kernels.
    t1 = MAX // 4
    t2 = MAX // 2
    t3 = (MAX * 3) // 4

    s0, b0 = 8192, 0
    s1, b1 = 32768, 4096
    s2, b2 = 49152, 8192
    s3, b3 = 57344, 4096

    if x < t1:
        y = (x * s0 + b0) >> Q
    elif x < t2:
        y = (x * s1 + b1) >> Q
    elif x < t3:
        y = (x * s2 + b2) >> Q
    else:
        y = (x * s3 + b3) >> Q
    return clamp(y, 0, MAX)


def reference_kernel(idxs_k_i: list[list[int]], w_k_i: list[list[int]], values: list[int]) -> list[int]:
    v = [u32(x) for x in values]
    for _ in range(STEPS):
        out = [0] * N
        for i in range(N):
            acc = 0
            for k in range(DEG):
                j = idxs_k_i[k][i]
                w = w_k_i[k][i]
                acc = u32(acc + u32(v[j] * w))
            x = u32(acc >> Q)
            x = clamp(x, 0, MAX)
            x = pwl_activate(x)
            out[i] = u32(x)
        v = out
    return v


def build_mem(seed: int) -> tuple[list[int], int, int, int, int, list[list[int]], list[list[int]], list[int]]:
    rng = random.Random(seed)

    # Graph stored transposed: idxs[k][i] and w[k][i]
    idxs = [[0] * N for _ in range(DEG)]
    w = [[0] * N for _ in range(DEG)]
    for i in range(N):
        for k in range(DEG):
            idxs[k][i] = rng.randrange(0, N)
            # weights in Q16, biased to smaller values to avoid overflow saturation
            w[k][i] = rng.randrange(0, 1 << 16)

    values0 = [rng.randrange(0, MAX + 1) for _ in range(N)]

    header_words = 16
    idxs_words = DEG * N
    w_words = DEG * N
    values_words = N
    tmp_words = N
    mem_words = header_words + idxs_words + w_words + values_words + tmp_words + 64
    mem = [0] * mem_words

    idxs_p = header_words
    w_p = idxs_p + idxs_words
    values_p = w_p + w_words
    tmp_p = values_p + values_words

    # Fill header (see C kernels)
    mem[1] = N
    mem[6] = values_p
    mem[7] = tmp_p
    mem[8] = idxs_p
    mem[9] = w_p

    # Write graph
    for k in range(DEG):
        base = idxs_p + k * N
        mem[base : base + N] = idxs[k]
    for k in range(DEG):
        base = w_p + k * N
        mem[base : base + N] = [u32(x) for x in w[k]]

    mem[values_p : values_p + N] = [u32(x) for x in values0]
    mem[tmp_p : tmp_p + N] = [0] * N

    return mem, values_p, tmp_p, idxs_p, w_p, idxs, w, values0


def compile_c(c_path: Path, out_json: Path) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["cargo", "run", "-q", "--", str(c_path), str(out_json)],
        cwd=COMPILER_DIR,
        check=True,
    )


def run_kernel(program_json: Path, mem: list[int]) -> tuple[list[int], int]:
    kernel = load_kernel_from_rust(program_json)
    m = Machine(mem_dump=list(mem), program=kernel.instrs, debug_info=kernel.debug_info(), trace=False)
    m.run(max_cycles=2_000_000)
    return m.mem, m.cycle


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--kernel", choices=["naive", "opt", "both"], default="both")
    args = ap.parse_args()

    mem, values_p, _tmp_p, _idxs_p, _w_p, idxs, w, values0 = build_mem(args.seed)
    ref = reference_kernel(idxs, w, values0)

    kernels = []
    if args.kernel in ("naive", "both"):
        kernels.append(("naive", COMPILER_DIR / "test" / "t28_graph_pwl_naive.c"))
    if args.kernel in ("opt", "both"):
        kernels.append(("opt", COMPILER_DIR / "test" / "t28_graph_pwl_opt.c"))

    for name, c_path in kernels:
        out_json = COMPILER_DIR / "output" / f"t28_{name}.json"
        compile_c(c_path, out_json)
        final_mem, cycles = run_kernel(out_json, mem)
        got = final_mem[values_p : values_p + N]
        ok = got == ref
        print(f"{name}: cycles={cycles} scratch_size={SCRATCH_SIZE} ok={ok}")
        if not ok:
            # show first mismatch
            for i, (a, b) in enumerate(zip(got, ref)):
                if a != b:
                    raise SystemExit(f"mismatch at i={i}: got={a} ref={b}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

