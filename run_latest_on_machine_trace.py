from __future__ import annotations

import argparse
import os
import random
from pathlib import Path

from load_from_rust import load_kernel_from_rust
from problem import CoreState, Input, Machine, Tree, build_mem_image


def find_latest_json(output_dir: Path) -> Path:
    candidates = list(output_dir.glob("*.json")) + list(output_dir.glob("**/*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON program files found under {output_dir}")
    return max(candidates, key=lambda p: p.stat().st_mtime)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description=(
            "Load the most recent compiler JSON from compiler/output/, run it on the "
            "Python Machine, and write trace.json (no reference-kernel comparison)."
        )
    )
    ap.add_argument(
        "--json",
        type=str,
        default=None,
        help="Optional explicit path to a compiled JSON (overrides --latest).",
    )
    ap.add_argument("--forest-height", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--rounds", type=int, default=16)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument(
        "--max-cycles",
        type=int,
        default=None,
        help="Optional safety limit; if exceeded, Machine raises RuntimeError.",
    )
    return ap.parse_args()


def main() -> None:
    repo_root = Path(__file__).resolve().parent
    compiler_output = repo_root / "compiler" / "output"

    args = parse_args()
    os.chdir(repo_root)  # Machine writes trace.json relative to CWD.

    random.seed(args.seed)

    program_json = Path(args.json) if args.json is not None else find_latest_json(compiler_output)
    print(f"loading compiled program: {program_json}")

    kernel = load_kernel_from_rust(program_json)

    forest = Tree.generate(args.forest_height)
    inp = Input.generate(forest, batch_size=args.batch_size, rounds=args.rounds)
    mem = build_mem_image(forest, inp)

    machine = Machine(
        mem_dump=mem,
        program=kernel.instrs,
        debug_info=kernel.debug_info(),
        n_cores=1,
        trace=True,
    )

    # flow_pause instructions set cores to PAUSED; keep calling run() until STOPPED.
    while any(c.state != CoreState.STOPPED for c in machine.cores):
        machine.run(max_cycles=args.max_cycles)

    print(f"done: cycles={machine.cycle}")
    print("wrote trace.json")


if __name__ == "__main__":
    main()

