from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from problem import DebugInfo


@dataclass
class RustKernel:
    instrs: list[dict[str, list[tuple[Any, ...]]]]

    def debug_info(self) -> DebugInfo:
        # Rust compiler currently emits only instruction bundles,
        # not a symbolic scratch map.
        return DebugInfo(scratch_map={})


def _normalize_instrs(raw: list[dict[str, list[list[Any]]]]) -> list[dict[str, list[tuple[Any, ...]]]]:
    normalized: list[dict[str, list[tuple[Any, ...]]]] = []
    for bundle in raw:
        out_bundle: dict[str, list[tuple[Any, ...]]] = {}
        for engine, slots in bundle.items():
            out_bundle[engine] = [tuple(slot) for slot in slots]
        normalized.append(out_bundle)
    return normalized


def load_kernel_from_rust(path: str | Path) -> RustKernel:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected top-level list in {p}, got {type(payload)}")
    return RustKernel(instrs=_normalize_instrs(payload))


def load_latest_kernel_from_rust(directory: str | Path) -> RustKernel:
    d = Path(directory)
    candidates = list(d.glob("*.json")) + list(d.glob("**/*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON program files found under {d}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"loading rust program: {latest}")
    return load_kernel_from_rust(latest)
