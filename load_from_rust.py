from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from problem import DebugInfo


@dataclass
class RustKernel:
    instrs: list[dict[str, list[tuple[Any, ...]]]]
    _scratch_map: dict[int, tuple[str, int]]

    def debug_info(self) -> DebugInfo:
        return DebugInfo(scratch_map=dict(self._scratch_map))


def _normalize_instrs(raw: list[dict[str, list[list[Any]]]]) -> list[dict[str, list[tuple[Any, ...]]]]:
    normalized: list[dict[str, list[tuple[Any, ...]]]] = []
    for bundle in raw:
        out_bundle: dict[str, list[tuple[Any, ...]]] = {}
        for engine, slots in bundle.items():
            out_bundle[engine] = [tuple(slot) for slot in slots]
        normalized.append(out_bundle)
    return normalized


def _parse_scratch_map(obj: Any) -> dict[int, tuple[str, int]]:
    """Parse `debug_info.scratch_map` from JSON (string keys -> int)."""
    if not isinstance(obj, dict):
        return {}
    out: dict[int, tuple[str, int]] = {}
    for k, v in obj.items():
        addr = int(k)
        if isinstance(v, (list, tuple)) and len(v) == 2:
            name, ln = v[0], v[1]
            if isinstance(name, str) and isinstance(ln, int):
                out[addr] = (name, ln)
    return out


def load_kernel_from_rust(path: str | Path) -> RustKernel:
    p = Path(path)
    payload = json.loads(p.read_text(encoding="utf-8"))

    # New format: {"instructions": [...], "debug_info": {"scratch_map": {...}}}
    if isinstance(payload, dict):
        if "instructions" not in payload:
            raise ValueError(
                f"Expected top-level list or object with 'instructions' in {p}, got {p!s}"
            )
        raw = payload["instructions"]
        if not isinstance(raw, list):
            raise ValueError(f"'instructions' must be a list in {p}")
        dbg = payload.get("debug_info") or {}
        sm = _parse_scratch_map(dbg.get("scratch_map") if isinstance(dbg, dict) else {})
        return RustKernel(instrs=_normalize_instrs(raw), _scratch_map=sm)

    # Legacy: top-level JSON array of instruction bundles
    if isinstance(payload, list):
        return RustKernel(instrs=_normalize_instrs(payload), _scratch_map={})

    raise ValueError(f"Expected top-level list or object in {p}, got {type(payload)}")


def load_latest_kernel_from_rust(directory: str | Path) -> RustKernel:
    d = Path(directory)
    candidates = list(d.glob("*.json")) + list(d.glob("**/*.json"))
    if not candidates:
        raise FileNotFoundError(f"No JSON program files found under {d}")
    latest = max(candidates, key=lambda p: p.stat().st_mtime)
    print(f"loading rust program: {latest}")
    return load_kernel_from_rust(latest)
