#!/usr/bin/env python3
"""
Visualize scratch register allocation from compiler JSON (debug_info.scratch_lifetime).

Generate with the Rust compiler:
  cargo run -- <input.c> <out.json> --trace-scratch

Then:
  python tools/visualize_scratch_lifetime.py <out.json> -o scratch_viz.html

Requires: pip install plotly (optional: kaleido for static PNG export).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


# list of 30 different colors for the bars
COLORS = [
    "rgba(70, 110, 180, 0.8)",
    "rgba(255, 165, 0, 0.8)",
    "rgba(0, 255, 0, 0.8)",
    "rgba(255, 0, 0, 0.8)",
    "rgba(0, 0, 255, 0.8)",
    "rgba(255, 255, 0, 0.8)",
    "rgba(0, 255, 255, 0.8)",
    "rgba(128, 0, 128, 0.8)",
    "rgba(128, 128, 128, 0.8)",
    "rgba(128, 128, 0, 0.8)",
    "rgba(0, 128, 128, 0.8)",
    "rgba(0, 128, 0, 0.8)",
    "rgba(128, 0, 128, 0.8)",
    "rgba(0, 0, 128, 0.8)",
    "rgba(128, 0, 0, 0.8)",
]


def load_trace(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    dbg = data.get("debug_info") or {}
    trace = dbg.get("scratch_lifetime")
    if not trace:
        print(
            "error: debug_info.scratch_lifetime missing. Recompile with --trace-scratch.",
            file=sys.stderr,
        )
        sys.exit(2)
    return trace


def plot_scratch_lifetime_html(
    trace: dict,
    out_html: Path,
    title: str,
    max_intervals: int,
) -> None:
    import plotly.graph_objects as go

    intervals = trace.get("intervals") or []
    events = trace.get("events") or []

    rows = sorted(
        intervals,
        key=lambda r: (r["alloc_bundle"], r.get("reg_name", ""), r["reg"]),
    )
    if len(rows) > max_intervals:
        rows = rows[:max_intervals]

    max_base = max([r["scratch_base"] for r in rows])
    bases = [i for i in range(max_base + 1)]
    labels = [f"base {b}" for b in bases]

    # labels = [f"{r['reg_name']} (r{r['reg']}) @{r['scratch_base']}" for r in rows]
    # labels = [f"base: {r['scratch_base']}" for r in rows]
    y_pos = list(range(len(bases)))

    fig = go.Figure()
    interval_by_reg = {}
    for r in rows:
        # Intervals are disjoint per register in this trace, so last-write is fine.
        interval_by_reg[r["reg"]] = r

    # Build interval bars in one trace to keep rendering/hover responsive.
    bar_x = []
    bar_y = []
    bar_base = []
    bar_width = []
    bar_color = []
    bar_hover = []
    for i, r in enumerate(rows):
        a, b = r["alloc_bundle"], r["free_bundle"]
        scratch_base, width = r["scratch_base"], r["width"]
        for wi in range(width):
            bar_x.append(max(0, b - a))
            bar_y.append(scratch_base + wi)
            bar_base.append(a)
            bar_width.append(0.88 if width == 1 or wi in (0, width - 1) else 1.5)
            bar_color.append(COLORS[i % len(COLORS)])
            bar_hover.append(
                (
                    f"{r['reg_name']} r{r['reg']}<br>"
                    f"scratch [{scratch_base}..{scratch_base + width - 1}] ({width} words)<br>"
                    f"bundles [{a}..{b}]<br>"
                    f"steps [{r['alloc_step']}..{r['free_step']}]"
                )
            )
    if bar_x:
        fig.add_trace(
            go.Bar(
                x=bar_x,
                y=bar_y,
                base=bar_base,
                width=bar_width,
                orientation="h",
                showlegend=False,
                marker=dict(
                    color=bar_color,
                    line=dict(color="rgba(40,40,40,0.5)", width=0.6),
                ),
                hovertemplate="%{customdata}<extra></extra>",
                customdata=bar_hover,
            )
        )

    # Events: overlay allocator transitions on the same timeline.
    max_ev = 15_000
    ev_sample = (
        events if len(events) <= max_ev else events[:: len(events) // max_ev + 1]
    )

    kind_colors = {
        "alloc": "#2ca02c",
        "use_read": "#1f77b4",
        "use_def": "#ff7f0e",
        "free": "#d62728",
    }
    kind_labels = {
        "alloc": "alloc",
        "use_read": "read",
        "use_def": "def",
        "free": "free",
    }
    kind_symbols = {
        "alloc": "diamond",
        "use_read": "circle",
        "use_def": "triangle-up",
        "free": "x",
    }

    def resolve_event_lane(event: dict) -> float | None:
        base = event.get("scratch_base")
        width = event.get("width", 1)
        if base is not None:
            return base + (max(1, width) - 1) / 2.0
        reg = event.get("reg")
        if reg is None:
            return None
        interval = interval_by_reg.get(reg)
        if not interval:
            return None
        return interval["scratch_base"] + (max(1, interval["width"]) - 1) / 2.0

    for kind, color in kind_colors.items():
        pts = [e for e in ev_sample if e.get("kind") == kind]
        if not pts:
            continue
        xs = []
        ys = []
        hover = []
        for e in pts:
            y = resolve_event_lane(e)
            if y is None:
                continue
            xs.append(e["bundle_index"])
            ys.append(y)
            hover.append(
                (
                    f"{kind_labels[kind]} {e.get('reg_name', '?')} r{e.get('reg', '?')}<br>"
                    f"bundle {e.get('bundle_index')} | step {e.get('step')}<br>"
                    f"width {e.get('width', '?')}"
                )
            )
        if not xs:
            continue
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers",
                marker=dict(
                    size=7,
                    color=color,
                    opacity=0.9,
                    symbol=kind_symbols[kind],
                    line=dict(color="white", width=0.7),
                ),
                name=kind_labels[kind],
                hovertemplate="%{customdata}<extra></extra>",
                customdata=hover,
            ),
        )

    fig.update_xaxes(
        title_text="bundle index",
        showgrid=True,
        gridcolor="rgba(200,200,200,0.2)",
        zeroline=False,
    )
    fig.update_yaxes(
        title_text="scratch base",
        tickmode="array",
        tickvals=y_pos,
        ticktext=labels,
        autorange="reversed",
        showgrid=True,
        gridcolor="rgba(200,200,200,0.2)",
        zeroline=False,
    )
    fig.update_layout(
        title_text=title,
        template="plotly_white",
        barmode="overlay",
        height=max(520, min(2600, 280 + len(rows) * 12)),
        margin=dict(l=180, r=28, t=72, b=52),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.7)",
        ),
        hovermode="closest",
        plot_bgcolor="rgba(248,250,252,1.0)",
    )
    out_html.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(out_html, include_plotlyjs="cdn")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "json", type=Path, help="Compiler output JSON with scratch_lifetime"
    )
    ap.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("scratch_lifetime.html"),
        help="Output HTML path (default: scratch_lifetime.html)",
    )
    ap.add_argument(
        "--max-intervals",
        type=int,
        default=400,
        help="Max live-interval rows to plot (default: 400)",
    )
    ap.add_argument("--title", default="", help="Chart title")
    args = ap.parse_args()

    trace = load_trace(args.json)
    title = args.title or f"Scratch lifetime — {args.json.name}"

    try:
        plot_scratch_lifetime_html(trace, args.output, title, args.max_intervals)
    except ImportError as e:
        print("error: install plotly:  pip install plotly", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)

    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
