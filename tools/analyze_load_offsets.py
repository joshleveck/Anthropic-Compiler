import ast
import csv
import json
from collections import defaultdict

import matplotlib.pyplot as plt


TRACE_PATH = "trace.json"
CSV_OUT = "load_offset_node_numbers.csv"
PLOT_OUT = "load_offset_node_numbers.png"
HIST_OUT = "load_offset_node_histogram.png"


def iter_trace_events(path: str):
    """Yield JSON objects from trace file, tolerating bracket/comma wrappers."""
    with open(path, "r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or line in {"[", "]"}:
                continue
            if line.endswith(","):
                line = line[:-1]
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # Skip malformed trailer lines if present.
                continue


def main():
    events = list(iter_trace_events(TRACE_PATH))

    # Build scratch addr -> variable name map from metadata events.
    scratch_addr_to_name = {}
    for ev in events:
        if ev.get("ph") == "M" and ev.get("name") == "thread_name" and ev.get("pid") == 1:
            tid = ev.get("tid")
            if isinstance(tid, int) and tid >= 100000:
                scratch_addr_to_name[tid - 100000] = ev.get("args", {}).get("name", "")

    # Gather scalar forest pointer and time series of vector address registers (vtmp_b_*).
    forest_ptr = None
    vtmp_b_history = defaultdict(list)  # key: scratch base addr, val: list[(ts, [8 vals])]
    for ev in events:
        if ev.get("ph") != "X" or ev.get("pid") != 1:
            continue
        tid = ev.get("tid")
        if not isinstance(tid, int) or tid < 100000:
            continue
        scratch_addr = tid - 100000
        var_name = scratch_addr_to_name.get(scratch_addr, "")
        raw_name = ev.get("name", "")

        if var_name.startswith("forest_values_p-"):
            try:
                forest_ptr = int(raw_name)
            except (TypeError, ValueError):
                pass
            continue

        if var_name.startswith("vtmp_b_"):
            try:
                vec = [int(x.strip()) for x in raw_name.split(",") if x.strip()]
            except ValueError:
                continue
            if len(vec) == 8:
                vtmp_b_history[scratch_addr].append((int(ev.get("ts", -1)), vec))

    # Ensure histories are time-ordered for binary-style scan.
    for addr in vtmp_b_history:
        vtmp_b_history[addr].sort(key=lambda x: x[0])

    if forest_ptr is None:
        raise RuntimeError("Could not locate forest_values_p in trace.")

    # Resolve each load_offset to concrete memory address via current vtmp_b lane register.
    rows = []
    for ev in events:
        if ev.get("ph") != "X" or ev.get("pid") != 0 or ev.get("name") != "load_offset":
            continue
        slot_str = ev.get("args", {}).get("slot")
        if not slot_str:
            continue
        try:
            op, _dest, addr_base, offset = ast.literal_eval(slot_str)
        except (SyntaxError, ValueError):
            continue
        if op != "load_offset":
            continue
        ts = int(ev.get("ts", -1))
        hist = vtmp_b_history.get(addr_base)
        if not hist or offset < 0 or offset >= 8:
            continue

        vec = None
        for t, v in hist:
            if t <= ts:
                vec = v
            else:
                break
        if vec is None:
            continue

        mem_addr = vec[offset]
        node_number = mem_addr - forest_ptr
        rows.append((ts, addr_base, offset, mem_addr, forest_ptr, node_number))

    if not rows:
        raise RuntimeError("No load_offset events could be resolved.")

    # Write CSV for inspection.
    with open(CSV_OUT, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "ts",
                "addr_base_register",
                "offset",
                "resolved_mem_addr",
                "forest_ptr",
                "node_number",
            ]
        )
        writer.writerows(rows)

    node_numbers = [r[-1] for r in rows]

    # Plot only node number sequence, as requested.
    plt.figure(figsize=(14, 5))
    plt.plot(range(len(node_numbers)), node_numbers, linewidth=0.8)
    plt.title("Tree Node Number Accessed by load_offset")
    plt.xlabel("load_offset event index")
    plt.ylabel("node_number")
    plt.tight_layout()
    plt.savefig(PLOT_OUT, dpi=180)
    plt.close()

    # Histogram of node access frequency.
    plt.figure(figsize=(12, 5))
    bins = range(min(node_numbers), max(node_numbers) + 2)
    plt.hist(node_numbers, bins=bins, edgecolor="black", linewidth=0.3)
    plt.title("Histogram of Tree Node Access Frequency (load_offset)")
    plt.xlabel("node_number")
    plt.ylabel("access_count")
    plt.tight_layout()
    plt.savefig(HIST_OUT, dpi=180)
    plt.close()

    print(f"forest_ptr: {forest_ptr}")
    print(f"resolved load_offset events: {len(rows)}")
    print(f"node_number min/max: {min(node_numbers)} / {max(node_numbers)}")
    print(f"wrote: {CSV_OUT}")
    print(f"wrote: {PLOT_OUT}")
    print(f"wrote: {HIST_OUT}")


if __name__ == "__main__":
    main()
