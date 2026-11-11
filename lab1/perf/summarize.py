#!/usr/bin/env python3
"""Average toplev results over trials.

Reads results/<label>_l<level>_t<trial>.txt (written by perf/run_toplev.sh) and
writes results/summary.csv with one row per (label, level): the mean of every
Top-down metric (% of slots) across trials, its standard deviation, the mean
time per kernel call, and the minimum MUX coverage seen.
"""
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

RESULTS_DIR = "results"
FILE_RE = re.compile(r"^(?P<label>.+)_l(?P<level>\d)_t(?P<trial>\d+)\.txt$")
METRIC_RE = re.compile(r"^\S+\s+(?P<name>[A-Za-z_.]+)\s+%\s+\S+\s+(?P<value>-?[\d.]+)")
MUX_RE = re.compile(r"^MUX\s+%\s+(?P<value>[\d.]+)")
RESULT_RE = re.compile(r"^RESULT .*per_call_us=(?P<us>[\d.]+)")


def parse(path):
    metrics, mux, per_call_us = {}, None, None
    with open(path) as f:
        for line in f:
            if m := RESULT_RE.match(line):
                per_call_us = float(m["us"])
            elif m := MUX_RE.match(line):
                mux = float(m["value"])
            elif m := METRIC_RE.match(line):
                metrics[m["name"]] = float(m["value"])
    return metrics, mux, per_call_us


def main():
    runs = defaultdict(list)  # (label, level) -> [(metrics, mux, us)]
    metric_order = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_l*_t*.txt"))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        metrics, mux, us = parse(path)
        if not metrics or us is None:
            print(f"warning: incomplete run, skipped: {path}")
            continue
        if mux is not None and mux < 95:
            print(f"warning: MUX {mux}% (counters multiplexed): {path}")
        for name in metrics:
            if name not in metric_order:
                metric_order.append(name)
        runs[(m["label"], int(m["level"]))].append((metrics, mux, us))

    header = ["label", "level", "trials", "per_call_us", "min_mux"]
    for name in metric_order:
        header += [name, name + "_std"]

    out_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (label, level), trials in sorted(runs.items()):
            muxes = [t[1] for t in trials if t[1] is not None]
            row = [label, level, len(trials),
                   f"{statistics.mean(t[2] for t in trials):.3f}",
                   f"{min(muxes):.1f}" if muxes else ""]
            for name in metric_order:
                values = [t[0][name] for t in trials if name in t[0]]
                if values:
                    std = statistics.stdev(values) if len(values) > 1 else 0.0
                    row += [f"{statistics.mean(values):.2f}", f"{std:.2f}"]
                else:
                    row += ["", ""]
            writer.writerow(row)
            print(f"{label:22s} L{level}  n={len(trials)}  {row[3]:>14s} us/call")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
