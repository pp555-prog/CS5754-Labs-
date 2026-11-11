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
    runtimes()


TIME_FILE_RE = re.compile(r"^(?P<label>.+)_time\.txt$")
REAL_RE = re.compile(r"^real\s+(?P<s>[\d.]+)")
FIELD_RE = re.compile(r"(\w+)=(\S+)")


def runtimes():
    """results/<label>_time.txt (un-profiled runs under `time -p`) -> results/runtime.csv:
    wall-clock `real` of the whole program, plus the driver's own timing of the
    kernel loop (per call) and, for sparse, of the one-off CSR build."""
    out_path = os.path.join(RESULTS_DIR, "runtime.csv")
    header = ["label", "kernel", "size", "density", "nnz", "iters", "real_s", "loop_s", "per_call_us", "csr_build_us"]
    rows = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_time.txt"))):
        label = TIME_FILE_RE.match(os.path.basename(path))["label"]
        real, fields = None, {}
        with open(path) as f:
            for line in f:
                if m := REAL_RE.match(line):
                    real = float(m["s"])
                elif line.startswith("RESULT"):
                    fields = dict(FIELD_RE.findall(line))
        if real is None or not fields:
            print(f"warning: incomplete timing run, skipped: {path}")
            continue
        rows.append([label, fields["kernel"], fields["size"], fields["density"], fields["nnz"], fields["iters"],
                     f"{real:.2f}", fields["total_s"], fields["per_call_us"], fields["csr_build_us"]])
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)
    for r in rows:
        print(f"{r[0]:22s} real {r[6]:>8s} s  {r[8]:>14s} us/call")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
