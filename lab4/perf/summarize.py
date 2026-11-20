#!/usr/bin/env python3
"""Average toplev results over trials (and over cores, for multithreaded runs).

Reads results/<label>_l<level>_t<trial>.txt (written by perf/run_toplev.sh) and
writes results/summary.csv with one row per (label, level): the mean of every
Top-down metric (% of slots) across trials, its standard deviation, the mean
time per kernel call, the number of cores profiled, and the minimum MUX
coverage seen. When toplev counts several cores it prints one line per core
(prefixed S<socket>-C<core>); those are averaged into one value per metric, so
each bar is the mean core's slot breakdown.
"""
import csv
import glob
import os
import re
import statistics
from collections import defaultdict

RESULTS_DIR = "results"
FILE_RE = re.compile(r"^(?P<label>.+)_l(?P<level>\d)_t(?P<trial>\d+)\.txt$")
# Per-core lines start with S<socket>-C<core> (per-thread ones add -T<thread>)
METRIC_RE = re.compile(r"^(?:(?P<core>S\d+-C\d+)(?:-T\d+)?\s+)?\S+\s+(?P<name>[A-Za-z_.]+)\s+%\s+\S+\s+(?P<value>-?[\d.]+)")
MUX_RE = re.compile(r"^(?:S\d+-C\d+(?:-T\d+)?\s+)?MUX\s+%\s+(?P<value>[\d.]+)")
RESULT_RE = re.compile(r"^RESULT .*per_call_us=(?P<us>[\d.]+)")
PINNED_RE = re.compile(r"^PINNED .*ncores=(?P<n>\d+)")


def parse(path):
    # toplev may print one aggregate line per metric for all --core cores, or
    # one line per core; either way the profiled core count comes from PINNED.
    per_core = defaultdict(list)  # metric -> values, one per printed line
    cores, muxes, per_call_us = set(), [], None
    with open(path) as f:
        for line in f:
            if m := PINNED_RE.match(line):
                cores.update(range(int(m["n"])))
            elif m := RESULT_RE.match(line):
                per_call_us = float(m["us"])
            elif m := MUX_RE.match(line):
                muxes.append(float(m["value"]))
            elif m := METRIC_RE.match(line):
                per_core[m["name"]].append(float(m["value"]))
    metrics = {name: statistics.mean(v) for name, v in per_core.items()}
    return metrics, (min(muxes) if muxes else None), per_call_us, max(len(cores), 1)


def main():
    runs = defaultdict(list)  # (label, level) -> [(metrics, mux, us, ncores)]
    metric_order = []
    for path in sorted(glob.glob(os.path.join(RESULTS_DIR, "*_l*_t*.txt"))):
        m = FILE_RE.match(os.path.basename(path))
        if not m:
            continue
        metrics, mux, us, ncores = parse(path)
        if not metrics or us is None:
            print(f"warning: incomplete run, skipped: {path}")
            continue
        if mux is not None and mux < 95:
            print(f"warning: MUX {mux}% (counters multiplexed): {path}")
        for name in metrics:
            if name not in metric_order:
                metric_order.append(name)
        runs[(m["label"], int(m["level"]))].append((metrics, mux, us, ncores))

    header = ["label", "level", "trials", "cores", "per_call_us", "min_mux"]
    for name in metric_order:
        header += [name, name + "_std"]

    out_path = os.path.join(RESULTS_DIR, "summary.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for (label, level), trials in sorted(runs.items()):
            muxes = [t[1] for t in trials if t[1] is not None]
            row = [label, level, len(trials), max(t[3] for t in trials),
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
            print(f"{label:18s} L{level}  n={len(trials)}  cores={row[3]}  {row[4]:>14s} us/call  mux={row[5]}")
    print(f"wrote {out_path}")
    runtimes()


TIME_FILE_RE = re.compile(r"^(?P<label>.+)_time\.txt$")
REAL_RE = re.compile(r"^real\s+(?P<s>[\d.]+)")
FIELD_RE = re.compile(r"(\w+)=(\S+)")


def runtimes():
    """results/<label>_time.txt (un-profiled runs under `time -p`) -> results/runtime.csv:
    wall-clock `real` of the whole program, the driver's own timing of the kernel
    loop (per call), and per-call speedup over naive matmul at the same size,
    plus parallel efficiency (speedup / threads)."""
    out_path = os.path.join(RESULTS_DIR, "runtime.csv")
    header = ["label", "kernel", "size", "threads", "iters", "real_s", "loop_s", "per_call_us",
              "speedup_vs_naive", "efficiency", "checksum"]
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
        rows.append({"label": label, "kernel": fields["kernel"], "size": int(fields["size"]),
                     "threads": int(fields["threads"]), "iters": fields["iters"], "real_s": real,
                     "loop_s": fields["total_s"], "per_call_us": float(fields["per_call_us"]),
                     "checksum": fields["checksum"]})

    naive = {r["size"]: r for r in rows if r["kernel"] == "matmul"}
    rows.sort(key=lambda r: (r["size"], r["kernel"] != "matmul", r["threads"]))
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for r in rows:
            base = naive.get(r["size"])
            speedup = base["per_call_us"] / r["per_call_us"] if base else None
            # The driver adds one output element per iteration, so compare per iteration
            # (float accumulation over different iteration counts leaves tiny rounding).
            if base:
                mine = float(r["checksum"]) / int(r["iters"])
                ref = float(base["checksum"]) / int(base["iters"])
                if abs(mine - ref) > 1e-3 * max(1.0, abs(ref)):
                    print(f"warning: checksum/iter {mine:.6f} != naive {ref:.6f}: {r['label']}")
            writer.writerow([r["label"], r["kernel"], r["size"], r["threads"], r["iters"], f"{r['real_s']:.2f}",
                             r["loop_s"], f"{r['per_call_us']:.1f}",
                             f"{speedup:.2f}" if speedup else "",
                             f"{speedup / r['threads']:.2f}" if speedup else "", r["checksum"]])
            print(f"{r['label']:18s} real {r['real_s']:>7.2f} s  {r['per_call_us']:>14.1f} us/call"
                  + (f"  x{speedup:.2f}" if speedup else ""))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
