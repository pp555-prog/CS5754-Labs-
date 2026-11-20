#!/usr/bin/env python3
"""Horizontal stacked Top-down bar charts from results/summary.csv.

Writes results/topdown_l1.svg and results/topdown_l2.svg: one bar per matmul
variant and input size (grouped by size: naive matmul, then matmul_thread with
1, 2, 4, 8 threads), segments = % of pipeline slots averaged over trials and,
for multithreaded runs, over the profiled cores.
Standard library only (the course server has no matplotlib/pip). SVGs open in
any browser and can be inserted into Word or printed to PDF.
"""
import csv
import os
import re
from xml.sax.saxutils import escape

RESULTS_DIR = "results"

# Fixed categorical order (validated reference palette, light mode). Adjacent
# segments in each stack follow this order, which is what the palette is
# validated for.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
DARK_FILLS = {"#2a78d6", "#008300", "#4a3aa7", "#e34948"}  # white value labels on these
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
FONT = "system-ui, -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif"

L1 = [
    ("Frontend_Bound", "Frontend bound"),
    ("Bad_Speculation", "Bad speculation"),
    ("Backend_Bound", "Backend bound"),
    ("Retiring", "Retiring"),
]
L2 = [
    ("Frontend_Bound.Fetch_Latency", "FE: fetch latency"),
    ("Frontend_Bound.Fetch_Bandwidth", "FE: fetch bandwidth"),
    ("Bad_Speculation.Branch_Mispredicts", "Bad spec: branch mispredicts"),
    ("Bad_Speculation.Machine_Clears", "Bad spec: machine clears"),
    ("Backend_Bound.Memory_Bound", "BE: memory bound"),
    ("Backend_Bound.Core_Bound", "BE: core bound"),
    ("Retiring.Light_Operations", "Retiring: light ops"),
    ("Retiring.Heavy_Operations", "Retiring: heavy ops"),
]

# Labels written by perf/run_toplev.sh: naive-n<size> (Lab 1 matmul, 1 core)
# or thread<T>-n<size> (matmul_thread with T threads, mean over its T cores)
LABEL_RE = re.compile(r"^(?:(?P<naive>naive)|thread(?P<threads>\d+))-n(?P<size>\d+)$")


def configs_for(rows, level):
    """(label, row name, group) ordered by size, naive first, then by thread
    count; group = size separates bars with a hairline."""
    found = []
    for label, lvl in rows:
        m = LABEL_RE.match(label)
        if lvl != level or not m:
            continue
        size = int(m["size"])
        threads = 0 if m["naive"] else int(m["threads"])
        name = f"naive  n={size}" if threads == 0 else f"thread x{threads}  n={size}"
        found.append(((size, threads), label, name, size))
    return [(label, name, group) for _, label, name, group in sorted(found)]


LABEL_MIN = 8.0  # only direct-label segments at least this wide (% slots)

# Layout (px)
WIDTH = 760
PLOT_X = 170  # left edge of the plot area (row labels sit to its left)
PLOT_W = 560
ROW_H = 24
BAR_H = 16
LEGEND_COLS = 4
LEGEND_COL_W = (WIDTH - 20) / LEGEND_COLS
LEGEND_ROW_H = 18


def load():
    with open(os.path.join(RESULTS_DIR, "summary.csv")) as f:
        return {(r["label"], int(r["level"])): r for r in csv.DictReader(f)}


def value(row, metric):
    v = row.get(metric, "")
    return max(float(v), 0.0) if v else 0.0  # toplev can report tiny negatives


def text(x, y, s, size=11, color=INK, anchor="start", weight="normal"):
    return (f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{color}" text-anchor="{anchor}" '
            f'font-weight="{weight}" dominant-baseline="middle">{escape(s)}</text>')


def plot(rows, level, metrics, name):
    configs = configs_for(rows, level)
    legend_rows = -(-len(metrics) // LEGEND_COLS)
    legend_y = 44
    plot_y = legend_y + legend_rows * LEGEND_ROW_H + 10
    plot_h = len(configs) * ROW_H
    height = plot_y + plot_h + 48

    def sx(pct):
        return PLOT_X + pct / 100 * PLOT_W

    out = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{height}" '
           f'viewBox="0 0 {WIDTH} {height}" font-family="{FONT}">',
           f'<rect width="{WIDTH}" height="{height}" fill="{SURFACE}"/>',
           text(10, 20, f"Top-down level {level}: % of pipeline slots (mean of trials and cores)", size=14, weight="600")]

    # Legend: fixed series order, wraps into rows of LEGEND_COLS
    for i, ((_, label), color) in enumerate(zip(metrics, SERIES)):
        lx = 10 + (i % LEGEND_COLS) * LEGEND_COL_W
        ly = legend_y + (i // LEGEND_COLS) * LEGEND_ROW_H
        out.append(f'<rect x="{lx:.1f}" y="{ly - 5:.1f}" width="10" height="10" rx="2" fill="{color}"/>')
        out.append(text(lx + 15, ly, label, size=11, color=INK_2))

    # Recessive vertical grid + x ticks
    for pct in range(0, 101, 20):
        x = sx(pct)
        out.append(f'<line x1="{x:.1f}" y1="{plot_y}" x2="{x:.1f}" y2="{plot_y + plot_h}" stroke="{GRID}" stroke-width="1"/>')
        out.append(text(x, plot_y + plot_h + 14, f"{pct}%", size=10, color=MUTED, anchor="middle"))
    out.append(f'<line x1="{PLOT_X}" y1="{plot_y + plot_h}" x2="{PLOT_X + PLOT_W}" y2="{plot_y + plot_h}" stroke="{AXIS}" stroke-width="1"/>')

    for r, (key, label, group) in enumerate(configs):
        row = rows[(key, level)]
        y_mid = plot_y + r * ROW_H + ROW_H / 2
        out.append(text(PLOT_X - 8, y_mid, label, size=11, anchor="end"))
        left = 0.0
        for (metric, _), color in zip(metrics, SERIES):
            width = value(row, metric)
            if width > 0:
                # surface-colored stroke = 2px gap between adjacent segments
                out.append(f'<rect x="{sx(left):.1f}" y="{y_mid - BAR_H / 2:.1f}" width="{width / 100 * PLOT_W:.1f}" '
                           f'height="{BAR_H}" fill="{color}" stroke="{SURFACE}" stroke-width="1"/>')
                if width >= LABEL_MIN:
                    out.append(text(sx(left + width / 2), y_mid, f"{width:.0f}", size=10, anchor="middle",
                                    color="#ffffff" if color in DARK_FILLS else INK))
            left += width
        # Hairline between input sizes
        if r < len(configs) - 1 and configs[r + 1][2] != group:
            y = plot_y + (r + 1) * ROW_H
            out.append(f'<line x1="10" y1="{y}" x2="{PLOT_X + PLOT_W}" y2="{y}" stroke="{AXIS}" stroke-width="0.75"/>')

    out.append("</svg>")
    path = os.path.join(RESULTS_DIR, f"{name}.svg")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    print(f"wrote {path}")


def main():
    rows = load()
    plot(rows, 1, L1, "topdown_l1")
    plot(rows, 2, L2, "topdown_l2")


if __name__ == "__main__":
    main()
