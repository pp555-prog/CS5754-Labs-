#!/bin/bash
# Top-down profiling of matmul (Lab 1 baseline) and selected matmul_blocking
# tile configurations, at several input sizes, levels 1 and 2, N trials.
#
# Run from ~/lab2 on the login node, after perf/sweep_tiles.sh has built
# perf/bin/profile_<variant> (one binary per tile configuration):
#   nohup bash perf/run_toplev.sh 2 > results/toplev.log 2>&1 &
#
# Each (variant, size, level, trial) is a separate srun job on the ece5755
# partition (Sapphire Rapids, 5-minute limit). The node is shared with other
# users, so perf/on_core.sh pins the program to whichever core Slurm assigned
# and toplev counts only that core (its PINNED line is kept in each file).
# Output goes to results/<variant>-n<size>_l<level>_t<trial>.txt; finished
# runs are skipped on re-run, so the script can be restarted after an
# interruption. Then: python3 perf/summarize.py && python3 perf/plot.py
set -u
TRIALS=${1:-2}
OUT=results
mkdir -p "$OUT"

# Variants to profile: "naive" is the Lab 1 matmul; the others are tags from
# VARIANTS in perf/sweep_tiles.sh (pick the best and a contrasting one)
VARIANTS=(naive t169_32_9 t64_64_64 t32_256_32)

# size iters: same inputs as the runtime sweep
SIZES=(
    "64   20000"
    "512  20"
    "1536 1"
)

run() { # file, level, binary, profile args...
    local f=$1 level=$2 bin=$3
    shift 3
    if [ -s "$f" ] && grep -q '^RESULT' "$f"; then
        return
    fi
    echo "[$(date +%T)] $f"
    srun -p ece5755 --time=00:05:00 --chdir="$PWD" \
        bash perf/on_core.sh toplev "$level" "$bin" "$@" > "$f" 2>&1
    grep '^RESULT' "$f" || echo "  !! no RESULT line, check $f"
}

for v in "${VARIANTS[@]}"; do
    if [ ! -x "perf/bin/profile_$v" ]; then
        echo "missing perf/bin/profile_$v: run perf/sweep_tiles.sh first (or add $v to its VARIANTS)"
        exit 1
    fi
done

for s in "${SIZES[@]}"; do
    read -r size iters <<< "$s"
    for v in "${VARIANTS[@]}"; do
        kernel=matmul_blocking
        [ "$v" = naive ] && kernel=matmul
        for level in 1 2; do
            for t in $(seq 1 "$TRIALS"); do
                run "$OUT/${v}-n${size}_l${level}_t${t}.txt" "$level" "perf/bin/profile_$v" $kernel $size $iters
            done
        done
        # Setup-only baseline (iters = 0): shows how much of a run is not the kernel
        run "$OUT/${v}-n${size}-setup_l1_t1.txt" 1 "perf/bin/profile_$v" $kernel $size 0
    done
done

echo "[$(date +%T)] done; now run: python3 perf/summarize.py && python3 perf/plot.py"
