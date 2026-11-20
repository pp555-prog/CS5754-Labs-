#!/bin/bash
# Top-down profiling and timing of matmul (Lab 1 baseline) vs. matmul_thread
# with 1, 2, 4 and 8 threads, over several input sizes.
#
# Run from ~/lab4 on the login node, after `make profile`:
#   nohup bash perf/run_toplev.sh 1 > results/toplev.log 2>&1 &
#   tail -f results/toplev.log
#
# Each (variant, size) gets:
#   <label>_l1_t<k>.txt, <label>_l2_t<k>.txt  toplev levels 1 and 2, TRIALS trials
#   <label>_time.txt                           un-profiled run under `time -p` (real)
# Labels: naive-n<size> (matmul, 1 core) and thread<T>-n<size> (matmul_thread, T threads).
# Each is a separate srun job on the ece5755 partition (Sapphire Rapids,
# 5-minute limit) with 8 CPUs (-c 8). perf/on_core.sh pins the program to T of
# them (one thread per physical core) and toplev counts only those T cores.
# Finished runs are skipped on re-run, so the script can be restarted.
# Then: python3 perf/summarize.py && python3 perf/plot.py
set -u
TRIALS=${1:-1}
TARGET_S=${TARGET_S:-6}    # aim for ~6 s of kernel time per run
OUT=results
mkdir -p "$OUT"

SIZES=(256 512 1024 1536)
THREADS=(1 2 4 8)

# Iterations so one run's kernel loop takes ~TARGET_S, from the -O0 cost of
# naive matmul measured in Labs 1-3 (~3.3 ns per multiply-add), assuming
# perfect scaling with the thread count.
iters_for() { # size threads
    awk -v n="$1" -v t="$2" -v target="$TARGET_S" 'BEGIN {
        est = n * n * n * 3.3e-9 / t;
        it = int(target / est + 0.5); if (it < 1) it = 1; print it }'
}

done_ok() { [ -s "$1" ] && grep -q '^RESULT' "$1"; }

# srun one job; $1 = output file, $2 = threads (NCPUS and MATMUL_THREADS), rest = on_core.sh args
job() {
    local f=$1 t=$2
    shift 2
    done_ok "$f" && return
    echo "[$(date +%T)] $f"
    srun -p ece5755 -c 8 --time=00:05:00 --chdir="$PWD" \
        env NCPUS="$t" MATMUL_THREADS="$t" bash perf/on_core.sh "$@" > "$f" 2>&1
    grep -E '^(RESULT|real)' "$f" || echo "  !! no RESULT line, check $f"
}

run_variant() { # label threads kernel size
    local label=$1 t=$2 kernel=$3 size=$4
    local iters
    iters=$(iters_for "$size" "$t")
    job "$OUT/${label}_time.txt" "$t" run bash -c 'time -p "$@"' _ ./perf/profile "$kernel" "$size" "$iters"
    for level in 1 2; do
        for k in $(seq 1 "$TRIALS"); do
            job "$OUT/${label}_l${level}_t${k}.txt" "$t" toplev "$level" ./perf/profile "$kernel" "$size" "$iters"
        done
    done
}

if [ ! -x perf/profile ]; then
    echo "missing perf/profile: run 'make profile' first"
    exit 1
fi

for size in "${SIZES[@]}"; do
    run_variant "naive-n${size}" 1 matmul "$size"
    for t in "${THREADS[@]}"; do
        run_variant "thread${t}-n${size}" "$t" matmul_thread "$size"
    done
done

echo "[$(date +%T)] done; now run: python3 perf/summarize.py && python3 perf/plot.py"
