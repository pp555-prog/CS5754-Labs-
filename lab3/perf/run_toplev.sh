#!/bin/bash
# Top-down profiling and timing of matmul (Lab 1 baseline) vs. matmul_sparse
# over several input sizes and densities (fraction of nonzeros in A).
#
# Run from ~/lab3 on the login node, after `make profile`:
#   nohup bash perf/run_toplev.sh 1 > results/toplev.log 2>&1 &
#   tail -f results/toplev.log
#
# Each (kernel, size, density) gets:
#   <label>_l1_t<k>.txt, <label>_l2_t<k>.txt  toplev levels 1 and 2, TRIALS trials
#   <label>-setup_l1_t1.txt                    iters = 0: setup alone (input generation,
#                                              CSR build), to show the loop dominates
#   <label>_time.txt                           un-profiled run under `time -p` (real)
# Each is a separate srun job on the ece5755 partition (Sapphire Rapids,
# 5-minute limit). The node is shared, so perf/on_core.sh pins the program to
# whichever core Slurm assigned and toplev counts only that core.
# Finished runs are skipped on re-run, so the script can be restarted.
# Then: python3 perf/summarize.py && python3 perf/plot.py
set -u
TRIALS=${1:-1}
TARGET_S=${TARGET_S:-6}    # aim for ~6 s of kernel time per run
OUT=results
mkdir -p "$OUT"

SIZES=(128 512 1536)
DENSITIES=(1 10 50 100)    # % nonzeros in A (sparsity = 100 - density)

# Iterations so one run's kernel loop takes ~TARGET_S, from -O0 costs measured
# in Labs 1-2: naive ~3.3 ns per multiply-add (all n^3, zeros included);
# sparse ~1.8 ns per multiply-add on the nonzeros only (n^3 * density), plus
# ~1 ns per output element for allocating and zeroing C.
iters_for() { # kernel size density%
    awk -v k="$1" -v n="$2" -v d="$3" -v t="$TARGET_S" 'BEGIN {
        if (k == "matmul") est = n * n * n * 3.3e-9;
        else               est = n * n * n * (d / 100) * 1.8e-9 + n * n * 1e-9;
        it = int(t / est + 0.5); if (it < 1) it = 1; print it }'
}

done_ok() { [ -s "$1" ] && grep -q '^RESULT' "$1"; }

profile() { # file, level, profile args...
    local f=$1 level=$2
    shift 2
    done_ok "$f" && return
    echo "[$(date +%T)] $f"
    srun -p ece5755 --time=00:05:00 --chdir="$PWD" \
        bash perf/on_core.sh toplev "$level" ./perf/profile "$@" > "$f" 2>&1
    grep '^RESULT' "$f" || echo "  !! no RESULT line, check $f"
}

timed() { # file, profile args...
    local f=$1
    shift
    done_ok "$f" && return
    echo "[$(date +%T)] $f"
    srun -p ece5755 --time=00:05:00 --chdir="$PWD" \
        bash perf/on_core.sh run bash -c 'time -p "$@"' _ ./perf/profile "$@" > "$f" 2>&1
    grep -E '^(RESULT|real)' "$f" || echo "  !! no RESULT line, check $f"
}

if [ ! -x perf/profile ]; then
    echo "missing perf/profile: run 'make profile' first"
    exit 1
fi

for size in "${SIZES[@]}"; do
    for d in "${DENSITIES[@]}"; do
        density=$(awk -v d="$d" 'BEGIN { print d / 100 }')
        for kernel in matmul matmul_sparse; do
            variant=naive
            [ "$kernel" = matmul_sparse ] && variant=sparse
            label="${variant}-n${size}-d${d}"
            iters=$(iters_for "$kernel" "$size" "$d")
            timed "$OUT/${label}_time.txt" "$kernel" "$size" "$density" "$iters"
            for level in 1 2; do
                for t in $(seq 1 "$TRIALS"); do
                    profile "$OUT/${label}_l${level}_t${t}.txt" "$level" "$kernel" "$size" "$density" "$iters"
                done
            done
            profile "$OUT/${label}-setup_l1_t1.txt" 1 "$kernel" "$size" "$density" 0
        done
    done
done

echo "[$(date +%T)] done; now run: python3 perf/summarize.py && python3 perf/plot.py"
