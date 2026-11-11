#!/bin/bash
# Top-down profiling of every kernel at 3 input sizes, levels 1 and 2, N trials.
#
# Run from ~/lab1 on the login node after `make profile`:
#   nohup bash perf/run_toplev.sh 3 > results/run.log 2>&1 &
#
# Each (config, level, trial) is a separate srun job on the ece5755 partition
# (Sapphire Rapids, 5-minute limit), pinned to core 0. Output goes to
# results/<label>_l<level>_t<trial>.txt; finished runs are skipped on re-run,
# so the script can be restarted after an interruption.
set -u
TRIALS=${1:-3}
OUT=results
TOPLEV="python3 /classes/ece5755/pmu-tools/toplev.py --core S0-C0 -v --no-desc --force-cpu spr"
mkdir -p "$OUT"

# label  kernel  size  iters  [conv: channels filters kernelSize]
# Iteration counts calibrated on the ece5755 node for ~10 s per run (-O0 build).
CONFIGS=(
    "relu_1e3      relu    1000      3000000"
    "relu_1e5      relu    100000    9000"
    "relu_1e7      relu    10000000  90"
    "softmax_1e3   softmax 1000      1200000"
    "softmax_1e5   softmax 100000    12000"
    "softmax_1e7   softmax 10000000  100"
    "linear_256    linear  256       50000"
    "linear_1024   linear  1024      3000"
    "linear_4096   linear  4096      200"
    "matmul_64     matmul  64        12500"
    "matmul_256    matmul  256       200"
    "matmul_1024   matmul  1024      3"
    "conv_28c1f32k3   conv 28  8000  1 32 3"
    "conv_64c3f16k3   conv 64  1200  3 16 3"
    "conv_224c3f64k5  conv 224 10    3 64 5"
)

run() { # file, level, profile args...
    local f=$1 level=$2
    shift 2
    if [ -s "$f" ] && grep -q '^RESULT' "$f"; then
        return
    fi
    echo "[$(date +%T)] $f"
    srun -p ece5755 --time=00:05:00 --chdir="$PWD" \
        $TOPLEV -l"$level" taskset -c 0 ./perf/profile "$@" > "$f" 2>&1
    grep '^RESULT' "$f" || echo "  !! no RESULT line, check $f"
}

for cfg in "${CONFIGS[@]}"; do
    read -r label kernel size iters extra <<< "$cfg"
    for level in 1 2; do
        for t in $(seq 1 "$TRIALS"); do
            run "$OUT/${label}_l${level}_t${t}.txt" "$level" $kernel $size $iters $extra
        done
    done
    # Setup-only baseline (iters = 0): shows how much of a run is not the kernel
    run "$OUT/${label}-setup_l1_t1.txt" 1 $kernel $size 0 $extra
done

echo "[$(date +%T)] done; now run: python3 perf/summarize.py"
