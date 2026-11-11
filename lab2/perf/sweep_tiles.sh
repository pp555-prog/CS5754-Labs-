#!/bin/bash
# Runtime sweep of matmul_blocking tile sizes against the naive matmul baseline.
#
# Run from ~/lab2 on the login node:
#   nohup bash perf/sweep_tiles.sh 2 > results/sweep.log 2>&1 &
#
# Builds one perf/profile binary per tile configuration (BLOCK_SIZE_I/J/K are
# compile-time macros), then times every (variant, size) on the ece5755 node
# (Sapphire Rapids, 5-minute limit), N trials each. The node is shared with
# other users, so each run is pinned to whichever core Slurm assigned
# (perf/on_core.sh) rather than core 0, and the core is recorded per row.
# Appends one CSV row per run to results/sweep.csv:
#   variant,I,J,K,size,iters,trial,total_s,per_call_us,cpu
# Finished (variant, size, trial) rows are skipped on re-run.
set -u
TRIALS=${1:-2}
OUT=results
CSV=$OUT/sweep.csv
mkdir -p "$OUT" perf/bin
[ -s "$CSV" ] || echo "variant,I,J,K,size,iters,trial,total_s,per_call_us,cpu" > "$CSV"

CC_FLAGS="-I. -lm -I/usr/include/hdf5/serial -L/usr/lib/x86_64-linux-gnu/hdf5/serial -lhdf5"
SRC="perf/profile.c utils/data_utils.c kernel/conv.c kernel/functional.c kernel/matrix_ops.c kernel/linear.c kernel/nn.c kernel/attention.c"

# Tile configurations: tag I J K ("naive" runs the Lab 1 matmul)
VARIANTS=(
    "naive      0   0   0"
    "t169_32_9  169 32  9"
    "t32_32_32  32  32  32"
    "t64_64_64  64  64  64"
    "t32_256_32 32  256 32"
)

# size iters: ~5-30 s per naive run at -O0 (about 3 ns per multiply-add)
# 64: 3 x 16 KB matrices fit in the 48 KB L1d; 512: 3 MB total, exceeds the 2 MB L2;
# 1536: 27 MB total, about the size of L3
SIZES=(
    "64   20000"
    "512  20"
    "1536 1"
)

for v in "${VARIANTS[@]}"; do
    read -r tag bi bj bk <<< "$v"
    if [ "$tag" = naive ]; then
        gcc -o perf/bin/profile_$tag $SRC $CC_FLAGS || exit 1
    else
        gcc -o perf/bin/profile_$tag -DBLOCK_SIZE_I=$bi -DBLOCK_SIZE_J=$bj -DBLOCK_SIZE_K=$bk $SRC $CC_FLAGS || exit 1
    fi
done

for s in "${SIZES[@]}"; do
    read -r size iters <<< "$s"
    for v in "${VARIANTS[@]}"; do
        read -r tag bi bj bk <<< "$v"
        kernel=matmul_blocking
        [ "$tag" = naive ] && kernel=matmul
        for t in $(seq 1 "$TRIALS"); do
            # Skip only completed rows (failed runs have empty timing fields)
            if grep -q "^$tag,.*,$size,[0-9]*,$t,[0-9]" "$CSV"; then
                continue
            fi
            raw=$(srun -p ece5755 --time=00:05:00 --chdir="$PWD" \
                bash perf/on_core.sh run ./perf/bin/profile_$tag $kernel $size $iters 2>&1)
            line=$(grep '^RESULT' <<< "$raw")
            if [ -z "$line" ]; then
                echo "!! $tag size=$size trial=$t failed:"
                echo "$raw"
                continue
            fi
            total=$(sed -n 's/.*total_s=\([0-9.]*\).*/\1/p' <<< "$line")
            per=$(sed -n 's/.*per_call_us=\([0-9.]*\).*/\1/p' <<< "$line")
            cpu=$(sed -n 's/^PINNED cpu=\([0-9]*\).*/\1/p' <<< "$raw")
            echo "$tag,$bi,$bj,$bk,$size,$iters,$t,$total,$per,$cpu" | tee -a "$CSV"
        done
    done
done

echo "done; per-call means:"
awk -F, 'NR > 1 && $9 != "" { s[$1 " size=" $5] += $9; n[$1 " size=" $5]++ }
         END { for (k in s) printf "%-28s %14.1f us\n", k, s[k] / n[k] }' "$CSV" | sort -k2,2 -k3,3n
