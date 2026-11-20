#!/bin/bash
# Run a command pinned to NCPUS of the CPUs Slurm assigned to this job,
# optionally under toplev counting only those cores. Used inside srun on the
# shared ece5755 node, where core 0 may belong to another user's job.
#
# Usage (inside srun, e.g. srun -p ece5755 -c 8 ...):
#   [NCPUS=n] perf/on_core.sh run <cmd...>              pin <cmd> to n CPUs
#   [NCPUS=n] perf/on_core.sh toplev <level> <cmd...>   same, profiled with toplev -l<level>
# NCPUS defaults to 1 (the naive matmul); use NCPUS = MATMUL_THREADS for
# matmul_thread so every thread has its own core and only those cores are counted.
#
# CPUs on distinct physical cores are chosen first (SMT siblings share a core's
# pipeline, so two threads on siblings would not be two cores of parallelism).
# Prints a PINNED line with the chosen CPUs, the job's allowed list, and the
# toplev core names (S<socket>-C<core>) so every result file records them.
set -u
TOPLEV="python3 /classes/ece5755/pmu-tools/toplev.py"
NCPUS=${NCPUS:-1}

# Expand this job's allowed CPU list (e.g. "17,49" or "16-23") into one per line
allowed=$(awk '/^Cpus_allowed_list/ { print $2 }' /proc/self/status)
cpus=$(echo "$allowed" | tr ',' '\n' | awk -F- '{ if (NF == 2) for (c = $1; c <= $2; c++) print c; else print $1 }')

# Pick NCPUS CPUs: first one per distinct physical core, then SMT siblings if needed
chosen=()
cores=()
seen=" "
for pass in distinct siblings; do
    for cpu in $cpus; do
        [ ${#chosen[@]} -ge "$NCPUS" ] && break
        topo=/sys/devices/system/cpu/cpu$cpu/topology
        core="S$(cat $topo/physical_package_id)-C$(cat $topo/core_id)"
        case " ${chosen[*]} " in *" $cpu "*) continue ;; esac
        if [ "$pass" = distinct ]; then
            case "$seen" in *" $core "*) continue ;; esac
        fi
        chosen+=("$cpu")
        case "$seen" in *" $core "*) ;; *) cores+=("$core"); seen="$seen$core " ;; esac
    done
done
if [ ${#chosen[@]} -lt "$NCPUS" ]; then
    echo "on_core.sh: asked for $NCPUS CPUs but the job only has: $allowed" >&2
    exit 2
fi
cpu_list=$(IFS=,; echo "${chosen[*]}")
core_list=$(IFS=,; echo "${cores[*]}")
echo "PINNED cpus=$cpu_list allowed=$allowed cores=$core_list ncores=${#cores[@]}"
[ ${#cores[@]} -lt "$NCPUS" ] && echo "WARNING: only ${#cores[@]} physical cores for $NCPUS CPUs (SMT siblings used)"

mode=$1
shift
case $mode in
    run)
        exec taskset -c "$cpu_list" "$@"
        ;;
    toplev)
        level=$1
        shift
        # TOPLEV_EXTRA: extra toplev options, e.g. --nodes '!+Backend_Bound*/3,+MUX'
        # to measure only a subtree (fewer events, so less multiplexing at -l3).
        # set -f keeps the * in node patterns from being glob-expanded.
        set -f
        exec $TOPLEV --core "$core_list" -l"$level" -v --no-desc --force-cpu spr ${TOPLEV_EXTRA:-} taskset -c "$cpu_list" "$@"
        ;;
    *)
        echo "unknown mode: $mode" >&2
        exit 2
        ;;
esac
