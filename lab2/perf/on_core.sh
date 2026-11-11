#!/bin/bash
# Run a command pinned to the core Slurm assigned to this job, optionally under
# toplev counting only that core. Used inside srun on the shared ece5755 node,
# where core 0 may belong to another user's job (taskset -c 0 then fails).
#
# Usage (inside srun):
#   perf/on_core.sh run <cmd...>              pin <cmd> to the assigned core
#   perf/on_core.sh toplev <level> <cmd...>   same, profiled with toplev -l<level>
#
# Prints a PINNED line with the logical CPU, the job's allowed CPU list, and
# the toplev core name (S<socket>-C<core>) so every result file records it.
set -u
TOPLEV="python3 /classes/ece5755/pmu-tools/toplev.py"

# First CPU of this job's allowed set (e.g. "17" from "17,49" or "17-18")
allowed=$(awk '/^Cpus_allowed_list/ { print $2 }' /proc/self/status)
cpu=${allowed%%[,-]*}
topo=/sys/devices/system/cpu/cpu$cpu/topology
core="S$(cat $topo/physical_package_id)-C$(cat $topo/core_id)"
echo "PINNED cpu=$cpu allowed=$allowed core=$core smt_siblings=$(cat $topo/thread_siblings_list)"

mode=$1
shift
case $mode in
    run)
        exec taskset -c "$cpu" "$@"
        ;;
    toplev)
        level=$1
        shift
        # TOPLEV_EXTRA: extra toplev options, e.g. --nodes '!+Backend_Bound*/3,+MUX'
        # to measure only a subtree (fewer events, so less multiplexing at -l3).
        # set -f keeps the * in node patterns from being glob-expanded.
        set -f
        exec $TOPLEV --core "$core" -l"$level" -v --no-desc --force-cpu spr ${TOPLEV_EXTRA:-} taskset -c "$cpu" "$@"
        ;;
    *)
        echo "unknown mode: $mode" >&2
        exit 2
        ;;
esac
