#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# runners/cuda/run_all.sh OUT.jsonl [oal-matrix options...]
# Run the CUDA entrant of every workload and control on the GPU of this machine and write
# oal-result/1 lines to OUT.jsonl. Needs nvcc, nvidia-smi and one NVIDIA GPU; the lab never runs
# this on CI. The records carry the GPU name, compute capability and driver in machine.runner and
# the label below, so the board can tell them from CPU rows.
# Exit: oal-matrix's status (0 when every line was written), 2 on a usage error, 4 without a GPU.
# INTERIM host glue (POSIX sh). Successor: an Omega runner script once omega-mv has one.
set -eu
[ $# -ge 1 ] || { echo "usage: run_all.sh OUT.jsonl [oal-matrix options]" >&2; exit 2; }
out=$1; shift
here=$(cd "$(dirname "$0")/../.." && pwd)
q=$(nvidia-smi --query-gpu=name,compute_cap,driver_version --format=csv,noheader 2>/dev/null | sed -n 1p) || true
[ -n "$q" ] || { echo "run_all.sh: no NVIDIA GPU visible" >&2; exit 4; }
name=$(echo "$q" | cut -d, -f1 | sed 's/^ *//;s/ *$//')
cc=$(echo "$q" | cut -d, -f2 | tr -d ' .')
drv=$(echo "$q" | cut -d, -f3 | tr -d ' ')
export OAL_CUDA_RUN=1
export OAL_CUDA_ARCH="${OAL_CUDA_ARCH:-sm_$cc}"
export OAL_RUNNER_LABEL="${OAL_RUNNER_LABEL:-$name sm_$cc driver $drv}"
export OAL_TIMING_LABEL="${OAL_TIMING_LABEL:-GPU: launch + cudaDeviceSynchronize, host steady_clock, per repeat}"
export OAL_EVIDENCE="${OAL_EVIDENCE:-QUALIFIED-on-$name candidate: GPU run, driver $drv, $(date -u +%Y-%m-%dT%H:%MZ)}"
echo "run_all.sh: $OAL_RUNNER_LABEL arch $OAL_CUDA_ARCH" >&2
exec python3 "$here/bin/oal-matrix" --backends cuda --controls "$@" > "$out"
