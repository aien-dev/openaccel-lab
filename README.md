# OpenAccel Lab

A vendor-neutral, execution-verified conformance and performance lab for accelerator kernels.

Each workload in `workloads/` has one float64 reference, a written tolerance contract and seeded
inputs that are byte-identical for every entrant. The entrants are **OpenCL C, Triton, HIP, CUDA
and Omega**. None of them referees the others. CI runs every entrant that a free GitHub runner can
host, judges it against the reference, and publishes the board:

**Board: https://aien-dev.github.io/openaccel-lab/**

Wrong kernels are part of the lab. Each control in `controls/` has one defect written into every
entrant: an off-by-one index, an exclusive scan where an inclusive one belongs, a reduction that
drops its last element, or a softmax without the max subtraction. The gate fails if any control
passes on any backend.

## What the numbers mean

- **CI rows are CPU evidence.** On the runners, OpenCL runs on PoCL, Triton in its interpreter, HIP
  on HIP-CPU and Omega as native AArch64. The CUDA source is compiled with nvcc but not run. Every
  CI timing is labelled host CPU wall clock. These timings say nothing about GPU performance.
- **GPU rows are receipts.** They live in `receipts/`, each with its machine, driver, date and who
  ran it. Rented-GPU rows imported from aien-dev/omega-gpu-bench are marked as imported.
- **Timings are distributions.** The board shows min / p10 / p25 / median / p75 / p90 / max and the
  sample count. It never shows a mean.
- **Only PASS counts.** `COMPILED`, `NOT_RUN` and `NO_SOURCE` never count as passing.

## Use it

```sh
git clone https://github.com/aien-dev/openaccel-lab && cd openaccel-lab
sh tools/fetch.sh all                # pinned omc, PoCL, HIP-CPU, hipify-perl; sha256-checked
export OCL_ICD_VENDORS=$PWD/.oal/icd TRITON_INTERPRET=1
bin/oal-gen vecadd --seed 1 > case.bin                 # one case, docs/CONTAINER.md
bin/oal-run opencl vecadd < case.bin > out.bin         # one entrant, one case
bin/oal-check vecadd case.bin out.bin                  # one verdict line; exit 0 PASS, 1 FAIL
bin/oal-matrix --controls > results.jsonl              # everything; docs/RESULTS.md
bin/oal-gate results.jsonl                             # exit 1 on any failure or passing control
bin/oal-shrink omega gemm_offbyone                     # smallest shape that still fails
```

Each tool does one job. Data goes to stdout, diagnostics to stderr, and exit codes mean what the
header of each tool says. To add a workload or an entrant, see `docs/AUTHORING.md`. The tolerance
rules, with their Wolfram checks, are in `docs/TOLERANCE.md`.

## GPU runs

`runners/cuda/run_all.sh OUT.jsonl` runs the CUDA entrant of every workload on the local NVIDIA
GPU and labels the lines with the GPU, compute capability and driver. Rows from AMD and Intel GPUs
are welcome as receipts in the same format.

## Limits

- HIP-CPU is archived upstream and does not implement every HIP feature. Workloads stay inside
  what it supports (see `docs/AUTHORING.md`).
- The Triton interpreter checks semantics, not Triton's GPU code generation.
- The lab takes source code only. It never inspects or translates NVIDIA binaries, SASS or other
  generated machine artifacts.

## Connected repos

- [aien-dev/omega-gpu-bench](https://github.com/aien-dev/omega-gpu-bench) at `be6e65ac`: rented-GPU receipts and their board. Its receipts are imported unchanged under `receipts/imported/`.
- [aien-dev/omega-kernel-template](https://github.com/aien-dev/omega-kernel-template) at `23a04c9f`: the single-kernel Omega check on AArch64 and SPIR-V/PoCL. Its PoCL release asset is the lab's arm64 OpenCL runtime.
- [aien-dev/omega-mv](https://github.com/aien-dev/omega-mv) at `369dbc96`: the Omega compiler used by the Omega entrant.

Pins are in `toolchain.lock`, and outside code is listed in `DEPENDENCIES.md`. License: Apache-2.0.
