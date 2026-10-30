# Result lines (`oal-result/1`)

`bin/oal-matrix` writes one JSON object per line per (workload, backend, case, seed). `bin/oal-gate`
and `bin/oal-board` read only these lines.

| field | meaning |
|---|---|
| `workload`, `family`, `control`, `expect` | from the manifest; `control` lines are expected to fail |
| `backend` | `omega`, `opencl`, `triton`, `hip` or `cuda` |
| `params`, `seed` | the case; `bin/oal-gen WORKLOAD --seed S --set k=v ...` rebuilds it byte for byte |
| `case_sha256`, `case_bytes` | sha256 and size of the container the entrant received |
| `status` | `PASS`, `FAIL` (ran, outside tolerance), `RUNTIME_FAIL`, `COMPILE_FAIL`, `TIMEOUT`, `COMPILED` (CUDA built but no GPU run), `NOT_RUN` (toolchain absent), `NO_SOURCE` (no entrant file) |
| `err_ratio` | largest `abs(got - ref) / bound` over every output element (see TOLERANCE.md) |
| `outputs` | per output: `pass`, `failed` count, `elements`, `first_bad` (index, got, ref, bound) |
| `outputs_sha256` | sha256 of the entrant's whole stdout, for bitwise agreement across backends |
| `times_ns` | every timed launch, in order; never averaged |
| `timing_label` | what the times mean on this machine; CPU runners say so |
| `evidence` | evidence level and machine, for example `EXERCISED: CPU runner` |
| `machine` | os, arch, cpu, cpu count, `runner` label, CI run URL |
| `versions`, `toolchain`, `build` | runtime versions the entrant reported and the build command used |
| `lab_commit` | commit of this repository |

Only `PASS` counts toward the gate's "at least 3 backends per workload". `COMPILED`, `NOT_RUN` and
`NO_SOURCE` are never counted as passing.

GPU result lines live under `receipts/gpu/*.jsonl` and use the same schema, with `machine.runner`
naming the GPU, compute capability and driver, and `evidence` stating who ran them, when and how.
Imported receipts from aien-dev/omega-gpu-bench sit under `receipts/imported/` unchanged, with a
`SHA256SUMS` file.
