# Dependencies

Everything below is outside code the lab runs. Entrant toolchains are the thing being measured,
so they are kept; the host glue is INTERIM and each piece names its Omega successor.

## INTERIM host glue (first-party, to be replaced by seeded Omega)

| file | language | job | Omega successor |
|---|---|---|---|
| `bin/oal-gen`, `oal/case.py`, `oal/manifest.py` | Python 3.12 | build the case container from a manifest and seed | `oal-gen` unit: splitmix64 and FNV-1a are integer-only and already expressible in seeded Omega |
| `bin/oal-check`, `oal/check.py`, `oal/ref.py`, `workloads/*/reference.py` | Python + numpy | float64 reference and tolerance check | `oal-check` unit with float64 references written in Omega; numpy stays only as an independent cross-check oracle |
| `bin/oal-run`, `oal/run.py`, `bin/oal-matrix`, `oal/matrix.py` | Python | build and launch entrants, write result lines | `oal-run` unit once seeded Omega has a process-spawn word |
| `bin/oal-gate`, `bin/oal-shrink`, `bin/oal-board` | Python + numpy | gate, minimise failures, fold the board data | Omega units over JSON lines |
| `oal/entrants/opencl.py` | Python + pyopencl | OpenCL host program | the OpenCL host path in aien-dev/omega-mv `seed/spirv/` |
| `oal/entrants/triton.py` | Python | Triton launcher (Triton is a Python DSL; this stays as long as Triton is an entrant) | none needed: it is the entrant's own host language |
| `runners/cuda/host.cu` | CUDA C++ | CUDA and HIP host program (also hipified for HIP) | stays C++: it is the CUDA/HIP entrant's own host language |
| `tools/fetch.sh`, `runners/cuda/run_all.sh` | POSIX sh | fetch pinned tools; run CUDA entrants on a GPU | Omega fetch-and-verify unit when seeded Omega has an HTTPS client word |
| `site/index.html` | HTML + JS | render the board in a browser | page generated from seeded Omega (as aienos.com pages already are) |

## Entrant toolchains and runtimes

| dependency | version / pin | what it does here | Omega replacement |
|---|---|---|---|
| Omega compiler `omc` | aien-dev/omega-mv `369dbc96`, sha256 `2c824805…` | compiles the Omega entrant to native AArch64 | it is the Omega entrant |
| PoCL (AArch64) | 7.2 at pocl `888c9774`, aien-dev/omega-kernel-template release `pocl-7.2-888c9774-noble-arm64.1`, sha256 `68892ec0…` | OpenCL C on the CPU of the arm64 runner | none: OpenCL is an entrant, not glue |
| PoCL (x86-64) | Ubuntu 24.04 `pocl-opencl-icd` | OpenCL C on the CPU of the x86 runner | none |
| clang-18, libLLVMSPIRVLib 18.1, ocl-icd | Ubuntu 24.04 packages | PoCL's compiler and the OpenCL loader | none |
| pyopencl | 2026.1.4 (PyPI) | OpenCL host API from Python | see `oal/entrants/opencl.py` above |
| Triton | 3.8.0 (PyPI), run with `TRITON_INTERPRET=1` | Triton entrant on CPU (the interpreter: correctness only, timings are not Triton GPU performance) | none: Triton is an entrant |
| PyTorch | 2.14.1 CPU wheel | tensors for the Triton interpreter | none |
| numpy | 2.5.4 (PyPI) | float64 reference arithmetic, quantiles | Omega float64 references (see above) |
| HIP-CPU | ROCm/HIP-CPU `e112c935` (MIT, archived upstream) | runs HIP source on the CPU with TBB | none: HIP is an entrant |
| oneTBB | Ubuntu 24.04 `libtbb-dev` | HIP-CPU's thread pool | none |
| hipify-perl | ROCm/HIPIFY `7d531fc2`, sha256 `3bd75514…` | turns each `kernel.cu` into HIP source, so the HIP entrant is the CUDA source hipified, not a second hand-written port | none |
| nvcc | Ubuntu 24.04 `nvidia-cuda-toolkit` on CI (compile only); CUDA 13.0 V13.0.88 on the GB10 | builds the CUDA entrant | none: CUDA is an entrant |
| GitHub Actions | checkout `11bd7190`, upload-artifact `ea165f8d`, download-artifact `9000827c`, upload-pages-artifact `fc324d35`, deploy-pages `368f8252` | CI and Pages | none |

Python packages come from PyPI and the PyTorch CPU index; everything else from GitHub or the
Ubuntu archive, at the pins in `toolchain.lock`.
