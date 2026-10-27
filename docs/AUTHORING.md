# Writing a workload

A workload is one directory `workloads/<name>/` (or `controls/<name>/` for a deliberately wrong
kernel). The directory name, the `name` field and the reference module agree. Everything an entrant
needs is in the directory; nothing outside it is workload-specific.

```
workloads/<name>/
  workload.json      manifest: shapes, generators, tolerance contract, entrant sources
  reference.py       reference(p, x): float64 numpy oracle, never an entrant
  omega.om           Omega entrant (kernel + main), compiled after lib/omega/oal.om
  kernel.cl          OpenCL C entrant (one or more __kernel functions)
  triton_kernel.py   Triton entrant: launch(p, t)
  kernel.cu          CUDA entrant: oal_launch(p, buf); the HIP entrant is this file through hipify-perl
```

## workload.json

| field | meaning |
|---|---|
| `name`, `family`, `summary`, `provenance` | identity; provenance says who wrote it and under what licence |
| `params` | integer parameters with their default values, in container order |
| `derived` | optional `{name: expression}` computed from params after overrides, appended in order |
| `cases` | list of parameter overrides run by `oal-matrix`; `{}` is the default case. Include an edge shape (a size of 1, a size that is not a multiple of any block). |
| `shrink` | per parameter `{"min": m, "multiple_of": k}`: how far `oal-shrink` may reduce it |
| `inputs` | tensors `{name, dtype, shape, gen}`; dtype one of f32 f64 i32 i8 u8; shape a list of expressions |
| `outputs` | tensors `{name, dtype, shape}` |
| `tolerance` | per output: `{"kind": "exact"}`, `{"kind": "bound", "rtol": expr, "atol": expr, "why": text}` or `{"kind": "int", "atol": n}` |
| `timing` | `{"warmup": w, "repeats": r}` |
| `backends` | `omega: {src}`, `opencl: {src, launches, scratch?, options?}`, `triton: {src}`, `cuda: {src}` |
| `scratch` | optional top-level list of tensors `{name, dtype, shape}` that OpenCL/Triton/CUDA entrants may use as device temporaries (zeroed for Triton/CUDA, undefined for OpenCL) |
| `expect` | controls only: `"fail"` |
| `reference_from` | controls only: the workload whose reference judges this control |

Expressions are Python arithmetic over parameter names plus `gamma(n)`, `u` (2^-24), `max`, `min`,
`log2`, `ceil`. `gamma(n)` is n·u/(1−n·u), Higham's bound on n roundings in binary32.

Generators (`gen`), all from splitmix64 with stream seed `seed ^ fnv1a64("<workload>/<input>")`:

- `{"kind":"uniform","lo":a,"hi":b}`: a + (b−a)·k·2^-24
- `{"kind":"normal4","mean":m,"std":s}`: Irwin–Hall sum of four uniforms, unit variance, then scaled
- `{"kind":"grid","klo":i,"khi":j,"shift":s}`: integer k in [i, j] times 2^-s, exact in f16 when |k| < 2048
- `{"kind":"int","lo":i,"hi":j}`: integers; required for i32, i8, u8 inputs
- `{"kind":"const","value":v}`, `{"kind":"iota","mod":m}`

Float scalars (alpha, eps) are written as constants in every source or passed as one-element f32
inputs; integer scalars are parameters.

## The tolerance contract

`exact` is for outputs made by at most one correctly rounded operation per element (add, mul,
copy, max, integer work). Everything else uses `bound` with a bound you can defend: state it in the
`why` field, and derive any numeric constant through the Wolfram Language evaluator, quoting the
query and result (`WolframLanguageEvaluator: <query> -> <result>`). The bound must hold for any
summation order and for fused multiply-add, because OpenCL C contracts `a*b+c` by default and GPU
compilers fuse; a recursive dot product of length n in any order is within gamma(n)·Σ|a_i b_i|
(Higham Thm 3.1/3.5), so pass that sum as the `scale` from `reference.py` and use `rtol: gamma(n)`.
For a transcendental, budget its documented accuracy: OpenCL `exp` ≤ 3 ulp (OpenCL C 3.0 §7.4),
CUDA `expf` ≤ 2 ulp (CUDA C Programming Guide, Mathematical Functions), so allow at least 4·u
relative per exp on top of the arithmetic bound. A bound so loose that a wrong kernel passes is a
defect: every workload family needs a control that fails it.

## reference.py

```python
import numpy as np

def reference(p, x):          # p: params dict, x: inputs as numpy arrays (manifest dtypes)
    a = x["a"].astype(np.float64)
    return {"y": a * 2.0}     # or {"y": (value, scale)} with an elementwise scale for "bound"
```

Compute in float64 from the exact input values, in the obvious order, with no float32 shortcut. The
reference shape must equal the manifest shape.

## Entrants

All entrants see the same case: parameters in manifest order (`params`, then `derived`), inputs in
manifest order, outputs in manifest order. Outputs start filled with 0xFF bytes (NaN for f32), so an
element the kernel never writes fails the check.

**Omega (`omega.om`).** Compiled as `lib/omega/oal.om` + `omega.om` by the pinned omc
(aien-dev/omega-mv 369dbc96), program mode, Linux AArch64. The prelude reads the case and gives:
`oal_begin`, `oal_p k` (parameter k, long), `oal_warm`, `oal_reps`, loaders that copy input k into a
pool and return its element count (`oal_fa..oal_fe k` into f32 pools `fa..fe`, `oal_ia/oal_ic k` into
int pools `ia/ic`, `oal_qa k` into byte pool `qa`, values −128..127), stores that copy a pool to output
k (`oal_so k n` from `fo`, `oal_sp k n` from `fp`, `oal_sio k n` from `io`, `oal_sqo k n` from `qo`),
`oal_tick ns` (one timing line), `oal_end` (writes stdout) and `oal_expf x` (binary64 e^x). Pools hold
4194304 elements; a workload may declare its own globals (names up to 8 characters, not clashing with
the prelude). Globals cannot be passed as arguments (error 19); functions use them by name. A
function takes at most 6 parameters. The pattern is `workloads/vecadd/omega.om`: a `kern` function
over globals, a main that loads, times `oal_reps` calls after `oal_warm` untimed ones with `monons`,
stores and ends. Language reference: `docs/one_layer/OMEGESE.md` in aien-dev/omega-mv; the pinned
compiler is the authority on what compiles.

**OpenCL (`kernel.cl` + `launches`).** Each launch is `{"kernel", "global": [exprs], "local":
[exprs] or null, "args": [...]}`, run in order once per timed repeat. An argument is a tensor name
(input, output or scratch buffer), a parameter expression (passed as `int`), `"f32:<expr>"` (a float),
or `"local:<dtype>:<count expr>"` (`__local` memory). No fast-math options.

**Triton (`triton_kernel.py`).** Define `launch(p, t)`; `t` maps every input, output and scratch name
to a contiguous CPU torch tensor. CI runs with `TRITON_INTERPRET=1`: keep shapes small and blocks
powers of two.

**CUDA and HIP (`kernel.cu`).** `#include <cuda_runtime.h>`, the kernels, then
`void oal_launch(const long long *p, void *const *buf)` that launches them; `buf` holds device
pointers for inputs, outputs, then scratch, in manifest order. The HIP entrant is this file through
`hipify-perl -hip-kernel-execution-syntax`, built against HIP-CPU (ROCm/HIP-CPU e112c935) with g++:
use static `__shared__` arrays only, no templates on kernels, no warp intrinsics, no dynamic
parallelism, and `__syncthreads()` only where every thread of the block reaches it. CUDA is built
with nvcc on CI and run only on rented GPUs (`runners/cuda/run_all.sh`).

## Controls

A control is a copy of a workload with the same defect written into every entrant (for example the
k loop stops one short). It sets `"expect": "fail"` and `"reference_from": "<workload>"`. `oal-gate`
fails CI if a control passes anywhere it ran.
