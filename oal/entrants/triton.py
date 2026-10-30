# SPDX-License-Identifier: Apache-2.0
"""Triton entrant host: python -m oal.entrants.triton WORKLOAD < case > outputs.

Imports workloads/<name>/<backends.triton.src> and calls its launch(p, t): p maps parameter names to
ints, t maps every input, output and scratch name to a CPU torch tensor. In CI TRITON_INTERPRET=1 is
set, so the kernel runs in Triton's numpy interpreter: a correctness entrant, never a GPU timing.
Output tensors start as 0xFF bytes (NaN for f32). Exit 3 on an exception in launch.

INTERIM host glue (Python + torch). Successor: none planned; Triton is a foreign entrant and its
host is part of what is measured.
"""
import importlib.util
import os
import sys
import time
import traceback

import numpy as np

from oal import case, manifest

TORCH_DT = {"f32": "float32", "f64": "float64", "i32": "int32", "i8": "int8", "u8": "uint8"}


def main():
    import torch
    import triton
    name = sys.argv[1]
    m = manifest.load(name)
    spec = m["backends"]["triton"]
    blob = sys.stdin.buffer.read()
    pvals, sizes, ins, warm, reps = case.unpack(blob)
    params = dict(zip(manifest.params_of(m).keys(), pvals))
    sys.stderr.write("oal-version triton %s torch %s interpret=%s\n" % (
        triton.__version__, torch.__version__, os.environ.get("TRITON_INTERPRET", "0")))
    path = os.path.join(m["_dir"], spec["src"])
    mod_spec = importlib.util.spec_from_file_location("oal_triton_" + name, path)
    mod = importlib.util.module_from_spec(mod_spec)
    mod_spec.loader.exec_module(mod)
    t = {}
    for k, s in enumerate(m["inputs"]):
        arr = np.frombuffer(ins[k], dtype=manifest.DTYPES[s["dtype"]][0]).reshape(manifest.shape(s, params)).copy()
        t[s["name"]] = torch.from_numpy(arr)
    for s in m["outputs"]:
        dims = manifest.shape(s, params)
        raw = np.full(manifest.nbytes(s, params), 0xFF, np.uint8)
        t[s["name"]] = torch.from_numpy(raw.view(manifest.DTYPES[s["dtype"]][0]).reshape(dims).copy())
    for s in m.get("scratch", []):
        t[s["name"]] = torch.zeros(manifest.shape(s, params), dtype=getattr(torch, TORCH_DT[s["dtype"]]))
    try:
        for r in range(warm + reps):
            t0 = time.perf_counter_ns()
            mod.launch(params, t)
            t1 = time.perf_counter_ns()
            if r >= warm:
                sys.stderr.write("oal-time-ns %d\n" % (t1 - t0))
    except Exception:
        traceback.print_exc(file=sys.stderr)
        sys.exit(3)
    w = sys.stdout.buffer
    for s in m["outputs"]:
        raw = t[s["name"]].contiguous().numpy().tobytes()
        w.write(raw)
        w.write(b"\0" * (case.pad8(len(raw)) - len(raw)))
    w.flush()


if __name__ == "__main__":
    main()
