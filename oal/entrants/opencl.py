# SPDX-License-Identifier: Apache-2.0
"""OpenCL C entrant host: python -m oal.entrants.opencl WORKLOAD [SCRATCH_BYTES...] < case > outputs.

Builds workloads/<name>/<backends.opencl.src> on the first OpenCL CPU or GPU device (OAL_OPENCL_PLATFORM
picks a platform by substring), runs every launch in backends.opencl.launches, and times warmup+repeats
rounds of all launches plus clFinish. Output buffers start as 0xFF bytes so an unwritten element is NaN
(f32) or -1 (integers). Exit 2 on a build failure, 3 on a runtime failure.

INTERIM host glue (Python + pyopencl). Successor: the Omega OpenCL host of aien-dev/omega-mv
seed/spirv (opencl-host.c today), once it takes a case container.
"""
import os
import sys
import time

import numpy as np
import pyopencl as cl

from oal import case, manifest


def pick_device():
    want = os.environ.get("OAL_OPENCL_PLATFORM", "")
    for plat in cl.get_platforms():
        if want and want.lower() not in plat.name.lower():
            continue
        devs = plat.get_devices()
        if devs:
            return plat, devs[0]
    raise SystemExit("opencl: no platform matching %r" % want)


def main():
    name = sys.argv[1]
    m = manifest.load(name)
    spec = m["backends"]["opencl"]
    blob = sys.stdin.buffer.read()
    pvals, sizes, ins, warm, reps = case.unpack(blob)
    params = dict(zip(manifest.params_of(m).keys(), pvals))
    plat, dev = pick_device()
    sys.stderr.write("oal-version opencl %s | %s | %s\n" % (plat.name, plat.version, dev.name))
    ctx = cl.Context([dev])
    q = cl.CommandQueue(ctx)
    src = open(os.path.join(m["_dir"], spec["src"])).read()
    try:
        prog = cl.Program(ctx, src).build(options=spec.get("options", ""))
    except cl.RuntimeError as e:
        sys.stderr.write("opencl build failed: %s\n" % e)
        sys.exit(2)
    mf = cl.mem_flags
    bufs = {}
    for k, t in enumerate(m["inputs"]):
        bufs[t["name"]] = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=np.frombuffer(ins[k], np.uint8).copy())
    nin = len(m["inputs"])
    outs = []
    for k, t in enumerate(m["outputs"]):
        host = np.full(sizes[nin + k], 0xFF, np.uint8)
        bufs[t["name"]] = cl.Buffer(ctx, mf.READ_WRITE | mf.COPY_HOST_PTR, hostbuf=host)
        outs.append(host)
    for t in m.get("scratch", []):
        nb = max(1, manifest.nbytes(t, params))
        bufs[t["name"]] = cl.Buffer(ctx, mf.READ_WRITE, size=nb)
    launches = []
    for L in spec["launches"]:
        kern = getattr(prog, L["kernel"])
        args = []
        for a in L["args"]:
            if a in bufs:
                args.append(bufs[a])
            elif isinstance(a, str) and a.startswith("local:"):
                _, dt, count = a.split(":")
                args.append(cl.LocalMemory(manifest.DTYPES[dt][1] * int(manifest.evaluate(count, params))))
            elif isinstance(a, str) and a.startswith("f32:"):
                args.append(np.float32(manifest.evaluate(a[4:], params)))
            else:
                args.append(np.int32(manifest.evaluate(a, params)))
        gsz = tuple(int(manifest.evaluate(g, params)) for g in L["global"])
        lsz = tuple(int(manifest.evaluate(g, params)) for g in L["local"]) if L.get("local") else None
        launches.append((kern, gsz, lsz, args))
    try:
        for r in range(warm + reps):
            t0 = time.perf_counter_ns()
            for kern, gsz, lsz, args in launches:
                kern(q, gsz, lsz, *args)
            q.finish()
            t1 = time.perf_counter_ns()
            if r >= warm:
                sys.stderr.write("oal-time-ns %d\n" % (t1 - t0))
        for k, t in enumerate(m["outputs"]):
            cl.enqueue_copy(q, outs[k], bufs[t["name"]])
        q.finish()
    except cl.Error as e:
        sys.stderr.write("opencl runtime failure: %s\n" % e)
        sys.exit(3)
    w = sys.stdout.buffer
    for h in outs:
        w.write(h.tobytes())
        w.write(b"\0" * (case.pad8(h.size) - h.size))
    w.flush()


if __name__ == "__main__":
    main()
