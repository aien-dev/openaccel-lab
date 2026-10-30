# SPDX-License-Identifier: Apache-2.0
"""Backends: how each entrant is built and run. Every entrant is a program that reads one case
container on stdin and writes the output tensors on stdout (docs/CONTAINER.md); this module only
builds the program, runs it with a timeout and reads back stdout, the exit status and the
"oal-time-ns" / "oal-version" lines on stderr.

Statuses: PASS / FAIL come later from oal.check; here a run ends RAN, COMPILE_FAIL, RUNTIME_FAIL,
TIMEOUT, COMPILED (built, deliberately not run: CUDA without a GPU), NOT_RUN (toolchain or platform
missing) or NO_SOURCE (the workload has no entrant for this backend).

INTERIM host glue (Python). Successor: oal-run in seeded Omega (fork/execprog/pipefds/waitpid
host words exist in omc 369dbc96).
"""
import hashlib
import os
import platform
import shutil
import subprocess
import sys

from . import manifest

ROOT = manifest.ROOT
TOOLS = os.path.join(ROOT, ".oal")
BACKENDS = ("omega", "opencl", "triton", "hip", "cuda")


def _sha(path):
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _workdir(backend, name):
    d = os.path.join(TOOLS, "build", backend, name)
    os.makedirs(d, exist_ok=True)
    return d


def _scratch_args(m, params):
    return [str(max(1, manifest.nbytes(s, params))) for s in m.get("scratch", [])]


def _sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, **kw)


def omc_path():
    return os.environ.get("OAL_OMC", os.path.join(TOOLS, "tools", "omc"))


def probe(backend):
    """(available, detail): whether this machine can build/run the backend at all."""
    if backend == "omega":
        if platform.system() != "Linux" or platform.machine() not in ("aarch64", "arm64"):
            return False, "Omega program mode emits Linux AArch64 only; this host is %s %s" % (
                platform.system(), platform.machine())
        p = omc_path()
        return (os.access(p, os.X_OK), "omc %s sha256 %s" % (p, _sha(p)) if os.path.exists(p) else "omc missing at %s" % p)
    if backend == "opencl":
        try:
            import pyopencl  # noqa: F401
        except ImportError:
            return False, "pyopencl not installed"
        return True, "pyopencl"
    if backend == "triton":
        try:
            import triton  # noqa: F401
            import torch  # noqa: F401
        except ImportError:
            return False, "triton or torch not installed"
        return True, "triton"
    if backend == "hip":
        inc = os.environ.get("OAL_HIPCPU", os.path.join(TOOLS, "hip-cpu", "include"))
        hip = os.environ.get("OAL_HIPIFY", os.path.join(TOOLS, "tools", "hipify-perl"))
        if not os.path.isdir(inc) or not os.path.exists(hip):
            return False, "HIP-CPU headers (%s) or hipify-perl (%s) missing" % (inc, hip)
        return True, "HIP-CPU %s, hipify-perl sha256 %s" % (inc, _sha(hip))
    if backend == "cuda":
        nvcc = shutil.which(os.environ.get("OAL_NVCC", "nvcc"))
        if not nvcc:
            return False, "nvcc not found"
        return True, nvcc
    raise ValueError(backend)


def _build_omega(m, wd):
    src = os.path.join(m["_dir"], m["backends"]["omega"]["src"])
    unit = os.path.join(wd, "unit.om")
    with open(unit, "wb") as f:
        f.write(open(os.path.join(ROOT, "lib", "omega", "oal.om"), "rb").read())
        f.write(open(src, "rb").read())
    exe = os.path.join(wd, "prog")
    with open(unit, "rb") as fi, open(exe, "wb") as fo:
        r = subprocess.run([omc_path()], stdin=fi, stdout=fo, stderr=subprocess.PIPE)
    if r.returncode != 0:
        os.unlink(exe)
        return None, r.stderr.decode(errors="replace").strip() or "omc exit %d" % r.returncode
    os.chmod(exe, 0o755)
    return [exe], "omc exit 0, unit sha256 %s" % _sha(unit)


def _cuda_unit(m, wd, hip):
    src = os.path.join(m["_dir"], m["backends"]["cuda"]["src"])
    unit = os.path.join(wd, "prog.cu")
    with open(unit, "wb") as f:
        f.write(open(src, "rb").read())
        f.write(b"\n")
        f.write(open(os.path.join(ROOT, "runners", "cuda", "host.cu"), "rb").read())
    if not hip:
        return unit, None
    hipify = os.environ.get("OAL_HIPIFY", os.path.join(TOOLS, "tools", "hipify-perl"))
    out = os.path.join(wd, "prog.hip.cpp")
    # HIP-CPU has no triple-chevron launch: ask hipify for hipLaunchKernelGGL calls.
    r = _sh(["perl", hipify, "-quiet-warnings", "-hip-kernel-execution-syntax", unit])
    if r.returncode != 0:
        return None, "hipify-perl: " + r.stderr.decode(errors="replace")
    with open(out, "wb") as f:
        f.write(r.stdout)
    return out, None


def _build_hip(m, wd):
    unit, err = _cuda_unit(m, wd, hip=True)
    if unit is None:
        return None, err
    inc = os.environ.get("OAL_HIPCPU", os.path.join(TOOLS, "hip-cpu", "include"))
    exe = os.path.join(wd, "prog")
    cxx = os.environ.get("CXX", "g++")
    extra = os.environ.get("OAL_HIP_FLAGS", "").split()
    # HIP-CPU e112c935 external/libco/aarch64.inl calls VALGRIND_STACK_REGISTER even when valgrind.h is
    # absent; define it away (no-op on x86, where that file is not used).
    shim = ["-DVALGRIND_STACK_REGISTER(a,b)=0"]
    r = _sh([cxx, "-std=c++17", "-O2", "-I", inc] + shim + extra + ["-x", "c++", unit, "-o", exe, "-ltbb", "-lpthread"])
    if r.returncode != 0:
        return None, r.stderr.decode(errors="replace")[-4000:]
    return [exe], "%s -std=c++17 -O2 HIP-CPU, hipified unit sha256 %s" % (cxx, _sha(unit))


def _build_cuda(m, wd):
    unit, _ = _cuda_unit(m, wd, hip=False)
    nvcc = shutil.which(os.environ.get("OAL_NVCC", "nvcc"))
    arch = os.environ.get("OAL_CUDA_ARCH", "sm_80")
    exe = os.path.join(wd, "prog")
    stamp = os.path.join(wd, "prog.stamp")
    want = "%s %s" % (arch, _sha(unit))
    # Reuse a build of the identical unit for the same arch, so a GPU run can be split into a build
    # phase and a run phase (no compiler inside a GPU lock).
    if os.path.exists(exe) and os.path.exists(stamp) and open(stamp).read().split("\n")[0] == want:
        return [exe], open(stamp).read().split("\n", 1)[1].strip() + " (reused build)"
    r = _sh([nvcc, "-O2", "-arch=" + arch, "-o", exe, unit])
    if r.returncode != 0:
        return None, r.stderr.decode(errors="replace")[-4000:]
    v = _sh([nvcc, "--version"]).stdout.decode().strip().splitlines()[-1]
    log = "nvcc -O2 -arch=%s (%s)" % (arch, v)
    with open(stamp, "w") as f:
        f.write(want + "\n" + log + "\n")
    return [exe], log


_BUILT = {}


def build(backend, m):
    """(argv or None, log). argv is the entrant program without scratch arguments; one build per
    (backend, workload) per process."""
    key = (backend, m["name"])
    if key not in _BUILT:
        _BUILT[key] = _build(backend, m)
    return _BUILT[key]


def _build(backend, m):
    wd = _workdir(backend, m["name"])
    if backend == "omega":
        return _build_omega(m, wd)
    if backend == "opencl":
        return [sys.executable, "-m", "oal.entrants.opencl", m["name"]], "pyopencl host"
    if backend == "triton":
        return [sys.executable, "-m", "oal.entrants.triton", m["name"]], "triton host"
    if backend == "hip":
        return _build_hip(m, wd)
    if backend == "cuda":
        return _build_cuda(m, wd)
    raise ValueError(backend)


def source_for(backend, m):
    b = m["backends"]
    if backend == "hip":
        return b.get("hip") or ("cuda" in b and {"from": "cuda"})
    return b.get(backend)


def execute(backend, m, params, blob, timeout):
    """Build and run one entrant on one case. Returns a dict; outputs (bytes) only when status RAN."""
    res = {"status": None, "times_ns": [], "versions": [], "log": ""}
    if not source_for(backend, m):
        res["status"] = "NO_SOURCE"
        return res
    ok, detail = probe(backend)
    res["toolchain"] = detail
    if not ok:
        res["status"] = "NOT_RUN"
        res["log"] = detail
        return res
    argv, log = build(backend, m)
    res["build"] = log
    if argv is None:
        res["status"] = "COMPILE_FAIL"
        res["log"] = log
        return res
    if backend == "cuda" and os.environ.get("OAL_CUDA_RUN") != "1":
        res["status"] = "COMPILED"
        res["log"] = "built; not run (no GPU on this runner; OAL_CUDA_RUN unset)"
        return res
    env = dict(os.environ)
    env["PYTHONPATH"] = ROOT + os.pathsep + env.get("PYTHONPATH", "")
    try:
        r = subprocess.run(argv + _scratch_args(m, params), input=blob, capture_output=True,
                           timeout=timeout, env=env, cwd=ROOT)
    except subprocess.TimeoutExpired:
        res["status"] = "TIMEOUT"
        res["log"] = "no exit within %ds" % timeout
        return res
    err = r.stderr.decode(errors="replace")
    other = []
    for line in err.splitlines():
        if line.startswith("oal-time-ns "):
            res["times_ns"].append(int(line.split()[1]))
        elif line.startswith("oal-version "):
            res["versions"].append(line[len("oal-version "):])
        else:
            other.append(line)
    res["log"] = "\n".join(other)[-4000:]
    res["exit"] = r.returncode
    if r.returncode != 0:
        res["status"] = "COMPILE_FAIL" if (backend == "opencl" and r.returncode == 2) else "RUNTIME_FAIL"
        if backend == "omega" and r.returncode > 100:
            res["log"] = ("trap %d. " % (r.returncode - 100)) + res["log"]
        return res
    res["status"] = "RAN"
    res["outputs"] = r.stdout
    return res
