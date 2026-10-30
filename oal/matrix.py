# SPDX-License-Identifier: Apache-2.0
"""One result record per (workload, case, backend): the JSONL schema in docs/RESULTS.md.

INTERIM host glue (Python). Successor: oal-matrix in seeded Omega.
"""
import os
import platform
import subprocess
import time

from . import case, check, manifest, ref, run

SCHEMA = "oal-result/1"


def runner_info():
    cpu = platform.processor() or ""
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith(("model name", "cpu part", "hardware")):
                    cpu = line.split(":", 1)[1].strip()
                    break
    except OSError:
        pass
    return {"os": platform.system(), "arch": platform.machine(), "cpu": cpu, "ncpu": os.cpu_count(),
            "ci": os.environ.get("GITHUB_ACTIONS") == "true",
            "runner": os.environ.get("OAL_RUNNER_LABEL", os.environ.get("RUNNER_NAME", "local")),
            "run_url": ("%s/%s/actions/runs/%s" % (os.environ["GITHUB_SERVER_URL"], os.environ["GITHUB_REPOSITORY"],
                        os.environ["GITHUB_RUN_ID"]) if os.environ.get("GITHUB_RUN_ID") else None)}


def lab_commit():
    try:
        return subprocess.run(["git", "-C", manifest.ROOT, "rev-parse", "HEAD"], capture_output=True,
                              text=True).stdout.strip() or None
    except OSError:
        return None


def cases_of(m):
    return [dict(c) for c in m.get("cases", [{}])]


def make_case(m, overrides, seed):
    params = manifest.params_of(m, overrides)
    tensors = case.inputs(m, params, seed)
    t = m["timing"]
    blob = case.pack(m, params, tensors, t.get("warmup", 2), t.get("repeats", 10))
    return params, tensors, blob


def run_one(backend, m, overrides, seed, timeout=600, refs_cache=None):
    params, tensors, blob = make_case(m, overrides, seed)
    rec = {"schema": SCHEMA, "workload": m["name"], "family": m.get("family"), "control": m["_control"],
           "expect": m.get("expect", "pass"), "backend": backend, "params": params, "seed": seed,
           "case_sha256": case.sha256(blob), "case_bytes": len(blob), "when": int(time.time()),
           "lab_commit": lab_commit(), "machine": runner_info(),
           "timing_label": os.environ.get("OAL_TIMING_LABEL") or m["timing"].get(
               "label", "host CPU wall clock; not GPU performance"),
           "evidence": os.environ.get("OAL_EVIDENCE", "EXERCISED: CPU runner")}
    r = run.execute(backend, m, params, blob, timeout)
    rec.update({k: r[k] for k in ("status", "times_ns", "versions", "log") if k in r})
    for k in ("toolchain", "build", "exit"):
        if k in r:
            rec[k] = r[k]
    if r["status"] != "RAN":
        return rec
    rec["outputs_sha256"] = case.sha256(r["outputs"])
    try:
        got = case.split_outputs(m, params, r["outputs"])
    except ValueError as e:
        rec["status"] = "RUNTIME_FAIL"
        rec["log"] = "output stream: %s" % e
        return rec
    key = (m["name"], case.sha256(blob))
    if refs_cache is not None and key in refs_cache:
        refs = refs_cache[key]
    else:
        refs = ref.compute(m, params, tensors)
        if refs_cache is not None:
            refs_cache[key] = refs
    v = check.compare(m, params, refs, got)
    rec["status"] = "PASS" if v["pass"] else "FAIL"
    rec["err_ratio"] = v["err_ratio"]
    rec["outputs"] = v["outputs"]
    return rec
