# SPDX-License-Identifier: Apache-2.0
"""Independent reference: workloads/<name>/reference.py defines reference(p, x) -> {output: value or
(value, scale)}, computed in float64 numpy from the exact input values. The reference is the oracle,
not an entrant; it never appears on the board as a backend. A control may borrow the reference of
the workload it breaks with "reference_from".
"""
import importlib.util
import os

import numpy as np

from . import manifest


def load(m):
    src_name = m.get("reference_from", m["name"])
    d = manifest.workload_dir(src_name)
    path = os.path.join(d, "reference.py")
    spec = importlib.util.spec_from_file_location("oal_ref_" + src_name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.reference


def compute(m, params, inputs):
    fn = load(m)
    x = {k: np.asarray(v) for k, v in inputs.items()}
    out = fn(dict(params), x)
    refs = {}
    for spec in m["outputs"]:
        v = out[spec["name"]]
        if isinstance(v, tuple):
            refs[spec["name"]] = (np.asarray(v[0], dtype=np.float64), np.asarray(v[1], dtype=np.float64))
        else:
            refs[spec["name"]] = (np.asarray(v, dtype=np.float64), None)
        want = tuple(manifest.shape(spec, params))
        if refs[spec["name"]][0].shape != want:
            raise ValueError("%s reference %s has shape %s, manifest says %s" % (
                m["name"], spec["name"], refs[spec["name"]][0].shape, want))
    return refs
