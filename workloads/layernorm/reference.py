# SPDX-License-Identifier: Apache-2.0
import numpy as np

U = 2.0 ** -24
EPS = float(np.float32(1e-5))   # every entrant writes eps as the f32 constant 1e-5


def _g(n):
    return n * U / (1.0 - n * U)


def reference(p, x):
    D = p["D"]
    v = x["x"].astype(np.float64)
    g = x["gamma"].astype(np.float64)[None, :]
    b = x["beta"].astype(np.float64)[None, :]
    mu = v.mean(axis=1, keepdims=True)
    c = v - mu
    var = (c * c).mean(axis=1, keepdims=True)
    vv = var + EPS
    s = 1.0 / np.sqrt(vv)
    y = c * s * g + b
    # error bound; derivation in workload.json tolerance.why
    em = _g(D + 1) * np.abs(v).sum(axis=1, keepdims=True) / D
    dc = em + U * (np.abs(c) + em)
    rho = (1.0 + em * em / vv) * (1.0 + _g(D + 5)) - 1.0
    phi = (1.0 + rho) * (1.0 + 4.0 * U) - 1.0
    ag = np.abs(g)
    bound = ag * s * (dc * (1.0 + phi) + np.abs(c) * phi)
    bound = bound + _g(3) * (ag * s * (np.abs(c) + dc) * (1.0 + phi) + np.abs(b))
    return {"y": (y, bound)}
