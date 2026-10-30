# SPDX-License-Identifier: Apache-2.0
import numpy as np

U = 2.0 ** -24


def _g(n):
    return n * U / (1.0 - n * U)


def reference(p, x):
    D = p["D"]
    v = x["x"].astype(np.float64)
    a = v - v.max(axis=1, keepdims=True)
    e = np.exp(a)
    y = e / e.sum(axis=1, keepdims=True)
    # error bound; derivation in workload.json tolerance.why
    eps = (1.0 + 6.0 * U) * np.exp(U * np.abs(a)) - 1.0
    E = eps.max(axis=1, keepdims=True)
    gs, g2 = _g(D - 1), _g(2)
    hi = (1.0 + eps) * (1.0 + g2) / ((1.0 - E) * (1.0 - gs)) - 1.0
    lo = 1.0 - (1.0 - eps) * (1.0 - g2) / ((1.0 + E) * (1.0 + gs))
    return {"y": (y, y * np.maximum(hi, lo))}
