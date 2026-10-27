# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    r, c = p["R"], p["C"]
    v = x["x"].astype(np.float64).reshape(r, c)
    amax = np.abs(v).max(axis=1)
    s = amax / 127.0
    y = np.zeros_like(v)
    nz = amax > 0
    y[nz] = 127.0 * v[nz] / amax[nz, None]
    q = np.clip(np.rint(y), -127.0, 127.0)  # rint: round half to even
    return {"q": q, "s": s}
