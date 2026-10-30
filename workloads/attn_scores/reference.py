# SPDX-License-Identifier: Apache-2.0
import numpy as np

MASKED = float(np.float32(-1e30))


def reference(p, x):
    t, d = p["T"], p["Dh"]
    q = x["q"].astype(np.float64).reshape(t, d)
    k = x["k"].astype(np.float64).reshape(t, d)
    s = 0.125 * (q @ k.T)
    scale = 0.125 * (np.abs(q) @ np.abs(k).T)
    masked = np.triu(np.ones((t, t), dtype=bool), 1)
    s[masked] = MASKED
    scale[masked] = 0.0
    return {"s": (s, scale)}
