# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    r, c, g, nb = p["R"], p["C"], p["G"], p["NB"]
    w = x["w"].astype(np.int64).reshape(r, nb)
    s = x["s"].astype(np.float64).reshape(r, g)
    nib = np.empty((r, c), dtype=np.int64)
    nib[:, 0::2] = w & 15
    nib[:, 1::2] = w >> 4
    nib = np.where(nib >= 8, nib - 16, nib)
    return {"y": nib.astype(np.float64) * s[:, np.arange(c) // 32]}
