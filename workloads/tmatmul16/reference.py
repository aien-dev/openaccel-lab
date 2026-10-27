# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    a = x["a"].astype(np.float64)
    b = x["b"].astype(np.float64)
    c = np.einsum("tik,tkj->tij", a, b)
    scale = np.einsum("tik,tkj->tij", np.abs(a), np.abs(b))
    return {"c": (c, scale)}
