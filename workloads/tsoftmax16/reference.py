# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    v = x["x"].astype(np.float64)
    e = np.exp(v - v.max(axis=2, keepdims=True))
    return {"y": e / e.sum(axis=2, keepdims=True)}
