# SPDX-License-Identifier: Apache-2.0
import numpy as np

EPS = float(np.float32(1e-6))   # every entrant writes eps as the f32 constant 1e-6


def reference(p, x):
    v = x["x"].astype(np.float64)
    w = x["w"].astype(np.float64)[None, :]
    ms = (v * v).mean(axis=1, keepdims=True)
    return {"y": v / np.sqrt(ms + EPS) * w}
