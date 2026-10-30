# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    h = p["h"]
    v = x["x"].astype(np.float64)
    c = x["cos"].astype(np.float64)[:, None, :]
    s = x["sin"].astype(np.float64)[:, None, :]
    x1, x2 = v[..., :h], v[..., h:]
    y = np.concatenate([x1 * c - x2 * s, x1 * s + x2 * c], axis=-1)
    sc = np.concatenate([np.abs(x1 * c) + np.abs(x2 * s), np.abs(x1 * s) + np.abs(x2 * c)], axis=-1)
    return {"y": (y, sc)}
