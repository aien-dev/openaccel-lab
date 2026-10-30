# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    a = x["x"].astype(np.float64)
    return {"y": (np.array([a.sum()]), np.array([np.abs(a).sum()]))}
