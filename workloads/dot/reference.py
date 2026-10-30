# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    pr = x["a"].astype(np.float64) * x["b"].astype(np.float64)
    return {"y": (np.array([pr.sum()]), np.array([np.abs(pr).sum()]))}
