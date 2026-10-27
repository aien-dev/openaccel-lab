# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    v = x["x"].astype(np.float64)
    return {"y": v / (1.0 + np.exp(-v))}
