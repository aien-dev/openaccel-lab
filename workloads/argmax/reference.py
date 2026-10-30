# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    a = x["x"].astype(np.float64)
    return {"y": np.array([np.flatnonzero(a == a.max())[0]], dtype=np.float64)}
