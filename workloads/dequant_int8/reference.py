# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    r, c = p["R"], p["C"]
    q = x["q"].astype(np.float64).reshape(r, c)
    s = x["s"].astype(np.float64).reshape(r)
    return {"y": q * s[:, None]}
