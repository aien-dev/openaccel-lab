# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    M, N = p["M"], p["N"]
    a = x["A"].astype(np.float64).reshape(M, N)
    v = x["x"].astype(np.float64)
    return {"y": (a @ v, np.abs(a) @ np.abs(v))}
