# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    M, N, K = p["M"], p["N"], p["K"]
    a = x["A"].astype(np.float64).reshape(M, K)
    b = x["B"].astype(np.float64).reshape(K, N)
    bias = x["bias"].astype(np.float64)
    pre = a @ b + bias[None, :]
    scale = np.abs(a) @ np.abs(b) + np.abs(bias)[None, :]
    return {"C": (np.maximum(pre, 0.0), scale)}
