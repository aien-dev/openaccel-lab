# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    M, N, K = p["M"], p["N"], p["K"]
    a = x["A"].astype(np.float64).reshape(M, K)
    b = x["B"].astype(np.float64).reshape(N, K)
    return {"C": (a @ b.T, np.abs(a) @ np.abs(b).T)}
