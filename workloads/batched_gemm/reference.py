# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    Bt, M, N, K = p["BATCH"], p["M"], p["N"], p["K"]
    a = x["A"].astype(np.float64).reshape(Bt, M, K)
    b = x["B"].astype(np.float64).reshape(Bt, K, N)
    return {"C": (a @ b, np.abs(a) @ np.abs(b))}
