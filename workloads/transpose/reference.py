# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    a = x["a"].astype(np.float64).reshape(p["R"], p["C"])
    return {"b": np.ascontiguousarray(a.T)}
