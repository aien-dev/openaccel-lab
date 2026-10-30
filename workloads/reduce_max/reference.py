# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    return {"y": np.array([x["x"].astype(np.float64).max()])}
