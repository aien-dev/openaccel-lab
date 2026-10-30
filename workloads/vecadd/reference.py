# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    return {"y": x["a"].astype(np.float64) + x["b"].astype(np.float64)}
