# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    return {"y": np.cumsum(x["x"].astype(np.int64)).astype(np.float64)}
