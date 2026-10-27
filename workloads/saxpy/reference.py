# SPDX-License-Identifier: Apache-2.0
import numpy as np


def reference(p, x):
    alpha = x["alpha"].astype(np.float64)[0]
    ax = alpha * x["x"].astype(np.float64)
    y = x["y_in"].astype(np.float64)
    return {"y": (ax + y, np.abs(ax) + np.abs(y))}
