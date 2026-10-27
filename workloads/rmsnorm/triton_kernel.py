# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _rmsnorm(x, w, y, D, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < D
    v = tl.load(x + r * D + k, mask=m, other=0.0)
    rs = 1.0 / tl.sqrt(tl.sum(v * v, axis=0) / D + 1e-6)
    tl.store(y + r * D + k, v * rs * tl.load(w + k, mask=m, other=0.0), mask=m)


def launch(p, t):
    _rmsnorm[(p["R"],)](t["x"], t["w"], t["y"], p["D"], BLOCK=triton.next_power_of_2(p["D"]))
