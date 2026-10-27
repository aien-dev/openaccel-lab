# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _layernorm(x, g, b, y, D, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < D
    v = tl.load(x + r * D + k, mask=m, other=0.0)
    mu = tl.sum(v, axis=0) / D
    c = tl.where(m, v - mu, 0.0)
    var = tl.sum(c * c, axis=0) / D
    rs = 1.0 / tl.sqrt(var + 1e-5)
    out = c * rs * tl.load(g + k, mask=m, other=0.0) + tl.load(b + k, mask=m, other=0.0)
    tl.store(y + r * D + k, out, mask=m)


def launch(p, t):
    _layernorm[(p["R"],)](t["x"], t["gamma"], t["beta"], t["y"], p["D"], BLOCK=triton.next_power_of_2(p["D"]))
