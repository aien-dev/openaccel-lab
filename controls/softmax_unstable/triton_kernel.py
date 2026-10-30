# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _softmax(x, y, D, BLOCK: tl.constexpr):
    r = tl.program_id(0)
    k = tl.arange(0, BLOCK)
    m = k < D
    # CONTROL: the max subtraction is deliberately omitted
    v = tl.load(x + r * D + k, mask=m, other=-float("inf"))
    e = tl.exp(v)
    tl.store(y + r * D + k, e / tl.sum(e, axis=0), mask=m)


def launch(p, t):
    _softmax[(p["R"],)](t["x"], t["y"], p["D"], BLOCK=triton.next_power_of_2(p["D"]))
