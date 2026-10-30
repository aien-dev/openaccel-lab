# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _quant_int8(x, q, s, C, BC: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BC)
    live = c < C
    xv = tl.load(x + r * C + c, mask=live, other=0.0)
    sc = tl.max(tl.abs(xv), axis=0) / 127.0
    tl.store(s + r, sc)
    y = tl.where(sc > 0.0, xv / tl.where(sc > 0.0, sc, 1.0), 0.0)
    # round half to even on |y|: |y| - floor(|y|) is exact for |y| < 2^23
    a = tl.abs(y)
    f = tl.floor(a)
    frac = a - f
    fi = f.to(tl.int32)
    up = (frac > 0.5) | ((frac == 0.5) & ((fi & 1) == 1))
    qi = tl.where(up, fi + 1, fi)
    qi = tl.where(y < 0.0, -qi, qi)
    qi = tl.minimum(tl.maximum(qi, -127), 127)
    tl.store(q + r * C + c, qi.to(tl.int8), mask=live)


def launch(p, t):
    _quant_int8[(p["R"],)](t["x"], t["q"], t["s"], p["C"], BC=triton.next_power_of_2(p["C"]))
