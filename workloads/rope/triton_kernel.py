# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _rope(x, cs, sn, y, H, h, BLOCK: tl.constexpr):
    th = tl.program_id(0)           # t*H + j
    t = th // H
    i = tl.arange(0, BLOCK)
    m = i < h
    o = th * 2 * h + i
    c = tl.load(cs + t * h + i, mask=m, other=0.0)
    s = tl.load(sn + t * h + i, mask=m, other=0.0)
    x1 = tl.load(x + o, mask=m, other=0.0)
    x2 = tl.load(x + o + h, mask=m, other=0.0)
    tl.store(y + o, x1 * c - x2 * s, mask=m)
    tl.store(y + o + h, x1 * s + x2 * c, mask=m)


def launch(p, t):
    _rope[(p["T"] * p["H"],)](t["x"], t["cos"], t["sin"], t["y"], p["H"], p["h"],
                              BLOCK=triton.next_power_of_2(p["h"]))
