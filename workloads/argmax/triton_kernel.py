# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _amax(x, y, n, BLOCK: tl.constexpr):
    # one program: lane j keeps the first maximum of x[j], x[j+BLOCK], ... (strict > keeps the
    # earlier index); then the smallest index among the lanes that hold the overall maximum.
    lane = tl.arange(0, BLOCK)
    bv = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    bi = tl.full([BLOCK], 2147483647, dtype=tl.int32)
    for off in range(0, n, BLOCK):
        i = off + lane
        v = tl.load(x + i, mask=i < n, other=float("-inf"))
        take = (v > bv) & (i < n)
        bv = tl.where(take, v, bv)
        bi = tl.where(take, i, bi)
    m = tl.max(bv, axis=0)
    tl.store(y, tl.min(tl.where(bv == m, bi, 2147483647), axis=0))


def launch(p, t):
    _amax[(1,)](t["x"], t["y"], p["n"], BLOCK=4096)
