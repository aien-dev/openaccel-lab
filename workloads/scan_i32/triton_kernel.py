# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _scan(x, y, n, BLOCK: tl.constexpr):
    # one program walks the input in blocks: block-local inclusive cumsum plus the running carry
    lane = tl.arange(0, BLOCK)
    carry = tl.sum(tl.zeros([BLOCK], dtype=tl.int32), axis=0)
    for off in range(0, n, BLOCK):
        i = off + lane
        m = i < n
        v = tl.load(x + i, mask=m, other=0)
        tl.store(y + i, tl.cumsum(v, axis=0) + carry, mask=m)
        carry += tl.sum(v, axis=0)


def launch(p, t):
    _scan[(1,)](t["x"], t["y"], p["n"], BLOCK=4096)
