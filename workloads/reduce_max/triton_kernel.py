# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _rmax(x, y, n, BLOCK: tl.constexpr):
    # one program: BLOCK lane maxima over the strided input, then a tree max of the lanes
    lane = tl.arange(0, BLOCK)
    acc = tl.full([BLOCK], float("-inf"), dtype=tl.float32)
    for off in range(0, n, BLOCK):
        i = off + lane
        acc = tl.maximum(acc, tl.load(x + i, mask=i < n, other=float("-inf")))
    tl.store(y, tl.max(acc, axis=0))


def launch(p, t):
    _rmax[(1,)](t["x"], t["y"], p["n"], BLOCK=4096)
