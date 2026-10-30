# SPDX-License-Identifier: Apache-2.0
# CONTROL (deliberately wrong): sums x[0..n-2] only, dropping the last element.
import triton
import triton.language as tl


@triton.jit
def _rsum(x, y, n, BLOCK: tl.constexpr):
    # one program: BLOCK lane accumulators over the strided input, then a tree sum of the lanes
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, n, BLOCK):
        i = off + lane
        acc += tl.load(x + i, mask=i < n, other=0.0)
    tl.store(y, tl.sum(acc, axis=0))


def launch(p, t):
    _rsum[(1,)](t["x"], t["y"], p["n"] - 1, BLOCK=4096)
