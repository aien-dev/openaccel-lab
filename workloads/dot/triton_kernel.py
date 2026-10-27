# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _dot(a, b, y, n, BLOCK: tl.constexpr):
    # one program: BLOCK lane accumulators of a[i]*b[i] over the strided input, then a tree sum
    lane = tl.arange(0, BLOCK)
    acc = tl.zeros([BLOCK], dtype=tl.float32)
    for off in range(0, n, BLOCK):
        i = off + lane
        m = i < n
        acc += tl.load(a + i, mask=m, other=0.0) * tl.load(b + i, mask=m, other=0.0)
    tl.store(y, tl.sum(acc, axis=0))


def launch(p, t):
    _dot[(1,)](t["a"], t["b"], t["y"], p["n"], BLOCK=4096)
