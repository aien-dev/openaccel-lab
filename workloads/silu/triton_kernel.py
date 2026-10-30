# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _silu(x, y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    v = tl.load(x + i, mask=m, other=0.0)
    tl.store(y + i, v / (1.0 + tl.exp(-v)), mask=m)


def launch(p, t):
    n = p["n"]
    _silu[(triton.cdiv(n, 1024),)](t["x"], t["y"], n, BLOCK=1024)
