# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _vecadd(a, b, y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    tl.store(y + i, tl.load(a + i, mask=m) + tl.load(b + i, mask=m), mask=m)


def launch(p, t):
    n = p["n"]
    _vecadd[(triton.cdiv(n, 1024),)](t["a"], t["b"], t["y"], n, BLOCK=1024)
