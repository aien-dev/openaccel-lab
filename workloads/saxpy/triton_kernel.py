# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _saxpy(alpha, x, y_in, y, n, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = i < n
    a = tl.load(alpha)
    tl.store(y + i, a * tl.load(x + i, mask=m) + tl.load(y_in + i, mask=m), mask=m)


def launch(p, t):
    n = p["n"]
    _saxpy[(triton.cdiv(n, 1024),)](t["alpha"], t["x"], t["y_in"], t["y"], n, BLOCK=1024)
