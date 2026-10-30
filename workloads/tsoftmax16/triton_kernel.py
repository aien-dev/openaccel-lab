# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _tsoftmax16(x, y, NR: tl.constexpr):
    r = tl.program_id(0) * NR + tl.arange(0, NR)
    c = tl.arange(0, 32)
    v = tl.load(x + r[:, None] * 32 + c[None, :])
    m = tl.max(v, axis=1)
    e = tl.exp(v - m[:, None])
    s = tl.sum(e, axis=1)
    tl.store(y + r[:, None] * 32 + c[None, :], e / s[:, None])


def launch(p, t):
    _tsoftmax16[(p["TILES"],)](t["x"], t["y"], NR=16)
