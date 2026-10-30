# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _transpose(a, b, R, C, BR: tl.constexpr, BC: tl.constexpr):
    r = tl.program_id(0) * BR + tl.arange(0, BR)
    c = tl.program_id(1) * BC + tl.arange(0, BC)
    m = (r[:, None] < R) & (c[None, :] < C)
    tile = tl.load(a + r[:, None] * C + c[None, :], mask=m)
    tl.store(b + c[None, :] * R + r[:, None], tile, mask=m)


def launch(p, t):
    R, C = p["R"], p["C"]
    _transpose[(triton.cdiv(R, 32), triton.cdiv(C, 32))](t["a"], t["b"], R, C, BR=32, BC=32)
