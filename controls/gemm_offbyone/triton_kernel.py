# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _tmatmul16(a, b, c, KN: tl.constexpr):
    t = tl.program_id(0)
    r = tl.arange(0, 16)
    km = r < KN
    av = tl.load(a + t * 256 + r[:, None] * 16 + r[None, :], mask=km[None, :], other=0.0)
    bv = tl.load(b + t * 256 + r[:, None] * 16 + r[None, :], mask=km[:, None], other=0.0)
    acc = tl.dot(av, bv, input_precision="ieee", out_dtype=tl.float32)
    tl.store(c + t * 256 + r[:, None] * 16 + r[None, :], acc)


def launch(p, t):
    _tmatmul16[(p["TILES"],)](t["a"], t["b"], t["c"], KN=15)
