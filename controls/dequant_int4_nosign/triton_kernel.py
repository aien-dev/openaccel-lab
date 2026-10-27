# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _dequant_int4(w, s, y, C, G, NB, BC: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BC)
    live = c < C
    b = tl.load(w + r * NB + c // 2, mask=live, other=0).to(tl.int32)
    nib = tl.where((c % 2) == 1, b >> 4, b & 15)
    nib = tl.where(nib >= 8, nib - 16, nib)
    sc = tl.load(s + r * G + c // 32, mask=live, other=0.0)
    tl.store(y + r * C + c, nib.to(tl.float32) * sc, mask=live)


def launch(p, t):
    _dequant_int4[(p["R"],)](t["w"], t["s"], t["y"], p["C"], p["G"], p["NB"], BC=triton.next_power_of_2(p["C"]))
