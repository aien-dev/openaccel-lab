# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _dequant_int8(q, s, y, C, BC: tl.constexpr):
    r = tl.program_id(0)
    c = tl.arange(0, BC)
    live = c < C
    qv = tl.load(q + r * C + c, mask=live, other=0).to(tl.float32)
    tl.store(y + r * C + c, qv * tl.load(s + r), mask=live)


def launch(p, t):
    _dequant_int8[(p["R"],)](t["q"], t["s"], t["y"], p["C"], BC=triton.next_power_of_2(p["C"]))
