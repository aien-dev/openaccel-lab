# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _attention(q, k, v, o, D: tl.constexpr, BJ: tl.constexpr):
    i = tl.program_id(0)
    j = tl.arange(0, BJ)
    c = tl.arange(0, D)
    live = j <= i
    qv = tl.load(q + i * D + c)
    kv = tl.load(k + j[:, None] * D + c[None, :], mask=live[:, None], other=0.0)
    s = tl.where(live, tl.sum(kv * qv[None, :], axis=1) * 0.125, -1e30)
    m = tl.max(s, axis=0)
    e = tl.where(live, tl.exp(s - m), 0.0)
    pw = e / tl.sum(e, axis=0)
    vv = tl.load(v + j[:, None] * D + c[None, :], mask=live[:, None], other=0.0)
    tl.store(o + i * D + c, tl.sum(pw[:, None] * vv, axis=0))


def launch(p, t):
    n = p["T"]
    _attention[(n,)](t["q"], t["k"], t["v"], t["o"], D=p["Dh"], BJ=triton.next_power_of_2(n))
