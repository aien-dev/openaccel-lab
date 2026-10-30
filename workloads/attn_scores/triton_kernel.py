# SPDX-License-Identifier: Apache-2.0
import triton
import triton.language as tl


@triton.jit
def _attn_scores(q, k, s, T, D: tl.constexpr, BJ: tl.constexpr):
    i = tl.program_id(0)
    j = tl.arange(0, BJ)
    c = tl.arange(0, D)
    qv = tl.load(q + i * D + c)
    kv = tl.load(k + j[:, None] * D + c[None, :], mask=j[:, None] < T, other=0.0)
    acc = tl.sum(kv * qv[None, :], axis=1) * 0.125
    out = tl.where(j <= i, acc, -1e30)
    tl.store(s + i * T + j, out, mask=j < T)


def launch(p, t):
    n = p["T"]
    _attn_scores[(n,)](t["q"], t["k"], t["s"], n, D=p["Dh"], BJ=triton.next_power_of_2(n))
