# SPDX-License-Identifier: Apache-2.0
# Blocked loop: each program owns BM rows, accumulates elementwise products over BN-wide column blocks
# in f32 and reduces each row with tl.sum (a summation tree, covered by the any-order gamma(N) bound).
import triton
import triton.language as tl


@triton.jit
def _gemv(A, x, y, M, N, BM: tl.constexpr, BN: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.arange(0, BN)
    acc = tl.zeros((BM,), dtype=tl.float32)
    for j0 in range(0, N, BN):
        jj = j0 + rn
        a = tl.load(A + rm[:, None] * N + jj[None, :], mask=(rm[:, None] < M) & (jj[None, :] < N), other=0.0)
        v = tl.load(x + jj, mask=jj < N, other=0.0)
        acc += tl.sum(a * v[None, :], axis=1)
    tl.store(y + rm, acc, mask=rm < M)


def launch(p, t):
    M, N = p["M"], p["N"]
    _gemv[(triton.cdiv(M, 32),)](t["A"], t["x"], t["y"], M, N, BM=32, BN=64)
