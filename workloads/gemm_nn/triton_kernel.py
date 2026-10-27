# SPDX-License-Identifier: Apache-2.0
# tl.dot with input_precision="ieee": full f32 products and f32 accumulation (the interpreter runs
# np.matmul in float32; on a GPU this forbids TF32), so the gamma(K) dot-product bound applies.
import triton
import triton.language as tl


@triton.jit
def _gemm_nn(A, B, C, M, N, K, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rn = tl.program_id(1) * BN + tl.arange(0, BN)
    rk = tl.arange(0, BK)
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for k0 in range(0, K, BK):
        kk = k0 + rk
        a = tl.load(A + rm[:, None] * K + kk[None, :], mask=(rm[:, None] < M) & (kk[None, :] < K), other=0.0)
        b = tl.load(B + kk[:, None] * N + rn[None, :], mask=(kk[:, None] < K) & (rn[None, :] < N), other=0.0)
        acc = tl.dot(a, b, acc, input_precision="ieee")
    tl.store(C + rm[:, None] * N + rn[None, :], acc, mask=(rm[:, None] < M) & (rn[None, :] < N))


def launch(p, t):
    M, N, K = p["M"], p["N"], p["K"]
    _gemm_nn[(triton.cdiv(M, 16), triton.cdiv(N, 16))](t["A"], t["B"], t["C"], M, N, K, BM=16, BN=16, BK=16)
