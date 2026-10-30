// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__global__ void attn_scores(const float *q, const float *k, float *s, int T, int D)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= T || j >= T)
        return;
    if (j > i) {
        s[i * T + j] = -1e30f;
        return;
    }
    float acc = 0.0f;
    for (int c = 0; c < D; c++)
        acc += q[i * D + c] * k[j * D + c];
    s[i * T + j] = acc * 0.125f;
}

void oal_launch(const long long *p, void *const *buf)
{
    int T = (int)p[0];
    int D = (int)p[1];
    dim3 block(16, 16);
    dim3 grid((T + 15) / 16, (T + 15) / 16);
    attn_scores<<<grid, block>>>((const float *)buf[0], (const float *)buf[1], (float *)buf[2], T, D);
}
