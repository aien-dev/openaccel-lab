// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// attn_p: one thread per row i writes the softmax weights pw[i][j], j <= i.
__global__ void attn_p(const float *q, const float *k, float *pw, int T, int D)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T)
        return;
    float m = -1e30f;
    for (int j = 0; j <= i; j++) {
        float acc = 0.0f;
        for (int c = 0; c < D; c++)
            acc += q[i * D + c] * k[j * D + c];
        float s = acc * 0.125f;
        pw[i * T + j] = s;
        m = fmaxf(m, s);
    }
    float z = 0.0f;
    for (int j = 0; j <= i; j++) {
        float e = expf(pw[i * T + j] - m);
        pw[i * T + j] = e;
        z += e;
    }
    for (int j = 0; j <= i; j++)
        pw[i * T + j] = pw[i * T + j] / z;
}

// attn_o: o[i][c] = sum_{j <= i} pw[i][j] v[j][c].
__global__ void attn_o(const float *pw, const float *v, float *o, int T, int D)
{
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int i = blockIdx.y;
    if (i >= T || c >= D)
        return;
    float acc = 0.0f;
    for (int j = 0; j <= i; j++)
        acc += pw[i * T + j] * v[j * D + c];
    o[i * D + c] = acc;
}

void oal_launch(const long long *p, void *const *buf)
{
    int T = (int)p[0];
    int D = (int)p[1];
    float *pw = (float *)buf[4];
    attn_p<<<(T + 63) / 64, 64>>>((const float *)buf[0], (const float *)buf[1], pw, T, D);
    dim3 grid((D + 63) / 64, T);
    attn_o<<<grid, 64>>>(pw, (const float *)buf[2], (float *)buf[3], T, D);
}
