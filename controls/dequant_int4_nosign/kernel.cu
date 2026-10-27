// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

__global__ void dequant_int4(const unsigned char *w, const float *s, float *y, int R, int C, int G, int NB)
{
    long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long long)R * C)
        return;
    int r = (int)(i / C);
    int c = (int)(i % C);
    int b = w[r * NB + c / 2];
    int nib = (c & 1) ? (b >> 4) : (b & 15);
    nib = nib >= 8 ? nib - 16 : nib;
    y[i] = (float)nib * s[r * G + c / 32];
}

void oal_launch(const long long *p, void *const *buf)
{
    long long n = p[0] * p[1];
    dequant_int4<<<(unsigned)((n + 255) / 256), 256>>>((const unsigned char *)buf[0], (const float *)buf[1],
                                                       (float *)buf[2], (int)p[0], (int)p[1], (int)p[2], (int)p[3]);
}
