// SPDX-License-Identifier: Apache-2.0
#include <cuda_runtime.h>

// One 16x16 block per tile; A and B are staged in shared memory, C[i][j] = sum_k A[i][k] B[k][j].
__global__ void tmatmul16(const float *a, const float *b, float *c)
{
    __shared__ float la[256];
    __shared__ float lb[256];
    long long t = blockIdx.x;
    int i = threadIdx.y, j = threadIdx.x, l = i * 16 + j;
    la[l] = a[t * 256 + l];
    lb[l] = b[t * 256 + l];
    __syncthreads();
    float s = 0.0f;
    for (int k = 0; k < 16; k++)
        s += la[i * 16 + k] * lb[k * 16 + j];
    c[t * 256 + l] = s;
}

void oal_launch(const long long *p, void *const *buf)
{
    long long tiles = p[0];
    tmatmul16<<<(unsigned)tiles, dim3(16, 16)>>>((const float *)buf[0], (const float *)buf[1], (float *)buf[2]);
}
