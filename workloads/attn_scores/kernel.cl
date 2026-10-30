// SPDX-License-Identifier: Apache-2.0
__kernel void attn_scores(__global const float *q, __global const float *k, __global float *s,
                          const int T, const int D)
{
    int i = get_global_id(0);
    int j = get_global_id(1);
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
