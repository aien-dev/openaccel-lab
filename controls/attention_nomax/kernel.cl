// SPDX-License-Identifier: Apache-2.0
// CONTROL attention_nomax: exp(s_j) without subtracting the row max m (m is still computed).
// attn_p: one work-item per row i writes the softmax weights pw[i][j], j <= i.
__kernel void attn_p(__global const float *q, __global const float *k, __global float *pw,
                     const int T, const int D)
{
    int i = get_global_id(0);
    if (i >= T)
        return;
    float m = -1e30f;
    for (int j = 0; j <= i; j++) {
        float acc = 0.0f;
        for (int c = 0; c < D; c++)
            acc += q[i * D + c] * k[j * D + c];
        float s = acc * 0.125f;
        pw[i * T + j] = s;
        m = fmax(m, s);
    }
    float z = 0.0f;
    for (int j = 0; j <= i; j++) {
        float e = exp(pw[i * T + j]);
        pw[i * T + j] = e;
        z += e;
    }
    for (int j = 0; j <= i; j++)
        pw[i * T + j] = pw[i * T + j] / z;
}

// attn_o: o[i][c] = sum_{j <= i} pw[i][j] v[j][c].
__kernel void attn_o(__global const float *pw, __global const float *v, __global float *o,
                     const int T, const int D)
{
    int i = get_global_id(0);
    int c = get_global_id(1);
    if (i >= T || c >= D)
        return;
    float acc = 0.0f;
    for (int j = 0; j <= i; j++)
        acc += pw[i * T + j] * v[j * D + c];
    o[i * D + c] = acc;
}
