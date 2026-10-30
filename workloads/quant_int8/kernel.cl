// SPDX-License-Identifier: Apache-2.0
// One work-item per row: absmax, scale = absmax / 127, q = clamp(rint(x / scale), -127, 127).
__kernel void quant_int8(__global const float *x, __global char *q, __global float *s,
                         const int R, const int C)
{
    int r = get_global_id(0);
    if (r >= R)
        return;
    float amax = 0.0f;
    for (int c = 0; c < C; c++)
        amax = fmax(amax, fabs(x[r * C + c]));
    float sc = amax / 127.0f;
    s[r] = sc;
    for (int c = 0; c < C; c++) {
        float y = sc > 0.0f ? x[r * C + c] / sc : 0.0f;
        float v = fmin(fmax(rint(y), -127.0f), 127.0f);
        q[r * C + c] = (char)(int)v;
    }
}
