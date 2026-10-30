// SPDX-License-Identifier: Apache-2.0
// One work-item per 32-column row: max, exp(x - max) with its sum, then divide.
__kernel void tsoftmax16(__global const float *x, __global float *y, const int tiles)
{
    int r = get_global_id(0);
    if (r >= tiles * 16)
        return;
    __global const float *xr = x + r * 32;
    __global float *yr = y + r * 32;
    float m = xr[0];
    for (int j = 1; j < 32; j++)
        m = fmax(m, xr[j]);
    float s = 0.0f;
    for (int j = 0; j < 32; j++) {
        float e = exp(xr[j] - m);
        yr[j] = e;
        s += e;
    }
    for (int j = 0; j < 32; j++)
        yr[j] = yr[j] / s;
}
