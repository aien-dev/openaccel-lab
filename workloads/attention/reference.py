# SPDX-License-Identifier: Apache-2.0
"""Single-head causal attention O = softmax(mask(Q K^T * 0.125)) V, with the elementwise error bound.

The bound (see workload.json "why") is computed per element, so the scale returned for "o" is the
whole bound and the manifest uses rtol 1. Notation for row i, n = i + 1 unmasked columns j:
  s_j = 0.125 sum_c q_c k_jc (exact), a_j = 0.125 sum_c |q_c k_jc|, e_j = gamma(Dh) a_j, E = max_j e_j,
  d_j = s_j - max s, p_j = exp(d_j) / sum exp(d), tau_j = e_j + E + u (|d_j| + e_j + E),
  theta_j = exp(tau_j) (1 + EXP) - 1                     relative error of each exp(score - max)
  K = sum_j p_j ((1 + theta_j)(1 + gamma(n - 1)) - 1)    relative error of the softmax denominator
  c_j = (1 + theta_j)(1 + gamma(n + 1))(1 + u) / (1 - K) - 1
  bound_ic = sum_j p_j |v_jc| c_j
"""
import numpy as np

U = 2.0 ** -24
EXP = 6.0 * U  # OpenCL exp <= 3 ulp (OpenCL C 3.0 s7.4); 1 ulp <= 2u relative, so <= 6u


def _gamma(n):
    return n * U / (1.0 - n * U)


def reference(p, x):
    t, d = p["T"], p["Dh"]
    q = x["q"].astype(np.float64).reshape(t, d)
    k = x["k"].astype(np.float64).reshape(t, d)
    v = x["v"].astype(np.float64).reshape(t, d)
    s_all = 0.125 * (q @ k.T)
    a_all = 0.125 * (np.abs(q) @ np.abs(k).T)
    g = _gamma(d)
    o = np.zeros((t, d))
    bound = np.zeros((t, d))
    for i in range(t):
        n = i + 1
        s = s_all[i, :n]
        e = g * a_all[i, :n]
        big_e = e.max()
        dj = s - s.max()
        w = np.exp(dj)
        pj = w / w.sum()
        tau = e + big_e + U * (np.abs(dj) + e + big_e)
        theta = np.expm1(tau) * (1.0 + EXP) + EXP
        kk = float(np.sum(pj * ((1.0 + theta) * (1.0 + _gamma(n - 1)) - 1.0)))
        c = (1.0 + theta) * (1.0 + _gamma(n + 1)) * (1.0 + U) / (1.0 - kk) - 1.0
        o[i] = pj @ v[:n]
        bound[i] = (pj * c) @ np.abs(v[:n])
    return {"o": (o, bound)}
