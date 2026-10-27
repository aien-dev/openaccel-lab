# Tolerance contract

Every workload states, per output, how close an entrant has to come to the float64 reference.
`oal/check.py` applies the contract; the entrant never sees it.

| kind | rule | used for |
|---|---|---|
| `exact` | bitwise equal to the reference rounded to the output dtype | integer outputs, copies, elementwise ops whose float32 result is correctly rounded |
| `int` | `abs(got - ref) <= atol` on integers | quantisation, where round-half cases may differ by one step |
| `bound` | `abs(got - ref) <= c * gamma(n) * scale + atol` | reductions, dot products, matmuls, normalisations |

`gamma(n) = n*u / (1 - n*u)` with `u = 2^-24` (float32 unit roundoff) is the standard forward error
bound for a length-`n` floating-point sum or dot product computed in any order (Higham, *Accuracy
and Stability of Numerical Algorithms*, 2nd ed., Section 3.1). `scale` is the reference's own
magnitude term, for example `sum |a_i * b_i|` for a dot product, so the bound does not depend on
cancellation in the exact answer. `c` and `atol` are written in each `workload.json` with a one-line
reason. A record's `err_ratio` is the largest `abs(got - ref) / bound` over all elements:
0 means bit-exact, 1 means at the bound, above 1 fails.

The bound allows any summation order on purpose. A tree reduction on a GPU, a serial loop on a CPU
and a Kahan sum all pass; a dropped term, an off-by-one index or an exclusive scan in place of an
inclusive one does not, and the controls in `controls/` prove that on every backend.

## Transcendentals in the Omega entrant

Seeded Omega has no `exp` word at omega-mv 369dbc96, so `lib/omega/oal.om` implements `oal_expf`
in float64: `x = n*ln2 + r` with `n = round(x / ln2)`, a two-part `ln2` (Cody-Waite) and a degree-11
Taylor polynomial for `exp(r)`, then scaling by `2^n` through the exponent bits. The result is
rounded to float32 by the caller. Every constant was checked with Wolfram:

- `WolframLanguageEvaluator: N[Log[2] - (SetPrecision[0.6931471803691238,40] + SetPrecision[1.9082149292705877*^-10,40]), 20]`
  -> `1.16122272293625318506558032769199*^-26` (the split of ln2 is exact far below float64 precision).
- `WolframLanguageEvaluator: N[1/Log[2], 20]` -> `1.4426950408889634074` (the `oal_expf` constant
  `1.4426950408889634`).
- `WolframLanguageEvaluator: Table[{k, N[1/k!, 17]}, {k, 2, 11}]` -> `0.5, 0.16666666666666667,
  0.041666666666666667, 0.0083333333333333333, 0.0013888888888888889, 0.00019841269841269841,
  0.000024801587301587302, 2.7557319223985891*^-6, 2.7557319223985891*^-7, 2.5052108385441719*^-8`
  (the polynomial coefficients).
- `WolframLanguageEvaluator: N[MaxValue[{Abs[Exp[r] - (1 + Sum[r^k/k!, {k, 1, 11}])]/Exp[r], -Log[2]/2 <= r <= Log[2]/2}, r], 6]`
  -> `8.63527402609310923645372*^-15` (largest relative truncation error of the polynomial on the
  reduced range).

That truncation error is 1.45e-7 of the float32 unit roundoff `2^-24`
(`WolframLanguageEvaluator: N[8.63527402609310923645372*^-15/2^-24, 6]` -> `1.44876*^-7`), so after the
final rounding to float32 the Omega `expf` differs from a correctly rounded `expf` only on inputs
that lie within that distance of a rounding boundary. Workloads that use `exp` (softmax, silu,
attention) carry it in their `bound` term through `c`, not through a special case per entrant.
