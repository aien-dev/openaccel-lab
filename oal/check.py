# SPDX-License-Identifier: Apache-2.0
"""The tolerance contract (docs/TOLERANCE.md).

The reference (workloads/<name>/reference.py) computes in float64 from the exact input values and
returns, per output, the value and optionally a scale array. The contract kinds are:

  exact   every element equals the reference rounded once to the output dtype (bitwise for floats,
          with -0.0 == 0.0); used where one correctly rounded operation, or none, makes each element.
  bound   |got - ref| <= rtol * scale + atol elementwise; scale defaults to |ref|. rtol and atol are
          manifest expressions (gamma(n), u, parameters), so the bound follows the shape.
  int     |got - ref| <= atol on integers (atol 0 is exact).

NaN or infinity in an output fails unless the reference has the same value at that element.
The verdict carries err_ratio = max(|got - ref| / bound) so the board can show how much of the
bound each entrant used (1.0 = at the limit).

INTERIM host glue (Python + numpy). Successor: oal-check in seeded Omega (f32/f64 words exist).
"""
import numpy as np

from . import manifest


def _bound_arrays(contract, params, ref, scale):
    rtol = float(manifest.evaluate(contract.get("rtol", 0.0), params))
    atol = float(manifest.evaluate(contract.get("atol", 0.0), params))
    sc = np.abs(ref) if scale is None else np.asarray(scale, dtype=np.float64)
    return rtol * sc + atol, rtol, atol


def compare(m, params, refs, got):
    """refs: name -> (value float64 array, scale or None). got: name -> array. Returns verdict dict."""
    worst = 0.0
    details = []
    ok = True
    for spec in m["outputs"]:
        name = spec["name"]
        contract = m["tolerance"][name]
        ref, scale = refs[name]
        ref = np.asarray(ref, dtype=np.float64)
        g = np.asarray(got[name])
        if g.shape != ref.shape:
            details.append({"output": name, "pass": False, "why": "shape %s vs %s" % (g.shape, ref.shape)})
            ok = False
            continue
        kind = contract["kind"]
        gd = g.astype(np.float64)
        bad_nonfinite = ~np.isfinite(gd) & ~(np.isnan(gd) & np.isnan(ref)) & ~(gd == ref)
        if kind == "exact":
            want = ref.astype(manifest.DTYPES[spec["dtype"]][0]).astype(np.float64)
            err = np.abs(gd - want)
            err[np.isnan(gd) & np.isnan(want)] = 0.0
            bound = np.zeros_like(err)
            fail = (err != 0) | bad_nonfinite
            ratio = float(np.nanmax(np.where(err > 0, np.inf, 0.0))) if err.size else 0.0
        elif kind in ("bound", "int"):
            bound, rtol, atol = _bound_arrays(contract, params, ref, scale)
            err = np.abs(gd - ref)
            fail = ~(err <= bound) | bad_nonfinite
            with np.errstate(divide="ignore", invalid="ignore"):
                r = np.where(bound > 0, err / bound, np.where(err > 0, np.inf, 0.0))
            ratio = float(np.nanmax(r)) if r.size else 0.0
        else:
            raise ValueError("%s: unknown tolerance kind %r" % (name, kind))
        nfail = int(np.count_nonzero(fail))
        d = {"output": name, "pass": nfail == 0, "kind": kind, "elements": int(fail.size),
             "failed": nfail, "err_ratio": ratio if np.isfinite(ratio) else None,
             "max_abs_err": float(np.nanmax(err)) if err.size else 0.0}
        if nfail:
            flat = np.flatnonzero(fail.ravel())
            i = int(flat[0])
            d["first_bad"] = {"index": [int(x) for x in np.unravel_index(i, fail.shape)],
                              "got": float(gd.ravel()[i]), "ref": float(ref.ravel()[i]),
                              "bound": float(np.asarray(bound).ravel()[i]) if np.size(bound) else 0.0}
            ok = False
        if np.isfinite(ratio):
            worst = max(worst, ratio)
        else:
            worst = float("inf")
        details.append(d)
    return {"pass": ok, "err_ratio": worst if np.isfinite(worst) else None, "outputs": details}
