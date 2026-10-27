# SPDX-License-Identifier: Apache-2.0
"""Workload manifests: load workloads/<name>/workload.json and evaluate its shape and tolerance
expressions against the case parameters. Expressions are integer/float arithmetic over parameter
names plus gamma(n) and u (unit roundoff 2^-24); nothing else is evaluated.

INTERIM host glue (Python). Successor: an Omega manifest reader once seeded Omega has the JSON
reader of seed/json/ wired into a program-mode library.
"""
import ast
import json
import math
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DTYPES = {"f32": ("float32", 4), "f64": ("float64", 8), "i32": ("int32", 4), "i8": ("int8", 1),
          "u8": ("uint8", 1)}
U32 = 2.0 ** -24  # unit roundoff of binary32, round to nearest


def gamma(n):
    """gamma_n = n u / (1 - n u), the classic bound on n roundings (Higham, ASNA 2nd ed. 3.1)."""
    n = float(n)
    if n * U32 >= 1.0:
        raise ValueError("gamma(%g) undefined: n u >= 1" % n)
    return n * U32 / (1.0 - n * U32)


_FUNCS = {"gamma": gamma, "max": max, "min": min, "log2": math.log2, "ceil": math.ceil}
_OPS = (ast.Expression, ast.BinOp, ast.UnaryOp, ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv,
        ast.Mod, ast.Pow, ast.USub, ast.UAdd, ast.Constant, ast.Name, ast.Load, ast.Call)


def evaluate(expr, params):
    """Evaluate a shape or tolerance expression. Numbers pass through unchanged."""
    if isinstance(expr, (int, float)):
        return expr
    tree = ast.parse(str(expr), mode="eval")
    for node in ast.walk(tree):
        if not isinstance(node, _OPS):
            raise ValueError("manifest expression %r: %s not allowed" % (expr, type(node).__name__))
        if isinstance(node, ast.Call) and not (isinstance(node.func, ast.Name) and node.func.id in _FUNCS):
            raise ValueError("manifest expression %r: unknown function" % expr)
    env = dict(_FUNCS)
    env["u"] = U32
    env.update(params)
    return eval(compile(tree, "<manifest>", "eval"), {"__builtins__": {}}, env)


def shape(spec, params):
    dims = [int(evaluate(d, params)) for d in spec["shape"]]
    if any(d < 1 for d in dims):
        raise ValueError("%s: non-positive dimension %s" % (spec["name"], dims))
    return dims


def nbytes(spec, params):
    n = 1
    for d in shape(spec, params):
        n *= d
    return n * DTYPES[spec["dtype"]][1]


def workload_dir(name):
    for top in ("workloads", "controls"):
        d = os.path.join(ROOT, top, name)
        if os.path.isfile(os.path.join(d, "workload.json")):
            return d
    raise FileNotFoundError("no workload %r under workloads/ or controls/" % name)


def load(name):
    d = workload_dir(name)
    with open(os.path.join(d, "workload.json")) as f:
        m = json.load(f)
    if m.get("name") != name:
        raise ValueError("%s/workload.json names %r" % (d, m.get("name")))
    m["_dir"] = d
    m["_control"] = os.path.basename(os.path.dirname(d)) == "controls"
    for key in ("params", "inputs", "outputs", "tolerance", "timing", "backends"):
        if key not in m:
            raise ValueError("%s: missing %r" % (name, key))
    for t in m["inputs"] + m["outputs"]:
        if t["dtype"] not in DTYPES:
            raise ValueError("%s: tensor %s has dtype %s" % (name, t["name"], t["dtype"]))
    for o in m["outputs"]:
        if o["name"] not in m["tolerance"]:
            raise ValueError("%s: output %s has no tolerance contract" % (name, o["name"]))
    return m


def all_names(include_controls=False):
    tops = ["workloads"] + (["controls"] if include_controls else [])
    names = []
    for top in tops:
        base = os.path.join(ROOT, top)
        if not os.path.isdir(base):
            continue
        for n in sorted(os.listdir(base)):
            if os.path.isfile(os.path.join(base, n, "workload.json")):
                names.append(n)
    return names


def params_of(m, overrides=None):
    p = dict(m["params"])
    for k, v in (overrides or {}).items():
        if k not in p:
            raise ValueError("%s has no parameter %r" % (m["name"], k))
        p[k] = int(v)
    for k, expr in m.get("derived", {}).items():
        p[k] = int(evaluate(expr, p))
    return p
