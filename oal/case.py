# SPDX-License-Identifier: Apache-2.0
"""Deterministic inputs and the case container (docs/CONTAINER.md).

Every input value is made from splitmix64 integers with IEEE basic operations only (+ - * /),
never with a libm function, so two machines produce byte-identical inputs. The case sha256 is
recorded with every result so the board can show that two runners ran the same bytes.

INTERIM host glue (Python + numpy). Successor: oal-gen in seeded Omega (splitmix64 and the
container writer need only integer words and write).
"""
import hashlib
import struct

import numpy as np

from . import manifest

MAGIC = b"OALCASE1"
M64 = np.uint64(0xFFFFFFFFFFFFFFFF)


def fnv1a64(text):
    h = 0xCBF29CE484222325
    for b in text.encode():
        h = ((h ^ b) * 0x100000001B3) & 0xFFFFFFFFFFFFFFFF
    return h


def splitmix64(seed, count):
    """count outputs of splitmix64 (Steele, Lea, Flood 2014) starting from state seed."""
    with np.errstate(over="ignore"):
        idx = np.arange(1, count + 1, dtype=np.uint64)
        z = np.uint64(seed) + idx * np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))


def unit24(bits):
    """Top 24 bits as a float64 in [0, 1), exact."""
    return (bits >> np.uint64(40)).astype(np.float64) * (2.0 ** -24)


def generate(spec, count, stream_seed):
    g = spec["gen"]
    kind = g["kind"]
    if kind == "uniform":
        v = g["lo"] + (g["hi"] - g["lo"]) * unit24(splitmix64(stream_seed, count))
    elif kind == "normal4":
        # Irwin-Hall: sum of four U(0,1) minus 2, scaled to unit variance (var = 4/12).
        b = splitmix64(stream_seed, 4 * count).reshape(count, 4)
        s = unit24(b).sum(axis=1) - 2.0
        v = g.get("mean", 0.0) + g.get("std", 1.0) * s * np.sqrt(3.0)
    elif kind == "grid":
        # integers k in [klo, khi] times 2^-shift: exactly representable in narrow formats.
        span = np.uint64(g["khi"] - g["klo"] + 1)
        k = (splitmix64(stream_seed, count) % span).astype(np.int64) + g["klo"]
        v = k.astype(np.float64) * (2.0 ** -g["shift"])
    elif kind == "int":
        span = np.uint64(g["hi"] - g["lo"] + 1)
        v = (splitmix64(stream_seed, count) % span).astype(np.int64) + g["lo"]
    elif kind == "const":
        v = np.full(count, g["value"], dtype=np.float64)
    elif kind == "iota":
        v = (np.arange(count, dtype=np.int64) % g.get("mod", count)).astype(np.float64)
    else:
        raise ValueError("unknown generator %r" % kind)
    return v


def inputs(m, params, seed):
    """Dict name -> numpy array (manifest dtype and shape), in manifest order."""
    out = {}
    for spec in m["inputs"]:
        dims = manifest.shape(spec, params)
        count = int(np.prod(dims))
        stream = (seed ^ fnv1a64(m["name"] + "/" + spec["name"])) & 0xFFFFFFFFFFFFFFFF
        v = generate(spec, count, stream)
        np_dtype = manifest.DTYPES[spec["dtype"]][0]
        if spec["dtype"] in ("i32", "i8", "u8"):
            arr = np.asarray(v).astype(np.int64)
            info = np.iinfo(np_dtype)
            if arr.min() < info.min or arr.max() > info.max:
                raise ValueError("%s: generator leaves %s range" % (spec["name"], spec["dtype"]))
            arr = arr.astype(np_dtype)
        else:
            arr = np.asarray(v, dtype=np.float64).astype(np_dtype)
        out[spec["name"]] = arr.reshape(dims)
    return out


def pad8(n):
    return (n + 7) // 8 * 8


def pack(m, params, tensors, warmup, repeats):
    """The container bytes: header, int64 params, tensor byte sizes, then padded input data."""
    names = list(params)
    head = MAGIC + struct.pack("<5Q", len(names), len(m["inputs"]), len(m["outputs"]), warmup, repeats)
    head += struct.pack("<%dq" % len(names), *[int(params[k]) for k in names])
    sizes = [manifest.nbytes(s, params) for s in m["inputs"] + m["outputs"]]
    head += struct.pack("<%dQ" % len(sizes), *sizes)
    body = b""
    for spec in m["inputs"]:
        raw = np.ascontiguousarray(tensors[spec["name"]]).tobytes()
        body += raw + b"\0" * (pad8(len(raw)) - len(raw))
    return head + body


def unpack(blob):
    """Inverse of pack for Python entrants: (params list, sizes list, inputs bytes list, warmup, repeats)."""
    if blob[:8] != MAGIC:
        raise ValueError("not an OALCASE1 container")
    np_, nin, nout, warm, reps = struct.unpack_from("<5Q", blob, 8)
    o = 48
    params = list(struct.unpack_from("<%dq" % np_, blob, o))
    o += 8 * np_
    sizes = list(struct.unpack_from("<%dQ" % (nin + nout), blob, o))
    o += 8 * (nin + nout)
    ins = []
    for k in range(nin):
        ins.append(blob[o:o + sizes[k]])
        o += pad8(sizes[k])
    return params, sizes, ins, warm, reps


def split_outputs(m, params, raw):
    """Cut an entrant's stdout into named output arrays; raises on a short or long stream."""
    out = {}
    o = 0
    for spec in m["outputs"]:
        n = manifest.nbytes(spec, params)
        if o + n > len(raw):
            raise ValueError("output stream ends inside %s (%d of %d bytes)" % (spec["name"], len(raw) - o, n))
        out[spec["name"]] = np.frombuffer(raw[o:o + n], dtype=manifest.DTYPES[spec["dtype"]][0]).reshape(
            manifest.shape(spec, params))
        o += pad8(n)
    if o != len(raw):
        raise ValueError("output stream has %d bytes, expected %d" % (len(raw), o))
    return out


def sha256(blob):
    return hashlib.sha256(blob).hexdigest()
