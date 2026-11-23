"""Write a library composite's one-step function as JavaScript, with no equation typed.

``javascript(fn, example_input, name)`` traces ``fn`` with ``jax.make_jaxpr`` and translates
the trace, equation by equation, into one JavaScript function. Nothing about the model is
written here: whatever composite ``fn`` steps is read at run time from its trace.

The translation is scalar. Every array the trace holds is split into its entries; each entry
is a constant or a slot of one ``Float64Array``. Shapes are fixed by the trace, so routing
(slices, concatenations, broadcasts, gathers at fixed indices, pads, transposes) costs no
code: it is worked out here, by running the routing primitive itself on arrays of entry
numbers. Arithmetic becomes one JavaScript assignment per entry. Constants are folded by
running the primitive in JAX, common expressions are computed once, and entries no output
needs are dropped. A ``scan`` or ``while`` stays a JavaScript loop over slots and a ``cond``
an ``if``, so a 52-halving search is written once, not 52 times. The code is cut into
functions of bounded length so that V8 optimizes each one.

Semantics. Doubles are IEEE as in JAX with x64. Unsigned 32-bit integers are exact
(``Math.imul``, ``>>> 0``); signed 64-bit integers are carried as doubles, exact below 2^53.
Two simplifications differ from JAX only where JAX's value is not finite: ``0 * x`` is taken
as ``0`` and ``x + 0`` as ``x``. Sums are taken in index order, so a sum may differ from
XLA's in its last bits. A primitive not handled here stops the export with its name.

A JavaScript ternary evaluates only the branch it takes, so a guarded expression such as
``where(x > c, exp(-1/x), 0)`` costs nothing on the untaken side, as it costs XLA nothing to
select. The first seconds of a page run slowly while V8 optimizes the large functions.

From the library checkout run, for the feeding page's model with N casings (N = 1 .. 6; the
page loads one file per count), stepping on (state, places, knobs), with its readouts
(``observe``), its start (``start``) and its search curve (``curve``),
    .venv/bin/python scripts/export_step_js.py --casings N --output viewer/feeding-slot-N.js
and rerun it for every N after any change to the feeding parts (``tests/test_feeding_page.py``
fails until then). The tests are ``tests/test_export_step_js.py``; ``viewer/exported-step-driver.js``
runs an export under node.
"""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import jax
import jax.numpy as jnp
from jax.extend import core

SLOT = re.compile(r"T\[(\d+)\]")
INTEGRAL = re.compile(r"^\(?-?\d+(\.0)?\)?$")
LOAD = re.compile(r"[TC]\[\d+\]")
RANGE = re.compile(r"T\[(\d+)\+")

CALLS = {"jit", "pjit", "closed_call", "core_call", "custom_jvp_call", "custom_vjp_call",
         "custom_vjp_call_jaxpr", "remat", "checkpoint", "remat2", "named_call"}

# Primitives that only move entries. The value is which operands carry entries; the others
# (indices, predicates) must be constants.
ROUTING = {"broadcast_in_dim": None, "reshape": (0,), "squeeze": (0,), "expand_dims": (0,),
           "transpose": (0,), "rev": (0,), "slice": (0,), "concatenate": None, "stack": None,
           "split": (0,), "unstack": (0,), "pad": (0, 1), "gather": (0,), "dynamic_slice": (0,),
           "dynamic_update_slice": (0, 1), "scatter": (0, 2), "copy": (0,), "copy_p": (0,)}

UNARY = {name: f"Math.{name}" for name in
         ("exp", "log", "sin", "cos", "tan", "tanh", "sqrt", "abs", "floor", "ceil", "expm1",
          "log1p", "sign", "asin", "acos", "atan", "sinh", "cosh", "asinh", "acosh", "atanh", "cbrt")}
COMPARE = {"lt": "<", "gt": ">", "le": "<=", "ge": ">=", "eq": "===", "ne": "!=="}
COMMUTATIVE = {"add", "add_any", "mul", "max", "min", "eq", "ne", "and", "or", "xor"}
NUMPY = {"add": np.add, "add_any": np.add, "sub": np.subtract, "mul": np.multiply,
         "neg": np.negative, "lt": np.less, "gt": np.greater, "le": np.less_equal,
         "ge": np.greater_equal, "eq": np.equal, "ne": np.not_equal}

PRELUDE = """const F2U = x => x !== x || x <= 0 ? 0 : x >= 4294967295 ? 4294967295 : Math.trunc(x);
const F2I = x => x !== x ? 0 : x <= -2147483648 ? -2147483648 : x >= 2147483647 ? 2147483647 : Math.trunc(x);
const F2L = x => x !== x ? 0 : Math.trunc(x);
const RINT = x => { const r = Math.round(x); return r - x === 0.5 && r % 2 !== 0 ? r - 1 : r; };
const AWAY = x => Math.sign(x) * Math.round(Math.abs(x));"""


DTYPE = {"f": np.float64, "b": np.bool_, "u32": np.uint32, "i32": np.int32, "i64": np.int64}


def kind(dtype) -> str:
    """How a JAX dtype is carried in a double: float, bool, uint32, int32, or a wide integer."""
    d = np.dtype(dtype)
    if d == np.bool_: return "b"
    if d == np.float64: return "f"
    if d == np.uint32: return "u32"
    if d.kind in "iu" and d.itemsize <= 4: return "i32"
    if d.kind in "iu": return "i64"
    raise NotImplementedError(f"dtype {d}")


def literal(x) -> str:
    if isinstance(x, (bool, np.bool_)): return "1" if x else "0"
    if isinstance(x, (int, np.integer)): return str(int(x)) if x >= 0 else f"({int(x)})"
    x = float(x)
    if math.isnan(x): return "NaN"
    if math.isinf(x): return "Infinity" if x > 0 else "(-Infinity)"
    text = repr(x)
    return f"({text})" if text.startswith("-") else text


def js(atom) -> str:
    return atom if isinstance(atom, str) else literal(atom)


def entries(value) -> np.ndarray:
    """An array of constants as an object array of Python numbers."""
    return np.asarray(np.asarray(value)).astype(object)


def constant(a) -> bool:
    return not any(isinstance(x, str) for x in np.asarray(a, dtype=object).flat)


def typed(a, dtype):
    a = np.asarray(a, dtype=object)
    return np.array(a.tolist(), dtype=dtype).reshape(a.shape)


def subjaxpr(params):
    for key in ("jaxpr", "call_jaxpr", "fun_jaxpr"):
        if key in params:
            return params[key]
    raise NotImplementedError(f"no jaxpr among {sorted(params)}")


def opened(closed):
    if isinstance(closed, core.ClosedJaxpr):
        return closed.jaxpr, list(closed.consts)
    return closed, []


class Translation:
    """One traced function written as JavaScript statements over the slots ``T``."""

    def __init__(self):
        self.size = 0
        self.allocations = []      # (base, length) of every slot block, in order
        self.scopes = [{}]         # expression text -> slot, innermost last
        self.blocks = [[]]         # statement lists, innermost last
        self.ranges = {}           # base -> length, for blocks read at a run-time index
        self.negated = {}          # slot -> what it is the negative of
        self.selects = set()       # slots holding a select, whose cases are never folded in
        self.doubles = {}          # whole-number constants read as doubles, value -> index in C
        self.depth = 0

    # ---- statements ----

    def allocate(self, n):
        base = self.size
        self.size += n
        self.allocations.append((base, n))
        return base

    def emit(self, statement):
        self.blocks[-1].append(statement)

    def value(self, text):
        for scope in reversed(self.scopes):
            if text in scope:
                return scope[text]
        slot = self.allocate(1)
        self.emit(("set", slot, text))
        atom = f"T[{slot}]"
        self.scopes[-1][text] = atom
        return atom

    @contextmanager
    def block(self):
        self.scopes.append({})
        self.blocks.append([])
        try:
            yield self.blocks[-1]
        finally:
            self.scopes.pop()
            self.blocks.pop()

    # ---- scalars ----

    def scalar(self, name, args, kinds, out, params=None, prim=None, dtypes=None, out_dtype=None):
        if all(not isinstance(a, str) for a in args):
            return self.fold(name, args, dtypes or [DTYPE[k] for k in kinds], out_dtype or DTYPE[out],
                             prim, params)
        simple = self.simplify(name, args, kinds, out, params)
        if simple is not None:
            return simple
        text = self.render(name, [js(a) for a in args], kinds, out, params)
        atom = self.value(text)
        if name == "neg" and kinds[0] == "f":
            self.negated[atom] = args[0]
        if name == "select_n":
            self.selects.add(int(atom[2:-1]))
        return atom

    def fold(self, name, args, dtypes, out_dtype, prim, params):
        xs =[np.asarray(a, dtype=d) for a, d in zip(args, dtypes)]
        with np.errstate(all="ignore"):
            if name in NUMPY and all(kind(d) in "fb" for d in dtypes):
                r = NUMPY[name](*xs)
            elif name == "div" and all(kind(d) == "f" for d in dtypes):
                r = np.divide(*xs)
            elif name == "select_n":
                r = xs[1 + int(xs[0])]
            elif prim is None:
                prim = getattr(jax.lax, f"{name}_p")
                r = prim.bind(*map(jnp.asarray, xs), **(params or {}))
            else:
                r = prim.bind(*map(jnp.asarray, xs), **(params or {}))
        return np.asarray(r, dtype=out_dtype).item()

    def simplify(self, name, args, kinds, out, params):
        a = args[0]
        b = args[1] if len(args) > 1 else None
        const = lambda x, v: not isinstance(x, str) and x == v
        zero = 0.0 if out == "f" else 0
        if name in ("add", "add_any") and out != "b":
            if const(a, 0): return b
            if const(b, 0): return a
        if name == "sub" and out != "b":
            if const(b, 0): return a
            if const(a, 0) and out == "f": return self.scalar("neg", [b], kinds[1:], out)
        if name == "mul" and out != "b":
            if const(a, 0) or const(b, 0): return zero
            if const(a, 1): return b
            if const(b, 1): return a
            if out == "f" and const(a, -1): return self.scalar("neg", [b], kinds[1:], out)
            if out == "f" and const(b, -1): return self.scalar("neg", [a], kinds[:1], out)
        if name == "div" and out == "f":
            if const(b, 1): return a
            if const(b, -1): return self.scalar("neg", [a], kinds[:1], out)
        if name == "neg" and out == "f" and a in self.negated:
            return self.negated[a]
        if name == "select_n":
            if not isinstance(a, str):
                return args[1 + min(max(int(a), 0), len(args) - 2)]
            if all(c == args[1] for c in args[2:]):
                return args[1]
        if name in ("copy", "copy_p", "reduce_precision", "optimization_barrier"):
            return a
        if name == "convert_element_type":
            source = kinds[0]
            if source == out or (out in ("f", "i64") and source in ("b", "u32", "i32", "i64")) \
                    or (out in ("i32", "u32", "i64") and source == "b"):
                return a
        return None

    def render(self, name, a, kinds, out, params):
        k = kinds[0]
        wrap = {"u32": "({})>>>0", "i32": "({})|0"}.get(out, "{}")
        if name in COMMUTATIVE and len(a) == 2 and a[1] < a[0]:
            a = [a[1], a[0]]
        if name in ("add", "add_any"):
            return wrap.format(f"{a[0]}+{a[1]}")
        if name == "sub":
            return wrap.format(f"{a[0]}-{a[1]}")
        if name == "mul":
            if out in ("u32", "i32"):
                return wrap.format(f"Math.imul({a[0]},{a[1]})")
            return f"{a[0]}*{a[1]}"
        if name == "div":
            return f"{a[0]}/{a[1]}" if out == "f" else wrap.format(f"Math.trunc({a[0]}/{a[1]})")
        if name == "rem":
            return f"{a[0]}%{a[1]}"
        if name in ("copy", "copy_p", "reduce_precision", "optimization_barrier"):
            return a[0]
        if name == "neg":
            return wrap.format(f"-{a[0]}")
        if name in ("max", "min"):
            return f"Math.{name}({a[0]},{a[1]})"
        if name in ("pow", "atan2"):
            return f"Math.{name}({a[0]},{a[1]})"
        if name in COMPARE:
            return f"{a[0]}{COMPARE[name]}{a[1]}?1:0"
        if name == "select_n":
            pred, cases = a[0], a[1:]
            if out == "f":
                # A whole-number literal is a small integer to V8; a case that is one would make
                # the select's value change kind, and V8 discard the function when it does.
                # Such cases are read as doubles from the constant array C.
                cases = [self.double(c) if INTEGRAL.match(c) else c for c in cases]
            if len(cases) == 2:
                return f"{pred}!==0?{cases[1]}:{cases[0]}"
            text = cases[-1]
            for i in reversed(range(len(cases) - 1)):
                text = f"{pred}<={i}?{cases[i]}:({text})"
            return text
        if name in ("and", "or", "xor"):
            op = {"and": "&", "or": "|", "xor": "^"}[name]
            return wrap.format(f"{a[0]}{op}{a[1]}")
        if name == "not":
            return f"1-{a[0]}" if out == "b" else wrap.format(f"~{a[0]}")
        if name == "shift_right_logical":
            if out != "u32":
                return f"({a[0]}>>>{a[1]})|0"
            if a[1].isdigit():
                return f"{a[0]}>>>{a[1]}" if int(a[1]) < 32 else "0"
            return f"{a[1]}>=32?0:{a[0]}>>>{a[1]}"
        if name == "shift_left":
            return wrap.format(f"{a[0]}<<{a[1]}")
        if name == "shift_right_arithmetic":
            return f"{a[0]}>>{a[1]}"
        if name in UNARY:
            return f"{UNARY[name]}({a[0]})"
        if name == "logistic":
            return f"1/(1+Math.exp(-{a[0]}))"
        if name == "rsqrt":
            return f"1/Math.sqrt({a[0]})"
        if name == "one_minus_square":
            return f"1-{a[0]}*{a[0]}"
        if name == "square":
            return f"{a[0]}*{a[0]}"
        if name == "exp2":
            return f"Math.pow(2,{a[0]})"
        if name == "is_finite":
            return f"Number.isFinite({a[0]})?1:0"
        if name == "round":
            method = int(params["rounding_method"])
            return f"RINT({a[0]})" if method == 1 else f"AWAY({a[0]})"
        if name == "clamp":
            return f"Math.min(Math.max({a[1]},{a[0]}),{a[2]})"
        if name == "convert_element_type":
            if out == "b":
                return f"{a[0]}!==0?1:0"
            if k == "f" and out != "f":
                return {"u32": "F2U({})", "i32": "F2I({})", "i64": "F2L({})"}[out].format(a[0])
            if out == "f":
                return a[0]
            return wrap.format(a[0])
        raise NotImplementedError(f"scalar primitive {name} for {kinds} -> {out}")

    def double(self, text):
        value = float(text.strip("()"))
        if value not in self.doubles:
            self.doubles[value] = len(self.doubles)
        return f"C[{self.doubles[value]}]"

    def integer_pow(self, x, y, k):
        # The same multiplications, in the same order, as JAX's lowering of integer_pow.
        if y == 0:
            return 1.0 if k == "f" else 1
        reciprocal, y = y < 0, abs(y)
        acc = None
        while y > 0:
            if y & 1:
                acc = x if acc is None else self.scalar("mul", [acc, x], [k, k], k)
            y >>= 1
            if y > 0:
                x = self.scalar("mul", [x, x], [k, k], k)
        return self.scalar("div", [1.0, acc], [k, k], k) if reciprocal else acc

    # ---- equations ----

    def evaluate(self, closed, args):
        jaxpr, consts = opened(closed)
        env = {}
        for v, c in zip(jaxpr.constvars, consts, strict=True):
            env[v] = entries(c)
        for v, a in zip(jaxpr.invars, args, strict=True):
            env[v] = np.asarray(a, dtype=object).reshape(v.aval.shape)

        def read(v):
            return entries(v.val) if isinstance(v, core.Literal) else env[v]

        for eq in jaxpr.eqns:
            outs = self.equation(eq, [read(v) for v in eq.invars])
            for v, o in zip(eq.outvars, outs, strict=True):
                env[v] = np.asarray(o, dtype=object).reshape(v.aval.shape)
        return [read(v) for v in jaxpr.outvars]

    def equation(self, eq, args):
        name, p = eq.primitive.name, eq.params
        if name in CALLS:
            return self.evaluate(subjaxpr(p), args)
        if name == "scan":
            return self.scan(eq, args)
        if name == "while":
            return self.loop_while(eq, args)
        if name == "cond":
            return self.cond(eq, args)
        dtypes = [v.aval.dtype for v in eq.invars]
        if all(constant(a) for a in args):
            return self.fold_equation(eq, args, dtypes)
        if name in ROUTING:
            return self.route(eq, args, dtypes)
        if name == "scatter-add":
            return self.scatter_add(eq, args, dtypes)
        if name.startswith("reduce_") and name != "reduce_precision":
            return [self.reduce(eq, args[0], dtypes[0])]
        if name == "dot_general":
            return [self.dot(eq, args, dtypes)]
        if name == "integer_pow":
            k = kind(dtypes[0])
            return [np.frompyfunc(lambda x: self.integer_pow(x, p["y"], k), 1, 1)(args[0])]
        if eq.primitive.multiple_results:
            raise NotImplementedError(f"primitive {name}")
        kinds = [kind(d) for d in dtypes]
        out_dtype = eq.outvars[0].aval.dtype
        out = kind(out_dtype)
        prim = eq.primitive
        self.render(name, ["x"] * len(args), kinds, out, p)    # fails loudly on an unknown primitive

        def each(*xs):
            return self.scalar(name, list(xs), kinds, out, p, prim, dtypes, out_dtype)
        shape = eq.outvars[0].aval.shape
        xs = np.broadcast_arrays(*[np.asarray(a, dtype=object) for a in args])
        return [np.array(np.frompyfunc(each, len(xs), 1)(*xs), dtype=object).reshape(shape)]

    def fold_equation(self, eq, args, dtypes):
        if eq.primitive.name == "empty2":
            return [entries(np.zeros(v.aval.shape, v.aval.dtype)) for v in eq.outvars]
        outs = eq.primitive.bind(*[jnp.asarray(typed(a, d)) for a, d in zip(args, dtypes)], **eq.params)
        if not eq.primitive.multiple_results:
            outs = [outs]
        return [entries(o) for o in outs]

    def route(self, eq, args, dtypes):
        name, p = eq.primitive.name, dict(eq.params)
        carried = ROUTING[name]
        if carried is None:
            carried = tuple(range(len(args)))
        pool, ids, offset = [], [], 0
        for i, a in enumerate(args):
            if i in carried:
                a = np.asarray(a, dtype=object)
                pool.extend(a.ravel().tolist())
                ids.append(jnp.asarray(np.arange(offset, offset + a.size, dtype=np.int64).reshape(a.shape)))
                offset += a.size
            elif not constant(a):
                raise NotImplementedError(f"{name} with an index computed at run time")
            else:
                ids.append(jnp.asarray(typed(a, dtypes[i])))
        fill = None
        if name == "gather" and "FILL" in str(p.get("mode")):
            fill, p["fill_value"] = p.get("fill_value"), -1
            if fill is None:
                fill = math.nan
        outs = eq.primitive.bind(*ids, **p)
        if not eq.primitive.multiple_results:
            outs = [outs]
        pool = np.array(pool + [fill], dtype=object)
        results = []
        for o in outs:
            o = np.asarray(o)
            results.append(pool[np.where(o < 0, len(pool) - 1, o)])
        return results

    def scatter_add(self, eq, args, dtypes):
        operand, indices, updates = args
        if not constant(indices):
            raise NotImplementedError("scatter-add with indices computed at run time")
        updates = np.asarray(updates, dtype=object)
        u = updates.size
        index = jnp.asarray(typed(indices, dtypes[1]))
        zeros = jnp.zeros(np.shape(operand), jnp.float64)
        hits = jax.vmap(lambda e: eq.primitive.bind(zeros, index, e.reshape(updates.shape), **eq.params))(
            jnp.eye(u, dtype=jnp.float64))
        hits = np.asarray(hits).reshape(u, -1)
        k = kind(dtypes[0])
        out = np.asarray(operand, dtype=object).ravel().copy()
        flat = updates.ravel()
        for i in range(u):
            for j in np.nonzero(hits[i])[0]:
                for _ in range(int(round(hits[i, j]))):
                    out[j] = self.scalar("add", [out[j], flat[i]], [k, k], k)
        return [out.reshape(np.shape(operand))]

    def reduce(self, eq, a, dtype):
        name = eq.primitive.name[len("reduce_"):]
        k = kind(dtype)
        op = {"sum": "add", "prod": "mul", "max": "max", "min": "min", "and": "and", "or": "or",
              "xor": "xor"}.get(name)
        if op is None:
            raise NotImplementedError(f"primitive reduce_{name}")
        unit = {"add": 0, "mul": 1, "max": -math.inf, "min": math.inf, "and": True, "or": False,
                "xor": False}[op]
        a = np.asarray(a, dtype=object)
        axes = tuple(eq.params["axes"])
        kept = [d for d in range(a.ndim) if d not in axes]
        rows = np.transpose(a, kept + list(axes)).reshape(int(np.prod([a.shape[d] for d in kept])), -1)
        out = []
        for row in rows:
            acc = None
            for x in row:
                acc = x if acc is None else self.scalar(op, [acc, x], [k, k], k)
            out.append(unit if acc is None else acc)
        return np.array(out, dtype=object).reshape([a.shape[d] for d in kept])

    def dot(self, eq, args, dtypes):
        (lc, rc), (lb, rb) = eq.params["dimension_numbers"]
        a, b = (np.asarray(x, dtype=object) for x in args)
        lf = [d for d in range(a.ndim) if d not in lc and d not in lb]
        rf = [d for d in range(b.ndim) if d not in rc and d not in rb]
        size = lambda x, dims: int(np.prod([x.shape[d] for d in dims]))
        A = np.transpose(a, list(lb) + lf + list(lc)).reshape(size(a, lb), size(a, lf), size(a, lc))
        B = np.transpose(b, list(rb) + list(rc) + rf).reshape(size(b, rb), size(b, rc), size(b, rf))
        k = kind(eq.outvars[0].aval.dtype)
        out = np.empty((A.shape[0], A.shape[1], B.shape[2]), dtype=object)
        for n in range(A.shape[0]):
            for i in range(A.shape[1]):
                for j in range(B.shape[2]):
                    acc = None
                    for c in range(A.shape[2]):
                        t = self.scalar("mul", [A[n, i, c], B[n, c, j]], [k, k], k)
                        acc = t if acc is None else self.scalar("add", [acc, t], [k, k], k)
                    out[n, i, j] = 0.0 if acc is None else acc
        return out.reshape(eq.outvars[0].aval.shape)

    # ---- control ----

    def slots_for(self, values):
        """Fresh slots holding the given arrays' entries, written now; returns the slot arrays."""
        flat = [np.asarray(v, dtype=object) for v in values]
        base = self.allocate(sum(v.size for v in flat))
        out, offset = [], base
        for v in flat:
            for j, x in enumerate(v.ravel()):
                self.emit(("set", offset + j, js(x)))
            out.append(np.array([f"T[{offset + j}]" for j in range(v.size)], dtype=object).reshape(v.shape))
            offset += v.size
        return base, out

    def carry_update(self, base, carry, new):
        """Write a loop's new carry over its slots, reading every old value first."""
        flat = [x for v in new for x in np.asarray(v, dtype=object).ravel()]
        n = len(flat)
        staged = []
        for x in flat:
            if isinstance(x, str) and (any(base <= int(s) < base + n for s in SLOT.findall(x)) or RANGE.search(x)):
                slot = self.allocate(1)
                self.emit(("set", slot, x))
                x = f"T[{slot}]"
            staged.append(x)
        for j, x in enumerate(staged):
            if x != f"T[{base + j}]":
                self.emit(("set", base + j, js(x)))

    def scan(self, eq, args):
        p = eq.params
        if "ft_in" in p:      # JAX 0.11: the operands' grouping is a flat tree (consts, carry, xs)
            nc, nk, _ = (len(group) for group in p["ft_in"].unpack())
        else:
            nc, nk = p["num_consts"], p["num_carry"]
        length = p["length"]
        consts, carry, xs = args[:nc], args[nc:nc + nk], args[nc + nk:]
        body = p["jaxpr"]
        if all(constant(a) for a in args) or length <= 1:
            ys = []
            for i in range(length):
                t = length - 1 - i if p["reverse"] else i
                outs = self.evaluate(body, consts + carry + [np.asarray(x, dtype=object)[t] for x in xs])
                carry, y = outs[:nk], outs[nk:]
                ys.append(y)
            if p["reverse"]:
                ys = ys[::-1]
            stacked = [np.array([y[j] for y in ys], dtype=object).reshape(v.aval.shape)
                       for j, v in enumerate(eq.outvars[nk:])]
            return carry + stacked
        base, carry_slots = self.slots_for(carry)
        index = f"i{self.depth}"
        at = f"({length - 1}-{index})" if p["reverse"] else index
        xs_atoms = []
        for x in xs:
            x = np.asarray(x, dtype=object)
            stride = x[0].size
            xbase, _ = self.slots_for([x])
            self.ranges[xbase] = x.size
            xs_atoms.append(np.array([f"T[{xbase}+({at}*{stride}+{k})]" for k in range(stride)],
                                     dtype=object).reshape(x.shape[1:]))
        ys_shapes = [v.aval.shape for v in eq.outvars[nk:]]
        ys_bases = []
        for shape in ys_shapes:
            ybase = self.allocate(int(np.prod(shape)))
            self.ranges[ybase] = int(np.prod(shape))
            ys_bases.append(ybase)
        writes = set(range(base, base + sum(np.asarray(c, dtype=object).size for c in carry)))
        for ybase, shape in zip(ys_bases, ys_shapes):
            writes |= set(range(ybase, ybase + int(np.prod(shape))))
        self.depth += 1
        with self.block() as statements:
            outs = self.evaluate(body, consts + carry_slots + xs_atoms)
            for ybase, shape, y in zip(ys_bases, ys_shapes, outs[nk:]):
                stride = int(np.prod(shape[1:]))
                for k, value in enumerate(np.asarray(y, dtype=object).ravel()):
                    self.emit(("put", ybase, f"T[{ybase}+({at}*{stride}+{k})]", js(value)))
            self.carry_update(base, carry, outs[:nk])
        self.depth -= 1
        self.emit(("loop", f"for(let {index}=0;{index}<{length};{index}++)", statements, writes))
        ys = [np.array([f"T[{ybase + j}]" for j in range(int(np.prod(shape)))], dtype=object).reshape(shape)
              for ybase, shape in zip(ys_bases, ys_shapes)]
        return carry_slots + ys

    def loop_while(self, eq, args):
        p = eq.params
        cn, bn = p["cond_nconsts"], p["body_nconsts"]
        cconsts, bconsts, carry = args[:cn], args[cn:cn + bn], args[cn + bn:]
        if all(constant(a) for a in args):
            while True:
                (go,) = self.evaluate(p["cond_jaxpr"], cconsts + carry)
                if not constant(go):
                    break
                if not bool(np.asarray(go, dtype=object).item()):
                    return carry
                carry = self.evaluate(p["body_jaxpr"], bconsts + carry)
                if not all(constant(c) for c in carry):
                    break
            raise NotImplementedError("a while loop whose constant start leaves constants")
        base, carry_slots = self.slots_for(carry)
        writes = set(range(base, base + sum(np.asarray(c, dtype=object).size for c in carry)))
        self.depth += 1
        with self.block() as statements:
            (go,) = self.evaluate(p["cond_jaxpr"], cconsts + carry_slots)
            self.emit(("raw", f"if({js(np.asarray(go, dtype=object).item())}===0)break;"))
            outs = self.evaluate(p["body_jaxpr"], bconsts + carry_slots)
            self.carry_update(base, carry, outs)
        self.depth -= 1
        self.emit(("loop", "for(;;)", statements, writes))
        return carry_slots

    def cond(self, eq, args):
        index, operands = np.asarray(args[0], dtype=object).item(), args[1:]
        branches = eq.params["branches"]
        if not isinstance(index, str):
            return self.evaluate(branches[min(max(int(index), 0), len(branches) - 1)], operands)
        shapes = [v.aval.shape for v in eq.outvars]
        base = self.allocate(sum(int(np.prod(s)) for s in shapes))
        writes = set(range(base, base + sum(int(np.prod(s)) for s in shapes)))
        lists = []
        for branch in branches:
            with self.block() as statements:
                outs = self.evaluate(branch, operands)
                flat = [x for o in outs for x in np.asarray(o, dtype=object).ravel()]
                for j, x in enumerate(flat):
                    self.emit(("set", base + j, js(x)))
            lists.append(statements)
        self.emit(("if", index, lists, writes))
        out, offset = [], base
        for s in shapes:
            n = int(np.prod(s))
            out.append(np.array([f"T[{offset + j}]" for j in range(n)], dtype=object).reshape(s))
            offset += n
        return out

    # ---- clean-up and writing ----

    def reads(self, text):
        found = {int(s) for s in SLOT.findall(text)}
        for base in RANGE.findall(text):
            base = int(base)
            found.update(range(base, base + self.ranges[base]))
        return found

    def prune(self, statements, live):
        """Drop statements no live slot depends on; ``live`` grows with what the kept ones read."""
        kept = []
        for s in reversed(statements):
            tag = s[0]
            if tag == "set":
                if s[1] in live:
                    kept.append(s)
                    live |= self.reads(s[2])
            elif tag == "put":
                if live & set(range(s[1], s[1] + self.ranges[s[1]])):
                    kept.append(s)
                    live |= self.reads(s[3])
            elif tag == "raw":
                kept.append(s)
                live |= self.reads(s[1])
            elif tag == "loop":
                if s[3] & live:
                    live |= s[3]
                    while True:
                        before = len(live)
                        body = self.prune(s[2], live)
                        if len(live) == before:
                            break
                    kept.append(("loop", s[1], body, s[3]))
            elif tag == "if":
                if s[3] & live:
                    branches = [self.prune(b, live) for b in s[2]]
                    live |= self.reads(s[1])
                    kept.append(("if", s[1], branches, s[3]))
        kept.reverse()
        return kept

    def uses(self, statements, returned):
        """How often each slot is read, and the slots that are written more than once or read
        at a run-time index, which are never folded into the expression that reads them."""
        count, fixed = {}, set()

        def read(text):
            for s in SLOT.findall(text):
                count[int(s)] = count.get(int(s), 0) + 1
            for base in RANGE.findall(text):
                fixed.update(range(int(base), int(base) + self.ranges[int(base)]))

        def walk(ss):
            for s in ss:
                tag = s[0]
                if tag == "set": read(s[2])
                elif tag == "put": read(s[3]); fixed.update(range(s[1], s[1] + self.ranges[s[1]]))
                elif tag == "raw": read(s[1])
                elif tag == "loop": fixed.update(s[3]); walk(s[2])
                elif tag == "if": fixed.update(s[3]); read(s[1]); [walk(b) for b in s[2]]
        walk(statements)
        for text in returned:
            read(text)
        return count, fixed

    def inline(self, statements, count, fixed, longest=400):
        """Fold each slot read exactly once into the expression that reads it, when that
        expression is in the same block and nothing between writes what the slot's expression
        reads. The arithmetic is unchanged; only the stores and loads go."""
        out, pending = [], {}       # slot -> (position in out, expression, slots it reads)

        def substitute(text):
            def one(m):
                s = int(m.group(1))
                if s in pending:
                    position, expression, _ = pending.pop(s)
                    out[position] = None
                    return f"({expression})"
                return m.group(0)
            return SLOT.sub(one, text)

        def written(slots):
            for s in [s for s, (_, _, r) in pending.items() if r & slots]:
                del pending[s]

        for s in statements:
            tag = s[0]
            if tag == "set":
                if s[1] in self.selects:
                    # Fold into the predicate only. A case folded in would run only when taken,
                    # and V8 discards a function when code it has never run first runs.
                    head, tail = s[2].split("?", 1)
                    text = substitute(head) + "?" + tail
                else:
                    text = substitute(s[2])
                written({s[1]})
                out.append(("set", s[1], text))
                if (count.get(s[1], 0) == 1 and s[1] not in fixed and s[1] not in self.selects
                        and len(text) < longest):
                    pending[s[1]] = (len(out) - 1, text, self.reads(text))
            elif tag == "put":
                out.append(("put", s[1], s[2], substitute(s[3])))
                written(set(range(s[1], s[1] + self.ranges[s[1]])))
            elif tag == "raw":
                out.append(("raw", substitute(s[1])))
            elif tag == "loop":
                out.append(("loop", s[1], self.inline(s[2], count, fixed, longest), s[3]))
                written(s[3] | self.writes_within(s[2]))
            elif tag == "if":
                cond = substitute(s[1])
                out.append(("if", cond, [self.inline(b, count, fixed, longest) for b in s[2]], s[3]))
                written(s[3] | {x for b in s[2] for x in self.writes_within(b)})
        return [s for s in out if s is not None]

    def writes_within(self, statements):
        """The slots a block writes."""
        found = set()
        for s in statements:
            tag = s[0]
            if tag == "set": found.add(s[1])
            elif tag == "put": found.update(range(s[1], s[1] + self.ranges[s[1]]))
            elif tag in ("loop", "if"):
                found |= s[3]
                for b in (s[2] if tag == "if" else [s[2]]):
                    found |= self.writes_within(b)
        return found

    def slots_in(self, statements, returned):
        found = set()
        def walk(ss):
            for s in ss:
                tag = s[0]
                if tag == "set": found.add(s[1]); found.update(self.reads(s[2]))
                elif tag == "put": found.update(range(s[1], s[1] + self.ranges[s[1]])); found.update(self.reads(s[3]))
                elif tag == "raw": found.update(self.reads(s[1]))
                elif tag == "loop": walk(s[2])
                elif tag == "if": found.update(self.reads(s[1])); [walk(b) for b in s[2]]
        walk(statements)
        for text in returned:
            found.update(self.reads(text))
        return found

    def compact(self, statements, keep):
        """Renumber the slot blocks still used so that they are contiguous."""
        new, offset = {}, 0
        used = sorted(keep)
        allocations = sorted(self.allocations)
        starts = [b for b, _ in allocations]
        live_blocks = set()
        for s in used:
            i = bisect.bisect_right(starts, s) - 1
            live_blocks.add(i)
        for i in sorted(live_blocks):
            b, n = allocations[i]
            new[b] = offset
            offset += n
        def where(slot):
            i = bisect.bisect_right(starts, slot) - 1
            b = allocations[i][0]
            return new[b] + slot - b
        sub = lambda text: RANGE.sub(lambda m: f"T[{where(int(m.group(1)))}+",
                                     SLOT.sub(lambda m: f"T[{where(int(m.group(1)))}]", text))

        def walk(ss):
            out = []
            for s in ss:
                tag = s[0]
                if tag == "set": out.append(("set", where(s[1]), sub(s[2])))
                elif tag == "put": out.append(("put", where(s[1]), sub(s[2]), sub(s[3])))
                elif tag == "raw": out.append(("raw", sub(s[1])))
                elif tag == "loop": out.append(("loop", s[1], walk(s[2]), moved(s[3])))
                elif tag == "if": out.append(("if", sub(s[1]), [walk(b) for b in s[2]], moved(s[3])))
            return out
        moved = lambda slots: {where(x) for x in slots if bisect.bisect_right(starts, x) - 1 in live_blocks}
        statements = walk(statements)
        self.ranges = {where(b): n for b, n in self.ranges.items()
                       if bisect.bisect_right(starts, b) - 1 in live_blocks}
        self.selects = moved(self.selects)
        return statements, offset, sub

    def lines(self, s, indent, local=frozenset(), rename=lambda text: text):
        pad = "  " * indent
        tag = s[0]
        if tag == "set":
            target = f"const v{s[1]}" if s[1] in local else f"T[{s[1]}]"
            text = rename(s[2])
            hoisted = []
            if s[1] in self.selects:
                # Load every case before choosing: a load V8 has never run discards the function.
                head, tail = text.split("?", 1)
                for j, load in enumerate(dict.fromkeys(LOAD.findall(tail))):
                    hoisted.append(f"{pad}const w{s[1]}_{j}={load};")
                    tail = tail.replace(load, f"w{s[1]}_{j}")
                text = head + "?" + tail
            return [*hoisted, f"{pad}{target}={text};"]
        if tag == "put": return [f"{pad}{s[2]}={rename(s[3])};"]
        if tag == "raw": return [f"{pad}{rename(s[1])}"]
        if tag == "loop":
            return [f"{pad}{s[1]}{{", *[l for t in s[2] for l in self.lines(t, indent + 1, local, rename)],
                    f"{pad}}}"]
        if tag == "if":
            branches = s[2]
            out = []
            for i, b in enumerate(branches):
                head = (f"if({rename(s[1])}<={i}){{" if i == 0 else
                        "}else{" if i == len(branches) - 1 else f"}}else if({rename(s[1])}<={i}){{")
                out.append(pad + head)
                out.extend(l for t in b for l in self.lines(t, indent + 1, local, rename))
            out.append(pad + "}")
            return out
        raise AssertionError(tag)


def javascript(fn, example_input, name, fields=(), chunk=1500):
    """JavaScript source for ``function name(...)`` computing ``fn`` on flat arrays.

    ``example_input`` is one array or a tuple of arrays, one per argument of ``fn``; the
    JavaScript function takes the same arguments as plain or typed arrays and returns the
    one output as an Array, or several outputs as an object keyed by ``fields``.
    """
    inputs = example_input if isinstance(example_input, tuple) else (example_input,)
    inputs = tuple(np.asarray(x) for x in inputs)
    ir = jax.make_jaxpr(fn)(*inputs)
    t = Translation()
    args = []
    for x in inputs:
        base = t.allocate(x.size)
        args.append(np.array([f"T[{base + j}]" for j in range(x.size)], dtype=object).reshape(x.shape))
    outs = t.evaluate(ir, args)
    flat_outs = [np.asarray(o, dtype=object).ravel() for o in outs]
    live = set(range(sum(x.size for x in inputs)))
    for o in flat_outs:
        for x in o:
            if isinstance(x, str):
                live |= t.reads(x)
    statements = t.prune(t.blocks[0], live)
    returned = ["[" + ",".join(js(x) for x in o) + "]" for o in flat_outs]
    count, fixed = t.uses(statements, returned)
    fixed.update(range(sum(x.size for x in inputs)))
    statements = t.inline(statements, count, fixed)
    keep = t.slots_in(statements, returned) | set(range(sum(x.size for x in inputs)))
    statements, size, sub = t.compact(statements, keep)
    returned = [sub(r) for r in returned]

    # Each branch of a ``cond`` becomes its own function, so that the first run of a branch
    # (the search's choice, say) does not make V8 throw away and rebuild a large function.
    branches = []

    def extract(ss):
        out = []
        for s in ss:
            if s[0] == "if":
                calls = []
                for b in s[2]:
                    entry = [f"{name}_b{len(branches)}", None]
                    branches.append(entry)
                    calls.append([("raw", f"{entry[0]}(T);")])
                    entry[1] = extract(b)
                out.append(("if", s[1], calls, s[3]))
            else:
                out.append(s)
        return out
    statements = extract(statements)

    # Cut the top-level statements into functions of about ``chunk`` lines. A slot written and
    # read only inside one function becomes a local constant there.
    groups, current, lines = [], [], 0
    for s in statements:
        current.append(s)
        lines += len(t.lines(s, 1))
        if lines >= chunk:
            groups.append(current)
            current, lines = [], 0
    if current:
        groups.append(current)
    names = [f"{name}_{g}" for g in range(len(groups))] + [n for n, _ in branches]
    called = len(groups)
    groups += [b for _, b in branches]
    home, readers = {}, {}

    def visit(ss, g):
        for s in ss:
            tag = s[0]
            texts = {"set": lambda: [s[2]], "put": lambda: [s[3]], "raw": lambda: [s[1]],
                     "loop": lambda: [], "if": lambda: [s[1]]}[tag]()
            for text in texts:
                for x in SLOT.findall(text):
                    readers.setdefault(int(x), set()).add(g)
            if tag == "set" and s[1] not in fixed_after:
                home[s[1]] = g
            if tag == "loop":
                visit(s[2], g)
            elif tag == "if":
                for b in s[2]:
                    visit(b, g)
    fixed_after = set(range(sum(x.size for x in inputs)))
    for base, n in t.ranges.items():
        fixed_after.update(range(base, base + n))

    def fixed_in(ss):
        for s in ss:
            if s[0] == "loop":
                fixed_after.update(s[3]); fixed_in(s[2])
            elif s[0] == "if":
                fixed_after.update(s[3]); [fixed_in(b) for b in s[2]]
    for group in groups:
        fixed_in(group)
    for g, group in enumerate(groups):
        visit(group, g)
    for text in returned:
        for x in SLOT.findall(text):
            readers.setdefault(int(x), set()).add(-1)
    local = {s for s, g in home.items() if readers.get(s, set()) <= {g}}
    rename = lambda text: SLOT.sub(lambda m: f"v{m.group(1)}" if int(m.group(1)) in local else m.group(0), text)
    functions, calls = [], []
    for g, group in enumerate(groups):
        body = [l for s in group for l in t.lines(s, 1, local, rename)]
        if any("C[" in l for l in body):
            body.insert(0, f"  const C={name}_C;")
        functions.append(f"function {names[g]}(T){{\n" + "\n".join(body) + "\n}")
        if g < called:
            calls.append(f"  {names[g]}(T);")
    arguments = [f"a{i}" for i in range(len(inputs))]
    loads, offset = [], 0
    for a, x in zip(arguments, inputs):
        loads.append(f"  for(let k=0;k<{x.size};k++)T[{offset}+k]={a}[k];")
        offset += x.size
    if fields:
        result = "{" + ",".join(f"{k}:{v}" for k, v in zip(fields, returned, strict=True)) + "}"
    else:
        result = returned[0] if len(returned) == 1 else "[" + ",".join(returned) + "]"
    main = "\n".join([f"function {name}({','.join(arguments)}){{",
                      f"  const T={name}_T;", *loads, *calls, f"  return {result};", "}"])
    doubles = ",".join(literal(v) for v in t.doubles)
    return "\n".join([f"const {name}_T=new Float64Array({max(size, 1)});",
                      f"const {name}_C=new Float64Array([{doubles}]);", *functions, main])


def module(parts, global_name, metadata, header):
    """A browser-or-node file defining the generated functions and some data beside them."""
    names = [n for n, _ in parts]
    body = "\n".join(code for _, code in parts)
    api = "{..." + json.dumps(metadata, allow_nan=False) + "," + ",".join(names) + "}"
    return (f"// {header}\n(function(root){{\n'use strict';\n{PRELUDE}\n{body}\n"
            f"const api={api};\n"
            f"if(typeof module!=='undefined'&&module.exports)module.exports=api;\n"
            f"else root.{global_name}=api;\n}})(typeof globalThis!=='undefined'?globalThis:this);\n")


# ---------------------------------------------------------------------------
# Feeding with a separator mouth and a slot (``feeding_composed.feeding_with_slot``): the
# step and a readout function take the places and the page's knobs as inputs.
# ---------------------------------------------------------------------------

PLACES, FOODS, H = (4., 8., -5.), (10., 10., .05), .004
KNOBS = ("A", "csee", "ktau", "searchSeed")
# The example places and foods of an export for up to six casings: the page's three, then three more.
PAGE_PLACES, PAGE_FOODS = PLACES + (-8., 1., -2.), FOODS + (10., 10., 10.)
EMPTY = np.zeros(0)


def stores(arr):
    """The composite's stores in the order of its flat state: for each primitive part with a
    state, its label, the labels of the named arrangements it sits inside, where its entries
    start and how many there are. Read off ``arr.construction``; checked against the flat
    starting state."""
    from jax.flatten_util import ravel_pytree
    found, offset = [], 0

    def named(a):
        return a.label and not a.label.startswith("(")

    def walk(a, path):
        nonlocal offset
        if a.construction:
            operation, f, g = a.construction
            inner = path + [a.label] if named(a) and (not path or path[-1] != a.label) else path
            if operation == "compose" and not g.construction and named(g) and (not inner or inner[-1] != g.label):
                inner = inner + [g.label]
            walk(f, inner)
            walk(g, inner)
            return
        size = ravel_pytree(a.P.init)[0].size
        if size:
            found.append(dict(label=a.P.label or a.label, path=path, start=offset, size=size))
            offset += size

    walk(arr, [])
    flat = np.asarray(ravel_pytree(arr.P.init)[0])
    if offset != flat.size:
        raise ValueError(f"the stores cover {offset} entries of {flat.size}")
    return found


def _factors(a, count):
    """The ``count`` factors of a left-folded ``tensor_arrangements``, in order."""
    out = []
    for _ in range(count - 1):
        operation, a, last = a.construction
        if operation != "tensor":
            raise ValueError("expected a tensor of parts")
        out.append(last)
    return [a, *reversed(out)]


def _parts_at(filled, count, q, m, n):
    """For a ``feeding_parts.fill`` of ``count`` parts, at its configuration ``q``, its
    internal values ``m`` and the inputs ``n`` it is sent: each part with its configuration
    and the inputs its wiring sends it. Read off the composite's construction; the values are
    the library's own ``out_f`` and ``in_f``."""
    operation, tensor, wiring = filled.construction
    if operation != "compose":
        raise ValueError("expected a filled wiring")
    q_t, q_w = q[:tensor.P.Q_dim], q[tensor.P.Q_dim:]
    sent = wiring.in_f(q_w, tensor.out_f(q_t, m), n)
    parts, qa, na, ma = [], 0, 0, 0
    for part in _factors(tensor, count):
        parts.append((part, q_t[qa:qa + part.P.Q_dim], m[ma:ma + part.out_dim_M], sent[na:na + part.in_dim_N]))
        qa, na, ma = qa + part.P.Q_dim, na + part.in_dim_N, ma + part.out_dim_M
    return parts


def _input_loads(part, q, m, n):
    """The load ``part`` puts on its inputs at ``(q, m, n)``: its own one-form's component there."""
    return part.omega(q, m, n)[2]


# The battery's rate, split by what draws on it or charges it (``observe``'s ``pays``).
# Since P109 (k) each is what that organ's cord draws from the robot's battery.
PAYS = ("gripper", "lever", "wheels", "eye", "memory", "hands (latches and lids)", "mouth", "basal cost", "gut wall",
        "in all")
# The grams the mouth moves, per unit of duration (``observe``'s ``flows``).
FLOWS = ("taken", "into the chamber", "husks into the collector", "taken up by the gut wall",
         "collector to the pile")
# The rates ``observe`` reads off the body's wiring (``observe``'s ``told_motors``).
MOTORS = ("torque", "grip rate", "lift rate", "scan free")
SAMPLES = np.linspace(-9., 9., 181)


def slot_model(count, places=None, foods=None, h=H, knobs=KNOBS):
    """The library's four Runge-Kutta rounds of ``feeding_with_slot(places, foods, learning=True)``
    on the composite's own flat state, as ``step(state, places, params)``; the page's readouts
    as ``observe(state, places, params)``; the composite's starting state as
    ``start(places, foods, params, usual, returnWeights, cart, battery)``; and the usual yield and
    the search density along the line as ``curve(state, at)``, ``at`` one more place beside the
    fixed ``SAMPLES``. ``params`` holds the ``knobs`` in order, every other parameter is
    ``DEFAULTS``. Returns the functions, example inputs and the metadata."""
    from jax.flatten_util import ravel_pytree
    from arrangements.feeding import CENTRES, DEFAULTS, _search_map, density_at, usual_at
    from arrangements.feeding_composed import feeding_with_slot
    from arrangements.functors import Phi
    from arrangements.interpretation import trivial_omega
    from arrangements.pcK import PhiK

    # Example places and foods only (both are inputs of every function): the page's three, then more.
    places = np.asarray(PAGE_PLACES[:count] if places is None else places, dtype=float)
    foods = np.asarray(PAGE_FOODS[:count] if foods is None else foods, dtype=float)
    if len(places) != count or len(foods) != count:
        raise ValueError("one place and one food per casing")
    params = np.asarray([DEFAULTS[k] for k in knobs], dtype=float)
    parameters = lambda params: {**DEFAULTS, **{k: params[i] for i, k in enumerate(knobs)}}

    def build(places, params, stages=4, foods=foods, **kw):
        return feeding_with_slot(tuple(places[i] for i in range(count)), tuple(foods[i] for i in range(count)),
                                 learning=True, stages=stages, h=h if stages == 4 else 1., parameters=parameters(params),
                                 **kw)

    concrete = feeding_with_slot(tuple(float(x) for x in places), tuple(float(x) for x in foods), learning=True,
                                 stages=4, h=h)
    initial, unravel = ravel_pytree(concrete.P.init)
    # The one-round composite's state, flat, has the same layout (its rates are read there).
    initial1, unravel1 = ravel_pytree(feeding_with_slot(tuple(float(x) for x in places), tuple(float(x) for x in foods),
                                                        learning=True).P.init)
    if initial1.shape != initial.shape or not np.array_equal(initial1, initial):
        raise ValueError("the one-round and four-round states differ")
    layout = stores(concrete)
    casing = 0
    for s in layout:      # number the casings in order: each one's stores begin with its food
        if s["path"] and s["path"][0].startswith("casing("):
            casing += s["label"] == "food"
            s["casing"] = casing - 1
    empty = jnp.zeros(0)
    inputs = (empty, trivial_omega(0))
    loads = lambda _: (jnp.zeros(2), empty)

    def step(flat, places, params):
        dynamics = PhiK(build(places, params))
        return ravel_pytree(dynamics.with_state(unravel(flat)).run_one(inputs, loads)[1])[0]

    def start(places, foods, params, usual, weights, cart, battery):
        arr = build(places, params, foods=foods, cart=cart[0], battery=battery[0],
                    memory={'usual': usual, 'returnWeights': weights})
        return ravel_pytree(arr.P.init)[0]

    def store(flat, label):
        (s,) = [s for s in layout if s["label"] == label]
        return flat[s["start"]:s["start"] + s["size"]]

    def observe(flat, places, params):
        arr = build(places, params)
        q = arr.P.reads[0](unravel(flat))
        _, world, coupling = arr.construction
        _, inside, wiring = coupling.construction
        q_world, q_coupling = q[:world.P.Q_dim], q[world.P.Q_dim:]
        q_inside, q_wiring = q_coupling[:inside.P.Q_dim], q_coupling[inside.P.Q_dim:]
        out = world.out_f(q_world, empty)
        sent = wiring.in_f(q_wiring, inside.out_f(q_inside, out), empty)
        told = sent[:8]
        richness, masses = _search_map(store(flat, "usual yield"), store(flat, "unexplored"))

        # The rate of every store: one Euler step of unit length of the same composite (``Phi``).
        euler = Phi(build(places, params, stages=1))
        rate = ravel_pytree(euler.step(unravel1(flat))[1](inputs)[1](loads(None))[1])[0] - flat

        # What each part puts on the battery: its own one-form at the inputs it is sent.
        robot = _factors(world, count + 2)[0]
        _, filler, seeker = robot.construction
        q_robot = q_world[:robot.P.Q_dim]
        q_filler, q_seeker = q_robot[:filler.P.Q_dim], q_robot[filler.P.Q_dim:]
        # The seeker's parts (P109 (k)): the body, the mouth, the collector, the slot, the mouth's
        # battery and cord, the satiety store and its motor, the basal cost.
        body, mouth, collector, _, _, mouth_cord, _, _, basal = _parts_at(seeker, 9, q_seeker, empty, told)
        handed = seeker.in_f(q_seeker, filler.out_f(q_filler, empty), told)     # what the slot hands the filler
        gut = _input_loads(filler, q_filler, empty, handed)
        mouth_loads = _input_loads(*mouth)
        settle_rate = mouth[0].omega(*mouth[1:])[0]
        export = _input_loads(*collector)
        # ``feeding_composed.body``'s parts, in its order: six stores (the fourth the wheels), the
        # memory's stores, the organs' batteries, then the gripper, the lever, the eye, one learner
        # per memory store, and the organs' cords (``feeding_composed.ORGANS``: drive, grip, lift,
        # eye, brain). What each organ draws on the robot's battery is its cord's load there.
        size = _left_spine(body[0].construction[1])
        organs = 5
        memory = (size - 9 - 2 * organs) // 2
        roles = [None, None, None, None, None, None, *[None] * memory, *[None] * organs, None, None, None,
                 *[None] * memory, "wheels", "gripper", "lever", "eye", "memory"]
        by_name = {}
        for role, (part, *at) in zip(roles, _parts_at(body[0], size, *body[1:]), strict=True):
            if role:
                by_name.setdefault(role, []).append((part, at))
        battery_of = lambda name: sum((_input_loads(part, *at)[0] for part, at in by_name.get(name, [])), 0.)
        motors = [battery_of(name) for name in ("gripper", "lever", "wheels", "eye", "memory")]
        # The coupling's pieces: per casing a contact and a latch, then taking, one lid motor per
        # casing, spilling, the hands' battery and the hands' cord, which carries the latches' and
        # lids' draw.
        pieces = _factors(inside.construction[2], 3 * count + 4)
        q_pieces = q_inside[inside.construction[1].P.Q_dim:]
        sent_pieces, qa, na = sent[8:], 0, 0
        hands = 0.
        for k, piece in enumerate(pieces):
            if k == 3 * count + 3:
                hands = _input_loads(piece, q_pieces[qa:qa + piece.P.Q_dim], empty,
                                     sent_pieces[na:na + piece.in_dim_N])[0]
            qa, na = qa + piece.P.Q_dim, na + piece.in_dim_N
        pays = jnp.stack([*motors, hands, _input_loads(*mouth_cord)[0], _input_loads(*basal)[0], gut[1],
                          store(rate, "battery")[0]])
        flows = jnp.stack([told[5], settle_rate[0], mouth_loads[2], -gut[0], export[0]])

        # What the body's wiring tells the wheels, the gripper's and lever's motors, and the eye.
        parts_of = lambda i: _parts_at(body[0], size, *body[1:])[i]
        at_part = 6 + memory + organs
        wheels, gripper, lever, eye = (parts_of(3), parts_of(at_part), parts_of(at_part + 1), parts_of(at_part + 2))
        grip_motor = _parts_at(gripper[0], 1, *gripper[1:])[0][3]
        lift_motor = _parts_at(lever[0], 1, *lever[1:])[0][3]
        told_motors = jnp.stack([wheels[3][1], grip_motor[2], lift_motor[2], eye[3][3]])
        return (told, out[:7], out[7:7 + 6 * count], out[7 + 6 * count:], store(flat, "usual yield"),
                richness, masses, store(flat, "destination"), rate, pays, flows, told_motors)

    def curve(flat, at):
        """The usual yield and the search density at ``SAMPLES`` and at ``at``."""
        xs = jnp.concatenate([jnp.asarray(SAMPLES), at])
        usual = store(flat, "usual yield")
        _, masses = _search_map(usual, store(flat, "unexplored"))
        return usual_at(usual, xs), density_at(masses, xs)

    from arrangements.feeding_composed import SLOT_SEEKER_INPUTS, SLOT_SEEKER_OUTPUTS
    zeros = np.zeros(len(CENTRES))
    examples = dict(step=(np.asarray(initial), places, params),
                    start=(places, foods, params, zeros, zeros, np.zeros(1), np.asarray([6.])),
                    curve=(np.asarray(initial), np.zeros(1)))
    names = ("mouthPass", "mouthCost", "absorbYield", "uptake", "clearance", "chamberCap", "collectorCap",
             "leat", "b0", "memoryCost", "x0", "xg", "wLocal", "battCap", "sateRoom", "hungryRoom", "lidOpen")
    metadata = dict(casings=count, places=places.tolist(), foods=foods.tolist(), h=h, knobs=list(knobs),
                    defaults=params.tolist(), initial=np.asarray(initial).tolist(), stores=layout,
                    entryStore=[i for i, s in enumerate(layout) for _ in range(s["size"])],
                    parameters={k: DEFAULTS[k] for k in names}, centres=list(CENTRES), samples=SAMPLES.tolist(),
                    observed=dict(told=list(SLOT_SEEKER_INPUTS), seeker=list(SLOT_SEEKER_OUTPUTS),
                                  casings=["food", "height", "red", "place", "lid", "speed"],
                                  pile=["mass", "acceptance"],
                                  usual="usual yield per centre", richness="richness per centre",
                                  masses="search mass per centre", destination="destination store",
                                  rate="the rate of every entry of the state", pays=list(PAYS), flows=list(FLOWS),
                                  told_motors=list(MOTORS)))
    return dict(step=step, observe=observe, start=start, curve=curve), examples, metadata


def _left_spine(tensor):
    """How many factors a left-folded tensor has whose first factor is not itself a tensor."""
    size = 1
    while tensor.construction and tensor.construction[0] == "tensor":
        tensor, size = tensor.construction[1], size + 1
    return size


OBSERVED = ("told", "seeker", "casings", "pile", "usual", "richness", "masses", "destination",
            "rate", "pays", "flows", "told_motors")


def export_slot(path, count, places=None, foods=None, h=H):
    functions, examples, metadata = slot_model(count, places, foods, h)
    began = time.time()
    parts = [("step", javascript(functions["step"], examples["step"], "step")),
             ("observe", javascript(functions["observe"], examples["step"], "observe", fields=OBSERVED)),
             ("start", javascript(functions["start"], examples["start"], "start")),
             ("curve", javascript(functions["curve"], examples["curve"], "curve", fields=("usual", "density")))]
    metadata["exportSeconds"] = round(time.time() - began, 1)
    text = module(parts, f"FeedingSlot{count}", metadata,
                  f"Generated from arrangements.feeding_composed.feeding_with_slot by "
                  f"scripts/export_step_js.py (--slot --casings {count}); no equation is typed here.")
    Path(path).write_text(text)
    return text


if __name__ == "__main__":
    jax.config.update("jax_enable_x64", True)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--casings", type=int, required=True)
    parser.add_argument("--places", type=float, nargs="*", help="one place per casing (default: the page's)")
    parser.add_argument("--slot", action="store_true",
                        help="accepted for older command lines: feeding_with_slot is the only feeding export")
    parser.add_argument("--h", type=float, default=H, help="Runge-Kutta step length")
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    text = export_slot(a.output, a.casings, a.places, h=a.h)
    print(f"wrote {a.output}: {len(text)} characters")
