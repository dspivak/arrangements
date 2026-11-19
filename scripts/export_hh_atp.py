"""Build the offline membrane step and observation functions from the library's assembled IMC system.

From the library checkout run .venv/bin/python scripts/export_hh_atp.py.
The small exporter supports the scalar arithmetic and static array indexing used here.
Unsupported JAX operations fail the build. No membrane equations are supplied by this file.
"""
import argparse
import json
from itertools import accumulate
from pathlib import Path

import jax
from jax.extend import core
import numpy as np
from arrangements import hh_atp


class ScalarCode:
    def __init__(self):
        self.names = {}
        self.lines = []

    def bind(self, expression):
        if expression not in self.names:
            name = f"t{len(self.names)}"
            self.names[expression] = name
            self.lines.append(f"  const {name} = {expression};")
        return self.names[expression]

    def scalar(self, op, *xs):
        # Eliminate the zeros and scalar copies created by tensor products and port routing.
        a, *rest = xs
        b = rest[0] if rest else None
        if op in ("add", "add_any"):
            if a == "0": return b
            if b == "0": return a
        if op == "mul":
            if "0" in xs: return "0"
            if a == "1": return b
            if b == "1": return a
        binary = {"add": "+", "add_any": "+", "sub": "-", "mul": "*", "div": "/",
                  "lt": "<", "gt": ">"}
        if op in binary:
            return self.bind(f"{a} {binary[op]} {b}")
        if op == "select_n":
            return self.bind(f"{a} ? {rest[1]} : {b}")
        if op == "neg":
            return self.bind(f"-({a})")
        if op == "logistic":
            return self.bind(f"1 / (1 + Math.exp(-({a})))")
        if op == "integer_pow":
            return self.bind(f"Math.pow({a}, {b})")
        if op in ("abs", "log", "exp", "expm1", "tanh"):
            return self.bind(f"Math.{op}({a})")
        raise ValueError(f"Unsupported scalar operation {op}")

    def evaluate(self, graph, constants, inputs):
        def literal(value):
            def number(x):
                if x in (0, 1): return str(int(x))
                return json.dumps(float(x), allow_nan=False)
            return np.asarray(np.frompyfunc(number, 1, 1)(np.asarray(value)), dtype=object)

        env = dict(zip(graph.constvars, map(literal, constants), strict=True))
        env.update(zip(graph.invars, inputs, strict=True))
        def read(v):
            return literal(v.val) if isinstance(v, core.Literal) else env[v]
        for eq in graph.eqns:
            args = [read(v) for v in eq.invars]
            op, p = eq.primitive.name, eq.params
            a = args[0]
            if op == "jit":
                nested = p["jaxpr"]
                results = self.evaluate(nested.jaxpr, nested.consts, args)
            elif op == "split":
                results = np.split(a, list(accumulate(p["sizes"]))[:-1], axis=p["axis"])
            elif op == "unstack":
                results = list(np.moveaxis(a, p["axis"], 0))
            else:
                if op == "convert_element_type": value = a
                elif op == "concatenate": value = np.concatenate(args, axis=p["dimension"])
                elif op == "stack": value = np.stack(args, axis=p["axis"])
                elif op == "squeeze": value = np.squeeze(a, axis=p["dimensions"])
                elif op == "slice":
                    step = p["strides"] or (1,) * a.ndim
                    value = a[tuple(map(slice, p["start_indices"], p["limit_indices"], step))]
                elif op == "broadcast_in_dim":
                    shape = [1] * len(p["shape"])
                    for i, size in zip(p["broadcast_dimensions"], a.shape, strict=True):
                        shape[i] = size
                    value = np.broadcast_to(a.reshape(shape), p["shape"])
                elif op == "pad":
                    if any(inner or lo < 0 or hi < 0 for lo, hi, inner in p["padding_config"]):
                        raise ValueError("Expected nonnegative edge padding")
                    value = np.pad(a, [(lo, hi) for lo, hi, _ in p["padding_config"]],
                                   constant_values=args[1].item())
                elif op == "dot_general":
                    if p["dimension_numbers"] != (((1,), (0,)), ((), ())) or a.ndim != 2 or args[1].ndim != 1:
                        raise ValueError("Expected an ordinary matrix-vector product")
                    value = [self.sum(self.scalar("mul", x, y) for x, y in zip(row, args[1], strict=True)) for row in a]
                elif op == "reduce_sum" and p["axes"] == (0,) and a.ndim == 1:
                    value = self.sum(a)
                else:
                    if op == "integer_pow": args.append(str(p["y"]))
                    value = np.frompyfunc(lambda *xs: self.scalar(op, *xs), len(args), 1)(*args)
                results = [value]
            env.update((k, np.asarray(v, dtype=object)) for k, v in zip(eq.outvars, results, strict=True))
        return [read(v) for v in graph.outvars]

    def sum(self, terms):
        result = "0"
        for term in terms:
            result = self.scalar("add", result, term)
        return result


def javascript(fn, initial, name, fields=()):
    ir = jax.make_jaxpr(fn)(initial)
    code = ScalarCode()
    inputs = np.array([f"state[{i}]" for i in range(initial.size)], dtype=object)
    arrays = code.evaluate(ir.jaxpr, ir.consts, [inputs])
    values = ["[" + ", ".join(a.tolist()) + "]" for a in arrays]
    result = ("{" + ", ".join(k + ": " + v for k, v in zip(fields, values, strict=True)) + "}"
              if fields else values[0])
    return "\n".join([f"function {name}(state) {{", *code.lines, f"  return {result};", "}"])


def export(path):
    state, step, observe, jacobian = hh_atp.live_membrane()
    parts = [javascript(step, state, "step"),
             javascript(observe, state, "observe", ["q", "values", "loads", "total"])]
    metadata = dict(initial=state.tolist(), jacobian=jacobian.tolist(),
                    outside=hh_atp.OUTSIDE.tolist(), dtMs=1000 * hh_atp.STEP_SECONDS)
    header = "// Generated from arrangements.hh_atp by scripts/export_hh_atp.py.\n"
    path.write_text(header + "(function(root){\n'use strict';\n" + "\n".join(parts)
                    + "\nroot.MembraneDynamics = {..." + json.dumps(metadata, allow_nan=False)
                    + ", step, observe};\n})(globalThis);\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path(__file__).resolve().parents[1] / "viewer" / "hh-atp-dynamics.js")
    export(parser.parse_args().output)
