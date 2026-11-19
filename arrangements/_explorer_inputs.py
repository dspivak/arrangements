"""Coordinate dependencies used by the array explorer, independent of the observed values."""
from __future__ import annotations

import numpy as np
import jax
from jax.extend import core


# Bit zero marks an operation whose coordinate-level dependencies are unknown.
# All remaining bits identify coordinates in the three input vectors.
_ELEMENTWISE = set("""
abs acos acosh add add_any and asin asinh atan atan2 atanh cbrt ceil
clamp complex conj convert_element_type cos cosh div eq exp expm1 floor ge gt imag
integer_pow is_finite le log log1p logistic lt max min mul ne neg not or pow real rem
round rsqrt select_n sign sin sinh sqrt stop_gradient sub tan tanh xor
""".split())


def _union(array):
    return np.bitwise_or.reduce(array.ravel(), initial=0)


def _propagate(graph, constants, inputs):
    env = dict(zip(graph.invars, inputs))
    known = dict(zip(graph.constvars, constants))
    env.update((var, np.zeros(np.shape(x), dtype=object)) for var, x in known.items())

    def read(var):
        return np.zeros(var.aval.shape, dtype=object) if isinstance(var, core.Literal) else env[var]

    def constant(var):
        return var.val if isinstance(var, core.Literal) else known.get(var)

    for eq in graph.eqns:
        op, p = eq.primitive.name, eq.params
        args = [read(var) for var in eq.invars]
        a = args[0] if args else None
        if op in ('jit', 'pjit') and 'jaxpr' in p:
            nested = p['jaxpr']
            results = _propagate(nested.jaxpr, nested.consts, args)
        elif op == 'split':
            results = np.split(a, np.cumsum(p['sizes'])[:-1], axis=p['axis'])
        elif op == 'unstack':
            results = np.moveaxis(a, p['axis'], 0)
        else:
            if op in _ELEMENTWISE:
                value = 0
                for arg in args:
                    value = np.bitwise_or(value, arg)
                if op == 'integer_pow' and p['y'] == 0:
                    value = np.zeros(a.shape, dtype=object)
                if op == 'sub' and eq.invars[0] is eq.invars[1]:
                    value = np.zeros(a.shape, dtype=object)
                if op == 'mul':
                    for var in eq.invars:
                        factor = constant(var)
                        if factor is not None:
                            value = np.where(np.asarray(factor) == 0, 0, value)
            elif op == 'slice':
                steps = p['strides'] or (1,) * a.ndim
                value = a[tuple(map(slice, p['start_indices'], p['limit_indices'], steps))]
            elif op == 'concatenate':
                value = np.concatenate(args, axis=p['dimension'])
            elif op == 'squeeze':
                value = np.squeeze(a, axis=p['dimensions'])
            elif op == 'transpose':
                value = np.transpose(a, p['permutation'])
            elif op == 'reshape':
                value = (a if p['dimensions'] is None else a.transpose(p['dimensions'])).reshape(p['new_sizes'])
            elif op == 'broadcast_in_dim':
                shape = [1] * len(p['shape'])
                for dim, size in zip(p['broadcast_dimensions'], a.shape):
                    shape[dim] = size
                value = np.broadcast_to(a.reshape(shape), p['shape'])
            elif op.startswith('reduce_') and op in ('reduce_sum', 'reduce_prod', 'reduce_max', 'reduce_min', 'reduce_or', 'reduce_and'):
                value = np.bitwise_or.reduce(a, axis=p['axes'], initial=0)
            elif (op == 'dot_general' and a.ndim == 2 and args[1].ndim == 1
                  and p['dimension_numbers'] == (((1,), (0,)), ((), ()))):
                terms = a | args[1]
                for var in eq.invars:
                    factor = constant(var)
                    if factor is not None:
                        terms = np.where(np.asarray(factor) == 0, 0, terms)
                value = np.bitwise_or.reduce(terms, axis=1, initial=0)
            else:
                possible = 0
                for arg in args:
                    possible |= _union(arg)
                results = [np.full(var.aval.shape, possible | int(bool(possible)), dtype=object)
                           for var in eq.outvars]
                env.update(zip(eq.outvars, results))
                continue
            results = [value]
        env.update((var, np.asarray(x, dtype=object)) for var, x in zip(eq.outvars, results))
    return [read(var) for var in graph.outvars]


def coordinate_inputs(fn, args, sizes):
    """For each result return input indices and whether only an upper bound is known."""
    count = sum(x.size for x in args)
    start, symbols = 0, []
    for x in args:
        symbols.append(np.array([1 << (i+1) for i in range(start, start+x.size)], dtype=object).reshape(x.shape))
        start += x.size
    try:
        ir = jax.make_jaxpr(fn)(*args)
        results = _propagate(ir.jaxpr, ir.consts, symbols)
    except (TypeError, ValueError, NotImplementedError, AttributeError):
        # A callable can be interpretable on its actual values but opaque to tracing.
        results = [np.full(size, (1 << (count+1))-1, dtype=object) for size in sizes]
    return [[([i for i in range(count) if int(mask) & (1 << (i+1))], bool(int(mask) & 1))
             for mask in result.flat] for result in results]
