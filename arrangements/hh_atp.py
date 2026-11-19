"""Hodgkin–Huxley membrane with finite ion and ATP stores, assembled in IMC.

Quantities are SI, with amounts and total currents for a one-square-centimetre
patch. Gate systems nest into channel systems; the channels, stores, leak
and ATPase fill the membrane wiring. live_membrane exposes the composed
library step and local observations to the browser exporter.
"""
import math

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.coalgebra import CotCoalg
from arrangements.functors import open_step
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

jax.config.update("jax_enable_x64", True)
F = 96485.33212
RT = 8.314462618 * 279.45
C = 1e-6
G_NA, G_K, G_L = 0.120, 0.036, 0.0003
E_L = -0.054387
CAPACITY = 1e-6 / F
TOTAL = 5e-9
ATP0 = 3e-9
H_REF = 50000 - RT * math.log(ATP0 / (TOTAL - ATP0))
NA_BATH, K_BATH = 440e-9, 10e-9  # bath concentrations times 1 microlitre
EMPTY = jnp.zeros(0)


def hh_rates(v):
    w = 1000 * v + 65  # millivolts above -65 mV, depolarization positive

    def trap(z):
        small = jnp.abs(z) < 1e-6
        safe = jnp.where(small, 1.0, z)
        return jnp.where(small, 1 + z / 2 + z**2 / 12, safe / -jnp.expm1(-safe))

    alpha = jnp.array([trap((w - 25) / 10), 0.07 * jnp.exp(-w / 20),
                       0.1 * trap((w - 10) / 10)])
    beta = jnp.array([4 * jnp.exp(-w / 18), jax.nn.sigmoid((w - 30) / 10),
                      0.125 * jnp.exp(-w / 80)])
    return alpha * 1000, beta * 1000  # inverse seconds


def soft_limit(z):
    # A smooth convex blend equal to z up to 1/2 and to tanh(z) past 0.9.
    def bump(x):
        return jnp.where(x > 0, jnp.exp(-1 / jnp.where(x > 0, x, 1)), 0.)
    a, b = bump(jnp.abs(z) - 0.5), bump(0.9 - jnp.abs(z))
    w = a / (a + b)
    return (1 - w) * z + w * jnp.tanh(z)


def store(name, initial, read, radius=None, clamped=False):
    def update(s, xi, dt):
        if clamped:
            return s
        delta = dt * xi
        if radius is None:
            return s + delta
        b = radius(s)
        return s + b * soft_limit(delta / b)

    param = CotCoalg(1, jnp.array([initial]), read,
                     lambda s, xi: update(s, xi, 1.), name, timed_step=update)
    return Arrangement(param, 0, 0, 0, 1, lambda q, m: q,
                       lambda q, m, n: EMPTY, zero_form, name)


def process(name, dimension, field):
    return Arrangement(trivial_parameter(), 0, 0, dimension, 0,
                       lambda q, m: EMPTY, lambda q, m, n: EMPTY,
                       lambda q, m, n: (EMPTY, EMPTY, field(n)), name)


def wiring(label, inner_inputs, inner_outputs, outer_inputs, outer_outputs, route, publish):
    """A stateless affine lens. Its transpose carries loads back to its parts."""
    return Arrangement(trivial_parameter(), inner_inputs, inner_outputs, outer_inputs, outer_outputs,
                       lambda q, m: publish(m), lambda q, m, n: route(m, n), zero_form, label)


def fill(parts, router):
    return compose_seq(tensor_arrangements(parts), router)


def capacitor():
    return store("capacitor", C * -0.065, lambda charge: charge / C)


def ion_pool(name, bath, reversal, clamp=False):
    return store(name, bath * math.exp(-F * reversal / RT), lambda s: RT * jnp.log(s / bath),
                 lambda s: s, clamp)


def atp_pool(clamp=False):
    return store("ATP", ATP0, lambda a: H_REF + RT * jnp.log(a / (TOTAL - a)),
                 lambda a: a * (1 - a / TOTAL), clamp)


def gate(name):
    """An open voltage-driven gate. Its output is the permissive fraction x."""
    i = ("m", "h", "n").index(name)
    alpha, beta = hh_rates(-0.065)
    fraction = store(name, (alpha / (alpha + beta))[i], lambda s: s, lambda s: s * (1 - s))

    def inclination(inputs):
        v, x = inputs
        a, b = hh_rates(v)
        return jnp.array([0., a[i] * (1 - x) - b[i] * x])

    law = process(name + " kinetics", 2, inclination)
    router = wiring(name + " gate wiring", 2, 1, 1, 1,
                    lambda m, n: jnp.array([n[0], m[0]]), lambda m: m)
    return fill([fraction, law], router)


def sodium_channel():
    """Input (V, sodium chemical potential), output gate fractions (m, h)."""
    pore = process("Na pore", 3, lambda w: jnp.array([-G_NA * w[1]**3 * w[2] * w[0] / F**2, 0, 0]))
    router = wiring("sodium channel wiring", 5, 2, 2, 2,
                    lambda m, n: jnp.array([n[0], n[0], F*n[0]+n[1], m[0], m[1]]),
                    lambda m: m)
    return fill([gate("m"), gate("h"), pore], router)


def potassium_channel():
    """Input (V, potassium chemical potential), output gate fraction n."""
    pore = process("K pore", 2, lambda w: jnp.array([-G_K * w[1]**4 * w[0] / F**2, 0]))
    router = wiring("potassium channel wiring", 3, 1, 2, 1,
                    lambda m, n: jnp.array([n[0], F*n[0]+n[1], m[0]]), lambda m: m)
    return fill([gate("n"), pore], router)


def sodium_potassium_pump(capacity=CAPACITY):
    """Input (V, sodium and potassium chemical potentials, H), four load components."""
    law = process("ATPase", 1, lambda w: -capacity * jnp.tanh(w / (2 * RT)))
    def affinity(m, n):
        v, na, k, hydrolysis = n
        return jnp.array([hydrolysis + 3*(F*v + na) - 2*(F*v + k)])
    return fill([law], wiring("pump affinity", 1, 0, 4, 0, affinity, lambda m: EMPTY))


def leak():
    law = process("leak", 1, lambda w: -G_L * w)
    return fill([law], wiring("leak drop", 1, 0, 1, 0, lambda m, n: n - E_L, lambda m: EMPTY))


def membrane_wiring():
    """Open slots: four stores, sodium channel, potassium channel, leak, and pump."""
    return wiring("membrane wiring", 9, 7, 0, 2,
                  lambda m, n: jnp.stack([m[i] for i in (0, 1, 0, 2, 0, 0, 1, 2, 3)]),
                  lambda m: jnp.stack([m[0], m[3]]))


def membrane(capacity=CAPACITY, clamp=False):
    stores = [capacitor(), ion_pool("sodium", NA_BATH, .050, clamp),
              ion_pool("potassium", K_BATH, -.077, clamp), atp_pool(clamp)]
    return fill([*stores, sodium_channel(), potassium_channel(), leak(),
                 sodium_potassium_pump(capacity)], membrane_wiring())


def amounts(state):
    return np.array([float(a[0]) for a in jax.tree_util.tree_leaves(state) if a.size])


PROCESS_ORDER = ("Na pore", "K pore", "leak", "ATPase", "m kinetics", "h kinetics", "n kinetics")
STEP_SECONDS = 1e-5
OUTSIDE = jnp.array([10e-6, 0.0])


def process_observations(system, q):
    """Read the primitive process ports through this particular tree of tensors and affine wirings."""
    readings = {}

    def visit(part, config, inputs):
        if not part.construction:
            if part.in_dim_N:
                readings[part.label] = inputs, part.omega(config, EMPTY, inputs)[2]
            return
        kind, left, right = part.construction
        if kind == "tensor":
            visit(left, config[:left.P.Q_dim], inputs[:left.in_dim_N])
            visit(right, config[left.P.Q_dim:], inputs[left.in_dim_N:])
        elif kind == "compose" and right.P.Q_dim == 0:
            outputs = left.out_f(config, EMPTY)
            visit(left, config, right.in_f(EMPTY, outputs, inputs))
        else:
            raise ValueError("Expected a tensor or a stateless enclosing wiring")
    visit(system, q, EMPTY)
    return tuple(jnp.concatenate([readings[name][side] for name in PROCESS_ORDER]) for side in (0, 1))


def live_membrane():
    """Flat-state callables for browser export and comparison, backed by ordinary library dynamics."""
    from jax.flatten_util import ravel_pytree

    system = membrane()
    initial, unpack = ravel_pytree(system.P.init)
    library_step = open_step(system, EMPTY, OUTSIDE, duration=STEP_SECONDS)
    def step(state):
        return ravel_pytree(library_step(unpack(state)))[0]

    def observe(state):
        q = system.P.position(unpack(state))
        values, loads = process_observations(system, q)
        own, _, _ = system.omega(q, EMPTY, EMPTY)
        total = own + jax.vjp(lambda config: system.out_f(config, EMPTY), q)[1](OUTSIDE)[0]
        return q, values, loads, total

    q0 = system.P.position(system.P.init)
    jacobian = jax.jacrev(lambda q: process_observations(system, q)[0])(q0)
    return initial, step, observe, jacobian


def record_membrane(steps=1200, stride=5):
    """Sample an ordinary run for a figure or reference check. The dynamics have no end time."""
    state, step, observe, jacobian = live_membrane()
    step, observe = jax.jit(step), jax.jit(observe)
    frames = []
    for i in range(steps + 1):
        if i % stride == 0 or i == steps:
            q, values, loads, total = observe(state)
            frames.append(dict(ms=i*STEP_SECONDS*1000, state=state.tolist(), q=q.tolist(),
                               values=values.tolist(), loads=loads.tolist(), total=total.tolist()))
        if i < steps:
            state = step(state)
    return dict(frames=frames, jacobian=jacobian.tolist(), outside=OUTSIDE.tolist())
