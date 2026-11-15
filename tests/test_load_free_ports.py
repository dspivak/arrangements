"""Walled arrangements on the input side (book Ch 3 ``def.arr_vocabulary`` (ix), Ch 4 ``def.walled`` and
``prop.walled``; book plan P103 (c), Claude Opus 5.5, e06f45e7, 2026-11-15).

An arrangement is load-free at an input port when its inclination field has component ``0`` there. The couple of
Ch 3 ``ex.copying_learner`` (``copying.couple``) receives ``(x, kappa, y)`` and pushes back ``0`` at ``x`` and at
``kappa`` and ``kappa (x - y)`` at ``y``: load-free at ``x`` and ``kappa``, not at ``y``.

The copier (``copying.copy_pair_out``) is its wiring ``copier_wire`` with its boxes filled by two oscillators and the
couple. The wiring is static (inclination field ``0``) and sends its external input ``kappa`` only to the couple's
``kappa``, a walled port: the routing condition (ii) holds, so by ``prop.walled`` the copier is load-free at
``kappa``. A wiring that sends ``kappa`` to the couple's ``y`` instead (an unwalled port) still has inclination field
``0``, so (i) holds, but (ii) fails, and the composite pushes back at ``kappa``. And the identity on ``<3 | 0>``,
load-free at every port, filled with the couple is the couple, which is not load-free at ``y``.

Every check reads the inclination field of an arrangement built with ``compose_seq`` and ``parallel_arrangements``,
the composite's field being the book's sum of pullbacks (Ch 3 ``eqn.stateless_composite_effort``), at configurations
drawn at random and at the zero configuration; on ``R^n`` only, as the library is.
"""

import jax
import jax.numpy as jnp

from arrangements.arrangement import Arrangement, zero_form
from arrangements.copying import copier_wire, copy_pair_out, couple, osc
from arrangements.wiring import compose_seq, parallel_arrangements, trivial_parameter

Z = jnp.zeros(0)


def pushback(arr, q, n):
    """The component of ``arr``'s inclination field at its external input, at ``(q, (), n)`` (a system)."""
    return arr.omega(q, Z, n)[2]


def configurations(arr, k=4, seed=0):
    keys = jax.random.split(jax.random.PRNGKey(seed), k)
    return [jnp.zeros(arr.P.Q_dim)] + [jax.random.normal(key, (arr.P.Q_dim,)) for key in keys]


INPUTS = [jnp.array(v) for v in ([0.0], [1.0], [-0.7], [2.3])]


def test_couple_load_free_at_x_and_kappa_not_y():
    c = couple()
    for n in ([0.3, 1.0, -0.2], [1.5, 0.0, 0.4], [-1.0, 2.0, 1.0]):
        p = pushback(c, jnp.zeros(0), jnp.array(n))
        assert jnp.allclose(p[:2], 0.0)
        assert jnp.allclose(p[2], n[1] * (n[0] - n[2]))
    assert not jnp.allclose(pushback(c, jnp.zeros(0), jnp.array([1.0, 1.0, 0.0]))[2], 0.0)


def test_copier_load_free_at_kappa():
    """The composite copier: (i) and (ii) hold for its wiring, and its pushback at kappa is 0 everywhere tried."""
    cp = copy_pair_out(0.1, 0.05, [0.2, 1.0], [-0.4, 1.3])
    assert cp.in_dim_N == 1
    for q in configurations(cp):
        for n in INPUTS:
            assert jnp.allclose(pushback(cp, q, n), 0.0)


def misrouted_wire():
    """Static like ``copier_wire``, but sending the external kappa to the couple's y (unwalled) and y to its kappa."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=3, out_dim_M=2, in_dim_N=1, out_dim_N=2,
        out_f=lambda q, m: m, in_f=lambda q, m, n_in: jnp.array([m[0], m[1], n_in[0]]), omega=zero_form,
        label="misrouted_wire",
    )


def test_misrouted_wiring_pushes_back_at_kappa():
    """(i) holds (inclination field 0) but (ii) fails: the composite pushes back y's load at kappa."""
    boxes = parallel_arrangements(
        parallel_arrangements(osc(0.1, 0.05, [0.2, 1.0]), osc(0.1, 0.05, [-0.4, 1.3])), couple())
    for wire in (copier_wire(), misrouted_wire()):
        q, m = jnp.zeros(wire.P.Q_dim), jnp.array([0.5, -0.3])
        assert jnp.allclose(wire.omega(q, m, jnp.array([1.0]))[2], 0.0)
    bad = compose_seq(boxes, misrouted_wire())
    nonzero = [not jnp.allclose(pushback(bad, q, n), 0.0) for q in configurations(bad) for n in INPUTS]
    assert any(nonzero)


def test_identity_filled_with_couple_is_the_couple():
    """The identity on <3 | 0> has inclination field 0, so it is load-free at all three ports; filled with the couple
    it pushes back as the couple does, not load-free at y."""
    ident = Arrangement(
        P=trivial_parameter(), in_dim_M=3, out_dim_M=0, in_dim_N=3, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n_in: n_in, omega=zero_form, label="id3",
    )
    n = jnp.array([1.2, 0.5, -0.3])
    assert jnp.allclose(ident.omega(jnp.zeros(0), Z, n)[2], 0.0)
    comp = compose_seq(couple(), ident)
    q = jnp.zeros(comp.P.Q_dim)
    assert jnp.allclose(pushback(comp, q, n), pushback(couple(), jnp.zeros(0), n))
    assert not jnp.allclose(pushback(comp, q, n)[2], 0.0)


def test_identity_removes_walls_but_does_not_add():
    """The identity on <2 | 0> as (N, {a}) -> (N, W'): its in-transfer sends the external a to the internal a. Filled
    with a box load-free at a only, the composite is load-free at a and not at b: W' = {a} is allowed, {a, b} is not."""
    def omega(q, m, n_in):
        return jnp.zeros_like(q), jnp.zeros_like(m), jnp.array([0.0, n_in[0] - n_in[1]])
    box = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n_in: Z, omega=omega, label="box_a",
    )
    ident = Arrangement(
        P=trivial_parameter(), in_dim_M=2, out_dim_M=0, in_dim_N=2, out_dim_N=0,
        out_f=lambda q, m: Z, in_f=lambda q, m, n_in: n_in, omega=zero_form, label="id2",
    )
    comp = compose_seq(box, ident)
    p = pushback(comp, jnp.zeros(comp.P.Q_dim), jnp.array([1.0, 0.0]))
    assert jnp.allclose(p[0], 0.0)
    assert not jnp.allclose(p[1], 0.0)
