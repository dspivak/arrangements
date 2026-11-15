"""Walled arrangements on the output side (book Ch 3 ``def.arr_vocabulary`` (x), Ch 4 ``def.walled``,
``prop.walled`` and ``ex.follower``; book plan P103 (a), Claude Opus 5.5, e1aedebb, 2026-11-14; walled names since
P103 (c), e06f45e7, 2026-11-15).

The follower ``follow : 1 -> <1 | R x R>`` has the response-integrator on ``Q = R^2`` with constant response
``sharp(xi_1, xi_2) = (xi_1, xi_1)``: a load on the leader ``q_1`` moves both coordinates, a load on the follower
``q_2`` moves neither. It reads out ``(q_1, q_2)`` at its two output ports ``a`` and ``b``. So it is unilateral at
``b`` and not at ``a``: a load told at ``b`` leaves its step unchanged, one told at ``a`` does not.

Beside a rod segment (response 1), into a static arrangement ``psi`` that transfers out the value at ``b``, the
composite is again unilateral at its one port (routing (ii) holds); if ``psi`` transfers out the rod segment's
temperature instead (routing fails), the composite is not. The identity on ``<1 | R>`` filled with the rod segment
is the rod segment, which is not unilateral, though the identity alone is (its configuration space is ``R^0``).

Every check is a step of an arrangement through the dynamics functor: ``open_step`` (functors.py) compiles one step of
``Phi(arr)`` for a system told a fixed force at its outputs, checked against ``Phi`` itself; composites are built with
``compose_seq`` and ``parallel_arrangements``. A follower is a mathematical idealization: its response is not
symmetric, and with a symmetric response a port is unilateral only if the configuration's motion never changes what
it reads out.
"""

import jax
import jax.numpy as jnp
import pytest

from arrangements.arrangement import Arrangement, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import open_step
from arrangements.rvect import constant
from arrangements.wiring import compose_seq, parallel_arrangements, trivial_parameter

Z = jnp.zeros(0)
TAU = 0.5


def system(P, out_dim, out_f, push, label):
    """A system ``1 -> <1 | R^out_dim>`` with integrator ``P`` and a constant inclination ``push`` on its configuration."""
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=out_dim,
        out_f=out_f, in_f=lambda q, m, n: Z,
        omega=lambda q, m, n: (jnp.asarray(push, dtype=float), jnp.zeros_like(m), jnp.zeros_like(n)),
        label=label,
    )


def follow():
    P = configuration(constant([[1.0, 0.0], [1.0, 0.0]]), init=[0.3, -0.2])
    return system(P, 2, lambda q, m: q, [0.7, 0.0], "follow")


def rod():
    P = configuration(constant([[1.0]]), init=[1.5])
    return system(P, 1, lambda q, m: q, [-0.4], "rod")


def static(out_dim_M, pick, label):
    """A static arrangement ``<1 | R^out_dim_M> -> <1 | R>`` transferring out the value at index ``pick``."""
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=out_dim_M, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: m[pick:pick + 1], in_f=lambda q, m, n: Z, omega=zero_form, label=label,
    )


def stepped(arr, xi_out):
    return open_step(arr, Z, xi_out, duration=TAU)(arr.P.init)


def flat(s):
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(s)])


@pytest.mark.parametrize("eta", [1.0, -2.5])
def test_follower_unilateral_at_b_not_at_a(eta):
    f = follow()
    base = flat(stepped(f, [0.0, 0.0]))
    at_b = flat(stepped(f, [0.0, eta]))
    at_a = flat(stepped(f, [eta, 0.0]))
    assert jnp.allclose(at_b, base)                                  # unilateral at b
    assert jnp.allclose(at_a - base, TAU * jnp.array([eta, eta]))    # a load at a moves leader and follower


@pytest.mark.parametrize("eta", [1.0, -2.5])
def test_routing_from_walled_port_keeps_unilateral(eta):
    """psi reads b: a two-input composite (follow beside rod), unilateral at its one port."""
    comp = compose_seq(parallel_arrangements(follow(), rod()), static(3, 1, "psi_b"))
    assert jnp.allclose(flat(stepped(comp, [eta])), flat(stepped(comp, [0.0])))


@pytest.mark.parametrize("eta", [1.0, -2.5])
def test_routing_from_unwalled_port_fails(eta):
    """psi reads the rod segment's temperature: static, so (i) holds, but (ii) fails and so does the composite."""
    comp = compose_seq(parallel_arrangements(follow(), rod()), static(3, 2, "psi_theta"))
    diff = flat(stepped(comp, [eta])) - flat(stepped(comp, [0.0]))
    assert jnp.allclose(diff, TAU * jnp.array([0.0, 0.0, eta]))


@pytest.mark.parametrize("eta", [1.0, -2.5])
def test_identity_filled_with_rod_is_the_rod(eta):
    """The identity on <1|R> is unilateral at its port (configuration R^0) but not walled-to-walled from an unwalled
    port; filled with the rod segment it steps as the rod segment, which a load at its output moves."""
    r = rod()
    comp = compose_seq(r, static(1, 0, "id"))
    for xi in ([0.0], [eta]):
        assert jnp.allclose(flat(stepped(comp, xi)), flat(stepped(r, xi)))
    assert not jnp.allclose(flat(stepped(comp, [eta])), flat(stepped(comp, [0.0])))


@pytest.mark.parametrize("eta", [1.0, -2.5])
def test_sensor_is_load_free_and_unilateral(eta):
    """Ch 4 sec.predictor_active's sensor (predictor.sensor) is the follower with its leader pulled toward g(x):
    a load on its readout leaves its step unchanged, and its inclination has no input component."""
    from arrangements.predictor import sensor
    s = sensor(lambda x: 2.0 * x, 1.0, 1, 1, y0=[0.4])
    x = jnp.array([0.9])
    base = open_step(s, x, [0.0], duration=TAU)(s.P.init)
    pushed = open_step(s, x, [eta], duration=TAU)(s.P.init)
    assert jnp.allclose(flat(pushed), flat(base))                                     # unilateral
    assert jnp.allclose(s.omega(s.P.init, Z, x)[2], 0.0)                              # load-free
    assert jnp.allclose(flat(base)[0], 0.4 + TAU * (2.0 * 0.9 - 0.4))                 # the report moves toward g(x)


def test_identity_on_walled_interface_is_a_unit():
    """The identity on <1 | R x R>, walled at b, composed after the follower steps as the follower, at every load."""
    f = follow()
    ident = Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=2,
        out_f=lambda q, m: m, in_f=lambda q, m, n: Z, omega=zero_form, label="id",
    )
    comp = compose_seq(f, ident)
    for xi in ([0.0, 0.0], [0.0, 1.3], [0.8, 0.0]):
        assert jnp.allclose(flat(stepped(comp, xi)), flat(stepped(f, xi)))
