"""Two tanks, three couplers (book sec.two_tanks; ``background/note-retroactivity.tex`` §3).

A tank is an arrangement ``<R^0|R^0> -> <R^0|R>`` with parameter the configuration
coalgebra on its volume ``v``, response ``sharp_v = A(h(v))/c`` (its capacitance, so
its shape is encoded once), output the base pressure ``e = c h(v)``, and zero
forcing. A coupler is a *stateless* arrangement ``<R^0|R^2> -> <R^0|R^0>`` whose
ports receive the two pressures and whose forcing on ``M^+ = R^2`` is the
returned covector ``F(e)``; the closed system is
``compose_seq(tensor(tank_1, tank_2), coupler)``. Because ``out_f = c h(v)`` pulls
a port covector back by ``c h'(v) = c/A(h)`` and the response multiplies by
``A(h)/c``, the displacement is ``v' = v + F(e)`` whatever the tank's shape.

The note writes its one-forms in CDLM's sign (``dot q = -sharp omega``); the
book's forcing is the note's covector with a minus sign (AGENTDOCPLAN.md §4.9),
and the coalgebra adds:

* pipe:      ``omega = -dV``, ``V = (kappa/2)((e_1 - e_2) + c z)^2``, so
             ``F = (-I, +I)``, ``I = kappa((e_1 - e_2) + c z)``; exact.
* waterfall: ``omega = -phi(e_1)(de_1 - de_2)``, ``F = (-phi(e_1), +phi(e_1))``;
             ``d omega = phi'(e_1) de_1 ^ de_2 != 0``; tank 1 never feels tank 2.
* pump:      ``omega = p (de_1 - de_2)``, ``F = (+p, -p)``; exact (a linear potential).

Reciprocity lemma (note §3): for a stateless coupler returning ``F_i(e)`` at port
``i``, ``omega = dV`` implies ``dF_i/de_j = (dF_j/de_i)^T``; in general ``d omega``
is the antisymmetric part of ``DF``. Tested here on the pipe and the waterfall.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import ReactiveVectorSpace
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.chain import IN_POS

C = 9.81          # weight density
ALPHA = 0.3       # waterfall orifice coefficient


# ---------------------------------------------------------------------------
# The components.
# ---------------------------------------------------------------------------


def tank(area_of_height, height_of_volume, v0, label="tank"):
    """A tank with cross-section ``A(h)`` and height ``h(v)``: state ``v``, response
    ``sharp_v = A(h(v))/c``, output pressure ``e = c h(v)``, zero forcing."""
    Q = ReactiveVectorSpace(
        dim=1, sharp_fn=lambda v: jnp.array([[area_of_height(height_of_volume(v[0])) / C]]))
    return Arrangement(
        P=configuration(Q, init=jnp.array([v0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda v, m_out: C * jnp.array([height_of_volume(v[0])]),
        in_f=lambda v, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label=label,
    )


def cylinder(A, v0):
    return tank(lambda h: A, lambda v: v / A, v0, label=f"cyl({A})")


def cone(a, v0):
    """A conical tank, ``A(h) = a h``, ``v = a h^2 / 2``: a position-dependent response."""
    return tank(lambda h: a * h, lambda v: jnp.sqrt(2.0 * v / a), v0, label=f"cone({a})")


def coupler(F, label):
    """A stateless two-port coupler with returned covector ``F : R^2 -> (R^2)^*``."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: jnp.zeros(0),
        in_f=lambda q, e, n_in: jnp.zeros(0),
        omega=lambda q, e, n_in: (jnp.zeros(0), F(e), jnp.zeros(0)),
        label=label,
    )


def pipe(kappa, z):
    V = lambda e: 0.5 * kappa * ((e[0] - e[1]) + C * z) ** 2
    omega = exact(lambda q, e, n: -V(e))
    c = coupler(lambda e: omega(jnp.zeros(0), e, jnp.zeros(0))[1], "pipe")
    return c, lambda e: kappa * ((e[0] - e[1]) + C * z)


def phi(e1):
    return ALPHA * jnp.tanh(e1)  # the orifice law: smooth, phi' != 0


def waterfall():
    return coupler(lambda e: jnp.array([-phi(e[0]), phi(e[0])]), "waterfall")


def pump(p):
    return coupler(lambda e: jnp.array([p, -p]), "pump")


def returned_covector(cpl):
    """``F(e)``: the coupler's forcing restricted to its ports ``M^+ = R^2``."""
    return lambda e: cpl.omega(jnp.zeros(0), e, jnp.zeros(0))[1]


def d_omega(F):
    """The exterior derivative ``(d omega)_{ij} = d_i F_j - d_j F_i`` of ``omega = sum_j F_j de_j``,
    i.e. the antisymmetric part of the returned-covector Jacobian (note §3)."""
    def at(e):
        J = jax.jacobian(F)(e)   # J[j, i] = d F_j / d e_i
        return J.T - J
    return at


def closed(tanks, cpl):
    return compose_seq(tensor_arrangements(tanks), cpl)


def run(O, state, T):
    traj = [state]
    for _ in range(T):
        _, _, state = O.with_state(state).run_one(IN_POS, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        traj.append(state)
    return traj


def volumes(state):
    (v1, v2), _ = state
    return float(v1[0]), float(v2[0])


# ---------------------------------------------------------------------------
# Reciprocity (DESIGN.md §5).
# ---------------------------------------------------------------------------


def test_reciprocity_pipe_is_exact():
    """The pipe comes from a potential: the returned-covector Jacobian is symmetric, d omega = 0."""
    cpl, _ = pipe(kappa=0.7, z=0.25)
    dw = d_omega(returned_covector(cpl))
    rng = np.random.default_rng(1)
    for _ in range(5):
        e = jnp.asarray(rng.standard_normal(2))
        np.testing.assert_allclose(dw(e), jnp.zeros((2, 2)), atol=1e-12)


def test_reciprocity_of_every_exact_coupler():
    """The lemma's "if": ``omega = dV`` for any smooth ``V`` gives symmetric ``DF``."""
    V = lambda e: jnp.sin(e[0] * e[1]) + jnp.exp(-e[0] ** 2) * e[1] ** 3
    omega = exact(lambda q, e, n: V(e))
    dw = d_omega(lambda e: omega(jnp.zeros(0), e, jnp.zeros(0))[1])
    rng = np.random.default_rng(2)
    for _ in range(5):
        e = jnp.asarray(rng.standard_normal(2))
        np.testing.assert_allclose(dw(e), jnp.zeros((2, 2)), atol=1e-12)


def test_reciprocity_waterfall_is_not_exact():
    """``d omega = phi'(e_1) de_1 ^ de_2 != 0``: one-way influence forces non-exactness."""
    dw = d_omega(returned_covector(waterfall()))
    dphi = jax.grad(phi)
    rng = np.random.default_rng(3)
    for _ in range(5):
        e = jnp.asarray(rng.standard_normal(2))
        expected = dphi(e[0]) * jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(dw(e), expected, atol=1e-12)
        assert abs(float(dphi(e[0]))) > 1e-3   # genuinely non-zero at these points


def test_retroactivity_pipe_yes_waterfall_no():
    """Port 1's load depends on ``e_2`` through the pipe, and not through the waterfall."""
    cpl, _ = pipe(kappa=0.7, z=0.0)
    F_pipe, F_fall = returned_covector(cpl), returned_covector(waterfall())
    e, e2 = jnp.array([1.0, 0.3]), jnp.array([1.0, -2.0])
    assert not np.allclose(F_pipe(e)[0], F_pipe(e2)[0])
    np.testing.assert_allclose(F_fall(e)[0], F_fall(e2)[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Two tanks, three couplers (DESIGN.md §5).
# ---------------------------------------------------------------------------


def _tanks():
    return [cylinder(A=2.0, v0=3.0), cone(a=1.5, v0=1.0)]


def test_pipe_step_matches_the_notes_two_lines():
    """``v_1' = v_1 - I``, ``v_2' = v_2 + I`` with ``I = kappa((e_1 - e_2) + c z)`` at the
    current pressures, for any tank shapes (the loading is automatic)."""
    kappa, z = 0.7, 0.25
    cpl, I = pipe(kappa, z)
    tanks = _tanks()
    O = Phi(closed(tanks, cpl))
    s0, s1 = run(O, O.state, 1)
    v1, v2 = volumes(s0)
    e = jnp.array([tanks[0].out_f(jnp.array([v1]), jnp.zeros(0))[0],
                   tanks[1].out_f(jnp.array([v2]), jnp.zeros(0))[0]])
    I0 = float(I(e))
    np.testing.assert_allclose(volumes(s1), (v1 - I0, v2 + I0), rtol=1e-12)


def test_waterfall_and_pump_steps():
    """Waterfall: ``v_1' = v_1 - phi(e_1)``, ``v_2' = v_2 + phi(e_1)``; pump: ``v_1' = v_1 + p``, ``v_2' = v_2 - p``."""
    tanks = _tanks()
    v1, v2 = 3.0, 1.0
    e1 = float(tanks[0].out_f(jnp.array([v1]), jnp.zeros(0))[0])

    O = Phi(closed(tanks, waterfall()))
    _, s1 = run(O, O.state, 1)
    np.testing.assert_allclose(volumes(s1), (v1 - float(phi(e1)), v2 + float(phi(e1))), rtol=1e-12)

    O = Phi(closed(tanks, pump(0.2)))
    _, s1 = run(O, O.state, 1)
    np.testing.assert_allclose(volumes(s1), (v1 + 0.2, v2 - 0.2), rtol=1e-12)


def test_total_water_conserved_when_covectors_are_opposite():
    """Total water is conserved by the pipe (its two covectors are equal and opposite);
    the waterfall as written also lands its stream in tank 2, so it conserves too --
    what it loses is reciprocity, not water. A leak (``F = (-phi, 0)``) does not."""
    cpl, _ = pipe(kappa=0.02, z=0.05)  # small steps: the cone's sqrt needs v > 0
    for c in (cpl, waterfall()):
        O = Phi(closed(_tanks(), c))
        totals = [sum(volumes(s)) for s in run(O, O.state, 20)]
        np.testing.assert_allclose(totals, totals[0], rtol=1e-12)

    leak = coupler(lambda e: jnp.array([-phi(e[0]), 0.0]), "leak")
    O = Phi(closed(_tanks(), leak))
    totals = [sum(volumes(s)) for s in run(O, O.state, 20)]
    assert totals[-1] < totals[0] - 1e-3


def test_pipe_equalizes_levels():
    """For real: with ``z = 0`` the pipe drives the two pressures together."""
    cpl, _ = pipe(kappa=0.05, z=0.0)
    tanks = _tanks()
    O = Phi(closed(tanks, cpl))
    traj = run(O, O.state, 400)
    def gap(s):
        v1, v2 = volumes(s)
        return abs(float(tanks[0].out_f(jnp.array([v1]), jnp.zeros(0))[0]
                         - tanks[1].out_f(jnp.array([v2]), jnp.zeros(0))[0]))
    assert gap(traj[-1]) < 1e-3 * gap(traj[0])


# ---------------------------------------------------------------------------
# Composition agrees with dynamics for a non-exact coupler (DESIGN.md §5).
# ---------------------------------------------------------------------------


def test_composition_agrees_with_dynamics_for_the_waterfall():
    """``Phi(compose_seq(tensor(tanks), waterfall))`` and
    ``Phi(tensor(tanks)).then(Phi(waterfall))`` agree step for step."""
    tanks = _tanks()
    first = Phi(closed(tanks, waterfall()))
    second = Phi(tensor_arrangements(tanks)).then(Phi(waterfall()))
    t1 = run(first, first.state, 10)
    t2 = run(second, second.state, 10)
    for a, b in zip(t1, t2):
        np.testing.assert_allclose(volumes(a), volumes(b), rtol=1e-12)
