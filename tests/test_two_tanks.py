"""Two tanks joined by a pipe, a waterfall, or a pump (book Chapter 2, ``sec.two_tanks``: ``sec.one_tank``,
``sec.pipe``, ``sec.waterfall``, ``sec.pump``; linked from the footnote at the end of ``sec.pump``).

For a reader of the book. A tank (``sec.one_tank``) is a part whose state is the volume of water in it,
``v`` here and ``s`` in the book. It reads out the pressure at its base, which with water's weight per unit
volume taken as ``1``, as the book does, is the height of the water: ``e`` here, ``q = s/A`` in the book.
The flow that comes back at its port, ``F`` here and the force ``xi`` in the book, is added to its volume,
``v' = v + F``, whatever the tank's shape, because the readout's transpose divides by the cross-section
and the response multiplies by it (``eqn.tank_step``, ``exc.tank_shape``). ``cylinder(A, v0)`` is the
straight-walled tank of cross-section ``A``; ``cone(a, v0)`` is a tank whose cross-section ``a h`` grows
with the height. The part joining two tanks is a *coupler*: it has no state, receives both pressures, and
returns a flow to each.

* ``pipe(kappa)`` returns ``(-I, +I)`` with ``I = kappa (e_1 - e_2)``, minus the differential of the cost
  function ``(kappa/2)(e_1 - e_2)^2``, ``kappa`` its conductance (``eqn.pipe_step``); ``pipe(kappa, z)``
  raises the second tank's base by a height ``z``.
* ``waterfall(kappa)`` returns ``(-I, +I)`` with ``I = kappa e_1``, the spout's flow, which depends on
  the upper tank's pressure only (``eqn.waterfall_step``): the lower tank never reaches the upper.
* ``pump(I0)`` returns ``(+I0, -I0)``: water from the second tank into the first at a fixed rate,
  whatever the levels (``eqn.pump_step``).

``closed(tanks, coupler)`` wires them as the book's picture does and ``Phi`` runs the whole; ``run``
takes steps and ``volumes`` reads the state. The book's hand calculation for the pipe (``sec.pipe``:
unit areas, ``kappa = 1/4``, from ``(4, 0)``),
run from the library's folder with it on the path, as the README's *Run it* says
(``PYTHONPATH=. .venv/bin/python example.py``):

    from arrangements.functors import Phi
    from tests.test_two_tanks import cylinder, pipe, closed, run, volumes
    O = Phi(closed([cylinder(1.0, 4.0), cylinder(1.0, 0.0)], pipe(0.25)))
    print([volumes(s) for s in run(O, O.state, 3)])
    # [(4.0, 0.0), (3.0, 1.0), (2.5, 1.5), (2.25, 1.75)]

What the tests check. ``test_book_steps_by_hand``: the hand calculations of ``sec.pipe``,
``sec.waterfall`` (from both starts) and ``sec.pump``. Then, between a cylinder and a cone: each
coupler's step is the book's; the pipe evens out the levels; the two sensitivities of ``sec.waterfall``,
``dF_1/de_2`` and ``dF_2/de_1``, agree through the pipe and differ through the waterfall, which is
``prop.closed`` at work, the pipe's pushback minus the differential of a cost function and the
waterfall's not; the total water is conserved whenever the two flows are equal and opposite, and not
through a leak; and wiring the tanks to the coupler before running, or running the tanks and then the
coupler, gives the same steps.

For agents: the antisymmetric part of the returned covector's Jacobian is ``d omega``, the reciprocity
lemma of ``background/note-retroactivity.tex`` §3; forcings carry the book's sign (STATUS.md §1). The
file's earlier head and what changed on 2026-10-30 are in ``notes/P31-B-CH2-HISTORY-2026-10-30.md``.
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

C = 1.0           # water's weight per unit volume, 1 as in the book (sec.one_tank): pressure is height


# ---------------------------------------------------------------------------
# The components.
# ---------------------------------------------------------------------------


def tank(area_of_height, height_of_volume, v0, label="tank"):
    """A tank of any shape (exc.tank_shape): cross-section ``A(h)`` at height ``h``, height ``h(v)`` at
    volume ``v``. Its state is ``v``, it reads out the pressure ``e = c h(v)``, its response multiplies
    by ``A(h(v))/c``, and it has no internal inclination (zero forcing)."""
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
    """The straight-walled tank of sec.one_tank, cross-section ``A``, holding ``v0`` at the start."""
    return tank(lambda h: A, lambda v: v / A, v0, label=f"cyl({A})")


def cone(a, v0):
    """A cone-shaped tank, ``A(h) = a h``, ``v = a h^2 / 2``: its response depends on its state."""
    return tank(lambda h: a * h, lambda v: jnp.sqrt(2.0 * v / a), v0, label=f"cone({a})")


def coupler(F, label):
    """The part joining two tanks: no state, two input ports receiving ``(e_1, e_2)``, and the pushback
    ``F(e)``, the pair of flows it returns (``exc.pipe_stateless``: all of its 1-form is its pushback)."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: jnp.zeros(0),
        in_f=lambda q, e, n_in: jnp.zeros(0),
        omega=lambda q, e, n_in: (jnp.zeros(0), F(e), jnp.zeros(0)),
        label=label,
    )


def pipe(kappa, z=0.0):
    """The pipe (sec.pipe): the cost function ``(kappa/2)((e_1 - e_2) + c z)^2``, so it returns
    ``(-I, +I)`` with ``I = kappa ((e_1 - e_2) + c z)``; ``z`` raises the second tank's base by a height
    ``z``, and the book's pipe has ``z = 0``. Its pushback is minus the differential of the cost."""
    V = lambda e: 0.5 * kappa * ((e[0] - e[1]) + C * z) ** 2
    omega = exact(lambda q, e, n: -V(e))
    return coupler(lambda e: omega(jnp.zeros(0), e, jnp.zeros(0))[1], "pipe")


def pipe_flow(kappa, z=0.0):
    """The pipe's flow ``I(e)`` of eqn.pipe_step, written out, for checking the step by hand."""
    return lambda e: kappa * ((e[0] - e[1]) + C * z)


def waterfall(kappa):
    """The waterfall (sec.waterfall): the spout's flow ``I = kappa e_1`` leaves the first tank and lands in
    the second, ``(-I, +I)``, whatever the second tank's pressure (eqn.waterfall_step)."""
    return coupler(lambda e: jnp.array([-kappa * e[0], kappa * e[0]]), "waterfall")


def pump(I0):
    """The pump (sec.pump): ``(+I0, -I0)``, water from the second tank into the first at the rate ``I0``,
    whatever the pressures (eqn.pump_step)."""
    return coupler(lambda e: jnp.array([I0, -I0]), "pump")


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
    """The two tanks wired to the coupler, closed: the picture of sec.pipe."""
    return compose_seq(tensor_arrangements(tanks), cpl)


def run(O, state, T):
    """``T`` steps of the dynamics ``O = Phi(closed(...))`` from ``state``; the ``T + 1`` states."""
    traj = [state]
    for _ in range(T):
        _, _, state = O.with_state(state).run_one(IN_POS, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        traj.append(state)
    return traj


def volumes(state):
    """The two volumes ``(v_1, v_2)`` held in a state."""
    (v1, v2), _ = state
    return float(v1[0]), float(v2[0])


# ---------------------------------------------------------------------------
# The book's hand calculations (sec.pipe, sec.waterfall, sec.pump).
# ---------------------------------------------------------------------------


def _by_hand(v0, cpl, T):
    O = Phi(closed([cylinder(1.0, v0[0]), cylinder(1.0, v0[1])], cpl))
    return [volumes(s) for s in run(O, O.state, T)]


def test_book_steps_by_hand():
    """Unit areas. Pipe, ``kappa = 1/4``, from ``(4, 0)``: ``(3, 1)``, ``(2.5, 1.5)``, ``(2.25, 1.75)``, the
    gap halved each step, toward ``(2, 2)``. Waterfall, ``kappa = 1/2``, from ``(4, 0)``: ``(2, 2)``,
    ``(1, 3)``, ``(1/2, 7/2)``; from ``(4, 8)``: ``(2, 10)``, ``(1, 11)``, ``(1/2, 23/2)``, the first tank's
    volumes the same, since nothing about the second tank reaches it. Pump, ``I0 = 1``, from ``(2, 6)``:
    ``(3, 5)``, ``(4, 4)``, ``(5, 3)``, the flow ``1`` at every step,
    on past the point where a pipe would rest."""
    np.testing.assert_allclose(_by_hand((4.0, 0.0), pipe(0.25), 3),
                               [(4, 0), (3, 1), (2.5, 1.5), (2.25, 1.75)], atol=1e-12)
    far = _by_hand((4.0, 0.0), pipe(0.25), 40)[-1]
    np.testing.assert_allclose(far, (2, 2), atol=1e-9)
    np.testing.assert_allclose(_by_hand((4.0, 0.0), waterfall(0.5), 3),
                               [(4, 0), (2, 2), (1, 3), (0.5, 3.5)], atol=1e-12)
    np.testing.assert_allclose(_by_hand((4.0, 8.0), waterfall(0.5), 3),
                               [(4, 8), (2, 10), (1, 11), (0.5, 11.5)], atol=1e-12)
    np.testing.assert_allclose(_by_hand((2.0, 6.0), pump(1.0), 3),
                               [(2, 6), (3, 5), (4, 4), (5, 3)], atol=1e-12)


# ---------------------------------------------------------------------------
# Reciprocity (notes/DESIGN.md §5).
# ---------------------------------------------------------------------------


def test_reciprocity_pipe_is_exact():
    """The pipe comes from a potential: the returned-covector Jacobian is symmetric, d omega = 0."""
    dw = d_omega(returned_covector(pipe(kappa=0.7, z=0.25)))
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
    """sec.waterfall: the sensitivities are ``dF_1/de_2 = 0`` and ``dF_2/de_1 = kappa``, so
    ``d omega = kappa de_1 ^ de_2 != 0`` and no cost function has the waterfall's pushback
    (exc.waterfall_cost)."""
    kappa = 0.5
    dw = d_omega(returned_covector(waterfall(kappa)))
    rng = np.random.default_rng(3)
    for _ in range(5):
        e = jnp.asarray(rng.standard_normal(2))
        expected = kappa * jnp.array([[0.0, 1.0], [-1.0, 0.0]])
        np.testing.assert_allclose(dw(e), expected, atol=1e-12)


def test_retroactivity_pipe_yes_waterfall_no():
    """Port 1's load depends on ``e_2`` through the pipe, and not through the waterfall."""
    F_pipe, F_fall = returned_covector(pipe(0.7)), returned_covector(waterfall(0.5))
    e, e2 = jnp.array([1.0, 0.3]), jnp.array([1.0, -2.0])
    assert not np.allclose(F_pipe(e)[0], F_pipe(e2)[0])
    np.testing.assert_allclose(F_fall(e)[0], F_fall(e2)[0], atol=1e-12)


# ---------------------------------------------------------------------------
# Two tanks, three couplers (notes/DESIGN.md §5).
# ---------------------------------------------------------------------------


def _tanks():
    return [cylinder(A=2.0, v0=3.0), cone(a=1.5, v0=1.0)]


def test_pipe_step_matches_the_notes_two_lines():
    """``v_1' = v_1 - I``, ``v_2' = v_2 + I`` with ``I = kappa((e_1 - e_2) + c z)`` at the
    current pressures, for any tank shapes (the loading is automatic)."""
    kappa, z = 0.7, 0.25
    cpl, I = pipe(kappa, z), pipe_flow(kappa, z)
    tanks = _tanks()
    O = Phi(closed(tanks, cpl))
    s0, s1 = run(O, O.state, 1)
    v1, v2 = volumes(s0)
    e = jnp.array([tanks[0].out_f(jnp.array([v1]), jnp.zeros(0))[0],
                   tanks[1].out_f(jnp.array([v2]), jnp.zeros(0))[0]])
    I0 = float(I(e))
    np.testing.assert_allclose(volumes(s1), (v1 - I0, v2 + I0), rtol=1e-12)


def test_waterfall_and_pump_steps():
    """Waterfall: ``v_1' = v_1 - kappa e_1``, ``v_2' = v_2 + kappa e_1`` (eqn.waterfall_step); pump:
    ``v_1' = v_1 + I0``, ``v_2' = v_2 - I0`` (eqn.pump_step), between the cylinder and the cone."""
    tanks = _tanks()
    v1, v2 = 3.0, 1.0
    e1 = float(tanks[0].out_f(jnp.array([v1]), jnp.zeros(0))[0])

    O = Phi(closed(tanks, waterfall(0.5)))
    _, s1 = run(O, O.state, 1)
    np.testing.assert_allclose(volumes(s1), (v1 - 0.5 * e1, v2 + 0.5 * e1), rtol=1e-12)

    O = Phi(closed(tanks, pump(0.2)))
    _, s1 = run(O, O.state, 1)
    np.testing.assert_allclose(volumes(s1), (v1 + 0.2, v2 - 0.2), rtol=1e-12)


def test_total_water_conserved_when_covectors_are_opposite():
    """Total water is conserved by the pipe (its two covectors are equal and opposite);
    the waterfall as written also lands its stream in tank 2, so it conserves too --
    what it loses is reciprocity, not water. A leak (``F = (-phi, 0)``) does not."""
    for c in (pipe(0.02, 0.05), waterfall(0.1)):   # small steps: the cone's sqrt needs v > 0
        O = Phi(closed(_tanks(), c))
        totals = [sum(volumes(s)) for s in run(O, O.state, 20)]
        np.testing.assert_allclose(totals, totals[0], rtol=1e-12)

    leak = coupler(lambda e: jnp.array([-0.1 * e[0], 0.0]), "leak")
    O = Phi(closed(_tanks(), leak))
    totals = [sum(volumes(s)) for s in run(O, O.state, 20)]
    assert totals[-1] < totals[0] - 1e-3


def test_pipe_equalizes_levels():
    """For real: with ``z = 0`` the pipe drives the two pressures together."""
    tanks = _tanks()
    O = Phi(closed(tanks, pipe(0.05)))
    traj = run(O, O.state, 400)
    def gap(s):
        v1, v2 = volumes(s)
        return abs(float(tanks[0].out_f(jnp.array([v1]), jnp.zeros(0))[0]
                         - tanks[1].out_f(jnp.array([v2]), jnp.zeros(0))[0]))
    assert gap(traj[-1]) < 1e-3 * gap(traj[0])


# ---------------------------------------------------------------------------
# Composition agrees with dynamics for a non-exact coupler (notes/DESIGN.md §5).
# ---------------------------------------------------------------------------


def test_composition_agrees_with_dynamics_for_the_waterfall():
    """``Phi(compose_seq(tensor(tanks), waterfall))`` and
    ``Phi(tensor(tanks)).then(Phi(waterfall))`` agree step for step."""
    tanks = _tanks()
    first = Phi(closed(tanks, waterfall(0.5)))
    second = Phi(tensor_arrangements(tanks)).then(Phi(waterfall(0.5)))
    t1 = run(first, first.state, 10)
    t2 = run(second, second.state, 10)
    for a, b in zip(t1, t2):
        np.testing.assert_allclose(volumes(a), volumes(b), rtol=1e-12)
