"""The duration input (book P9.5; Ch 3 ``def.duration_integrator``).

Durations are positive (David, 2026-10-10, "yes" to durations the positive numbers, the differential equation the
limit as they shrink); a law's value at ``tau = 0`` below is its limit there, the smooth extension the book calls
normalized, not a step of no time. An integrator is first told how long the coming step lasts, then shows its look-ahead for that duration,
then receives a covector there: ``u : S -> (y^{R>=0} <| cot(Q))(S)`` (David, 2026-10-10: "output a function
… from duration to look ahead"), and parts side by side share one duration. In the library a part's
``look_ahead(s, duration)`` and ``timed_step(s, xi, duration)`` are that law, ``position_at`` and
``step_over`` call them, and ``Phi``'s step, ``PCMorphism``'s composites, ``run_one`` and ``closed_step``
carry a duration to every box; ``duration=None`` is the unit step, the path every run took before durations
existed.

Predictions, written before the code (book `notes/runs/p95-duration-33faae02/RESULTS.md`):
1. every existing test passes unchanged;
2. one step of a composite of configuration parts held for ``tau`` moves the state ``tau`` times as far
   as the unit step, since the covector is read before the step;
3. as ``tau`` shrinks to 0 it does not move (its extension at 0 is the state);
4. Euler is first order: two half steps differ from one unit step by ``O(tau^2)``, and over a fixed
   elapsed time the gap shrinks about in half when the duration is halved;
5. a composite with a part that has no duration law, handed a duration, raises;
6. ``Phi`` of a composite and the composite of the ``Phi``s agree at a duration other than 1.

Added with the look-ahead (David's "yes," 2026-10-10), predictions before the code:
7. the phase part at duration 1 is its unit step, and at duration 0 does not move;
9. the damped part (friction a push held for the duration, David's "yes," 2026-10-10) is its unit step at
   duration 1, still at 0, and its momentum with no other push keeps ``1 - c tau`` of itself;
8. a part whose look-ahead is half the coming duration is the leapfrog step in one round: second order
   (the error falls by about 4 when the duration halves) where the full look-ahead falls by about 2.

The rod is Ch 3's grouped rod: three pieces of heat capacity 4, joined by conductance 1, insulated at
both ends (held ends with no spring), from ``(8, 4, 0)``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.coalgebra import CotCoalg, configuration, damped, gyro, phase, phase_on_cotangent
from arrangements.functors import Phi, closed_step
from arrangements.rvect import diagonal
from arrangements.wiring import held_chain

from tests.chain import IN_POS, HeldChain, harmonic_particle

NO_DIR = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def grouped_rod(parameter=configuration):
    """Ch 3's grouped rod as ``wiring.held_chain``: the first piece has no spring to the left end and the
    right end none to the last piece, so both ends are insulated; capacity 4 is ``sharp = 1/4``."""
    parts = [harmonic_particle(parameter, 4.0, 0.0)] + [harmonic_particle(parameter, 4.0, 1.0)] * 2
    return held_chain(parts, 0.0, parameter)


class Rod(HeldChain):
    """``HeldChain``'s state conversions for the grouped rod, stepped at a duration."""

    def __init__(self, parameter=configuration):
        self.K, self.two_stage = 3, False
        self.arr = grouped_rod(parameter)
        self.phase = False
        self._steps = {}

    def at(self, q, tau):
        if tau not in self._steps:
            self._steps[tau] = closed_step(self.arr, duration=tau)
        return self.flat(self._steps[tau](self.state(q)))


def test_unit_step_is_the_books():
    """The grouped rod's unit step, (8,4,0) to (7,4,1), with and without an explicit duration 1."""
    rod = Rod()
    q0 = jnp.array([8.0, 4.0, 0.0])
    np.testing.assert_allclose(rod.at(q0, None), [7.0, 4.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(rod.at(q0, 1.0), [7.0, 4.0, 1.0], atol=1e-12)


def test_two_half_steps_are_the_books():
    """Two steps held for 1/2 each: (7.5, 4, 0.5), then Ch 3's (7.0625, 4, 0.9375)."""
    rod = Rod()
    q = rod.at(jnp.array([8.0, 4.0, 0.0]), 0.5)
    np.testing.assert_allclose(q, [7.5, 4.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(rod.at(q, 0.5), [7.0625, 4.0, 0.9375], atol=1e-12)


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.5, 2.0])
def test_one_step_scales_with_duration(tau):
    """Predictions 2 and 3: the move is ``tau`` times the unit move, and nothing at ``tau = 0``."""
    rod = Rod()
    q0 = jnp.array([3.0, -1.0, 2.5])
    unit = rod.at(q0, None) - q0
    np.testing.assert_allclose(rod.at(q0, tau) - q0, tau * unit, atol=1e-12)


def test_first_order_in_duration():
    """Prediction 4: over elapsed time 1, the gap to a fine run roughly halves as the duration halves."""
    rod = Rod()
    q0 = jnp.array([8.0, 4.0, 0.0])

    def run(n):
        q = q0
        for _ in range(n):
            q = rod.at(q, 1.0 / n)
        return q

    fine = run(256)
    gaps = [float(jnp.max(jnp.abs(run(n) - fine))) for n in (4, 8, 16)]
    for a, b in zip(gaps, gaps[1:]):
        assert 1.6 < a / b < 2.4, gaps
    # and the total heat is held at every duration (the forces are a difference across each joint)
    np.testing.assert_allclose(jnp.sum(run(4)), 12.0, atol=1e-12)


def no_law(Q):
    """A configuration part stripped of its duration law: what a part not yet given one looks like."""
    from dataclasses import replace
    return replace(configuration(Q), timed_step=None, label="no law")


def test_part_without_a_duration_law_raises():
    """Prediction 5: a rod with a part that has no duration law, handed a duration, raises."""
    arr = grouped_rod(no_law)
    assert arr.P.timed_step is None
    with pytest.raises(TypeError, match="no duration law"):
        Phi(arr).run_one(IN_POS, NO_DIR, 0.5)
    Phi(arr).run_one(IN_POS, NO_DIR)  # the unit step still runs


def test_phi_composes_at_a_duration():
    """Prediction 6: ``parallel`` of two ``Phi``s hands one duration to both factors and agrees with stepping
    each alone at that duration (``Phi`` of a wired composite is the rod tests above)."""
    a = harmonic_particle(configuration, 2.0, 1.0)
    b = harmonic_particle(configuration, 5.0, 3.0)
    Oa, Ob = Phi(a), Phi(b)
    sa, sb = jnp.array([1.5]), jnp.array([-0.5])
    Oa, Ob = Oa.with_state(sa), Ob.with_state(sb)
    tau = 0.3
    pos = IN_POS

    def dir_from(_o):
        return (jnp.zeros(1), jnp.array([0.25]))

    _, _, na = Oa.run_one(pos, dir_from, tau)
    _, _, nb = Ob.run_one(pos, dir_from, tau)
    _, _, nab = Oa.parallel(Ob).run_one((pos, pos), lambda _o: (dir_from(None), dir_from(None)), tau)
    np.testing.assert_allclose(nab[0], na, atol=1e-12)
    np.testing.assert_allclose(nab[1], nb, atol=1e-12)
    # and the step is the Euler family: s + tau * sharp(xi), xi = -kappa (s - y)
    np.testing.assert_allclose(na, sa + tau * (1 / 2.0) * (-1.0 * (sa - 0.25)), atol=1e-12)
    np.testing.assert_allclose(nb, sb + tau * (1 / 5.0) * (-3.0 * (sb - 0.25)), atol=1e-12)


def test_phase_unit_and_still():
    """Prediction 7: a rod of phase parts at duration 1 is its unit step; at duration 0 it does not move."""
    arr = grouped_rod(phase)
    s0 = closed_step(arr, check=0)(arr.P.init)                 # any state off the start
    s0 = closed_step(arr, check=0)(s0)
    one, unit = closed_step(arr, duration=1.0)(s0), closed_step(arr)(s0)
    zero = closed_step(arr, duration=0.0)(s0)
    import jax
    for a, b in zip(jax.tree_util.tree_leaves(one), jax.tree_util.tree_leaves(unit)):
        np.testing.assert_allclose(a, b, atol=1e-12)
    for a, b in zip(jax.tree_util.tree_leaves(zero), jax.tree_util.tree_leaves(s0)):
        np.testing.assert_allclose(a, b, atol=1e-12)


def lookahead_spring(frac):
    """A mass 1 on a spring of constant 1 as one closed box whose look-ahead is ``frac`` of the duration:
    coast ``frac tau``, take the push there, coast the rest. ``frac = 1`` is the phase part's law,
    ``frac = 1/2`` the leapfrog (drift, kick, drift)."""
    from arrangements.arrangement import Arrangement, exact
    from arrangements.coalgebra import CotCoalg
    def timed(s, xi, t):                     # coast frac*t, push, coast the rest on the new momentum
        q_seen = s[0] + frac * t * s[1]
        p = s[1] + t * xi
        return (q_seen + (1 - frac) * t * p, p)

    P = CotCoalg(
        Q_dim=1, init=(jnp.array([1.0]), jnp.array([0.0])),
        position=lambda s: s[0] + frac * s[1],
        step=lambda s, xi: timed(s, xi, 1.0),
        look_ahead=lambda s, t: s[0] + frac * t * s[1],
        timed_step=timed,
    )
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
                       omega=exact(lambda q, m, n: -0.5 * q[0] ** 2), label=f"spring({frac})")


def test_half_look_ahead_is_second_order():
    """Prediction 8: over elapsed time 5, halving the duration cuts the error by about 4 with a half
    look-ahead and about 2 with the full one; one push per step either way."""
    exact_q = np.cos(5.0)

    def err(frac, n):
        arr = lookahead_spring(frac)
        step = closed_step(arr, check=0, duration=5.0 / n)
        s = arr.P.init
        for _ in range(n):
            s = step(s)
        return abs(float(s[0][0]) - exact_q)

    half = [err(0.5, n) for n in (50, 100, 200)]
    full = [err(1.0, n) for n in (50, 100, 200)]
    for a, b in zip(half, half[1:]):
        assert 3.6 < a / b < 4.4, half
    for a, b in zip(full, full[1:]):
        assert 1.6 < a / b < 2.4, full


def test_damped_friction_is_a_push_held_for_the_duration():
    """Prediction 9: a rod of damped parts at duration 1 is its unit step, at 0 is still, and a free damped
    mass keeps ``1 - c tau`` of its momentum over a step of ``tau``."""
    import jax
    c = 0.2
    arr = grouped_rod(lambda Q: damped(Q, c))
    s0 = closed_step(arr, check=0)(closed_step(arr, check=0)(arr.P.init))
    for tau, want in ((1.0, closed_step(arr)(s0)), (0.0, s0)):
        got = closed_step(arr, duration=tau)(s0)
        for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
            np.testing.assert_allclose(a, b, atol=1e-12)
    P = damped(diagonal(jnp.array([1.0])), c, init=(jnp.array([0.0]), jnp.array([2.0])))
    for tau in (0.25, 0.5, 2.0):
        _, xi = P.step_over(P.init, jnp.zeros(1), tau)
        np.testing.assert_allclose(xi, (1 - c * tau) * 2.0, atol=1e-12)


@pytest.mark.parametrize("make", [
    lambda Q: gyro(Q, damping=0.1, gamma=0.3, J=jnp.array([[0.0]])),
    lambda Q: phase_on_cotangent(Q, damping=0.1),
])
def test_gyro_and_cotangent_parts_at_duration_one_and_zero(make):
    """The gyro and phase-on-T*Q parts, every push held for the duration: duration 1 is the unit step, 0 is still."""
    P = make(diagonal(jnp.array([0.5])))
    s0 = (jnp.array([0.3]), jnp.array([1.2]))
    xi = jnp.full((P.Q_dim,), 0.7)
    import jax
    for tau, want in ((1.0, P.step(s0, xi)), (0.0, s0)):
        got = P.step_over(s0, xi, tau)
        for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
            np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(P.position_at(s0, 1.0), P.position(s0), atol=1e-12)


def _module_parts():
    from arrangements.arm import geared_joints
    from arrangements.belief import belief_coalgebra
    from arrangements.genetic_code import tensor_all
    from arrangements.lay_pairs import phase_dt
    return [
        ("geared", geared_joints(jnp.array([0.2, -0.1]), 0.6).P, None),
        ("belief", belief_coalgebra(3, jnp.array([0.1, -0.2, 0.3]), tau=0.5), None),
        ("phase_dt", phase_dt(0.1, 0.4, -0.3), None),
        ("tensor_all", tensor_all([harmonic_particle(configuration, 2.0, 1.0),
                                   harmonic_particle(phase, 3.0, 1.0)]).P, None),
    ]


@pytest.mark.parametrize("name", ["geared", "belief", "phase_dt", "tensor_all"])
def test_module_parts_at_duration_one_and_zero(name):
    """The module-built continuous parts (book P9.5): at duration 1 each is its unit step and shows its
    position; at duration 0 each is still."""
    import jax
    P = dict((n, p) for n, p, _ in _module_parts())[name]
    s0 = P.init
    xi = jnp.linspace(0.3, 0.9, P.Q_dim)
    s1 = P.step(s0, xi)                               # a state off the start
    for tau, want in ((1.0, P.step(s1, xi)), (0.0, s1)):
        got = P.step_over(s1, xi, tau)
        for a, b in zip(jax.tree_util.tree_leaves(got), jax.tree_util.tree_leaves(want)):
            np.testing.assert_allclose(a, b, atol=1e-12)
    np.testing.assert_allclose(P.position_at(s1, 1.0), P.position(s1), atol=1e-12)


def test_short_steps_of_one_example_each_are_full_batch_gradient_flow():
    """The data source shows the next example every step, whatever the duration (David, 2026-10-10:
    "They do 0.01, bam, bam, bam … we just show them 100 new data points per second"). Over a fixed elapsed
    time, shorter steps run the examples faster, and the drawn network's parameters approach full-batch
    gradient flow on the mean loss, first order in the duration: one-example-per-step training averages
    over the data as the steps shrink. Measured at writing: distance 0.0502, 0.0162, 0.0047, 0.0005 at
    durations 1, 0.3, 0.1, 0.01."""
    import jax
    from arrangements.network import drawn_network, parameters
    h, m, n = 4, 1, 1
    X = np.linspace(-2, 2, 12).reshape(-1, 1)
    Y = np.sin(X)
    p0 = np.random.default_rng(0).normal(0, 0.5, h * m + h + n * h + n)
    eta, T = 0.05, 40.0

    def net(p):
        W1, b1, W2, b2 = p[:4].reshape(4, 1), p[4:8], p[8:12].reshape(1, 4), p[12:13]
        return (W2 @ jnp.tanh(W1 @ X.T + b1[:, None])).T + b2

    g = jax.jit(jax.grad(lambda p: jnp.mean(0.5 * jnp.sum((net(p) - Y) ** 2, axis=1))))
    p = jnp.asarray(p0)
    for _ in range(20000):
        p = p - (T / 20000) * eta * g(p)
    A = drawn_network(h, m, n, p0, eta, X, Y)

    def dist(tau):
        step, s = closed_step(A, duration=tau, check=0), A.P.init
        for _ in range(int(round(T / tau))):
            s = step(s)
        return float(jnp.max(jnp.abs(parameters(A, s, m, len(p0)) - p)))

    d1, d2 = dist(0.1), dist(0.01)
    assert d2 < 0.002 and 6 < d1 / d2 < 14, (d1, d2)


def test_counting_parts_take_one_step_per_step():
    """The runner's drum and copier and system identification's delay take one step per step, whatever the
    duration (David, 2026-10-10, "if you're happy with this and if the math works, then let's do it")."""
    import jax
    from arrangements import system_id
    src = open(system_id.__file__).read()
    assert "timed_step=lambda s, xi, t: jnp.full(2, s[1] + xi[1])" in src
    from arrangements.network import example_coalgebra
    P = example_coalgebra(jnp.arange(6.0).reshape(3, 2))
    for tau in (1e-9, 0.01, 1.0, 3.0):          # durations are positive; the next example at every one
        assert int(P.step_over(P.init, jnp.zeros(2), tau)) == 1


def _flat(state):
    import jax
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(state)])


def test_leapfrog_two_rounds_at_a_duration():
    """The two-round leapfrog told a duration before round 1 (David, 2026-10-10, "carry it through"): at
    duration 1 it is its unit step; a wave chain run over a fixed time with shorter steps converges at second
    order (the gap to a fine run falls by about 4 when the duration halves)."""
    from arrangements.leapfrog import leapfrog
    from arrangements.pc2 import Phi2
    arr = held_chain([harmonic_particle(leapfrog, 1.0, 1.0)] * 3, 1.0, leapfrog)
    O = Phi2(arr)
    nil = lambda _o: (jnp.zeros(0), jnp.zeros(0))
    s0 = O.run_one(IN_POS, nil)[2]                       # a state off the start
    s0 = jax_tree_set_first_q(arr, s0)
    a = O.with_state(s0).run_one(IN_POS, nil, 1.0)[2]
    b = O.with_state(s0).run_one(IN_POS, nil)[2]
    np.testing.assert_allclose(_flat(a), _flat(b), atol=1e-12)

    def run(n, T=4.0):
        s = s0
        for _ in range(n):
            s = O.with_state(s).run_one(IN_POS, nil, T / n)[2]
        return _flat(s)

    fine = run(256)
    g = [float(jnp.max(jnp.abs(run(k) - fine))) for k in (16, 32, 64)]
    for x, y in zip(g, g[1:]):
        assert 3.4 < x / y < 4.6, g


def jax_tree_set_first_q(arr, s):
    """Displace the first particle of a held leapfrog chain so the run is not at rest."""
    import jax
    leaves, tree = jax.tree_util.tree_flatten(s)
    k = next(i for i, l in enumerate(leaves) if jnp.size(l) == 1 and i > 1)
    leaves[k] = leaves[k] + 0.5
    return jax.tree_util.tree_unflatten(tree, leaves)


def test_rk4_at_a_duration_is_rk4_with_the_step_scaled():
    """RK4 of step ``h`` told a duration ``tau`` is RK4 of step ``h tau``: one ``PhiK`` step of each agrees."""
    from arrangements.rk4 import rk4
    from arrangements.pcK import PhiK
    from arrangements.arrangement import Arrangement, exact
    Q = diagonal(jnp.array([1.0]))

    def spring(P):
        return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
                           out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, n: jnp.zeros(0),
                           omega=exact(lambda q, m, n: -0.5 * q[0] ** 2), label="spring")

    nil = lambda _o: (jnp.zeros(0), jnp.zeros(0))
    x0 = jnp.array([1.3])
    a = PhiK(spring(rk4(Q, 0.2, x0))).run_one(IN_POS, nil, 0.5)[1]
    b = PhiK(spring(rk4(Q, 0.1, x0))).run_one(IN_POS, nil)[1]
    np.testing.assert_allclose(a, b, atol=1e-14)


def test_langevin_jostling_grows_with_the_square_root_of_the_duration():
    """The random step over a duration: the push held for it and the Gaussian kick scaled by its square root."""
    from arrangements.rand import euler_maruyama
    Q = diagonal(jnp.array([0.5]))
    P = euler_maruyama(Q, temperature=0.8, init=jnp.array([0.2]))
    xi, z = jnp.array([1.0]), jnp.array([0.7])
    for tau in (1.0, 0.25, 0.04):
        got = P.step_over(P.init, xi, tau).ret(z)
        want = 0.2 + tau * 0.5 * 1.0 + jnp.sqrt(2 * 0.8) * jnp.sqrt(tau) * jnp.sqrt(0.5) * 0.7
        np.testing.assert_allclose(got, want, atol=1e-12)
    np.testing.assert_allclose(P.step_over(P.init, xi, 1.0).ret(z), P.step(P.init, xi).ret(z), atol=1e-12)
