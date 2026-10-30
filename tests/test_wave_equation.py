"""Waves along a chain of three masses, and the chain of any length with its ends held (book Chapter 2,
``sec.waves`` in ``sec.heat_waves``; linked, with ``test_heat_equation.py``,
from the footnote in ``sec.waves``
on what makes the difference between heat and waves).

For a reader of the book. The same wiring ``eqn.chain_wd`` as the rod's, filled with masses that slide
without friction. A mass's state is a pair, its position ``q_i`` and its momentum ``p_i``; it reads out
the position it is headed to, ``q_i + p_i/m_i`` (``eqn.mass_readout``), and that is what the springs
receive. The force ``xi_i`` that comes back goes to both parts of the state, as ``(xi_i, xi_i/m_i)``; its
internal inclination is minus the differential of its kinetic energy, ``(0, -p_i/m_i)``; and its response
function ``eqn.wave_response`` sends the force into the momentum and the inclination into the position,
so one step is ``q_i' = q_i + p_i/m_i``, ``p_i' = p_i + xi_i`` (``eqn.wave_step``, symplectic Euler). The
springs are those of ``sec.springs``. In code (``tests/chain.py``) a mass is
``one_port_box(phase, 1/m, ([q0], [p0]), "mass")``: the ``phase`` coalgebra holds ``(q, p)``, presents
``q + p/m``, and steps as the book's response function does; a spring is ``spring(kappa)``; the chain is
``book_chain(masses, [kappa_1, kappa_2])``, its ends free as in the book, so the total momentum stays
put. The book's hand steps (unit masses, spring constants ``1/4``, from positions ``(8, 0, 0)`` at
rest), run from the library's folder with it on the path, as the README's *Run it* says
(``PYTHONPATH=. .venv/bin/python example.py``):

    import jax.numpy as jnp
    from arrangements.coalgebra import phase
    from tests.chain import one_port_box, book_chain, closed_run, flat_state
    masses = [one_port_box(phase, 1.0, (jnp.array([q]), jnp.zeros(1)), "mass") for q in (8.0, 0.0, 0.0)]
    _, states = closed_run(book_chain(masses, [0.25, 0.25]), 3)
    for s in states:
        print(flat_state(s).reshape(3, 2).T)   # row 0 the positions, row 1 the momenta
    # (8,0,0),(0,0,0); (8,0,0),(-2,2,0); (6,2,0),(-3,2.5,0.5); (3,4.5,0.5),(-2.625,1.125,1.5)

What the tests check. ``test_book_chain_by_hand``: those steps; that the masses read out the position
they are headed to; the total momentum ``0`` and the positions' sum ``8`` at every step
(``exc.waves_sum``); every point of the plot ``eqn.wave_plot``; and that the masses do not settle, the
middle one passing ``8/3``. The rest run the chain of ``K`` harmonic particles with both ends held at
``0`` by walls (``wiring.held_chain``), each particle a box carrying its own spring to its left
neighbour, as ``test_heat_equation.py`` does with ``configuration`` in place of ``phase``: the step is the
book's; the recurrence ``m (q(t+1) - 2 q(t) + q(t-1)) = kappa Lap q(t)`` holds at every step; the open
chain's one-step update and the one-form it presents at its left port are checked against the formulas
of the paper this code grew from; and the same wiring filled two ways gives two arrangements with the
same maps and different states (``test_heat_and_wave_are_two_arrangements_on_one_wiring``).

For agents: ``regress/wave.npz`` holds the CDLM trajectories this reproduces (states identical,
covectors negated: STATUS.md §1); the pinned ends once computed in code each step are kept as
``_step_by_port`` and checked equal to the held chain. The file's earlier head is in
``notes/P31-B-CH2-HISTORY-2026-10-30.md``.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi
from arrangements.wiring import compose_chain

from tests.chain import (
    IN_POS, HeldChain, book_chain, closed_run, flat_state, harmonic_particle, load_regress, one_port_box,
    phase_flat, phase_state,
)


def _wave_chain(K, m, kappa):
    return compose_chain([harmonic_particle(phase, m, kappa)] * K)


def _composite_final(q, p, q_prev_extern, xi_N_extern, m, kappa):
    """Reference implementation of CDLM eqn.potlens_composite_final, with the
    external covector ``xi_N_extern`` a forcing (added, not subtracted)."""
    K = q.shape[0]
    qt = q + p / m  # presented positions q~ (forces are evaluated here)
    q_new = qt
    p_new = []
    for i in range(K):
        if i < K - 1:
            q_prev = q_prev_extern if i == 0 else qt[i - 1]
            q_next = qt[i + 1]
            p_i_new = p[i] + kappa * (q_prev + q_next - 2.0 * qt[i])
        else:
            q_prev = qt[i - 1] if K > 1 else q_prev_extern
            p_i_new = p[i] + kappa * (q_prev - qt[i]) + xi_N_extern
        p_new.append(p_i_new)
    return q_new, jnp.stack(p_new)


def _held(K, m, kappa):
    """The chain with its ends held by boxes, closed."""
    return HeldChain([harmonic_particle(phase, m, kappa)] * K, kappa, phase)


def _step_by_port(O, q, p, kappa, m):
    """The old pinned step, ends held in code: q_prev = 0 fed in, xi_N = -kappa * q~_K computed."""
    xi_N = -kappa * float(q[-1] + p[-1] / m)
    _, _, (q, p) = _run(O, q, p, xi_N, 0.0)
    return q, p


def _run(O, q, p, xi_N, q_prev):
    K = q.shape[0]
    out_pos, out_dir, new_state = O.with_state(phase_state(q, p)).run_one(
        IN_POS, lambda _o: (jnp.atleast_1d(xi_N), jnp.atleast_1d(q_prev)))
    return out_pos, out_dir, phase_flat(new_state, K)


# ---------------------------------------------------------------------------
# The chain as the book draws it (sec.waves): three masses, two springs, free ends.
# ---------------------------------------------------------------------------


def _masses(qs, kappas=(0.25, 0.25), m=1.0):
    """Three masses ``m`` at the positions ``qs``, at rest, joined by springs of constants ``kappas``:
    ``book_chain`` of ``one_port_box(phase, 1/m, ...)``."""
    masses = [one_port_box(phase, 1.0 / m, (jnp.array([q]), jnp.zeros(1)), "mass") for q in qs]
    return book_chain(masses, list(kappas))


def _positions_and_momenta(states, K):
    """The stored ``(q, p)`` of each mass at each state: arrays ``q[t, i]``, ``p[t, i]``."""
    qp = np.stack([np.asarray(flat_state(s)).reshape(K, 2) for s in states])
    return qp[:, :, 0], qp[:, :, 1]


WAVE_PLOT = np.array([   # eqn.wave_plot: steps 0..24 of the first, middle and last mass, 3 decimals
    [8, 0, 0], [8, 0, 0], [6, 2, 0], [3, 4.5, 0.5], [0.375, 5.625, 2], [-0.938, 4.531, 4.406],
    [-0.883, 2.039, 6.844], [-0.098, 0.018, 8.08], [0.716, -0.017, 7.301], [1.347, 1.961, 4.692],
    [2.131, 4.468, 1.401], [3.5, 5.624, -1.124], [5.399, 4.562, -1.961], [7.09, 2.078, -1.168],
    [7.527, 0.036, 0.437], [6.092, -0.034, 1.942], [3.125, 1.922, 2.953], [-0.142, 4.436, 3.706],
    [-2.265, 5.623, 4.642], [-2.416, 4.593, 5.823], [-0.814, 2.118, 6.697], [1.52, 0.054, 6.426],
    [3.488, -0.05, 4.562], [4.571, 1.884, 1.545], [4.983, 4.404, -1.387],
])


def test_book_chain_by_hand():
    """sec.waves' hand steps, unit masses and spring constants 1/4, from (8,0,0) at rest: the positions and
    momenta go (8,0,0),(0,0,0) -> (8,0,0),(-2,2,0) -> (6,2,0),(-3,5/2,1/2) -> positions (3,4.5,0.5);
    each mass reads out the position it is headed to (eqn.mass_readout); each step is eqn.wave_step with
    the free-ended forces eqn.chain_forces at the read-out positions; the total momentum is 0 and the
    positions sum to 8 at every step (exc.waves_sum); the plot eqn.wave_plot, point by point; and the
    middle mass passes 8/3 and the chain never settles."""
    K = 3
    readouts, states = closed_run(_masses((8.0, 0.0, 0.0)), 24)
    q, p = _positions_and_momenta(states, K)
    np.testing.assert_allclose(q[:4], [[8, 0, 0], [8, 0, 0], [6, 2, 0], [3, 4.5, 0.5]], atol=1e-12)
    np.testing.assert_allclose(p[:3], [[0, 0, 0], [-2, 2, 0], [-3, 2.5, 0.5]], atol=1e-12)
    np.testing.assert_allclose(np.stack(readouts), q + p, atol=1e-12)          # unit masses: q + p/m
    L = 0.25 * np.array([[-1.0, 1.0, 0.0], [1.0, -2.0, 1.0], [0.0, 1.0, -1.0]])   # the forces, free ends
    for t in range(24):
        np.testing.assert_allclose(q[t + 1], q[t] + p[t], atol=1e-12)
        np.testing.assert_allclose(p[t + 1], p[t] + L @ q[t + 1], atol=1e-12)
    np.testing.assert_allclose(p.sum(axis=1), 0.0, atol=1e-12)
    np.testing.assert_allclose(q.sum(axis=1), 8.0, atol=1e-12)
    np.testing.assert_allclose(q, WAVE_PLOT, atol=6e-4)
    assert q[3, 1] > 8.0 / 3.0 and q[:, 1].min() < 0.0 and q[:, 1].max() > 5.0


# ---------------------------------------------------------------------------
# The chain of any length with its ends held by walls (wiring.held_chain), and the open chain.
# ---------------------------------------------------------------------------


def test_state_update_matches_composite_final():
    """Apply Phi to a 5-particle phase chain; check eqn.potlens_composite_final."""
    K = 5
    m, kappa = 1.7, 2.3
    O = Phi(_wave_chain(K, m, kappa))

    rng = np.random.default_rng(20260507)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        p = jnp.asarray(rng.standard_normal(K))
        q_prev = float(rng.standard_normal())
        xi_N = float(rng.standard_normal())

        _, _, (new_q, new_p) = _run(O, q, p, xi_N, q_prev)

        ref_q, ref_p = _composite_final(q, p, q_prev, xi_N, m, kappa)
        np.testing.assert_allclose(np.asarray(new_q), np.asarray(ref_q), atol=1e-10)
        np.testing.assert_allclose(np.asarray(new_p), np.asarray(ref_p), atol=1e-10)


def test_omega_N_is_harmonic_spring_forcing():
    """The published one-form on N^-: omega_N(q_0) = kappa(q_1 - q_0), i.e. the
    forcing ``-dk`` of the first spring seen from the input port (minus CDLM's eqn.omegaprime)."""
    K = 5
    m, kappa = 1.0, 0.7
    O = Phi(_wave_chain(K, m, kappa))

    q = jnp.array([0.4, -0.2, 1.1, 0.0, 2.3])
    p = jnp.zeros(K)
    _, fiber = O.step(phase_state(q, p))
    out_pos, _ = fiber(IN_POS)
    out_n, omega_N = out_pos  # omega_N is the one-form (a callable)
    np.testing.assert_allclose(np.asarray(out_n), np.asarray(q[K - 1:K]), atol=1e-10)
    # omega_N(q_0) = kappa(q_1 - q_0): value at 0 is b, the 1D slope is A.
    b_N = omega_N(jnp.zeros(1))
    A_N = omega_N(jnp.ones(1)) - b_N
    np.testing.assert_allclose(np.asarray(A_N), np.array([-kappa]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(b_N), np.array([kappa * float(q[0])]), atol=1e-10)


def test_discrete_wave_equation_with_pinned_ends():
    """eqn.discrete_wave: m * ddot q_i = kappa (q_{i-1} - 2 q_i + q_{i+1}) at every step.

    Boundary conditions: q_0 := 0, xi_N := -kappa * q~_K (presented), a forcing.
    The recurrence is the centered one: m ddot q_i(t) = kappa Lap(q(t+1))_i.
    """
    K = 5
    T = 10
    m, kappa = 1.5, 0.9
    H = _held(K, m, kappa)

    rng = np.random.default_rng(42)
    q = jnp.asarray(rng.standard_normal(K))
    p = jnp.zeros(K)

    q_traj = [q]
    for _ in range(T + 2):
        q, p = H.step(q, p)
        q_traj.append(q)

    q_arr = np.stack([np.asarray(x) for x in q_traj], axis=0)
    for t in range(T):
        ddot_q = q_arr[t + 2] - 2.0 * q_arr[t + 1] + q_arr[t]
        q_aug = np.concatenate([[0.0], q_arr[t + 1], [0.0]])
        laplacian = q_aug[:-2] - 2.0 * q_aug[1:-1] + q_aug[2:]
        residual = m * ddot_q - kappa * laplacian
        np.testing.assert_allclose(residual, np.zeros(K), atol=1e-10)


def test_held_ends_are_the_port_fed_run():
    """The held chain's ``Phi`` equals the old run that held the ends in code, 200 steps."""
    K, m, kappa = 5, 1.5, 0.9
    H, O = _held(K, m, kappa), Phi(_wave_chain(K, m, kappa))
    rng = np.random.default_rng(1)
    q = a = jnp.asarray(rng.standard_normal(K))
    p = b = jnp.asarray(rng.standard_normal(K))
    for _ in range(200):
        q, p = H.step(q, p)
        a, b = _step_by_port(O, a, b, kappa, m)
    np.testing.assert_allclose(np.asarray(q), np.asarray(a), atol=1e-12)
    np.testing.assert_allclose(np.asarray(p), np.asarray(b), atol=1e-12)


def test_heat_and_wave_are_two_arrangements_on_one_wiring():
    """STATUS.md §1: the integrator is part of the arrangement. The same wiring
    ``compose_chain`` on the same particle data gives two arrangements, one with
    ``configuration`` parameters and one with ``phase``; ``Phi`` takes no
    integrator, and the two differ in their parameter, not in their maps."""
    K, m, kappa = 3, 1.5, 0.9
    heat = compose_chain([harmonic_particle(configuration, m, kappa)] * K)
    wave = compose_chain([harmonic_particle(phase, m, kappa)] * K)

    # same interface and same lens data
    assert (heat.in_dim_M, heat.out_dim_M, heat.in_dim_N, heat.out_dim_N) == \
           (wave.in_dim_M, wave.out_dim_M, wave.in_dim_N, wave.out_dim_N)
    q = jnp.array([0.3, -1.0, 2.0]); n_in = jnp.array([0.7])
    np.testing.assert_allclose(heat.out_f(q, jnp.zeros(0)), wave.out_f(q, jnp.zeros(0)))
    for a, b in zip(heat.omega(q, jnp.zeros(0), n_in), wave.omega(q, jnp.zeros(0), n_in)):
        np.testing.assert_allclose(a, b)

    # different parameters: state R^K vs T^*R^K, hence different dynamics from Phi alone
    O_heat, O_wave = Phi(heat), Phi(wave)
    s_heat, s_wave = O_heat.state, O_wave.state
    assert heat.P.Q_dim == wave.P.Q_dim == K
    assert isinstance(s_heat[0][0][0], jnp.ndarray) and isinstance(s_wave[0][0][0], tuple)
    import inspect
    assert list(inspect.signature(Phi).parameters) == ["arr"]


def test_regression_matches_cdlm():
    """The CDLM outputs (``regress/wave.npz``) are reproduced to rtol 1e-10 (STATUS.md §1):
    states identical, environment covectors negated, published field negated."""
    R = load_regress("wave")

    K, m, kappa = R["cf_params"]; K = int(K)
    O = Phi(_wave_chain(K, m, kappa))
    for i in range(R["cf_q"].shape[0]):
        _, _, (nq, np_) = _run(O, jnp.asarray(R["cf_q"][i]), jnp.asarray(R["cf_p"][i]),
                               -float(R["cf_xiN"][i]), float(R["cf_qprev"][i]))
        np.testing.assert_allclose(np.asarray(nq), R["cf_new_q"][i], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(np.asarray(np_), R["cf_new_p"][i], rtol=1e-10, atol=1e-12)

    K2, m2, kappa2 = R["field_params"]; K2 = int(K2)
    O2 = Phi(_wave_chain(K2, m2, kappa2))
    _, fiber = O2.step(phase_state(jnp.asarray(R["field_q"]), jnp.zeros(K2)))
    (out_n, omega_N), _ = fiber(IN_POS)
    np.testing.assert_allclose(np.asarray(out_n), R["field_out_n"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(omega_N(jnp.zeros(1))), -R["field_at_0"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.asarray(omega_N(jnp.ones(1))), -R["field_at_1"], rtol=1e-10, atol=1e-12)

    K3, T3, m3, kappa3 = R["pin_params"]; K3, T3 = int(K3), int(T3)
    H3 = _held(K3, m3, kappa3)
    q, p = jnp.asarray(R["pin_q"][0]), jnp.asarray(R["pin_p"][0])
    qs, ps = [q], [p]
    for _ in range(T3 + 2):
        q, p = H3.step(q, p)
        qs.append(q); ps.append(p)
    np.testing.assert_allclose(np.stack([np.asarray(x) for x in qs]), R["pin_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.stack([np.asarray(x) for x in ps]), R["pin_p"], rtol=1e-10, atol=1e-12)
