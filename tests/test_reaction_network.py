"""Mass-action reaction networks as ``Phi`` of configuration arrangements (dap/reaction_network.py).

The frontier under test (module docstring of reaction_network.py):

* MONOLITHIC, exact: for detailed-balanced networks -- a unimolecular chain
  A<->B<->C with unequal rates and the bimolecular A+B<->C -- one ``Phi``
  step of the Onsager/free-energy arrangement IS the forward-Euler
  mass-action step ``x + dt * sum_r zeta_r v_r(x)``, to machine precision
  (keystone);
* convergence to the detailed-balance equilibrium OF THE INITIAL CONDITION'S
  stoichiometric compatibility class, from generic positive starts, with the
  free energy decreasing monotonically and the class invariants conserved to
  rounding;
* COMPOSITIONAL, exact for unimolecular: the species-box/reaction-box/wire
  composite (built by compose_seq/tensor, forcing emerging from
  pullback-and-sum -- checked against the closed formula) steps identically to
  both the monolithic encoding and the ground truth;
* OBSTRUCTION: the bimolecular A+B<->C has NO species-box realization -- the
  g-free separability certificate (opposite-Jacobian-entry ratio must be
  independent of third species) fails, reversibly and irreversibly, while the
  unimolecular chain passes it (positive control);
* networks without a detailed-balance point (Wegscheider-violating triangle)
  are rejected by the constructors;
* COMPOSITIONAL, ARBITRARY Petri net (doubled transition boxes + species as
  affine wiring): one tick is the exact forward-Euler mass-action step of a
  bimolecular, one-way, detailed-balance-free net; eps' is frozen and its value
  never reaches the dynamics; the composite sharp is nilpotent and
  non-symmetric (the certificate that nothing is being descended); the total
  forcing emerges from compose_seq; boxes are local; conservation laws hold
  structurally.
* COMPOSITIONAL, ARBITRARY Petri net IN LOG COORDINATES (doubled species boxes
  + stateless transitions + prism wiring): one tick is the exact forward-Euler
  mass-action step in log-concentration coordinates ``x * exp(dt (Z^T f)/x)``;
  concentrations stay positive by construction at a ``dt`` where encoding 4
  overshoots negative; it converges to mass action as ``dt -> 0``; ``y'`` is
  frozen and irrelevant; the moiety trade-off is dual to encoding 4's
  (conservation drifts, shrinking with ``dt``); the wire is a bijection and
  the forcing emerges from compose_seq.
* DIRECT (new, book only): the mass-action field as a NON-EXACT forcing on the
  plain ``configuration(Q)`` with Euclidean sharp ``dt * I``. It agrees step
  for step with the monolithic trick realization on detailed-balanced
  networks, and it runs the networks the trick's no-go excludes -- a
  Wegscheider-violating triangle and an irreversible reaction -- whose forcing
  is provably not ``dU``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.reaction_network import (
    check_detailed_balance,
    crn_arrangement,
    crn_dynamics,
    crn_step,
    flat_state,
    free_energy,
    log_mass_action_step,
    mass_action_arrangement,
    mass_action_dynamics,
    mass_action_rhs,
    onsager_rvect,
    petri_arrangement,
    petri_concentrations,
    petri_dynamics,
    petri_extents,
    petri_incidences,
    petri_initial_state,
    petri_log_arrangement,
    petri_log_concentrations,
    petri_log_dynamics,
    petri_log_incidences,
    petri_log_initial_state,
    petri_log_species_box,
    petri_log_transition_box,
    petri_log_wire,
    petri_step,
    petri_transition_box,
    petri_wire,
    separability_ratio,
    step_matrix,
    unimolecular_arrangement,
    unimolecular_dynamics,
    unimolecular_stoichiometry,
)

from tests.chain import load_regress


# ---------------------------------------------------------------------------
# The two fixture networks (rates chosen so detailed balance is float-exact).
# ---------------------------------------------------------------------------

# Chain A <-> B <-> C, unequal rates: kf1=2, kb1=1, kf2=0.75, kb2=1.
CHAIN_EDGES = [(0, 1), (1, 2)]
CHAIN_KF = [2.0, 0.75]
CHAIN_KB = [1.0, 1.0]
CHAIN_XEQ = jnp.array([1.0, 2.0, 1.5])  # 2*1=1*2, 0.75*2=1*1.5: exact
CHAIN_ALPHA, CHAIN_BETA = unimolecular_stoichiometry(3, CHAIN_EDGES)

# Bimolecular A + B <-> C: kf=2, kb=1.
BI_ALPHA = jnp.array([[1.0, 1.0, 0.0]])
BI_BETA = jnp.array([[0.0, 0.0, 1.0]])
BI_KF = [2.0]
BI_KB = [1.0]
BI_XEQ = jnp.array([1.0, 0.5, 1.0])  # 2*1*0.5 = 1*1: exact

NONE = jnp.zeros(0)


def _random_positive(rng, n, lo=0.2, hi=3.0):
    return jnp.asarray(rng.uniform(lo, hi, size=n))


def _omega_q(arr, q):
    """The parameter component of a closed arrangement's forcing at ``q``."""
    return arr.omega(q, NONE, NONE)[0]


# ---------------------------------------------------------------------------
# (i) One-step exactness against the true mass-action Euler step.
# ---------------------------------------------------------------------------


def test_chain_one_step_exact_monolithic():
    """Keystone: ``Phi`` step of the Onsager arrangement = x + dt * Z^T v(x)."""
    rhs = mass_action_rhs(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB)
    rng = np.random.default_rng(0)
    for dt in (1.0, 0.05):
        O = crn_dynamics(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt)
        for _ in range(5):
            x = _random_positive(rng, 3)
            new_x = crn_step(O, x)
            exact = x + dt * rhs(x)
            np.testing.assert_allclose(
                np.asarray(new_x), np.asarray(exact), rtol=0.0, atol=1e-14
            )


def test_bimolecular_one_step_exact_monolithic():
    """The SAME keystone survives molecularity: A+B<->C steps exactly."""
    rhs = mass_action_rhs(BI_ALPHA, BI_BETA, BI_KF, BI_KB)
    rng = np.random.default_rng(1)
    for dt in (1.0, 0.02):
        O = crn_dynamics(BI_ALPHA, BI_BETA, BI_KF, BI_KB, BI_XEQ, dt)
        for _ in range(5):
            x = _random_positive(rng, 3)
            new_x = crn_step(O, x)
            exact = x + dt * rhs(x)
            np.testing.assert_allclose(
                np.asarray(new_x), np.asarray(exact), rtol=0.0, atol=1e-14
            )


def test_one_step_exact_at_equilibrium_guard():
    """The log-mean guard branch (balanced fluxes) is also exact: the step fixes x_eq."""
    for dt in (1.0, 0.1):
        O = crn_dynamics(BI_ALPHA, BI_BETA, BI_KF, BI_KB, BI_XEQ, dt)
        new_x = crn_step(O, BI_XEQ)
        np.testing.assert_allclose(np.asarray(new_x), np.asarray(BI_XEQ), atol=1e-15)


# ---------------------------------------------------------------------------
# (ii)+(iii) Convergence to the in-class detailed-balance point; free energy
# monotone; class invariants conserved to rounding.
# ---------------------------------------------------------------------------


def test_chain_converges_free_energy_monotone():
    """From a generic positive start the chain reaches c * x_eq, c = sum(x0)/sum(x_eq).

    The compatibility class of the chain is {sum x = sum x0}; on it the unique
    detailed-balance point is the scaled equilibrium ray. F decreases at every
    step; total mass is conserved to rounding (the increment lies in span of
    the zeta_r, which kill 1 for unimolecular reactions).
    """
    dt = 0.05
    O = crn_dynamics(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt)
    F = free_energy(CHAIN_XEQ)
    x0 = jnp.array([3.0, 0.2, 1.0])
    target = (jnp.sum(x0) / jnp.sum(CHAIN_XEQ)) * CHAIN_XEQ

    x = x0
    f_prev = float(F(x))
    max_mass_drift = 0.0
    for _ in range(6000):
        x = crn_step(O, x)
        f_now = float(F(x))
        assert f_now <= f_prev + 1e-12
        f_prev = f_now
        max_mass_drift = max(max_mass_drift, abs(float(jnp.sum(x) - jnp.sum(x0))))
    assert max_mass_drift < 1e-12
    np.testing.assert_allclose(np.asarray(x), np.asarray(target), atol=1e-9)


def test_bimolecular_converges_free_energy_monotone():
    """A+B<->C reaches the in-class equilibrium; F monotone; A+C, B+C conserved.

    The class of x0 is {A+C = A0+C0, B+C = B0+C0}; its detailed-balance point
    solves kf (A0-s)(B0-s) = kb (C0+s) for the feasible root s*.
    """
    kf, kb = BI_KF[0], BI_KB[0]
    dt = 0.02
    O = crn_dynamics(BI_ALPHA, BI_BETA, BI_KF, BI_KB, BI_XEQ, dt)
    F = free_energy(BI_XEQ)
    A0, B0, C0 = 2.0, 1.2, 0.3
    x0 = jnp.array([A0, B0, C0])
    # kf s^2 - (kf(A0+B0) + kb) s + (kf A0 B0 - kb C0) = 0
    roots = np.roots([kf, -(kf * (A0 + B0) + kb), kf * A0 * B0 - kb * C0])
    s = min(r.real for r in roots if abs(r.imag) < 1e-12 and r.real < min(A0, B0))
    target = jnp.array([A0 - s, B0 - s, C0 + s])
    assert abs(kf * target[0] * target[1] - kb * target[2]) < 1e-12  # sanity

    x = x0
    f_prev = float(F(x))
    max_inv_drift = 0.0
    for _ in range(8000):
        x = crn_step(O, x)
        f_now = float(F(x))
        assert f_now <= f_prev + 1e-12
        f_prev = f_now
        max_inv_drift = max(
            max_inv_drift,
            abs(float(x[0] + x[2] - (A0 + C0))),
            abs(float(x[1] + x[2] - (B0 + C0))),
        )
    assert max_inv_drift < 1e-12
    np.testing.assert_allclose(np.asarray(x), np.asarray(target), atol=1e-9)


def test_onsager_sharp_symmetric_psd():
    """K(x) is symmetric PSD on x > 0 (the structured-realization claim is not vacuous)."""
    rng = np.random.default_rng(2)
    for alpha, beta, kf, kb in (
        (CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB),
        (BI_ALPHA, BI_BETA, BI_KF, BI_KB),
    ):
        Q = onsager_rvect(alpha, beta, kf, kb, dt=0.7)
        for _ in range(5):
            x = _random_positive(rng, 3)
            K = np.asarray(Q.sharp_at(x))
            np.testing.assert_allclose(K, K.T, atol=1e-14)
            assert np.min(np.linalg.eigvalsh(K)) > -1e-13


# ---------------------------------------------------------------------------
# (iv) Compositionality: unimolecular emerges from compose_seq; bimolecular
# provably cannot.
# ---------------------------------------------------------------------------


def test_unimolecular_compositional_potential_emerges():
    """The composite forcing (via compose_seq/tensor, written nowhere by hand)
    equals -d sum_r (w_r/2)(x_i/xeq_i - x_j/xeq_j)^2; the composite sharp is the
    direct sum dt * diag(x_eq)."""
    dt = 0.1
    arr = unimolecular_arrangement(3, CHAIN_EDGES, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt)
    assert arr.P.Q_dim == 3
    w = check_detailed_balance(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ)

    def U(x):
        a = x / CHAIN_XEQ
        return sum(
            0.5 * float(w[r]) * (a[i] - a[j]) ** 2 for r, (i, j) in enumerate(CHAIN_EDGES)
        )

    rng = np.random.default_rng(3)
    for _ in range(5):
        x = _random_positive(rng, 3)
        np.testing.assert_allclose(
            np.asarray(_omega_q(arr, x)), np.asarray(-jax.grad(U)(x)), atol=1e-13
        )
        np.testing.assert_allclose(
            np.asarray(step_matrix(arr.P, x)), np.asarray(dt * jnp.diag(CHAIN_XEQ)), atol=1e-15
        )


def test_unimolecular_compositional_step_exact():
    """``Phi`` of the composed arrangement = ground truth = monolithic, per step."""
    rhs = mass_action_rhs(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB)
    rng = np.random.default_rng(4)
    for dt in (1.0, 0.05):
        O_comp = unimolecular_dynamics(3, CHAIN_EDGES, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt)
        O_mono = crn_dynamics(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt)
        for _ in range(5):
            x = _random_positive(rng, 3)
            step_comp = crn_step(O_comp, x)
            exact = x + dt * rhs(x)
            np.testing.assert_allclose(
                np.asarray(step_comp), np.asarray(exact), rtol=0.0, atol=1e-14
            )
            np.testing.assert_allclose(
                np.asarray(step_comp), np.asarray(crn_step(O_mono, x)), rtol=0.0, atol=1e-14
            )


def test_bimolecular_species_box_obstruction():
    """The g-free certificate: any species-box realization x_k' = -g_k(x_k) dU/dx_k
    forces (dV_j/dx_i)/(dV_i/dx_j) = g_j(x_j)/g_i(x_i), separable. For A+B<->C
    the (A, C) ratio is (kf/kb) * x_B: doubling x_B (with x_A, x_C FIXED)
    doubles it, so no g's and U exist -- in any per-species coordinates, log
    included. Irreversibly (kb = 0) the denominator entry dV_A/dx_C vanishes
    against a nonzero dV_C/dx_A: production without back-dependence is exactly
    what a shared potential cannot return."""
    rhs = mass_action_rhs(BI_ALPHA, BI_BETA, BI_KF, BI_KB)
    ratio_AC = separability_ratio(rhs, 0, 2)
    p1 = jnp.array([1.0, 1.0, 1.0])
    p2 = jnp.array([1.0, 2.0, 1.0])  # only x_B changed; x_A, x_C identical
    r1, r2 = float(ratio_AC(p1)), float(ratio_AC(p2))
    np.testing.assert_allclose(r1, BI_KF[0] / BI_KB[0], atol=1e-13)       # = 2
    np.testing.assert_allclose(r2, 2.0 * BI_KF[0] / BI_KB[0], atol=1e-13)  # = 4
    assert abs(r2 - r1) > 1.0  # not separable: no species-box realization

    # Irreversible A+B->C: dV_A/dx_C = 0 while dV_C/dx_A = kf * x_B != 0.
    rhs_irrev = mass_action_rhs(BI_ALPHA, BI_BETA, BI_KF, [0.0])
    J = jax.jacobian(rhs_irrev)(p2)
    assert abs(float(J[0, 2])) == 0.0
    np.testing.assert_allclose(float(J[2, 0]), BI_KF[0] * 2.0, atol=1e-13)


def test_chain_passes_separability_positive_control():
    """The unimolecular chain PASSES the certificate: the (A, B) ratio is the
    constant kf1/kb1 = xeq_B/xeq_A wherever defined, matching the constant
    species sharps g_i = dt * xeq_i actually found by the construction."""
    rhs = mass_action_rhs(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB)
    ratio_AB = separability_ratio(rhs, 0, 1)
    rng = np.random.default_rng(5)
    expected = CHAIN_KF[0] / CHAIN_KB[0]
    assert abs(expected - float(CHAIN_XEQ[1] / CHAIN_XEQ[0])) < 1e-15
    for _ in range(5):
        x = _random_positive(rng, 3)
        np.testing.assert_allclose(float(ratio_AB(x)), expected, atol=1e-13)


# ---------------------------------------------------------------------------
# Scope limits are enforced, not silently absorbed.
# ---------------------------------------------------------------------------


def test_wegscheider_violating_triangle_rejected():
    """A<->B<->C<->A with cycle affinity != 0 has no detailed-balance point;
    both constructors refuse it rather than change the kinetics."""
    edges = [(0, 1), (1, 2), (2, 0)]
    kf, kb = [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]  # prod kf / prod kb = 1/2 != 1
    alpha, beta = unimolecular_stoichiometry(3, edges)
    x_eq = jnp.ones(3)
    with pytest.raises(ValueError):
        crn_arrangement(alpha, beta, kf, kb, x_eq, dt=0.1)
    with pytest.raises(ValueError):
        unimolecular_arrangement(3, edges, kf, kb, x_eq, dt=0.1)


def test_irreversible_rejected():
    """kb = 0 (irreversible) breaks the log-mean Onsager structure; rejected."""
    with pytest.raises(ValueError):
        crn_arrangement(BI_ALPHA, BI_BETA, BI_KF, [0.0], BI_XEQ, dt=0.1)


# ---------------------------------------------------------------------------
# 4. Arbitrary Petri nets: doubled transition boxes, species as wiring
#    (reaction_network.py module docstring, 4; the paper's "Petri nets" item).
#
# The fixture net is deliberately outside everything above: bimolecular
# (A + B -> C), a molecularity-2 product (C -> 2D), a three-species transition
# (D + A -> B), all transitions ONE-WAY, no detailed-balance point.
# ---------------------------------------------------------------------------

PETRI_M = jnp.array([[1., 1., 0., 0.],   # A + B -> C
                     [0., 0., 1., 0.],   # C -> A + B
                     [0., 0., 1., 0.],   # C -> 2D
                     [0., 0., 0., 2.],   # 2D -> C
                     [1., 0., 0., 1.]])  # D + A -> B
PETRI_N = jnp.array([[0., 0., 1., 0.],
                     [1., 1., 0., 0.],
                     [0., 0., 0., 2.],
                     [0., 0., 1., 0.],
                     [0., 1., 0., 0.]])
PETRI_R = jnp.array([1.3, 0.7, 0.9, 0.4, 0.6])
PETRI_X0 = jnp.array([1.4, 0.8, 0.5, 1.1])
PETRI_T = int(PETRI_M.shape[0])
# every transition is one-way, so the ground truth is mass action with kb = 0
PETRI_RHS = mass_action_rhs(PETRI_M, PETRI_N, PETRI_R, jnp.zeros(PETRI_T))


def _petri_fluxes(x):
    return PETRI_R * jnp.prod(x[None, :] ** PETRI_M, axis=1)


def test_petri_one_step_exact():
    """Keystone: one ``Phi`` tick of the Petri arrangement = x + dt * sum_t zeta_t f_t(x).

    Checked from a moving state, not just from eps = 0, so the affine wiring
    x = x0 + Z^T eps is exercised.
    """
    for dt in (1.0, 0.05, 0.001):
        O = petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt)
        s = petri_initial_state(PETRI_T)
        for _ in range(6):
            x = petri_concentrations(PETRI_M, PETRI_N, PETRI_X0, s)
            s = petri_step(O, s)
            got = petri_concentrations(PETRI_M, PETRI_N, PETRI_X0, s)
            np.testing.assert_allclose(
                np.asarray(got), np.asarray(x + dt * PETRI_RHS(x)), rtol=0.0, atol=1e-6
            )


def test_petri_long_run_matches_euler():
    """300 ticks track 300 forward-Euler steps of mass action."""
    dt = 0.01
    O = petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt)
    s, xe = petri_initial_state(PETRI_T), PETRI_X0
    for _ in range(300):
        s, xe = petri_step(O, s), xe + dt * PETRI_RHS(xe)
    got = petri_concentrations(PETRI_M, PETRI_N, PETRI_X0, s)
    np.testing.assert_allclose(np.asarray(got), np.asarray(xe), rtol=0.0, atol=1e-5)


def test_petri_eps_prime_frozen_and_irrelevant():
    """eps' never moves, and its value never reaches the dynamics.

    It is a direction to differentiate in, not a state: the forcing is LINEAR
    in it, so ``-dU/deps' = f_t(x)`` is the rate itself and does not depend on
    eps'.
    """
    dt = 0.01
    O = petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt)
    s1 = petri_initial_state(PETRI_T, eps_prime=1.0)
    s2 = petri_initial_state(PETRI_T, eps_prime=-3.7)
    for _ in range(50):
        s1, s2 = petri_step(O, s1), petri_step(O, s2)
    np.testing.assert_array_equal(np.asarray(s1[1::2]), np.full(PETRI_T, 1.0))
    np.testing.assert_array_equal(np.asarray(s2[1::2]), np.full(PETRI_T, -3.7))
    np.testing.assert_array_equal(
        np.asarray(petri_extents(s1)), np.asarray(petri_extents(s2))
    )


def test_petri_sharp_is_nilpotent_and_not_symmetric():
    """The honesty certificate: this realization descends nothing.

    The composite sharp -- the coalgebra's displacement per covector,
    ``step_matrix`` -- is block-diagonal with 2x2 transistor blocks, strictly
    upper triangular, hence nilpotent and non-symmetric, so it is neither an
    Onsager mobility nor any metric, and no potential is being minimized.
    """
    dt = 0.01
    arr = petri_arrangement(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt)
    S = step_matrix(arr.P, petri_initial_state(PETRI_T))
    assert arr.P.Q_dim == 2 * PETRI_T
    assert not np.allclose(np.asarray(S), np.asarray(S).T)
    np.testing.assert_allclose(np.asarray(S @ S), np.zeros_like(np.asarray(S)), atol=0)
    np.testing.assert_allclose(np.asarray(np.tril(np.asarray(S))), 0.0, atol=0)
    for t in range(PETRI_T):
        np.testing.assert_allclose(float(S[2 * t, 2 * t + 1]), dt, atol=0)


def test_petri_potential_emerges_from_composition():
    """The total forcing -d(-sum_t eps'_t f_t(x)) is assembled by compose_seq.

    It is written nowhere by hand: each box knows only its own rate constant
    and its own reactant multiplicities.
    """
    arr = petri_arrangement(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt=0.1)

    def U(s):
        x = petri_concentrations(PETRI_M, PETRI_N, PETRI_X0, s)
        return -jnp.sum(s[1::2] * _petri_fluxes(x))

    rng = np.random.default_rng(3)
    for _ in range(5):
        s = jnp.asarray(rng.uniform(-0.3, 0.3, size=2 * PETRI_T))
        np.testing.assert_allclose(
            np.asarray(_omega_q(arr, s)), np.asarray(-jax.grad(U)(s)), rtol=1e-6
        )


def test_petri_boxes_are_local():
    """Each transition box sees only the species it touches -- nothing global."""
    incid = petri_incidences(PETRI_M, PETRI_N)
    assert incid == [[0, 1, 2], [0, 1, 2], [2, 3], [2, 3], [0, 1, 3]]
    for t, St in enumerate(incid):
        box = petri_transition_box(PETRI_M[t, jnp.asarray(St, dtype=int)], float(PETRI_R[t]))
        assert (box.in_dim_N, box.out_dim_N) == (len(St), 1)
        assert box.P.Q_dim == 2
    wire = petri_wire(PETRI_M, PETRI_N, PETRI_X0)
    assert (wire.out_dim_M, wire.in_dim_M) == (PETRI_T, sum(len(St) for St in incid))
    assert wire.P.Q_dim == 0  # static: stateless and (below) forcing-free
    xi_q, xi_m, xi_n = wire.omega(NONE, jnp.zeros(PETRI_T), NONE)
    assert float(jnp.sum(jnp.abs(xi_q)) + jnp.sum(jnp.abs(xi_m)) + jnp.sum(jnp.abs(xi_n))) == 0.0


def test_petri_moiety_conservation_is_structural():
    """c . x stays c . x0 for every conservation law c (Z c = 0), at every tick.

    The state is the extent vector and x is recomputed as x0 + Z^T eps, so this
    holds by construction rather than by the integrator being accurate.
    """
    Z = PETRI_N - PETRI_M
    c = jnp.array([0.5, 1.5, 2.0, 1.0])
    np.testing.assert_allclose(np.asarray(Z @ c), np.zeros(PETRI_T), atol=1e-12)
    dt = 0.05  # deliberately coarse: accuracy is irrelevant to the claim
    O = petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt)
    s = petri_initial_state(PETRI_T)
    for _ in range(200):
        s = petri_step(O, s)
        x = petri_concentrations(PETRI_M, PETRI_N, PETRI_X0, s)
        np.testing.assert_allclose(
            float(jnp.dot(c, x)), float(jnp.dot(c, PETRI_X0)), rtol=1e-6
        )


def test_petri_malformed_nets_rejected():
    """Shapes, integrality, and signs are checked rather than silently absorbed."""
    with pytest.raises(ValueError):  # m, n shapes disagree
        petri_arrangement(PETRI_M, PETRI_N[:, :3], PETRI_R, PETRI_X0)
    with pytest.raises(ValueError):  # wrong number of rate constants
        petri_arrangement(PETRI_M, PETRI_N, PETRI_R[:2], PETRI_X0)
    with pytest.raises(ValueError):  # non-integer arc multiplicity
        petri_arrangement(PETRI_M.at[0, 0].set(0.5), PETRI_N, PETRI_R, PETRI_X0)
    with pytest.raises(ValueError):  # negative rate constant
        petri_arrangement(PETRI_M, PETRI_N, PETRI_R.at[0].set(-1.0), PETRI_X0)


# ---------------------------------------------------------------------------
# (v) The log-coordinate Petri encoding (module docstring, 5).
# ---------------------------------------------------------------------------

PETRI_LOG_STEP = log_mass_action_step  # alias for readability below


def test_petri_log_one_step_exact():
    """Keystone: one ``Phi`` tick = x * exp(dt * (Z^T f)(x) / x), from moving states."""
    for dt in (1.0, 0.05, 0.001):
        O = petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt)
        ref = PETRI_LOG_STEP(PETRI_M, PETRI_N, PETRI_R, dt)
        s = petri_log_initial_state(PETRI_X0)
        for _ in range(6):
            x = petri_log_concentrations(s)
            s = petri_step(O, s)
            np.testing.assert_allclose(
                np.asarray(petri_log_concentrations(s)),
                np.asarray(ref(x)),
                rtol=1e-6,
            )


def test_petri_log_positive_where_euler_overshoots():
    """Stiff decay A -> 0 at rate 3, dt = 0.5: plain forward Euler leaves the
    positive orthant on the FIRST step (x(1 - 3 dt) < 0); the log tick is
    x e^{-3 dt}: positive, finite, and monotone to the equilibrium, at any dt."""
    m1, n1 = jnp.array([[1.0]]), jnp.array([[0.0]])
    r1, x01 = jnp.array([3.0]), jnp.array([1.0])
    dt = 0.5
    euler = x01 + dt * mass_action_rhs(m1, n1, r1, jnp.zeros(1))(x01)
    assert bool(jnp.any(euler < 0))  # encoding 4's chemical reading fails here
    O = petri_log_dynamics(m1, n1, r1, dt)
    s = petri_log_initial_state(x01)
    prev = float(x01[0])
    for _ in range(100):
        s = petri_step(O, s)
        x = float(petri_log_concentrations(s)[0])
        assert jnp.isfinite(x) and 0 < x < prev
        prev = x
    np.testing.assert_allclose(prev, 0.0, atol=1e-6)


def test_petri_log_positivity_is_not_stability():
    """The honesty frontier: on the 4-species net at dt = 1 the log encoding
    does NOT rescue the integration -- each tick is positive, but the exponent
    (Z^T f)/x grows without bound and the iteration blows up (in floats,
    overflowing to inf and underflowing to 0). Positivity by construction
    removes only the orthant-exit failure mode; dt must still be small for
    stability and accuracy."""
    dt = 1.0
    assert bool(jnp.any(PETRI_X0 + dt * PETRI_RHS(PETRI_X0) < 0))
    O = petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt)
    s = petri_log_initial_state(PETRI_X0)
    for _ in range(20):
        s = petri_step(O, s)
    x = petri_log_concentrations(s)
    assert not bool(jnp.all(jnp.isfinite(jnp.log(x))))  # degenerate: 0 or inf


def test_petri_log_converges_to_mass_action():
    """Over a fixed horizon the log-Euler trajectory converges to the
    mass-action flow as dt -> 0 (first-order scheme: error shrinks with dt)."""
    horizon = 0.3
    x_ref = PETRI_X0
    dt_ref = 5e-4
    for _ in range(int(round(horizon / dt_ref))):
        x_ref = x_ref + dt_ref * PETRI_RHS(x_ref)

    errs = {}
    for dt in (0.05, 0.01):
        O = petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt)
        s = petri_log_initial_state(PETRI_X0)
        for _ in range(int(round(horizon / dt))):
            s = petri_step(O, s)
        errs[dt] = float(jnp.max(jnp.abs(petri_log_concentrations(s) - x_ref)))
    assert errs[0.01] < errs[0.05]
    assert errs[0.01] < 0.05


def test_petri_log_yprime_frozen_and_irrelevant():
    """y' never moves and its value never reaches the y-dynamics: the forcing's
    y'-derivative is y'-free, and the exp Jacobian only feeds the discarded
    xi-slot."""
    dt = 0.01
    O = petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt)
    s1 = petri_log_initial_state(PETRI_X0, y_prime=1.0)
    s2 = petri_log_initial_state(PETRI_X0, y_prime=-3.7)
    for _ in range(50):
        s1, s2 = petri_step(O, s1), petri_step(O, s2)
    np.testing.assert_array_equal(np.asarray(s1[1::2]), np.full(4, 1.0))
    np.testing.assert_array_equal(np.asarray(s2[1::2]), np.full(4, -3.7))
    np.testing.assert_array_equal(np.asarray(s1[0::2]), np.asarray(s2[0::2]))


def test_petri_log_conservation_drift_is_dual_to_extent_encoding():
    """The trade-off, measured: encoding 4 conserves c . x structurally
    (test_petri_moiety_conservation_is_structural); here c . x drifts, and
    the drift shrinks as dt -> 0 over a fixed horizon."""
    Z = PETRI_N - PETRI_M
    c = jnp.array([0.5, 1.5, 2.0, 1.0])
    np.testing.assert_allclose(np.asarray(Z @ c), np.zeros(PETRI_T), atol=1e-12)
    horizon = 1.0
    drift = {}
    for dt in (0.2, 0.02):
        O = petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt)
        s = petri_log_initial_state(PETRI_X0)
        for _ in range(int(round(horizon / dt))):
            s = petri_step(O, s)
        x = petri_log_concentrations(s)
        drift[dt] = abs(float(jnp.dot(c, x)) - float(jnp.dot(c, PETRI_X0)))
    assert drift[0.2] > 1e-6  # it genuinely drifts (no structural conservation)
    assert drift[0.02] < drift[0.2]


def test_petri_log_potential_emerges_from_composition():
    """The total forcing -d(-sum_t f_t(x) sum_s zeta_ts y'_s / x_s) is assembled
    by compose_seq; each transition box knows only its own arc data and rate."""
    arr = petri_log_arrangement(PETRI_M, PETRI_N, PETRI_R, dt=0.1)
    Z = PETRI_N - PETRI_M

    def U(s):
        y, yp = s[0::2], s[1::2]
        x = jnp.exp(y)
        return -jnp.sum(_petri_fluxes(x) * (Z @ (yp / x)))

    rng = np.random.default_rng(7)
    for _ in range(5):
        y = jnp.asarray(rng.uniform(-0.5, 0.5, size=4))
        yp = jnp.asarray(rng.uniform(-1.0, 1.0, size=4))
        s = jnp.stack([y, yp], axis=1).reshape(-1)
        np.testing.assert_allclose(
            np.asarray(_omega_q(arr, s)), np.asarray(-jax.grad(U)(s)), rtol=1e-6
        )


def test_petri_log_wire_is_prism_and_boxes_are_local():
    """The wire is a port bijection; species boxes carry no reaction data
    (not even x0); transitions are stateless; the composite sharp is the
    species' block-diagonal nilpotent."""
    degrees, perm = petri_log_incidences(PETRI_M, PETRI_N)
    assert degrees == [3, 3, 4, 3]
    assert sorted(perm) == list(range(2 * sum(degrees)))  # bijection
    wire = petri_log_wire(PETRI_M, PETRI_N)
    assert (wire.out_dim_M, wire.in_dim_M) == (2 * sum(degrees), 2 * sum(degrees))
    assert wire.P.Q_dim == 0

    sp = petri_log_species_box(degree=3, dt=0.1)
    assert (sp.in_dim_N, sp.out_dim_N) == (0, 6)
    assert sp.P.Q_dim == 2  # (y, y'), and nothing else -- no rates, no x0
    tr = petri_log_transition_box(jnp.array([1.0, 1.0]), jnp.array([-1.0, 1.0]), 0.5)
    assert tr.P.Q_dim == 0  # stateless

    arr = petri_log_arrangement(PETRI_M, PETRI_N, PETRI_R, dt=0.1)
    assert arr.P.Q_dim == 8  # 2 per species; transitions contribute none
    S = step_matrix(arr.P, petri_log_initial_state(PETRI_X0))
    np.testing.assert_allclose(np.asarray(S @ S), np.zeros_like(np.asarray(S)), atol=0)


def test_petri_log_x0_validation():
    """The chart is y = log x: zero or negative initial concentrations are
    rejected here, and petri_check rejects negative x0 in encoding 4 too."""
    with pytest.raises(ValueError):
        petri_log_initial_state(jnp.array([1.0, 0.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        petri_log_initial_state(jnp.array([1.0, -0.5, 1.0, 1.0]))
    with pytest.raises(ValueError):
        petri_arrangement(PETRI_M, PETRI_N, PETRI_R, PETRI_X0.at[0].set(-1.0))


# ---------------------------------------------------------------------------
# 6. The DIRECT non-exact realization (module docstring, 6; DESIGN.md §6(c)).
# The trick of 1 moves out of the sharp and into ``omega``.
# ---------------------------------------------------------------------------


def test_direct_agrees_with_trick_step_for_step():
    """On a detailed-balanced network the DIRECT realization (plain
    ``configuration(dt*I)``, forcing the mass-action field by hand) and the
    TRICK realization (Onsager sharp, exact forcing ``-dF``) are the same
    dynamics, step for step, on both fixture networks: single steps from random
    positive points at both step sizes, and a whole trajectory at the ``dt``
    that keeps the trick realization inside the positive orthant (at ``dt = 1``
    the Euler step leaves it after a few steps and the Onsager sharp's
    ``log`` -- not the direct realization -- goes ``nan``)."""
    rng = np.random.default_rng(11)
    for alpha, beta, kf, kb, x_eq, dts in (
        (CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, (1.0, 0.05)),
        (BI_ALPHA, BI_BETA, BI_KF, BI_KB, BI_XEQ, (1.0, 0.02)),
    ):
        for dt in dts:
            O_trick = crn_dynamics(alpha, beta, kf, kb, x_eq, dt)
            O_direct = mass_action_dynamics(alpha, beta, kf, kb, dt)
            for _ in range(5):  # single steps from random positive points
                x = _random_positive(rng, 3)
                np.testing.assert_allclose(
                    np.asarray(crn_step(O_direct, x)), np.asarray(crn_step(O_trick, x)),
                    rtol=1e-12, atol=0.0,
                )
        dt = dts[1]
        O_trick = crn_dynamics(alpha, beta, kf, kb, x_eq, dt)
        O_direct = mass_action_dynamics(alpha, beta, kf, kb, dt)
        x = _random_positive(rng, 3)
        for _ in range(20):  # step for step along one trajectory
            x_trick, x_direct = crn_step(O_trick, x), crn_step(O_direct, x)
            np.testing.assert_allclose(
                np.asarray(x_direct), np.asarray(x_trick), rtol=1e-12, atol=0.0
            )
            x = x_trick


def test_direct_runs_what_the_no_go_excludes():
    """The DIRECT realization has no detailed-balance hypothesis, so it runs the
    networks §1 rejects and the §3 obstruction excludes: the
    Wegscheider-violating triangle (``crn_arrangement`` raises) and the
    irreversible A+B->C. Its forcing is genuinely NOT ``dU``: the Jacobian of
    the mass-action field is non-symmetric, so no potential has it as a
    gradient."""
    edges = [(0, 1), (1, 2), (2, 0)]
    kf, kb = [1.0, 1.0, 1.0], [1.0, 1.0, 2.0]  # cycle affinity != 0
    alpha, beta = unimolecular_stoichiometry(3, edges)
    with pytest.raises(ValueError):                       # the trick cannot
        crn_arrangement(alpha, beta, kf, kb, jnp.ones(3), dt=0.1)

    dt = 0.01
    rhs = mass_action_rhs(alpha, beta, kf, kb)
    O = mass_action_dynamics(alpha, beta, kf, kb, dt)     # the direct one can
    x = jnp.array([1.0, 0.3, 2.0])
    for _ in range(200):
        expected = x + dt * rhs(x)
        x = crn_step(O, x)
        np.testing.assert_allclose(np.asarray(x), np.asarray(expected), rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(float(jnp.sum(x)), 3.3, rtol=1e-9)  # mass conserved

    # Irreversible A + B -> C: kb = 0, which the Onsager log-mean cannot take.
    with pytest.raises(ValueError):
        crn_arrangement(BI_ALPHA, BI_BETA, BI_KF, [0.0], BI_XEQ, dt=0.1)
    rhs_irrev = mass_action_rhs(BI_ALPHA, BI_BETA, BI_KF, [0.0])
    arr = mass_action_arrangement(BI_ALPHA, BI_BETA, BI_KF, [0.0], dt=0.01)
    O2 = mass_action_dynamics(BI_ALPHA, BI_BETA, BI_KF, [0.0], dt=0.01)
    x = jnp.array([2.0, 1.2, 0.3])
    for _ in range(200):
        expected = x + 0.01 * rhs_irrev(x)
        x = crn_step(O2, x)
        np.testing.assert_allclose(np.asarray(x), np.asarray(expected), rtol=0.0, atol=1e-14)

    # Not exact: d omega != 0, i.e. the field's Jacobian is not symmetric.
    J = np.asarray(jax.jacobian(lambda q: _omega_q(arr, q))(jnp.array([1.0, 2.0, 0.7])))
    assert not np.allclose(J, J.T)


# ---------------------------------------------------------------------------
# Regression against CDLM (``regress/reaction_network.npz``, DESIGN.md §3).
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """Every distinct dynamics of the old test file, reproduced to rtol 1e-10.

    Closed systems throughout, so no environment covector is translated; the
    recorded composite potential gradients ``*_grad_U`` are CDLM's ``dU`` and
    the new forcing is their negative (DESIGN.md §3).
    """
    R = load_regress("reaction_network")
    tol = dict(rtol=1e-10, atol=1e-12)

    def one_steps(make_O, dts, xs):
        got, k = [], 0
        for dt in dts:
            O = make_O(float(dt))
            for _ in range(5):
                got.append(crn_step(O, jnp.asarray(xs[k])))
                k += 1
        return np.stack([np.asarray(v) for v in got])

    def run(O, s, T):
        traj = [np.asarray(s)]
        for _ in range(T):
            s = crn_step(O, s)
            traj.append(np.asarray(s))
        return np.stack(traj)

    # 1. monolithic (trick) one-step, both networks, both dt.
    np.testing.assert_allclose(
        one_steps(lambda dt: crn_dynamics(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt),
                  R["mono_chain_dts"], R["mono_chain_x"]), R["mono_chain_new"], **tol)
    np.testing.assert_allclose(
        one_steps(lambda dt: crn_dynamics(BI_ALPHA, BI_BETA, BI_KF, BI_KB, BI_XEQ, dt),
                  R["mono_bi_dts"], R["mono_bi_x"]), R["mono_bi_new"], **tol)

    # 1. the two long convergence trajectories.
    np.testing.assert_allclose(
        run(crn_dynamics(CHAIN_ALPHA, CHAIN_BETA, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, 0.05),
            jnp.array([3.0, 0.2, 1.0]), 6000), R["conv_chain_traj"], **tol)
    np.testing.assert_allclose(
        run(crn_dynamics(BI_ALPHA, BI_BETA, BI_KF, BI_KB, BI_XEQ, 0.02),
            jnp.array([2.0, 1.2, 0.3]), 8000), R["conv_bi_traj"], **tol)

    # 2. compositional unimolecular: steps, and the emergent forcing (negated dU).
    np.testing.assert_allclose(
        one_steps(lambda dt: unimolecular_dynamics(3, CHAIN_EDGES, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, dt),
                  R["uni_dts"], R["uni_x"]), R["uni_new"], **tol)
    arr = unimolecular_arrangement(3, CHAIN_EDGES, CHAIN_KF, CHAIN_KB, CHAIN_XEQ, 0.1)
    np.testing.assert_allclose(
        np.stack([np.asarray(_omega_q(arr, jnp.asarray(x))) for x in R["uni_grad_x"]]),
        -R["uni_grad_U"], **tol)

    # 4. Petri (extent) encoding: three dt, the long run, the eps' = -3.7 run,
    #    and the emergent forcing.
    for dt in (1.0, 0.05, 0.001):
        np.testing.assert_allclose(
            run(petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt), petri_initial_state(PETRI_T), 6),
            R[f"petri_traj_{dt:g}"], **tol)
    np.testing.assert_allclose(
        run(petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, 0.01), petri_initial_state(PETRI_T), 300),
        R["petri_long_traj"], **tol)
    np.testing.assert_allclose(
        run(petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, 0.01),
            petri_initial_state(PETRI_T, eps_prime=-3.7), 50), R["petri_epsp_traj"], **tol)
    arr = petri_arrangement(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, dt=0.1)
    np.testing.assert_allclose(
        np.stack([np.asarray(_omega_q(arr, jnp.asarray(s))) for s in R["petri_grad_s"]]),
        -R["petri_grad_U"], **tol)

    # 5. log-Petri encoding: three dt, the stiff decay, the two convergence runs,
    #    the y' = -3.7 run, and the emergent forcing.
    for dt in (1.0, 0.05, 0.001):
        np.testing.assert_allclose(
            run(petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt), petri_log_initial_state(PETRI_X0), 6),
            R[f"plog_traj_{dt:g}"], **tol)
    m1, n1, r1, x01 = jnp.array([[1.0]]), jnp.array([[0.0]]), jnp.array([3.0]), jnp.array([1.0])
    np.testing.assert_allclose(
        run(petri_log_dynamics(m1, n1, r1, 0.5), petri_log_initial_state(x01), 100),
        R["plog_stiff_traj"], **tol)
    for dt in (0.05, 0.01):
        np.testing.assert_allclose(
            run(petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, dt),
                petri_log_initial_state(PETRI_X0), int(round(0.3 / dt))), R[f"plog_conv_{dt:g}"], **tol)
    np.testing.assert_allclose(
        run(petri_log_dynamics(PETRI_M, PETRI_N, PETRI_R, 0.01),
            petri_log_initial_state(PETRI_X0, y_prime=-3.7), 50), R["plog_yp_traj"], **tol)
    arr = petri_log_arrangement(PETRI_M, PETRI_N, PETRI_R, dt=0.1)
    np.testing.assert_allclose(
        np.stack([np.asarray(_omega_q(arr, jnp.asarray(s))) for s in R["plog_grad_s"]]),
        -R["plog_grad_U"], **tol)


def test_flat_state_round_trip():
    """The composite's state is the tensor's nested tuple; ``flat_state`` is the
    flat vector CDLM's ``direct_sum`` used, and it is what the helpers take."""
    O = petri_dynamics(PETRI_M, PETRI_N, PETRI_R, PETRI_X0, 0.01)
    assert isinstance(O.state, tuple)                      # nested, not flat
    np.testing.assert_allclose(np.asarray(flat_state(O.state)), np.zeros(2 * PETRI_T))
    s = petri_step(O, petri_initial_state(PETRI_T))
    assert np.asarray(s).shape == (2 * PETRI_T,)           # helpers stay flat
