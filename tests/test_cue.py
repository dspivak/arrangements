"""A transient cue selects an autonomous commitment (the attention paper).

The mathematics is ``background/affine_quadratic_operad.tex`` in the book repository,
cited by its own labels (eq:context-preference, eq:cue-parent-cost, eq:autonomous-cost,
eq:context-dynamics, prop:capture, eq:capture-condition, eq:short-cue, and the numerical
table at the end). **Every docstring below says which of three things its test does** --
reproduce a closed form the paper proves, check a discrete-time analogue the book still
has to prove (STATUS.md §3.2), or merely illustrate (STATUS.md §3.2).

The parent is the rendezvous parent with the requested separation promoted to an internal
context coordinate: ``Q_* = P^2 x R``, configuration ``(x_1, x_2, g)``, same placement,
same transfers, same subsidiaries (imported from tests/test_rendezvous.py, where they are
built and explained). Only the cost changes, to eq:cue-parent-cost.

**The paper's table is RK4 of the ODE at step 1e-3; the book's dynamics is the discrete
displacement map** (STATUS.md §1). So no test here asserts the table's digits of a
discrete run. What is asserted is: the capture condition holds at the end of the cue; the
autonomous run afterwards reaches the same limit; ``V`` decreases at every step under a
mobility below ``2/L``; and the discrete trajectory approaches the table's sampled values
as the step shrinks, the observed differences being recorded in the docstrings.

The discrete map has no time step separate from its mobilities: one covector gives one
displacement, so the mobility *is* the step. The paper's "all five mobilities equal to
``1``" therefore becomes, here, all five mobilities equal to ``h``, with the prescribed
cue waveform sampled at ``t_k = k h`` -- explicit Euler on eq:meeting-dynamics and
eq:context-dynamics. That correspondence is a thing the book has to state; it is not in
the paper, whose mobilities and time step are independent.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.attention import attention_arrangement, identity_transfers, make_input, message
from arrangements.coalgebra import configuration
from arrangements.rvect import diagonal
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_rendezvous import (
    DH, DP, EDIM, MDIM, exterior, run_closed, run_phi, subsidiary,
)

LAM = 64.0            # the paper's lambda; the capture threshold is lambda/64 = 1
T_CUE = 0.2           # the cue duration
AMP = -12.0           # the cue amplitude: u(t) = -12 b_0(t / 0.2)
REST = (1.0, 0.0, 1.0, 0.0, 1.0)      # the resting state (g, z_1, z_2, x_1, x_2)

# The paper's table (RK4 of the ODE at step 1e-3), (g, z_1, z_2, x_1, x_2) by time.
TABLE = {
    0.0: (1.0, 0.0, 1.0, 0.0, 1.0),
    0.2: (-0.0363, 0.0984, 0.9016, 0.0057, 0.9943),
    1.0: (0.0103, 0.3386, 0.6614, 0.1498, 0.8502),
    5.0: (0.0009, 0.4850, 0.5150, 0.4646, 0.5354),
}


# ---------------------------------------------------------------------------
# The preference cost, the cue waveform, and the parent arrangement.
# ---------------------------------------------------------------------------


def phi_lam(g, lam=LAM):
    """eq:context-preference: ``phi_lambda(g) = (lambda/4) g^2 (1 - g)^2``, whose minima
    ``g = 0`` and ``g = 1`` encode the two supplied modes, meeting and separation by one
    unit."""
    return 0.25 * lam * g ** 2 * (1.0 - g) ** 2


def bump(s):
    """The paper's smooth bump ``b_0(s) = exp(1 - [4 s (1 - s)]^{-1})`` on ``(0, 1)``,
    zero outside; nonnegative, smooth, supported in ``[0, 1]`` (eq:short-cue's ``b``)."""
    s = jnp.asarray(s, float)
    inside = (s > 0.0) & (s < 1.0)
    safe = jnp.where(inside, s, 0.5)
    return jnp.where(inside, jnp.exp(1.0 - 1.0 / (4.0 * safe * (1.0 - safe))), 0.0)


def cue(t):
    """``u(t) = -12 b_0(t / 0.2)``, the paper's concrete cue: zero before ``0`` and after
    ``T = 0.2``."""
    return AMP * bump(t / T_CUE)


def cue_parent_cost(q, m_out, n_in, lam=LAM):
    """eq:cue-parent-cost on ``Q_* x D_2``:

        k_*((x, g), m_1, m_2, ((u, w_0), p_0))
            = (1/2)(g - (h_2 - h_1))^2 + phi_lambda(g) + (1/2) u^2 - u g + 2.

    The cue tilts the cost in the context coordinate; the terms ``u^2/2 + 2`` make it
    nonnegative without changing the internal force that tilt produces. At each
    configuration it is quadratic in the interaction variables, and the quartic term
    depends only on the configuration, so this is an operation of ``A_B``."""
    g, u = q[2], n_in[0]
    r = g - (m_out[MDIM] - m_out[0])
    return 0.5 * r ** 2 + phi_lam(g, lam) + 0.5 * u ** 2 - u * g + 2.0


def parent_cue(x0=(0.0, 1.0), g0=1.0, mobilities=(1.0, 1.0, 1.0)):
    """``a_* in A_B(2)``: ``Q_* = P^2 x R``, ``q = (x_1, x_2, g)``, placement
    ``l_i(x, g) = x_i`` and transfers ``T_ij = M`` exactly as in the rendezvous (the
    sending and receiving maps are eq:meeting-maps, independent of ``g``), and cost
    eq:cue-parent-cost. ``mobilities`` are ``(beta_1, beta_2, eta)``."""
    return attention_arrangement(
        2, DH, DP,
        placement=lambda q: jnp.reshape(q[:2], (2, DP)),
        transfers=identity_transfers(2, DH),
        cost=cue_parent_cost,
        parameter=configuration(diagonal(jnp.asarray(mobilities, float)),
                                init=jnp.asarray([x0[0], x0[1], g0], float)),
        label="a_*",
    )


def cue_system(state=REST, mobility=0.01):
    """``a_*(S_1, S_2) : I -> B``, by composition in ``arr``; configuration
    ``(z_1, z_2, x_1, x_2, g)`` in that order. ``state`` is ``(g, z_1, z_2, x_1, x_2)``,
    the paper's ordering. All five mobilities equal ``mobility``, which is the step."""
    g0, z1, z2, x1, x2 = state
    parts = [subsidiary(z1, mobility, "S1"), subsidiary(z2, mobility, "S2")]
    return compose_seq(tensor_arrangements(parts),
                       parent_cue((x1, x2), g0, (mobility, mobility, mobility)).arr())


def V(y, lam=LAM):
    """eq:autonomous-cost, ``V = K(g, z, x) + phi_lambda(g)``, on the composite's
    configuration ``y = (z_1, z_2, x_1, x_2, g)``. The substituted cost is
    ``V + u^2/2 - u g + 2``; after the cue, ``u = 0`` and the internal dynamics are
    gradient descent on ``V``."""
    z1, z2, x1, x2, g = y[0], y[1], y[2], y[3], y[4]
    r = g - (z2 - z1)
    return (0.5 * r ** 2 + 0.5 * (x1 - z1) ** 2 + 0.5 * (x2 - z2) ** 2 + phi_lam(g, lam))


def paper_order(y):
    """The composite's ``(z_1, z_2, x_1, x_2, g)`` as the table's ``(g, z_1, z_2, x_1, x_2)``."""
    return np.asarray(y)[..., [4, 0, 1, 2, 3]]


def run_cue(mobility, until, state=REST):
    """The discrete run from ``state`` to time ``until`` with all mobilities equal to the
    step ``mobility``, driving the prescribed cue at ``t_k = k * mobility``."""
    steps = int(round(until / mobility))
    b = cue_system(state, mobility)
    return run_closed(b, steps,
                      lambda k: make_input(message(float(cue(k * mobility)), 1.0), 0.0))


_Vv = jax.jit(jax.vmap(V))
_Hv = jax.jit(jax.vmap(jax.hessian(V)))


# ---------------------------------------------------------------------------
# The cost: nonnegativity, and the cue covector in the forcing.
# ---------------------------------------------------------------------------


def test_cue_cost_nonnegativity_decomposition():
    """REPRODUCES A CLOSED FORM the paper proves (the display after eq:cue-parent-cost):

        phi_lambda(g) + u^2/2 - u g + 2 = (1/2)(u - g)^2 + phi_lambda(g) - g^2/2 + 2 >= 0,

    an identity in ``(g, u)``, together with the two-case bound behind it: for
    ``|g| <= 2`` the inequality follows from ``phi_lambda >= 0`` (and the ``+2``), and for
    ``|g| >= 2`` from ``phi_lambda(g) >= lambda g^4 / 16 >= g^2 / 2`` when
    ``lambda >= 2``. Both cases are checked over their ranges, and the whole
    eq:cue-parent-cost is checked nonnegative, as def:arrangement requires."""
    r = np.random.default_rng(401)
    for _ in range(200):
        g, u = r.standard_normal(2) * 3.0
        lhs = float(phi_lam(g) + 0.5 * u ** 2 - u * g + 2.0)
        rhs = float(0.5 * (u - g) ** 2 + phi_lam(g) - 0.5 * g ** 2 + 2.0)
        np.testing.assert_allclose(lhs, rhs, atol=1e-12)
        assert lhs >= 0.0

    for lam in (2.0, LAM):
        for g in np.linspace(-2.0, 2.0, 81):
            assert float(phi_lam(g, lam)) - 0.5 * g ** 2 + 2.0 >= 0.0
        for g in np.concatenate([np.linspace(-20.0, -2.0, 41), np.linspace(2.0, 20.0, 41)]):
            assert float(phi_lam(g, lam)) >= lam * g ** 4 / 16.0 - 1e-12
            assert lam * g ** 4 / 16.0 >= 0.5 * g ** 2 - 1e-12

    a = parent_cue()
    for _ in range(50):
        q = jnp.asarray(r.standard_normal(3) * 2.0)
        m = jnp.asarray(r.standard_normal(2 * MDIM) * 2.0)
        e = jnp.asarray(r.standard_normal(EDIM) * 2.0)
        assert float(a.cost(q, m, e)) >= 0.0


def test_the_cue_covector_and_the_context_dynamics_are_in_the_forcing():
    """REPRODUCES A CLOSED FORM the paper states (eq:context-dynamics and the sentence
    after it): the composite's forcing has, on the context coordinate,

        -d_g (V + u^2/2 - u g) = -r - phi_lambda'(g) + u,

    which is eq:context-dynamics' bracket, and on the exterior message it has "the cue
    covector ``(g - u) du``". The cue waveform is prescribed (it is supplied as the
    exterior input at each step) while the context ``g`` remains dynamic, so that covector
    is published upstream and drives nothing here."""
    b = cue_system(mobility=1.0)
    dphi = jax.grad(phi_lam)
    r_ = np.random.default_rng(402)
    for _ in range(6):
        z1, z2, x1, x2, g, u = r_.standard_normal(6)
        q = jnp.asarray([z1, z2, x1, x2, g])
        xi_q, _, xi_e = b.omega(q, jnp.zeros(0), exterior(u))
        r_val = g - (z2 - z1)
        np.testing.assert_allclose(float(xi_q[4]),
                                   -r_val - float(dphi(g)) + u, atol=1e-12)
        np.testing.assert_allclose(float(xi_e[0]), g - u, atol=1e-12)   # (g - u) du
        # the other four are the rendezvous forces of eq:meeting-dynamics, unchanged
        np.testing.assert_allclose(np.asarray(xi_q)[:4],
                                   [-r_val + x1 - z1, r_val + x2 - z2,
                                    z1 - x1, z2 - x2], atol=1e-12)


def test_the_substituted_cost_is_V_plus_the_cue_terms():
    """REPRODUCES A CLOSED FORM the paper states (the line after eq:autonomous-cost): with
    reference position ``p_0 = 0``, "The substituted cost is ``V + u^2/2 - u g + 2``".
    The composite is built by substitution and never handed ``V``; what is checked is that
    its whole configuration forcing is ``-dV`` once ``u = 0``, and ``-d(V - u g)`` in
    general."""
    b = cue_system(mobility=1.0)
    gradV = jax.grad(V)
    r_ = np.random.default_rng(403)
    for u in (0.0, -3.7):
        for _ in range(4):
            q = jnp.asarray(r_.standard_normal(5))
            xi_q, _, _ = b.omega(q, jnp.zeros(0), exterior(u))
            expected = -np.asarray(gradV(q))
            expected[4] += u                 # the only effect of the cue on the state
            np.testing.assert_allclose(np.asarray(xi_q), expected, atol=1e-12)


def test_the_resting_state_has_zero_cost_and_is_a_critical_point():
    """REPRODUCES A CLOSED FORM the paper's "Finite cues suffice" paragraph relies on:
    the resting state ``(g, z_1, z_2, x_1, x_2) = (1, 0, 1, 0, 1)`` has ``r = 0``,
    ``x_i = z_i`` and ``phi_lambda(1) = 0``, so ``V = 0`` and the autonomous forcing
    vanishes. Without a cue the discrete run does not move at all: the second mode is
    genuinely held, not merely approached."""
    y0 = jnp.asarray([0.0, 1.0, 0.0, 1.0, 1.0])
    np.testing.assert_allclose(float(V(y0)), 0.0, atol=1e-15)
    b = cue_system(mobility=0.01)
    tr = run_phi(b, 20, exterior(0.0))
    np.testing.assert_allclose(tr[-1], tr[0], atol=1e-15)


# ---------------------------------------------------------------------------
# prop:capture, discretely.
# ---------------------------------------------------------------------------


def test_capture_condition_holds_at_the_end_of_the_cue():
    """CHECKS THE DISCRETE ANALOGUE the book has to prove (STATUS.md §3.2) of
    eq:capture-condition, for the paper's concrete data: ``lambda = 64``, unit mobilities,
    ``u(t) = -12 b_0(t / 0.2)``, from the resting state ``(1, 0, 1, 0, 1)``. At ``T = 0.2``
    the discrete run has ``g(T) < 1/2`` and ``V(T) < lambda/64 = 1``, and the physical
    positions have barely changed.

    The paper's RK4 value is ``V(0.2) ~ 0.3836``. The discrete map at step ``h`` overshoots
    it, and converges to it from above as ``h`` shrinks: ``0.3980`` at ``h = 0.02``,
    ``0.3926`` at ``0.01``, ``0.3891`` at ``0.005``, ``0.3859`` at ``0.002``, ``0.3848`` at
    ``0.001``. None of those digits is asserted; what is asserted is the capture
    condition itself at each step size, and that the sequence decreases toward the paper's
    number. The margin is wide, but it is the *discrete* run's ``V`` that has to satisfy
    eq:capture-condition, not the ODE's: at a coarse enough step it would not."""
    vals = []
    for h in (0.02, 0.01, 0.005):
        tr = run_cue(h, T_CUE)
        y = jnp.asarray(tr[-1])
        g = float(y[4])
        Vt = float(V(y))
        assert g < 0.5                       # eq:capture-condition, first half
        assert Vt < LAM / 64.0               # eq:capture-condition, second half
        assert abs(float(y[2])) < 0.02 and abs(float(y[3]) - 1.0) < 0.02  # x barely moved
        vals.append(Vt)
    assert vals[0] > vals[1] > vals[2] > 0.3836


def test_the_autonomous_run_completes_the_rendezvous():
    """CHECKS THE DISCRETE ANALOGUE (§5.9 item 1) of prop:capture's conclusion: after the
    cue, ``g -> 0`` and ``z_1, z_2, x_1, x_2 -> c`` with ``c`` from eq:meeting-point.
    Here ``S = 0 + 1 + 0 + 1 = 2`` with equal mobilities, so ``c = 1/2`` and the limit is
    ``(0, 1/2, 1/2, 1/2, 1/2)``. Run to ``t = 40`` at step ``0.01``; the slowest transverse
    mode decays like ``exp(-(2 - sqrt 2) t)``, so at the paper's last sampled time
    ``t = 20`` the coordinates are still ``6e-6`` from the limit -- which is why the
    table's ``0.5000`` is a rounding and not an arrival."""
    tr = run_cue(0.01, 40.0)
    np.testing.assert_allclose(paper_order(tr[-1]), [0.0, 0.5, 0.5, 0.5, 0.5], atol=1e-9)

    at20 = paper_order(tr[int(round(20.0 / 0.01))])
    err20 = float(np.max(np.abs(at20 - np.array([0.0, 0.5, 0.5, 0.5, 0.5]))))
    assert 1e-6 < err20 < 1e-4


def test_V_decreases_at_every_step_after_the_cue_and_not_during_it():
    """CHECKS THE DISCRETE ANALOGUE (§5.9 item 1) of prop:capture's proof, whose
    continuous statement is ``Vdot <= 0`` after the cue. Discretely that is the descent
    lemma, and it needs the mobility below ``2/L`` (STATUS.md §3.2, third bullet);
    at ``h = 0.01`` and ``L ~ 40.3`` (see the next test) it holds, and ``V`` decreases at
    *every* one of the four thousand steps, strictly.

    During the cue it does not, and the paper does not claim it does: ``u`` is an external
    tilt, ``V`` climbs from ``0`` to about ``0.39``, and the capture condition is what
    replaces the descent argument over that interval. Both are asserted here, because a
    book that states the discrete descent lemma has to say where it does not apply."""
    h = 0.01
    tr = run_cue(h, 40.0)
    Vs = np.asarray(_Vv(jnp.asarray(tr)))
    k = int(round(T_CUE / h))

    assert np.all(np.diff(Vs[k:]) < 0.0)
    assert np.max(np.diff(Vs[:k + 1])) > 0.1      # it climbs while the cue is on
    np.testing.assert_allclose(Vs[0], 0.0, atol=1e-15)
    assert Vs[-1] < 1e-15


def test_gradient_lipschitz_constant_on_the_visited_region():
    """CHECKS THE DISCRETE ANALOGUE (§5.9 item 1) of what prop:capture needs and the paper
    does not: the cost is **not** quadratic -- ``phi_lambda`` is quartic -- so the
    discrete argument wants the descent lemma on a neighbourhood of the sublevel set, with
    each mobility below ``2/L``, ``L`` the Lipschitz constant of ``grad V`` there.

    ``L`` is estimated numerically as the largest spectral norm of ``grad^2 V`` over the
    states the run visits. Over the autonomous region -- the one prop:capture's descent
    argument is about -- ``L = 40.25`` at ``h = 0.01`` (``39.85`` at ``0.02``, ``40.36`` at
    ``0.005``), so ``2/L = 0.0497`` and the step used above sits inside it by a factor of
    five. Over the *whole* run the estimate is larger, ``61.93``, and it is attained in the
    middle of the cue at ``t = 0.16``, where ``g`` undershoots to ``-0.133``: the cue
    carries the state through a stiffer part of the quartic than anything the autonomous
    phase visits. ``2/61.93 = 0.0323`` is still above ``h``, so the whole run is inside the
    bound, but a book that states the discrete restatement has to pick which region it
    means.

    **The hypothesis is not asserted globally, and it is false globally**: ``grad^2 V``
    has the entry ``1 + phi_lambda''(g) = 33 - 192 g + 192 g^2`` at ``lambda = 64``,
    unbounded in ``g``, so any ``L`` is a statement about a region. That is checked too."""
    h = 0.01
    tr = run_cue(h, 40.0)
    norms = np.linalg.norm(np.asarray(_Hv(jnp.asarray(tr))), 2, axis=(1, 2))
    k = int(round(T_CUE / h))
    L_auto, L_all = float(norms[k:].max()), float(norms.max())

    assert 39.0 < L_auto < 42.0
    assert 60.0 < L_all < 65.0
    assert float(tr[:, 4].min()) < -0.1        # the undershoot that makes L_all larger
    assert h < 2.0 / L_all < 2.0 / L_auto
    # the same bound read the other way: the step used is a fifth of the critical one
    assert 2.0 / L_auto > 4.0 * h

    # not global: far out in g the Hessian norm is arbitrarily large
    far = jnp.asarray([[0.0, 0.0, 0.0, 0.0, gg] for gg in (10.0, 100.0)])
    big = np.linalg.norm(np.asarray(_Hv(far)), 2, axis=(1, 2))
    assert big[0] > 10.0 * L_all and big[1] > 90.0 * big[0]


def test_the_discrete_run_approaches_the_tables_values_as_the_step_shrinks():
    """CHECKS THE DISCRETE ANALOGUE (§5.9 item 1) against the paper's table, which is RK4
    of the ODE at step ``1e-3`` -- **the table's digits are not asserted of the discrete
    run**. What is asserted is first-order convergence to them: the discrete map with
    mobility ``h`` is explicit Euler on eq:meeting-dynamics and eq:context-dynamics, so
    halving ``h`` should halve the difference. Observed maximum differences over the five
    coordinates:

        t = 0.2:   1.205e-2 (h = 0.02),  5.984e-3 (0.01),  2.975e-3 (0.005)
        t = 1.0:   3.220e-3 (h = 0.02),  1.643e-3 (0.01),  8.400e-4 (0.005)
        t = 5.0:   1.960e-4 (h = 0.02),  1.061e-4 (0.01),  6.370e-5 (0.005)

    -- ratios ``0.497, 0.497`` and ``0.510, 0.511``, which is the first order. At
    ``t = 5`` the ratios are no longer clean (``0.54, 0.60``) because the table is quoted
    to four decimals, so its own rounding is ``5e-5`` and differences near ``1e-4`` are
    reading the printed digits rather than the discretization; only that the difference is
    below ``2e-4`` is asserted there. Continued to ``h = 0.001`` (not run here, it is
    twenty thousand steps) the ``t = 0.2`` difference is ``5.878e-4``."""
    steps = [0.02, 0.01, 0.005]
    runs = {h: run_cue(h, 5.0) for h in steps}
    for t in (0.2, 1.0):
        errs = []
        for h in steps:
            got = paper_order(runs[h][int(round(t / h))])
            errs.append(float(np.max(np.abs(got - np.array(TABLE[t])))))
        assert errs[1] < 0.6 * errs[0] and errs[2] < 0.6 * errs[1]
        assert errs[2] < 4e-3

    for h in steps:
        got = paper_order(runs[h][int(round(5.0 / h))])
        assert float(np.max(np.abs(got - np.array(TABLE[5.0])))) < 2e-4
