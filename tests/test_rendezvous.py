"""Rendezvous: "A coarse commitment acquires subsidiary plans" (the attention paper).

The mathematics is ``background/affine_quadratic_operad.tex`` in the book repository,
cited by its own labels (eq:meeting-maps, eq:subsidiary-cost, eq:parent-meeting-cost,
eq:meeting-cost, eq:meeting-dynamics, prop:meeting, eq:meeting-point). **Every docstring
below says which of three things its test does** -- reproduce a closed form the paper
proves, check a discrete-time analogue the book still has to prove (STATUS.md §3.2
item 1), or merely illustrate (STATUS.md §3.2).

The parent ``a_m in A_B(2)`` is an attention arrangement (pure placement transfers, a
private prediction cost on the requested separation). The two subsidiaries are *not*
operations of ``A_B``: the paper's ``S_i : I -> B`` are components, CDLM systems on the
monoidal unit interface, and an arity-zero operation of ``A_B`` would send the empty sum
``0_M`` rather than ``(z_i, 1)``. The paper says so explicitly ("The components being
arranged can be arbitrary CDLM systems ``I -> B`` ... its empty outgoing sum does not
restrict a component to emitting zero"). So each ``S_i`` is a plain ``Arrangement`` with
``M = <R^0|R^0>``, and the substitution of prop:operad is the composition in ``arr``

    compose_seq(tensor_arrangements([S_1, S_2]), a_m.arr()) : I -> B.

**eq:meeting-cost is never written into the composite.** ``K`` appears in this file only
as the closed form the composite is *checked against*, exactly as the harmonic potential
is in ``compose_graph`` (wiring.py) and the total potential is in ``hopfield.py``: the
composite's forcing comes from ``compose_seq``'s pullback-and-sum of the parent's
prediction cost and the two subsidiary costs.

The example-specific pieces -- the subsidiary's ``(1/2)(p - z)^2``, the parent's
``(1/2)(g - (h_2 - h_1))^2`` -- live here and not in ``dap/attention.py``: they are the
paper's modeling choices, not operad structure. ``tests/test_cue.py`` imports them.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    attention_arrangement, identity_transfers, input_dim, make_input, message, msg_dim,
)
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.rvect import diagonal, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

DH, DP = 1, 1                      # H = W = P = R (the paper's choice for this example)
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)     # dim M = 2, dim E = 3
IN_POS = (jnp.zeros(0), trivial_omega(0))       # the unit interface has nothing to send


# ---------------------------------------------------------------------------
# The components and the parent arrangement.
# ---------------------------------------------------------------------------


def subsidiary(z0, alpha, label="S"):
    """A subsidiary ``S_i : I -> B``: configuration ``z_i in P``, an *expected position*.

    It sends ``(z_i, 1) in M``; its receiving map to the one-point input space of ``I``
    is unique (``in_f`` is the bang); and its cost is eq:subsidiary-cost,
    ``k_i(z_i, (m, p)) = (1/2)(p - z_i)^2``, so it predicts the position coordinate of
    the input it receives. Mobility ``alpha`` for ``z_i``, as the paper's integrator
    choice ("a coordinate ``y`` with mobility ``mu`` responds to a force ``f dy`` with
    velocity ``ydot = mu f``"); in the book's discrete time one covector gives one
    displacement, so ``alpha`` is the whole step.
    """
    return Arrangement(
        P=configuration(euclidean(1, alpha), init=jnp.array([float(z0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda z, m_out: message(z, 1.0),
        in_f=lambda z, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda z, m_out, n_in: -0.5 * (n_in[MDIM] - z[0]) ** 2),
        label=label,
    )


def meeting_cost(q, m_out, n_in):
    """eq:parent-meeting-cost: ``k_m = (1/2)(g - (h_2 - h_1))^2``.

    The parent predicts the requested separation ``g`` -- the signal of the exterior
    message ``m_0 = (g, w_0)`` -- from the child signals. "The same formula defines the
    cost for any exterior weight, which it ignores", and this private prediction
    ``h_2 - h_1`` differs from the public sending map of eq:meeting-maps.
    """
    h1, h2 = m_out[0], m_out[MDIM]
    return 0.5 * (n_in[0] - (h2 - h1)) ** 2


def parent_meeting(x0=(0.0, 1.0), betas=(1.0, 1.0)):
    """``a_m in A_B(2)``: ``Q_m = P^2``, ``l_i(x) = x_i``, ``T_ij = M`` for ``i != j``,
    cost eq:parent-meeting-cost. Mobilities ``beta_i`` for the physical positions."""
    return attention_arrangement(
        2, DH, DP,
        placement=lambda x: jnp.reshape(x, (2, DP)),
        transfers=identity_transfers(2, DH),
        cost=meeting_cost,
        parameter=configuration(diagonal(jnp.asarray(betas, float)),
                                init=jnp.asarray(x0, float)),
        label="a_m",
    )


def rendezvous(z0=(0.0, 1.0), x0=(0.0, 1.0), alphas=(1.0, 1.0), betas=(1.0, 1.0)):
    """``a_m(S_1, S_2) : I -> B``, by composition in ``arr`` (prop:operad).

    The composite's configuration is ``(z_1, z_2, x_1, x_2)`` in that order, which is the
    order ``compose_seq`` gives the tensored parameter ``P_{S_1} (x) P_{S_2} (x) P_{a_m}``.
    """
    parts = [subsidiary(z0[i], alphas[i], f"S{i + 1}") for i in range(2)]
    return compose_seq(tensor_arrangements(parts), parent_meeting(x0, betas).arr())


def exterior(g, w_0=1.0, p_0=0.0):
    """The exterior input ``e = ((g, w_0), p_0) in E`` -- the maintained request."""
    return make_input(message(g, w_0), p_0)


def K_full(g, w_0, p_0, z1, z2, x1, x2):
    """eq:meeting-cost, with the reference position ``p_0`` left free.

    The paper writes ``K(g, z, x)`` after fixing ``p_0 = 0``; the composite cost is this
    function, which the subsidiaries make depend on ``p_0`` as well (their cost predicts
    the position they *receive*, ``p_0 + x_i``) and not at all on the exterior weight
    ``w_0``. This is the closed form the composite is checked against; nothing builds it.
    """
    r = g - (z2 - z1)
    return 0.5 * r ** 2 + 0.5 * (p_0 + x1 - z1) ** 2 + 0.5 * (p_0 + x2 - z2) ** 2


def K(g, z1, z2, x1, x2):
    """eq:meeting-cost at the paper's ``p_0 = 0``."""
    return K_full(g, 1.0, 0.0, z1, z2, x1, x2)


# ---------------------------------------------------------------------------
# Running a closed arrangement.
# ---------------------------------------------------------------------------


def run_phi(arr, steps, e):
    """``Phi(arr)`` run for ``steps`` steps on a constant exterior input ``e`` and zero
    exterior covector; returns the flattened configurations."""
    O = Phi(arr)
    s, out = O.state, [np.asarray(flat_state(O.state))]
    zero_xi = jnp.zeros(arr.out_dim_N)
    for _ in range(steps):
        _, _, s = O.with_state(s).run_one(IN_POS, lambda _o: (zero_xi, e))
        out.append(np.asarray(flat_state(s)))
    return np.array(out)


def run_closed(arr, steps, e_of_step):
    """The same run, with the covector triple specialized once and jitted.

    For a closed arrangement (``in_dim_M = out_dim_M = 0``, no incoming one-form) and a
    zero exterior covector ``xi_N``, the interpretation's triple
    ``xi_Q = (out_f)^* xi_N|_Q + (in_f)^* omega_M|_Q + omega|_Q`` collapses to its last
    term, so one step of ``Phi`` is ``s |-> u(s, omega(q(s), (), e)|_Q)``. That identity
    is not assumed here: ``test_the_jitted_closed_step_is_Phi`` checks the two agree
    exactly. The long runs below use this form because it is some fifty times faster.
    """
    P, omega, empty = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s, e):
        xi_q, _, _ = omega(P.position(s), empty, e)
        return P.step(s, xi_q)

    s, out = P.init, [np.asarray(flat_state(P.init))]
    for t in range(steps):
        s = step(s, e_of_step(t))
        out.append(np.asarray(flat_state(s)))
    return np.array(out)


def hessian_of_K():
    """``H``, the Hessian of ``K(0, -, -)`` in ``(z_1, z_2, x_1, x_2)`` (prop:meeting)."""
    return np.asarray(jax.hessian(lambda y: K(0.0, y[0], y[1], y[2], y[3]))(jnp.zeros(4)))


def spectrum(H, D):
    """The eigenvalues of ``D^{1/2} H D^{1/2}``, ``D = diag(mobilities)`` (STATUS.md §3.2)."""
    R = np.sqrt(np.diag(np.asarray(D, float)))
    return np.linalg.eigvalsh(R @ H @ R)


# ---------------------------------------------------------------------------
# The arrangement's maps and the cost that emerges from substitution.
# ---------------------------------------------------------------------------


def test_meeting_maps_are_the_papers():
    """REPRODUCES A CLOSED FORM the paper states (eq:meeting-maps): with ``T_ij = M``
    for ``i != j`` and ``l_i(x) = x_i``,

        sigma_m(x, m_1, m_2) = m_1 + m_2,
        rho_m(x, m_1, m_2, (m_0, p_0)) = ((m_0 + m_2, p_0 + x_1), (m_0 + m_1, p_0 + x_2)),

    the zero diagonal being what keeps ``m_i`` out of child ``i``'s own input."""
    a = parent_meeting()
    x = jnp.array([0.3, -1.2])
    m1, m2 = message(2.0, 1.0), message(-0.5, 3.0)
    m0, p0 = message(0.7, 1.0), 0.25
    e = make_input(m0, p0)

    np.testing.assert_allclose(a.sigma(x, jnp.concatenate([m1, m2])), m1 + m2, atol=1e-14)
    got = a.rho(x, jnp.concatenate([m1, m2]), e).reshape(2, EDIM)
    np.testing.assert_allclose(got[0], make_input(m0 + m2, p0 + x[0]), atol=1e-14)
    np.testing.assert_allclose(got[1], make_input(m0 + m1, p0 + x[1]), atol=1e-14)


def test_meeting_cost_emerges_from_substitution():
    """REPRODUCES A CLOSED FORM the paper proves (eq:meeting-cost, via eq:composite-cost
    and the chain rule eq:chain-rule): substituting the two subsidiaries into ``a_m``
    gives the cost

        K = (1/2)(g - (z_2 - z_1))^2 + (1/2)(x_1 - z_1)^2 + (1/2)(x_2 - z_2)^2.

    ``K`` is not handed to the composite -- the composite is
    ``compose_seq(tensor(S_1, S_2), a_m.arr())`` and its forcing is pullback-and-sum of
    the parent's prediction cost and the two subsidiary costs. What is checked is that
    the whole one-form, on the configuration *and* on the exterior input, is ``-dK``.

    It is checked with the reference position ``p_0`` free, which shows the one place the
    composite is larger than the paper's display: the subsidiary predicts the position it
    *receives*, ``p_0 + x_i``, so the composite cost depends on ``p_0``, and eq:meeting-cost
    is its restriction to the paper's ``p_0 = 0``. It does not depend on the exterior
    weight ``w_0``, as eq:parent-meeting-cost says.
    """
    b = rendezvous()
    grad_K = jax.grad(K_full, argnums=(0, 1, 2, 3, 4, 5, 6))
    r = np.random.default_rng(201)
    for _ in range(6):
        z1, z2, x1, x2, g, p0, w0 = r.standard_normal(7)
        q = jnp.asarray([z1, z2, x1, x2])
        e = exterior(g, w_0=w0, p_0=p0)
        xi_q, xi_m, xi_e = b.omega(q, jnp.zeros(0), e)
        dg, dw0, dp0, dz1, dz2, dx1, dx2 = grad_K(g, w0, p0, z1, z2, x1, x2)

        np.testing.assert_allclose(xi_q, -np.array([dz1, dz2, dx1, dx2]), atol=1e-13)
        np.testing.assert_allclose(xi_e, -np.array([dg, dw0, dp0]), atol=1e-13)
        assert xi_m.shape == (0,)
        np.testing.assert_allclose(float(dw0), 0.0, atol=1e-15)


def test_state_force_coefficients_at_the_resting_state():
    """REPRODUCES A CLOSED FORM the paper states (eq:meeting-dynamics and the sentence
    after it): "At ``g = 0`` and ``(z_1, z_2, x_1, x_2) = (0, 1, 0, 1)``, the state-force
    coefficients are ``(1, -1, 0, 0)``. The expectations begin changing before the
    physical positions do."

    The upstream force ``-r dg`` of the same paragraph is checked too: with ``r = -1``
    here, the exterior input carries the covector ``+1 dg``, and maintaining the request
    is the boundary condition supplied by holding ``e`` fixed in the runs below.
    """
    b = rendezvous()
    q = jnp.array([0.0, 1.0, 0.0, 1.0])
    xi_q, _, xi_e = b.omega(q, jnp.zeros(0), exterior(0.0))

    np.testing.assert_allclose(xi_q, np.array([1.0, -1.0, 0.0, 0.0]), atol=1e-14)
    r_val = 0.0 - (1.0 - 0.0)
    np.testing.assert_allclose(xi_e[0], -r_val, atol=1e-14)   # the covector -r dg
    np.testing.assert_allclose(xi_e[1], 0.0, atol=1e-14)      # nothing on the weight


def test_the_jitted_closed_step_is_Phi():
    """ILLUSTRATES nothing about the paper; it licenses ``run_closed`` (see its
    docstring). ``Phi`` and the specialized closed step agree *exactly*, bit for bit,
    over fifty steps, so the long runs below are ``Phi``'s trajectory."""
    b = rendezvous(alphas=(0.4, 0.4), betas=(0.4, 0.4))
    e = exterior(0.0)
    a = run_phi(b, 50, e)
    c = run_closed(b, 50, lambda t: e)
    assert float(np.max(np.abs(a - c))) == 0.0


# ---------------------------------------------------------------------------
# prop:meeting, discretely.
# ---------------------------------------------------------------------------


def test_meeting_point_for_equal_mobilities():
    """CHECKS THE DISCRETE ANALOGUE the book still has to prove (STATUS.md §3.2
    item 1). prop:meeting is a statement about the *flow* eq:meeting-dynamics; here the
    same data are run as the book's displacement map, one covector one step, with all
    four mobilities ``0.4`` -- inside the bound of STATUS.md §3.2, see
    ``test_the_mobility_bound_is_real``. From ``(z, x) = (0, 1, 0, 1)`` the run converges
    to the common point eq:meeting-point predicts, ``c = 1/2``."""
    b = rendezvous(alphas=(0.4, 0.4), betas=(0.4, 0.4))
    tr = run_closed(b, 400, lambda t: exterior(0.0))
    np.testing.assert_allclose(tr[-1], np.full(4, 0.5), atol=1e-12)


def test_meeting_point_with_a_less_mobile_component():
    """CHECKS THE DISCRETE ANALOGUE (as above) of the paper's second number: "Taking
    ``beta_2 = 1/10`` and the other mobilities equal to ``1`` gives ``c = 11/13``, closer
    to the less mobile component." The mobility *ratios* are the paper's; all four are
    scaled by a common step ``0.4``, which eq:meeting-point leaves invariant -- ``c`` is a
    ratio of reciprocal mobilities -- and which the mobility bound requires."""
    step = 0.4
    b = rendezvous(alphas=(step, step), betas=(step, step * 0.1))
    tr = run_closed(b, 3000, lambda t: exterior(0.0))
    np.testing.assert_allclose(tr[-1], np.full(4, 11.0 / 13.0), atol=1e-9)


def test_S_is_conserved_exactly_by_the_discrete_map():
    """CHECKS THE DISCRETE ANALOGUE (STATUS.md §3.2, third bullet) of the
    conserved quantity in prop:meeting's proof,

        S = z_1/alpha_1 + z_2/alpha_2 + x_1/beta_1 + x_2/beta_2.

    The mechanism is exact and not merely accurate: with ``r = g - (z_2 - z_1)`` the four
    state forces are ``(-r + (x_1 - z_1), r + (x_2 - z_2), -(x_1 - z_1), -(x_2 - z_2))``,
    which cancel identically -- ``K`` at fixed ``g`` depends only on differences -- and the
    step adds ``alpha_i`` (resp. ``beta_i``) times the ``i``th force to the ``i``th
    coordinate, so ``S`` is unchanged for *any* mobilities and any step, not to first
    order in a step size. What a floating-point run can show is that identity to within a
    single rounding, and that is what is asserted: the four computed force components sum
    to within one ulp of the largest of them (usually exactly ``0.0``), and ``S``
    recomputed along a five-hundred-step run drifts by a few ulps of ``S`` itself -- the
    rounding of the divisions ``z_i / alpha_i``, not a dynamical tolerance.
    """
    alphas, betas = (0.4, 0.4), (0.4, 0.25)
    b = rendezvous(alphas=alphas, betas=betas)
    mob = np.array(list(alphas) + list(betas))
    eps = np.finfo(float).eps

    r = np.random.default_rng(202)
    for _ in range(8):
        q = jnp.asarray(r.standard_normal(4))
        xi_q, _, _ = b.omega(q, jnp.zeros(0), exterior(float(r.standard_normal())))
        f = np.asarray(xi_q)
        assert abs(float(f.sum())) <= eps * float(np.max(np.abs(f)))

    tr = run_closed(b, 500, lambda t: exterior(0.0))
    S = (tr / mob).sum(axis=1)
    drift = float(np.max(np.abs(S - S[0])))
    assert drift <= 8.0 * eps * abs(float(S[0]))


# ---------------------------------------------------------------------------
# The mobility bound (STATUS.md §3.2), exhibited failing.
# ---------------------------------------------------------------------------


def test_the_kernel_of_the_hessian_is_the_conserved_direction():
    """REPRODUCES A CLOSED FORM the paper proves (prop:meeting's proof): "The Hessian of
    ``K(0, -, -)`` is positive semidefinite with kernel ``z_1 = z_2 = x_1 = x_2``."

    It matters for the discrete restatement the book owes (§5.9 item 1). notes/ATTENTION-DESIGN.md
    §4 states the bound as "converges iff every eigenvalue of ``D^{1/2} H D^{1/2}`` lies in
    ``(0, 2)``", which no mobility satisfies here: ``H`` has the eigenvalue ``0``, and that
    is not a failure of convergence but the conserved ``S``. The book's statement must read
    *every nonzero eigenvalue*, the zero eigenvalue being the direction along which the
    limit is fixed by the initial condition rather than approached.
    """
    H = hessian_of_K()
    ev = np.linalg.eigvalsh(H)
    assert ev[0] > -1e-12 and abs(ev[0]) < 1e-12          # positive semidefinite, one zero
    assert ev[1] > 1e-6                                    # the rest strictly positive
    np.testing.assert_allclose(H @ np.ones(4), np.zeros(4), atol=1e-13)


def test_the_mobility_bound_is_real():
    """CHECKS THE DISCRETE ANALOGUE the book has to prove (§5.9 item 1), and exhibits the
    bound *failing*. For equal mobilities ``mu`` the map is ``y -> y - mu H y`` about the
    limit, so the transverse modes contract iff ``mu lambda in (0, 2)`` for every nonzero
    eigenvalue ``lambda`` of ``H``. Here ``lambda_max(H) = 2 + sqrt 2``, so the critical
    mobility is ``2 / (2 + sqrt 2) = 2 - sqrt 2 = 0.5857864...``.

    Just inside (``mu = 0.55``, largest eigenvalue ``1.8778``) the run reaches ``1/2``.
    Just outside (``mu = 0.62``, largest eigenvalue ``2.1168``) it leaves every scale --
    ``|y| > 1e12`` within three hundred steps. A bound never exhibited failing is not a
    bound.
    """
    H = hessian_of_K()
    mu_crit = 2.0 / np.linalg.eigvalsh(H)[-1]
    np.testing.assert_allclose(mu_crit, 2.0 - np.sqrt(2.0), atol=1e-12)

    inside, outside = 0.55, 0.62
    assert inside < mu_crit < outside
    assert spectrum(H, [inside] * 4)[-1] < 2.0 < spectrum(H, [outside] * 4)[-1]

    conv = run_closed(rendezvous(alphas=(inside,) * 2, betas=(inside,) * 2),
                      300, lambda t: exterior(0.0))
    np.testing.assert_allclose(conv[-1], np.full(4, 0.5), atol=1e-12)

    div = run_closed(rendezvous(alphas=(outside,) * 2, betas=(outside,) * 2),
                     300, lambda t: exterior(0.0))
    assert float(np.max(np.abs(div[-1]))) > 1e12


def test_the_observed_rate_is_the_spectral_radius_on_both_sides_of_the_bound():
    """CHECKS THE DISCRETE ANALOGUE (§5.9 item 1) sharply, which is what makes the bound
    an *iff* rather than a sufficient condition. Straddling ``2 - sqrt 2`` by one part in
    six thousand -- ``mu = 0.5857`` and ``mu = 0.5859`` -- the observed per-step growth of
    the transverse component, ``(|y_600 - c| / |y_300 - c|)^{1/300}``, agrees with
    ``|1 - mu lambda_max(H)|`` to six decimals, and crosses ``1`` exactly where the
    eigenvalue crosses ``2``: contracting at ``0.999705``, growing at ``1.000388``."""
    H = hessian_of_K()
    for mu, growing in [(0.5857, False), (0.5859, True)]:
        predicted = abs(1.0 - spectrum(H, [mu] * 4)[-1])
        tr = run_closed(rendezvous(alphas=(mu,) * 2, betas=(mu,) * 2),
                        600, lambda t: exterior(0.0))
        amp = np.linalg.norm(tr - 0.5, axis=1)
        observed = (amp[600] / amp[300]) ** (1.0 / 300.0)
        np.testing.assert_allclose(observed, predicted, atol=1e-6)
        assert bool(observed > 1.0) is growing
