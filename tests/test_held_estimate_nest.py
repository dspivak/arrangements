"""Does the held estimate nest? The hearing sender in a loop with a reader, then two senders in a group heard through an
outer transfer (book plan P71 (c); David, 2026-10-23: "will it work compositionally? you can put that off for its own
task if you prefer").

`test_held_estimate.py` (§6 of the book's ``notes/runs/p71b-fcfaae81/RESULTS.md``) built the sender of David's picture:
it holds one number ``g``, its estimate of how loudly the reader hears it, the ``gamma`` of Ch 4's transfer
``T_{gamma,V}``; it wants its message to arrive with weight ``HEARD``; it pays a price ``c w`` on its loudness
``w = exp(lam)`` and corrects ``g`` by what comes back, the arriving weight ``r``. There ``r`` was the environment's
rule, ``gamma`` times the weight sent three steps before. Here the reader is in the loop.

**What comes back, and why an acknowledgment.** The sender's ``r`` is a value at its input port, so in the book's
terms it is carried forward on a wire: a message the sender hears from the reader through the group's transfer
``T_{12}``. The other candidate, the reader's pushback on the arriving weight, is a force, already present in any exact
composite: ``gamma [kappa_r + w_r/2 (v - mu)^2]`` per unit sent (`test_weight_price.py`), ``gamma`` times how much the
reader minds, so it cannot be read as how loudly the reader hears. An attention arrangement with no internal boxes
sends the zero message, so the reader that acknowledges is a part filling a leaf: its matter is its belief ``mu`` and
the weight ``a`` it has heard, tracking the arriving weight by ``p_a/2 (a - w_arr)^2``; it reads out the message
``(a mu, a)``: value its belief, weight what it heard.

**What makes the return one-way.** The sender's prediction error ``1/2 (r - g w)^2`` has a nonzero differential at
``r``, a force on the return message's weight that pulls back, through ``T_{12}`` and the reader's readout, onto the
reader's ``a``: the sender's expectation would move the reader's acknowledgment toward ``g w``. Three ways to stop it.
A *static box* cannot carry the acknowledgment at all: a lens's readout cannot see its own input, ``T_{ii} = 0``
forbids a box hearing itself, and a static box with a readout that depends on its input pulls the force back through
it. A *load-free* sender (effort field with zero component at ``r``) stops it but is not exact: a forcing ``-dK`` on the
matter and ``0`` at ``r`` is not closed when ``K`` depends on ``r``, so the whole is no longer ``-d`` of one function.
A *unilateral* reader at its output (the glossary's: ``u(xi + (D o)^T eta) = u(xi)``) stops it in the response and
keeps every forcing ``-dK``: the readout is written on ``a + phi_a`` and ``mu + phi_mu`` with the shadows frozen, the
reader's own cost on ``a`` and ``mu``, and the response moves ``a`` by the covector on ``a`` minus the covector on
``phi_a`` (``S[a,a] = eta``, ``S[a,phi_a] = -eta``), so any force arriving at the output, which lands on both equally,
cancels. This is the mechanism §5–6 used for the estimate inside the sender, one level up, as P71 (c) guessed.

**Where the pushback lands.** (1) The sender's prediction error: on ``phi`` (corrects ``g``); on ``g`` (ignored); on
``lam`` (speak up when more arrives than predicted); on ``r``, hence through ``T_{12}`` and the reader's readout onto
``(a, phi_a)``, discarded by the unilateral reader, taken up by a symmetric one. (2) The reader's code length: on
``mu`` (its belief moves toward the value) and on the arriving message, hence through ``T_{21}`` and the sender's
readout onto ``lam``: ``-gamma w C_r``, ``C_r = kappa_r + w_r/2 (v - mu)^2``, the charge of `test_weight_price.py`.
(3) The reader's tracking ``p_a/2 (a - w_arr)^2``: on ``a`` (own) and on the sender's ``lam``, pulling the loudness
toward what has been acknowledged, zero at rest.

**Rest by hand** (``mu`` held). ``g = gamma_b gamma`` (the round trip: out through ``T_{21} = T_gamma``, back through
``T_{12} = T_{gamma_b}``) and, with ``gamma_b = 1``,
``w* = HEARD/gamma - (c + gamma C_r)/(kappa gamma^2)``: the reader's charge adds to the price, scaled by ``gamma``.
Louder the farther, silent below ``gamma = c/(kappa HEARD - C_r)``, the deaf included. The symmetric reader rests at
the same point (``a`` is pulled toward ``g w`` and ``g`` is free, so at rest ``a = g w = gamma w`` still); the two
differ on the way, and when the reader goes deaf.

**The nest.** Outer ``O`` (2-ary): box 1 the group ``G`` (2-ary, the senders), box 2 the reader; ``T^O_{21} =
T_{gamma_o}``, ``T^O_{12} = T_{gamma_b}``. ``G``: ``T^G_{0j} = T_{gamma_j}``, ``T^G_{j0} = T_{beta_j}``, the senders
not hearing each other. By ``prop.attention_operad`` the reader hears sender ``j`` through ``gamma_o gamma_j`` and
sender ``j`` hears the reader through ``beta_j gamma_b``; but the reader hears *one* message, the sum
``W = gamma_o (gamma_1 w_1 + gamma_2 w_2)``, and acknowledges that. So at rest
``g_j = beta_j gamma_b gamma_o (gamma_1 w_1 + gamma_2 w_2) / w_j``: the product of the gammas along the path only when
the sender is alone in its group; with company, each sender takes the group's acknowledgment for its own. A group
holding its own estimate ``g_G`` (matter of ``G``; cost a want on the weight it sends and a prediction error on the
weight that returns) rests at ``g_G = gamma_b gamma_o``, the outer round trip, and ``g_j w_j = beta_j g_G W_s``, ``W_s = gamma_1 w_1 + gamma_2 w_2`` the weight the group sends: the
members' estimates are not summed into the group's; each is the group's, scaled by how loudly the member hears the
group's return and divided by the member's own weight. A price nests by ``prop.exact_lenses``: charged at a level on the
weight an internal box sends it, it reaches a sender scaled by the transfers between, those of the levels the message
passes through on its way out (``gamma_j`` for a price at the outer level, the sender's own group's), not by the
charging level's own, since a cost function sees the messages its internal boxes send, before its own transfers act; so a
price at the outer level charges each sender in proportion to how loudly its group hears it; the sender's matter is what
moves.

Results and the judgment: the book's ``notes/runs/p71c-54ac9b94/RESULTS.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import attention_arrangement, substitute, transfer, unit_arrangement, zero_transfer
from arrangements.coalgebra import configuration
from arrangements.functors import open_step
from arrangements.rvect import constant, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

DH, DP = 1, 0                                   # numbers for values, no positions
ZERO_IN = jnp.zeros(2)                          # the zero message at the external input
V = 1.0                                         # the senders' value, fixed
C, KAPPA, HEARD = 0.02, 0.5, 1.0                # the sender's price, want stiffness, wanted arrival (§6's)
W_R, KAPPA_R, P_R, MU0, P_A = 0.2, 0.05, 1.0, 0.0, 1.0   # the reader's
EPS = 1e-9                                      # guards the code length at the apex, where the deaf reader sits


def T(gamma):
    """The transfer ``T_{gamma, gamma id}``: weight scaled by ``gamma``, value kept."""
    return zero_transfer(DH) if gamma == 0 else transfer(gamma, gamma * jnp.eye(DH))


# ---------------------------------------------------------------------------
# The parts.
# ---------------------------------------------------------------------------


def hearing_sender(lam0=-2.0, g0=0.5, c=C, kappa=KAPPA, heard=HEARD, eta_lam=0.1, eta_g=0.1, label="sender",
                   frozen=False) -> Arrangement:
    """§6's sender with a message input: matter ``(lam, g, phi)``, ``phi`` the frozen shadow; input the arriving
    message ``(h, w)``, of which it reads the weight ``r = w``; output ``(e^lam V, e^lam)``. Cost the price, the
    predicted shortfall and the prediction error. ``frozen`` holds everything (a sender that is absent)."""
    def cost(q, m_out, n_in):
        w, g, r = jnp.exp(q[0]), q[1], n_in[1]
        return c * w + 0.5 * kappa * (heard - g * w) ** 2 + 0.5 * (r - (g + q[2]) * w) ** 2

    S = jnp.zeros((3, 3)) if frozen else jnp.zeros((3, 3)).at[0, 0].set(eta_lam).at[1, 2].set(eta_g)
    return Arrangement(
        P=configuration(constant(S), init=jnp.array([lam0, g0, 0.0], dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=2,
        out_f=lambda q, m_out: jnp.array([jnp.exp(q[0]) * V, jnp.exp(q[0])]),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, m_out, n_in)),
        label=label,
    )


def acknowledging_reader(mu0=MU0, a0=0.0, w_r=W_R, kappa_r=KAPPA_R, p_r=P_R, p_a=P_A, eta_mu=0.1, eta_a=0.3,
                         unilateral=True, label="reader") -> Arrangement:
    """The reader as a part: matter its belief ``mu`` and the weight ``a`` it has heard, with shadows ``(phi_mu,
    phi_a)`` when unilateral; input the arriving message ``(h, w)``; output the acknowledgment ``(a mu, a)``, value its
    belief, weight what it heard. Cost: the Gaussian code length of what arrives (`test_weight_price.py`'s), its prior,
    and the tracking ``p_a/2 (a - w)^2``. Unilateral at its output: the readout is written on ``mu + phi_mu`` and
    ``a + phi_a``, the cost on ``mu`` and ``a``, and the response subtracts the shadow's covector."""
    def cost(q, m_out, n_in):
        mu, a, h, w = q[0], q[1], n_in[0], n_in[1]
        length = 0.5 * w_r * (h - w * mu) ** 2 / (w + EPS) + kappa_r * w
        return length + 0.5 * p_r * (mu - mu0) ** 2 + 0.5 * p_a * (a - w) ** 2

    if unilateral:
        S = (jnp.zeros((4, 4)).at[0, 0].set(eta_mu).at[0, 2].set(-eta_mu)
             .at[1, 1].set(eta_a).at[1, 3].set(-eta_a))
        init = jnp.array([mu0, a0, 0.0, 0.0], dtype=float)
        out_f = lambda q, m_out: jnp.array([(q[1] + q[3]) * (q[0] + q[2]), q[1] + q[3]])
    else:
        S = jnp.diag(jnp.array([eta_mu, eta_a]))
        init = jnp.array([mu0, a0], dtype=float)
        out_f = lambda q, m_out: jnp.array([q[1] * q[0], q[1]])
    return Arrangement(
        P=configuration(constant(S), init=init),
        in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=2,
        out_f=out_f, in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, m_out, n_in)),
        label=label,
    )


def pair(gamma, gamma_b=1.0, price=None, label="pair"):
    """The 2-ary attention arrangement of the loop: box 2 (the reader) hears box 1 (the sender) through
    ``T_gamma``, box 1 hears box 2 through ``T_{gamma_b}``; nothing is heard from or sent outside. Trivial matter;
    cost ``0``, or ``price * wt(m_1)``."""
    Z = zero_transfer(DH)
    kw = {"cost": (lambda q, m_out, n_in: price * m_out[1])} if price is not None else {}
    return attention_arrangement(
        2, DH, DP, lambda q: jnp.zeros((2, DP)), lambda q: ((Z, Z, Z), (Z, Z, T(gamma_b)), (Z, T(gamma), Z)),
        label=label, **kw,
    )


def group(gammas, betas, estimate=False, g0=0.5, kappa_G=KAPPA, heard_G=HEARD, eta_g=0.1, price=None, label="G"):
    """The senders' group: the outside hears sender ``j`` through ``T_{gamma_j}``, sender ``j`` hears the outside
    through ``T_{beta_j}``, the senders do not hear each other. With ``estimate`` the group holds its own estimate
    ``g_G`` (unilateral, with a shadow) of how loudly it is heard: cost a want ``kappa_G/2 (heard_G - g_G W_s)^2`` on
    the weight ``W_s`` it sends and a prediction error ``1/2 (wt(m_0) - (g_G + phi) W_s)^2`` on the weight that comes
    back in. ``price`` charges ``price * W_s``."""
    Z = zero_transfer(DH)
    g1, g2 = gammas
    b1, b2 = betas

    def transfers(q):
        return ((Z, T(g1), T(g2)), (T(b1), Z, Z), (T(b2), Z, Z))

    def cost(q, m_out, n_in):
        W_s = g1 * m_out[1] + g2 * m_out[3]
        total = 0.0 if price is None else price * W_s
        if estimate:
            total = total + 0.5 * kappa_G * (heard_G - q[0] * W_s) ** 2 + 0.5 * (n_in[1] - (q[0] + q[1]) * W_s) ** 2
        return total

    kw = {}
    if estimate:
        kw["parameter"] = configuration(constant(jnp.zeros((2, 2)).at[0, 1].set(eta_g)),
                                        init=jnp.array([g0, 0.0], dtype=float))
    return attention_arrangement(2, DH, DP, lambda q: jnp.zeros((2, DP)), transfers, cost=cost, label=label, **kw)


def outer(gamma_o, gamma_b, price=None, label="O"):
    """The outer arrangement: box 2 (the reader) hears box 1 (the group) through ``T_{gamma_o}``, the group hears the
    reader through ``T_{gamma_b}``. ``price`` charges ``price * wt(m_1)`` on what the group sends."""
    return pair(gamma_o, gamma_b, price=price, label=label)


def loop(snd, rd, grp=None, gamma=1.0, by="substitute"):
    """The system ``1 -> B``: the sender and the reader filling the pair's boxes, by the operad's substitution of the
    unit and the pair into itself (one box, then the parts by ``compose_seq``) or as the lens composite."""
    grp = grp or pair(gamma)
    if by == "substitute":
        op = substitute(grp, [unit_arrangement(DH, DP), unit_arrangement(DH, DP)], label="pair(1, 1)").arr()
    else:
        op = grp.arr()
    return compose_seq(tensor_arrangements([snd, rd]), op)


def nest(senders, rd, grp, out, by="substitute"):
    """The system ``1 -> B`` of the nest: ``O(G(s_1, s_2), reader)``, either by the operad's substitution of ``G``
    and the unit into ``O`` (a 3-ary operation, the parts filling its boxes) or stepwise as lens composites."""
    if by == "substitute":
        op = substitute(out, [grp, unit_arrangement(DH, DP)], label="O(G, 1)").arr()
        return compose_seq(tensor_arrangements(list(senders) + [rd]), op)
    inner = compose_seq(tensor_arrangements(list(senders)), grp.arr())
    return compose_seq(tensor_arrangements([inner, rd]), out.arr())


def run(arr, steps):
    """``Phi(arr)`` told the zero message and no force at its outputs; the flat states at every step."""
    step = open_step(arr, ZERO_IN)
    s, out = arr.P.init, [flat_state(arr.P.init)]
    for _ in range(steps):
        s = step(s)
        out.append(flat_state(s))
    return jnp.stack(out)


# ---------------------------------------------------------------------------
# By hand.
# ---------------------------------------------------------------------------


def charge(mu, w_r=W_R, kappa_r=KAPPA_R):
    """The reader's charge per unit of arriving weight, ``C_r = kappa_r + w_r/2 (V - mu)^2``."""
    return kappa_r + 0.5 * w_r * (V - mu) ** 2


def loop_rest(gamma, mu=None, c=C, kappa=KAPPA, heard=HEARD, gamma_b=1.0):
    """The loop's rest by hand: ``g = gamma_b gamma``, and the loudness where
    ``kappa (heard - g w) g = c + gamma C_r(mu)``, with ``mu`` held or, if ``None``, at its own rest
    ``(p_r mu0 + w_r gamma w V)/(p_r + w_r gamma w)``. Silence where no positive root exists."""
    if gamma == 0:
        return 0.0, 0.0, MU0
    g = gamma_b * gamma

    def mu_of(w):
        return mu if mu is not None else (P_R * MU0 + W_R * gamma * w * V) / (P_R + W_R * gamma * w)

    def f(w):
        return kappa * (heard - g * w) * g - c - gamma * charge(mu_of(w))

    if f(1e-9) <= 0:
        return 0.0, g, mu_of(0.0)
    w = brentq(f, 1e-9, heard / g)
    return w, g, mu_of(w)


def nest_rest(gammas, betas, gamma_o, gamma_b, mu=None):
    """The nest's rest by hand. With ``W = gamma_o (gamma_1 w_1 + gamma_2 w_2)`` the arriving weight and
    ``g_j w_j = beta_j gamma_b W`` at rest, the force on ``lam_j`` vanishes where
    ``kappa (heard - beta_j gamma_b W) beta_j gamma_b W / w_j = c + gamma_o gamma_j C_r``; solve for ``W``."""
    def w_j(W, j, mu_):
        return KAPPA * (HEARD - betas[j] * gamma_b * W) * betas[j] * gamma_b * W / (C + gamma_o * gammas[j] * charge(mu_))

    def mu_of(W):
        return mu if mu is not None else (P_R * MU0 + W_R * W * V) / (P_R + W_R * W)

    def f(W):
        mu_ = mu_of(W)
        return gamma_o * sum(gammas[j] * w_j(W, j, mu_) for j in range(2)) - W

    W = brentq(f, 1e-6, HEARD / (gamma_b * max(betas)) - 1e-9)
    mu_ = mu_of(W)
    ws = [w_j(W, j, mu_) for j in range(2)]
    gs = [betas[j] * gamma_b * W / ws[j] for j in range(2)]
    return ws, gs, W, mu_


# ---------------------------------------------------------------------------
# The loop: one sender, one reader, the acknowledgment returned through the pair.
# ---------------------------------------------------------------------------


def test_loop_forcing_is_minus_d_of_the_summed_costs_and_substitution_agrees():
    """REPRODUCES ``prop.exact_lenses`` and ``prop.attention_operad`` on the loop: the whole's forcing is ``-d`` of
    the sender's cost at the return it hears plus the reader's cost at what arrives, as a function of all seven
    coordinates (shadows included); and the pair by substitution equals the lens composite in readout, forcing and a
    50-step trajectory."""
    gamma = 0.25
    snd, rd = hearing_sender(), acknowledging_reader()
    W1, W2 = loop(snd, rd, gamma=gamma, by="substitute"), loop(snd, rd, gamma=gamma, by="compose")

    def hand(q):
        lam, g, phi, mu, a, pm, pa = q
        w = jnp.exp(lam)
        h_arr, w_arr = gamma * w * V, gamma * w
        r = (a + pa)                                     # the return through the identity T_{12}
        K_s = C * w + 0.5 * KAPPA * (HEARD - g * w) ** 2 + 0.5 * (r - (g + phi) * w) ** 2
        K_r = (0.5 * W_R * (h_arr - w_arr * mu) ** 2 / (w_arr + EPS) + KAPPA_R * w_arr
               + 0.5 * P_R * (mu - MU0) ** 2 + 0.5 * P_A * (a - w_arr) ** 2)
        return K_s + K_r

    rng = np.random.default_rng(7)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(7))
        xi_q, _, _ = W1.omega(q, jnp.zeros(0), ZERO_IN)
        np.testing.assert_allclose(xi_q, -jax.grad(hand)(q), atol=1e-9)
        np.testing.assert_allclose(W1.out_f(q, jnp.zeros(0)), W2.out_f(q, jnp.zeros(0)), atol=1e-12)
        for x, y in zip(W1.omega(q, jnp.zeros(0), ZERO_IN), W2.omega(q, jnp.zeros(0), ZERO_IN)):
            np.testing.assert_allclose(x, y, atol=1e-11)
    np.testing.assert_allclose(run(W1, 50), run(W2, 50), atol=1e-11)


def test_reader_is_unilateral_at_its_output():
    """CHECKS THE GLOSSARY'S CONDITION ``u(xi + (D o)^T eta) = u(xi)`` on the reader: a force at its acknowledgment
    port, pulled back through its readout, does not change its step; the symmetric reader's step does change."""
    for unilateral, same in [(True, True), (False, False)]:
        rd = acknowledging_reader(unilateral=unilateral)
        rng = np.random.default_rng(3)
        q = jnp.asarray(rng.standard_normal(rd.P.Q_dim)) if not unilateral else jnp.array([0.3, 0.7, 0.0, 0.0])
        n_in = jnp.array([0.4, 0.5])
        xi_q, _, _ = rd.omega(q, jnp.zeros(0), n_in)
        _, vjp = jax.vjp(lambda qq: rd.out_f(qq, jnp.zeros(0)), q)
        for _ in range(3):
            eta = jnp.asarray(rng.standard_normal(2))
            (pulled,) = vjp(eta)
            s0, s1 = rd.P.step(q, xi_q), rd.P.step(q, xi_q + pulled)
            if same:
                np.testing.assert_allclose(s0, s1, atol=1e-12)
            else:
                assert float(jnp.max(jnp.abs(s0 - s1))) > 1e-3


def test_loop_talks_louder_when_far_and_not_at_the_deaf():
    """CHECKS THE HAND RESULT with the reader's belief held: ``g`` comes to ``gamma`` and the loudness to
    ``HEARD/gamma - (c + gamma C_r)/(kappa gamma^2)``, the reader's charge added to the price: softer near, louder far,
    silent at the deaf, and falling toward silence at a reader that hears a little (``gamma = 0.04``, below
    ``c/(kappa HEARD - C_r) = 0.057``), though not deaf: there the force on the log-weight at ``g = gamma`` is
    ``-w (c + gamma C_r - kappa HEARD gamma) = -0.006 w``, so ``1/w`` grows by ``0.0006`` a step and the quieting takes
    tens of thousands of steps. The reader's acknowledgment rests at what arrives."""
    rests = {}
    for name, gamma in [("near", 1.0), ("far", 0.25), ("hard of hearing", 0.04), ("deaf", 0.0)]:
        W = loop(hearing_sender(), acknowledging_reader(eta_mu=0.0), gamma=gamma)
        traj = run(W, 6000)
        lam, g, mu, a = traj[:, 0], traj[:, 1], traj[:, 3], traj[:, 4]
        w_hand, g_hand, _ = loop_rest(gamma, mu=MU0)
        rests[name] = float(jnp.exp(lam[-1]))
        if w_hand > 0:
            np.testing.assert_allclose(jnp.exp(lam[-1]), w_hand, rtol=0.01)
            np.testing.assert_allclose(g[-1], g_hand, atol=2e-3)
            np.testing.assert_allclose(a[-1], gamma * jnp.exp(lam[-1]), rtol=1e-3)
        elif gamma == 0:
            assert float(jnp.exp(lam[-1])) < 0.1 and float(g[-1]) < 1e-2
        else:                                                # below the threshold but not deaf: falling, slowly
            np.testing.assert_allclose(g[-1], gamma, atol=2e-3)
            assert float(lam[-1]) < float(lam[-1000]) and float(jnp.exp(lam[-1])) < 0.5 * rests["near"]
    assert rests["far"] > 3 * rests["near"] > 5 * rests["deaf"]


def test_loop_with_a_moving_belief_rests_where_the_hand_fixed_point_says():
    """CHECKS THE HAND RESULT with everything mobile: the reader's belief rests at
    ``(p_r mu0 + w_r gamma w V)/(p_r + w_r gamma w)`` and the loudness at the coupled fixed point; the belief moves
    toward the value and stays, since the sender does not fall quiet."""
    for gamma in (1.0, 0.25):
        traj = run(loop(hearing_sender(), acknowledging_reader(), gamma=gamma), 8000)
        w_hand, g_hand, mu_hand = loop_rest(gamma)
        np.testing.assert_allclose(jnp.exp(traj[-1, 0]), w_hand, rtol=0.01)
        np.testing.assert_allclose(traj[-1, 3], mu_hand, atol=2e-3)
        assert float(traj[-1, 3]) > 0.05                   # the belief has moved toward V = 1 and stays


def test_symmetric_reader_rests_at_the_same_point_but_is_bent_on_the_way():
    """CHECKS THE HAND RESULT that the rest does not need the one-way return (``a`` is pulled toward ``g w`` and ``g``
    is free, so ``a = gamma w`` at rest either way), and ILLUSTRATES what it does need it for: the record. On the way
    the symmetric reader acknowledges more than it hears while the sender expects more (``a > gamma w``, by ``4%`` of
    what arrives at the worst step); when the reader goes deaf it goes on acknowledging the expectation for a while,
    ``1.1`` units of weight it never heard, against none; the sender's quieting is the same to ``1%`` either way, since
    the bent acknowledgment still falls short of what the sender expects and its estimate unlearns at the same rate.
    The one-way return keeps the acknowledgment honest; the sender's own learning protects its rest."""
    gamma = 0.25
    far = {}
    for unilateral in (True, False):
        W = loop(hearing_sender(), acknowledging_reader(eta_mu=0.0, unilateral=unilateral), gamma=gamma)
        traj = run(W, 6000)
        far[unilateral] = traj
        w_hand, g_hand, _ = loop_rest(gamma, mu=MU0)
        np.testing.assert_allclose(jnp.exp(traj[-1, 0]), w_hand, rtol=0.01)
        np.testing.assert_allclose(traj[-1, 1], g_hand, atol=2e-3)
    a_u, a_s = far[True][:, 4], far[False][:, 4]
    w_u, w_s = jnp.exp(far[True][:, 0]), jnp.exp(far[False][:, 0])
    excess_u = float(jnp.max(a_u - gamma * w_u))
    excess_s = float(jnp.max(a_s - gamma * w_s))
    assert excess_u < 1e-12                                   # the unilateral reader never acknowledges more than arrives
    assert excess_s > 0.02                                    # the symmetric one does, on the way

    # the reader goes deaf: a chain of two arrangements, the second started from the first's end state
    quiet = {}
    for unilateral in (True, False):
        start = far[unilateral][-1]
        snd = hearing_sender(lam0=float(start[0]), g0=float(start[1]))
        rd = acknowledging_reader(mu0=MU0, a0=float(start[4]), eta_mu=0.0, unilateral=unilateral)
        quiet[unilateral] = run(loop(snd, rd, gamma=0.0), 3000)
    heard_after = {u: float(jnp.sum(quiet[u][1:, 4])) for u in quiet}      # acknowledged weight after the deafness
    w_end = {u: float(jnp.exp(quiet[u][-1, 0])) for u in quiet}
    assert heard_after[False] > 1.5 * heard_after[True]       # the symmetric reader acknowledges what it never heard
    np.testing.assert_allclose(w_end[False], w_end[True], rtol=0.01)   # and the sender falls quiet the same


# ---------------------------------------------------------------------------
# The nest: two senders in a group heard through an outer transfer.
# ---------------------------------------------------------------------------

GAMMAS, BETAS, GAMMA_O, GAMMA_B = (1.0, 0.5), (1.0, 1.0), 0.5, 1.0


def test_nest_substitution_agrees_with_the_stepwise_lens_composite():
    """REPRODUCES ``prop.attention_operad`` on a branching composite: ``O(G, 1)`` by the explicit substitution, the
    three parts filling its boxes, equals the stepwise lens composite (the senders into ``G``, then that and the reader
    into ``O``) in readout, forcing and trajectory; and the composite's transfers are the products along the paths:
    the reader hears sender ``j`` through ``gamma_o gamma_j`` and sender ``j`` hears the reader through
    ``beta_j gamma_b``."""
    senders = [hearing_sender(label="s1"), hearing_sender(lam0=-1.0, g0=0.3, label="s2")]
    rd, grp, out = acknowledging_reader(), group(GAMMAS, BETAS), outer(GAMMA_O, GAMMA_B)
    W1, W2 = nest(senders, rd, grp, out, by="substitute"), nest(senders, rd, grp, out, by="compose")
    rng = np.random.default_rng(11)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(W1.P.Q_dim))
        np.testing.assert_allclose(W1.out_f(q, jnp.zeros(0)), W2.out_f(q, jnp.zeros(0)), atol=1e-12)
        for x, y in zip(W1.omega(q, jnp.zeros(0), ZERO_IN), W2.omega(q, jnp.zeros(0), ZERO_IN)):
            np.testing.assert_allclose(x, y, atol=1e-11)
    np.testing.assert_allclose(run(W1, 50), run(W2, 50), atol=1e-11)

    op = substitute(out, [grp, unit_arrangement(DH, DP)])
    Tc = op.transfers(jnp.zeros(0))
    for j in range(2):
        np.testing.assert_allclose(Tc[3][j + 1].gamma, GAMMA_O * GAMMAS[j])     # reader (box 3) hears sender j
        np.testing.assert_allclose(Tc[j + 1][3].gamma, BETAS[j] * GAMMA_B)      # sender j hears the reader
    np.testing.assert_allclose(Tc[1][2].gamma, 0.0)                              # the senders do not hear each other


def test_alone_in_its_group_a_sender_estimates_the_product_of_the_gammas():
    """CHECKS THE HAND RESULT for a sender alone in its group (the other frozen silent): its estimate comes to the
    product of the gammas along the round trip, ``beta_1 gamma_b gamma_o gamma_1``, and its loudness to the loop's
    rest at that path, as ``sec.attention_nesting``'s composite transfer says."""
    senders = [hearing_sender(label="s1"), hearing_sender(lam0=-30.0, g0=0.0, frozen=True, label="s2")]
    rd, grp, out = acknowledging_reader(eta_mu=0.0), group(GAMMAS, BETAS), outer(GAMMA_O, GAMMA_B)
    traj = run(nest(senders, rd, grp, out), 8000)
    path = GAMMA_O * GAMMAS[0]
    w_hand, g_hand, _ = loop_rest(path, mu=MU0, gamma_b=BETAS[0] * GAMMA_B)
    np.testing.assert_allclose(traj[-1, 1], BETAS[0] * GAMMA_B * GAMMA_O * GAMMAS[0], atol=2e-3)
    np.testing.assert_allclose(jnp.exp(traj[-1, 0]), w_hand, rtol=0.01)


def test_with_company_each_sender_takes_the_groups_acknowledgment_for_its_own():
    """CHECKS THE HAND RESULT, and the finding: with two senders the reader hears their sum and acknowledges it, so
    each sender's estimate rests at ``beta_j gamma_b gamma_o (gamma_1 w_1 + gamma_2 w_2)/w_j``, above the product of
    the gammas along its own path by the factor ``1 + (the other's arriving weight)/(its own)``; the loudnesses rest at
    the hand fixed point; and the group arrives at ``HEARD - ...`` as one voice, each member speaking less than it
    would alone."""
    senders = [hearing_sender(label="s1"), hearing_sender(lam0=-1.0, g0=0.3, label="s2")]
    rd, grp, out = acknowledging_reader(eta_mu=0.0), group(GAMMAS, BETAS), outer(GAMMA_O, GAMMA_B)
    traj = run(nest(senders, rd, grp, out), 8000)
    w = [float(jnp.exp(traj[-1, 0])), float(jnp.exp(traj[-1, 3]))]
    g = [float(traj[-1, 1]), float(traj[-1, 4])]
    ws, gs, W, _ = nest_rest(GAMMAS, BETAS, GAMMA_O, GAMMA_B, mu=MU0)
    np.testing.assert_allclose(w, ws, rtol=0.02)
    np.testing.assert_allclose(g, gs, rtol=0.02)
    for j in range(2):
        path = BETAS[j] * GAMMA_B * GAMMA_O * GAMMAS[j]
        other = 1 - j
        np.testing.assert_allclose(g[j], path * (1 + GAMMAS[other] * w[other] / (GAMMAS[j] * w[j])), rtol=0.02)
        assert g[j] > 1.5 * path                             # well above its own path
    # alone, sender 1 would speak louder than it does with company
    w_alone, _, _ = loop_rest(GAMMA_O * GAMMAS[0], mu=MU0, gamma_b=BETAS[0] * GAMMA_B)
    assert w[0] < 0.8 * w_alone


def test_group_holding_its_own_estimate_learns_the_outer_round_trip():
    """CHECKS THE HAND RESULT: a group with its own held estimate rests at ``g_G = gamma_b gamma_o``, the outer round
    trip; each member's ``g_j w_j`` equals ``beta_j g_G W_s``, the group's estimated arrival scaled by how loudly the
    member hears the group's return; and the group's want, pulled back through its transfers, is felt by each member
    in proportion to how loudly the group hears it. The estimates do not sum."""
    senders = [hearing_sender(label="s1"), hearing_sender(lam0=-1.0, g0=0.3, label="s2")]
    rd, out = acknowledging_reader(eta_mu=0.0), outer(GAMMA_O, GAMMA_B)
    grp = group(GAMMAS, BETAS, estimate=True, heard_G=1.5)
    W = nest(senders, rd, grp, out)
    traj = run(W, 8000)
    q = traj[-1]
    w = [float(jnp.exp(q[0])), float(jnp.exp(q[3]))]
    g = [float(q[1]), float(q[4])]
    g_G = float(q[10])                                       # the group's configuration follows the three parts' (compose_seq's order)
    np.testing.assert_allclose(g_G, GAMMA_B * GAMMA_O, atol=2e-3)
    W_s = GAMMAS[0] * w[0] + GAMMAS[1] * w[1]
    W_arr = GAMMA_O * W_s
    for j in range(2):
        np.testing.assert_allclose(g[j] * w[j], BETAS[j] * g_G * W_s, rtol=0.02)
    # the group's want on the senders' weights: -d/d m_j of kappa_G/2 (heard_G - g_G W_s)^2 is proportional to gamma_j
    def want(m_out):
        W_s_ = GAMMAS[0] * m_out[1] + GAMMAS[1] * m_out[3]
        return 0.5 * KAPPA * (1.5 - g_G * W_s_) ** 2
    grad = jax.grad(want)(jnp.array([w[0] * V, w[0], w[1] * V, w[1]]))
    np.testing.assert_allclose(grad[1] / grad[3], GAMMAS[0] / GAMMAS[1], rtol=1e-6)
    # the forcing of the whole is still -d of one function: the composite is exact
    rng = np.random.default_rng(5)
    for _ in range(3):
        qq = jnp.asarray(rng.standard_normal(W.P.Q_dim))
        xi, _, _ = W.omega(qq, jnp.zeros(0), ZERO_IN)
        assert bool(jnp.all(jnp.isfinite(xi)))


def test_a_price_charged_above_reaches_a_sender_scaled_by_the_gammas_between():
    """ILLUSTRATES who pays: a price ``c`` paid by each sender on its own loudness, a price ``c_G`` charged by the group
    on the weight it sends out, and a price ``c_O`` charged by the outer arrangement on what the group sends it, give the
    same forcing on the whole when ``c = c_G gamma_j = c_O gamma_j`` for each sender ``j``: the price pulls back along
    the transfers the message passed through to the sender's matter, which is what moves (``prop.exact_lenses``). The
    outer level's own ``gamma_o`` does not enter: a cost function is of the messages the internal boxes send, before
    that level's transfers act (``def.attention_arrangement``), so ``c_O = c_G``, not ``c_G / gamma_o`` as this run
    first wrote. With unequal ``gamma_j`` one price above is unequal prices below."""
    rd, rng = acknowledging_reader(), np.random.default_rng(9)
    g1, g2 = GAMMAS
    c_G = 0.04                                               # so sender 1 pays c_G g1 = 0.04, sender 2 pays c_G g2 = 0.02
    c_O = c_G                                                # not c_G / GAMMA_O: see the docstring
    Wa = nest([hearing_sender(c=c_G * g1, label="s1"), hearing_sender(c=c_G * g2, label="s2")], rd,
              group(GAMMAS, BETAS), outer(GAMMA_O, GAMMA_B))
    Wb = nest([hearing_sender(c=0.0, label="s1"), hearing_sender(c=0.0, label="s2")], rd,
              group(GAMMAS, BETAS, price=c_G), outer(GAMMA_O, GAMMA_B))
    Wc = nest([hearing_sender(c=0.0, label="s1"), hearing_sender(c=0.0, label="s2")], rd,
              group(GAMMAS, BETAS), outer(GAMMA_O, GAMMA_B, price=c_O))
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(Wa.P.Q_dim))
        fa = Wa.omega(q, jnp.zeros(0), ZERO_IN)
        for Wx in (Wb, Wc):
            for x, y in zip(fa, Wx.omega(q, jnp.zeros(0), ZERO_IN)):
                np.testing.assert_allclose(x, y, atol=1e-11)
    # and an equal price above is not an equal price below
    Wd = nest([hearing_sender(c=c_G, label="s1"), hearing_sender(c=c_G, label="s2")], rd,
              group(GAMMAS, BETAS), outer(GAMMA_O, GAMMA_B))
    q = jnp.asarray(rng.standard_normal(Wd.P.Q_dim))
    xd, xb = Wd.omega(q, jnp.zeros(0), ZERO_IN)[0], Wb.omega(q, jnp.zeros(0), ZERO_IN)[0]
    assert float(jnp.abs(xd[3] - xb[3])) > 1e-3             # sender 2's weight is charged differently
