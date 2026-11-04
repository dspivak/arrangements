"""A price on the weight a part sends: one sender, one reader, one exact attention arrangement (book plan P71 (b)).

The book's Ch 4 ``def.attention_arrangement`` lets a cost function push on the messages the internal boxes send. P71 (b)
asks what happens when a reader pays the Gaussian code length of what arrives and the sender pays a price on the
weight it sends: does the sender fall quiet when the reader already expects its value, and speak up when it does not?

**What runs in ``arr``.** Values are numbers (``dim_H = 1``), there are no positions (``dim_P = 0``). Three pieces:

* the **sender**, a part filling the interface ``B`` (an ``Arrangement``, not an attention operation: a 0-ary
  attention arrangement reads out the empty sum ``0_M`` and so sends nothing). Its matter is ``(v, lam)``, its
  value and its log-weight; it reads out the message ``(w v, w)`` with ``w = exp(lam)``, so what it sends lies in the
  message cone and silence is the limit ``lam -> -inf``; its cost is the price ``c w`` plus an anchor
  ``kappa_s/2 (v - v_0)^2`` holding its value;
* the **reader**, a 0-ary attention arrangement whose matter is its belief ``mu``; its cost is the Gaussian code
  length of what arrives under the belief ``N(mu, 1/w_r)``, the arriving message ``(h, w)`` read as ``w`` observations
  of the value ``h/w``: ``w [ w_r/2 (h/w - mu)^2 + kappa_r ]``, ``kappa_r`` the per-observation constant
  ``log(2 pi / w_r)/2 - log delta`` of a code length (P71 (a)), plus its prior ``p_r/2 (mu - mu_0)^2``;
* the **group**, a 2-ary attention arrangement with trivial matter and one transfer, the reader hears the sender
  through the identity; its cost is ``0`` (``group_price`` puts the price ``c wt(m_1)`` here instead, a degree-one
  polynomial in the messages).

The reader fills the group's second box by operad substitution; the sender fills the first by ``compose_seq``; the
whole is a system ``1 -> B`` run by ``Phi`` (``open_step``) with the zero message at its external input.

**Checked.** The forcing of the whole is ``-d`` of one function of ``(v, lam, mu)``, the three costs summed at the
delivered points (``prop.exact_lenses``); the pair as the operad's substitution and as the lens composite agree
(``prop.attention_operad``), and the unit law holds; the price paid by the sender and the same price charged by the
group give the same forcing. Under ``Phi`` the sender always falls quiet, faster the more the reader disagrees, and
the reader's belief returns to its prior as the message fades; a slow-to-quiet sender moves the reader further. The
variant in which the weight is the sender's precision (the code length ``w/2 (v - mu)^2 - log(w)/2``) has the rest
weight ``1/(2c + (v - mu)^2)``: loudest when the reader already agrees.

**Mechanism** (by hand, in the module's ``RESULTS``): in one exact arrangement the term that makes the weight pull the
reader toward the sender's value, ``w f(mu - v)``, is the term that charges the sender ``f(mu - v)`` per unit of
weight; mixed partials are symmetric (Ch 2 ``prop.closed``), so the weight at rest is a nonincreasing function of
the disagreement, whatever the price. Results and the judgment: the book's ``notes/runs/p71b-fcfaae81/RESULTS.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    attention_arrangement, identity_transfer, substitute, unit_arrangement, zero_transfer,
)
from arrangements.coalgebra import configuration
from arrangements.functors import Phi, open_step
from arrangements.rvect import diagonal, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

DH, DP = 1, 0                       # numbers for values, no positions
C, KAPPA_S, W_R, KAPPA_R, P_R = 0.1, 1.0, 1.0, 0.5, 1.0
ZERO_IN = jnp.zeros(2)              # the zero message at the group's external input


# ---------------------------------------------------------------------------
# The three pieces.
# ---------------------------------------------------------------------------


def sender(v0, lam0, c=C, kappa_s=KAPPA_S, eta_v=0.1, eta_lam=0.1, price="linear", label="sender") -> Arrangement:
    """The sender, a part filling ``B``: matter ``(v, lam)``, readout ``(e^lam v, e^lam)``, cost ``c w`` (or ``c w^2``)
    plus ``kappa_s/2 (v - v0)^2``. With ``price="none"`` the sender pays nothing for its weight."""
    def cost(q, m_out, n_in):
        v, w = q[0], jnp.exp(q[1])
        pay = {"linear": c * w, "quadratic": c * w ** 2, "none": 0.0}[price]
        return pay + 0.5 * kappa_s * (v - v0) ** 2

    return Arrangement(
        P=configuration(diagonal(jnp.array([eta_v, eta_lam])), init=jnp.array([v0, lam0], dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=2,
        out_f=lambda q, m_out: jnp.array([jnp.exp(q[1]) * q[0], jnp.exp(q[1])]),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, m_out, n_in)),
        label=label,
    )


def reader(mu0, w_r=W_R, kappa_r=KAPPA_R, p_r=P_R, eta_mu=0.1, weight_is="length"):
    """The reader, a 0-ary attention arrangement with matter ``mu``.

    ``weight_is="length"``: the arriving ``(h, w)`` is ``w`` observations of ``h/w`` under ``N(mu, 1/w_r)``, code
    length ``w [w_r/2 (h/w - mu)^2 + kappa_r]``. ``weight_is="precision"``: the arriving message is the Gaussian
    ``N(h/w, 1/w)`` and the reader pays the code length of its belief under it, ``w/2 (h/w - mu)^2 - log(w)/2``.
    Both are smooth where ``w > 0`` and singular at the apex ``w = 0``, where a message has no value.
    """
    def cost(q, m_out, n_in):
        mu, h, w = q[0], n_in[0], n_in[1]
        err2 = (h - w * mu) ** 2 / w          # = w (h/w - mu)^2
        if weight_is == "length":
            length = 0.5 * w_r * err2 + kappa_r * w
        else:
            length = 0.5 * err2 - 0.5 * jnp.log(w)
        return length + 0.5 * p_r * (mu - mu0) ** 2

    Z = zero_transfer(DH)
    return attention_arrangement(
        0, DH, DP, lambda q: jnp.zeros((0, DP)), lambda q: ((Z,),), cost=cost,
        parameter=configuration(euclidean(1, eta_mu), init=jnp.array([mu0], dtype=float)), label="reader",
    )


def group(price=None):
    """The 2-ary attention arrangement: box 2 (the reader) hears box 1 (the sender) through the identity; nothing is
    heard from or sent outside. Cost ``0``, or the price ``c wt(m_1)`` on the sender's weight when ``price`` is given."""
    Z, I = zero_transfer(DH), identity_transfer(DH)
    cost = (lambda q, m_out, n_in: price * m_out[1]) if price is not None else None
    kw = {"cost": cost} if cost is not None else {}
    return attention_arrangement(
        2, DH, DP, lambda q: jnp.zeros((2, DP)), lambda q: ((Z, Z, Z), (Z, Z, Z), (Z, I, Z)), label="group", **kw,
    )


def pair_operation(rd, grp=None):
    """The 1-ary attention operation ``group(1, reader)``: the reader substituted, the sender's box left open."""
    return substitute(grp or group(), [unit_arrangement(DH, DP), rd], label="group(1, reader)")


def whole(snd, rd, grp=None, by="substitute"):
    """The system ``1 -> B``: the sender filling the open box of the pair, the pair either as the operad's
    substitution (one box) or as the lens composite of the unit, the reader and the group."""
    if by == "substitute":
        op = pair_operation(rd, grp).arr()
    else:
        op = compose_seq(tensor_arrangements([unit_arrangement(DH, DP).arr(), rd.arr()]), (grp or group()).arr())
    return compose_seq(snd, op)


def hand_cost(q, v0, mu0, c=C, kappa_s=KAPPA_S, w_r=W_R, kappa_r=KAPPA_R, p_r=P_R):
    """The three costs summed at the delivered points, by hand, as a function of ``(v, lam, mu)``."""
    v, lam, mu = q
    w = jnp.exp(lam)
    return (c * w + 0.5 * kappa_s * (v - v0) ** 2
            + w * (0.5 * w_r * (v - mu) ** 2 + kappa_r)
            + 0.5 * p_r * (mu - mu0) ** 2)


def run(arr, steps):
    """``Phi(arr)`` told the zero message and no force at its outputs, ``steps`` unit steps; the flat states
    ``(v, lam, mu)`` at every step."""
    step = open_step(arr, ZERO_IN)
    s, out = arr.P.init, [flat_state(arr.P.init)]
    for _ in range(steps):
        s = step(s)
        out.append(flat_state(s))
    return jnp.stack(out)


# ---------------------------------------------------------------------------
# The costs nest, and the whole descends one function.
# ---------------------------------------------------------------------------


def test_forcing_is_minus_d_of_the_summed_costs():
    """REPRODUCES ``prop.exact_lenses`` on this nest: the whole's forcing on ``(v, lam, mu)`` is ``-d`` of the
    price plus the reader's length at the delivered message plus the anchors, at several points."""
    v0, mu0 = 1.0, 0.0
    W = whole(sender(v0, 0.0), reader(mu0))
    r = np.random.default_rng(1)
    for _ in range(5):
        q = jnp.asarray(r.standard_normal(3))
        xi_q, _, _ = W.omega(q, jnp.zeros(0), ZERO_IN)
        np.testing.assert_allclose(xi_q, -jax.grad(hand_cost)(q, v0, mu0), atol=1e-10)


def test_substitution_agrees_with_the_lens_composite_and_the_unit():
    """REPRODUCES ``prop.attention_operad`` for this pair: ``group(1, reader)`` by the explicit substitution equals the
    lens composite of the unit, the reader and the group, in readout, delivery, forcing and trajectory, with the sender
    filling the open box either way; and substituting the pair into the unit, or the unit into its open box, changes
    nothing (the unit laws), the costs summing with ``0`` for the unit."""
    v0, mu0 = 1.0, -0.5
    rd = reader(mu0)
    W1, W2 = whole(sender(v0, 0.2), rd, by="substitute"), whole(sender(v0, 0.2), rd, by="compose")
    r = np.random.default_rng(2)
    for _ in range(5):
        q = jnp.asarray(r.standard_normal(3))
        np.testing.assert_allclose(W1.out_f(q, jnp.zeros(0)), W2.out_f(q, jnp.zeros(0)), atol=1e-12)
        for x, y in zip(W1.omega(q, jnp.zeros(0), ZERO_IN), W2.omega(q, jnp.zeros(0), ZERO_IN)):
            np.testing.assert_allclose(x, y, atol=1e-11)
    np.testing.assert_allclose(run(W1, 50), run(W2, 50), atol=1e-11)

    op = pair_operation(rd)
    u = unit_arrangement(DH, DP)
    outer = substitute(u, [op], label="1(group(1, reader))")
    inner = substitute(op, [u], label="group(1, reader)(1)")
    for _ in range(4):
        q = jnp.asarray(r.standard_normal(1))      # the reader's mu; the units have no configuration
        m = jnp.asarray(r.standard_normal(2))      # the message in the open box
        e = jnp.asarray(r.standard_normal(2))      # the external input
        for other in (outer, inner):
            np.testing.assert_allclose(op.cost(q, m, e), other.cost(q, m, e), atol=1e-12)
            np.testing.assert_allclose(op.sigma(q, m), other.sigma(q, m), atol=1e-12)
            np.testing.assert_allclose(op.rho(q, m, e), other.rho(q, m, e), atol=1e-12)
    # the pair's cost is the reader's at the delivered message, the group's being 0
    q, m, e = jnp.array([0.3]), jnp.array([0.8, 0.4]), jnp.zeros(2)
    np.testing.assert_allclose(op.cost(q, m, e), rd.cost(q, jnp.zeros(0), m), atol=1e-12)


def test_price_paid_by_the_sender_equals_price_charged_by_the_group():
    """ILLUSTRATES one home for the price: the sender paying ``c w`` on its own configuration, and the group
    charging ``c wt(m_1)`` on the message it carries, give the same forcing on the whole, since the pushback on the
    message's weight lands on the sender's matter through its readout."""
    v0, mu0 = 1.0, 0.0
    Wa = whole(sender(v0, 0.0), reader(mu0))
    Wb = whole(sender(v0, 0.0, price="none"), reader(mu0), grp=group(price=C))
    r = np.random.default_rng(3)
    for _ in range(5):
        q = jnp.asarray(r.standard_normal(3))
        for x, y in zip(Wa.omega(q, jnp.zeros(0), ZERO_IN), Wb.omega(q, jnp.zeros(0), ZERO_IN)):
            np.testing.assert_allclose(x, y, atol=1e-11)


# ---------------------------------------------------------------------------
# What the pair does under Phi.
# ---------------------------------------------------------------------------


def test_sender_falls_quiet_faster_the_more_the_reader_disagrees():
    """CHECKS THE HAND RESULT: with ``v`` and ``mu`` held (zero mobility) the log-weight obeys
    ``lam' = -eta w (c + kappa_r + w_r/2 (v - mu)^2)``, so ``1/w`` grows linearly at the rate in brackets: silence
    is approached, never reached, and sooner the larger the disagreement. Nothing ever raises the weight."""
    for v0, mu0 in [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0)]:
        W = whole(sender(v0, 0.0, eta_v=0.0, eta_lam=0.05), reader(mu0, eta_mu=0.0))
        traj = run(W, 400)
        w = jnp.exp(traj[:, 1])
        assert bool(jnp.all(jnp.diff(traj[:, 1]) < 0))                       # the weight only falls
        rate = C + KAPPA_R + 0.5 * W_R * (v0 - mu0) ** 2
        inv_w_hand = 1.0 / w[0] + 0.05 * rate * jnp.arange(401)             # Euler on 1/w, to first order
        np.testing.assert_allclose(1.0 / w[-1], inv_w_hand[-1], rtol=0.03)
    # the three disagreements, final weights in order: more disagreement, quieter
    finals = []
    for v0 in (0.0, 1.0, 2.0):
        W = whole(sender(v0, 0.0, eta_v=0.0, eta_lam=0.05), reader(0.0, eta_mu=0.0))
        finals.append(float(jnp.exp(run(W, 400)[-1, 1])))
    assert finals[0] > finals[1] > finals[2]


def test_reader_moves_while_the_message_lasts_and_returns_to_its_prior():
    """ILLUSTRATES the transient: with everything mobile, the reader's belief is pulled toward the sender's value
    while the weight lasts and drifts back to its prior as the sender quiets; the sender's value is pulled toward the
    reader's belief by the same term (reciprocity) and returns to its anchor. A sender slow to quiet moves the reader
    further than a quick one."""
    v0, mu0 = 1.0, 0.0
    far = []
    for eta_lam in (0.5, 0.02):
        W = whole(sender(v0, 1.0, eta_v=0.05, eta_lam=eta_lam), reader(mu0, eta_mu=0.1))
        traj = run(W, 3000)
        mu, v, w = traj[:, 2], traj[:, 0], jnp.exp(traj[:, 1])
        assert float(jnp.max(mu)) > 0.05                                   # the reader moved toward v = 1
        assert float(jnp.min(v)) < v0 - 0.01                               # and the sender toward mu = 0
        assert abs(float(mu[-1])) < 0.5 * float(jnp.max(mu))               # and is on its way back
        assert float(w[-1]) < 0.1 * float(w[0])
        far.append(float(jnp.max(mu)))
    assert far[1] > far[0]                                                  # the patient sender gets further


def test_weight_as_precision_rests_loudest_on_agreement():
    """CHECKS THE HAND RESULT for the other reading of weight: with the reader paying ``w/2 (v - mu)^2 - log(w)/2`` and
    the sender ``c w``, the weight comes to rest at ``1/(2c + (v - mu)^2)``, so a sender the reader already agrees with
    is the loudest."""
    for v0 in (0.0, 1.0, 3.0):
        W = whole(sender(v0, 0.0, eta_v=0.0, eta_lam=0.1), reader(0.0, eta_mu=0.0, weight_is="precision"))
        traj = run(W, 1500)
        np.testing.assert_allclose(jnp.exp(traj[-1, 1]), 1.0 / (2 * C + v0 ** 2), rtol=1e-3)
