"""A sender that holds an estimate of its return (book plan P71 (b), second part; David: "2. isn't that beautiful? is it
like dopamine? please check").

`test_weight_price.py` found that in one exact arrangement under a symmetric response a sender's weight never rises with
the reader's disagreement, since the term that lets the weight pull the reader charges the sender for it. David's
answer: the sender gets something for sending, later ("pass the salt" brings the salt). The reading taken ("2"): the
sender holds a prediction of its return as matter and pays now for the predicted shortfall; the prediction is corrected
when the return arrives.

**What runs in ``arr``.** One exact arrangement, the sender, a system ``1 -> <R | M>``: it receives the return ``r``
(the salt) at its input and sends the message ``(w v, w)``, ``w = exp(lam)``, its value ``v`` fixed. Its matter is its
log-weight ``lam`` and its estimate of the return, ``pi = theta_0 + theta_1 w``, a baseline plus a return per unit of
weight. Its cost is

    c w  +  kappa/2 (want - pi)^2  +  1/2 (r - pi)^2 :

the price, the predicted shortfall, and the prediction error. The force on the weight is
``-c + theta_1 [kappa (want - pi) + (r - pi)]``: the sender speaks up when it predicts that weight brings salt it is short
of, and more when the salt exceeds the prediction.

**Two kinds of matter for the estimate.** *Symmetric* (``unilateral=False``): the estimate descends the whole cost, so
the want pushes it up as the prediction error pushes it toward the return (wishful thinking), and it rests at
``pi = (kappa want + r)/(1 + kappa)``, exactly where the bracket in the force on the weight vanishes: raising the
estimate is free and raising the weight costs ``c``, so the sender believes instead of asking and falls silent whatever
the reader does. *Unilateral* (``unilateral=True``, the default): the estimate's matter carries shadow coordinates
``phi``, frozen at ``0``, the prediction error is written on ``theta + phi``, and the response moves ``theta`` by the
covector on ``phi`` (the kernel condition of ``amplifier.py``'s stage and ``logic.py``'s MOSFET), so the estimate
takes responsibility for the prediction error alone; the want's push on it lands in the kernel. The forcing is still
``-d`` of one cost; only the response is not symmetric, which the book's integrator allows (Ch 3 ``def.arrangement``,
and the glossary's *Unilateral*). This is the actor and critic of reinforcement learning, the critic kept apart.

**What is the environment's.** The return schedule: the salt arriving at step ``t`` is a rule applied to the weight
sent ``DELAY`` steps earlier, as the experimenter's juice in a conditioning experiment. Four schedules: *deaf*
(``r = 0``), *attentive* (``r = w/(1+w)``), *generous* (``r = 1`` whatever is sent), and *attentive then deaf*. The
rule is computed outside the arrangement, between its steps; nothing of the sender's is.

**Checked.** The forcing is ``-d`` of the cost (``jax.grad`` of the sum written by hand); the compiled step equals
``Phi``'s. Symmetric: the estimate rests at the wishful value and the weight falls under every schedule. Unilateral,
deaf: the weight falls (extinction) and the estimate rests at ``0``. Attentive: the weight rises from a whisper and
rests where ``kappa (want - r) theta_1 = c`` with the estimate equal to the return; the prediction error is positive
early and ``0`` at rest. Generous: the weight falls, the estimate rests at ``1``. Attentive then deaf: the prediction
error dips below ``0`` at the withdrawal and returns to ``0`` as the estimate relearns, and the weight falls. Against
dopamine's reward prediction error (Schultz, Dayan and Montague, Science 1997; Montague, Dayan and Sejnowski, J.
Neurosci. 1996): the burst at an unpredicted return, the silence at a predicted one and the dip at a withheld one are
here; the shift of the response to the earliest predictor is not, since this sender predicts its return from its
present message only (the Rescorla–Wagner error, not the temporal-difference one). Results and the judgment: the
book's ``notes/runs/p71b-fcfaae81/RESULTS.md`` §5.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.rvect import constant

C, KAPPA, WANT, V = 0.1, 0.5, 1.0, 1.0
DELAY = 3


def held_estimate_sender(lam0=-2.0, th0=0.0, th1=0.0, c=C, kappa=KAPPA, want=WANT,
                         eta_lam=0.1, eta_theta=0.1, unilateral=True, decay=0.0, label="sender") -> Arrangement:
    """The sender: matter ``(lam, theta_0, theta_1)`` and, when unilateral, the shadow ``(phi_0, phi_1)`` frozen at
    ``0``; input the return ``r``; output the message ``(e^lam v, e^lam)``. ``decay`` adds ``decay/2 theta_1^2`` to the
    estimate's own cost (Ch 2 ``exc.nn_decay``'s weight decay), so the return per unit of weight is forgotten without
    evidence."""
    def cost(q, m_out, n_in):
        lam, th = q[0], q[1:3]
        w = jnp.exp(lam)
        pi_want = th[0] + th[1] * w                      # the estimate the want reads
        if unilateral:
            tp = th + q[3:5]                             # theta + phi, the prediction error's estimate
            pi_err = tp[0] + tp[1] * w
        else:
            tp, pi_err = th, pi_want
        return (c * w + 0.5 * kappa * (want - pi_want) ** 2 + 0.5 * (n_in[0] - pi_err) ** 2
                + 0.5 * decay * tp[1] ** 2)

    if unilateral:
        S = jnp.zeros((5, 5)).at[0, 0].set(eta_lam).at[1, 3].set(eta_theta).at[2, 4].set(eta_theta)
        init = jnp.array([lam0, th0, th1, 0.0, 0.0], dtype=float)
    else:
        S = jnp.diag(jnp.array([eta_lam, eta_theta, eta_theta]))
        init = jnp.array([lam0, th0, th1], dtype=float)
    return Arrangement(
        P=configuration(constant(S), init=init),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=2,
        out_f=lambda q, m_out: jnp.array([jnp.exp(q[0]) * V, jnp.exp(q[0])]),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, m_out, n_in)),
        label=label,
    )


def hand_cost(q, r, c=C, kappa=KAPPA, want=WANT):
    """The unilateral sender's cost by hand, ``q = (lam, theta_0, theta_1, phi_0, phi_1)``."""
    w = jnp.exp(q[0])
    pi_want = q[1] + q[2] * w
    pi_err = (q[1] + q[3]) + (q[2] + q[4]) * w
    return c * w + 0.5 * kappa * (want - pi_want) ** 2 + 0.5 * (r - pi_err) ** 2


def estimate(q):
    return q[1] + q[2] * jnp.exp(q[0])


SCHEDULES = {
    "deaf": lambda w, t: 0.0,
    "attentive": lambda w, t: w / (1.0 + w),
    "generous": lambda w, t: 1.0,
    "attentive_then_deaf": lambda w, t: (w / (1.0 + w)) if t < 1500 else 0.0,
}


def run(arr, schedule, steps, delay=DELAY, check=2, estimate_fn=None):
    """``Phi(arr)`` stepped with the return the environment's rule gives for the weight sent ``delay`` steps earlier (``0``
    before any was). Returns the states, the returns ``r`` and the prediction errors ``r - pi`` at each step. The
    compiled step is checked against ``Phi.run_one`` on its first ``check`` steps."""
    P, omega, z = arr.P, arr.omega, jnp.zeros(0)

    @jax.jit
    def step(s, r):
        q = P.position(s)
        xi_q, _, _ = omega(q, z, jnp.array([r]))
        return P.step(s, xi_q)

    est = estimate if estimate_fn is None else estimate_fn
    O, IN = Phi(arr), (z, trivial_omega(0))
    s = P.init
    weights, states, returns, errors = [], [s], [], []
    for t in range(steps):
        w_now = float(jnp.exp(P.position(s)[0]))
        weights.append(w_now)
        w_then = weights[t - delay] if t >= delay else 0.0
        r = float(schedule(w_then, t))
        if t < check:
            _, _, s_phi = O.with_state(s).run_one(IN, lambda _o: (jnp.zeros(2), jnp.array([r])))
        errors.append(r - float(est(P.position(s))))
        returns.append(r)
        s = step(s, r)
        if t < check:
            np.testing.assert_allclose(s, s_phi, rtol=1e-10, atol=1e-12)
        states.append(s)
    return jnp.stack(states), jnp.asarray(returns), jnp.asarray(errors)


def test_forcing_is_minus_d_of_the_cost():
    """REPRODUCES ``def.system`` (v) for this sender: its forcing is ``-d`` of the price plus the predicted shortfall plus
    the prediction error, on all five coordinates; and the shadow coordinates never move."""
    a = held_estimate_sender()
    rng = np.random.default_rng(4)
    for _ in range(5):
        q, r = jnp.asarray(rng.standard_normal(5)), float(rng.standard_normal())
        xi_q, _, _ = a.omega(q, jnp.zeros(0), jnp.array([r]))
        np.testing.assert_allclose(xi_q, -jax.grad(hand_cost)(q, r), atol=1e-10)
    st, _, _ = run(a, SCHEDULES["attentive"], 20)
    np.testing.assert_allclose(st[:, 3:5], 0.0, atol=0.0)


def test_symmetric_estimate_believes_instead_of_asking():
    """CHECKS THE HAND RESULT for the symmetric matter: under every schedule the estimate rests at the wishful
    ``(kappa want + r)/(1 + kappa)``, the bracket in the force on the weight vanishes there, and the weight falls under
    the price alone, attentive reader or not."""
    for name, r_rest in [("deaf", 0.0), ("generous", 1.0)]:
        a = held_estimate_sender(lam0=0.0, th0=0.3, th1=0.3, unilateral=False)
        st, r, err = run(a, SCHEDULES[name], 3000)
        np.testing.assert_allclose(estimate(st[-1]), (KAPPA * WANT + r_rest) / (1 + KAPPA), atol=1e-3)
        assert float(jnp.exp(st[-1, 0])) < 0.05
    a = held_estimate_sender(lam0=-2.0, th0=0.0, th1=0.5, unilateral=False)
    st, r, err = run(a, SCHEDULES["attentive"], 4000)
    w = jnp.exp(st[:, 0])
    assert float(jnp.max(w)) < 1.1 * float(w[0]) and float(w[-1]) < 0.2 * float(w[0])   # never spoke up


def test_deaf_reader_extinguishes_the_asking():
    """CHECKS THE HAND RESULT: with no return ever, the estimate rests at ``0`` and the weight falls (extinction); the
    first prediction error is a dip, since the sender started expecting something."""
    st, r, err = run(held_estimate_sender(lam0=0.0, th0=0.3, th1=0.3), SCHEDULES["deaf"], 3000)
    w = jnp.exp(st[:, 0])
    assert float(w[-1]) < 0.05 * float(w[0])
    np.testing.assert_allclose(estimate(st[-1]), 0.0, atol=1e-3)
    assert float(err[0]) < 0


def test_attentive_reader_makes_the_sender_speak_up_and_the_error_vanish():
    """CHECKS THE HAND RESULT: from a whisper, the weight rises and rests where ``kappa (want - r(w)) theta_1 = c`` with
    the estimate equal to the return; the prediction error is positive while the sender learns that speaking brings
    salt, and ``0`` at rest."""
    st, r, err = run(held_estimate_sender(lam0=-2.0, th0=0.0, th1=0.5), SCHEDULES["attentive"], 4000)
    w = jnp.exp(st[:, 0])
    assert float(w[-1]) > 5 * float(w[0])                   # spoke up
    q = st[-1]
    np.testing.assert_allclose(err[-1], 0.0, atol=2e-3)     # predicted
    np.testing.assert_allclose(KAPPA * (WANT - r[-1]) * q[2], C, rtol=0.02)
    assert float(jnp.max(err[:300])) > 0.05                 # a burst early, when the salt exceeded the estimate


def test_generous_reader_lets_the_sender_fall_quiet():
    """CHECKS THE HAND RESULT: salt whatever is sent: the estimate rests at the return and the weight falls under the
    price alone; the first return is unpredicted, a burst."""
    st, r, err = run(held_estimate_sender(lam0=0.0, th0=0.0, th1=0.5), SCHEDULES["generous"], 3000)
    w = jnp.exp(st[:, 0])
    assert float(w[-1]) < 0.05 * float(w[0])
    np.testing.assert_allclose(estimate(st[-1]), 1.0, atol=1e-3)
    assert float(err[0]) >= 0.5                             # the estimate was 0.5, the salt 1


def test_withheld_return_gives_a_dip_and_the_asking_persists_as_a_habit():
    """ILLUSTRATES the omission response, and a finding: attentive for 1500 steps, then deaf. The prediction error, near
    ``0`` once learned, dips below ``0`` when the salt stops and returns to ``0`` as the estimate relearns; but the
    weight stays up. The estimate has two parameters and is corrected at one operating point, so it relearns the level
    (``pi = 0``) along the line ``theta_0 + theta_1 w = 0`` and keeps a positive return per unit of weight, which keeps
    ``kappa (want - pi) theta_1 = c`` satisfied: the sender goes on asking a deaf reader, a habit its evidence cannot
    break while its weight does not vary."""
    st, r, err = run(held_estimate_sender(lam0=-2.0, th0=0.0, th1=0.5), SCHEDULES["attentive_then_deaf"], 4000)
    w = jnp.exp(st[:, 0])
    assert abs(float(err[1499])) < 2e-2                     # predicted, just before the withdrawal
    assert float(jnp.min(err[1500:1520])) < -0.1            # the dip
    assert abs(float(err[-1])) < 1e-2                       # relearned
    assert float(st[-1, 2]) > 0.1                           # a return per unit of weight remains
    assert float(w[-1]) > 0.7 * float(w[1499])              # and the asking goes on
    np.testing.assert_allclose(KAPPA * (WANT - 0.0) * st[-1, 2], C, rtol=0.02)


def test_decay_on_the_slope_lets_the_habit_extinguish():
    """CHECKS THE HAND RESULT: with Ch 2's weight decay on the return per unit of weight (``decay/2 theta_1^2`` in the
    estimate's cost), the slope is forgotten without evidence: after the withdrawal ``theta_1`` falls, the force on the
    weight becomes ``-c``, and the sender falls quiet; before it, the attentive rest stands, shrunk a little. The decay
    is a trade: at ``0.02`` the slope is forgotten faster than one operating point can teach it and the sender never
    speaks up (``w`` falls from ``0.135`` to ``0.085`` while the salt is on offer); at ``0.002`` it speaks up by a factor
    ``7`` and, the salt withheld, is down by a factor ``27`` after 6500 steps."""
    st, r, err = run(held_estimate_sender(lam0=-2.0, th0=0.0, th1=0.5, decay=0.002), SCHEDULES["attentive_then_deaf"], 8000)
    w = jnp.exp(st[:, 0])
    assert float(w[1499]) > 5 * float(w[0])                 # spoke up while it paid
    assert float(jnp.min(err[1500:1520])) < -0.1            # the dip
    assert float(st[-1, 2]) < 0.1                           # the slope mostly forgotten
    assert float(w[-1]) < 0.05 * float(w[1499])             # quiet again


# ---------------------------------------------------------------------------
# David's picture: the sender wants to be heard, and estimates how loudly the reader hears it.
# ---------------------------------------------------------------------------
#
# David (2026-10-23, to the salt estimate): "certainly not 'i expect this much salt': i don't want mountains of salt on
# my soup! i might want to know if they guy is deaf or far away. if they're either, i talk louder. i want them to hear
# me. that's it. i don't want to be too loud, and i don't think shouting at them is going to get the job done better
# unless they wouldn't otherwise hear".
#
# So the sender holds one number, its estimate ``g`` of how loudly the reader hears it, the ``gamma`` of Ch 4's
# transfer ``T_{gamma,V}``; its want is that its message arrive with weight ``HEARD``; what comes back is the arriving
# weight ``r = gamma w`` (the reader's acknowledgment), which corrects the estimate. Its cost is
#
#     c w  +  kappa/2 (HEARD - g w)^2  +  1/2 (r - g w)^2 .
#
# At rest ``g = gamma`` and the loudness is ``w* = HEARD/gamma - c/(kappa gamma^2)``, so the arriving weight is
# ``gamma w* = HEARD - c/(kappa gamma)``: near (``gamma`` large) the sender is heard at nearly the weight it wants and
# speaks softly; the farther the reader, the louder it speaks and the less of it arrives; at ``gamma = c/(kappa HEARD)``
# speaking stops paying and below it, the deaf reader included, the sender is silent: shouting at the deaf does not get
# the job done. One number, so the estimate is identified at any one loudness and the habit of the salt estimate cannot
# form. The want still pushes the estimate (wishful hearing), so the
# estimate's matter is unilateral as above; with a symmetric response the sender believes it is heard and falls quiet.

HEARD, C_HEAR = 1.0, 0.02
GAMMAS = {"near": 1.0, "far": 0.25, "deaf": 0.0}


def hearing_sender(lam0=-2.0, g0=0.5, c=C_HEAR, kappa=KAPPA, heard=HEARD, eta_lam=0.1, eta_g=0.1,
                   unilateral=True, label="sender") -> Arrangement:
    """The sender of David's picture: matter its log-weight and its estimate ``g`` of how loudly it is heard (with a
    shadow coordinate when unilateral); input the arriving weight ``r``; output the message ``(e^lam v, e^lam)``."""
    def cost(q, m_out, n_in):
        w, g = jnp.exp(q[0]), q[1]
        g_err = g + q[2] if unilateral else g
        return c * w + 0.5 * kappa * (heard - g * w) ** 2 + 0.5 * (n_in[0] - g_err * w) ** 2

    if unilateral:
        S = jnp.zeros((3, 3)).at[0, 0].set(eta_lam).at[1, 2].set(eta_g)
        init = jnp.array([lam0, g0, 0.0], dtype=float)
    else:
        S = jnp.diag(jnp.array([eta_lam, eta_g]))
        init = jnp.array([lam0, g0], dtype=float)
    return Arrangement(
        P=configuration(constant(S), init=init),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=2,
        out_f=lambda q, m_out: jnp.array([jnp.exp(q[0]) * V, jnp.exp(q[0])]),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q, m_out, n_in)),
        label=label,
    )


def hearing_rest(gamma, c=C_HEAR, kappa=KAPPA, heard=HEARD):
    """The loudness at rest, by hand: ``heard/gamma - c/(kappa gamma^2)`` where that is positive, else ``0``."""
    return max(0.0, heard / gamma - c / (kappa * gamma ** 2)) if gamma > 0 else 0.0


def hearing_estimate(q):
    return q[1] * jnp.exp(q[0])


def heard_schedule(gamma):
    return lambda w, t: gamma * w


def test_hearing_sender_talks_louder_when_far_and_not_at_all_when_deaf():
    """CHECKS THE HAND RESULT: near, far and deaf readers; the estimate comes to the reader's ``gamma`` and the loudness
    to ``HEARD/gamma - c/(kappa gamma^2)``: quieter near than far, and silent when deaf; the arriving weight is
    ``HEARD - c/(kappa gamma)``, so the far reader hears less of a louder sender."""
    rests = {}
    for name, gamma in GAMMAS.items():
        st, r, err = run(hearing_sender(), heard_schedule(gamma), 4000, estimate_fn=hearing_estimate)
        w, g = jnp.exp(st[:, 0]), st[:, 1]
        rests[name] = float(w[-1])
        if gamma > 0:
            np.testing.assert_allclose(g[-1], gamma, atol=2e-3)
            np.testing.assert_allclose(w[-1], hearing_rest(gamma), rtol=0.01)
            np.testing.assert_allclose(r[-1], HEARD - C_HEAR / (KAPPA * gamma), rtol=0.01)
            np.testing.assert_allclose(err[-1], 0.0, atol=1e-6)
        else:
            assert float(g[-1]) < 1e-2 and float(w[-1]) < 0.1
    assert rests["far"] > 3 * rests["near"] > 5 * rests["deaf"]


def test_hearing_sender_falls_quiet_when_the_reader_goes_deaf_no_habit():
    """CHECKS THE HAND RESULT: far for 1500 steps, then deaf. One number is identified at any one loudness, so the
    estimate falls to ``0`` and the loudness with it; the habit of the two-number salt estimate does not form."""
    sched = lambda w, t: (GAMMAS["far"] * w) if t < 1500 else 0.0
    st, r, err = run(hearing_sender(), sched, 6000, estimate_fn=hearing_estimate)
    w, g = jnp.exp(st[:, 0]), st[:, 1]
    np.testing.assert_allclose(w[1499], hearing_rest(GAMMAS["far"]), rtol=0.01)
    assert float(jnp.min(err[1500:1520])) < -0.3            # the acknowledgment stops: a dip
    assert float(g[-1]) < 1e-2 and float(w[-1]) < 0.1 * float(w[1499])


def test_symmetric_hearing_estimate_believes_it_is_heard():
    """CHECKS THE HAND RESULT for the symmetric matter: the want pushes the estimate up (wishful hearing) to where the
    force on the loudness vanishes, and the far reader is spoken to no louder than at the start."""
    st, r, err = run(hearing_sender(unilateral=False), heard_schedule(GAMMAS["far"]), 4000, estimate_fn=hearing_estimate)
    w, g = jnp.exp(st[:, 0]), st[:, 1]
    assert float(w[-1]) < 0.3 * hearing_rest(GAMMAS["far"])
    assert float(g[-1] * w[-1]) > float(r[-1]) + 0.1        # believes more arrives than does
