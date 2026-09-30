"""A redundant hand: a goal set at the top, a gap filled below (book ``AGENTDOCPLAN.md`` §5.9 item 43,
its sequel; Ch 1 ``sec.gap_filling`` and §1.7's gap-filling footnote in ``sec.check_claims``).

QUESTION. Item 43 (``active_inference.hierarchy``) showed a prediction discharged by the level
below through the pullback alone, but its hand is a scalar: every level predicts the same number,
so no level leaves anything unspecified and the book's "fillable gaps" (Lashley and Bernstein's
redundant joints) are not in it (the Ch 1 domain check, Group E2, "Gap-filling"). Does the chain,
with a hand that has a spare degree of freedom, set only what the top wants and leave the rest to
the level below?

MODEL. Item 43's chain with the scalar hand replaced by a two-joint hand ``I -> <0|1>``: state
``(q1, q2)``, Euclidean response, its own cost ``1/2 k1 (q1 - c1)^2 + 1/2 k2 (q2 - c2)^2`` (each
joint's spring toward a comfort posture ``(c1, c2)``), publishing only the endpoint
``e = q1 + q2``. Level 1 (``predictor``) predicts ``e``; level 2 predicts level 1 and wants ``g``.
Exact wiring by ``compose_seq``; nothing goes down but pullbacks.

PREDICTION (fixed in ``notes/ch1-tests-2026-09-29/redundant-hand.md`` in the book repository
before this file existed). (P1) the endpoint rests at item 43's chain of springs,
``e* = (kappa c + k_s g)/(kappa + k_s)`` with ``kappa = k1 k2/(k1 + k2)`` the joints in series,
``c = c1 + c2``, ``1/k_s = 1/Pi1 + 1/Pi2 + 1/pi_g``; (P2) the split along ``(1, -1)``, which the
endpoint does not see, is set by the joints alone: ``k1 (q1 - c1) = k2 (q2 - c2)`` at rest for
every goal and precision; (P3) a change of the comfort split, or of the stiffnesses at fixed
``kappa``, moves the split and nothing any level sees; (P4) the published value has zero
derivative along ``(1, -1)`` and the pullback from above is a multiple of ``(1, 1)``.

RESULT. All held (numbers in the note): at ``k1 = k2 = 2`` the rest is item 43's
``(0.5714, 0.7143, 0.8571)`` with each joint at ``0.2857``; at ``k1 = 1.5, k2 = 3`` the levels'
rests are the same to 1e-12 and the softer joint takes twice the share; the joint balance holds to
1e-12 at every goal tried; with equal joints the messages up the chain are identical, step by step,
across runs that differ only in the comfort split or the starting split. With a non-diagonal
response (joints coupled in how they move, ``test_rest_is_metric_free_path_is_not``) the rest is
unchanged but the top's pull does move the split on the way there (from comfort, the split swings
to -0.070 at step 20 of a reach whose endpoint goes to 0.571, then returns): "the top never acts on the
split" is a statement about the rest and about Euclidean response, not about every path.

WHAT IT DOES NOT SHOW. The hand still stops short of ``g`` (item 43's proportional offset; no
integrating level is built). The hand is linear, one-dimensional at the endpoint, with a quadratic
comfort cost; a planar two-link arm, whose null direction turns with the posture, is not run. The
level below that fills the gap is the hand's own cost, not a separate predicting level with its
own goal; and the redundancy is resolved by the cheapest posture, one of many rules a body might
use. The rendezvous parent (``tests/test_rendezvous.py``, which fixes only a separation) is not
composed under this chain here.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import active_inference as ai
from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq

P = 4.0  # every precision, as in item 43's tests


def joint_hand(k1, k2, c1, c2, response, init=(0.0, 0.0)) -> Arrangement:
    """Two joints ``(q1, q2)``, each with its spring to comfort ``c_i``; publishes ``q1 + q2``."""
    return Arrangement(
        P=configuration(response, init=jnp.asarray(init, dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q[:1] + q[1:],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in:
                    -0.5 * k1 * (q[0] - c1) ** 2 - 0.5 * k2 * (q[1] - c2) ** 2),
        label="joint-hand",
    )


def chain(k1, k2, c1, c2, g, init=(0.0, 0.0), response=None):
    eta = 1.0 / (k1 + k2 + 4 * P)
    h = joint_hand(k1, k2, c1, c2, euclidean(2, eta) if response is None else response, init)
    return compose_seq(compose_seq(h, ai.predictor(P, eta)),
                       ai.predictor(P, eta, goal=(P, g), top=True))


def messages(s):
    """What crosses each wire upward: the endpoint, then level 1's prediction; and level 2's state."""
    (q, mu1), mu2 = s
    return np.array([float(q[0] + q[1]), float(mu1[0]), float(mu2[0])])


def settle(k1, k2, c1, c2, g, steps=40000, **kw):
    (q, mu1), mu2 = s = ai.run(chain(k1, k2, c1, c2, g, **kw), steps)
    return np.array([float(q[0]), float(q[1])]), messages(s)


def closed_form(k1, k2, c1, c2, g):
    """Joints in series under item 43's chain of springs; the split by equal joint load."""
    kappa, c, ks = k1 * k2 / (k1 + k2), c1 + c2, 1.0 / (3.0 / P)
    e = (kappa * c + ks * g) / (kappa + ks)
    load = kappa * (e - c)
    mu1 = e + load / P
    mu2 = mu1 + load / P
    return np.array([c1 + load / k1, c2 + load / k2]), np.array([e, mu1, mu2])


CASES = [
    (2.0, 2.0, 0.0, 0.0, 1.0),
    (1.5, 3.0, 0.0, 0.0, 1.0),
    (2.0, 2.0, 0.3, -0.3, 1.0),
    (1.5, 3.0, 0.2, 0.1, -0.7),
    (10.0, 0.5, -0.4, 0.9, 2.0),
]


@pytest.mark.parametrize("k1,k2,c1,c2,g", CASES)
def test_P1_P2_rest_is_the_series_spring_and_the_joints_split_it(k1, k2, c1, c2, g):
    """(P1) endpoint and levels at the closed form; (P2) joints at equal load; 1e-10."""
    q, msg = settle(k1, k2, c1, c2, g)
    q_cf, msg_cf = closed_form(k1, k2, c1, c2, g)
    print("case", (k1, k2, c1, c2, g), "q", q, "messages", msg)
    assert np.max(np.abs(msg - msg_cf)) < 1e-10
    assert np.max(np.abs(q - q_cf)) < 1e-10
    assert abs(k1 * (q[0] - c1) - k2 * (q[1] - c2)) < 1e-10


def test_equal_joints_item_43_numbers():
    """At ``k1 = k2 = 2`` (series stiffness 1) the levels rest at item 43's numbers; 1e-10."""
    q, msg = settle(2.0, 2.0, 0.0, 0.0, 1.0)
    assert np.max(np.abs(msg - [0.5714285714285715, 0.7142857142857144, 0.8571428571428572])) < 1e-10
    assert np.max(np.abs(q - [0.2857142857142857] * 2)) < 1e-10


@pytest.mark.parametrize("g", [-2.0, 0.0, 0.5, 3.0])
def test_P2_the_goal_moves_the_endpoint_not_the_rule(g):
    """Equal joints: split ``q1 - q2`` is the comfort split at every goal; unequal: equal load."""
    q, msg = settle(2.0, 2.0, 0.3, -0.3, g)
    assert abs((q[0] - q[1]) - 0.6) < 1e-10
    q, msg = settle(1.5, 3.0, 0.3, -0.3, g)
    assert abs(1.5 * (q[0] - 0.3) - 3.0 * (q[1] + 0.3)) < 1e-10


def test_P3_what_the_top_sees_is_blind_to_the_split():
    """Comfort split changed at fixed ``c1 + c2``, and stiffnesses changed at fixed series
    stiffness: the split moves, every level's rest does not; 1e-10."""
    qa, ma = settle(2.0, 2.0, 0.0, 0.0, 1.0)
    qb, mb = settle(2.0, 2.0, 0.5, -0.5, 1.0)
    qc, mc = settle(1.5, 3.0, 0.0, 0.0, 1.0)
    assert np.max(np.abs(ma - mb)) < 1e-10 and np.max(np.abs(ma - mc)) < 1e-10
    assert abs((qb[0] - qb[1]) - (qa[0] - qa[1])) > 0.9
    assert abs((qc[0] - qc[1]) - (qa[0] - qa[1])) > 0.1


def test_P4_messages_never_carry_the_split():
    """Equal joints, Euclidean response: runs differing only in the comfort split, or only in the
    starting split, send the same messages at every step (1e-12)."""
    rec = lambda s: messages(s)
    _, a = ai.run(chain(2.0, 2.0, 0.0, 0.0, 1.0), 300, record=rec)
    _, b = ai.run(chain(2.0, 2.0, 0.7, -0.7, 1.0), 300, record=rec)
    _, c = ai.run(chain(2.0, 2.0, 0.0, 0.0, 1.0, init=(1.3, -1.3)), 300, record=rec)
    assert max(np.max(np.abs(x - y)) for x, y in zip(a, b)) < 1e-12
    assert max(np.max(np.abs(x - y)) for x, y in zip(a, c)) < 1e-12


def test_mechanism_the_pullback_lies_along_the_endpoint_gradient():
    """The composite's forcing on the joints, minus the hand's own, is a multiple of ``(1, 1)``;
    and the published value has zero derivative along ``(1, -1)``, at random states (1e-12)."""
    k1, k2, c1, c2 = 1.5, 3.0, 0.2, 0.1
    arr = chain(k1, k2, c1, c2, 1.0)
    h = joint_hand(k1, k2, c1, c2, euclidean(2, 1.0))
    rng = np.random.default_rng(0)
    for _ in range(5):
        s = jnp.asarray(rng.normal(size=4))
        xi_q, _, _ = arr.omega(s, jnp.zeros(0), jnp.zeros(0))
        own, _, _ = h.omega(s[:2], jnp.zeros(0), jnp.zeros(0))
        from_above = np.asarray(xi_q[:2] - own)
        assert abs(from_above[0] - from_above[1]) < 1e-12
        e_mu1 = float(s[0] + s[1] - s[2])
        assert abs(from_above[0] - (-P * e_mu1)) < 1e-12
        J = jax.jacobian(lambda q: h.out_f(q, jnp.zeros(0)))(s[:2])
        assert abs(float((J @ jnp.array([1.0, -1.0]))[0])) < 1e-15


def test_rest_is_metric_free_path_is_not():
    """Joints coupled in their response (``sharp = eta [[1/2, 1/4], [1/4, 1]]``): same rest as
    Euclidean (1e-10), but the top's pull reaches the split on the way (split leaves its rest by
    more than 1e-3 from a start at comfort)."""
    k1 = k2 = 2.0
    eta = 1.0 / (k1 + k2 + 4 * P)
    coupled = constant(eta * jnp.array([[0.5, 0.25], [0.25, 1.0]]))
    q, msg = settle(k1, k2, 0.0, 0.0, 1.0, steps=120000, response=coupled)
    q_cf, msg_cf = closed_form(k1, k2, 0.0, 0.0, 1.0)
    assert np.max(np.abs(msg - msg_cf)) < 1e-10 and np.max(np.abs(q - q_cf)) < 1e-10
    s0 = chain(k1, k2, 0.0, 0.0, 1.0, init=(0.0, 0.0), response=coupled)
    _, splits = ai.run(s0, 400, record=lambda s: float(s[0][0][0] - s[0][0][1]))
    _, splits_e = ai.run(chain(k1, k2, 0.0, 0.0, 1.0), 400,
                         record=lambda s: float(s[0][0][0] - s[0][0][1]))
    print("coupled: max |split| on the path", max(abs(x) for x in splits),
          "euclidean", max(abs(x) for x in splits_e))
    assert max(abs(x) for x in splits_e) < 1e-12
    assert max(abs(x) for x in splits) > 1e-3
