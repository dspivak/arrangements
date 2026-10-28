"""Attention arrangements with voice (book P73 (c), `notes/runs/p73-c-stiffness-01e0eec3/STATEMENT.md`,
"The voice: the matter as one more sender"; Claude Fable 5.1, 01e0eec3, 2026-10-27).

Besides its transfers, an arrangement has for each box ``i in {0..n}`` a smooth map
``v_i : Q -> Cone(V_i)``, what box ``i`` hears from the matter; readout and delivery
gain the term ``v_i(q)``. The claim checked here is that these still form an operad
containing the paper's (``v = 0``) with the same map into ``arr``: ``substitute``'s
explicit voice formulas agree with lens composition in ``arr``, the unit (silent) is
a unit, and substitution is associative. A ``0``-ary operation with voice reads out
its state, which the paper's operad cannot do (its ``0``-ary readout is the empty sum).

Every docstring says which of three things its test does (STATUS.md §3.2).
"""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np

from arrangements.attention import (
    attention_arrangement, input_dim, message, msg_dim, prediction_cost, substitute,
    unit_arrangement, validate,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean

from tests.test_attention_operad import (
    DH, DP, _agrees_with_composition, assert_same_operation, fixed_transfers, permute_blocks,
)

MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)


def voiced_part(n, seed, q_dim=3, label=None):
    """An ``n``-ary arrangement whose matter is a configuration integrator on ``R^q_dim``,
    with random positive transfers, a prediction-error need, and a voice to every box:
    value affine in ``q``, loudness ``softplus`` of a linear form plus ``0.1`` (so
    always a positive-weight message, a cone point)."""
    r = np.random.default_rng(seed)
    gammas = r.uniform(0.3, 1.5, (n + 1, n + 1))
    for i in range(n + 1):
        gammas[i, i] = 0.0
    T = fixed_transfers(n, gammas)
    A = jnp.asarray(r.standard_normal((EDIM, n * MDIM)) * 0.3)
    c = jnp.asarray(r.standard_normal(EDIM) * 0.3)
    VA = jnp.asarray(r.standard_normal((n + 1, DH, q_dim)) * 0.5)
    Vb = jnp.asarray(r.standard_normal((n + 1, DH)) * 0.5)
    Vc = jnp.asarray(r.standard_normal((n + 1, q_dim)) * 0.5)
    L = jnp.asarray(r.standard_normal((n, DP, q_dim)) * 0.3)

    def voices(q):
        rows = []
        for i in range(n + 1):
            w = jnp.logaddexp(0.0, Vc[i] @ q) + 0.1
            rows.append(message(VA[i] @ q + Vb[i], w))
        return jnp.stack(rows)

    a = attention_arrangement(
        n, DH, DP,
        placement=lambda q: jnp.einsum("ipk,k->ip", L, q) if n else jnp.zeros((0, DP)),
        transfers=T, cost=prediction_cost(lambda q, m: A @ m + c + 0.1 * jnp.sum(q)),
        parameter=configuration(euclidean(q_dim, 0.05), init=jnp.asarray(r.standard_normal(q_dim) * 0.5)),
        label=label or f"voiced{n}", check=False,
    )
    return validate(replace(a, voices=voices))


def test_voiced_substitution_agrees_with_composition_two_level_nest():
    """CHECKS A DISCRETE-TIME ANALOGUE THE BOOK MUST STILL PROVE (STATEMENT.md,
    "The voice", *proposed*): with voices, ``substitute``'s explicit formulas for the
    composite's voices agree with the lens composite in ``arr`` in readout, delivery,
    the whole forcing, and ``Phi`` trajectories."""
    a = voiced_part(2, 1)
    _agrees_with_composition(a, [voiced_part(2, 2), voiced_part(1, 3)], seed=201)


def test_voiced_substitution_with_a_speaking_leaf():
    """As above on an uneven nest whose ``0``-ary child has only a voice to the
    exterior: the composite's readout to the parent is that voice through the
    parent's transfer, and the ``0``-ary child's need acts on what it receives."""
    a = voiced_part(3, 4)
    b = _agrees_with_composition(a, [voiced_part(0, 5), voiced_part(2, 6), voiced_part(1, 7)], seed=202)
    assert b.n == 3


def test_silent_unit_is_a_unit_with_voices():
    """The paper's unit (silent) is a unit for voiced arrangements: ``a(1, ..., 1) = a``
    and ``1(a) = a`` as operations, voices included."""
    one = unit_arrangement(DH, DP)
    a = voiced_part(3, 8)
    q = jnp.asarray(np.random.default_rng(9).standard_normal(a.P.Q_dim) * 0.5)
    right = substitute(a, [one, one, one])
    left = substitute(one, [a])
    assert right.voices is not None and left.voices is not None
    assert_same_operation(right, a, q, q, seed=10)
    assert_same_operation(left, a, q, q, seed=11)
    np.testing.assert_allclose(right.voice(q), a.voice(q), atol=1e-12)
    np.testing.assert_allclose(left.voice(q), a.voice(q), atol=1e-12)


def test_voiced_substitution_is_associative():
    """Both bracketings of a three-level voiced nest agree as operations (voices
    included), up to the reordering of configuration factors."""
    a = voiced_part(2, 31)
    a1, a2 = voiced_part(2, 32), voiced_part(1, 33)
    g = [voiced_part(1, 34), voiced_part(2, 35), voiced_part(1, 36)]
    left = substitute(substitute(a, [a1, a2]), g)
    right = substitute(a, [substitute(a1, g[:2]), substitute(a2, g[2:])])
    sizes = [p.P.Q_dim for p in g] + [a1.P.Q_dim, a2.P.Q_dim, a.P.Q_dim]
    r = np.random.default_rng(103)
    q_left = jnp.asarray(r.standard_normal(left.P.Q_dim) * 0.5)
    q_right = permute_blocks(q_left, sizes, [0, 1, 3, 2, 4, 5])
    assert_same_operation(left, right, q_left, q_right, seed=104)
    np.testing.assert_allclose(left.voice(q_left), right.voice(q_right), atol=1e-11)


def test_a_zero_ary_operation_with_voice_reads_out_its_state():
    """ILLUSTRATES the gap the voice closes: the paper's ``0``-ary readout is the empty
    sum ``0_M``; with a voice it is ``v_0(q)``, a message that depends on the matter."""
    leaf = voiced_part(0, 41)
    q = leaf.P.position(leaf.P.init)
    out = leaf.sigma(q, jnp.zeros(0))
    np.testing.assert_allclose(out, leaf.voice(q)[0], atol=1e-14)
    assert float(out[-1]) > 0.0                     # a positive-weight message
    silent = replace(leaf, voices=None)
    np.testing.assert_allclose(silent.sigma(q, jnp.zeros(0)), jnp.zeros(MDIM), atol=1e-14)
