"""Discrete choice as an integrator (book ``AGENTDOCPLAN.md`` §8.2 P25 (b); notes/DISCRETE-CHOICE.md).

David's question (2026-09-27): does discrete choice come into the framework "through softmax /
stochasticity"? The claim under test is Chapter 2's -- learning and mechanics are arrangements
of one kind -- which stage (a) (``test_bayes_integrator.py``) widened to exponential-family Bayes.

The derivation was fixed in writing before ``arrangements/choice.py`` existed
(notes/DISCRETE-CHOICE.md §1). For ``n`` options with values ``v_i`` at temperature ``T``:

* the CHOOSER is the configuration coalgebra on the simplex with the Shahshahani response
  ``sharp_p = dt (diag(p) - p p^T)`` and the EXACT forcing ``d(<v,p> + T H(p))``. Its rest
  point is the logit, and the rest point does not involve ``dt``; at ``T = 0`` the step is the
  forward-Euler replicator step for the linear fitness ``v``.
* THE MECHANISM IS THE RESPONSE'S: ``1^T sharp_p = dt (1 - sum p) p^T`` is zero exactly on the
  simplex, so a jump preserves ``sum p = 1`` exactly and annihilates the constant part of the
  forcing -- the normalization of the choice distribution is contributed by the response, not
  by the cost. The Euclidean response is the negative control.
* the SAME distribution is ONE Bayes step in the Legendre-dual chart: gauge-fixed natural
  parameters, inverse-Fisher response, affine forcing, ``lambda |-> lambda + t`` with
  ``t_i = (v_i - v_n)/T``, landing on the logit exactly from the uniform belief.
* a MADE CHOICE is a ``(cot(Q) tri rand)``-coalgebra whose branches are the constant maps onto
  the vertices, so a sampled collapse is inside ``poly(SmSet)``; ``argmax`` is not.

THE FIVE PREDICTIONS, verbatim from notes/DISCRETE-CHOICE.md §3, written before any
construction code:

1. **The rest point is the logit.** Iterating the chooser from the uniform ``p`` at a duration
   below the threshold converges to the logit: ``(1,0)`` at ``T = 1`` to ``0.7310585786300049``,
   and ``(2, 0.5, -1)`` at ``T = 0.7`` to
   ``(0.8841084533193249, 0.10372286648030102, 0.01216868020037423)``, each to relative error
   below 1e-12. The forcing is exact throughout, so choice needs no non-exact forcing. And
   ``sum_i p_i = 1`` is preserved by every *jump*, to machine precision, not to first order.
2. **The response supplies the normalization, and the Euclidean response does not.** With
   ``rvect.euclidean(n, eta)`` in place of the Shahshahani sharp -- the same cost, the same
   forcing, the same wiring, the same interface -- the same arrangement settles on the
   *unnormalized* Gibbs weights ``p_i = exp(v_i/T - 1)``, i.e. ``(1.0, 0.36787944117144233)``
   summing to ``1.3678794411714423`` at ``v = (1,0)``, ``T = 1``, and
   ``(6.405409432177272, 0.751477293075286, 0.08816268936235741)`` summing to
   ``7.245049414614916`` at ``v = (2, 0.5, -1)``, ``T = 0.7``. So it is the response's
   degeneracy along ``1``, not the cost, that makes the answer a distribution.
3. **Softmax is also one Bayes step, in the dual chart.** The natural chart / inverse-Fisher /
   affine-forcing arrangement reaches the same logit in ONE step at duration 1 from
   ``lambda = 0``, to relative error below 1e-12; and its response pushed forward along the
   Legendre map equals the Shahshahani sharp on the gauge-fixed block, to 1e-12. The mean-chart
   Euler jump on the *affine* cost (no entropy term) does NOT land on the logit, because a jump
   is affine in its own chart; it misses by order ``|t|^2``.
4. **A made choice is a sample, and it stays inside; ``argmax`` does not.** The collapsing
   ``CotCoalgR``, seeded with ``rand.key_sampler``, has empirical frequencies matching ``p``
   within ``4/sqrt(N)`` at ``N = 20000``; ``FiniteRV.bind`` gives its exact two-step law with no
   sampling, and that law equals the one-step law, so the collapsed chooser is absorbing. It
   tensors with an ordinary ``CotCoalg`` and wires unchanged. The deterministic limit fails
   instead: the derivative of the binary choice in the utility gap, at gap 1, falls
   ``0.196612 -> 0.000453958 -> 0`` at ``T = 1, 0.1, 0.01``, while at a tie the choice is
   exactly ``1/2`` at every ``T``.
5. **Choice is selection at a temperature.** At ``T = 0`` the chooser agrees step for step with
   the library's own ``replicator_arrangement`` for the symmetric payoff ``A = v 1^T + 1 v^T``
   (whose ``dU`` differs from ``v`` by a multiple of ``1``, which the response kills), to 1e-14.
   It approaches the best vertex without reaching it, since ``sharp_p`` vanishes there. And the
   duration matters for the path, not the destination: the measured stability threshold is
   ``dt = 2/T`` (2.0 at ``T = 1``, 4.0 at ``T = 0.5``) to within 1e-6, and the rest point is the
   same at every stable duration.

WHAT THE RUN FOUND (28 tests; verdicts and numbers in notes/DISCRETE-CHOICE.md §5).

Predictions 2, 3 and 4 held exactly as written. Prediction 5's two numerical parts held -- the
``T = 0`` chooser agrees with the library's replicator to 5.4e-20, and the multiplier at the
rest point is exactly ``1 - dt*T``. Two clauses did not hold as written:

* **Prediction 1's "at a duration below the threshold" names the wrong condition.** The
  threshold Prediction 5 supplies, ``dt*T < 2``, is not what binds. What binds is positivity of
  the TRANSIENT: a jump stays in the positive orthant only while
  ``dt * max_i(xibar - xi_i) < 1`` at the state it leaves, and outside that the entropy's ``log``
  of a negative coordinate gives ``nan``. From the uniform start at ``v = (2, 0.5, -1)``,
  ``T = 0.7``, that bound is ``dt < 2/3``, measured between 0.66 and 0.67, where ``dt*T < 2``
  would have allowed everything below ``dt = 2.857``. At ``T = 0`` the same bound reads
  ``dt * max_i(vbar - v_i) < 1`` and is measured between ``dt*spread`` 0.990 and 1.020.
* **Prediction 5's "approaches the best vertex without reaching it" is only half testable.** It
  is true of the map -- the vertex is an exact fixed point and the losing coordinates stay
  strictly positive throughout, down to 6.6e-199 -- but the best coordinate is already
  indistinguishable from 1.0 in double precision at step 64, so "not reached" is not observable
  in the state.

Three things not predicted:

* **The simplex is exactly invariant but not attracting**, which is the important one. The
  Jacobian at the rest point has a second, transverse eigenvalue ``1 + dt*T*(1 - log Z)``,
  above 1 whenever ``Z = sum_j exp(v_j/T) < e``. Then rounding-level sum error grows
  geometrically -- 8.4e-15 at step 100 to 7.0e-5 at step 600, and to ``nan`` soon after -- so
  Prediction 1's exactness holds PER JUMP and fails OVER A RUN. Adding a constant to every
  ``v_i`` changes no choice and multiplies ``Z``, so the numerics is not gauge-invariant where
  the model is; the shift repairs it.
* The collapsing chooser's absorption is visible in the exact two-step law's weights, which are
  ``e_i`` block by block, so no sampling is needed to see that it cannot reconsider.
* The reporting chooser, state ``(p, last)``, keeps moving under the same covector and records
  different vertices as it goes -- so whether a chooser can reconsider is settled by what its
  state holds, not by anything probabilistic.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import coalgebra
from arrangements.choice import (
    belief_arrangement,
    belief_to_distribution,
    choice_law,
    choice_step,
    collapsing_chooser,
    evidence,
    fisher,
    gibbs_weights,
    logit,
    logit_arrangement,
    logit_arrangement_euclidean,
    mean_chart_affine_arrangement,
    reporting_chooser,
    settle,
    soft_choice,
    trajectory,
)
from arrangements.functors import Phi
from arrangements.rand import FiniteRV, key_sampler, seeded
from arrangements.replicator import replicator_dynamics, replicator_step, shahshahani
from arrangements.rvect import euclidean

V2 = jnp.array([1.0, 0.0])
V3 = jnp.array([2.0, 0.5, -1.0])
T2 = 1.0
T3 = 0.7

LOGIT2 = jnp.array([0.7310585786300049, 0.2689414213699951])
LOGIT3 = jnp.array([0.8841084533193249, 0.10372286648030102, 0.01216868020037423])


def _rel(a, b):
    return float(jnp.max(jnp.abs(jnp.asarray(a) - jnp.asarray(b)) / jnp.abs(jnp.asarray(b))))


# ---------------------------------------------------------------------------
# Prediction 1: the rest point is the logit, and the forcing is exact.
# ---------------------------------------------------------------------------


def test_small_example_two_options_is_the_binary_logit():
    """The small example of notes §2, run first: v = (1,0), T = 1."""
    p = settle(logit_arrangement(V2, T2, dt=0.5), 400)
    assert _rel(p, LOGIT2) < 1e-12
    assert abs(float(jnp.sum(p)) - 1.0) < 1e-12


def test_rest_point_is_the_logit_three_options():
    p = settle(logit_arrangement(V3, T3, dt=0.5), 800)
    assert _rel(p, LOGIT3) < 1e-12


def test_rest_point_does_not_depend_on_the_duration():
    """The increment vanishes iff ``v_i - T log p_i`` is constant, a condition without ``dt``.

    Within the positivity bound below; ``dt = 0.9`` is outside it and gives ``nan``.
    """
    for dt in (0.1, 0.3, 0.6):
        assert _rel(settle(logit_arrangement(V3, T3, dt=dt), 4000), LOGIT3) < 1e-12


def test_the_binding_duration_bound_is_positivity_of_the_transient():
    """NOT PREDICTED, and it refutes ``dt*T < 2`` as the operative condition. A jump stays in
    the positive orthant only while ``dt * max_i(xibar - xi_i) < 1`` at the state it leaves,
    and outside that the entropy's ``log`` of a negative coordinate gives ``nan``. From the
    uniform start at ``v = (2, 0.5, -1)``, ``T = 0.7``, the spread is exactly 1.5, so the bound
    is ``dt < 2/3``: measured between 0.66 (holds) and 0.67 (``nan``), where ``dt*T < 2`` would
    have allowed everything below ``dt = 2.857``.
    """
    p = jnp.full(3, 1.0 / 3.0)
    xi = V3 - T3 * jnp.log(p) - T3
    assert float(jnp.max(jnp.dot(p, xi) - xi)) == pytest.approx(1.5, abs=1e-12)

    assert float(jnp.min(trajectory(logit_arrangement(V3, T3, dt=0.66), 300))) > 0.0
    assert bool(jnp.any(jnp.isnan(trajectory(logit_arrangement(V3, T3, dt=0.67), 300))))
    assert 0.67 * T3 < 2.0  # so the stability bound was not what bit


def test_every_jump_preserves_the_sum_exactly():
    """Not to first order: ``1^T sharp_p = dt (1 - sum p) p^T`` is zero on the simplex.

    Per jump; over a long run the transverse eigenvalue can amplify rounding
    (``test_simplex_is_invariant_but_not_attracting``), which is why ``T = 1`` is used here.
    """
    tr = trajectory(logit_arrangement(V3, T3, dt=0.5), 500)
    assert float(jnp.max(jnp.abs(jnp.sum(tr, axis=1) - 1.0))) < 1e-14


def test_the_choosers_forcing_is_exact():
    """``omega = dU`` by construction, so ``jax`` sees a symmetric Jacobian: choice needs no
    non-exact forcing, unlike the replicator field's direct realization (replicator.py)."""
    a = logit_arrangement(V3, T3)
    p = jnp.array([0.5, 0.3, 0.2])
    J = jax.jacfwd(lambda q: a.omega(q, jnp.zeros(0), jnp.zeros(0))[0])(p)
    assert float(jnp.max(jnp.abs(J - J.T))) < 1e-12


# ---------------------------------------------------------------------------
# Prediction 2: the response supplies the normalization.
# ---------------------------------------------------------------------------


def test_euclidean_response_settles_on_unnormalized_gibbs_weights():
    p = settle(logit_arrangement_euclidean(V2, T2, eta=0.2), 3000)
    assert _rel(p, gibbs_weights(V2, T2)) < 1e-12
    assert _rel(p, jnp.array([1.0, 0.36787944117144233])) < 1e-12
    assert abs(float(jnp.sum(p)) - 1.3678794411714423) < 1e-12


def test_euclidean_response_unnormalized_three_options():
    p = settle(logit_arrangement_euclidean(V3, T3, eta=0.05), 20000)
    assert _rel(p, gibbs_weights(V3, T3)) < 1e-10
    assert abs(float(jnp.sum(p)) - 7.245049414614916) < 1e-9
    # and it is NOT the logit: the miss is large, not a tolerance question
    assert float(jnp.max(jnp.abs(p - LOGIT3))) > 0.5


def test_the_constant_part_of_the_forcing_is_annihilated():
    """The mechanism, directly: ``sharp_p(c * 1) = 0`` exactly on the simplex, which is why the
    ``-T`` in ``xi_i = v_i - T log p_i - T`` and a Lagrange multiplier both drop out."""
    p = jnp.array([0.5, 0.3, 0.2])
    S = shahshahani(3, 1.0).sharp_at(p)
    assert float(jnp.max(jnp.abs(S @ jnp.ones(3)))) < 1e-15
    off = jnp.array([0.5, 0.3, 0.3])  # sum 1.1
    assert float(jnp.max(jnp.abs(shahshahani(3, 1.0).sharp_at(off) @ jnp.ones(3)))) > 1e-3


# ---------------------------------------------------------------------------
# Prediction 3: softmax is one Bayes step in the dual chart.
# ---------------------------------------------------------------------------


def test_one_bayes_step_from_the_uniform_belief_is_the_logit():
    lam = settle(belief_arrangement(V3, T3), 1)
    assert _rel(belief_to_distribution(lam), LOGIT3) < 1e-12
    assert float(jnp.max(jnp.abs(lam - evidence(V3, T3)))) < 1e-12


def test_the_two_charts_carry_the_same_response():
    """``G G^{-1} G = G``: the inverse-Fisher response pushed forward along the Legendre map is
    the Shahshahani sharp on the gauge-fixed block."""
    lam = jnp.array([0.3, -0.4])
    mu = belief_to_distribution(lam)
    shah = (jnp.diag(mu) - jnp.outer(mu, mu))[:2, :2]
    assert float(jnp.max(jnp.abs(fisher(3)(lam) - shah))) < 1e-12


def test_the_mean_chart_jump_on_the_affine_cost_misses_by_order_t_squared():
    """A ``configuration`` step is affine in its own chart, so the chart is part of the
    arrangement: the same response with the entropy term dropped linearizes softmax."""
    errs = {}
    for eps in (0.5, 0.25, 0.125):
        one = settle(mean_chart_affine_arrangement(eps * V3, T3, dt=1.0), 1)
        errs[eps] = float(jnp.max(jnp.abs(one - logit(eps * V3, T3))))
    assert errs[0.5] > 0.05  # it misses, and not slightly
    for eps in (0.5, 0.25):
        assert 3.0 < errs[eps] / errs[eps / 2] < 5.0  # second order: the ratio is near 4


def test_the_mean_chart_affine_arrangement_runs_to_a_vertex():
    """Without the entropy term the descent has no interior rest point at all."""
    p = settle(mean_chart_affine_arrangement(V3, T3, dt=0.2), 200)
    assert float(jnp.max(p)) > 1.0 - 1e-12
    assert float(jnp.max(jnp.abs(p - LOGIT3))) > 0.1


# ---------------------------------------------------------------------------
# Prediction 4: a made choice is a sample, and argmax is what is outside.
# ---------------------------------------------------------------------------


P_MADE = jnp.array([0.6, 0.3, 0.1])


def test_the_collapsing_chooser_is_a_rand_coalgebra_over_the_vertices():
    rv = collapsing_chooser(3, init=P_MADE).step(P_MADE, jnp.zeros(3))
    assert isinstance(rv, FiniteRV)
    assert float(jnp.max(jnp.abs(rv.weights - P_MADE))) == 0.0
    assert float(jnp.max(jnp.abs(rv.outcomes - jnp.eye(3)))) == 0.0


def test_the_exact_two_step_law_equals_the_one_step_law():
    """``FiniteRV.bind``, the ``tri``-monad multiplication, with no sampler: the collapsed
    chooser is absorbing, and the weights show it block by block."""
    rv2 = choice_law(3, P_MADE, 2)
    law = sum(rv2.weights[i] * rv2.ret(i) for i in range(rv2.n))
    assert float(jnp.max(jnp.abs(law - P_MADE))) == 0.0
    # nine branches, and only the diagonal ones carry weight
    assert rv2.n == 9
    assert float(jnp.max(jnp.abs(rv2.weights - jnp.diag(P_MADE).reshape(-1)))) < 1e-15


def test_a_seeded_chooser_has_the_right_frequencies():
    ordinary = seeded(collapsing_chooser(3, init=P_MADE), key_sampler(0))
    xi, N = jnp.zeros(3), 20000
    t = ordinary.init[1]
    counts = np.zeros(3)
    for _ in range(N):
        p_next, t = ordinary.step((P_MADE, t), xi)
        counts += np.asarray(p_next)
    assert float(np.max(np.abs(counts / N - np.asarray(P_MADE)))) < 4.0 / np.sqrt(N)


def test_the_stochastic_chooser_tensors_with_an_ordinary_coalgebra():
    """``rand.py``'s lift along the unit, so the wiring composes it unchanged."""
    T = collapsing_chooser(3).tensor(coalgebra.configuration(euclidean(2, 0.1)))
    assert T.Q_dim == 5 and T.kind is FiniteRV


def test_the_reporting_chooser_can_reconsider():
    """State ``(p, last)``: the drawn vertex is recorded and never fed back, so the same
    covector keeps moving the distribution -- where the collapsing chooser is absorbing."""
    o = seeded(reporting_chooser(shahshahani(3, 0.3)), key_sampler(1))
    st, xi = o.init, jnp.array([1.0, 0.0, -1.0])
    ps = []
    for _ in range(5):
        st = o.step(st, xi)
        ps.append(st[0][0])
    assert float(ps[-1][0]) > float(ps[0][0]) + 0.2  # it moved, and kept moving
    assert float(jnp.sum(ps[-1])) == pytest.approx(1.0, abs=1e-12)


def test_argmax_has_no_smooth_extension_across_a_tie():
    """The rest point is the logit (Prediction 1), so this differentiates the arrangement's
    rest point: off a tie the choice flattens to locally constant as ``T -> 0``, while at a
    tie it is ``1/n`` at every ``T``."""
    d = jax.grad(soft_choice, argnums=0)
    slopes = [float(d(1.0, T)) for T in (1.0, 0.1, 0.01)]
    assert slopes[0] == pytest.approx(0.196612, abs=1e-5)
    assert slopes[1] == pytest.approx(0.000453958, rel=1e-4)
    assert slopes[2] == 0.0
    for T in (1.0, 0.1, 0.01):
        assert float(soft_choice(0.0, T)) == 0.5


# ---------------------------------------------------------------------------
# Prediction 5: choice is selection at a temperature.
# ---------------------------------------------------------------------------


def test_at_zero_temperature_the_chooser_is_the_replicator():
    """``A = v 1^T + 1 v^T`` is symmetric and its ``dU`` differs from ``v`` by a multiple of
    ``1``, which the Shahshahani response kills, so the two agree step for step."""
    A = jnp.outer(V3, jnp.ones(3)) + jnp.outer(jnp.ones(3), V3)
    dt = 0.1
    O, Oc = replicator_dynamics(A, dt), Phi(logit_arrangement(V3, 0.0, dt=dt))
    p = q = jnp.array([0.5, 0.3, 0.2])
    for _ in range(50):
        p, q = replicator_step(O, p), choice_step(Oc, q)
    assert float(jnp.max(jnp.abs(p - q))) < 1e-14


def test_at_zero_temperature_the_best_vertex_is_approached_not_reached():
    """Approached in the map, and NOT OBSERVABLE as such in the state: the losing coordinates
    stay strictly positive for the whole run (down to 6.6e-199 here) and the vertex is an exact
    fixed point, since ``sharp_p`` vanishes there -- but the best coordinate is already
    indistinguishable from 1.0 in double precision at step 64. The prediction's wording is
    therefore only half testable, and this is the half that is.
    """
    tr = trajectory(logit_arrangement(V3, 0.0, dt=0.3), 600, s=jnp.array([0.4, 0.35, 0.25]))
    assert bool(jnp.all(tr[:, 1:] > 0.0))  # the losers never reach zero
    assert float(jnp.min(tr[:, 1:])) < 1e-100  # and fall without bound
    assert int(jnp.argmax(jnp.max(tr, axis=1) >= 1.0)) == 64  # float saturation, not arrival

    O = Phi(logit_arrangement(V3, 0.0, dt=0.3))
    vertex = jnp.array([1.0, 0.0, 0.0])
    assert bool(jnp.all(choice_step(O, vertex) == vertex))


def test_above_the_positivity_bound_a_coordinate_overshoots_below_zero():
    """Measured between ``dt*spread = 0.990`` and ``1.020``; ``spread = max_i v_i - min_i v_i``
    is 3.0 here, so the two durations are 0.33 and 0.34."""
    ok = trajectory(logit_arrangement(V3, 0.0, dt=0.33), 600, s=jnp.array([0.4, 0.35, 0.25]))
    bad = trajectory(logit_arrangement(V3, 0.0, dt=0.34), 600, s=jnp.array([0.4, 0.35, 0.25]))
    assert float(jnp.min(ok)) >= 0.0
    assert float(jnp.min(bad)) < -1e-11


@pytest.mark.parametrize("T,dt", [(1.0, 0.3), (1.0, 1.0), (0.5, 0.3), (0.5, 1.0), (2.0, 0.3)])
def test_the_multiplier_at_the_rest_point_is_one_minus_dt_times_T(T, dt):
    """The stability threshold is ``dt*T = 2``, which is §5.2's mobility-weighted-Hessian bound
    below 2 in this case. Measured as an eigenvalue of one step's Jacobian (notes §3 amendment).
    """
    O = Phi(logit_arrangement(V2, T, dt=dt))
    J = jax.jacfwd(lambda s: choice_step(O, s))(logit(V2, T))
    eigs = np.sort(np.real(np.linalg.eigvals(np.asarray(J))))
    assert float(np.min(np.abs(eigs - (1.0 - dt * T)))) < 1e-12


def test_the_simplex_is_invariant_but_not_attracting():
    """NOT PREDICTED. The transverse eigenvalue is ``1 + dt*T*(1 - log Z)``, above 1 whenever
    ``Z = sum_j exp(v_j/T) < e``; then rounding-level sum error grows geometrically and the run
    ends in ``nan``. ``v = (1,0)`` at ``T = 2`` has ``Z = 2.6487 < e``; at ``T = 1``,
    ``Z = 3.7183 > e``, and the sum then holds exactly.
    """
    Z_cold = float(jnp.sum(jnp.exp(V2 / 1.0)))
    Z_hot = float(jnp.sum(jnp.exp(V2 / 2.0)))
    assert Z_cold > np.e > Z_hot

    for T, dt in ((2.0, 0.9), (2.0, 1.0)):
        O = Phi(logit_arrangement(V2, T, dt=dt))
        J = jax.jacfwd(lambda s: choice_step(O, s))(logit(V2, T))
        eigs = np.real(np.linalg.eigvals(np.asarray(J)))
        predicted = 1.0 + dt * T * (1.0 - np.log(Z_hot))
        assert float(np.min(np.abs(eigs - predicted))) < 1e-10
        assert predicted > 1.0

    drift = jnp.abs(jnp.sum(trajectory(logit_arrangement(V2, 2.0, dt=0.9), 600), axis=1) - 1.0)
    assert float(drift[100]) < 1e-13 and float(drift[600]) > 1e-5  # rounding, amplified
    held = jnp.abs(jnp.sum(trajectory(logit_arrangement(V2, 1.0, dt=1.0), 600), axis=1) - 1.0)
    assert float(jnp.max(held)) == 0.0

    # the numerics is not gauge-invariant where the model is: shifting every v_i by +1 leaves
    # every choice alone and multiplies Z by e^{1/T}, which restores the attracting simplex.
    shifted = jnp.abs(jnp.sum(trajectory(logit_arrangement(V2 + 1.0, 2.0, dt=0.9), 600), axis=1) - 1.0)
    assert _rel(settle(logit_arrangement(V2 + 1.0, 2.0, dt=0.5), 400), logit(V2, 2.0)) < 1e-12
    assert float(jnp.max(shifted)) < 1e-13


# ===========================================================================
# Part two: choosing by basin (notes/DISCRETE-CHOICE.md §§6-9; David, 2026-09-27:
# "can you think of another way? how does nature do it? it's not basins or something?").
#
# Part one fed a FIXED value for each option in from outside, so there was one rest point and the
# only route to a discrete outcome was an external coin. Let an option's value depend on the
# shares already committed -- a symmetric payoff ``A`` in place of the linear ``<v,p>`` -- and the
# SAME arrangement has several stable rest points, each with a basin, and chooses with no coin and
# no temperature. One family, two knobs: the coupling ``A`` and the temperature ``T``.
#
# THIS ALSO WITHDRAWS Part one's "determinism is the obstruction, not the jump." A basin chooser is
# deterministic and entirely inside the framework. What has no smooth extension is ``argmax`` AS A
# FUNCTION OF THE VALUES; a basin chooser near a tie does not jump, it SLOWS, because the boundary
# is an unstable rest point.
#
# THE FIVE PREDICTIONS, verbatim from notes/DISCRETE-CHOICE.md §8, written before any new
# construction code:
#
# 1. **The basin boundary is ``a_2/(a_1+a_2)``.** At ``T = 0`` the flip in the outcome, located by
#    bisecting the starting share, sits at ``2/3`` for ``a = (1,2)`` and at ``4/5`` for
#    ``a = (1,4)``, to 1e-6. The interior rest point is unstable and the two vertices are stable,
#    checked as ``g'(p_1*) > 0`` against the closed form ``p_1*(1-p_1*)(a_1+a_2)`` to 1e-12.
# 2. **The worse option wins from a large enough share.** With ``a = (1,2)``, every start above 2/3
#    commits to option 1, whose payoff is half option 2's. So no book sentence may call this
#    chooser an optimizer.
# 3. **Deciding slows logarithmically at the boundary, at the derived rate.** Steps to decide,
#    against ``-log`` of the distance from ``p_1*``, is a straight line of slope
#    ``1/log(1 + dt g'(p_1*))`` -- 30.50 at ``a = (1,2)``, ``dt = 0.05``, and 15.30 at ``dt = 0.1``
#    -- fitted over distances from 1e-2 to 1e-8, to 2%. A chooser near a tie is slow, not
#    discontinuous.
# 4. **The pitchfork at ``T_c = a/2``.** For ``A = diag(1,1)``: below ``T = 0.5`` there are three
#    interior rest points and the outcome depends on the start; above it there is one, at
#    ``(1/2, 1/2)``, and every interior start reaches it. The measured critical temperature is 0.5
#    to 1e-4, and the committed share just below it grows like ``sqrt(T_c - T)`` (the pitchfork's
#    exponent, slope of ``log m`` against ``log(T_c - T)`` equal to 1/2, to 5%).
# 5. **No coupling, no basins, and it is one arrangement family.** With ``A = v 1^T + 1 v^T`` there
#    is exactly one rest point at every ``T > 0``, the logit of Part one, reached from every
#    interior start -- so the basins are the coupling's doing and nothing else's. And the same
#    constructor agrees step for step with ``replicator_arrangement`` at ``T = 0`` (to 1e-14) and
#    with Part one's ``logit_arrangement`` for the uncoupled payoff (to 1e-14).
#
# WHAT THE RUN FOUND (notes/DISCRETE-CHOICE.md §9): all five held. One arithmetic slip in
# Prediction 3's second number -- the derived slope at ``dt = 0.1`` is 15.4946, not the 15.30
# written; the formula was right and the arithmetic was not, and the test uses the formula.
# ===========================================================================


from arrangements.choice import (  # noqa: E402  -- Part two's constructors
    basin_boundary,
    boundary_rate,
    committed_share,
    critical_temperature,
    decide,
    game_arrangement,
    locate_boundary,
    rest_points_2,
    slowing_slope,
    uncoupled_payoff,
)

A12 = jnp.diag(jnp.array([1.0, 2.0]))
A14 = jnp.diag(jnp.array([1.0, 4.0]))
ASYM = jnp.diag(jnp.array([1.0, 1.0]))


# ---- Prediction 1: the basin boundary ----


@pytest.mark.parametrize("a,expected", [((1.0, 2.0), 2.0 / 3.0), ((1.0, 4.0), 4.0 / 5.0)])
def test_the_basin_boundary_is_the_closed_form(a, expected):
    av = jnp.array(a)
    assert float(basin_boundary(av)) == pytest.approx(expected, abs=1e-12)
    A = jnp.diag(av)
    assert locate_boundary(A, 0.0, 0.05, rounds=24, max_steps=2000) == pytest.approx(
        expected, abs=1e-6
    )


def test_the_boundary_is_unstable_and_the_vertices_are_not():
    """``g'(p_1*) = p_1*(1-p_1*)(a_1+a_2) > 0``, so the interior point repels and the two vertices
    attract: that is what makes it a basin boundary rather than a choice."""
    for a in ((1.0, 2.0), (1.0, 4.0)):
        av = jnp.array(a)
        p = float(basin_boundary(av))
        assert float(boundary_rate(av)) == pytest.approx(p * (1 - p) * sum(a), abs=1e-12)
        assert float(boundary_rate(av)) > 0.0
    O = Phi(game_arrangement(A12, 0.0, dt=0.05))
    for vertex in (jnp.array([1.0, 0.0]), jnp.array([0.0, 1.0])):
        assert bool(jnp.all(choice_step(O, vertex) == vertex))


# ---- Prediction 2: this chooser is not an optimizer ----


@pytest.mark.parametrize("p1", [0.67, 0.70, 0.80, 0.95])
def test_the_worse_option_wins_from_a_large_enough_share(p1):
    """Payoffs 1 and 2, and every start above 2/3 commits to the option paying 1."""
    winner, steps = decide(game_arrangement(A12, 0.0, dt=0.05), jnp.array([p1, 1.0 - p1]))
    assert winner == 0 and steps < 20000


def test_below_the_boundary_the_better_option_wins():
    winner, _ = decide(game_arrangement(A12, 0.0, dt=0.05), jnp.array([0.66, 0.34]))
    assert winner == 1


# ---- Prediction 3: slowing, not jumping ----


@pytest.mark.parametrize("dt", [0.05, 0.1])
def test_deciding_slows_logarithmically_at_the_derived_rate(dt):
    """The repair of Part one's overstatement: near a tie the chooser is slow, not discontinuous,
    because the boundary is a hyperbolic unstable point with jump multiplier ``1 + dt g'(p_1*)``."""
    a = jnp.array([1.0, 2.0])
    pstar = float(basin_boundary(a))
    xs, ys = [], []
    for e in (1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8):
        _, k = decide(game_arrangement(A12, 0.0, dt=dt), jnp.array([pstar + e, 1.0 - pstar - e]),
                      max_steps=20000)
        xs.append(-np.log(e))
        ys.append(k)
    slope = np.polyfit(xs, ys, 1)[0]
    assert slope / slowing_slope(a, dt) == pytest.approx(1.0, rel=0.02)
    assert ys == sorted(ys)  # monotonically slower the nearer the tie


# ---- Prediction 4: the pitchfork at T_c = a/2 ----


def test_the_critical_temperature_is_half_the_coupling():
    assert critical_temperature(1.0) == 0.5
    lo, hi = 0.1, 0.9
    for _ in range(26):
        mid = 0.5 * (lo + hi)
        if len(rest_points_2(ASYM, mid, grid=2001)) >= 3:
            lo = mid
        else:
            hi = mid
    assert lo == pytest.approx(0.5, abs=1e-4)  # grid-limited from below, never above


@pytest.mark.parametrize("T,count", [(0.1, 3), (0.3, 3), (0.45, 3), (0.5, 1), (0.7, 1)])
def test_three_rest_points_below_the_critical_temperature_and_one_above(T, count):
    rp = rest_points_2(ASYM, T, grid=2001)
    assert len(rp) == count
    if count == 3:
        m = committed_share(1.0, T)
        assert float(rp[2]) == pytest.approx(0.5 * (1 + m), abs=1e-4)
        assert float(rp[0]) == pytest.approx(0.5 * (1 - m), abs=1e-4)
    assert float(rp[len(rp) // 2]) == pytest.approx(0.5, abs=1e-9)


def test_no_vertex_is_a_rest_point_at_any_positive_temperature():
    """The entropy's ``log`` diverges at a corner while the payoff stays bounded, so a hedging
    chooser never fully commits -- what it commits to is a committed *share*."""
    for T in (0.01, 0.1, 0.4):
        rp = rest_points_2(ASYM, T, grid=2001)
        assert float(jnp.min(rp)) > 0.0 and float(jnp.max(rp)) < 1.0


def test_the_committed_share_grows_with_the_pitchfork_exponent_one_half():
    gaps = np.array([1e-4, 3e-4, 1e-3, 3e-3, 1e-2])
    ms = np.array([committed_share(1.0, 0.5 - g) for g in gaps])
    assert np.polyfit(np.log(gaps), np.log(ms), 1)[0] == pytest.approx(0.5, rel=0.05)


@pytest.mark.parametrize("T,starts,decides", [(0.7, (0.05, 0.3, 0.9), False),
                                             (0.2, (0.45, 0.55), True)])
def test_above_the_critical_temperature_the_start_stops_mattering(T, starts, decides):
    outs = [settle(game_arrangement(ASYM, T, dt=0.2, init=jnp.array([p, 1.0 - p])), 4000)
            for p in starts]
    if decides:
        assert float(outs[0][0]) < 0.1 and float(outs[1][0]) > 0.9  # the start picks the winner
    else:
        for s in outs:
            assert float(jnp.max(jnp.abs(s - 0.5))) < 1e-9  # every start hedges at 1/2


# ---- Prediction 5: no coupling, no basins; and one family ----


def test_without_coupling_there_is_exactly_one_rest_point():
    """So the basins are the coupling's doing and nothing else's."""
    Au = uncoupled_payoff(jnp.array([1.0, 0.0]))
    rp = rest_points_2(Au, 0.7, grid=2001)
    assert len(rp) == 1
    assert float(rp[0]) == pytest.approx(float(logit(V2, 0.7)[0]), abs=1e-6)


def test_the_uncoupled_game_is_part_ones_chooser_step_for_step():
    dt = 0.1
    Og = Phi(game_arrangement(uncoupled_payoff(V3), T3, dt=dt))
    Ol = Phi(logit_arrangement(V3, T3, dt=dt))
    p = q = jnp.array([0.5, 0.3, 0.2])
    for _ in range(60):
        p, q = choice_step(Og, p), choice_step(Ol, q)
    assert float(jnp.max(jnp.abs(p - q))) < 1e-14
    # and the shared destination is Part one's logit, which 60 steps at dt = 0.1 only approaches
    assert _rel(settle(game_arrangement(uncoupled_payoff(V3), T3, dt=0.3), 800), LOGIT3) < 1e-12


def test_at_zero_temperature_the_game_is_the_librarys_replicator_step_for_step():
    A = jnp.array([[1.0, 0.4, 0.0], [0.4, 1.5, 0.2], [0.0, 0.2, 2.0]])
    dt = 0.1
    Or, Og = replicator_dynamics(A, dt), Phi(game_arrangement(A, 0.0, dt=dt))
    x = y = jnp.array([0.5, 0.3, 0.2])
    for _ in range(60):
        x, y = replicator_step(Or, x), choice_step(Og, y)
    assert float(jnp.max(jnp.abs(x - y))) < 1e-14


def test_a_non_symmetric_payoff_is_rejected():
    """As ``replicator_arrangement``: a non-symmetric game is not a potential game, and silently
    symmetrizing it would change the dynamics."""
    with pytest.raises(ValueError):
        game_arrangement(jnp.array([[0.0, 1.0], [2.0, 0.0]]), 0.1)


# ===========================================================================
# Part three: composed with other things (notes/DISCRETE-CHOICE.md §§10-11).
#
# Part two closed by calling a wired chooser "untested," which David read as "we can't tell if
# discrete choice can actually affect other things." That sentence was worse than the truth and is
# withdrawn. That these arrangements COMPOSE is not in question -- ``arr`` is a category, ``Phi`` a
# functor, and ``test_composition.py`` checks both operad legs, non-exact forcing included.
# Composition is pullback-and-sum on the FORCING and leaves the integrator alone, so the only open
# question for a closed test is whether its conclusion survives an ADDED FORCING TERM. It does not
# survive unchanged here, and how it fails is the result.
#
# THE FIVE PREDICTIONS, verbatim from notes/DISCRETE-CHOICE.md §10, written before any composition
# code:
#
# 1. **Composition is faithful.** The composite's chooser-state trajectory equals, to 1e-14, the
#    closed chooser whose potential has ``b (q_1 - q_2)`` added by hand. So wiring does exactly
#    pullback-and-sum and leaves the integrator alone.
# 2. **An outside push moves the basin boundary, linearly, at rate ``1/a``.** The rest point is
#    ``p_1* = 1/2 - b/a``: with ``a = 1``, ``b = 0.1`` the boundary is at 0.4 and with ``b = -0.2``
#    at 0.7, located by bisection to 1e-6. So one box **can** move where another box's choice tips.
# 3. **Above a critical push bistability is destroyed: ``|b| >= a/2``.** With ``a = 1`` that push is
#    0.5: at ``b = 0.6`` every start commits to option 1, and at ``b = 0.4`` the start still decides.
# 4. **And it is hysteresis: releasing the push does not undo the flip.** Push a chooser started at
#    ``p_1 = 0.2`` past the threshold so it commits to option 1, then set ``b = 0``; it stays,
#    because a vertex is an exact fixed point and the response vanishes there. Checked to 0.
# 5. **The chooser affects downstream too.** With the chooser publishing its share of option 1 into
#    a downstream box whose forcing is ``-(y - share)^2/2``, the downstream state ends at 1 when the
#    chooser commits to option 1 and at 0 when it commits to option 2, to 1e-6.
#
# WHAT THE RUN FOUND: 2, 3, 4 and 5 held as written. **Prediction 1 was REFUTED as written, by
# something real: I forgot RETROACTIVITY.** With a downstream reader present the composite's forcing
# sums BOTH boxes' pullbacks, so the box that reads the choice also tilts it -- the three-box
# composite deviates from my hand-sum by 3.2e-2, and the boundary lands at 0.3876 instead of 0.4.
# Compare the right things and composition is exact: the two-box composite (nothing reading) equals
# the hand-summed push to **0.0**, and the three-box composite equals a hand-sum that includes the
# follower's pullback to 1.1e-16. So the answer to David's question is BOTH directions at once: a
# choice is moved by what pushes it and by what reads it, and the framework does that without being
# asked.
# ===========================================================================


from arrangements.choice import (  # noqa: E402  -- Part three's constructors
    chain_states,
    choice_chain,
    critical_push,
    follower,
    game_potential,
    locate_pushed_boundary,
    push_chain,
    push_states,
    pushed_boundary,
    pushed_chooser,
    pusher,
)
from arrangements.coalgebra import configuration as _configuration  # noqa: E402
from arrangements.replicator import shahshahani as _shahshahani  # noqa: E402
from arrangements.choice import _closed  # noqa: E402
from arrangements.wiring import compose_seq  # noqa: E402

DT3 = 0.2
TILT = jnp.array([1.0, -1.0])


def _hand_pushed(b, init, dt=DT3):
    """The closed chooser with the push added to its potential by hand."""
    return _closed(_configuration(_shahshahani(2, dt), init=jnp.asarray(init)),
                   lambda q: game_potential(ASYM, 0.0)(q) + b * jnp.dot(TILT, q), "hand")


def _run(arr, steps, unpack):
    O = Phi(arr)
    s = arr.P.init
    out = []
    for _ in range(steps):
        s = choice_step(O, s)
        out.append(unpack(s))
    return out


# ---- Prediction 1: faithfulness, and the retroactivity that refuted it as written ----


def test_the_two_box_composite_is_exactly_the_hand_summed_push():
    """Nothing downstream: composition contributes the upstream push and nothing else."""
    init = jnp.array([0.45, 0.55])
    got = _run(push_chain(0.1, ASYM, 0.0, dt=DT3, init=init), 40, push_states)
    want = _run(_hand_pushed(0.1, init), 40, lambda s: s)
    assert float(jnp.max(jnp.abs(jnp.stack(got) - jnp.stack(want)))) == 0.0


def test_a_downstream_reader_pushes_back_on_the_choice():
    """REFUTES Prediction 1 as written, and is the better answer. A box that only *reads* the
    chooser's share still tilts it, because the composite's forcing sums both boxes' pullbacks."""
    init = jnp.array([0.45, 0.55])
    with_reader = _run(choice_chain(0.1, ASYM, 0.0, dt=DT3, init=init), 40,
                       lambda s: chain_states(s)[0])
    without = _run(_hand_pushed(0.1, init), 40, lambda s: s)
    assert float(jnp.max(jnp.abs(jnp.stack(with_reader) - jnp.stack(without)))) > 1e-2


def test_the_three_box_composite_equals_the_hand_sum_that_includes_the_pullback():
    """And with the reader's pullback written in by hand, composition is exact again: the
    chooser's forcing is ``dU + b*tilt + (y - q_1) e_1`` and the follower descends ``-(y-q_1)^2/2``.
    """
    Q = _shahshahani(2, DT3)
    b, eta = 0.1, 0.3
    p, y = jnp.array([0.45, 0.55]), jnp.array([0.5])
    hp, hy = [], []
    for _ in range(40):
        xi = jax.grad(lambda q: game_potential(ASYM, 0.0)(q) + b * jnp.dot(TILT, q)
                      - 0.5 * (y[0] - q[0]) ** 2)(p)
        xiy = jax.grad(lambda yy: -0.5 * (yy[0] - p[0]) ** 2)(y)
        p, y = p + Q.apply_sharp(p, xi), y + eta * xiy
        hp.append(p)
        hy.append(y)
    arr = choice_chain(b, ASYM, 0.0, dt=DT3, init=jnp.array([0.45, 0.55]), eta=eta, y0=0.5)
    got = _run(arr, 40, chain_states)
    assert float(jnp.max(jnp.abs(jnp.stack([g[0] for g in got]) - jnp.stack(hp)))) < 1e-14
    assert float(jnp.max(jnp.abs(jnp.stack([g[1] for g in got]) - jnp.stack(hy)))) < 1e-14


# ---- Prediction 2: an outside push moves the boundary ----


@pytest.mark.parametrize("b", [0.1, -0.2, 0.0])
def test_an_outside_push_moves_the_basin_boundary_linearly(b):
    """``p_1* = 1/2 - b/a``: one box moves where another box's choice tips."""
    assert pushed_boundary(1.0, b) == pytest.approx(0.5 - b, abs=1e-12)
    located = locate_pushed_boundary(b, ASYM, 0.0, DT3, rounds=22, max_steps=2500)
    assert located == pytest.approx(pushed_boundary(1.0, b), abs=1e-6)


# ---- Prediction 3: above a critical push, the outside decides ----


def test_above_the_critical_push_every_start_commits_the_same_way():
    assert critical_push(1.0) == 0.5
    for p1 in (0.02, 0.2, 0.49):
        arr = push_chain(0.6, ASYM, 0.0, dt=DT3, init=jnp.array([p1, 1.0 - p1]))
        assert int(jnp.argmax(_run(arr, 600, push_states)[-1])) == 0
    # just below it, the start still decides: boundary at 1/2 - 0.4 = 0.1
    below = [int(jnp.argmax(_run(push_chain(0.4, ASYM, 0.0, dt=DT3,
                                           init=jnp.array([p1, 1.0 - p1])), 1200, push_states)[-1]))
             for p1 in (0.05, 0.2)]
    assert below == [1, 0]


# ---- Prediction 4: hysteresis ----


def test_releasing_the_push_does_not_undo_the_flip():
    """A vertex is an exact fixed point where the response vanishes, so the flip is permanent --
    which is what makes commitment worth the name."""
    start = jnp.array([0.2, 0.8])
    pushed = _run(push_chain(0.6, ASYM, 0.0, dt=DT3, init=start), 300, push_states)[-1]
    assert int(jnp.argmax(pushed)) == 0
    released = _run(push_chain(0.0, ASYM, 0.0, dt=DT3, init=pushed), 2000, push_states)
    assert int(jnp.argmax(released[-1])) == 0
    assert float(jnp.max(jnp.abs(jnp.stack(released) - pushed))) < 1e-12
    # and unpushed from the same start it would have gone the other way
    unpushed = _run(push_chain(0.0, ASYM, 0.0, dt=DT3, init=start), 600, push_states)[-1]
    assert int(jnp.argmax(unpushed)) == 1


# ---- Prediction 5: the choice is visible downstream ----


@pytest.mark.parametrize("p1,expected_choice,expected_y", [(0.2, 1, 0.0), (0.8, 0, 1.0)])
def test_the_choice_reaches_the_downstream_box(p1, expected_choice, expected_y):
    """Two runs differing only in the chooser's start: the composite's downstream state differs by
    the whole range, so which way an internal choice went decides the composite's outcome."""
    arr = choice_chain(0.0, ASYM, 0.0, dt=DT3, init=jnp.array([p1, 1.0 - p1]), eta=0.3, y0=0.5)
    p, y = chain_states(settle(arr, 3000))
    assert int(jnp.argmax(p)) == expected_choice
    assert float(y[0]) == pytest.approx(expected_y, abs=1e-6)


def test_the_pushed_chooser_rejects_a_non_symmetric_payoff():
    with pytest.raises(ValueError):
        pushed_chooser(jnp.array([[0.0, 1.0], [2.0, 0.0]]), 0.1)
