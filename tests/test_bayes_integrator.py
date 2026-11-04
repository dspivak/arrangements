"""Bayes' rule as an integrator (book ``AGENTDOCPLAN.md`` §8.2 P25 (a); notes/BAYES-INTEGRATOR.md).

David's question (2026-09-27): is Bayesian update "the same sort of thing, just a different
integrator"? The claim under test is Chapter 2's -- learning and mechanics are arrangements of
one kind -- which so far covers differentiable learning only.

The derivation was fixed in writing before this file existed (notes/BAYES-INTEGRATOR.md §1).
For a likelihood ``p(x|theta) = h(x) exp(<T(x), theta> - A(theta))`` and a belief in the
conjugate family ``q_lam(theta) = exp(<lam, Psi(theta)> - AA(lam))``, ``Psi = (theta, -A)``:

* Bayes is a translation, ``lam |-> lam + t(x)`` with ``t(x) = (T(x), 1)``;
* the cost ``k_x(lam) = E_{q_lam}[-log p(x|theta)] = -<t(x), mu(lam)> - log h(x)`` is affine
  in the expectation parameter ``mu = grad AA(lam)``, so the forcing is the *exact* one-form
  ``omega_x = -d k_x`` with components ``G(lam) t(x)``, where ``G = grad^2 AA`` is the belief
  family's Fisher information in natural coordinates;
* the **response** is ``sharp_lam = G(lam)^{-1} = grad^2 AA*(mu)``, the raised inverse Fisher
  metric -- the slot the two-layer learner fills with ``eta_LR * I`` and the spring with an
  inverse mass. Then ``lam + sharp_lam(omega_x) = lam + t(x)``: Bayes exactly, not to first
  order.

So in ``def.arr`` the response is ``grad^2 AA*`` and the forcing one-form is ``-d k_x`` on
``Q = Lam``; interface, forcing lens, wiring and ``Phi`` are untouched.

THE FIVE PREDICTIONS, verbatim from notes/BAYES-INTEGRATOR.md §3, written before any
construction code:

1. **Exactness.** ``n`` steps of the arrangement at duration 1, one datum per step, reproduce
   the closed-form posterior mean and variance to machine precision (relative error below
   1e-12).
2. **Batch equals sequential.** One step at duration 1 with the summed forcing
   ``-d sum_i k_{x_i}`` gives the same ``lam`` as the ``n`` sequential steps. Gradient descent
   with any response that is not this one fails this, since only a translation is additive.
3. **Duration is tempering.** Duration ``tau`` gives the belief updated by ``p(x|theta)^tau``
   normalized; so ``tau = 0`` is the identity, and two steps at ``tau = 1/2`` equal one step at
   ``tau = 1``.
4. **The response is what is doing the work.** The Euclidean response ``eta * I`` in the same
   arrangement -- same wiring, same forcing, same interface -- does *not* give the posterior at
   any ``eta``; nor does the Newton response ``(grad^2 k_x)^{-1}``, which is a different matrix
   because ``grad^2 k_x`` involves the third derivative of ``AA``.
5. **Composition survives.** Wiring the belief into a downstream box and running ``Phi`` on the
   composite reproduces the same posterior, so the integrator is compatible with the dynamics
   functor, not only with a hand-run step.

The library is the fixed-duration specialization (STATUS.md; the duration-input API is the book
plan's P9.5), so a duration ``tau`` enters as the scale ``sharp = tau * G^{-1}`` on the
response, which is what ``prop.duration_evaluation``'s "a one-step formula without a duration
means evaluation at one" licenses.

The small example is the conjugate Gaussian with unknown mean and known variance:
``Psi(m) = (m, -m^2/2)``, ``t(x) = (x/sig^2, 1/sig^2)``, belief ``N(lam_1/lam_2, 1/lam_2)``.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.rvect import ReactiveVectorSpace, euclidean, inverse_hessian
from arrangements.wiring import compose_seq, trivial_parameter


# ---------------------------------------------------------------------------
# The conjugate Gaussian belief family, in natural coordinates.
# ---------------------------------------------------------------------------

SIG2 = 0.7  # the known observation variance sig^2
DATA = jnp.array([1.3, -0.4, 2.2, 0.1, 1.9, -1.1, 0.6])
LAM0 = jnp.array([2.0 / 1.5, 1.0 / 1.5])  # prior N(m0=2.0, sig0^2=1.5)


def log_partition(lam):
    """``AA(lam)`` for ``q_lam(m) ∝ exp(lam_1 m - lam_2 m^2/2)``: a Gaussian normalizer."""
    return lam[0] ** 2 / (2.0 * lam[1]) - 0.5 * jnp.log(lam[1]) + 0.5 * jnp.log(2.0 * jnp.pi)


def fisher(lam):
    """``G(lam) = grad^2 AA(lam)``, the belief family's Fisher information."""
    return jax.hessian(log_partition)(lam)


def suff_stat(x):
    """``t(x) = (T(x), 1)`` for the Gaussian likelihood: ``(x/sig^2, 1/sig^2)``."""
    return jnp.array([x[0] / SIG2, 1.0 / SIG2])


def expected_nll(lam, x):
    """``k_x(lam) = E_{q_lam}[-log p(x|m)]``, the cost whose ``-dk`` is the forcing."""
    mean, var = lam[0] / lam[1], 1.0 / lam[1]
    return ((x[0] - mean) ** 2 + var) / (2.0 * SIG2) + 0.5 * jnp.log(2.0 * jnp.pi * SIG2)


def closed_form(lam0, data):
    """The textbook conjugate posterior ``(mean, variance)`` after ``data``."""
    n, xbar = len(data), float(np.mean(np.asarray(data)))
    prec = float(lam0[1]) + n / SIG2
    mean = (float(lam0[0]) + n * xbar / SIG2) / prec
    return mean, 1.0 / prec


def belief_stats(lam):
    """``(mean, variance)`` of ``q_lam``."""
    lam = np.asarray(lam)
    return lam[0] / lam[1], 1.0 / lam[1]


# ---------------------------------------------------------------------------
# The arrangement: a belief box whose datum arrives on its domain's output port.
# ---------------------------------------------------------------------------


def bayes_response(tau=1.0):
    """The Bayes response ``sharp_lam = tau * G(lam)^{-1}``, the raised inverse Fisher."""
    return ReactiveVectorSpace(dim=2, sharp_fn=lambda lam: tau * jnp.linalg.inv(fisher(lam)))


def belief(response, n_data=1, cost=expected_nll, lam0=LAM0):
    """The belief arrangement ``<R^0|R^{n_data}> -> <R^0|R^1>``.

    The environment supplies the datum (or a batch) on the domain's output port, exactly as
    ``learning.py``'s environment streams ``(x, label)``; the box reports the posterior mean
    forward and carries the cost ``k_x`` as its exact forcing ``omega = -dk``. The only thing
    that distinguishes it from the two-layer learner is ``response``.
    """
    return Arrangement(
        P=configuration(response, init=lam0),
        in_dim_M=0, out_dim_M=n_data, in_dim_N=0, out_dim_N=1,
        out_f=lambda lam, x: jnp.array([lam[0] / lam[1]]),
        in_f=lambda lam, x, n_in: jnp.zeros(0),
        omega=exact(lambda lam, x, n_in: -cost(lam, x)),
        label="belief",
    )


def _zero_dir(out_dim_N):
    return lambda _o: (jnp.zeros(out_dim_N), jnp.zeros(0))


def run(arr, data, state=None, out_dim_N=1):
    """Step ``Phi(arr)`` once per datum, the environment returning a zero output covector."""
    O = Phi(arr)
    s = O.state if state is None else state
    in_pos_omega = trivial_omega(0)
    for x in data:
        _, _, s = O.with_state(s).run_one(
            (jnp.atleast_1d(jnp.asarray(x, dtype=float)), in_pos_omega),
            _zero_dir(out_dim_N),
        )
    return s


# ---------------------------------------------------------------------------
# The derivation's two identities, checked before the predictions.
# ---------------------------------------------------------------------------


def test_forcing_is_the_fisher_matrix_times_the_sufficient_statistic():
    """``-d k_x|_lam = G(lam) t(x)``: the forcing the box publishes on ``Q``."""
    arr = belief(bayes_response())
    for lam in (LAM0, jnp.array([0.3, 2.4]), jnp.array([-1.7, 0.45])):
        for x in (jnp.array([1.3]), jnp.array([-0.4])):
            xi_Q, _, _ = arr.omega(lam, x, jnp.zeros(0))
            np.testing.assert_allclose(
                np.asarray(xi_Q), np.asarray(fisher(lam) @ suff_stat(x)), rtol=1e-12, atol=1e-12
            )


def test_the_parameter_covector_the_coalgebra_receives_is_that_forcing():
    """Through ``Phi``, the covector reaching the integrator is ``G(lam) t(x)``.

    A negative control on the plumbing: with a *nonzero* output covector the box would also
    receive the pullback along ``out_f``, so the pure-Bayes reading needs the zero one.
    """
    from arrangements.interpretation import interpretation

    arr = belief(bayes_response())
    _, direction_action = interpretation(arr)(LAM0)
    x = jnp.array([1.3])
    xi_Q, _, _ = direction_action(x, trivial_omega(0), jnp.zeros(1), jnp.zeros(0))
    np.testing.assert_allclose(
        np.asarray(xi_Q), np.asarray(fisher(LAM0) @ suff_stat(x)), rtol=1e-12, atol=1e-12
    )

    xi_Q_pushed, _, _ = direction_action(x, trivial_omega(0), jnp.ones(1), jnp.zeros(0))
    assert not np.allclose(np.asarray(xi_Q_pushed), np.asarray(xi_Q))


# ---------------------------------------------------------------------------
# Prediction 1: exactness.
# ---------------------------------------------------------------------------


def test_prediction_1_posterior_is_exact():
    """``n`` steps, one datum each, give the closed-form posterior to machine precision."""
    lam_n = run(belief(bayes_response()), DATA)
    mean, var = belief_stats(lam_n)
    mean_star, var_star = closed_form(LAM0, DATA)
    assert abs(mean - mean_star) / abs(mean_star) < 1e-12
    assert abs(var - var_star) / var_star < 1e-12

    # and the natural parameter is the prior translated by the summed sufficient statistics
    t_sum = np.sum([np.asarray(suff_stat(jnp.array([float(x)]))) for x in DATA], axis=0)
    np.testing.assert_allclose(np.asarray(lam_n), np.asarray(LAM0) + t_sum, rtol=1e-12)


# ---------------------------------------------------------------------------
# Prediction 2: batch equals sequential.
# ---------------------------------------------------------------------------


def test_prediction_2_batch_equals_sequential():
    """One step with the summed forcing equals the ``n`` sequential steps."""
    n = len(DATA)
    batch_cost = lambda lam, xs: sum(expected_nll(lam, xs[i:i + 1]) for i in range(n))
    lam_batch = run(belief(bayes_response(), n_data=n, cost=batch_cost), [DATA])
    lam_seq = run(belief(bayes_response()), DATA)
    np.testing.assert_allclose(np.asarray(lam_batch), np.asarray(lam_seq), rtol=1e-12)

    # the same comparison with the Euclidean response fails, as prediction 2 says it must
    lam_batch_e = run(belief(euclidean(2, 0.3), n_data=n, cost=batch_cost), [DATA])
    lam_seq_e = run(belief(euclidean(2, 0.3)), DATA)
    assert not np.allclose(np.asarray(lam_batch_e), np.asarray(lam_seq_e), rtol=1e-6)


# ---------------------------------------------------------------------------
# Prediction 3: duration is tempering.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("tau", [0.0, 0.25, 0.5, 1.0, 2.0])
def test_prediction_3_duration_tempers_the_likelihood(tau):
    """One step at ``tau`` is the belief updated by ``p(x|m)^tau``, i.e. ``lam + tau t(x)``."""
    x = jnp.array([1.3])
    lam_1 = run(belief(bayes_response(tau)), [x])
    np.testing.assert_allclose(
        np.asarray(lam_1), np.asarray(LAM0 + tau * suff_stat(x)), rtol=1e-12, atol=1e-14
    )


def test_prediction_3_two_half_steps_equal_one_whole_step():
    """Two steps at ``tau = 1/2`` on the same datum equal one at ``tau = 1``."""
    x = jnp.array([1.3])
    np.testing.assert_allclose(
        np.asarray(run(belief(bayes_response(0.5)), [x, x])),
        np.asarray(run(belief(bayes_response(1.0)), [x])),
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# Prediction 4: the response is what is doing the work.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("eta", [0.01, 0.1, 0.5, 1.0, 2.0])
def test_prediction_4_euclidean_response_misses_the_posterior(eta):
    """Same wiring, same forcing, same interface, Euclidean response: not the posterior."""
    lam_n = run(belief(euclidean(2, eta)), DATA)
    mean, var = belief_stats(lam_n)
    mean_star, var_star = closed_form(LAM0, DATA)
    assert abs(mean - mean_star) > 1e-6 or abs(var - var_star) > 1e-6


def test_prediction_4_newton_response_is_a_different_matrix_and_misses_it():
    """The Newton response ``(grad^2 k_x)^{-1}`` differs from ``G^{-1}`` and misses Bayes."""
    x = jnp.array([1.3])
    k_x = lambda lam: expected_nll(lam, x)
    hess_k = jax.hessian(k_x)(LAM0)
    assert not np.allclose(np.asarray(hess_k), np.asarray(fisher(LAM0)), rtol=1e-6)

    lam_1 = run(belief(inverse_hessian(k_x, 2), n_data=1), [x])
    assert not np.allclose(
        np.asarray(lam_1), np.asarray(LAM0 + suff_stat(x)), rtol=1e-6, atol=1e-8
    )


# ---------------------------------------------------------------------------
# Prediction 5: composition survives.
# ---------------------------------------------------------------------------


def reporter():
    """A downstream box ``<R^0|R^1> -> <R^0|R^1>``: trivial parameter, zero forcing, doubles."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: 2.0 * m,
        in_f=lambda q, m, n_in: jnp.zeros(0),
        omega=lambda q, m, n_in: (jnp.zeros(0), jnp.zeros(1), jnp.zeros(0)),
        label="reporter",
    )


def test_prediction_5_composite_gives_the_same_posterior():
    """``Phi(reporter . belief)`` moves the belief exactly as ``Phi(belief)`` does."""
    composite = compose_seq(belief(bayes_response()), reporter())
    state = run(composite, DATA)
    lam_n, _ = state
    np.testing.assert_allclose(
        np.asarray(lam_n), np.asarray(run(belief(bayes_response()), DATA)), rtol=1e-12
    )
    mean, var = belief_stats(lam_n)
    mean_star, var_star = closed_form(LAM0, DATA)
    assert abs(mean - mean_star) / abs(mean_star) < 1e-12
    assert abs(var - var_star) / var_star < 1e-12


# ---------------------------------------------------------------------------
# Two findings of the run itself, not predicted in advance (notes/BAYES-INTEGRATOR.md §5).
# ---------------------------------------------------------------------------


def expectation_reporter(response, lam0=LAM0):
    """The belief in the *learner's* shape: it reports ``(mu, x)`` forward and forces nothing.

    ``out_f`` is the Legendre map ``grad AA`` paired with the datum passed through, and
    ``omega = 0`` -- exactly ``learning.py``'s network, whose learning signal comes from
    whatever it is wired into rather than from its own forcing.
    """
    return Arrangement(
        P=configuration(response, init=lam0),
        in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=3,
        out_f=lambda lam, x: jnp.concatenate([jax.grad(log_partition)(lam), x]),
        in_f=lambda lam, x, n_in: jnp.zeros(0),
        omega=lambda lam, x, n_in: (jnp.zeros(2), jnp.zeros(1), jnp.zeros(0)),
        label="belief(mu)",
    )


def likelihood_box():
    """The downstream cost ``k(mu, x) = -<t(x), mu>``, closing the composite: the loss box."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, z: jnp.zeros(0),
        in_f=lambda q, z, n_in: jnp.zeros(0),
        omega=exact(lambda q, z, n_in: jnp.dot(suff_stat(z[2:3]), z[0:2])),
        label="likelihood",
    )


def test_bayes_also_has_the_learners_shape_loss_downstream():
    """The forcing may sit in a downstream likelihood box; the backward pass still gives Bayes.

    Not predicted in advance: prediction 5 asked only that composition not disturb the
    posterior. Here the belief box carries *no* forcing and reports its expectation parameters
    ``mu = grad AA(lam)``; the downstream box's ``omega = -dk`` pushes ``t(x)`` back on the
    ``mu`` port; the pullback along the Legendre map is ``(grad^2 AA)^T t(x) = G(lam) t(x)``,
    and the inverse-Fisher response turns that into ``lam + t(x)``. So Bayes occupies the
    learner's slot in ``def.arr`` as well as Newton's: the loss is a separate box and
    backpropagation supplies the parameter covector.
    """
    composite = compose_seq(expectation_reporter(bayes_response()), likelihood_box())
    lam_n, _ = run(composite, DATA, out_dim_N=0)
    mean, var = belief_stats(lam_n)
    mean_star, var_star = closed_form(LAM0, DATA)
    assert abs(mean - mean_star) / abs(mean_star) < 1e-12
    assert abs(var - var_star) / var_star < 1e-12
    np.testing.assert_allclose(
        np.asarray(lam_n), np.asarray(run(belief(bayes_response()), DATA)), rtol=1e-12
    )


def test_matches_khan_and_rue_equation_54():
    """The run's ``lam_n`` is the fixed point of the Bayesian learning rule, Eq. (54), p. 22.

    Khan and Rue (JMLR 24(281), 2023; arXiv:2107.04562v4) §5.1 Eq. (54) writes the conjugate
    posterior's natural parameter as ``lam* = grad_mu E_q*[log p(theta)] + sum_i grad_mu
    E_q*[log p(y_i|theta)]``, "equivalent to running the BLR for just one-iteration with
    rho_t = 1". Their natural gradient is ``F(lam)^{-1} grad_lam = grad_mu`` with
    ``F = grad^2_lam A`` (Eq. (4), p. 3), which is this arrangement's response. Both gradients
    are taken here by autodiff in ``lam`` and converted with ``G^{-1}``, so the check is not
    circular.
    """
    M0, SIG0_2 = 2.0, 1.5     # the prior N(m0, sig0^2) that LAM0 encodes

    def grad_mu(f, lam):
        return jnp.linalg.inv(fisher(lam)) @ jax.grad(f)(lam)

    def log_prior(lam):
        mean, var = lam[0] / lam[1], 1.0 / lam[1]
        return -((mean - M0) ** 2 + var) / (2.0 * SIG0_2) - 0.5 * jnp.log(2.0 * jnp.pi * SIG0_2)

    lam_n = run(belief(bayes_response()), DATA)
    rhs = grad_mu(log_prior, lam_n)
    np.testing.assert_allclose(np.asarray(rhs), np.asarray(LAM0), rtol=1e-10)  # = lam_0
    for x in DATA:
        xa = jnp.array([float(x)])
        rhs = rhs + grad_mu(lambda lam: -expected_nll(lam, xa), lam_n)
    np.testing.assert_allclose(np.asarray(rhs), np.asarray(lam_n), rtol=1e-10)


def test_the_forcing_of_a_bayes_arrangement_is_nowhere_zero():
    """Bayes never settles: its cost has no critical point in ``Lam``, so the forcing never dies.

    Not predicted in advance. ``-d k_x = G(lam) t(x)`` with ``G`` positive definite and
    ``t(x) = (T(x), 1)`` never zero (its last entry is the observation count), so the forcing
    vanishes nowhere on ``Lam``. The infimum of ``k_x`` over realizable beliefs is attained only
    on the *boundary* -- infinite precision, mean at the datum -- which is outside ``Lam``, and
    the orbit on a repeated datum runs to it as ``1/n``, never reaching it: the variance after
    ``n`` copies is exactly ``1/(1/sig0^2 + n/sig^2)``. So the arrangement has no driven steady
    state on repeated data, unlike a spring or a learner descending a loss with an interior
    minimum; what keeps one Bayes step from collapsing the belief is the unit step size, which
    is the ``KL`` term of the proximal form.
    """
    x = jnp.array([1.3])
    for lam in (LAM0, jnp.array([0.3, 2.4]), jnp.array([-1.7, 0.45]), jnp.array([94.0, 72.0])):
        xi_Q, _, _ = belief(bayes_response()).omega(lam, x, jnp.zeros(0))
        assert float(jnp.linalg.norm(xi_Q)) > 1e-6
        assert float(jnp.linalg.det(fisher(lam))) > 0.0

    lam, costs = LAM0, []
    for _ in range(50):
        lam = run(belief(bayes_response()), [x], state=lam)
        costs.append(float(expected_nll(lam, x)))
    mean, var = belief_stats(lam)
    np.testing.assert_allclose(var, 1.0 / (float(LAM0[1]) + 50.0 / SIG2), rtol=1e-12)
    assert abs(mean - 1.3) < 0.01 and var < 0.02        # running to the boundary as 1/n
    assert all(b < a for a, b in zip(costs, costs[1:]))  # monotone, toward a boundary infimum
    assert costs[-1] > 0.5 * float(jnp.log(2.0 * jnp.pi * SIG2))  # which it never reaches
