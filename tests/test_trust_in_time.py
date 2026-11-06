"""Learning whom to trust, run in time (book Ch 5 ``ex.learning_trust``, on Ch 4's communication networks).

David, 2026-10-29, of the batch run this replaces (``test_shared_evidence.py`` part 4, where the days sat in the
coordinates of one value space ``R^400`` and every member sent all their reports at every step): "seems like a total
violation of the model. how could this be at all reasonable? the whole thing is a time-based physics. now you're
cheating that? untenable." Then "yes please" to rebuilding it in time (Claude Opus 5.5, b0471e87). Rebuilt on values
and proportions by P95 (Claude Fable 5.1, bcc1d91e, 2026-11-06): the message weights are gone, and Bob's trust-levels
are softmax of three scores.

**The arrangement, as the book draws it.** Bob is ``phi : R, R, R -> R`` of the attention operad, silent: three
internal boxes of object ``R``, the outside hearing member ``i`` in the proportion ``q_i = softmax(s)_i`` (Ch 4
``eqn.network_integrator``: state the scores ``s``, configuration the proportions, readout softmax), the rest zero;
no voice constant; cost the book's one-day cost in ``ex.learning_trust`` (``eqn.trust_day_cost``),
``(1/2)(z - sum_i q_i y_i)^2``; the integrator pulls the load at ``q`` back to the scores along softmax and steps
at the learning rate ``mu``.

**Time.** A day lasts duration ``1``. Each member's box is filled by a part whose state is a clock ``t``, stepping
``t + duration``, that reads out ``y_i[floor t]``, that day's report, and ignores the force that comes back. The
outside of ``phi`` is filled by a world, an arrangement with ``phi``'s outer box as its one internal box, whose matter
is a clock as well and which delivers ``z[floor t]``, that day's true number, to Bob's input. Every day is a fresh
draw; nothing repeats. The whole is closed and run by ``closed_step`` (the dynamics functor ``Phi``, compiled,
checked against ``Phi`` on its first steps) at durations ``1/4`` and ``1/8``, which land exactly on the day
boundaries.

**Predictions, written before the run** (the agent's, from Ch 5 ``prop.trust_precisions``(b)): Bob's expected cost
on one day is ``(1/2) sum_i q_i^2 sigma_i^2``, least on the simplex at the precision shares ``(4, 1, 1/4) / 5.25``
``= (0.762, 0.190, 0.048)``, where the mixture's squared error is ``1/5.25 = 0.190``, below the best member's ``0.25``.
Since Bob's scores move slowly against the days (``mu`` small), what moves him over many days is the days' average of
his inclination, so (1) his proportions settle near those shares, with a day-to-day jitter that shrinks with ``mu``;
(2) halving the duration does not change the run beyond the jitter; (3) his readout's value on the last days has
squared error near ``0.190``. Each can fail.

**What happened** is recorded by the assertions below; the run takes a few seconds.

Reasonable and weird (the agent's judgment, not evidence): reasonable for a forecaster weighting staff by track record
or a body weighting senses by reliability, with trust moving a little each day, and now the weights are shares of
one, as a forecaster's are. Weird: the members' reports change only at day boundaries, so their parts' readouts jump
there (a report delivered once a day); the truth arrives the same day as the reports, so the story's wait between
report and truth is not in the model.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import CotCoalg
from arrangements.functors import closed_step
from arrangements.wiring import compose_seq, tensor_arrangements

jax.config.update("jax_enable_x64", True)

EMPTY = jnp.zeros(0)
SIGMAS = np.array([0.5, 1.0, 2.0])     # the members' errors; precisions 4, 1, 1/4
DAYS = 20000
MU = 0.05


def days(n=DAYS, sigmas=SIGMAS, seed=0):
    """True numbers ``z ~ N(0, 1)`` and reports ``y_i = z + sigma_i eps_i``, one of each per day."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    return z, z[None, :] + sigmas[:, None] * rng.standard_normal((len(sigmas), n))


def clock(label):
    """A part whose state is the time ``t``: position ``t``, a step of duration ``tau`` goes to ``t + tau`` whatever
    covector comes back (its unit step is a day)."""
    return CotCoalg(Q_dim=1, init=jnp.zeros(1), position=lambda s: s, step=lambda s, xi: s + 1.0, label=label,
                    timed_step=lambda s, xi, tau: s + tau)


def today(series, t):
    return jnp.asarray(series)[jnp.minimum(jnp.floor(t[0]).astype(int), len(series) - 1)]


def member(reports, label):
    """Member ``i``'s box filled: on day ``floor t`` it sends the value ``y_i`` for that day."""
    return Arrangement(
        P=clock(label), in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: jnp.array([today(reports, q)]),
        in_f=lambda q, m_out, n_in: EMPTY, omega=lambda q, m, n: (jnp.zeros(1), EMPTY, jnp.zeros(1)), label=label)


def scores(mu, label="scores"):
    """Bob's integrator (Ch 4 eqn.network_integrator, one row): state three scores ``s``, configuration
    ``softmax(s)``, update ``s + tau mu (d softmax)^T xi`` for the load ``xi`` at the configuration."""
    def pull(s, xi):
        _, vjp = jax.vjp(jax.nn.softmax, s)
        (xi_s,) = vjp(xi)
        return xi_s

    return CotCoalg(Q_dim=3, init=jnp.zeros(3), position=jax.nn.softmax,
                    step=lambda s, xi: s + mu * pull(s, xi), label=label,
                    timed_step=lambda s, xi, tau: s + tau * mu * pull(s, xi))


def bob_arrangement(mu):
    """``phi : R, R, R -> R``: out-transfer ``sum_i q_i y_i``; cost ``(1/2)(z - sum_i q_i y_i)^2`` on the value ``z``
    that comes in; the members hear nothing."""
    def potential(q, m_out, n_in):             # omega = dU, U = -cost (STATUS.md §1)
        return -0.5 * (n_in[0] - jnp.dot(q, m_out)) ** 2

    return Arrangement(
        P=scores(mu), in_dim_M=3, out_dim_M=3, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: jnp.array([jnp.dot(q, m_out)]),
        in_f=lambda q, m_out, n_in: jnp.zeros(3),
        omega=exact(potential), label="bob")


def world(truth, bob):
    """The outside of ``phi``: one internal box, Bob's; it delivers ``z`` for the day to his input, ignores what he
    reads out and the force he pushes back."""
    return Arrangement(
        P=clock("world"), in_dim_M=bob.in_dim_N, out_dim_M=bob.out_dim_N, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY,
        in_f=lambda q, m_out, n_in: jnp.array([today(truth, q)]),
        omega=lambda q, m, n: (jnp.zeros(1), jnp.zeros_like(m), EMPTY), label="world")


def town(z, Y, mu=MU):
    bob = compose_seq(tensor_arrangements([member(Y[i], f"member{i + 1}") for i in range(3)]), bob_arrangement(mu))
    return bob, compose_seq(bob, world(z, bob))


def run_in_time(duration, z, Y, mu=MU, n_days=DAYS):
    """Positions ``(t_1, t_2, t_3, q_1, q_2, q_3, t_world)`` on each day, and Bob's readout on each day, read off at
    the day's first step."""
    bob, system = town(z, Y, mu)
    step = closed_step(system, duration=duration)
    per_day = int(round(1 / duration))
    readout = jax.jit(lambda q: bob.out_f(q[:6], EMPTY))

    @jax.jit
    def one_day(s, _):
        q = system.P.position(s)
        r = readout(q)
        s = jax.lax.fori_loop(0, per_day, lambda _i, t: step(t), s)
        return s, (q, r)

    _, (qs, rs) = jax.lax.scan(one_day, system.P.init, None, length=n_days)
    return np.asarray(qs), np.asarray(rs)


def test_trust_learned_in_time():
    z, Y = days()
    shares = (1 / SIGMAS ** 2) / (1 / SIGMAS ** 2).sum()
    out = {}
    for duration in (0.25, 0.125):
        qs, rs = run_in_time(duration, z, Y)
        assert np.allclose(qs[:, 6], np.arange(DAYS)), "the world's clock counts days"
        assert np.allclose(qs[:, :3], qs[:, 6:7]), "the members' clocks keep the world's"
        props = qs[:, 3:6]
        assert np.allclose(props.sum(axis=1), 1.0), "Bob's proportions are a distribution every day"
        second_half = props[DAYS // 2:]
        assert np.abs(second_half.mean(axis=0) - shares).max() < 0.03, (second_half.mean(axis=0), shares)
        assert second_half.std(axis=0).max() < 0.05
        err = (rs[DAYS // 2:, 0] - z[DAYS // 2:]) ** 2
        assert 0.15 < err.mean() < 0.23, err.mean()
        assert err.mean() < ((Y[0, DAYS // 2:] - z[DAYS // 2:]) ** 2).mean(), "better than the most reliable member"
        out[duration] = props
    assert np.abs(out[0.25] - out[0.125]).max() < 0.01, "halving the duration changes nothing beyond jitter"
