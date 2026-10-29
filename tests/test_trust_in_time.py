"""Learning whom to trust, run in time (book Ch 4 ``ex.learning_trust``).

David, 2026-10-29, of the batch run this replaces (``test_shared_evidence.py`` part 4, where the days sat in the
coordinates of one value space ``R^400`` and every member sent all their reports at every step): "seems like a total
violation of the model. how could this be at all reasonable? the whole thing is a time-based physics. now you're
cheating that? untenable." Then "yes please" to rebuilding it in time (Claude Opus 5.5, b0471e87).

**The arrangement, as the book draws it.** Bob is ``phi : R, R, R -> R`` of the attention operad: three internal boxes
of color ``R``, transfers ``T_0i(q) = gamma_i C(id)`` with ``gamma_i = exp(-q_i^2 / 2)``, the rest zero; matter
``Q = R^3`` with no internal inclination, responding by the learning rate ``mu``; cost the private prediction
(``eqn.private_prediction``) on one day, ``(1/2)(z - sum_i gamma_i y_i)^2 + (1/2)(1 - sum_i gamma_i)^2``.

**Time.** A day lasts duration ``1``. Each member's box is filled by a part whose state is a clock ``t``, stepping
``t + duration``, that reads out ``msg(y_i[floor t])``, that day's report, and ignores the force that comes back. The
outside of ``phi`` is filled by a world, an arrangement with ``phi``'s outer box as its one internal box, whose matter
is a clock as well and which delivers ``msg(z[floor t])``, that day's true number, to Bob's input. Every day is a
fresh draw; nothing repeats. The whole is closed and run by ``closed_step`` (the dynamics functor ``Phi``, compiled,
checked against ``Phi`` on its first steps) at durations ``1/4`` and ``1/8``, which land exactly on the day
boundaries. A report changes only at a day boundary, so the clock parts' readouts are constant within a day and jump
between days: they are not smooth at the boundaries, which no step lands inside.

**Predictions, written before the run** (the agent's, from Ch 4's own calculation): Bob's expected cost on one day is
``(1 - gamma)^2 + (1/2) sum_i gamma_i^2 sigma_i^2``, ``gamma = sum_i gamma_i``, smallest at ``gamma_i`` proportional to
the precisions ``4 : 1 : 1/4`` with ``gamma = 10.5 / 11.5``, so at loudness ``(0.696, 0.174, 0.043)`` and positions
``(0.852, 1.870, 2.503)``. Since Bob's matter moves slowly against the days (``mu`` small), what moves him over many
days is the days' average of his inclination, so (1) his positions settle near those, with a day-to-day jitter that
shrinks with ``mu``; (2) halving the duration does not change the run beyond the jitter; (3) his readout's value on
the last days has squared error near ``1/5.25 = 0.190``, below the best member's ``0.25``. Each can fail.

**What happened** (2026-10-29, 2.8 s, passes; ``mu = 0.01``, 20000 days, seed 0). All three held. At duration ``1/4``:
positions at the end ``(0.858, 1.829, 2.519)``, over the second half ``(0.863, 1.848, 2.519)`` with day-to-day spread
``(0.021, 0.023, 0.017)``; loudness at the end ``(0.692, 0.188, 0.042)``, relative ``(0.751, 0.204, 0.045)``, total
``0.922``; readout squared error over the second half ``0.1949``, the best member's ``0.2581`` on those days. Duration
``1/8`` agrees to ``0.0015`` in every position over the whole run, duration ``1`` to ``0.011``. At ``mu = 0.05`` the
spread is about ``0.05``; at ``mu = 0.002`` 20000 days are too few for the third member to arrive. Each day's readout
is read at the day's first step, before Bob has learned from that day, so every error is on a day he had not seen.

Reasonable and weird (the agent's judgment, not evidence): reasonable for a forecaster weighting staff by track record
or a body weighting senses by reliability, with trust moving a little each day. Weird: the members' reports change
only at day boundaries, so their parts' readouts jump there (a report delivered once a day); the truth arrives the same
day as the reports, so the story's wait between report and truth is not in the model.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.attention import input_dim, make_input, message, msg_dim, signal, weight
from arrangements.coalgebra import CotCoalg
from arrangements.functors import closed_step
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_shared_evidence import hub

jax.config.update("jax_enable_x64", True)

EMPTY = jnp.zeros(0)
SIGMAS = np.array([0.5, 1.0, 2.0])     # the members' errors; precisions 4, 1, 1/4
DAYS = 20000
MU = 0.01


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
    """Member ``i``'s box filled: on day ``floor t`` it sends ``msg(y_i)`` for that day, weight 1."""
    return Arrangement(
        P=clock(label), in_dim_M=0, out_dim_M=0, in_dim_N=input_dim(1, 1), out_dim_N=msg_dim(1),
        out_f=lambda q, m_out: message(today(reports, q), 1.0),
        in_f=lambda q, m_out, n_in: EMPTY, omega=zero_form, label=label)


def world(truth, bob):
    """The outside of ``phi``: one internal box, Bob's; it delivers ``msg(z)`` for the day, placement ``0``, to his
    input, ignores what he reads out and the force he pushes back."""
    return Arrangement(
        P=clock("world"), in_dim_M=bob.in_dim_N, out_dim_M=bob.out_dim_N, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY,
        in_f=lambda q, m_out, n_in: make_input(message(today(truth, q), 1.0), 0.0),
        omega=zero_form, label="world")


def per_day_cost(q, m_out, e):
    """``eqn.private_prediction`` on one day, as ``test_shared_evidence.trust_system`` writes it with one trial."""
    ms = m_out.reshape(3, 2)
    pred = (jnp.exp(-q ** 2 / 2)[:, None] * ms).sum(0)
    d = e[:2] - pred
    return 0.5 * d[0] ** 2 + 0.5 * d[1] ** 2 + 0.5 * e[2] ** 2


def town(z, Y, mu=MU):
    bob_hub = hub(3, 1, lambda q: jnp.exp(-q ** 2 / 2), lambda q: q.reshape(3, 1), per_day_cost, mu, [1.0, 1.0, 1.0])
    bob = compose_seq(tensor_arrangements([member(Y[i], f"member{i + 1}") for i in range(3)]), bob_hub.arr())
    return bob, compose_seq(bob, world(z, bob))


def run_in_time(duration, z, Y, mu=MU, n_days=DAYS, every_days=100):
    """Positions ``(t_1, t_2, t_3, q_1, q_2, q_3, t_world)`` sampled every ``every_days`` days, and Bob's readout on
    each day, read off at the day's first step."""
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
    target_g = (10.5 / 11.5) * (1 / SIGMAS ** 2) / (1 / SIGMAS ** 2).sum()
    out = {}
    for duration in (0.25, 0.125):
        qs, rs = run_in_time(duration, z, Y)
        assert np.allclose(qs[:, 6], np.arange(DAYS)), "the world's clock counts days"
        assert np.allclose(qs[:, :3], qs[:, 6:7]), "the members' clocks keep the world's"
        last = slice(DAYS // 2, None)
        q_end, q_mean, q_sd = np.abs(qs[-1, 3:6]), np.abs(qs[last, 3:6]).mean(0), np.abs(qs[last, 3:6]).std(0)
        g = np.exp(-q_end ** 2 / 2)
        value = rs[:, 0] / rs[:, 1]
        err = np.mean((value[last] - z[last]) ** 2)
        best = np.mean((Y[0, last] - z[last]) ** 2)
        out[duration] = q_end
        print(f"duration {duration}: positions at the end {np.round(q_end, 3)}, over the last half "
              f"{np.round(q_mean, 3)} +- {np.round(q_sd, 3)}; loudness {np.round(g, 3)}, relative "
              f"{np.round(g / g.sum(), 3)}, total {g.sum():.3f}; readout squared error {err:.4f}, best member {best:.4f}")
        assert np.allclose(q_mean, np.sqrt(-2 * np.log(target_g)), atol=0.1)
        assert err < best and abs(err - 1 / 5.25) < 0.02
    assert np.allclose(out[0.25], out[0.125], atol=0.02)
