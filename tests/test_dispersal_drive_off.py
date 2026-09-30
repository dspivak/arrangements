"""Dispersal with the drive switched off: does the collective's message dissolve?

**The question.** The book's Ch 1 §1.7 says a form that plays "dissolves when the driving
stops or the parts stop staying attuned," and its footnote cites ``tests/test_dispersal.py``,
which shows the collective's message *returning* after a part is driven out, not dissolving
(book repo ``notes/DOMAIN-CHECK-CH1-2026-09-29.md``, Group B, run 1, objection 3). Here the
same collective is settled and then its parent's drive ``nu`` is switched off.

**The model.** ``test_dispersal.build``: ``n`` frozen parts (no configuration, no cost, no
need), each always sending ``(1, 1)``, on a spatial parent whose cost is
``(1/2)(nu - h~)^2`` on the total ``h~ = sum_j kappa_j``, ``kappa_j = exp(-p_j^2 / 2 tau)``,
``tau = 0.8``; the placements descend with mobility ``mu = 0.5`` per step; the value maps are
fixed. Settled at ``nu = 1.5`` with every ``kappa = 1/2``, ``p* = sqrt(2 tau ln 2) = 1.053``;
then ``nu = 0``; then, in test 3, ``nu = 1.5`` again.

**The prediction** (written first, book repo ``notes/ch1-tests-2026-09-29/dispersal-drive-off.md``).
Descent gives ``dp_j/dt = -mu (nu - h~)(p_j/tau) kappa_j``; at ``nu = 0`` that is
``+mu h~ p_j kappa_j / tau > 0``, every part pushed out, with no minimum at a finite placement.
By symmetry ``dp/dt = mu n p e^{-p^2/tau} / tau``, so the time to reach ``p`` is the quadrature
``t(p) = (tau/(mu n)) int_{p*}^{p} e^{q^2/tau}/q dq``: the placement drifts out like
``sqrt(tau log t)`` and never stops, and the message decays to ``0`` like ``t^{-1/2}`` (up to a
log), ``h~ ~ sqrt(n) tau / (p sqrt(2 mu t))``; a larger ``n`` is pushed out faster, in time
``prop. 1/n``. Switched back on, the parts are recalled in about
``(tau/(mu nu)) int^d e^{p^2/2tau}/p dp ~ e^{d^2/2tau}`` steps, which, since ``e^{d^2/tau}``
grows like the time off, grows like ``sqrt(T_off)``: always recalled, ever more slowly. Control:
with the parent's cost removed nothing moves and the message stays at ``1.5``.

**The result.** Every prediction holds (tests 1-4). ``n = 3``: ``h~ = 0.0591, 0.0160, 0.00449,
0.00129`` after ``10^2 .. 10^5`` steps off (log-log slope ``-0.55``), placement ``2.51 .. 3.52``,
each within 3% of the quadrature's time. Recall after ``10^2 .. 10^5`` steps off takes ``16, 32,
76, 199`` steps: leaving to a distance takes about the square of the time coming back from it
takes. Cost removed: nothing moves, the message stays ``1.5``. Target lowered to ``0.3``: the
parts move out and stop where ``kappa = 0.1``; the message follows the target, not zero.

**What it does not show.** "Driving stops" is modelled as the parent's target going to zero,
which is the parent *wanting* silence and pushing the parts away; if instead the parent stops
caring (its cost removed), nothing dissolves, because the parts are frozen and send forever
with no cost or dissipation of their own. So the run does not show a played form fading for
lack of drive; it shows a collective dispersing because the whole it sends to wants nothing
from it. The parts have no needs, no dynamics and no other place to go; the kernel is Gaussian
on a line, so "out" is ``kappa -> 0`` asymptotically, never ``0``. The "parts stop staying
attuned" half of the sentence is not run here.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.attention import signal
from arrangements.coalgebra import configuration
from arrangements.collective import (
    EXT, collective, frozen_part, position, run, sent_onward, split, trace,
)

from tests.test_dispersal import MU, SIG, TAU, build, distance
from tests.test_rendezvous import run_phi

NU = 1.5
P_STAR = distance(0.5)          # every kappa = 1/2 at nu = n/2, here n = 3
N = 3


def drift(n, nu_off, steps, positions=None):
    """From ``positions`` (default all at ``P_STAR``) run the collective with target
    ``nu_off``; returns ``(message, placements)`` at every step."""
    positions = [P_STAR] * n if positions is None else positions
    C = build(positions, nu_off)
    obs = lambda s: jnp.concatenate([signal(sent_onward(C, position(C, s))),
                                     split(C, position(C, s))[1]])
    _, tr = trace(C, steps, obs)
    tr = np.asarray(tr, float)
    return tr[:, 0], tr[:, 1:]


def t_quad(p, n, p0=P_STAR):
    """``(tau/(mu n)) int_{p0}^{p} e^{q^2/tau}/q dq``: the symmetric drift-out time."""
    q = np.linspace(p0, p, 200001)
    return TAU / (MU * n) * np.trapezoid(np.exp(q ** 2 / TAU) / q, q)


def recall_ode(d, n=N, nu=NU, dt=1e-3, t_max=5000.0):
    """The continuous-time recall ``dp/dt = -mu (nu - n kappa)(p/tau) kappa`` from ``d``,
    until ``n kappa >= 0.99 nu``; returns the time (steps are unit time)."""
    p, t = d, 0.0
    while n * np.exp(-p * p / (2 * TAU)) < 0.99 * nu and t < t_max:
        k = np.exp(-p * p / (2 * TAU))
        p -= dt * MU * (nu - n * k) * (p / TAU) * k
        t += dt
    return t


def recall(d, n=N, nu=NU, steps=20000):
    """Steps until the message is within 1% of ``nu``, from every part at ``d``, and the
    final placements."""
    C = build([d] * n, nu)
    _, sig = trace(C, steps, lambda s: signal(sent_onward(C, position(C, s)))[0])
    hit = np.nonzero(np.asarray(sig) >= 0.99 * nu)[0]
    q = np.asarray(position(C, run(C, steps)))
    return (int(hit[0]) + 1 if hit.size else None), np.asarray(split(C, q)[1])


# ---------------------------------------------------------------------------


def test_0_the_run_is_the_dynamics_functors_and_starts_settled():
    """LICENSES the runs: ``Phi`` agrees with ``collective.run`` at ``nu = 0``, and at
    ``nu = 1.5`` every part at ``P_STAR`` is a fixed point."""
    C = build([P_STAR, 1.2, 0.9], 0.0)
    np.testing.assert_allclose(np.asarray(position(C, run(C, 6))), run_phi(C.arr, 6, EXT)[-1],
                               atol=1e-6)
    C = build([P_STAR] * N, NU)
    q = np.asarray(position(C, run(C, 1000)))
    np.testing.assert_allclose(np.asarray(split(C, q)[1]), P_STAR, atol=1e-5)
    assert abs(float(signal(sent_onward(C, q))[0]) - NU) < 1e-5


def test_1_drive_off_the_message_decays_like_one_over_root_t_and_the_parts_never_stop():
    """REPRODUCES the quadrature and CHECKS the asymptotic rates: with ``nu = 0`` the
    message falls monotonically, the placements rise monotonically and together, each
    placement is reached within 5% of the quadrature's time, the log-log slope of the
    message over ``10^3 .. 10^5`` lies in ``[-0.6, -0.5]``, it matches
    ``sqrt(n) tau/(p sqrt(2 mu t))`` within 10% at ``10^5``, and the placement is still
    moving then (up more than 0.25 over the last decade)."""
    h, p = drift(N, 0.0, 100000)
    assert np.all(np.diff(h) <= 0.0) and np.all(np.diff(p[:, 0]) >= 0.0)
    np.testing.assert_allclose(p, p[:, :1] * np.ones((1, N)), atol=1e-6)
    for t in (100, 1000, 10000, 100000):
        assert abs(t_quad(p[t - 1, 0], N) / t - 1.0) < 0.05, (t, p[t - 1, 0])
    ts = np.array([1000, 10000, 100000])
    slopes = np.diff(np.log(h[ts - 1])) / np.diff(np.log(ts))
    assert np.all((slopes > -0.6) & (slopes < -0.5)), slopes
    t = 100000
    asym = np.sqrt(N) * TAU / (p[t - 1, 0] * np.sqrt(2 * MU * t))
    assert abs(h[t - 1] / asym - 1.0) < 0.10, (h[t - 1], asym)
    assert h[t - 1] < 0.002
    assert p[99999, 0] - p[9999, 0] > 0.25
    print("\n  steps off  message  placement")
    for t in (1, 10, 100, 1000, 10000, 100000):
        print(f"  {t:>6}  {h[t - 1]:.5f}  {p[t - 1, 0]:.4f}")
    print("  slopes", slopes)


def test_2_a_larger_collective_is_pushed_out_faster_in_proportion_to_n():
    """REPRODUCES the quadrature's ``1/n``: the steps for every part to pass ``p = 3.5``
    at ``n = 1, 3, 8``, times ``n``, agree within 10%, each within 5% of ``t_quad``. An
    asymmetric start (``kappa = 0.7, 0.5, 0.3``) sends every part out too."""
    times = {}
    for n in (1, 3, 8):
        _, p = drift(n, 0.0, 400000 // n)
        hit = np.nonzero(p[:, 0] >= 3.5)[0]
        assert hit.size, n
        times[n] = int(hit[0]) + 1
        assert abs(t_quad(3.5, n) / times[n] - 1.0) < 0.05, (n, times[n], t_quad(3.5, n))
    scaled = [n * t for n, t in times.items()]
    assert max(scaled) / min(scaled) < 1.10, times
    start = [distance(k) for k in (0.7, 0.5, 0.3)]
    h, p = drift(3, 0.0, 20000, positions=start)
    assert np.all(p[-1] > np.array(start) + 1.0), p[-1]
    assert h[-1] < 0.01
    print("\n  steps to p = 3.5:", times)


def test_3_switched_back_on_the_parts_are_recalled_in_about_root_time_off():
    """REPRODUCES the recall's quadrature (the continuous-time ODE, within 25%) and
    CHECKS its growth: after ``10^2 .. 10^5`` steps off, switching ``nu`` back to ``1.5``
    brings the message to 99% of ``nu`` every time, the parts back to ``P_STAR``, in a
    number of steps that grows with the time off, by more than 1.8 and less than
    ``sqrt 10`` per decade, and is always far shorter than the time off was."""
    _, p = drift(N, 0.0, 100000)
    rows = []
    for t_off in (100, 1000, 10000, 100000):
        d = float(p[t_off - 1, 0])
        t_back, final = recall(d)
        assert t_back is not None, t_off
        np.testing.assert_allclose(final, P_STAR, atol=1e-4)
        pred = recall_ode(d)
        assert abs(t_back / pred - 1.0) < 0.25, (t_off, t_back, pred)
        rows.append((t_off, d, t_back, pred))
    back = np.array([r[2] for r in rows], float)
    ratios = back[1:] / back[:-1]
    assert np.all((ratios > 1.8) & (ratios < np.sqrt(10.0))), ratios
    assert all(r[2] < 0.2 * r[0] for r in rows)
    print("\n  steps off  distance  steps to recall  ODE")
    for r in rows:
        print(f"  {r[0]:>6}  {r[1]:.4f}  {r[2]}  {r[3]:.1f}")


def test_4_controls_cost_removed_nothing_moves_target_lowered_the_parts_stop():
    """REPRODUCES two closed forms. With the parent's cost removed, the forcing is zero:
    after ``10^4`` steps the placements are at ``P_STAR`` and the message is ``1.5``
    exactly (the frozen parts keep sending; nothing dissolves). With the target lowered to
    ``0.3``, the parts move out and stop where ``3 kappa = 0.3``,
    ``p = distance(0.1) = 1.9194``, and the message is ``0.3``: it follows the target."""
    C = collective([frozen_part(SIG, 1.0, f"C{j}") for j in range(N)], tau=TAU,
                   cost=lambda q, m_out, n_in: 0.0, positions=[P_STAR] * N,
                   integrator=lambda Q, init: configuration(Q, init=init),
                   mobility=MU, value_mobility=0.0)
    q = np.asarray(position(C, run(C, 10000)))
    np.testing.assert_array_equal(np.asarray(split(C, q)[1]), P_STAR)
    assert abs(float(signal(sent_onward(C, q))[0]) - NU) < 1e-12
    h, p = drift(N, 0.3, 20000)
    np.testing.assert_allclose(p[-1], distance(0.1), atol=1e-4)
    assert abs(h[-1] - 0.3) < 1e-5
