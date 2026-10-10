"""Spells and timescales: stage B of the sustained-note run (STATUS.md §3.16, §3.17).

David, 2026-09-26: "maybe the incantation gets you to a new equilibrium that's less aligned
with your surroundings. but now you're singing this strange sustained note to them for a
while. that could change things at another time scale, or you might settle back down to your
old way after a few minutes, but still you can be a part of some longer-time-scale note that's
playing to inchant something else." Three pieces, run in the order he numbered them in the
reply: (1) a spell into a rest less aligned with the surrounds wears off; (3) a slow habit
races the fall back; (4) a population's note, held by no member, leaves a lasting mark in a
slower reader.

Readers are those of ``tests/test_sustained_note.py``: the committed cost
``(1/4)(q^2 - 1)^2 - u q``, Euler--Maruyama noise at ``T``, Euclidean response. The
*surrounds* are a weak note ``S = 0.1`` held on every reader, which makes the right well the
lower one. The statistics use vectorized copies of the step; ``test_the_habit_step_is_Phi``
checks the two-coordinate one pathwise against ``Phi`` of the seeded arrangement, as
``test_the_fast_step_is_Phi`` does for the one-coordinate one.

**Predictions, written before any of this ran (Claude, 2026-09-26).**

1. *Less aligned, and back.* Readers settled under the surrounds, then a spell of -0.6 for 200
   steps: just after, fewer than 5% are right; they relax back to the Gibbs value within 0.05, at
   rate ``r_L + r_R`` (Kramers, each over its own barrier) within a factor 2. A reader's mean
   stay, floor to the other floor, in the lower well over the higher is Kramers's ratio within a
   factor 2. **Could fail**, on the prefactors.
3. *The habit's race.* A second, slow coordinate ``w`` (response ``eta_w``) pays
   ``(beta/2)(q - w)^2``, ``beta = 0.1``: the habit follows where the reader is and pulls it
   back toward where it has been. From ``q = w = -1`` a spell of +0.6 for ``D`` steps, then
   nothing, ``T = 0.1``; a spell *sticks* if ``q`` averages positive over the last 500 of 3000
   steps after it. At ``D = 200``, the sticking fraction falls monotonically as the habit's time
   ``tau_w = 1/(eta_w beta)`` grows through 25, 100, 250, 1000, 2500: above 0.8 at 25, below 0.2
   at 2500, crossing 1/2 between 100 and 1000 (the fall-back time with the habit unmoved is
   estimated at about 250). At ``tau_w = 1000``, sticking rises with ``D`` through 200, 1000,
   3000 and exceeds 0.8 at 3000: a spell held long enough becomes a habit. **Could fail**: the
   crossing, the monotonicity.
4. *A note no member holds, and what it leaves.* 2000 members, the surrounds on for ``D_pop``
   steps then off; a slower committed reader ``R2`` (``eta_2 = 0.01``, quiet), starting left,
   receives the members' mean ``q``. (a) With ``D_pop = 6000``, members flip more than once each
   on average while the mean holds within 0.05 over the note's last half. (b) ``R2`` crosses right
   during it and is still right 6000 steps after the surrounds go, when the members' mean is back
   near 0. With ``D_pop = 1000`` it does not cross; the threshold between is reported. (c) Driven
   instead by one member's ``q``, ``R2`` switches at least twice in 20000 steps, while driven by
   the mean it never switches back. (a) and (b) are near-certain given the numbers; (c) **could
   fail** if one member's stays are too short for ``R2`` to follow.

**Composed since P43 (b) job 7** (Claude Opus 5.5, de71554a, 2026-10-10; David: "division is a box"). Test 4's
runs are ``Phi`` of ``population.population``: the members as boxes (the ``N``-fold tensor power of one noisy
member, checked equal to ``tensor_arrangements`` of copies), the surrounds' note copied to them through the
population's port, a ``+`` box and a ``1/N`` box for the mean, and ``R2`` reading the mean one way. Until
then the members were averaged in numpy and ``R2`` was a hand loop fed the mean, never checked against
``Phi``; those loops are kept (``members_mean``, ``slow_reader``) and checked equal on replayed noise, with
one difference that is the composition's: ``R2`` hears the mean the members present at a step, so it reads
it one step later than the hand loop, which fed it the members' new mean the same step. The noise of the
composed runs is the library's (``rand.key_sampler``), not numpy's, so their numbers differ by sampling.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements import population as pop
from arrangements.carving import note_step
from arrangements.rand import euler_maruyama, key_sampler, replay_sampler, seed
from arrangements.rvect import diagonal

from tests.chain import IN_POS
from tests.test_sustained_note import ETA, dk_com, fast, gibbs_right, k_com

S, T, SPELL, TP = 0.1, 0.1, 0.6, 200
BETA = 0.1


def landscape(u):
    r = np.sort(np.real([x for x in np.roots([1, 0, -1, -u]) if abs(x.imag) < 1e-9]))
    curv = lambda q: 3 * q ** 2 - 1
    return r, [k_com(x, u) for x in r], [curv(x) for x in r]


def kramers_from(well, u, temp):
    (wl, top, wr), (kl, kt, kr), (cl, ct, cr) = landscape(u)
    kw, cw = (kl, cl) if well == "L" else (kr, cr)
    return ETA * np.sqrt(cw * abs(ct)) / (2 * np.pi) * np.exp(-(kt - kw) / temp)


# ---------------------------------------------------------------------------
# 1. Less aligned, and back.
# ---------------------------------------------------------------------------


def test_a_spell_against_the_surrounds_wears_off():
    """Prediction 1, first half. Could fail."""
    N, n = 4000, 8000
    g = gibbs_right(S, T)
    settled = np.where(np.random.default_rng(10).random(N) < g, 1.0, -1.0)
    z = np.random.default_rng(11).standard_normal((TP + n, N))
    q = fast(dk_com, settled, [S - SPELL] * TP + [S] * n, T=T, z=z)
    f = (q[TP:] > 0).mean(axis=1)
    d = g - f
    idx = np.nonzero(d > 0.08)[0]
    idx = idx[: np.argmax(np.diff(idx) > 1) + 1] if np.any(np.diff(idx) > 1) else idx
    lam = -np.polyfit(idx, np.log(d[idx]), 1)[0]
    lam_k = kramers_from("L", S, T) + kramers_from("R", S, T)
    print(f"\n[1, against] right just after {f[0]:.3f}, end {f[-1]:.3f} (Gibbs {g:.3f}); "
          f"rate {lam:.3e}, Kramers {lam_k:.3e}, ratio {lam / lam_k:.2f}")
    assert f[0] < 0.05 and abs(f[-1] - g) < 0.05 and 0.5 < lam / lam_k < 2.0


def test_a_reader_stays_longer_in_the_lower_rest():
    """Prediction 1, second half. Could fail. Mean stay from a well's floor to the other
    well's floor: time at risk over exits, readers still in at the horizon counted at risk."""
    N, n = 2000, 40000
    (wl, top, wr), _, _ = landscape(S)
    c = np.sqrt(2.0 * T * ETA)
    stay = {}
    for well, q0, sd in (("L", wl, 12), ("R", wr, 13)):
        rng = np.random.default_rng(sd)
        q, t_out, alive = np.full(N, q0), np.full(N, n), np.ones(N, bool)
        for t in range(n):
            q = q - ETA * dk_com(q, S) + c * rng.standard_normal(N)
            hit = alive & ((q >= wr) if well == "L" else (q <= wl))
            t_out[hit] = t + 1
            alive &= ~hit
        stay[well] = t_out.sum() / max(1, (~alive).sum())
    ratio = stay["R"] / stay["L"]
    pred = kramers_from("L", S, T) / kramers_from("R", S, T)
    print(f"\n[1, stays] higher (left) {stay['L']:.0f} steps, lower (right) {stay['R']:.0f}; "
          f"ratio {ratio:.2f}, Kramers {pred:.2f}")
    assert 0.5 < ratio / pred < 2.0


# ---------------------------------------------------------------------------
# 3. The habit's race.
# ---------------------------------------------------------------------------


def habit_cost(x, u):
    q, w = x[0], x[1]
    return k_com(q, u) + 0.5 * BETA * (q - w) ** 2


def habit_reader(eta_w, x0, temp=None):
    Q = diagonal(jnp.array([ETA, eta_w]))
    init = jnp.asarray(x0, dtype=float)
    P = configuration(Q, init=init) if temp is None else euler_maruyama(Q, temp, init=init)
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda x, m_out: x[:1],
        in_f=lambda x, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda x, m_out, n_in: -habit_cost(x, n_in[0])),
        label="habit reader",
    )


def habit_fast(eta_w, q0, w0, notes, temp, rng):
    q, w = np.array(q0, float), np.array(w0, float)
    cq, cw = np.sqrt(2 * temp * ETA), np.sqrt(2 * temp * eta_w)
    qs = []
    for u in notes:
        z = rng.standard_normal((2,) + q.shape) if callable(getattr(rng, "standard_normal", None)) else next(rng)
        dq = dk_com(q, u) + BETA * (q - w)
        dw = BETA * (w - q)
        q, w = q - ETA * dq + cq * z[0], w - eta_w * dw + cw * z[1]
        qs.append(q.copy())
    return np.array(qs), w


def test_the_habit_step_is_Phi():
    """Machinery: the two-coordinate vectorized step is ``Phi`` of the habit reader, on
    replayed normal samples, to 1e-12."""
    notes = [SPELL] * 20 + [0.0] * 20
    z = np.random.default_rng(30).standard_normal((40, 2))
    arr = seed(habit_reader(0.04, [-1.0, -1.0], temp=T), replay_sampler(z))
    O, s, zero = Phi(arr), None, jnp.zeros(1)
    s, a = O.state, []
    for u in notes:
        _, _, s = O.with_state(s).run_one(IN_POS, lambda _o, u=u: (zero, jnp.array([u])))
        a.append(float(arr.P.position(s)[0]))
    b, _ = habit_fast(0.04, [-1.0], [-1.0], notes, T, iter(z[:, :, None]))
    assert np.max(np.abs(np.array(a) - b[:, 0])) < 1e-12


def stick_fraction(eta_w, D, N=2000, H=3000, sd=0):
    rng = np.random.default_rng(sd)
    qs, _ = habit_fast(eta_w, -np.ones(N), -np.ones(N), [SPELL] * D + [0.0] * H, T, rng)
    return (qs[-500:].mean(axis=0) > 0).mean()


@pytest.mark.xfail(strict=True, reason="prediction 3 refuted as designed, 2026-09-26; see docstring")
def test_the_habit_races_the_fall_back():
    """Prediction 3. Could fail, and **did** (2026-09-26). At ``D = 200`` the sticking fraction
    was 0.512, 0.520, 0.501, 0.463, 0.358 for ``tau_w`` = 25 ... 2500: about a half until the
    slowest habit, which pulled readers back. At ``tau_w = 1000`` it rose with ``D`` only to
    0.478, 0.521, 0.610. Kept failing on purpose (strict xfail) so the refutation stays on the
    record. Diagnosis, tested below as prediction 3': the horizon, 3000 steps, is longer than
    the time in which noise re-randomizes the reader *and* its habit together at ``T = 0.1``, so
    the measurement saw the joint equilibrium, one half, not the race."""
    taus = (25, 100, 250, 1000, 2500)
    fr = [stick_fraction(1.0 / (tau * BETA), 200, sd=40 + i) for i, tau in enumerate(taus)]
    print("\n[3, D = 200] tau_w: sticking  " + ", ".join(f"{t}: {f:.3f}" for t, f in zip(taus, fr)))
    Ds = (200, 1000, 3000)
    fd = [stick_fraction(1.0 / (1000 * BETA), D, sd=50 + i) for i, D in enumerate(Ds)]
    print("[3, tau_w = 1000] D: sticking  " + ", ".join(f"{D}: {f:.3f}" for D, f in zip(Ds, fd)))
    assert all(a >= b - 0.02 for a, b in zip(fr, fr[1:]))
    assert fr[0] > 0.8 and fr[-1] < 0.2
    half = [t for t, a, b in zip(taus, fr, fr[1:]) if a >= 0.5 > b]
    assert half and 100 <= half[0] <= 1000
    assert all(a <= b + 0.02 for a, b in zip(fd, fd[1:])) and fd[-1] > 0.8


# ---------------------------------------------------------------------------
# 4. A note no member holds, and what it leaves.
# ---------------------------------------------------------------------------

ETA2 = 0.01


def members_mean(D_pop, n_after, N=2000, sd=0):
    rng = np.random.default_rng(sd)
    q = -np.ones(N)
    c = np.sqrt(2 * T * ETA)
    means, flips, side = [], np.zeros(N), q > 0
    for t in range(D_pop + n_after):
        u = S if t < D_pop else 0.0
        q = q - ETA * dk_com(q, u) + c * rng.standard_normal(N)
        # a flip is a floor-to-floor passage, counted by the side of the barrier with a margin
        now = np.where(q > 0.5, True, np.where(q < -0.5, False, side))
        flips += now != side
        side = now
        means.append(q.mean())
    return np.array(means), flips


def slow_reader(notes, q0=-1.0):
    q, out = q0, []
    for u in notes:
        q = q - ETA2 * (q ** 3 - q - u)
        out.append(q)
    return np.array(out)


def composed_run(D_pop, n_after, N=2000, sd=0):
    """``Phi`` of the population (members, mean, ``R2``) under the surrounds for ``D_pop`` steps, then
    none: the members' mean after each step, the flips per member, and ``R2`` after each step."""
    import jax

    A = seed(pop.population(N, ETA, T, ETA2), key_sampler(sd))
    step = note_step(A)
    notes = jnp.asarray([S] * D_pop + [0.0] * n_after)

    def body(carry, u):
        s, side, flips = carry
        s = step(s, u)
        pos = A.P.position(s)
        q = pos[:N]
        now = jnp.where(q > 0.5, True, jnp.where(q < -0.5, False, side))
        return (s, now, flips + (now != side)), (jnp.mean(q), pos[N])

    (_, _, flips), (means, r2) = jax.lax.scan(body, (A.P.init, jnp.zeros(N, bool), jnp.zeros(N)), notes)
    return np.asarray(means), np.asarray(flips), np.asarray(r2)


def test_the_population_is_the_hand_loops():
    """The composed population equals the hand loops on replayed noise: members' mean to 1e-12 over
    300 steps at N = 2000, ``R2`` exactly once it is fed the mean one step earlier (the composition's
    timing); and the power of the member equals the tensor of four members. The prediction."""
    from arrangements.wiring import tensor_arrangements

    z = np.random.default_rng(0).standard_normal((50, 4))
    notes = [S] * 30 + [0.0] * 20
    A = seed(pop.population(4, ETA, T, ETA2), replay_sampler(z))
    B = seed(pop.population(4, ETA, T, ETA2,
                            members_arr=tensor_arrangements([pop.member(ETA, T)] * 4)), replay_sampler(z))
    sa, sb, a, b = note_step(A), note_step(B), A.P.init, B.P.init
    for u in notes:
        a, b = sa(a, u), sb(b, u)
    assert np.max(np.abs(np.asarray(A.P.position(a)) - np.asarray(B.P.position(b)))) < 1e-14

    N, n = 2000, 300
    Z = np.random.default_rng(60).standard_normal((n, N))
    A = seed(pop.population(N, ETA, T, ETA2), replay_sampler(Z))
    step, s, means, r2 = note_step(A), A.P.init, [], []
    for _ in range(n):
        s = step(s, S)
        pos = np.asarray(A.P.position(s))
        means.append(pos[:N].mean())
        r2.append(pos[N])
    m_old, _ = members_mean(n, 0, sd=60)
    assert np.max(np.abs(np.array(means) - m_old)) < 1e-12
    assert np.max(np.abs(np.array(r2) - slow_reader(np.concatenate([[-1.0], m_old[:-1]])))) < 1e-12


def test_a_note_no_member_holds_leaves_a_mark():
    """Prediction 4 (a), (b)."""
    D_pop, after = 6000, 6000
    m, flips, r2 = composed_run(D_pop, after, sd=60)
    late = m[D_pop // 2: D_pop]
    print(f"\n[4, D_pop = 6000] flips per member {flips.mean():.2f}; members' mean over the note's "
          f"last half {late.mean():.3f} +- {late.std():.3f}, at the end {m[-1]:.3f}; "
          f"R2 during {r2[D_pop - 1]:.3f}, at the end {r2[-1]:.3f}")
    assert flips.mean() > 1.0 and late.max() - late.min() < 0.1
    assert r2[D_pop - 1] > 0.9 and r2[-1] > 0.9 and abs(m[-1]) < 0.3
    thresholds = []
    for D in (1200, 1500, 2000, 3000, 4000):
        _, _, r = composed_run(D, 3000, sd=61)
        thresholds.append((D, r[-1] > 0))
    print("[4, threshold] D_pop: R2 right at the end  " + ", ".join(f"{D}: {r}" for D, r in thresholds))
    # Prediction 4(b)'s second half **failed** (2026-09-26): with D_pop = 1000, R2 did cross;
    # the members' mean rises faster than the agent estimated (e-fold about 400 steps, not 1000).
    # At 1000 itself the outcome is the draw's (P43 (b) job 7, de71554a, 2026-10-10, ten seeds each
    # way at these settings): R2 crossed in 4 of 10 composed runs and 6 of 10 hand-loop runs, never
    # at 800 or 900, always at 1200 and 1500. The 2026-09-26 run's one draw at 1000 crossed; the
    # composed run's draw (seed 61) does not. So the threshold is near 1000, a coin flip there.
    assert all(r for _, r in thresholds)


def test_one_member_is_not_a_held_note():
    """Prediction 4 (c). Could fail."""
    n = 20000
    _, _, r_one = composed_run(n, 0, N=1, sd=70)       # R2 reading one member: a population of one
    _, _, r_mean = composed_run(n, 0, sd=71)
    sw = lambda r: int(np.sum(np.diff(np.where(r > 0.5, 1, np.where(r < -0.5, -1, 0))[np.abs(r) > 0.5]) != 0))
    first_right = int(np.argmax(r_mean > 0.5))
    back = bool(np.any(r_mean[first_right:] < -0.5))
    print(f"\n[4c] R2 switches driven by one member: {sw(r_one)}; driven by the mean: {sw(r_mean)} "
          f"(ever back left after crossing: {back})")
    assert sw(r_one) >= 2 and not back


# ---------------------------------------------------------------------------
# After the two failures (Claude, 2026-09-26), predictions written before these ran:
#
# 3'. (i) At T = 0.1 with a fast habit (tau_w = 25), readers started at q = w = +1 relax to half
#     right with an e-fold time under 1500 steps, so a 3000-step horizon sees equilibrium.
#     (ii) At T = 0.05, same beta, spell and horizon, the race shows: sticking above 0.8 at
#     tau_w = 25, below 0.2 at 2500, falling monotonically, crossing 1/2 between 100 and 2500
#     (the fall-back time with the habit unmoved estimated at about 650 steps). Could fail.
# 4'. Observed, not predicted: the shortest population note that leaves R2 right, among
#     D_pop in 100, 200, 300, 400, 600, 800.
# ---------------------------------------------------------------------------


def test_the_first_horizon_saw_the_joint_equilibrium():
    """Prediction 3'(i). Could fail."""
    N, n = 2000, 3000
    qs, _ = habit_fast(1.0 / (25 * BETA), np.ones(N), np.ones(N), [0.0] * n, T, np.random.default_rng(80))
    d = (qs > 0).mean(axis=1) - 0.5
    idx = np.nonzero(d > 0.05)[0]
    idx = idx[: np.argmax(np.diff(idx) > 1) + 1] if np.any(np.diff(idx) > 1) else idx
    efold = -1.0 / np.polyfit(idx, np.log(d[idx]), 1)[0]
    print(f"\n[3'(i)] joint e-fold at T = 0.1, tau_w = 25: {efold:.0f} steps; right at 3000: {d[-1] + 0.5:.3f}")
    assert efold < 1500


@pytest.mark.xfail(strict=True, reason="prediction 3'(ii) refuted on its numbers, 2026-09-26")
def test_the_race_at_lower_noise():
    """Prediction 3'(ii). Could fail, and **did** on its numbers (2026-09-26): sticking 0.837,
    0.856, 0.817, 0.685, 0.509 for tau_w = 25 ... 2500. The direction held, a faster habit keeps
    more spells (0.84 against 0.51), but the slowest habit still kept half, not under 0.2, and the
    crossing of 1/2 was not reached. Not tuned further: a third guess at the parameters would be
    fitting the run to the prediction."""
    taus = (25, 100, 250, 1000, 2500)
    fr = []
    for i, tau in enumerate(taus):
        rng = np.random.default_rng(90 + i)
        qs, _ = habit_fast(1.0 / (tau * BETA), -np.ones(2000), -np.ones(2000), [SPELL] * 200 + [0.0] * 3000, 0.05, rng)
        fr.append((qs[-500:].mean(axis=0) > 0).mean())
    print("\n[3'(ii), T = 0.05, D = 200] tau_w: sticking  " + ", ".join(f"{t}: {f:.3f}" for t, f in zip(taus, fr)))
    assert all(a >= b - 0.02 for a, b in zip(fr, fr[1:]))
    assert fr[0] > 0.8 and fr[-1] < 0.2
    half = [t for t, a, b in zip(taus, fr, fr[1:]) if a >= 0.5 > b]
    assert half and 100 <= half[0] <= 2500


def test_the_shortest_population_note_that_leaves_a_mark():
    """4', observed (2026-09-26): no D_pop from 100 to 800 left R2 right (the members' mean
    peaked at 0.45 by 800). The threshold is near 1000 steps, where the members' mean has passed
    R2's tilt threshold 0.385 long enough for R2 to cross: at 1000 it crosses in about half the
    draws (4 of 10 composed, 6 of 10 by hand; P43 (b) job 7), at 1200 in every one."""
    rows = []
    for D in (100, 200, 300, 400, 600, 800):
        mm, _, r = composed_run(D, 3000, sd=62)
        rows.append((D, bool(r[-1] > 0), float(mm[:D].max())))
    print("\n[4'] D_pop: R2 right at the end, members' peak mean  "
          + ", ".join(f"{D}: {r} ({pk:.2f})" for D, r, pk in rows))
