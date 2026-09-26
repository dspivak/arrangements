"""A passing note moves a reader; a sustained note moves where it comes to rest.

The claim is the book's (plan §4.1 step 11; David, 2026-09-26, "i love it!"), and this file
is its first run. **Every docstring below says whether its test could have come out otherwise**
(STATUS.md §2, the last warning); the ones that could not are closed-form checks of the
machinery, and only the others are results.

The reader is one component ``I -> <R | R>``: configuration ``q in R`` with the Euclidean
response ``eta``, sending ``q`` and receiving the note ``u`` on its exterior input. Its cost is
one of two, both linear in ``u`` so that ``u = 0`` is the note's absence:

* the *uncommitted* reader, ``k(q; u) = (a/2) q^2 - u q``, one rest point, at ``u / a``;
* the *committed* reader, ``k(q; u) = (1/4)(q^2 - 1)^2 - u q``, two wells at ``q = +-1`` when
  ``u = 0``, barrier ``Delta V = 1/4`` at ``q = 0``; for ``|u| > 2/(3 sqrt 3)`` one well vanishes.

The note's source is left out of the picture: its schedule is supplied from outside, step by
step, as the exterior input (a partial picture, plan §9.1 *arrangement*). A note is *held* when
it stays on, *passing* when it is switched off after ``T_p`` steps. Noise is
``euler_maruyama`` (STATUS.md §3.12), whose chain is not the Gibbs law; the statistics below use
a vectorized copy of the step, and ``test_the_fast_step_is_Phi`` checks pathwise, on replayed
normal samples, that the copy is the ``Phi`` of the seeded arrangement.

**Predictions, written before any of this ran (Claude, 2026-09-26):**

1. Uncommitted, no noise: a held note moves the rest to ``u/a`` and the reader goes there at
   rate ``1 - eta a`` per step; a passing note moves the reader and, once off, it returns to 0 at
   the same rate. Closed form; cannot come out otherwise.
2. Committed, no noise: a passing note above threshold, on long enough to carry the reader past
   ``q = 0``, leaves it in the other well for good (the attention paper's cue and commitment,
   plan §5.4 Example 3); on for less, the reader returns to its own well. A held note below
   threshold only shifts the well the reader is in, and it never crosses. Guaranteed in one
   dimension; stated so the noisy runs have their baseline.
3. Committed, with noise (``T = 0.1``), a held note below threshold (``u = 0.1``): the fraction of
   readers in the right well goes from 0 to the two-well Gibbs value, within 0.05, and stays there
   while the note is held. **Could fail**: the chain's bias, or a slow approach. The point for the
   book: a note too weak to move a quiet reader moves where a noisy one rests.
4. The same, released after it has settled: the fraction relaxes to 1/2 as
   ``1/2 + c exp(-2 r t)``, with ``r`` Kramers's escape rate per step,
   ``eta sqrt(2)/(2 pi) exp(-Delta V / T)``, within a factor 2. **Could fail**, on the prefactor.
5. Over ``T in {1/16, 1/12, 1/10, 1/8}``, after a passing note above threshold flips every
   reader: ``log r`` against ``1/T`` has slope ``-Delta V = -1/4`` within 20%. **Could fail**. The
   point for the book: what a passing note leaves lasts as long as the reader's own commitment
   holds it against wear, and that time grows exponentially with the commitment's depth.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rand import euler_maruyama, replay_sampler, seed
from arrangements.rvect import euclidean

from tests.chain import IN_POS

ETA = 0.05            # the response; one covector, one displacement
A_UNC = 1.0           # the uncommitted reader's stiffness
DV = 0.25             # the committed reader's barrier at u = 0
U_C = 2.0 / (3.0 * np.sqrt(3.0))   # the tilt at which one well vanishes


def k_unc(q, u):
    return 0.5 * A_UNC * q ** 2 - u * q


def k_com(q, u):
    return 0.25 * (q ** 2 - 1.0) ** 2 - u * q


def dk_unc(q, u):
    return A_UNC * q - u


def dk_com(q, u):
    return q ** 3 - q - u


def reader(cost, q0, T=None):
    """The reader component ``I -> <R | R>``: it sends ``q`` and pays ``cost(q, u)`` on the
    note ``u`` it receives. With ``T`` its integrator is Euler--Maruyama at temperature ``T``."""
    Q = euclidean(1, ETA)
    init = jnp.array([float(q0)])
    P = configuration(Q, init=init) if T is None else euler_maruyama(Q, T, init=init)
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(q[0], n_in[0])),
        label="reader",
    )


def run_phi(arr, notes):
    """``Phi(arr)`` run on the note schedule ``notes`` (one exterior input per step), with a
    zero exterior covector; returns the positions."""
    O = Phi(arr)
    s, zero_xi, out = O.state, jnp.zeros(1), []
    for u in notes:
        _, _, s = O.with_state(s).run_one(IN_POS, lambda _o, u=u: (zero_xi, jnp.array([u])))
        out.append(float(arr.P.position(s)[0]))
    return np.array(out)


def fast(dk, q0, notes, T=0.0, z=None):
    """The same step, vectorized over readers: ``q' = q - eta dk(q, u) + sqrt(2 T eta) z``."""
    q = np.array(q0, dtype=float)
    c = np.sqrt(2.0 * T * ETA)
    out = []
    for t, u in enumerate(notes):
        q = q - ETA * dk(q, u) + (c * z[t] if T > 0 else 0.0)
        out.append(q.copy())
    return np.array(out)


def gibbs_right(u, T):
    """The fraction of ``exp(-k_com / T)`` on ``q > 0``."""
    q = np.linspace(-3.0, 3.0, 60001)
    w = np.exp(-(k_com(q, u) - k_com(q, u).min()) / T)
    return w[q > 0].sum() / w.sum()


def kramers(T):
    """Kramers's escape rate per step over the barrier at ``u = 0``."""
    return ETA * np.sqrt(2.0 * 1.0) / (2.0 * np.pi) * np.exp(-DV / T)


def relaxation_rate(frac, floor=0.06):
    """Fit ``log(frac - 1/2)`` linearly while ``frac - 1/2 > floor``; returns ``r`` with the
    relaxation ``exp(-2 r t)``."""
    d = frac - 0.5
    idx = np.nonzero(d > floor)[0]
    end = idx[np.nonzero(np.diff(idx) > 1)[0][0]] if np.any(np.diff(idx) > 1) else idx[-1]
    t = np.arange(end + 1)
    slope = np.polyfit(t, np.log(d[: end + 1]), 1)[0]
    return -slope / 2.0


# ---------------------------------------------------------------------------


def test_the_fast_step_is_Phi():
    """Machinery. The vectorized step is ``Phi`` of the reader, without noise and, on replayed
    normal samples, with it: pathwise agreement on 60 steps of a note switched off halfway."""
    notes = [0.6] * 30 + [0.0] * 30
    for cost, dk in ((k_unc, dk_unc), (k_com, dk_com)):
        a = run_phi(reader(cost, -1.0), notes)
        b = fast(dk, [-1.0], notes)[:, 0]
        assert np.max(np.abs(a - b)) < 1e-12
    z = np.random.default_rng(0).standard_normal((60, 1))
    arr = seed(reader(k_com, -1.0, T=0.1), replay_sampler(z))
    a = run_phi(arr, notes)
    b = fast(dk_com, [-1.0], notes, T=0.1, z=z)[:, 0]
    assert np.max(np.abs(a - b)) < 1e-12


def test_uncommitted_reader_held_and_passing():
    """Prediction 1, closed form: cannot come out otherwise."""
    A, n, Tp = 0.5, 200, 40
    c = 1.0 - ETA * A_UNC
    held = fast(dk_unc, [0.0], [A] * n)[:, 0]
    t = np.arange(1, n + 1)
    assert np.allclose(held, (A / A_UNC) * (1.0 - c ** t), atol=1e-12)
    passing = fast(dk_unc, [0.0], [A] * Tp + [0.0] * (n - Tp))[:, 0]
    assert np.allclose(passing[Tp:], passing[Tp - 1] * c ** np.arange(1, n - Tp + 1), atol=1e-12)
    assert abs(passing[-1]) < 1e-3 and abs(held[-1] - A / A_UNC) < 1e-3


def test_committed_reader_quiet():
    """Prediction 2: guaranteed in one dimension; the baseline for the noisy runs."""
    n = 2000
    held = fast(dk_com, [-1.0], [0.6] * n)[:, 0]
    root = np.real([r for r in np.roots([1, 0, -1, -0.6]) if abs(r.imag) < 1e-9][0])
    assert abs(held[-1] - root) < 1e-9 and root > 1.0
    crossing = int(np.argmax(held > 0.0)) + 1          # steps the note must stay on
    long = fast(dk_com, [-1.0], [0.6] * crossing + [0.0] * n)[:, 0]
    short = fast(dk_com, [-1.0], [0.6] * (crossing - 1) + [0.0] * n)[:, 0]
    assert abs(long[-1] - 1.0) < 1e-6 and abs(short[-1] + 1.0) < 1e-6
    weak = fast(dk_com, [-1.0], [0.1] * n)[:, 0]
    assert np.all(weak < 0.0)
    print(f"\n[quiet] held 0.6 rests at {root:.4f}; passing 0.6 must stay on {crossing} steps "
          f"(settling e-fold ~ {1 / (2 * ETA):.0f}); held 0.1 rests at {weak[-1]:.4f}")


def test_weak_held_note_moves_a_noisy_reader():
    """Prediction 3. Could fail."""
    T, u, N, n = 0.1, 0.1, 4000, 6000
    rng = np.random.default_rng(1)
    z = rng.standard_normal((n, N))
    q = fast(dk_com, -np.ones(N), [u] * n, T=T, z=z)
    frac = (q > 0).mean(axis=1)
    g = gibbs_right(u, T)
    late = frac[n // 2:].mean()
    print(f"\n[held 0.1, T=0.1] right fraction: quiet 0, noisy {late:.3f}, Gibbs {g:.3f}")
    assert abs(late - g) < 0.05


def test_released_note_fades_at_kramers_rate():
    """Prediction 4. Could fail, on the prefactor."""
    T, u, N, n_on, n_off = 0.1, 0.1, 4000, 6000, 6000
    rng = np.random.default_rng(2)
    z = rng.standard_normal((n_on + n_off, N))
    q = fast(dk_com, -np.ones(N), [u] * n_on + [0.0] * n_off, T=T, z=z)
    frac = (q[n_on:] > 0).mean(axis=1)
    r = relaxation_rate(frac, floor=0.04)
    print(f"\n[released 0.1, T=0.1] fraction at release {frac[0]:.3f}, end {frac[-1]:.3f}; "
          f"rate {r:.3e}, Kramers {kramers(T):.3e}, ratio {r / kramers(T):.2f}")
    assert 0.5 < r / kramers(T) < 2.0


def test_passing_mark_lasts_exponentially_in_the_commitment():
    """Prediction 5. Could fail.

    First run (2026-09-26) had ``Tp = 60``, written before the quiet crossing time was known;
    that is four steps short of the 64 a quiet reader needs, so only 60--65% of the noisy
    readers flipped and the prediction's premise ("flips every reader") was not met. The slope
    came out -0.2605 even so, from a short fitting window. A second run with ``Tp = 100``
    still left 12% of the readers on the left at ``T = 1/16`` (a noisy reader lingers where the
    vanished well was); rerun here with ``Tp = 200``. The short note is kept as its own,
    observed test below."""
    N, Tp = 4000, 200
    rows = []
    for T, n in ((1 / 16, 12000), (1 / 12, 6000), (1 / 10, 4000), (1 / 8, 3000)):
        rng = np.random.default_rng(int(1 / T))
        z = rng.standard_normal((Tp + n, N))
        q = fast(dk_com, -np.ones(N), [0.6] * Tp + [0.0] * n, T=T, z=z)
        frac = (q[Tp:] > 0).mean(axis=1)
        assert frac[0] > 0.95
        r = relaxation_rate(frac)
        rows.append((T, frac[0], r, kramers(T)))
    inv_T = np.array([1 / T for T, *_ in rows])
    slope = np.polyfit(inv_T, np.log([r for _, _, r, _ in rows]), 1)[0]
    print("\n[passing 0.6] T, right fraction after the note, rate, Kramers, ratio")
    for T, f0, r, kr in rows:
        print(f"  {T:.4f}  {f0:.3f}  {r:.3e}  {kr:.3e}  {r / kr:.2f}")
    print(f"  slope of log r against 1/T: {slope:.4f} (prediction -{DV})")
    assert abs(slope + DV) < 0.2 * DV


def test_a_note_too_short_for_a_quiet_reader_moves_some_noisy_ones():
    """**Observed, not predicted** (the first run of prediction 5, 2026-09-26). A note of 0.6
    held four steps short of the quiet crossing time moves no quiet reader and moves a
    majority of noisy ones: what counts as held long enough is relative to the reader's noise
    as well as to its settling time."""
    quiet = fast(dk_com, [-1.0], [0.6] * 64 + [0.0] * 2000)[:, 0]
    crossing = int(np.argmax(fast(dk_com, [-1.0], [0.6] * 200)[:, 0] > 0.0)) + 1
    short = fast(dk_com, [-1.0], [0.6] * (crossing - 4) + [0.0] * 2000)[:, 0]
    assert crossing == 64 and quiet[-1] > 0.9 and short[-1] < -0.9
    N = 4000
    rows = []
    for T in (1 / 16, 1 / 12, 1 / 10, 1 / 8):
        z = np.random.default_rng(int(1 / T)).standard_normal((crossing - 4 + 1, N))
        q = fast(dk_com, -np.ones(N), [0.6] * (crossing - 4) + [0.0], T=T, z=z)
        rows.append((T, (q[-1] > 0).mean()))
    print("\n[note 60 steps, 4 short] T, right fraction just after: "
          + ", ".join(f"{T:.4f}: {f:.3f}" for T, f in rows))
    assert all(0.5 < f < 0.8 for _, f in rows)


# ---------------------------------------------------------------------------
# David's wizard (2026-09-26): "the strange ripples can move them out of the most likely
# settling place and into a whole new one, but of course it's still one they have. but for
# example, there may be pressure to get there because it's a lower equilibrium with their
# surrounds, and the wizard may know that, so he sings the strange ripples, gets you out of
# your stuck place, and into accordance with the rest."
#
# The surrounds are a weak note held throughout, S = 0.1, which makes the right well the lower
# one; the wizard is a passing note of +-0.6 for 200 steps; T = 0.1.
#
# Predictions, written before this section ran (Claude, 2026-09-26):
# 6a. Surrounds alone, readers starting in the left well: the right fraction rises toward the
#     Gibbs value at rate lambda = r_LR + r_RL, each r Kramers's for its own barrier, within a
#     factor 2. Could fail, on the prefactor.
# 6b. The wizard into accordance (+0.6): right fraction near 1 at once, relaxing only to the
#     equilibrium, so at least 80% of the spell's effect is still there at the end. Guaranteed
#     by stationarity once 6a holds; stated for the contrast.
# 6c. The wizard against it (-0.6), readers starting settled: right fraction near 0 at once,
#     then all of the effect fades, back to the equilibrium.
# 6d. A reader's mean time in a well before it reaches the other well's floor: in the lower well
#     over in the higher, about Kramers's ratio (barriers and curvatures), within a factor 2;
#     roughly 7 at these numbers. Could fail.
# ---------------------------------------------------------------------------

S_SUR, T_W, U_W, TP_W = 0.1, 0.1, 0.6, 200


def landscape(u):
    """The two wells and the barrier of ``k_com(., u)``: positions, costs, curvatures."""
    r = np.sort(np.real([x for x in np.roots([1, 0, -1, -u]) if abs(x.imag) < 1e-9]))
    wl, top, wr = r
    curv = lambda q: 3 * q ** 2 - 1
    return (wl, top, wr), (k_com(wl, u), k_com(top, u), k_com(wr, u)), (curv(wl), curv(top), curv(wr))


def kramers_from(well, u, T):
    """Kramers's rate per step out of ``well`` ('L' or 'R') of ``k_com(., u)``."""
    (wl, top, wr), (kl, kt, kr), (cl, ct, cr) = landscape(u)
    kw, cw = (kl, cl) if well == "L" else (kr, cr)
    return ETA * np.sqrt(cw * abs(ct)) / (2 * np.pi) * np.exp(-(kt - kw) / T)


def test_the_surrounds_alone_move_readers_slowly():
    """Prediction 6a. Could fail."""
    N, n = 4000, 8000
    z = np.random.default_rng(61).standard_normal((n, N))
    q = fast(dk_com, -np.ones(N), [S_SUR] * n, T=T_W, z=z)
    frac = (q > 0).mean(axis=1)
    g = gibbs_right(S_SUR, T_W)
    lam_pred = kramers_from("L", S_SUR, T_W) + kramers_from("R", S_SUR, T_W)
    d = g - frac
    idx = np.nonzero(d > 0.08)[0]
    idx = idx[: np.argmax(np.diff(idx) > 1) + 1] if np.any(np.diff(idx) > 1) else idx
    lam = -np.polyfit(idx, np.log(d[idx]), 1)[0]
    half = int(np.argmax(frac > g / 2))
    print(f"\n[surrounds alone] rate {lam:.3e}, Kramers {lam_pred:.3e}, ratio {lam / lam_pred:.2f}; "
          f"half-way after {half} steps; end {frac[-1]:.3f} (Gibbs {g:.3f})")
    assert 0.5 < lam / lam_pred < 2.0


def test_the_wizard_with_and_against_the_surrounds():
    """Predictions 6b and 6c."""
    N, n = 4000, 8000
    g = gibbs_right(S_SUR, T_W)
    z = np.random.default_rng(62).standard_normal((TP_W + n, N))
    with_ = fast(dk_com, -np.ones(N), [S_SUR + U_W] * TP_W + [S_SUR] * n, T=T_W, z=z)
    fw = (with_[TP_W:] > 0).mean(axis=1)
    settled = np.where(np.random.default_rng(63).random(N) < g, 1.0, -1.0)
    z = np.random.default_rng(64).standard_normal((TP_W + n, N))
    against = fast(dk_com, settled, [S_SUR - U_W] * TP_W + [S_SUR] * n, T=T_W, z=z)
    fa = (against[TP_W:] > 0).mean(axis=1)
    kept_with = (fw[-1] - 0.0) / (fw[0] - 0.0)          # from all-left, the spell's effect is fw
    print(f"\n[wizard with]    right fraction just after {fw[0]:.3f}, end {fw[-1]:.3f}; "
          f"effect kept {kept_with:.2f}")
    print(f"[wizard against] right fraction just after {fa[0]:.3f}, end {fa[-1]:.3f} (equilibrium {g:.3f})")
    assert fw[0] > 0.95 and kept_with > 0.8
    assert fa[0] < 0.05 and abs(fa[-1] - g) < 0.05


def test_a_reader_stays_longer_in_the_lower_rest():
    """Prediction 6d. Could fail. Mean time from a well's floor to the other well's floor,
    estimated as time at risk over exits (readers not yet out by the horizon count as at risk)."""
    N, n = 2000, 40000
    (wl, top, wr), _, _ = landscape(S_SUR)
    out = {}
    for well, q0, seed in (("L", wl, 71), ("R", wr, 72)):
        rng = np.random.default_rng(seed)
        q = np.full(N, q0)
        exit_t = np.full(N, n)
        alive = np.ones(N, bool)
        c = np.sqrt(2.0 * T_W * ETA)
        for t in range(n):
            q = q - ETA * dk_com(q, S_SUR) + c * rng.standard_normal(N)
            hit = alive & ((q >= wr) if well == "L" else (q <= wl))
            exit_t[hit] = t + 1
            alive &= ~hit
        out[well] = exit_t.sum() / max(1, (~alive).sum())
    ratio = out["R"] / out["L"]
    pred = kramers_from("L", S_SUR, T_W) / kramers_from("R", S_SUR, T_W)
    print(f"\n[dwell] mean steps in the higher (left) well {out['L']:.0f}, in the lower (right) "
          f"{out['R']:.0f}; ratio {ratio:.2f}, Kramers {pred:.2f}")
    assert 0.5 < ratio / pred < 2.0
