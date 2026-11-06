"""A whole whose hearing is its own: the gated pair with its two links as one slow coordinate, kept up
by the parts' joint activity through the gate.

Book plan P73 (AGENTDOCPLAN.md §8.2), the run after `notes/runs/p73-structures-d5eec4fb/RESULTS.md`.
David, 2026-11-05 (d5eec4fb), asked whether in a self-sustaining whole the hearing is kept up by the
parts' own joint activity, fading when unused, or set from outside: "parts own joint activity"; then
"let's run the model". Everything not quoted from him is the agent's.

**The whole.** ``tests/test_mutual_need.py``'s gated pair (§5.9 item 12), with its two links
``beta = gamma`` no longer constants but one coordinate ``w`` of the *coupler's own integrator*: the
coupler is an arrangement ``<R^0|R^3> -> <R^0|R^0>`` whose parameter is the configuration coalgebra
on ``w`` with mobility ``h * ETA``, whose returned covector on the ports ``e = (q_A, q_B, s)`` reads
``w`` as the two loudnesses, and whose own effort on ``w`` grows through the gate and decays:

    F_A(e, w) = s g + w q_B,   F_B(e, w) = w q_A,   F_S = 0,
    xi_w(e, w) = g (1 - w) - LAM w,                 g = q_A q_B / (K + q_A q_B).

In the book's terms the coupler is an attention-like arrangement: its transfer maps are the scalar
loudnesses ``T_AB = T_BA = w`` read off its configuration (Ch 4 `def.attention_arrangement`), its effort
field is non-exact (the gate, item 12 test 5), and ``w`` is a slow part of the whole (Ch 1
`sec.held_form`: where the slow parts sit sets which forms the quick parts can settle into). The
composite forcing on ``Q = R^4``, configuration ``(q_A, q_B, s, w)``:

    xi_A = s g + w q_B - q_A,   xi_B = w q_A - q_B,   xi_S = 0,   xi_w = g (1 - w) - LAM w,

one step ``q + (h, h, 0, h ETA) xi`` (``alpha = b = K = 1``, ``s = 4``). The growth rule is a model
(Hebbian through the gate, bounded by 1), not a derivation: by hand, a linear Hebbian rule
``xi_w = q_A q_B - LAM w`` has no stable alive hearing, since joint activity grows with hearing faster
than the decay (the task file). The decay rate ``LAM`` is what the run varies.

**The sheets** (``notes/runs/p73-structures-d5eec4fb/gated_pair_fold.py``): at frozen ``w`` the quick
parts have the dead rest ``0``, a sink for every ``w < 1``, and for ``s >= 2 (1 - w^2)/sqrt(w)`` an
alive rest ``q_A^+(w)``, ``q_B = w q_A^+``, with a saddle between; at ``s = 4`` the alive sheet ends at the
fold ``w* = 0.2253``. The slow motion on the alive sheet is ``w' = ETA (g(w q_A^+(w)^2)(1 - w) - LAM w)``;
it has a stable rest for ``LAM`` below ``LAM_C = 2.024`` and none above (then the alive sheet drains
into the fold). P41 (g)'s theorem (§5.9 item 88; one slow variable) reads the whole's capacities off
the sheets: its attractors are the slow motion's rests on the sheets.

**Predictions, written before the arrangement was built** (`.scratch/p73-structures-d5eec4fb/hearing_kept_predictions.py`,
the reduced equations stepped by the same Euler rule; ``ETA = 0.05``, ``h = 0.01``):

  H0  ``Phi`` of the composite, ``run_closed`` and the ``fori_loop`` agree exactly over fifty steps.
  H1  ``LAM = 1``: from ``(4.9, 2.45, w = 0.5)`` the whole settles, in 600 time units, at the alive rest
      ``(q_A, q_B, w) = (4.7416, 2.2654, 0.4778)`` to ``1e-6``, the sheet's rest; from the low start
      ``(0.3, 0.3, w = 0.5)`` it settles at the dead rest, every amplitude and the hearing below ``1e-3``.
      Two attractors, alive with hearing kept up and dead with hearing gone.
  H2  ``LAM = 1``, the supply cut (``s = 0``) for a time ``T`` and restored: a cut of ``2.06`` revives
      to the alive rest; a cut of ``8.23`` kills, and after 600 units of restored supply the pair is dead
      and its hearing below ``1e-3``: death is not undone by the sun coming back. The threshold cut
      length is ``T* = 4.115`` to within ``0.05``.
  H3  ``LAM = 2.6 > LAM_C``: from the alive rest, with the supply on throughout, the whole dies within
      600 units and its hearing fades below ``1e-3``: death with the sun shining, by hearing decaying
      faster than use keeps it up.
  H4  (P41 (g)'s reading) at the whole's alive rest the quick parts sit exactly at the sheet's point over
      its hearing: ``q_A = q_A^+(w)`` and ``q_B = w q_A`` to ``1e-6``; and from four more starts at
      ``LAM = 1`` every run ends at one of the two rests.

Everything run here is one closed arrangement built by ``compose_seq(tensor(A, B, S), hearing)`` and
stepped by the dynamics functor (H0 licenses ``settle``); the reduced-equation numbers are the
predictions, not a second model. Reasonable or weird, by scale: `RESULTS.md` in the book repository.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

DISS = 1.0
KGATE = 1.0
SUPPLY = 4.0
MOB = 0.01          # h, each part's mobility
ETA = 0.05          # the hearing's mobility relative to the parts': its sharp is h ETA
LAM = 1.0           # the hearing's decay rate in the alive runs
LAM_HIGH = 2.6      # above LAM_C = 2.024: the alive sheet drains into the fold
HELD = 1e-3
EMPTY = jnp.zeros(0)
W_STAR = 0.22527042609891978

# the predictions' numbers (hearing_kept_predictions.py)
ALIVE_REST = np.array([4.74163411, 2.26536817, 0.47776107])
T_REVIVE, T_KILL, T_STAR = 2.058, 8.230, 4.115


def part(q0, b=DISS, h=MOB, label="part"):
    """A part as in test_mutual_need: amplitude ``q`` with mobility ``h``, published at its port,
    carrying its own dissipation."""
    return Arrangement(
        P=configuration(euclidean(1, h), init=jnp.array([float(q0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=exact(lambda q, m_out, n_in: -0.5 * b * q[0] ** 2),
        label=label,
    )


def supply(s=SUPPLY):
    """The drive: a box with zero response holding its value."""
    return Arrangement(
        P=configuration(constant(jnp.zeros((1, 1))), init=jnp.array([float(s)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label="S",
    )


def gate(q_A, q_B, K=KGATE):
    u = q_A * q_B
    return u / (K + u)


def hearing(w0, lam=LAM, eta=ETA, h=MOB, supply_on=True, label="hearing"):
    """The coupler with its own slow state ``w``, the two loudnesses: ``<R^0|R^3> -> <R^0|R^0>``.
    Its covector on the ports is the pair's wiring read at ``w``; its effort on ``w`` is the growth
    through the gate minus the decay."""

    def omega(q, e, n_in):
        w = q[0]
        q_A, q_B, s = e[0], e[1], e[2]
        s = s if supply_on else 0.0 * s
        g = gate(q_A, q_B)
        F = jnp.array([s * g + w * q_B, w * q_A, 0.0])
        xi_w = jnp.array([g * (1.0 - w) - lam * w])
        return (xi_w, F, EMPTY)

    return Arrangement(
        P=configuration(euclidean(1, h * eta), init=jnp.array([float(w0)])),
        in_dim_M=0, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: EMPTY,
        in_f=lambda q, e, n_in: EMPTY,
        omega=omega,
        label=label,
    )


def whole(q_A=4.9, q_B=2.45, w=0.5, s=SUPPLY, lam=LAM, eta=ETA, h=MOB, supply_on=True):
    """The closed arrangement ``compose_seq(tensor(A, B, S), hearing)``, configuration
    ``(q_A, q_B, s, w)`` in that order."""
    boxes = [part(q_A, h=h, label="A"), part(q_B, h=h, label="B"), supply(s)]
    return compose_seq(tensor_arrangements(boxes),
                       hearing(w, lam=lam, eta=eta, h=h, supply_on=supply_on))


def settle(arr, steps, init=None):
    """The final configuration after ``steps`` steps of the composite, inside a ``fori_loop``
    (H0 checks it agrees with ``Phi``). ``init`` overrides the arrangement's initial state."""
    P, omega = arr.P, arr.omega
    s0 = P.init if init is None else init

    def step(_i, st):
        xi_q, _, _ = omega(P.position(st), EMPTY, EMPTY)
        return P.step(st, xi_q)

    return np.asarray(flat_state(jax.lax.fori_loop(0, steps, step, s0)))


def steps_for(T, h=MOB):
    return int(round(T / h))


def qA_plus(w, s=SUPPLY):
    """The alive sheet at frozen hearing ``w`` (gated_pair_fold.py)."""
    Bq, c = 1.0 - w ** 2, w
    disc = (s * c) ** 2 - 4.0 * Bq * c * Bq
    if disc < 0.0:
        return float("nan")
    return (s * c + np.sqrt(disc)) / (2.0 * Bq * c)


def run_cut_then_restore(T_cut, T_after=600.0, lam=LAM):
    """From the alive rest: the supply cut for ``T_cut``, then restored for ``T_after``.
    The cut is the same whole with ``supply_on=False``, started at the alive rest; the restoration
    is the whole with the supply on, started where the cut left it."""
    cut = whole(*ALIVE_REST[:2], w=ALIVE_REST[2], lam=lam, supply_on=False)
    after_cut = settle(cut, steps_for(T_cut))
    on = whole(after_cut[0], after_cut[1], w=after_cut[3], lam=lam)
    return after_cut, settle(on, steps_for(T_after))


# ---------------------------------------------------------------------------
# H0: the runs are the dynamics functor's.
# ---------------------------------------------------------------------------


def test_H0_the_runs_are_phis():
    b = whole()
    print("label:", b.label)
    by_phi = run_phi(b, 50, EMPTY)
    by_closed = run_closed(b, 50, lambda t: EMPTY)
    assert float(np.max(np.abs(by_phi - by_closed))) == 0.0
    np.testing.assert_array_equal(settle(b, 50), by_closed[-1])


# ---------------------------------------------------------------------------
# H1 and H4: two attractors, alive with hearing kept up and dead with hearing gone.
# ---------------------------------------------------------------------------


def test_H1_H4_two_attractors_and_the_sheet():
    end = settle(whole(4.9, 2.45, w=0.5), steps_for(600.0))
    print("H1 alive rest:", end)
    np.testing.assert_allclose(end[[0, 1, 3]], ALIVE_REST, atol=1e-6)
    # H4: the quick parts sit at the sheet's point over the hearing
    w = end[3]
    np.testing.assert_allclose(end[0], qA_plus(w), atol=1e-6)
    np.testing.assert_allclose(end[1], w * end[0], atol=1e-6)
    assert w > W_STAR

    dead = settle(whole(0.3, 0.3, w=0.5), steps_for(600.0))
    print("H1 dead rest:", dead)
    assert max(abs(dead[0]), abs(dead[1]), abs(dead[3])) < HELD

    # H4: four more starts, each ends at one of the two rests
    for q_A, q_B, w0 in ((1.0, 1.0, 0.5), (3.0, 0.5, 0.3), (0.5, 3.0, 0.9), (6.0, 6.0, 0.25)):
        e = settle(whole(q_A, q_B, w=w0), steps_for(600.0))
        at_alive = np.allclose(e[[0, 1, 3]], ALIVE_REST, atol=1e-4)
        at_dead = max(abs(e[0]), abs(e[1]), abs(e[3])) < HELD
        print(f"H4 start ({q_A}, {q_B}, w={w0}) -> {e}  alive={at_alive} dead={at_dead}")
        assert at_alive or at_dead


# ---------------------------------------------------------------------------
# H2: the supply cut; death is not undone by the sun coming back.
# ---------------------------------------------------------------------------


def test_H2_a_cut_kills_past_a_threshold_and_the_sun_does_not_revive():
    after, back = run_cut_then_restore(T_REVIVE)
    print("H2 short cut: after cut", after, "after restoring", back)
    np.testing.assert_allclose(back[[0, 1, 3]], ALIVE_REST, atol=1e-6)

    after, back = run_cut_then_restore(T_KILL)
    print("H2 long cut: after cut", after, "after restoring", back)
    assert max(abs(back[0]), abs(back[1])) < HELD
    assert abs(back[3]) < HELD      # the hearing has faded too

    lo, hi = 2.0, 8.0
    for _ in range(8):
        mid = 0.5 * (lo + hi)
        _, back = run_cut_then_restore(mid, T_after=300.0)
        if abs(back[0]) > HELD:
            lo = mid
        else:
            hi = mid
    t_star = 0.5 * (lo + hi)
    print("H2 threshold cut length:", t_star, "(predicted", T_STAR, ")")
    assert abs(t_star - T_STAR) < 0.05


# ---------------------------------------------------------------------------
# H3: death with the sun shining, when hearing decays faster than use keeps it up.
# ---------------------------------------------------------------------------


def test_H3_death_with_the_supply_on_above_lam_c():
    end = settle(whole(*ALIVE_REST[:2], w=ALIVE_REST[2], lam=LAM_HIGH), steps_for(600.0))
    print("H3 end:", end)
    assert max(abs(end[0]), abs(end[1])) < HELD
    assert abs(end[3]) < HELD
