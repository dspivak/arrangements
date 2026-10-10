"""Many parts that keep one another whole: a ring and an all-to-all of parts in mutual need.

The book plan's P30 stage (e) (AGENTDOCPLAN.md §8.2 in the book repository; David,
2026-09-29, on §1.7's footnote "larger collectives are still being worked out": "then we
should do it now. we are at the point where we need to know"). The baseline is
``tests/test_mutual_need.py`` (§5.9 item 12): two parts, each a driven steady state whose
held form is supplied by the other through a gate, one supply entering from outside.

**The predictions, fixed before any construction code in
``notes/MANY-PARTS-PREDICTIONS-2026-09-29.md``.** The plan's three: (1) both wirings hold
together under the same supply condition as the pair; (2) cutting one part's supply ends
the ring, every part in turn, while all-to-all loses only that part and settles lower;
(3) the ring's collapse runs around it at a speed set by the parts' settling time. The
run's own, E1-E5, are in the note and are quoted in the docstrings below.

**The construction.** ``n`` parts, each item 12's part (``mn.part``): a configuration
coalgebra on its amplitude ``q_i`` with mobility ``h``, publishing ``q_i``, carrying its
own dissipation ``-d (b/2) q_i^2``. One supply box holding ``s`` (zero response). The
three are tensored and closed by a stateless coupler on the ports ``e = (q_1..q_n, s)``
returning

    F_i(e) = alpha ( s m_i g(r_i, q_i) + beta r_i ),     g(r, q) = r q / (K + r q),

where ``r_i`` is what part ``i`` receives from the parts it needs -- *ring*:
``r_i = q_{i-1}``; *all-to-all*: ``r_i = mean_{j != i} q_j`` -- and ``m_i`` is the supply
mask (a cut wire delivers ``0``, item 12's reading). With ``beta = 0`` a part with nothing
arriving has no drive at all, item 12's condition that no ungated path reach a part;
``beta > 0`` is that run's ungated internal coupling, run here as a control. The composite
forcing on ``Q = R^{n+1}`` is ``xi_i = F_i - b q_i``, ``xi_S = 0``, and one step is
``q + h xi``. Fixed points do not depend on ``h``.

Every docstring below says which of three things its test does (STATUS.md §3.7):
reproduce a closed form derived in the docstring, check a discrete analogue, or correct a
claim made on paper. The numbers: ``notes/MANY-PARTS-RESULTS.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement
from arrangements.network import fan_box, map_box, plus_box
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests import test_mutual_need as mn
from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

ALPHA, DISS, KGATE, MOB = mn.ALPHA, mn.DISS, mn.KGATE, mn.MOB
STEPS = mn.STEPS          # 20000 steps: 200 units of h-time
HELD = mn.HELD
EMPTY = mn.EMPTY
WIRINGS = ("ring", "all")


# ---------------------------------------------------------------------------
# The coupler, and the closed arrangement.
# ---------------------------------------------------------------------------


def F_many(n, wiring, beta=0.0, s_mask=None, out_mask=None, alpha=ALPHA, K=KGATE):
    """The returned covector ``F`` on the ports ``e = (q_1..q_n, s)`` (module docstring).
    ``s_mask[i] = 0`` cuts part ``i``'s supply wire; ``out_mask[i] = 0`` cuts its outgoing
    wire, so that the parts needing it receive ``0`` from it."""
    s_mask = jnp.ones(n) if s_mask is None else jnp.asarray(s_mask, float)
    out_mask = jnp.ones(n) if out_mask is None else jnp.asarray(out_mask, float)

    def F(e):
        q, s = e[:n], e[n]
        seen = q * out_mask
        if wiring == "ring":
            r = jnp.roll(seen, 1)                              # r_i = q_{i-1}
        else:
            r = (jnp.sum(seen) - seen) / (n - 1)               # the others' mean
        u = r * q
        g = u / (K + u)
        drive = alpha * (s * s_mask * g + beta * r)
        return jnp.concatenate([drive, jnp.zeros(1)])

    return F


def need_box(beta=0.0, alpha=ALPHA, K=KGATE, label="need"):
    """A part's need, ``<0|3> -> I`` on ``(q, r, s)``: the part's own amplitude, what arrives from the
    parts it needs, and the supply. It returns ``alpha (s g(r, q) + beta r)`` to the part and nothing to
    the senders or the supply (one way, as the coupler's returned covector was)."""
    def omega(_q, e, _n):
        q, r, s = e[0], e[1], e[2]
        u = r * q
        return EMPTY, jnp.stack([alpha * (s * u / (K + u) + beta * r), 0.0, 0.0]), EMPTY

    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=3, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, e: EMPTY, in_f=lambda q, e, n: EMPTY, omega=omega, label=label)


def mean_of(k):
    """The others' mean as boxes: a ``+`` box over ``k`` wires, then a box dividing by ``k``."""
    return compose_seq(plus_box(k), map_box(lambda v: v / k, 1, 1, f"/{k}"))


def collective(n, wiring, q0, s, beta=0.0, h=MOB, b=DISS, s_mask=None, out_mask=None, receive=None,
               alpha=ALPHA, K=KGATE):
    """The collective as parts (P43 (b) job 8, Claude Opus 5.5, de71554a, 2026-10-10): the ``n`` parts and
    the supply, a box presenting ``0`` (what a cut wire delivers), and per part a need box hearing the part,
    what arrives from the parts it needs and the supply. The wiring decides what arrives: *ring*, the
    previous part's wire; *all*, the others' wires through a mean box (or ``receive``, a box over the
    others). A cut supply or outgoing wire is the zero box wired in its place. Configuration
    ``(q_1..q_n, s)``. Equal to ``collective_one_coupler``, where one coupler held the wiring, the mean and
    the gates (``test_0b``)."""
    q0 = [float(q0)] * n if isinstance(q0, (int, float)) else list(q0)
    s_mask = [1.0] * n if s_mask is None else [float(x) for x in s_mask]
    out_mask = [1.0] * n if out_mask is None else [float(x) for x in out_mask]
    zero = map_box(lambda v: jnp.zeros(1), 0, 1, "cut wire")
    parts = [mn.part(q0[i], b=b, h=h, label=f"P{i}") for i in range(n)] + [mn.supply(s), zero]
    wire = lambda j: j if out_mask[j] else n + 1
    idx, stages = [], []
    for i in range(n):
        senders = [wire((i - 1) % n)] if wiring == "ring" else [wire(j) for j in range(n) if j != i]
        idx += [i] + senders + [n if s_mask[i] else n + 1]
        recv = (fan_box(1, [0], "neighbour") if wiring == "ring"
                else (receive(n - 1) if receive is not None else mean_of(n - 1)))
        stages.append(compose_seq(tensor_arrangements([fan_box(1, [0], "own"), recv, fan_box(1, [0], "supply")]),
                                  need_box(beta, alpha, K, f"need {i}")))
    routed = compose_seq(tensor_arrangements(parts), fan_box(n + 2, idx, "wiring"))
    return compose_seq(routed, tensor_arrangements(stages))


def collective_one_coupler(n, wiring, q0, s, beta=0.0, h=MOB, b=DISS, **kw):
    """The run before P43 (b): ``compose_seq(tensor(P_1..P_n, S), coupler)``, one coupler holding the
    wiring, the others' mean and the gates. Configuration ``(q_1..q_n, s)``."""
    q0 = [float(q0)] * n if isinstance(q0, (int, float)) else list(q0)
    parts = [mn.part(q0[i], b=b, h=h, label=f"P{i}") for i in range(n)] + [mn.supply(s)]
    return compose_seq(tensor_arrangements(parts), mn.coupler(F_many(n, wiring, beta, **kw), n + 1))


def trajectory(arr, steps):
    """The configurations at every step, by ``lax.scan`` of the same jitted step
    ``settle`` uses (test 0 checks the step is ``Phi``'s)."""
    P, omega = arr.P, arr.omega

    def step(state, _):
        xi_q, _, _ = omega(P.position(state), EMPTY, EMPTY)
        new = P.step(state, xi_q)
        return new, flat_state(new)

    _, out = jax.lax.scan(step, P.init, None, length=steps)
    return np.asarray(out)


def field(arr, n):
    """The composite forcing on the parts' amplitudes, as a function of them alone."""
    s = float(flat_state(arr.P.init)[n])

    def xi(q):
        return arr.omega(jnp.concatenate([q, jnp.array([s])]), EMPTY, EMPTY)[0][:n]

    return xi


# ---------------------------------------------------------------------------
# The closed forms.
# ---------------------------------------------------------------------------


def symmetric_roots(s, beta=0.0, alpha=ALPHA, b=DISS, K=KGATE):
    """At ``q_i = q`` both wirings give ``r = q``, so the nonzero fixed points solve
    ``(b - alpha beta) q (K + q^2) = alpha s q^2``:
    ``q± = [alpha s ± sqrt(alpha^2 s^2 - 4 (b - alpha beta)^2 K)] / (2 (b - alpha beta))``,
    and ``s_* = 2 (b - alpha beta) sqrt(K) / alpha``. Returns ``(q-, q+, s_*)``, ``nan``
    roots below ``s_*``."""
    B = b - alpha * beta
    s_star = 2.0 * B * np.sqrt(K) / alpha
    disc = (alpha * s) ** 2 - 4.0 * B ** 2 * K
    if disc < 0.0:
        return float("nan"), float("nan"), s_star
    root = np.sqrt(disc)
    return (alpha * s - root) / (2.0 * B), (alpha * s + root) / (2.0 * B), s_star


def survivors_roots(n, s, beta=0.0, alpha=ALPHA, b=DISS, K=KGATE):
    """All-to-all with one part dead: the survivors' symmetric state has ``r = c q``,
    ``c = (n-2)/(n-1)``, so ``(b - alpha beta c) q (K + c q^2) = alpha s c q^2`` and
    ``s_*' = 2 (b - alpha beta c) sqrt(K / c) / alpha``. Returns ``(q+', s_*')``."""
    c = (n - 2.0) / (n - 1.0)
    B = b - alpha * beta * c
    s_star = 2.0 * B * np.sqrt(K / c) / alpha
    disc = (alpha * s * c) ** 2 - 4.0 * c * B ** 2 * K
    if disc < 0.0:
        return float("nan"), s_star
    return (alpha * s * c + np.sqrt(disc)) / (2.0 * c * B), s_star


def circulant_eigs(n, wiring, s, beta=0.0, alpha=ALPHA, b=DISS, K=KGATE):
    """The Jacobian's eigenvalues at ``q+``, derived in the predictions note: with
    ``gam = K q / (K + q^2)^2``, the ring's are
    ``alpha s gam (1 + w^k) - b + alpha beta w^k``, ``w = e^{2 pi i / n}``; all-to-all's
    are the symmetric mode ``2 alpha s gam + alpha beta - b`` and, ``n - 1`` times,
    ``alpha s gam - b - (alpha s gam + alpha beta) / (n - 1)``."""
    _, q, _ = symmetric_roots(s, beta, alpha, b, K)
    gam = K * q / (K + q * q) ** 2
    if wiring == "ring":
        w = np.exp(2j * np.pi * np.arange(n) / n)
        return alpha * s * gam * (1.0 + w) - b + alpha * beta * w
    sym = 2.0 * alpha * s * gam + alpha * beta - b
    rest = alpha * s * gam - b - (alpha * s * gam + alpha * beta) / (n - 1)
    return np.array([sym] + [rest] * (n - 1), dtype=complex)


def held_all(q):
    return [mn.held(x) for x in q]


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0b_the_parts_are_the_one_coupler():
    """The collective as parts equals the one-coupler run, step for step over 2000 steps, on both
    wirings, from an uneven start, with beta 0 and 0.5, a cut supply and a cut outgoing wire. The
    prediction (the need boxes' covectors sum to the coupler's), not a premise."""
    q0 = [0.4, 1.3, 0.9, 1.1, 0.2]
    for wiring in WIRINGS:
        for beta in (0.0, 0.5):
            for kw in ({}, {"s_mask": [1, 0, 1, 1, 1]}, {"out_mask": [1, 1, 0, 1, 1]}):
                a = trajectory(collective(5, wiring, q0, 4.0, beta, **kw), 2000)
                o = trajectory(collective_one_coupler(5, wiring, q0, 4.0, beta, **kw), 2000)
                np.testing.assert_allclose(a, o, atol=1e-12)


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run forms used below; it establishes nothing about the predictions.
    ``Phi`` of the composite, ``run_closed``, ``mn.settle``'s ``fori_loop`` and
    ``trajectory``'s ``scan`` agree exactly over fifty steps."""
    for wiring in WIRINGS:
        arr = collective(3, wiring, 1.0, 4.0)
        by_phi = run_phi(arr, 50, EMPTY)
        by_closed = run_closed(arr, 50, lambda t: EMPTY)
        assert float(np.max(np.abs(by_phi - by_closed))) == 0.0
        np.testing.assert_array_equal(mn.settle(arr, 50), by_closed[-1])
        np.testing.assert_array_equal(trajectory(arr, 50), by_closed[1:])


# ---------------------------------------------------------------------------
# Test 1: prediction 1 and E1 -- one supply condition for every n and both wirings.
# ---------------------------------------------------------------------------


def test_1_prediction_1_the_supply_threshold_is_the_pairs_for_every_n():
    """REPRODUCES the closed forms for ``q+`` and ``s_*`` (module docstring) and the
    circulant eigenvalues, and so tests prediction 1 and E1.

    At ``beta = 0``, ``s_* = 2``: at ``s = 1.5`` nothing is held and at ``s = 2.5, 4``
    every part settles at ``q+`` (``2`` and ``3.732051``), for ``n = 2, 3, 5, 8`` and both
    wirings, from ``q0 = 1`` (above the saddle ``q-``). At ``beta = 0.5``, ``s_* = 1``:
    ``s = 0.8`` dies, ``s = 1.5`` holds at ``q+ = 1.618034``. The Jacobian's eigenvalues at
    ``q+`` are the circulant ones, every real part negative, and the largest is the
    symmetric mode's, ``2 alpha s gam + alpha beta - b``, which is the pair's: the
    threshold is carried by the mode that does not see ``n`` or the wiring (E1).
    """
    cases = [(0.0, 1.5, False), (0.0, 2.5, True), (0.0, 4.0, True),
             (0.5, 0.8, False), (0.5, 1.5, True)]
    for beta, s, expect in cases:
        q_lo, q_hi, s_star = symmetric_roots(s, beta)
        assert (s > s_star) == expect
        for n in (2, 3, 5, 8):
            for wiring in WIRINGS:
                arr = collective(n, wiring, 1.0, s, beta)
                q = mn.settle(arr)[:n]
                assert held_all(q) == [expect] * n, (beta, s, n, wiring, q)
                if expect:
                    assert q_lo < 1.0 < q_hi
                    np.testing.assert_allclose(q, q_hi, atol=1e-6)
                    J = np.asarray(jax.jacobian(field(arr, n))(jnp.asarray(q)))
                    eig = np.linalg.eigvals(J)
                    pred = circulant_eigs(n, wiring, s, beta)
                    np.testing.assert_allclose(np.sort_complex(eig), np.sort_complex(pred),
                                               atol=1e-6)
                    assert np.max(eig.real) < 0.0
                    gam = KGATE * q_hi / (KGATE + q_hi ** 2) ** 2
                    np.testing.assert_allclose(np.max(eig.real),
                                               2.0 * ALPHA * s * gam + ALPHA * beta - DISS,
                                               atol=1e-6)
                else:
                    assert float(np.max(q)) < 1e-30


# ---------------------------------------------------------------------------
# Test 2: prediction 2 and E2 -- cutting one part's supply.
# ---------------------------------------------------------------------------


def cut_supply(n, wiring, s, beta=0.0, which=0, steps=STEPS):
    """From the settled symmetric state, part ``which``'s supply wire delivers ``0``."""
    _, q_hi, _ = symmetric_roots(s, beta)
    mask = np.ones(n)
    mask[which] = 0.0
    return mn.settle(collective(n, wiring, q_hi, s, beta, s_mask=mask), steps)[:n]


def test_2_prediction_2_a_cut_ends_the_ring_at_beta_0_and_all_to_all_survives_above_s_star_prime():
    """REPRODUCES the survivors' closed form ``q+'`` and threshold ``s_*'`` (module
    docstring), and so tests prediction 2 and E2.

    *Ring, ``beta = 0``.* The cut part has no drive, dies at rate ``b``; its neighbour's
    gate then reads ``0`` and it dies too; around the ring every part dies: prediction 2's
    ring half holds, at ``n = 3, 5, 8`` and ``s = 2.5, 4``.

    *Ring, ``beta = 0.5``.* The cut part settles at ``alpha beta q_{i-1} / b``, not at
    zero, and the ring holds with that part degraded: the ring half of prediction 2 is a
    statement about a wiring with no ungated internal path, item 12's condition again.

    *All-to-all, ``beta = 0``.* The cut part dies; the survivors settle at ``q+'`` iff
    ``s > s_*' = 2 sqrt(K (n-1)/(n-2))``: at ``s = 4`` they hold for ``n = 3, 5, 8``
    (``3.414214`` at ``n = 3``, below the intact ``3.732051``); at ``s = 2.5`` they collapse
    at ``n = 3`` (``s_*' = 2.828``) and hold at ``n = 5, 8`` (``2.309``, ``2.160``). So a
    larger all-to-all collective survives a loss that a smaller one does not, at a supply
    where every intact collective held; the ring at ``beta = 0`` never does.
    """
    for s in (2.5, 4.0):
        for n in (3, 5, 8):
            q = cut_supply(n, "ring", s, 0.0)
            assert held_all(q) == [False] * n, (s, n, q)
            assert float(np.max(q)) < 1e-30

            q = cut_supply(n, "ring", s, 0.5)
            assert held_all(q) == [True] * n, (s, n, q)
            assert q[0] < 0.9 * np.min(q[1:])           # degraded, not dead
            np.testing.assert_allclose(q[0], ALPHA * 0.5 * q[-1] / DISS, atol=1e-6)

            q = cut_supply(n, "all", s, 0.0)
            q_surv, s_prime = survivors_roots(n, s)
            assert not mn.held(q[0])
            if s > s_prime:
                assert held_all(q[1:]) == [True] * (n - 1), (s, n, q)
                np.testing.assert_allclose(q[1:], q_surv, atol=1e-6)
                assert q_surv < symmetric_roots(s)[1]     # settles lower
            else:
                assert held_all(q[1:]) == [False] * (n - 1), (s, n, q)
    np.testing.assert_allclose(survivors_roots(3, 4.0)[0], 2.0 + np.sqrt(2.0), atol=1e-12)
    np.testing.assert_allclose([survivors_roots(n, 2.5)[1] for n in (3, 5, 8)],
                               [2.0 * np.sqrt(2.0), 2.0 * np.sqrt(4.0 / 3.0), 2.0 * np.sqrt(7.0 / 6.0)],
                               atol=1e-12)


# ---------------------------------------------------------------------------
# Test 3: prediction 3 and E3, E4 -- the front around the ring.
# ---------------------------------------------------------------------------


def death_times(n, s, b=DISS, h=MOB, steps=STEPS, out_cut=False, which=0):
    """The h-time at which each part first falls below ``HELD``, from the settled state,
    after part ``which``'s supply (or, ``out_cut``, its outgoing wire) is cut."""
    _, q_hi, _ = symmetric_roots(s, 0.0, b=b)
    mask = np.ones(n)
    mask[which] = 0.0
    kw = dict(out_mask=mask) if out_cut else dict(s_mask=mask)
    tr = trajectory(collective(n, "ring", q_hi, s, 0.0, h=h, b=b, **kw), steps)[:, :n]
    below = np.abs(tr) < HELD
    assert np.all(np.any(below, axis=0)), "every part must die within the run"
    return h * (np.argmax(below, axis=0) + 1)


def test_3_prediction_3_the_collapse_is_a_front_whose_delay_is_set_by_b_and_by_s():
    """CHECKS the discrete analogue of a front: the deaths go around the ring in order,
    the delay between successive deaths is the same for every part once the front is
    running, it doubles when ``b`` is halved with ``s/b`` fixed, and it grows with ``s``.
    E3 also named a value, ``(1/b) log(q+^2 / K)``; the docstring of the results note says
    how far that was from the measured delay, which is the result.
    """
    n = 8
    t = death_times(n, 4.0)
    order = np.argsort(t)
    np.testing.assert_array_equal(order, np.arange(n))         # 0, 1, ..., 7 in turn
    delays = np.diff(t)
    running = delays[2:]                                        # once the front is going
    assert np.max(running) - np.min(running) < 0.05 * np.mean(running)

    # halving b with s/b fixed doubles every delay (the field scales by 1/2)
    t_half = death_times(n, 2.0, b=0.5, steps=2 * STEPS)
    np.testing.assert_allclose(np.diff(t_half)[2:], 2.0 * running, rtol=2e-2)

    # the delay grows with s
    d = [np.mean(np.diff(death_times(n, s, steps=STEPS))[2:]) for s in (3.0, 4.0, 6.0)]
    assert d[0] < d[1] < d[2]
    ref = [np.log(symmetric_roots(s)[1] ** 2 / KGATE) / DISS for s in (3.0, 4.0, 6.0)]
    print("front delays", d, "E3's value", ref)

    # E4: cutting the outgoing wire starts the same front one part on
    t_out = death_times(n, 4.0, out_cut=True)
    assert np.argmin(t_out) == 1
    np.testing.assert_array_equal(np.argsort(t_out), np.r_[np.arange(1, n), 0])
    np.testing.assert_allclose(np.diff(t_out[1:])[2:], running[:4], rtol=2e-2)


# ---------------------------------------------------------------------------
# Test 4: E5 -- the restart threshold survives.
# ---------------------------------------------------------------------------


def test_4_E5_the_origin_is_stable_and_the_saddle_is_the_threshold():
    """REPRODUCES the closed forms: the forcing vanishes at the origin, the Jacobian
    there is ``-b I`` at ``beta = 0`` (``-b + alpha beta`` on the ring's circulant modes
    otherwise), and on the symmetric ray a collective started at ``0.9 q-`` dies while one
    at ``1.1 q-`` settles at ``q+``, for ``n = 3, 5`` and both wirings."""
    for wiring in WIRINGS:
        for n in (3, 5):
            q_lo, q_hi, _ = symmetric_roots(4.0)
            arr = collective(n, wiring, 0.0, 4.0)
            xi = field(arr, n)
            np.testing.assert_allclose(xi(jnp.zeros(n)), 0.0, atol=0.0)
            J = np.asarray(jax.jacobian(xi)(jnp.zeros(n)))
            np.testing.assert_allclose(J, -DISS * np.eye(n), atol=1e-12)
            below = mn.settle(collective(n, wiring, 0.9 * q_lo, 4.0))[:n]
            above = mn.settle(collective(n, wiring, 1.1 * q_lo, 4.0))[:n]
            assert held_all(below) == [False] * n
            np.testing.assert_allclose(above, q_hi, atol=1e-6)
    # the couplers are non-exact, as item 12's gate is: the ring's need is one-way
    n = 3
    q_pt = np.array([3.7, 3.7, 3.7, 4.0])
    dw_ring = mn.d_omega(F_many(n, "ring"))(q_pt)
    dw_all = mn.d_omega(F_many(n, "all"))(q_pt)
    assert np.max(np.abs(dw_ring)) > 1e-3
    print("d omega ring", dw_ring[:n, :n], "all", dw_all[:n, :n])
