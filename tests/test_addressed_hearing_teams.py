"""Addressed hearing teams: roles nobody assigned, at one level and two (book plan P104; ``hearing_teams.py``).

**The question** (David, 2026-11-15/16, bbb49fa7): "the most interesting thing we can do with a compositional theory of
predictors, while still being not too complicated"; "maybe in some conditions member 1 is more right and in other
conditions member 2 is more right, so bob's state tries to track conditions and hence how much to listen to each
member?" (2026-10-29, b0471e87); and "an arrangement of systems ... able to do something together that the pieces
couldn't do themselves ... organized groups each doing their own thing, reorganizing to do things better".

**The arrangements** (``hearing_teams.py``'s docstring): members that record what they hear and send the record as
their address beside a word ``a x + b``; hearing teams whose trust-levels are ``softmax(C_k a_k + D_k)`` of the
members' addresses; a team of teams is composition in arr (``compose_seq``, ``tensor_arrangements``), and
``closed_step`` runs the whole, so every load reaches every configuration by pullback-and-sum and no gradient is
written by hand. A context reaches the trust-levels only through an address a member sent, so it is recorded a step
before the outcome it helps predict arrives. Seeds: world ``s``, initial configuration ``100 + s`` (one level) or
``200 + s`` (two levels), ``s = 0..4``; every run starts its records at 0.

**Worlds.** ``f2``: ``y_t = f(y_{t-1}) + .3 eps``, ``f = .5x + 1`` for ``x < 0``, ``-.9x`` otherwise (floor .09; the
best line .222). ``zig``: the world sends ``(y_t, x_{t+1})``, contexts uniform on ``[-2, 2]``, ``y_t = zig(x_t) +
.1 eps``, ``zig`` four straight pieces with knots at -1, 0, 1 (floor .01); after the switch ``1 - zig(x + .5)``.
``f4``: ``y_t = f(y_{t-1}) + .3 eps``, four lines visited in a cycle (floor .09, best line .788).

**Predictions, written before the runs** (the book repo's ``notes/tasks/p104-addressed-teams-62f4c8d3.md``, from
bbb49fa7's scratch numbers): (1) one level on ``f2``: ``K = 1`` blurs (.17-.20), ``K = 10, 100`` give .105-.115 with
one line per piece in 5/5, one member alone .22-.23. (2) two levels on ``f4``: under .15 with four leaders in at least
4/5; flat about the same; alone about .79. (3) reorganization on ``f4 -> f4b``: back under .15 in at least 4/5.

**What happened** (2026-11-16, Claude Opus 5.5, 62f4c8d3). (1) held: ``K = 1`` .189 .181 .178 .183 .191; ``K = 10``
.108 .108 .107 .108 .108; ``K = 100`` .107 .106 .106 .107 .106; record drift .15-.20 at ``K = 1``, .013-.017 at 10,
.001-.002 at 100; alone .225-.227. Each piece is taken by one line; in seeds 0 and 4 the team's own voice takes one
piece and a member stays idle. (2) on ``f4`` failed: nested .158 .139 .156 .124 .143 with four leaders only in seed 3,
flat .144-.145 with two members used of four; four times longer and at eta .01, .1 the same states persist. A control,
bbb49fa7's hand-gradient tree handed the context exactly (the book repo's ``.scratch/p104-addressed-teams/
control_scratch.py``), sticks the same way (.130-.144, flat .142-.149), so the stuck states are the world's, not the
book form's. On ``zig`` (the world bbb49fa7's scratch used, contexts now recorded a step earlier): nested .012 .011
.012 .033 .027, flat .011-.013, alone .097; the top's boundary near 0 (seeds 1, 3, 4), where all four members are
heard most somewhere, or near 1 (0, 2), where one group covers three pieces and three members are. The members'
lines are not the four pieces: the gates stay soft and neighbouring lines blend (seed 1: ``+.56x+.97``,
``-1.89x+.04``, ``+1.87x+.05``, ``-.66x+1.09``), so a role is a stretch of contexts where one member is heard most
(shares .55-.9), not one piece. (3) on ``zig``: before .011-.012, just after the switch .048-.074, after .012 (seeds
1, 3, 4) and .026 (0, 2); the top's boundary moves from about 0 to about .6, or from about 1 to about 1.7.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements import hearing_teams as H

jax.config.update("jax_enable_x64", True)

SEEDS = range(5)


def f2(x):
    return 0.5 * x + 1.0 if x < 0 else -0.9 * x


def f4(x):
    return -x if x < -1 else (0.5 * x + 0.75 if x < 0 else (-x - 1.0 if x < 1 else x - 2.0))


def zig(x):
    return 1.0 - np.abs(((x + 2.0) % 2.0) - 1.0)


def zig_pairs(seed, T, switch=False):
    """Row ``t`` is ``(y_t, x_{t+1})``: the outcome and the next context."""
    rng = np.random.default_rng(seed)
    x = rng.uniform(-2, 2, T + 1)
    y = zig(x[:T]) + 0.1 * rng.standard_normal(T)
    if switch:
        h = T // 2
        y[h:] = 1.0 - zig(x[h:T] + 0.5) + 0.1 * rng.standard_normal(T - h)
    return np.stack([y, x[1:]], 1)


def one_level(seed, K, eta=0.01):
    rng = np.random.default_rng(100 + seed)
    ms = [H.member(K, eta, H.member_init(rng), f"m{k + 1}") for k in range(2)]
    return H.fill(ms, H.team(2, eta, H.team_init(2, rng, voice=True), voice=True))


def nested(seed, K=100.0, eta=0.03, obs_dim=2, scale=0.3):
    rng = np.random.default_rng(200 + seed)
    groups = []
    for name in "AB":
        ms = [H.member(K, eta, H.member_init(rng, scale), f"{name}{k + 1}", obs_dim=obs_dim) for k in range(2)]
        groups.append(H.fill(ms, H.team(2, eta, H.team_init(2, rng, scale=scale), residual=False, label=name,
                                         obs_dim=obs_dim)))
    return H.fill(groups, H.team(2, eta, H.team_init(2, rng, scale=scale), label="top", obs_dim=obs_dim))


def flat(seed, K=100.0, eta=0.03, obs_dim=2, scale=0.3):
    rng = np.random.default_rng(200 + seed)
    ms = [H.member(K, eta, H.member_init(rng, scale), f"m{k + 1}", obs_dim=obs_dim) for k in range(4)]
    return H.fill(ms, H.team(4, eta, H.team_init(4, rng, scale=scale), label="flat", obs_dim=obs_dim))


def alone(seed, K=100.0, eta=0.03, obs_dim=1):
    rng = np.random.default_rng(100 + seed)
    return H.fill([H.member(K, eta, H.member_init(rng), obs_dim=obs_dim)],
                  H.team(1, eta, H.team_init(1, rng), obs_dim=obs_dim))


def err(series, preds, start):
    y = np.asarray(series).reshape(len(series), -1)[:, 0]
    return float(np.mean((y - preds)[start:] ** 2))


# position layout of ``nested``: A1(3) A2(3) A(4) B1(3) B2(3) B(4) top(4), then the world's clock
def product_shares(q, x):
    g = np.asarray(H.team_shares(q[20:24], np.array([x, x]), False))
    hA = np.asarray(H.team_shares(q[6:10], np.array([x, x]), False))
    hB = np.asarray(H.team_shares(q[16:20], np.array([x, x]), False))
    return np.concatenate([g[0] * hA, g[1] * hB])


def leaders(q, lo=-2.0, hi=2.0):
    return sorted({int(np.argmax(product_shares(q, x))) for x in np.linspace(lo, hi, 81)})


def top_boundary(q):
    xs = np.linspace(-2.0, 2.5, 451)
    g = np.array([np.asarray(H.team_shares(q[20:24], np.array([x, x]), False))[0] for x in xs])
    return [float(xs[i]) for i in np.where(np.diff(np.sign(g - 0.5)) != 0)[0]]


def test_team_of_teams_hears_by_products():
    """The composite's out-transfer is the top's trust-levels times each group's, read off the addresses
    (``prop.hearing_mixes`` with address-dependent trust-levels): checked against the formula at random points."""
    whole = nested(0, obs_dim=1)
    rng = np.random.default_rng(7)
    for _ in range(5):
        q = jnp.asarray(rng.normal(0, 1, whole.P.Q_dim))
        got = whole.out_f(q, H.EMPTY)
        rec = np.asarray(q)[[0, 3, 10, 13]]
        words = np.asarray(q)[[1, 4, 11, 14]] * rec + np.asarray(q)[[2, 5, 12, 15]]
        g = np.asarray(H.team_shares(q[20:24], jnp.array([rec[:2].mean(), rec[2:].mean()]), False))
        hA = np.asarray(H.team_shares(q[6:10], jnp.asarray(rec[:2]), False))
        hB = np.asarray(H.team_shares(q[16:20], jnp.asarray(rec[2:]), False))
        shares = np.concatenate([g[0] * hA, g[1] * hB])
        assert np.isclose(shares.sum(), 1.0)
        assert np.allclose(np.asarray(got), [rec.mean(), shares @ words], atol=1e-12)


def test_one_level_roles_and_record_weight():
    T = 200000
    out = {}
    for K in (1.0, 10.0, 100.0):
        for s in SEEDS:
            y = H.ar_series(f2, T, 0.3, s)
            _, qs, p = H.run(one_level(s, K), y)
            drift = float(np.mean(np.abs(qs[1:, 0] - y[:-1])[T // 2:]))
            out[K, s] = (err(y, p, T // 2), drift)
            print(f"one level K={K:g} seed {s}: team {out[K, s][0]:.3f} record drift {drift:.4f}")
    lone = [err(y, H.run(alone(s, eta=0.01), y)[2], T // 2)
            for s in SEEDS for y in [H.ar_series(f2, T, 0.3, s)]]
    print("one member alone:", np.round(lone, 3))
    assert all(0.15 < out[1.0, s][0] < 0.22 for s in SEEDS), "K = 1: the team's load bends the records; roles blur"
    assert all(out[K, s][0] < 0.115 for K in (10.0, 100.0) for s in SEEDS)
    assert all(out[100.0, s][1] < 0.005 for s in SEEDS), "K = 100 keeps the records faithful"
    assert all(e > 0.2 for e in lone)


def test_two_levels_and_flat_on_zigzag():
    T = 400000
    for s in SEEDS:
        Y = zig_pairs(s, T)
        _, qs, p = H.run(nested(s), Y)
        _, _, pf = H.run(flat(s), Y)
        _, _, pa = H.run(alone(s, obs_dim=2), Y)
        e, ef, ea = err(Y, p, T // 2), err(Y, pf, T // 2), err(Y, pa, T // 2)
        L, b = leaders(qs[-1]), top_boundary(qs[-1])
        print(f"zig seed {s}: nested {e:.3f} leaders {L} top boundary {np.round(b, 2)}  flat {ef:.3f}  alone {ea:.3f}")
        assert e < 0.04 and ef < 0.02 and ea > 0.09
        assert len(L) >= 3 and len(b) == 1, "the top splits the contexts once, and at least three members lead somewhere"


def test_two_levels_on_the_cycle_world_stick():
    """The four-line cycle world: better than one member, short of four roles (the world's, per the control)."""
    T = 400000
    for s in SEEDS:
        y = H.ar_series(f4, T, 0.3, s)
        e = err(y, H.run(nested(s, obs_dim=1), y)[2], T // 2)
        ef = err(y, H.run(flat(s, obs_dim=1), y)[2], T // 2)
        print(f"f4 seed {s}: nested {e:.3f} flat {ef:.3f}")
        assert e < 0.2 and ef < 0.2


def test_reorganization_on_zigzag():
    T = 400000
    for s in SEEDS:
        Y = zig_pairs(s, 2 * T, switch=True)
        _, qs, p = H.run(nested(s), Y)
        sq = (Y[:, 0] - p) ** 2
        q = T // 4
        before, just, after = np.mean(sq[T - q:T]), np.mean(sq[T:T + 2000]), np.mean(sq[-q:])
        b0, b1 = top_boundary(qs[T]), top_boundary(qs[-1])
        print(f"reorg seed {s}: before {before:.3f} just after {just:.3f} after {after:.3f} "
              f"top boundary {np.round(b0, 2)} -> {np.round(b1, 2)}")
        assert before < 0.02 and just > 3 * before and after < 0.03
        assert len(b0) == 1 and len(b1) == 1 and b1[0] - b0[0] > 0.4, "the boundary between the groups moves"
