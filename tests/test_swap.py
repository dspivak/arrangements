"""The swap, piece 1 (book plan P37 (b)(vii); ``arrangements/swap.py``).

Predictions fixed before any code, in the book repo's ``notes/runs/p37-swap-build-4f4ee268/PREDICTIONS.md``
(Claude Opus 5.5, 4f4ee268, 2026-10-15); results in the same folder's ``RESULTS.md``. Each test is one prediction;
run them one node at a time with ``PYTEST_ADDOPTS=-s scripts/run_tests.sh``.
"""
import functools

import jax
import numpy as np
import pytest

from arrangements import swap as W

jax.config.update("jax_enable_x64", True)

NOISES = (0.25, 0.1)


def _casts(problem):
    (a0, b0), _ = W.PROBLEMS[problem]
    return W.family(a0, b0)


def _top_cast(problem, trust):
    """The most trusted cast, the base's two scenes merged."""
    tot = {}
    for c, q in zip(_casts(problem), trust):
        tot[c] = tot.get(c, 0.0) + q
    top = max(tot, key=tot.get)
    return top, tot[top]


@functools.lru_cache(maxsize=None)
def _s0(problem, sigma, trigger=True):
    return W.s0_run(problem, sigma, trigger=trigger)


@functools.lru_cache(maxsize=None)
def _s6(problem, sigma):
    return W.s6_run(problem, sigma)


def _s6_top(row):
    return (W.A_VALUES[int(np.argmax(row["snake"]))], W.B_VALUES[int(np.argmax(row["cat"]))])


@pytest.mark.parametrize("problem", ["P1", "P2", "P3"])
def test_1_s0_equals_the_hand_recurrence(problem):
    worst = 0.0
    for sigma in NOISES:
        rows, hand = _s0(problem, sigma), W.s0_by_hand(problem, sigma)
        worst = max(worst, max(np.abs(r["trust"] - h["trust"]).max() for r, h in zip(rows, hand)),
                    max(np.abs(r["sums"] - h["sums"]).max() for r, h in zip(rows, hand)))
    print(f"\n1 S0 {problem}: arr vs numpy, trust and summed misses, worst {worst:.1e}")
    assert worst < 1e-12


@pytest.mark.parametrize("problem", ["P1", "P2"])
def test_1_s6_equals_the_hand_recurrence(problem):
    worst = 0.0
    for sigma in NOISES:
        rows, hand = _s6(problem, sigma), W.s6_by_hand(problem, sigma)
        worst = max(worst, max(max(np.abs(r[k] - h[k]).max() for k in ("snake", "cat")) for r, h in zip(rows, hand)))
    print(f"\n1 S6 {problem}: arr vs numpy, both beliefs, worst {worst:.1e}")
    assert worst < 1e-12


def test_2_s0_as_in_the_hand_table():
    out = {}
    for problem in ("P1", "P2"):
        truth = W.PROBLEMS[problem][1]
        for sigma in NOISES:
            tops = [_top_cast(problem, r["trust"]) for r in _s0(problem, sigma)[:12]]
            out[(problem, sigma)] = (W.settled([t for t, _ in tops], truth), tops[-1][0], round(tops[-1][1], 4))
    for k, v in out.items():
        print(f"\n2 S0 {k[0]} sd {k[1]}: settled {v[0]}, end on {v[1]} with trust {v[2]}", end="")
    print()
    assert out[("P1", 0.25)][0] == 1 and out[("P1", 0.1)][0] == 1
    for sigma in NOISES:
        assert out[("P2", sigma)][0] is None and out[("P2", sigma)][1] == (0.2, 0.2) and out[("P2", sigma)][2] > 0.99


def test_3_s6_settles():
    out = {}
    for problem in ("P1", "P2"):
        truth = W.PROBLEMS[problem][1]
        for sigma in NOISES:
            rows = _s6(problem, sigma)
            tops = [_s6_top(r) for r in rows]
            out[(problem, sigma)] = (W.settled(tops, truth), tops,
                                     round(float(rows[-1]["snake"].max()), 4), round(float(rows[-1]["cat"].max()), 4))
    for k, v in out.items():
        print(f"\n3 S6 {k[0]} sd {k[1]}: settled {v[0]}; tops rounds 1-3 {v[1][:3]}; end tops' trust {v[2]}, {v[3]}",
              end="")
    print()
    for k, v in out.items():
        assert v[0] is not None and v[0] <= 2, k
        assert v[1][-1] == (0.3, 0.4) and v[2] > 0.99 and v[3] > 0.99, k


def test_4_s6_does_not_read_the_start():
    for sigma in NOISES:
        d = max(max(np.abs(r[k] - h[k]).max() for k in ("snake", "cat")) for r, h in zip(_s6("P1", sigma), _s6("P2", sigma)))
        print(f"\n4 S6 sd {sigma}: P1 vs P2, worst difference {d:.1e}")
        assert d == 0.0


def test_5_scenes_imagined():
    """Counted off the boxes: the scene boxes in each composite, times twelve rounds."""
    def count(arr):
        return sum(1 for part in _leaves(arr) if part.label.startswith(("scene(", "comparison(")))
    s0 = count(W.s0_arrangement(0.3, 0.2, (0.3, 0.4), 0.25))
    s6 = count(W.s6_arrangement((0.3, 0.4), 0.25))
    print(f"\n5 scenes a round: S0 {s0}, S6 {s6}; over twelve rounds S0 {12 * s0}, S6 {12 * s6} "
          f"(hand: S0 60 with five casts, S1 108, S6 72 with the top read)")
    assert (s0, s6) == (6, 12)


def _leaves(arr):
    """The boxes of a composite, by its label: the composites keep their factors' labels in theirs."""
    return [_Label(x) for x in _labels(arr.label)]


class _Label:
    def __init__(self, s):
        self.label = s


def _labels(s):
    import re
    return re.findall(r"(?:scene|comparison)\([0-9., ]+\)", s)


@pytest.mark.parametrize("sigma", NOISES)
def test_6_the_trigger_on_P3(sigma):
    rows = _s0("P3", sigma)
    casts = _casts("P3")
    i4, i6 = casts.index((0.3, 0.4)), casts.index((0.3, 0.6))
    bound = lambda t: 3 * sigma * np.sqrt(t)
    first = {}
    for name, i in (("cat 0.4", i4), ("cat 0.6", i6)):
        past = [t for t in range(1, 13) if abs(rows[t - 1]["sums"][i]) > bound(t)]
        first[name] = past[0] if past else None
        per = np.diff(np.concatenate([[0.0], [r["sums"][i] for r in rows[:12]]]))
        print(f"\n6 P3 sd {sigma} {name}: miss a round {np.round(per, 3).tolist()}, sum after 12 "
              f"{rows[11]['sums'][i]:.3f} against 3 sd {bound(12):.3f}, first past at round {first[name]}", end="")
    fire = [rows[t]["fire"] for t in range(1, 13)]          # the fire shown after t rounds is row t + 1's
    fired = [t for t, f in enumerate(fire, 1) if f]
    trust_end = np.round(rows[11]["trust"], 3).tolist()
    print(f"\n6 P3 sd {sigma}: trigger fires at rounds {fired}; trust at the end {trust_end}")
    if sigma == 0.1:
        assert first["cat 0.4"] is not None and first["cat 0.6"] is not None
        assert max(first.values()) <= 7 and fired and fired[0] <= 7
    else:
        assert first["cat 0.4"] is None and first["cat 0.6"] is None and not fired


@pytest.mark.parametrize("sigma", NOISES)
def test_7_the_trigger_on_P1(sigma):
    rows = _s0("P1", sigma)
    i = _casts("P1").index((0.3, 0.4))
    fired = [t for t in range(1, 13) if rows[t]["fire"]]
    print(f"\n7 P1 sd {sigma}: the truth's summed miss {max(abs(r['sums'][i]) for r in rows[:12]):.1e}, fires {fired}")
    assert max(abs(r["sums"][i]) for r in rows[:12]) == 0.0 and not fired


def test_8_read_one_way():
    for problem in ("P1", "P3"):
        for sigma in NOISES:
            d = max(np.abs(a["trust"] - b["trust"]).max()
                    for a, b in zip(_s0(problem, sigma), _s0(problem, sigma, trigger=False)))
            print(f"\n8 {problem} sd {sigma}: trust with the trigger vs without, worst {d:.1e}", end="")
            assert d == 0.0
    print()


@pytest.mark.parametrize("sigma", NOISES)
def test_9_the_mixture_miss_on_P3(sigma):
    rows = _s0("P3", sigma)
    hand = W.s0_by_hand("P3", sigma)
    fam = _casts("P3")
    # the mixture's miss each round: the trust before the round times each scene's miss that round
    q = np.full(6, 1 / 6)
    tot, path = 0.0, []
    for t in range(12):
        miss = hand[t]["sums"] - (hand[t - 1]["sums"] if t else 0.0)
        tot += float(q @ miss)
        path.append(round(tot, 3))
        q = rows[t]["trust"]
    print(f"\n9 P3 sd {sigma}: the mixture's summed miss {path}; 3 sd after 12 {3 * sigma * np.sqrt(12):.3f}")
    if sigma == 0.25:
        assert abs(tot) < 3 * sigma * np.sqrt(12)


@pytest.mark.parametrize("sigma", NOISES)
def test_7N_false_alarms_on_a_noisy_P1(sigma):
    """Arm N: the recording given bell noise of width sigma (the environment's schedule, an example box's second
    column), twenty seeds; the trigger fires in at most 2 of 20."""
    fired_seeds, worst = [], 0.0
    for seed in range(20):
        noise = np.random.default_rng(seed).normal(0.0, sigma, 12)
        rows = W.s0_run("P1", sigma, noise=noise)
        if seed == 0:
            hand = W.s0_by_hand("P1", sigma, noise=noise)
            worst = max(np.abs(r["trust"] - h["trust"]).max() for r, h in zip(rows, hand))
        fired = [t for t in range(1, 13) if rows[t]["fire"]]
        if fired:
            fired_seeds.append((seed, fired[0]))
    print(f"\n7N P1 noisy sd {sigma}: fired in {len(fired_seeds)} of 20 seeds (seed, first round) {fired_seeds}; "
          f"seed 0 arr vs numpy {worst:.1e}")
    assert worst < 1e-12 and len(fired_seeds) <= 2


# S6g, David's "generic snake" (predictions 10-12, added before its code).

@functools.lru_cache(maxsize=None)
def _s6g(problem, sigma):
    return W.s6g_run(problem, sigma)


def test_10_s6g_equals_the_hand_recurrence_and_ignores_the_start():
    worst, same = 0.0, 0.0
    for sigma in NOISES:
        for problem in ("P1", "P2"):
            rows, hand = _s6g(problem, sigma), W.s6g_by_hand(problem, sigma)
            worst = max(worst, max(max(np.abs(r[k] - h[k]).max() for k in ("snake", "cat")) for r, h in zip(rows, hand)))
        same = max(same, max(max(np.abs(r[k] - h[k]).max() for k in ("snake", "cat"))
                             for r, h in zip(_s6g("P1", sigma), _s6g("P2", sigma))))
    print(f"\n10 S6g arr vs numpy worst {worst:.1e}; P1 vs P2 {same:.1e}")
    assert worst < 1e-12 and same == 0.0


def test_11_s6g_scenes():
    import re
    n = len(re.findall(r"(?:scene|comparison)\(", W.s6g_arrangement((0.3, 0.4), 0.25).label))
    print(f"\n11 S6g scenes a round {n}, over twelve {12 * n}")
    assert n == 6


def test_12_s6g_settles_late():
    for sigma in NOISES:
        rows, hand = _s6g("P1", sigma), W.s6g_by_hand("P1", sigma)
        tops = [_s6_top(r) for r in rows]
        st = W.settled(tops, (0.3, 0.4))
        print(f"\n12 S6g sd {sigma}: settled {st}; tops {tops}; generic snake by round "
              f"{[round(h['abar'], 3) for h in hand]}; cat belief at the end {np.round(rows[-1]['cat'], 3).tolist()}", end="")
        assert st is not None and 1 < st <= 8 and tops[-1] == (0.3, 0.4)
    print()
