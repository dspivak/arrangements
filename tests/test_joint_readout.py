"""The joint readout (book plan P73 (b); ``arrangements/joint_readout.py``).

Predictions, written before the code in the book's ``notes/runs/p73-b-56518228/PREDICTIONS.md``; results and what
failed in ``RESULTS.md`` beside it. Each test below is one prediction at the settings the viewer page uses
(``scenes.JR_*``), except ``test_p75_rebuilt``, which uses p75's.

* P0: the p75 model rebuilt (redundant parts, fixed compositeness) gives p75's numbers.
* P1: a prime-maker for ``P`` at rest reads out ``P - 1 + kappa(separation)``: rounded, ``P`` iff the parts hear each
  other at more than half strength.
* P2: with one part far, it reads out ``P - 1``, where p75's finder read ``P``.
* P3/P5: the composite-makers are broken up and read out another composite; the prime-makers do not move.
* P4: a loose prime-maker is pulled together when its separation is below about 1.3, and comes apart above.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np
import pytest

from arrangements import joint_readout as jr
from arrangements.scenes import JR_ALPHA, JR_K, JR_MOB, JR_MS, JR_STEPS, JR_TAU

TAU = JR_TAU


def kab(d):
    return float(np.exp(-d ** 2 / (2 * TAU)))


def maker(N, sep, mobility=JR_MOB):
    s0 = (N - 1) + kab(sep)
    parts = [jr.piece_part(N - 1, s0, k=JR_K, mobility=JR_MS, label=f"{N}{x}") for x in "ab"]
    return jr.finder(parts, tau=TAU, placements=[sep / 2, -sep / 2], mobility=mobility, label=str(N))


def value(m):
    return float(m[0] / m[1])


def run(finders, outer, steps=JR_STEPS, positions=None, fixed_c=None):
    S = jr.scene(finders, tau=TAU, positions=positions or [0.3] * len(finders), mobility=outer, alpha=JR_ALPHA,
                 fixed_c=fixed_c)
    return S, jr.read(S, jr.trace(S, steps)[-1])


def test_p75_rebuilt():
    Ns, c = [5, 11, 57, 91], [0.0, 0.0, 1 / 3, 1 / 7]
    fs = [jr.finder([jr.redundant_part(N, f"{N}a"), jr.redundant_part(N, f"{N}b")], tau=TAU,
                    placements=[0.3, -0.3], mobility=0.3, label=str(N)) for N in Ns]
    S = jr.scene(fs, tau=TAU, positions=[0.3] * 4, mobility=0.3, fixed_c=c)
    r = jr.read(S, jr.trace(S, 400)[-1])
    assert np.allclose([float(r["inner"][f][0]) for f in range(4)], [0.3, 0.3, 1.01, 0.95], atol=0.01)
    assert np.allclose([float(x) for x in r["outer"]], [0.3, 0.3, 3.19, 2.92], atol=0.01)


@pytest.mark.parametrize("P", [5, 11])
@pytest.mark.parametrize("sep", [0.6, 1.0, 1.1, 2.0])
def test_hand_case(P, sep):
    # the finder alone: no listener cost (fixed_c = 0), whose push would move the parts' values
    S, r = run([maker(P, sep, mobility=0.0)], outer=0.0, steps=50, fixed_c=[0.0])
    v = value(r["readout"][0])
    assert v == pytest.approx(P - 1 + kab(sep), abs=1e-6)
    assert jr.is_prime(round(v)) == (kab(sep) > 0.5)


@pytest.mark.parametrize("P", [5, 11])
def test_one_part_far_is_not_enough(P):
    s0 = (P - 1) + kab(2.7)
    parts = [jr.piece_part(P - 1, s0, k=JR_K, mobility=JR_MS, label=x) for x in "ab"]
    f = jr.finder(parts, tau=TAU, placements=[0.3, 3.0], mobility=0.0, label=str(P))
    _, r = run([f], outer=0.0, steps=50)
    assert round(value(r["readout"][0])) == P - 1
    red = jr.finder([jr.redundant_part(P, "a"), jr.redundant_part(P, "b")], tau=TAU, placements=[0.3, 3.0],
                    mobility=0.0, label="p75")
    assert value(red[0].out_f(red[0].P.position(red[0].P.init), jr.EMPTY)) == pytest.approx(P)


@pytest.mark.parametrize("outer", [JR_MOB, 0.0])
def test_scatter(outer):
    S, r = run([maker(5, 0.6), maker(11, 0.6), maker(57, 0.6), maker(91, 0.6)], outer=outer)
    reads = [round(value(m)) for m in r["readout"]]
    assert reads == [5, 11, 56, 90]
    seps = [float(x[0] - x[1]) for x in r["inner"]]
    assert seps[:2] == pytest.approx([0.6, 0.6], abs=1e-6)
    assert min(seps[2:]) > (1.4 if outer else 5.0)


def test_loose_prime_maker():
    ends = []
    for sep in (1.0, 1.2, 1.3, 1.4, 1.5):
        _, r = run([maker(13, sep)], outer=JR_MOB)
        ends.append(round(value(r["readout"][0])))
    assert ends == [13, 13, 13, 12, 12]
