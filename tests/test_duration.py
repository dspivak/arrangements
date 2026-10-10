"""The duration input (book P9.5; Ch 3 ``def.duration_integrator``).

An integrator is handed, with each covector, how long it holds it: ``u : S -> (cot(Q) (x) y^{R>=0})(S)``,
and parts side by side share one duration (``prop.duration_parameters``). In the library a part's
``timed_step(s, xi, duration)`` is that law, ``step_over`` calls it, and ``Phi``'s fiber, ``PCMorphism``'s
composites, ``run_one`` and ``closed_step`` carry a duration to every box; ``duration=None`` is the unit
step, the path every run took before durations existed.

Predictions, written before the code (book `notes/runs/p95-duration-33faae02/RESULTS.md`):
1. every existing test passes unchanged;
2. one step of a composite of configuration parts held for ``tau`` moves the state ``tau`` times as far
   as the unit step, since the covector is read before the step;
3. at ``tau = 0`` it does not move;
4. Euler is first order: two half steps differ from one unit step by ``O(tau^2)``, and over a fixed
   elapsed time the gap shrinks about in half when the duration is halved;
5. a composite with a part that has no duration law, handed a duration, raises;
6. ``Phi`` of a composite and the composite of the ``Phi``s agree at a duration other than 1.

The rod is Ch 3's grouped rod: three pieces of heat capacity 4, joined by conductance 1, insulated at
both ends (held ends with no spring), from ``(8, 4, 0)``.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi, closed_step
from arrangements.rvect import diagonal
from arrangements.wiring import held_chain

from tests.chain import IN_POS, HeldChain, harmonic_particle

NO_DIR = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def grouped_rod(parameter=configuration):
    """Ch 3's grouped rod as ``wiring.held_chain``: the first piece has no spring to the left end and the
    right end none to the last piece, so both ends are insulated; capacity 4 is ``sharp = 1/4``."""
    parts = [harmonic_particle(parameter, 4.0, 0.0)] + [harmonic_particle(parameter, 4.0, 1.0)] * 2
    return held_chain(parts, 0.0, parameter)


class Rod(HeldChain):
    """``HeldChain``'s state conversions for the grouped rod, stepped at a duration."""

    def __init__(self, parameter=configuration):
        self.K, self.two_stage = 3, False
        self.arr = grouped_rod(parameter)
        self.phase = False
        self._steps = {}

    def at(self, q, tau):
        if tau not in self._steps:
            self._steps[tau] = closed_step(self.arr, duration=tau)
        return self.flat(self._steps[tau](self.state(q)))


def test_unit_step_is_the_books():
    """The grouped rod's unit step, (8,4,0) to (7,4,1), with and without an explicit duration 1."""
    rod = Rod()
    q0 = jnp.array([8.0, 4.0, 0.0])
    np.testing.assert_allclose(rod.at(q0, None), [7.0, 4.0, 1.0], atol=1e-12)
    np.testing.assert_allclose(rod.at(q0, 1.0), [7.0, 4.0, 1.0], atol=1e-12)


def test_two_half_steps_are_the_books():
    """Two steps held for 1/2 each: (7.5, 4, 0.5), then Ch 3's (7.0625, 4, 0.9375)."""
    rod = Rod()
    q = rod.at(jnp.array([8.0, 4.0, 0.0]), 0.5)
    np.testing.assert_allclose(q, [7.5, 4.0, 0.5], atol=1e-12)
    np.testing.assert_allclose(rod.at(q, 0.5), [7.0625, 4.0, 0.9375], atol=1e-12)


@pytest.mark.parametrize("tau", [0.0, 0.1, 0.5, 2.0])
def test_one_step_scales_with_duration(tau):
    """Predictions 2 and 3: the move is ``tau`` times the unit move, and nothing at ``tau = 0``."""
    rod = Rod()
    q0 = jnp.array([3.0, -1.0, 2.5])
    unit = rod.at(q0, None) - q0
    np.testing.assert_allclose(rod.at(q0, tau) - q0, tau * unit, atol=1e-12)


def test_first_order_in_duration():
    """Prediction 4: over elapsed time 1, the gap to a fine run roughly halves as the duration halves."""
    rod = Rod()
    q0 = jnp.array([8.0, 4.0, 0.0])

    def run(n):
        q = q0
        for _ in range(n):
            q = rod.at(q, 1.0 / n)
        return q

    fine = run(256)
    gaps = [float(jnp.max(jnp.abs(run(n) - fine))) for n in (4, 8, 16)]
    for a, b in zip(gaps, gaps[1:]):
        assert 1.6 < a / b < 2.4, gaps
    # and the total heat is held at every duration (the forces are a difference across each joint)
    np.testing.assert_allclose(jnp.sum(run(4)), 12.0, atol=1e-12)


def test_part_without_a_duration_law_raises():
    """Prediction 5: the phase part has no duration law yet, so a rod of phase parts handed one raises."""
    arr = grouped_rod(phase)
    assert arr.P.timed_step is None
    with pytest.raises(TypeError, match="no duration law"):
        Phi(arr).run_one(IN_POS, NO_DIR, 0.5)
    Phi(arr).run_one(IN_POS, NO_DIR)  # the unit step still runs


def test_phi_composes_at_a_duration():
    """Prediction 6: ``parallel`` of two ``Phi``s hands one duration to both factors and agrees with stepping
    each alone at that duration (``Phi`` of a wired composite is the rod tests above)."""
    a = harmonic_particle(configuration, 2.0, 1.0)
    b = harmonic_particle(configuration, 5.0, 3.0)
    Oa, Ob = Phi(a), Phi(b)
    sa, sb = jnp.array([1.5]), jnp.array([-0.5])
    Oa, Ob = Oa.with_state(sa), Ob.with_state(sb)
    tau = 0.3
    pos = IN_POS

    def dir_from(_o):
        return (jnp.zeros(1), jnp.array([0.25]))

    _, _, na = Oa.run_one(pos, dir_from, tau)
    _, _, nb = Ob.run_one(pos, dir_from, tau)
    _, _, nab = Oa.parallel(Ob).run_one((pos, pos), lambda _o: (dir_from(None), dir_from(None)), tau)
    np.testing.assert_allclose(nab[0], na, atol=1e-12)
    np.testing.assert_allclose(nab[1], nb, atol=1e-12)
    # and the step is the Euler family: s + tau * sharp(xi), xi = -kappa (s - y)
    np.testing.assert_allclose(na, sa + tau * (1 / 2.0) * (-1.0 * (sa - 0.25)), atol=1e-12)
    np.testing.assert_allclose(nb, sb + tau * (1 / 5.0) * (-3.0 * (sb - 0.25)), atol=1e-12)
