"""The feeding search's destination: the inverse of the cumulative curve, to rounding, with
its own derivative (book P109 (f), act (5); David, 2026-11-19: "yes as rec'd").

``feeding._destination`` halves ``[-9, 9]`` 52 times and carries the inverse's derivative by
``jax.custom_jvp`` (the implicit function theorem on ``C(m, x) = u sum(m)``). Checked here: the
value inverts the curve to rounding; the derivative in the draw and in each mass agrees with
finite differences of the halving. What runs: an ordinary function of the library, the circle
inside the memory's static map; no arrangement is stepped here.
"""
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

from arrangements.feeding import CENTRES, _cumulative, _destination, _search_map  # noqa: E402


def _masses():
    fresh = jnp.ones(len(CENTRES)).at[8:12].set(jnp.array([.05, 0., 0., .05]))
    usual = jnp.zeros(len(CENTRES)).at[4].set(.9).at[13].set(.7)
    return _search_map(usual, fresh)[1]


def _halve(masses, u):
    """The same 52 halvings, with no derivative rule: the finite differences' reference."""
    lo, hi = float(CENTRES[0]), float(CENTRES[-1])
    for _ in range(52):
        x = (lo + hi) / 2
        if float(_cumulative(masses, x)) < float(masses.sum()) * u:
            lo = x
        else:
            hi = x
    return (lo + hi) / 2


def test_the_destination_inverts_the_cumulative_curve_to_rounding():
    masses = _masses()
    for u in (.03, .2, .477, .5, .8, .97):
        x = _destination(masses, u)
        assert abs(float(_cumulative(masses, x)) / float(masses.sum()) - u) < 1e-12


def test_the_derivative_is_the_inverse_s_own():
    masses = _masses()
    worst = 0.
    for u in (.03, .2, .477, .5, .8, .97):
        gu = float(jax.grad(_destination, argnums=1)(masses, u))
        gm = np.asarray(jax.grad(_destination, argnums=0)(masses, u))
        fu = (_halve(masses, u + 1e-7) - _halve(masses, u - 1e-7)) / 2e-7
        e = 1e-6 * float(masses.sum())
        fm = np.array([(_halve(masses.at[i].add(e), u) - _halve(masses.at[i].add(-e), u)) / (2 * e)
                       for i in range(len(CENTRES))])
        assert gu > 0                                    # the halving alone would give 0
        worst = max(worst, abs(gu - fu) / abs(fu), np.abs(gm - fm).max() / np.abs(fm).max())
    assert worst < 1e-4, worst
