"""A coil moved past a coil (book plan P76 (f); ``notes/runs/moving-coil-2524b69e/RESULTS.md`` in the book's repo holds the
predictions, written first).

Every test runs the closed arrangement ``(coils_at . (tank || cart))`` through ``PhiK``; energies are read off its states
by ordinary code beside it. Everything is on ``R^n``. Claude Fable 5.2, session 2524b69e, 2026-11-10.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.interpretation import trivial_omega
from arrangements.moving_coil import field_energy, inductance, magnet_past_coil
from arrangements.pcK import PhiK

jax.config.update("jax_enable_x64", True)

_IN = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda op: (jnp.zeros(0), jnp.zeros(0))


def _unpack(s):
    return np.concatenate([np.asarray(x, float).ravel() for x in jax.tree_util.tree_leaves(s)])


def _readings(s, M0):
    q, x, p, l1, l2 = _unpack(s)
    lam = jnp.array([l1, l2])
    i = np.asarray(jnp.linalg.solve(inductance(1.0, 1.0, M0)(x), lam))
    H = float(field_energy(1.0, 1.0, M0)(lam, x))
    return dict(s=q, x=x, p=p, i2=i[1], KE=p * p / 2, PE=q * q / 2 + H, E=q * q / 2 + H + p * p / 2)


def _run(T, M0=0.5, tau=0.05, **kw):
    O = PhiK(magnet_past_coil(M0=M0, tau=tau, **kw))
    step = jax.jit(lambda s: O.with_state(s).run_one(_IN, _TRIV)[1])
    s = O.state
    out = [_readings(s, M0)]
    for _ in range(int(round(T / tau))):
        s = step(s)
        out.append(_readings(s, M0))
    return out


def test_label_and_first_step():
    arr = magnet_past_coil()
    assert arr.label == "(coils_at(L=1,1;M0=0.5) . (tank(1) || cart(1)))"
    O = PhiK(arr)
    plain = O.run_one(_IN, _TRIV)[1]
    jitted = jax.jit(lambda s: O.with_state(s).run_one(_IN, _TRIV)[1])(O.state)
    assert np.allclose(_unpack(plain), _unpack(jitted), atol=1e-14)


@pytest.mark.parametrize("x0", [-6.0, 0.5])
def test_a_still_coil_gets_nothing(x0):
    out = _run(10.0, x0=x0, p0=0.0)
    assert max(abs(o["s"]) for o in out) < 1e-12
    assert max(abs(o["i2"]) for o in out) < 1e-12
    assert abs(out[-1]["x"] - x0) < 1e-12


def test_moving_makes_current_and_the_cart_pays():
    out = _run(10.0)
    assert max(abs(o["s"]) for o in out) > 0.05
    approach = [o for o in out if o["x"] < -0.5]
    assert max(o["i2"] for o in approach) <= 0.0 and min(o["i2"] for o in approach) < -0.05
    assert approach[-1]["p"] < approach[0]["p"]
    lost = out[0]["KE"] - out[-1]["KE"]
    gained = out[-1]["PE"] - out[0]["PE"]
    assert lost > 0.1 and abs(lost - gained) < 1e-6


def test_energy_is_fourth_order_and_euler_drifts():
    band = {}
    for rule, tau in (("rk4", 0.1), ("rk4", 0.05), ("euler", 0.05)):
        E = np.array([o["E"] for o in _run(10.0, rule=rule, tau=tau)])
        band[rule, tau] = np.abs(E / E[0] - 1).max()
    assert 10 < band["rk4", 0.1] / band["rk4", 0.05] < 25
    assert band["euler", 0.05] > 100 * band["rk4", 0.05]


def test_no_shared_field_is_a_tensor():
    out = _run(10.0, M0=0.0)
    assert max(abs(o["s"]) for o in out) == 0.0
    assert max(abs(o["p"] - 1.5) for o in out) == 0.0
