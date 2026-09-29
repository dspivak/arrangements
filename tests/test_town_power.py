"""P29 (e): the residents take the plant's power and push back on it; smoke returns nothing.

Predictions fixed before code: notes/P29-E-PREDICTIONS-2026-09-29.md.
The revenue moves from inside the plant's cost onto the power wire, as the residents' push,
so every trajectory is stage (d)'s; the power port returns `benefit`, the smoke port zero.
"""
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.town import Settings, build, run


@pytest.mark.parametrize("stage", [1, 2, 3, 4])
@pytest.mark.parametrize("k", [.1, 1.0])
def test_power_wire_reproduces_d(stage, k):
    s = Settings(clean_cost=k)
    d = run(build(stage, s, clean=True), 30.0)
    e = run(build(stage, s, clean=True, power_to_town=True), 30.0)
    assert d.shape == e.shape
    assert np.max(np.abs(d - e)) < 1e-12


def test_power_pushes_back_smoke_does_not():
    s = Settings()
    town = build(1, s, clean=True, power_to_town=True)
    # the residents' box as build() makes it; its one-form on (power, smoke) at visited states
    from arrangements.town import own_cost
    box = own_cost('coughing', [0.0], 2,
                   lambda q, e: .5*s.recovery*q[0]**2 - q[0]*s.beta*e[1]**2/2, s.dt,
                   push=lambda q, e: jnp.array([s.benefit, 0.0]))
    trajectory = run(town, 30.0)
    for row in trajectory[::150]:
        q = jnp.array([row[town.slots['coughing']][0]])
        x, c = row[town.slots['output']][0], row[town.slots['cleanliness']][0]
        e = jnp.array([x, (1-c)*x])
        _, _, returned = box.omega(q, jnp.zeros(0), e)
        assert float(returned[0]) == s.benefit
        assert float(returned[1]) == 0.0
