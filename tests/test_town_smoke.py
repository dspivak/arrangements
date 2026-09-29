"""P29 stage 1: analytic recurrence, actual curl, and Phi on the composite.

Prediction 1 in notes/P29-TOWN-PREDICTIONS-2026-09-28.md. This verifies a chosen
one-way physical model, not a criterion for which real institutions should exist.
Reasonable: unpriced harm leaves this plant's profit unchanged. Artificial: one
noiseless, quadratic harm law. At a cell's scale only the asymmetric coupling is
an analogy; no biological or social inference follows from its non-exactness.

Result (2026-09-28): all 3 checks passed; production follows its exact recurrence,
ordinary and automatic derivatives give mismatch 1.4, and Phi agrees. Prediction 1 held.
Full results: notes/P29-TOWN-RESULTS.md.
"""
from dataclasses import replace
import jax
import jax.numpy as jnp
import numpy as np
from arrangements.town import Settings, build, run
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega


def test_smoke_does_not_return_a_push_and_production_matches_closed_form():
    c = Settings()
    town = build(1, c)
    trajectory = run(town, 5)
    expected = c.benefit + (c.output0-c.benefit)*(1-c.dt)**np.arange(len(trajectory))
    np.testing.assert_allclose(town.value(trajectory, 'output')[:, 0], expected, atol=2e-13)
    other = build(1, replace(c, beta=7.0))
    np.testing.assert_allclose(town.value(trajectory, 'output'),
                               other.value(run(other, 5), 'output'), atol=1e-13)
    assert other.value(run(other, 1), 'coughing')[-1, 0] > 0


def test_actual_composite_has_the_predicted_cross_partial_mismatch():
    town = build(1)
    point = jnp.array([.7, .2])
    jac = np.asarray(jax.jacfwd(town.field)(point))
    assert jac[0, 1] == 0
    np.testing.assert_allclose(jac[1, 0]-jac[0, 1], 1.4, atol=1e-12)
    # Finite differences check ordinary derivatives, without AD conventions.
    step = 1e-5
    up = np.asarray(town.field(point+jnp.array([step, 0.])))
    down = np.asarray(town.field(point-jnp.array([step, 0.])))
    np.testing.assert_allclose((up[1]-down[1])/(2*step), 1.4, atol=1e-9)


def test_phi_agrees_with_the_product_integrator_on_the_closed_composite():
    town = build(1)
    phi = Phi(town.arr)
    state = town.arr.P.init
    points = [np.asarray(town.arr.P.position(state))]
    for _ in range(5):
        _, _, state = phi.with_state(state).run_one(
            (jnp.zeros(0), trivial_omega(0)),
            lambda _: (jnp.zeros(0), jnp.zeros(0)))
        points.append(np.asarray(town.arr.P.position(state)))
    np.testing.assert_allclose(points, run(town, .1), atol=1e-13)
