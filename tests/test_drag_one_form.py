"""Quadratic-drag 1-form (EXTENSION; coalgebra.quadratic_drag_kick / coalgebra.gyro).

The blog's air drag -- force ``~ -|v| v`` -- as a 1-form
``omega_drag(q,xi) = (|v| flat(v), 0) : Q^* (+) Q`` taken *per gyro* (``flat`` the
Euclidean flat; ``quadratic_drag_kick`` computes it in coordinates). We verify the
rmk.adam split that makes it a legitimate integrator:

* MONOIDAL over ``(+)`` -- each gyro drags on its own velocity (the non-negotiable
  compositionality of rmk.adam);
* NATURAL only over per-gyro orthogonal maps -- the norm and the flat both come from
  the Euclidean structure of a gyro block -- a "smaller Q", which rmk.adam explicitly
  sanctions; the third test shows the restriction is *real* (a non-orthogonal per-gyro
  map breaks naturality), so it is not a vacuous claim;
* and the drag dissipates energy in a free rollout.

Here the drag lives inside the ``gyro`` coalgebra's step (the TRICK realization);
the same term written as a non-exact forcing on ``T^*Q`` (the DIRECT one) is checked
against it in test_gyroscope.py.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import gyro, quadratic_drag_kick
from arrangements.functors import Phi
from arrangements.gyroscope import gyro_arrangement, make_config
from arrangements.interpretation import trivial_omega

from tests.chain import load_regress

_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _rot(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def _block_diag(mats):
    return np.asarray(jax.scipy.linalg.block_diag(*[jnp.asarray(m) for m in mats]))


def test_drag_is_monoidal_over_oplus():
    """Each gyro drags on its own velocity: F is block-diagonal, so F on the whole
    space is the concatenation of F on each 2-D gyro block -- monoidality over (+)."""
    rng = np.random.default_rng(0)
    V = 4
    v = jnp.asarray(rng.standard_normal(2 * V))
    whole = np.asarray(quadratic_drag_kick(v, 2))
    per_block = np.concatenate(
        [np.asarray(quadratic_drag_kick(v[2 * i : 2 * i + 2], 2)) for i in range(V)]
    )
    np.testing.assert_allclose(whole, per_block, atol=1e-12)


def test_drag_is_natural_over_block_orthogonal():
    """F(O v) = O F(v) for per-gyro orthogonal O (block O(2)): natural over the
    orthogonal subcategory -- exactly the restriction rmk.adam permits."""
    rng = np.random.default_rng(1)
    V = 4
    v = jnp.asarray(rng.standard_normal(2 * V))
    O = jnp.asarray(_block_diag([_rot(t) for t in rng.uniform(0, 2 * np.pi, V)]))
    lhs = np.asarray(quadratic_drag_kick(O @ v, 2))
    rhs = np.asarray(O @ quadratic_drag_kick(v, 2))
    np.testing.assert_allclose(lhs, rhs, atol=1e-12)


def test_drag_is_NOT_natural_over_nonorthogonal():
    """A non-orthogonal per-gyro map (here area-preserving but not an isometry) breaks
    naturality -- the subcategory restriction is real, not vacuous, because |v_i| is
    not scale-invariant. This is why drag lives over the *orthogonal* subcategory."""
    rng = np.random.default_rng(2)
    V = 3
    v = jnp.asarray(rng.standard_normal(2 * V))
    D = jnp.asarray(_block_diag([np.diag([2.0, 0.5]) for _ in range(V)]))  # det 1, not orthogonal
    lhs = np.asarray(quadratic_drag_kick(D @ v, 2))
    rhs = np.asarray(D @ quadratic_drag_kick(v, 2))
    assert not np.allclose(lhs, rhs, atol=1e-6)


def _drag_network(cfg, kappa, grav, drag):
    """The 2x3 gyro network with the drag term alone in its ``gyro`` parameter."""
    return Phi(gyro_arrangement(
        cfg, inv_mass=jnp.ones(cfg.V), kappa=kappa, grav_k=grav,
        parameter=lambda Q: gyro(Q, damping=0.0, gamma=0.0, J=None, drag=drag, gyro_block=2),
    ))


def _free_rollout(O, cfg, q0, xi0, steps):
    state = (q0, xi0)
    qs, xis = [q0], [xi0]
    zero_force = jnp.zeros(cfg.n_in)
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(_IN_POS, lambda op: (jnp.zeros(cfg.n_out), zero_force))
        qs.append(state[0]); xis.append(state[1])
    return np.stack([np.asarray(x) for x in qs]), np.stack([np.asarray(x) for x in xis])


def test_drag_dissipates_energy():
    """A free (unforced) gyro network with drag > 0 bleeds out kinetic + potential
    energy -- the dissipative content of the 1-form, run through ``Phi`` with the
    ``gyro`` parameter."""
    cfg = make_config(rows=2, cols=3)  # 6 gyros
    V = cfg.V
    kappa = jnp.full(len(cfg.edges), 0.2)
    grav = jnp.array(0.1)
    O = _drag_network(cfg, kappa, grav, 0.3)

    rng = np.random.default_rng(0)
    q0 = jnp.asarray(0.5 * rng.standard_normal(2 * V))
    xi0 = jnp.asarray(0.8 * rng.standard_normal(2 * V))
    E = jnp.asarray(cfg.edges)

    def energy(q, xi):
        qv = q.reshape(V, 2)
        springs = 0.5 * jnp.sum(kappa * jnp.sum((qv[E[:, 0]] - qv[E[:, 1]]) ** 2, axis=1))
        return float(0.5 * jnp.sum(xi ** 2) + springs + 0.5 * grav * jnp.sum(qv ** 2))

    qs, xis = _free_rollout(O, cfg, q0, xi0, 60)
    H0 = energy(jnp.asarray(qs[0]), jnp.asarray(xis[0]))
    assert energy(jnp.asarray(qs[-1]), jnp.asarray(xis[-1])) < 0.7 * H0  # drag bled energy out


def test_regression_matches_cdlm():
    """The CDLM drag rollout (``regress/gyroscope.npz``) is reproduced to rtol 1e-10."""
    R = load_regress("gyroscope")
    kappa0, grav0, drag = R["drag_params"]
    cfg = make_config(rows=2, cols=3)
    O = _drag_network(cfg, jnp.full(len(cfg.edges), kappa0), jnp.array(grav0), drag)
    qs, xis = _free_rollout(O, cfg, jnp.asarray(R["drag_q"][0]), jnp.asarray(R["drag_xi"][0]),
                            R["drag_q"].shape[0] - 1)
    np.testing.assert_allclose(qs, R["drag_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(xis, R["drag_xi"], rtol=1e-10, atol=1e-12)
