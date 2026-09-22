"""R^d generalization of the prism graph-wiring (sec.graph_laplacian; Phase 3).

The paper's prism wiring (``compose_graph``) couples scalar (R^1) vertices; this is
its generalization to R^vdim vertices plus an optional on-site potential, so a network
of 2-D gyros is built BY COMPOSITION -- closing the audit's "U written directly, not
built by the graph-wiring construction" gap. We check that:

* the R^2 graph-Laplacian forcing ``-d sum_e (kappa/2)|q_tgt - q_src|^2`` EMERGES from
  the prism wiring on a hex topology, matching an independent hand-written oracle (it
  is *not* the forcing we write -- it comes out of ``compose_seq``'s pullback-and-sum);
* an on-site potential per vertex is added on top;
* the scalar case (vdim=1) is unchanged (backward compatibility).

CDLM's ``compose_seq`` summed potentials, so the old test inspected ``arr.U``; the
book's composite carries only the forcing ``omega`` (DESIGN.md §2), so here the
oracle is differentiated and compared with the forcing's ``Q``-component, in the
book's sign ``omega = -dU`` (DESIGN.md §3).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.gyroscope import hex_graph
from arrangements.wiring import compose_graph

from tests.chain import load_regress

_Z0 = jnp.zeros(0)


def _spring_oracle(q, edges, kappa, vdim):
    """Independent oracle: sum_e (kappa/2)|q_tgt - q_src|^2 (Euclidean per edge)."""
    qv = q.reshape(-1, vdim)
    return sum(0.5 * kappa * jnp.sum((qv[t] - qv[s]) ** 2) for (s, t) in edges)


def _forcing_Q(arr, q):
    """The composite forcing's ``Q``-component at ``q`` (the other two are on ``R^0``)."""
    xi_q, xi_m, xi_n = arr.omega(q, _Z0, _Z0)
    assert xi_m.shape == (0,) and xi_n.shape == (0,)
    return np.asarray(xi_q)


def test_r2_graph_laplacian_emerges_on_hex():
    """The 2-D spring coupling is produced by compose_graph (vdim=2), not hand-written."""
    edges, in_g, out_g = hex_graph(3, 3)
    V = 9
    m, kappa = 1.3, 0.7
    arr = compose_graph(V, edges, m, kappa, vdim=2)
    # closed system, parameter R^{2V}, mass-sharp -- everything from composition:
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)
    assert arr.P.Q_dim == 2 * V
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(2 * V))
        want = -jax.grad(lambda q: _spring_oracle(q, edges, kappa, 2))(q)
        np.testing.assert_allclose(_forcing_Q(arr, q), np.asarray(want), atol=1e-10)


def test_onsite_potential_is_added():
    """An on-site term (a per-gyro well) adds to the emergent spring potential."""
    edges, _, _ = hex_graph(2, 3)
    V = 6
    m, kappa, g = 1.0, 0.5, 0.3
    onsite = lambda qv: 0.5 * g * jnp.sum(qv ** 2)
    arr = compose_graph(V, edges, m, kappa, vdim=2, onsite=onsite)
    rng = np.random.default_rng(1)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(2 * V))

        def oracle(q):
            qv = q.reshape(V, 2)
            return _spring_oracle(q, edges, kappa, 2) + sum(0.5 * g * jnp.sum(qv[v] ** 2) for v in range(V))

        want = -jax.grad(oracle)(q)
        np.testing.assert_allclose(_forcing_Q(arr, q), np.asarray(want), atol=1e-10)


def test_scalar_graph_unchanged():
    """vdim=1 (the default) still gives the paper's scalar graph Laplacian."""
    edges = [(0, 1), (1, 2), (2, 0), (0, 2)]
    V = 3
    m, kappa = 1.3, 2.1
    arr = compose_graph(V, edges, m, kappa)  # default vdim=1, onsite=None
    rng = np.random.default_rng(2)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(V))
        want = -jax.grad(lambda q: sum(0.5 * kappa * (q[t] - q[s]) ** 2 for (s, t) in edges))(q)
        np.testing.assert_allclose(_forcing_Q(arr, q), np.asarray(want), atol=1e-10)


def test_regression_matches_cdlm():
    """CDLM's composite potential gradients ``dU`` (``regress/gyroscope.npz``) equal
    minus the composite forcing, to rtol 1e-10, on all three graphs."""
    R = load_regress("gyroscope")
    hex33, _, _ = hex_graph(3, 3)
    hex23, _, _ = hex_graph(2, 3)
    cases = {
        "hex": compose_graph(9, hex33, 1.3, 0.7, vdim=2),
        "on": compose_graph(6, hex23, 1.0, 0.5, vdim=2, onsite=lambda qv: 0.5 * 0.3 * jnp.sum(qv ** 2)),
        "sc": compose_graph(3, [(0, 1), (1, 2), (2, 0), (0, 2)], 1.3, 2.1),
    }
    for tag, arr in cases.items():
        got = np.stack([_forcing_Q(arr, jnp.asarray(q)) for q in R[f"vg_{tag}_q"]])
        np.testing.assert_allclose(got, -R[f"vg_{tag}_dU"], rtol=1e-10, atol=1e-12)
