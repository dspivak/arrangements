"""Graph Laplacian on a directed graph via prism wiring (CDLM sec.graph_laplacian).

The closed system ``wire_G = R^{varphi_G}((Part_v)_v)`` (CDLM eqn.graph_wire) is
built by *genuine composition* in ``arr`` -- the prism wiring ``graph_wire``
(image of ``varphi_G`` under ``R^-``, CDLM eqn.prism_f) composed with the
per-vertex harmonic particles ``Part_v``. We check:

* the composite forcing ``-d sum_e (kappa/2)(q_tgt - q_src)^2`` (minus CDLM's
  eqn.graph_potential) EMERGES from the wiring -- it is not written by hand;
  CDLM's ``compose_seq`` summed potentials and the old test inspected ``arr.U``,
  while the book's composite carries only ``omega`` (DESIGN.md §2), so the oracle
  is differentiated and compared with the forcing's ``Q``-component;
* the prism input map ``inpt f`` is a bijection on ``R^E`` (CDLM eqn.prism_f);
* the same graph with *phase* vertex parameters is the discrete graph-wave
  equation (CDLM prop.graph_laplacian, eqn.graph_laplacian_dynamics) and with
  *configuration* parameters the graph-heat equation (CDLM rmk.graph_heat), both
  against an independent Laplacian oracle -- one wiring, two arrangements (def:arr).

The test graph mixes in- and out-degrees and has a non-identity prism permutation,
so it exercises the broadcast (``outp f_v``) and routing (``inpt f``) non-trivially.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi
from arrangements.wiring import compose_graph, graph_wire

from tests.chain import IN_POS, conf_flat, conf_state, load_regress, phase_flat, phase_state

# directed triangle 0->1->2->0 with an extra chord 0->2:
#   out-degrees (2,1,1), in-degrees (1,1,2); the 0--2 bond is doubled.
V = 3
EDGES = [(0, 1), (1, 2), (2, 0), (0, 2)]

_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))
_Z0 = jnp.zeros(0)


def _laplacian(num_vertices, edges):
    """Independent oracle: combinatorial (symmetrized) Laplacian, +1 per directed edge."""
    L = np.zeros((num_vertices, num_vertices))
    for (s, t) in edges:
        L[s, s] += 1.0
        L[t, t] += 1.0
        L[s, t] -= 1.0
        L[t, s] -= 1.0
    return L


def _conf_step(O, q):
    _, _, new_state = O.with_state(conf_state(q)).run_one(IN_POS, _TRIV)
    return conf_flat(new_state, V)


def _phase_step(O, q, p):
    _, _, new_state = O.with_state(phase_state(q, p)).run_one(IN_POS, _TRIV)
    return phase_flat(new_state, V)


def test_graph_forcing_emerges_from_wiring():
    """Minus CDLM eqn.graph_potential: the closed forcing is ``-d`` of the edge sum,
    produced by ``compose_seq``'s pullback-and-sum."""
    m, kappa = 1.3, 2.1
    arr = compose_graph(V, EDGES, m, kappa)
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)  # closed
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(V))
        xi_q, xi_m, xi_n = arr.omega(q, _Z0, _Z0)
        assert xi_m.shape == (0,) and xi_n.shape == (0,)
        want = -jax.grad(lambda q: sum(0.5 * kappa * (q[t] - q[s]) ** 2 for (s, t) in EDGES))(q)
        np.testing.assert_allclose(np.asarray(xi_q), np.asarray(want), atol=1e-10)


def test_prism_routing_is_a_bijection():
    """CDLM eqn.prism_f: ``inpt f : sum in(v) -> sum out(v)`` is a bijection on R^E, so the
    prism routes the |E| out-ports onto the |E| in-ports with nothing lost or doubled."""
    wire = graph_wire(V, EDGES)
    E = len(EDGES)
    assert (wire.out_dim_M, wire.in_dim_M, wire.out_dim_N, wire.in_dim_N) == (E, E, 0, 0)
    m_out = jnp.arange(E, dtype=float)  # tag each out-port by its position
    routed = np.asarray(wire.in_f(_Z0, m_out, _Z0))
    assert sorted(routed.tolist()) == list(range(E))  # a permutation of {0,...,E-1}


def test_graph_wave_is_laplacian_dynamics():
    """CDLM prop.graph_laplacian / eqn.graph_laplacian_dynamics: m * ddot q = -kappa * L q(center)."""
    m, kappa = 1.5, 0.9
    O = Phi(compose_graph(V, EDGES, m, kappa, parameter=phase))
    L = _laplacian(V, EDGES)

    rng = np.random.default_rng(42)
    q, p = jnp.asarray(rng.standard_normal(V)), jnp.zeros(V)
    traj = [np.asarray(q)]
    for _ in range(12):
        q, p = _phase_step(O, q, p)
        traj.append(np.asarray(q))
    a = np.stack(traj)
    for t in range(len(a) - 2):
        ddot_q = a[t + 2] - 2.0 * a[t + 1] + a[t]
        residual = m * ddot_q + kappa * (L @ a[t + 1])
        np.testing.assert_allclose(residual, np.zeros(V), atol=1e-10)


def test_graph_heat_is_laplacian_descent():
    """CDLM rmk.graph_heat: the SAME wiring with *configuration* vertex parameters
    descends by q(t+1) = q(t) - (kappa/m) L q(t)."""
    m, kappa = 1.0, 0.2
    O = Phi(compose_graph(V, EDGES, m, kappa, parameter=configuration))
    L = _laplacian(V, EDGES)

    rng = np.random.default_rng(7)
    q = jnp.asarray(rng.standard_normal(V))
    for _ in range(50):
        q_prev = np.asarray(q)
        q = _conf_step(O, q)
        want = q_prev - (kappa / m) * (L @ q_prev)
        np.testing.assert_allclose(np.asarray(q), want, atol=1e-10)


def test_regression_matches_cdlm():
    """``regress/misc.npz``: the graph-wave and graph-heat trajectories are CDLM's
    (states unchanged; the sign moved from the integrator into the forcing)."""
    R = load_regress("misc")

    m, kappa, T = float(R["gl_wave_params"][0]), float(R["gl_wave_params"][1]), int(R["gl_wave_params"][2])
    O = Phi(compose_graph(V, EDGES, m, kappa, parameter=phase))
    q, p = jnp.asarray(R["gl_wave_q"][0]), jnp.asarray(R["gl_wave_p"][0])
    qs, ps = [np.asarray(q)], [np.asarray(p)]
    for _ in range(T):
        q, p = _phase_step(O, q, p)
        qs.append(np.asarray(q)); ps.append(np.asarray(p))
    np.testing.assert_allclose(np.stack(qs), R["gl_wave_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(np.stack(ps), R["gl_wave_p"], rtol=1e-10, atol=1e-12)

    m2, kappa2, T2 = float(R["gl_heat_params"][0]), float(R["gl_heat_params"][1]), int(R["gl_heat_params"][2])
    O2 = Phi(compose_graph(V, EDGES, m2, kappa2, parameter=configuration))
    q = jnp.asarray(R["gl_heat_q"][0])
    traj = [np.asarray(q)]
    for _ in range(T2):
        q = _conf_step(O2, q)
        traj.append(np.asarray(q))
    np.testing.assert_allclose(np.stack(traj), R["gl_heat_q"], rtol=1e-10, atol=1e-12)
