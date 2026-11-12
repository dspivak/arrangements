"""Graph Laplacian on a directed graph via prism wiring (CDLM sec.graph_laplacian).

The closed system ``wire_G = R^{varphi_G}((Part_v)_v)`` (CDLM eqn.graph_wire) is
built by *genuine composition* in ``arr`` -- the prism wiring ``graph_wire``
(image of ``varphi_G`` under ``R^-``, CDLM eqn.prism_f) composed with the
per-vertex harmonic particles ``Part_v``. We check:

* the composite forcing ``-d sum_e (kappa/2)(q_tgt - q_src)^2`` (minus CDLM's
  eqn.graph_potential) EMERGES from the wiring -- it is not written by hand;
  CDLM's ``compose_seq`` summed potentials and the old test inspected ``arr.U``,
  while the book's composite carries only ``omega`` (notes/DESIGN.md §2), so the oracle
  is differentiated and compared with the forcing's ``Q``-component;
* the prism input map ``inpt f`` is a bijection on ``R^E`` (CDLM eqn.prism_f);
* the same graph with *phase* vertex parameters is the discrete graph-wave
  equation (CDLM prop.graph_laplacian, eqn.graph_laplacian_dynamics) and with
  *configuration* parameters the graph-heat equation (CDLM rmk.graph_heat), both
  against an independent Laplacian oracle -- one wiring, two arrangements (def.imc).

The test graph mixes in- and out-degrees and has a non-identity prism permutation,
so it exercises the broadcast (``outp f_v``) and routing (``inpt f``) non-trivially.

The book's Ch 3 ``ex.graph_laplacian`` (Claude Opus 5.5, 00f2d232, 2026-11-12): the triangle it draws, edges
``a : 1 -> 2``, ``b : 2 -> 3``, ``c : 1 -> 3`` with their own spring constants, built as the book builds it (three part
boxes, the wiring's dots, a spring box per edge, ``wiring.boxes_and_couplers``). Predictions before the code: the closed
inclination at the positions ``x`` is ``-L x`` for the book's matrix ``eqn.graph_laplacian_triangle``, whatever fills
the mass boxes (rod segments or masses), and the rod segments' step at durations ``1/4`` and ``1`` is
``x - tau eta L x``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi, closed_step
from arrangements.rvect import diagonal
from arrangements.wiring import boxes_and_couplers, compose_graph, graph_wire, trivial_parameter

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


# ---------------------------------------------------------------------------
# The book's Ch 3 ex.graph_laplacian: its triangle, a spring constant per edge.
# ---------------------------------------------------------------------------


def _box(P):
    """A box of the book's picture filled with a part reading out its position."""
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1, out_f=lambda q, m: q,
                       in_f=lambda q, m, n: _Z0, omega=zero_form, label="part")


def _spring(k):
    """``spring_e``: stateless, receiving ``(x_v, x_v')``, cost ``k/2 (x_v - x_v')^2``."""
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0, out_f=lambda q, m: _Z0,
                       in_f=lambda q, m, n: _Z0, omega=exact(lambda q, m, n: -0.5 * k * (m[0] - m[1]) ** 2),
                       label=f"spring({k:g})")


def test_book_triangle_loads_are_minus_its_laplacian():
    ka, kb, kc = 0.7, 1.9, 0.4
    edges = [(0, 1), (1, 2), (0, 2)]               # a : 1 -> 2, b : 2 -> 3, c : 1 -> 3
    L = np.array([[ka + kc, -ka, -kc], [-ka, ka + kb, -kb], [-kc, -kb, kb + kc]])   # eqn.graph_laplacian_triangle
    x = jnp.array([1.0, -0.5, 2.0])
    eta = 0.3
    fills = {
        "rod segments": [_box(configuration(diagonal(jnp.array([eta])), init=x[i:i + 1])) for i in range(3)],
        "masses": [_box(phase(diagonal(jnp.array([1.0])), init=(x[i:i + 1], jnp.zeros(1)))) for i in range(3)],
    }
    for name, boxes in fills.items():
        arr = boxes_and_couplers(boxes, edges, [_spring(ka), _spring(kb), _spring(kc)])
        assert arr.label.count("spring") == 3, name
        xi_q, _, _ = arr.omega(x, _Z0, _Z0)
        np.testing.assert_allclose(np.asarray(xi_q), -L @ np.asarray(x), atol=1e-12, err_msg=name)
    arr = boxes_and_couplers(fills["rod segments"], edges, [_spring(ka), _spring(kb), _spring(kc)])
    for tau in (0.25, 1.0):
        s1 = closed_step(arr, duration=tau)(arr.P.init)
        flat = np.concatenate([np.ravel(np.asarray(a)) for a in jax.tree_util.tree_leaves(s1)])
        np.testing.assert_allclose(flat, np.asarray(x) - tau * eta * L @ np.asarray(x), atol=1e-12)
