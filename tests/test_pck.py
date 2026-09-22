"""General ``pc^(K)`` datatype + RK4 as ``pc^(4)`` (pcK.py, rk4.py; rmk.multistage).

``PCMorphismK`` is the K-fold substitution ``[p,q]^{∘K}``, the general case of the
two-stage ``PCMorphism2``. These tests exercise:

* the *general* structure -- ``parallel`` and ``pre_static`` compose correctly over
  the K rounds (here K = 4, RK4 instances);
* **subsumption** -- ``K = 1`` reproduces the single-stage ``Phi`` of the
  configuration arrangement (pc), and ``K = 2`` reproduces leapfrog (``pc^(2)``), so
  the general machinery contains both;
* **RK4 is really RK4** -- one macro-tick equals a hand-written RK4 step, and the
  global error of the staged ``PhiK`` falls like ``h^4`` (halving ``h`` cuts the
  error by ~16x). The ``h^4`` rate is the falsifiable proof that the four ``pc^(4)``
  rounds carry the right intermediate positions and Butcher weights;
* the CDLM RK4 trajectory (``regress/multistage.npz``) is reproduced to rtol 1e-10.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import CotCoalgK, configuration, iterate
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.leapfrog import leapfrog
from arrangements.pc2 import Phi2
from arrangements.pcK import PCMorphismK, PhiK
from arrangements.polynomial import identity_poly_map
from arrangements.rk4 import rk4
from arrangements.rvect import diagonal, euclidean

from tests.chain import load_regress

_IN_POS0 = (jnp.zeros(0), trivial_omega(0))  # closed position
_IN_DIR0 = (jnp.zeros(0), jnp.zeros(0))       # closed direction
_TRIV = lambda op: _IN_DIR0


def _closed(P, k):
    """A closed (I -> I) arrangement with parameter ``P`` and cost ``k(q)``."""
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -k(q)),
        label="closed",
    )


def _well(A, parameter):
    """A quadratic well ``k(q) = 1/2 q.A.q`` with Euclidean sharp.

    The configuration/RK4 flow is ``qdot = -A q`` (sharp = I), whose exact solution
    ``q(t) = e^{-At} q0`` lets us measure the integrator's order against a closed form.
    """
    A = jnp.asarray(A)
    return _closed(parameter(euclidean(A.shape[0], 1.0)), lambda q: 0.5 * q @ (A @ q))


def _rk4_well(A, h=0.1):
    return PhiK(_well(A, lambda Q: rk4(Q, h)))


def _oscillator(omega2, parameter):
    """A closed harmonic oscillator: Q = R, mass 1, cost (omega2/2) q^2 (for leapfrog)."""
    return _closed(parameter(diagonal(jnp.array([1.0]))), lambda q: 0.5 * omega2 * q[0] ** 2)


# ---------------------------------------------------------------------------
# Datatype basics.
# ---------------------------------------------------------------------------


def test_phik_rk4_is_a_general_pcK_morphism_with_K_4():
    """RK4 is an instance of the general datatype, with four interaction rounds."""
    O = _rk4_well(jnp.eye(2))
    assert isinstance(O, PCMorphismK)
    assert O.K == 4


def test_run_one_returns_K_out_positions():
    """A closed macro-tick emits one out-position per round (K = 4)."""
    O = _rk4_well(jnp.eye(2), h=0.1)
    out_poss, _new_state = O.run_one(_IN_POS0, _TRIV)
    assert len(out_poss) == 4


# ---------------------------------------------------------------------------
# Subsumption: K = 1 is pc, K = 2 is leapfrog.
# ---------------------------------------------------------------------------


def test_K1_reproduces_phi():
    """The K = 1 case of the general machinery IS the single-stage pc (``Phi``)."""
    A_mat = jnp.diag(jnp.array([1.0, 2.0, 4.0]))
    A = PhiK(_well(A_mat, lambda Q: iterate(configuration(Q), 1)))
    B = Phi(_well(A_mat, configuration))

    sa = sb = jnp.array([1.0, -0.5, 0.3])
    for _ in range(25):
        _outs, sa = A.with_state(sa).run_one(_IN_POS0, _TRIV)
        _op, _od, sb = B.with_state(sb).run_one(_IN_POS0, _TRIV)
    np.testing.assert_allclose(np.asarray(sa), np.asarray(sb), atol=1e-12)


def _leapfrog_K(Q):
    """Leapfrog (velocity Verlet) re-expressed as a two-round ``CotCoalgK``."""

    def advance1(s, xi_Q1):
        q, xi = s
        xi_half = xi + 0.5 * xi_Q1
        return (q + Q.apply_sharp(q, xi_half), xi_half)

    def finish(mid, xi_Q2):
        q2, xi_half = mid
        return (q2, xi_half + 0.5 * xi_Q2)

    return CotCoalgK(
        Q_dim=Q.dim,
        init=(jnp.zeros(Q.dim), jnp.zeros(Q.dim)),
        reads=(lambda s: s[0], lambda mid: mid[0]),
        advances=(advance1, finish),
        label="leapfrog-K2",
    )


def test_K2_reproduces_leapfrog():
    """The K = 2 case reproduces leapfrog exactly -- the same step as pc^(2) ``Phi2``."""
    A = PhiK(_oscillator(0.8, _leapfrog_K))
    B = Phi2(_oscillator(0.8, leapfrog))

    sa = sb = (jnp.array([1.0]), jnp.array([0.3]))
    for _ in range(30):
        _outs, sa = A.with_state(sa).run_one(_IN_POS0, _TRIV)
        _op1, _op2, sb = B.with_state(sb).run_one(_IN_POS0, _TRIV)
    np.testing.assert_allclose(np.asarray(sa[0]), np.asarray(sb[0]), atol=1e-12)
    np.testing.assert_allclose(np.asarray(sa[1]), np.asarray(sb[1]), atol=1e-12)


# ---------------------------------------------------------------------------
# Composition (general structure, exercised on K = 4).
# ---------------------------------------------------------------------------


def test_parallel_runs_two_systems_independently():
    """`A.parallel(B)` evolves the combined state exactly as A and B run separately."""
    A1, A2 = jnp.diag(jnp.array([1.0, 3.0])), jnp.diag(jnp.array([2.0]))
    A = _rk4_well(A1).parallel(_rk4_well(A2))
    B1, B2 = _rk4_well(A1), _rk4_well(A2)

    s1 = jnp.array([1.0, -0.5])
    s2 = jnp.array([0.7])
    A_state = (s1, s2)
    for _ in range(15):
        _outs, A_state = A.with_state(A_state).run_one(
            (_IN_POS0, _IN_POS0), lambda op: (_IN_DIR0, _IN_DIR0)
        )
        _o1, s1 = B1.with_state(s1).run_one(_IN_POS0, _TRIV)
        _o2, s2 = B2.with_state(s2).run_one(_IN_POS0, _TRIV)

    np.testing.assert_allclose(np.asarray(A_state[0]), np.asarray(s1), atol=1e-12)
    np.testing.assert_allclose(np.asarray(A_state[1]), np.asarray(s2), atol=1e-12)


def test_pre_static_identity_is_a_noop():
    """Post-composing every round with the identity poly map changes nothing."""
    base = _rk4_well(jnp.diag(jnp.array([1.0, 2.0])))
    comp = base.pre_static(identity_poly_map(base.tgt_poly))

    sb = sc = jnp.array([1.0, -0.4])
    for _ in range(15):
        _ob, sb = base.with_state(sb).run_one(_IN_POS0, _TRIV)
        _oc, sc = comp.with_state(sc).run_one(_IN_POS0, _TRIV)
    np.testing.assert_allclose(np.asarray(sc), np.asarray(sb), atol=1e-12)


# ---------------------------------------------------------------------------
# RK4 really is RK4.
# ---------------------------------------------------------------------------


def test_one_step_matches_hand_written_rk4():
    """One ``PhiK`` macro-tick equals a hand-rolled RK4 step of qdot = -A q."""
    A = np.array([[1.0, 0.2], [0.2, 3.0]])
    h = 0.1
    O = _rk4_well(jnp.asarray(A), h)
    x0 = jnp.array([0.6, -0.9])
    _outs, x_new = O.with_state(x0).run_one(_IN_POS0, _TRIV)

    f = lambda x: -A @ x
    x = np.asarray(x0)
    k1 = f(x)
    k2 = f(x + 0.5 * h * k1)
    k3 = f(x + 0.5 * h * k2)
    k4 = f(x + h * k3)
    x_ref = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
    np.testing.assert_allclose(np.asarray(x_new), x_ref, atol=1e-12)


def test_rk4_is_fourth_order():
    """Global error of the staged ``PhiK`` falls like h^4: halving h cuts it by ~16x."""
    A = jnp.diag(jnp.array([1.0, 2.0, 5.0]))
    x0 = jnp.array([1.0, 1.0, 1.0])
    T = 1.0
    exact_sol = np.asarray(jax.scipy.linalg.expm(-np.asarray(A) * T) @ np.asarray(x0))

    def err(N):
        O = _rk4_well(A, T / N)
        s = x0
        for _ in range(N):
            _outs, s = O.with_state(s).run_one(_IN_POS0, _TRIV)
        return float(np.linalg.norm(np.asarray(s) - exact_sol))

    e8, e16, e32 = err(8), err(16), err(32)
    assert e32 < 1e-6                  # the fine solution is accurate
    assert e8 / e16 > 12.0             # h^4 (=16), cleanly above h^3 (=8)
    assert e16 / e32 > 12.0


def test_regression_matches_cdlm():
    """The CDLM RK4 well trajectory (``regress/multistage.npz``) to rtol 1e-10."""
    R = load_regress("multistage")
    O = _rk4_well(jnp.asarray(R["rk4_A"]), float(R["rk4_h"]))
    x = jnp.asarray(R["rk4_x"][0])
    xs = [x]
    for _ in range(R["rk4_x"].shape[0] - 1):
        _outs, x = O.with_state(x).run_one(_IN_POS0, _TRIV)
        xs.append(x)
    np.testing.assert_allclose(np.stack([np.asarray(v) for v in xs]), R["rk4_x"], rtol=1e-10, atol=1e-12)
