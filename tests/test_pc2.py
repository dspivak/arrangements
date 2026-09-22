"""General pc^(2) datatype and its composition (pc2.py).

``Phi2`` of an arrangement with a ``leapfrog`` parameter is an instance of the
general two-stage coalgebra ``PCMorphism2``. These tests exercise the *general*
structure (not leapfrog-specific): that the two-round execution composes
correctly under ``parallel`` and ``pre_static``, and that ``Phi2`` works for any
``CotCoalg2``, e.g. the canonical lift ``as_two_stage(phase(Q))`` (two phase-Euler
steps per macro-tick), which equals running ``Phi`` of the phase arrangement twice.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import as_two_stage, phase
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.leapfrog import leapfrog
from arrangements.pc2 import PCMorphism2, Phi2
from arrangements.polynomial import identity_poly_map
from arrangements.rvect import diagonal


def _oscillator(omega2, parameter=leapfrog):
    """A closed (I -> I) harmonic oscillator: Q = R, mass 1, cost (omega2/2) q^2."""
    return Arrangement(
        P=parameter(diagonal(jnp.array([1.0]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * omega2 * q[0] ** 2),
        label="osc",
    )


_IN_POS0 = (jnp.zeros(0), trivial_omega(0))  # closed position
_IN_DIR0 = (jnp.zeros(0), jnp.zeros(0))                        # closed direction


def test_phi2_leapfrog_is_a_general_pc2_morphism():
    """Leapfrog dynamics is an instance of the general datatype, not a bespoke class."""
    assert isinstance(Phi2(_oscillator(1.0)), PCMorphism2)


def test_parallel_runs_two_systems_independently():
    """`A.parallel(B)` evolves the combined state exactly as A and B run separately."""
    o1, o2 = _oscillator(0.5), _oscillator(1.3)
    A = Phi2(o1).parallel(Phi2(o2))
    B1, B2 = Phi2(o1), Phi2(o2)

    s1 = (jnp.array([1.0]), jnp.array([0.0]))
    s2 = (jnp.array([0.5]), jnp.array([0.2]))
    A_state = (s1, s2)

    for _ in range(20):
        _, _, A_state = A.with_state(A_state).run_one(
            (_IN_POS0, _IN_POS0), lambda op: (_IN_DIR0, _IN_DIR0)
        )
        _, _, s1 = B1.with_state(s1).run_one(_IN_POS0, lambda op: _IN_DIR0)
        _, _, s2 = B2.with_state(s2).run_one(_IN_POS0, lambda op: _IN_DIR0)

    for got, want in ((A_state[0], s1), (A_state[1], s2)):
        np.testing.assert_allclose(np.asarray(got[0]), np.asarray(want[0]), atol=1e-10)
        np.testing.assert_allclose(np.asarray(got[1]), np.asarray(want[1]), atol=1e-10)


def test_pre_static_identity_is_a_noop():
    """Post-composing both rounds with the identity poly map changes nothing."""
    o = _oscillator(0.7)
    base = Phi2(o)
    comp = base.pre_static(identity_poly_map(base.tgt_poly))

    sb = sc = (jnp.array([1.0]), jnp.array([0.3]))
    for _ in range(15):
        _, _, sb = base.with_state(sb).run_one(_IN_POS0, lambda op: _IN_DIR0)
        _, _, sc = comp.with_state(sc).run_one(_IN_POS0, lambda op: _IN_DIR0)
    np.testing.assert_allclose(np.asarray(sc[0]), np.asarray(sb[0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(sc[1]), np.asarray(sb[1]), atol=1e-10)


def test_phi2_is_general():
    """``Phi2`` works for ANY two-stage coalgebra, not just leapfrog: the canonical
    two-stage lift of ``phase(Q)`` equals running ``Phi`` of the phase arrangement twice."""
    A = Phi2(_oscillator(0.6, parameter=lambda Q: as_two_stage(phase(Q))))
    B = Phi(_oscillator(0.6, parameter=phase))

    s = (jnp.array([1.0]), jnp.array([0.2]))
    _, _, a_state = A.with_state(s).run_one(_IN_POS0, lambda op: _IN_DIR0)
    _, _, b1 = B.with_state(s).run_one(_IN_POS0, lambda op: _IN_DIR0)
    _, _, b2 = B.with_state(b1).run_one(_IN_POS0, lambda op: _IN_DIR0)

    np.testing.assert_allclose(np.asarray(a_state[0]), np.asarray(b2[0]), atol=1e-10)
    np.testing.assert_allclose(np.asarray(a_state[1]), np.asarray(b2[1]), atol=1e-10)
