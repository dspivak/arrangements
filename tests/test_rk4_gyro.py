"""RK4 on the gyro phase system as pc^(4) (rk4.rk4_gyro / PhiK), and the single-stage
``damped`` / ``gyro`` coalgebras -- each in its TRICK and its DIRECT realization.

The faithful machine's integrator: classical RK4 stepping the phase ODE
``qdot = sharp(xi), xidot = xi_Q - drag*F(v) - gamma*(J v)`` -- the same forces as
``gyro`` but the blog's RK4 time-stepping instead of symplectic Euler. We verify:

* 4th-order accuracy: on a harmonic oscillator (closed form via matrix exp) the global
  error falls like h^4 (ratios -> 16) -- the falsifiable proof it is genuine RK4 run
  through the four pc^(4) rounds on the PHASE state;
* one macro-tick equals a hand-rolled RK4 step of the full phase ODE *including* the
  quadratic-drag and per-gyro precession 1-forms;
* the CDLM trajectories (``regress/multistage.npz``) for ``rk4_gyro``, ``damped`` and
  ``gyro`` are reproduced to rtol 1e-10;
* **trick vs direct**: CDLM smuggled the velocity-dependent drag and gyro 1-forms into
  the integrator because its arrangement carried only a potential on ``Q``. The book's
  arrangement carries a forcing on the base, so the direct realization enlarges the
  base to ``T^*Q`` (``phase_on_cotangent`` / ``rk4_gyro_on_cotangent``) and writes
  ``omega|_Q = -dU(q) + gyro_kick(sharp(xi))``, a genuinely non-exact one-form. The
  two agree step for step.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import damped, gyro, gyro_kick, phase_on_cotangent
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK
from arrangements.rk4 import rk4_gyro, rk4_gyro_on_cotangent
from arrangements.rvect import constant

from tests.chain import load_regress

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda op: (jnp.zeros(0), jnp.zeros(0))


def _complex_structure(V):
    """Block-diagonal ``J``: one ``[[0,-1],[1,0]]`` per 2-D gyro (gyroscope.py)."""
    return jnp.kron(jnp.eye(V), jnp.array([[0.0, -1.0], [1.0, 0.0]]))


def _oscillator(d, m, k, parameter):
    """Closed harmonic oscillator on R^d: cost (k/2)|q|^2, sharp = (1/m) I (forcing -k q)."""
    return Arrangement(
        P=parameter(constant(jnp.eye(d) / m)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, mo: jnp.zeros(0),
        in_f=lambda q, mo, ni: jnp.zeros(0),
        omega=exact(lambda q, mo, ni: -0.5 * k * jnp.sum(q ** 2)),
        label="osc",
    )


def _oscillator_on_cotangent(d, m, k, parameter, gamma, J, drag, gyro_block):
    """The DIRECT realization: base ``T^*Q = R^{2d}``, forcing ``omega|_Q = -k q + gyro_kick(v)``
    with ``v = sharp(xi) = xi / m`` read off the presented momentum; not exact."""
    Q = constant(jnp.eye(d) / m)

    def omega(qp, mo, ni):
        q, xi = qp[:d], qp[d:]
        kick = gyro_kick(Q.apply_sharp(q, xi), gamma, J, drag, gyro_block)
        return (jnp.concatenate([-k * q + kick, jnp.zeros(d)]), jnp.zeros(0), jnp.zeros(0))

    return Arrangement(
        P=parameter(Q),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, mo: jnp.zeros(0),
        in_f=lambda q, mo, ni: jnp.zeros(0),
        omega=omega,
        label="osc on T*Q",
    )


def test_rk4_gyro_is_fourth_order_on_oscillator():
    """Global error vs the exact phase flow e^{AT} falls like h^4 (ratios -> 16)."""
    d, m, k, T = 2, 1.3, 2.0, 1.0
    q0 = np.array([1.0, -0.5])
    xi0 = np.array([0.2, 0.7])
    # phase generator A on (q, xi): qdot = xi/m, xidot = -k q
    A = np.block([[np.zeros((d, d)), np.eye(d) / m], [-k * np.eye(d), np.zeros((d, d))]])
    z0 = np.concatenate([q0, xi0])
    exact_sol = np.asarray(jax.scipy.linalg.expm(jnp.asarray(A * T))) @ z0

    def err(N):
        O = PhiK(_oscillator(d, m, k, lambda Q: rk4_gyro(Q, T / N)))
        s = (jnp.asarray(q0), jnp.asarray(xi0))
        for _ in range(N):
            _, s = O.with_state(s).run_one(_IN_POS, _TRIV)
        got = np.concatenate([np.asarray(s[0]), np.asarray(s[1])])
        return float(np.linalg.norm(got - exact_sol))

    e8, e16, e32 = err(8), err(16), err(32)
    assert e32 < 1e-5
    assert e8 / e16 > 12.0
    assert e16 / e32 > 12.0


def test_rk4_gyro_one_step_matches_hand_written_with_drag_and_precession():
    """One ``PhiK`` tick = a hand-rolled RK4 step of the full phase ODE, drag + gamma in."""
    V, m, k, h = 2, 1.0, 0.5, 0.1
    dim = 2 * V
    J = _complex_structure(V)
    drag = 0.3
    gamma = jnp.array([0.4, 0.4, 0.0, 0.0])  # per-gyro
    O = PhiK(_oscillator(dim, m, k, lambda Q: rk4_gyro(Q, h, drag=drag, gamma=gamma, J=J, gyro_block=2)))

    q0 = jnp.array([0.3, -0.2, 0.1, 0.5])
    xi0 = jnp.array([0.2, 0.1, -0.3, 0.4])
    _, (q1, xi1) = O.with_state((q0, xi0)).run_one(_IN_POS, _TRIV)

    Jm = np.asarray(J)
    gam = np.asarray(gamma)

    def Fdrag(v):
        vg = v.reshape(V, 2)
        sp = np.sqrt((vg ** 2).sum(axis=1, keepdims=True))
        return (sp * vg).reshape(-1)

    def f(z):  # full phase vector field
        q, xi = z[:dim], z[dim:]
        v = xi / m
        dU = k * q
        return np.concatenate([v, -dU - drag * Fdrag(v) - gam * (Jm @ v)])

    z = np.concatenate([np.asarray(q0), np.asarray(xi0)])
    k1 = f(z)
    k2 = f(z + h / 2 * k1)
    k3 = f(z + h / 2 * k2)
    k4 = f(z + h * k3)
    z_new = z + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    got = np.concatenate([np.asarray(q1), np.asarray(xi1)])
    np.testing.assert_allclose(got, z_new, atol=1e-10)


def _gyro_setup(R):
    V, m, k, h, drag = R["gyro4_params"]
    V = int(V)
    J = jnp.asarray(R["gyro4_J"])
    gamma = jnp.asarray(R["gyro4_gamma"])
    return V, m, k, h, drag, J, gamma


def _run(O, s, T, multistage):
    qs, ps = [np.asarray(s[0])], [np.asarray(s[1])]
    for _ in range(T):
        if multistage:
            _, s = O.with_state(s).run_one(_IN_POS, _TRIV)
        else:
            _, _, s = O.with_state(s).run_one(_IN_POS, _TRIV)
        qs.append(np.asarray(s[0])); ps.append(np.asarray(s[1]))
    return np.stack(qs), np.stack(ps)


def test_regression_matches_cdlm():
    """``rk4_gyro``, ``damped`` and ``gyro`` reproduce CDLM's trajectories (rtol 1e-10)."""
    R = load_regress("multistage")
    V, m, k, h, drag, J, gamma = _gyro_setup(R)
    dim = 2 * V
    s0 = (jnp.asarray(R["gyro4_q"][0]), jnp.asarray(R["gyro4_p"][0]))

    O = PhiK(_oscillator(dim, m, k, lambda Q: rk4_gyro(Q, h, drag=drag, gamma=gamma, J=J, gyro_block=2)))
    qs, ps = _run(O, s0, R["gyro4_q"].shape[0] - 1, True)
    np.testing.assert_allclose(qs, R["gyro4_q"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(ps, R["gyro4_p"], rtol=1e-10, atol=1e-12)

    for tag, param in (("damped", lambda Q: damped(Q, 0.1)),
                       ("gyro", lambda Q: gyro(Q, damping=0.1, gamma=gamma, J=J, drag=drag, gyro_block=2))):
        O = Phi(_oscillator(dim, m, k, param))
        qs, ps = _run(O, s0, R[f"{tag}_q"].shape[0] - 1, False)
        np.testing.assert_allclose(qs, R[f"{tag}_q"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ps, R[f"{tag}_p"], rtol=1e-10, atol=1e-12)


def test_direct_one_form_realization_agrees_with_trick():
    """The gyro/drag terms as a non-exact forcing on ``T^*Q`` (direct) equal the
    same terms inside the step (trick), step for step, for ``gyro`` and ``rk4_gyro``."""
    R = load_regress("multistage")
    V, m, k, h, drag, J, gamma = _gyro_setup(R)
    dim = 2 * V
    s0 = (jnp.asarray(R["gyro4_q"][0]), jnp.asarray(R["gyro4_p"][0]))
    kw = dict(gamma=gamma, J=J, drag=drag, gyro_block=2)

    trick = Phi(_oscillator(dim, m, k, lambda Q: gyro(Q, damping=0.1, **kw)))
    direct = Phi(_oscillator_on_cotangent(dim, m, k, lambda Q: phase_on_cotangent(Q, 0.1), **kw))
    a, b = _run(trick, s0, 30, False), _run(direct, s0, 30, False)
    np.testing.assert_allclose(b[0], a[0], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(b[1], a[1], rtol=1e-12, atol=1e-13)

    trick4 = PhiK(_oscillator(dim, m, k, lambda Q: rk4_gyro(Q, h, **kw)))
    direct4 = PhiK(_oscillator_on_cotangent(dim, m, k, lambda Q: rk4_gyro_on_cotangent(Q, h), **kw))
    a, b = _run(trick4, s0, 10, True), _run(direct4, s0, 10, True)
    np.testing.assert_allclose(b[0], a[0], rtol=1e-12, atol=1e-13)
    np.testing.assert_allclose(b[1], a[1], rtol=1e-12, atol=1e-13)
