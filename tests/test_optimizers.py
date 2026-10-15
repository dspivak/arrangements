"""The parameter is the optimizer (coalgebra.damped).

One convex forcing -- ``omega = -dk`` for the quadratic well ``k(q) = 1/2 q.A.q`` --
carried by three arrangements that differ only in their parameter (def.arr: the
integrator is part of the arrangement, so these are three *different* objects of
``arr``, not one object read three ways as in CDLM's ``Phiconf``/``Phiphase``/
``Phidamped``):

* ``configuration(Q)`` -- descent: gradient descent, converges.
* ``phase(Q)``         -- conservative phase flow: oscillates forever, never converges.
* ``damped(Q, c)``     -- phase with friction: heavy-ball momentum, converges.

So the same forcing yields SGD, conservative dynamics, or momentum, selected by the
parameter alone -- the optimizer analogue of wave-vs-heat.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration, damped, phase
from arrangements.functors import Phi
from arrangements.rvect import euclidean

from tests.chain import IN_POS

_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _quadratic_well(A, eta=1.0, parameter=configuration):
    """Closed autonomous arrangement: cost ``k(q) = 1/2 q.A.q``, forcing ``-dk``,
    sharp ``eta*I``, parameter ``parameter(Q)``."""
    A = jnp.asarray(A, float)
    d = int(A.shape[0])
    return Arrangement(
        P=parameter(euclidean(d, eta)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * q @ (A @ q)),
        label="quadratic_well",
    )


def _conf_traj(O, q0, steps):
    q = jnp.asarray(q0, float)
    out = [q]
    for _ in range(steps):
        _, _, q = O.with_state(q).run_one(IN_POS, _TRIV)
        out.append(q)
    return np.stack([np.asarray(x) for x in out])


def _phase_traj(O, q0, steps):
    state = (jnp.asarray(q0, float), jnp.zeros(len(q0)))
    qs, ps = [np.asarray(state[0])], [np.asarray(state[1])]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(IN_POS, _TRIV)
        qs.append(np.asarray(state[0])); ps.append(np.asarray(state[1]))
    return np.stack(qs), np.stack(ps)


def _norms(qs):
    return np.linalg.norm(qs, axis=1)


def test_integrator_is_the_optimizer():
    A = jnp.diag(jnp.array([1.0, 3.0, 9.0]))  # convex, ill-conditioned
    q0 = jnp.array([1.0, 1.0, 1.0])

    conf = _norms(_conf_traj(Phi(_quadratic_well(A, 0.1, configuration)), q0, 600))
    phase_q, _ = _phase_traj(Phi(_quadratic_well(A, 0.1, phase)), q0, 600)
    damped_q, _ = _phase_traj(
        Phi(_quadratic_well(A, 0.1, lambda Q: damped(Q, 0.15))), q0, 600)

    # configuration parameter: descent, converges to the minimum q = 0.
    assert conf[-1] < 1e-3
    # phase parameter: conservative, never converges -- still oscillating.
    assert _norms(phase_q)[-100:].max() > 0.3
    # damped phase: momentum, converges.
    assert _norms(damped_q)[-1] < 1e-3


def test_regression_matches_cdlm():
    """``regress/misc.npz``: all three optimizer trajectories are CDLM's
    (states unchanged; the sign moved from the integrator into the forcing)."""
    from tests.chain import load_regress

    R = load_regress("misc")
    eta, damping, steps = float(R["opt_params"][0]), float(R["opt_params"][1]), int(R["opt_params"][2])
    A = jnp.diag(jnp.asarray(R["opt_diagA"]))
    q0 = jnp.asarray(R["opt_conf_q"][0])

    conf = _conf_traj(Phi(_quadratic_well(A, eta, configuration)), q0, steps)
    np.testing.assert_allclose(conf, R["opt_conf_q"], rtol=1e-10, atol=1e-12)

    for tag, parameter in (("phase", phase), ("damped", lambda Q: damped(Q, damping))):
        qs, ps = _phase_traj(Phi(_quadratic_well(A, eta, parameter)), q0, steps)
        np.testing.assert_allclose(qs, R[f"opt_{tag}_q"], rtol=1e-10, atol=1e-12)
        np.testing.assert_allclose(ps, R[f"opt_{tag}_p"], rtol=1e-10, atol=1e-12)
