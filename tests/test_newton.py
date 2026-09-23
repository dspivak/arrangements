"""Newton's method as ``Phi`` of a closed scalar arrangement (CDLM sec.newton_warmup).

A closed arrangement ``f : I -> I`` whose parameter is ``configuration(Q)`` with the
position-dependent response ``sharp_q = (T_q dk)^{-1}`` (the inverse Hessian of the
cost ``k``, rvect.inverse_hessian) and whose forcing is ``omega = -dk`` has, by
sec.displacement,

    Phi(f) : q |-> q - (T_q dk)^{-1}(dk|_q)   (CDLM eqn.newton),

one Newton step for a critical point of ``k``: the book's ``q + sharp(omega)`` with
``omega = -dk`` is CDLM's ``q - sharp(dU)``.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import inverse_hessian

from tests.chain import IN_POS, load_regress


def _scalar(k_scalar, dim, parameter=configuration):
    """The closed arrangement with cost ``k``, forcing ``-dk`` and Newton response."""
    return Arrangement(
        P=parameter(inverse_hessian(k_scalar, dim)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -k_scalar(q)),
        label="newton",
    )


_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def _newton_step(O, q):
    _, _, new_q = O.with_state(q).run_one(IN_POS, lambda _o: _IN_DIR)
    return new_q


def test_newton_quadratic_lands_in_one_step():
    """Quadratic k: the inverse-Hessian step reaches the critical point in one step."""
    a = 2.0
    k = lambda q: (q[0] - a) ** 2 + 1.0
    O = Phi(_scalar(k, 1))
    new_q = _newton_step(O, jnp.array([5.0]))
    np.testing.assert_allclose(np.asarray(new_q), np.array([a]), atol=1e-10)


def test_newton_exp_step_formula_and_convergence():
    """k(x) = e^x - x: one step is x -> x - 1 + e^{-x}, and the orbit reaches 0."""
    k = lambda q: jnp.exp(q[0]) - q[0]
    O = Phi(_scalar(k, 1))

    # exact step formula
    new_q = _newton_step(O, jnp.array([2.0]))
    np.testing.assert_allclose(np.asarray(new_q), np.array([2.0 - 1.0 + np.exp(-2.0)]), atol=1e-10)

    # quadratic convergence to the minimizer x = 0
    q = jnp.array([2.0])
    for _ in range(40):
        q = _newton_step(O, q)
    assert abs(float(q[0])) < 1e-10


def test_regression_matches_cdlm():
    """``regress/misc.npz``: the Newton orbits are CDLM's (states are unchanged, STATUS.md §1)."""
    R = load_regress("misc")
    a = float(R["newton_quad_a"])
    cases = {
        "newton_quad": (lambda q: (q[0] - a) ** 2 + 1.0),
        "newton_exp": (lambda q: jnp.exp(q[0]) - q[0]),
    }
    for tag, k in cases.items():
        old = R[tag]
        O = Phi(_scalar(k, 1))
        q = jnp.asarray(old[0])
        traj = [q]
        for _ in range(len(old) - 1):
            q = _newton_step(O, q)
            traj.append(q)
        new = np.stack([np.asarray(x) for x in traj])
        np.testing.assert_allclose(new, old, rtol=1e-10, atol=1e-12)
