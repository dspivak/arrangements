"""Stock-and-flow diagrams as arrangements (book Ch 4 ``sec.stock_flow``, ``prop.stock_flow``).

Prediction, fixed before the run (it is the proposition's claim): for every diagram with smooth rates and
acyclic auxiliaries, one step of ``arrangement_of(diagram, dt)`` is the diagram's forward-Euler step, so the
two agree to rounding over many steps; a flow between two stocks conserves their total exactly; the
epidemic's forcing is not exact (no cost function gives it).

  test_epidemic_is_its_euler_step     the book's example, 400 steps: TWO IMPLEMENTATIONS AGREEING
  test_epidemic_conserves_the_total   CLOSED FORM (the flows' pushes are equal and opposite)
  test_epidemic_forcing_not_exact     CLOSED FORM (the Jacobian of the forcing is not symmetric)
  test_random_diagrams                40 random diagrams, shared auxiliaries, clouds: TWO IMPLEMENTATIONS AGREEING
"""
import jax
import jax.numpy as jnp
import numpy as np

from arrangements.stock_flow import arrangement_of, euler, random_diagram, run

jax.config.update("jax_enable_x64", True)

BETA, RHO, DT = 0.6, 0.2, 0.05


def epidemic(levels=(990.0, 10.0, 0.0)):
    stocks = [("S", levels[0]), ("I", levels[1]), ("R", levels[2])]
    auxes = [("y", ["S", "I", "R"], lambda m: m[1] / (m[0] + m[1] + m[2]))]
    flows = [("infection", "S", "I", ["S", "y"], lambda m: BETA * m[0] * m[1]),
             ("recovery", "I", "R", ["I"], lambda m: RHO * m[0])]
    return stocks, auxes, flows


def test_epidemic_is_its_euler_step():
    d = epidemic()
    np.testing.assert_allclose(run(d, DT, 400), euler(d, DT, 400), rtol=0, atol=1e-9)


def test_epidemic_conserves_the_total():
    out = run(epidemic(), DT, 400)
    np.testing.assert_allclose(out.sum(axis=1), 1000.0, rtol=0, atol=1e-9)


def test_epidemic_forcing_not_exact():
    def F(q):
        y = q[1] / q.sum(); inf = BETA * q[0] * y; rec = RHO * q[1]
        return jnp.array([-inf, inf - rec, rec])
    J = np.asarray(jax.jacfwd(F)(jnp.array([700.0, 200.0, 100.0])))
    assert np.abs(J - J.T).max() > 0.1


def test_random_diagrams():
    rng = np.random.default_rng(20261008)
    for t in range(40):
        d = random_diagram(rng)
        err = np.abs(run(d, 0.05, 60) - euler(d, 0.05, 60)).max()
        print(f"diagram {t}: {len(d[0])} stocks, {len(d[1])} auxiliaries, {len(d[2])} flows; max err {err:.1e}", flush=True)
        assert err < 1e-10
