"""Copying a system with inputs: the copier and the comparison arrangement (book Chapter 3, ``ex.copying_pushback``;
linked from its last paragraph, "The composite runs in IMC").

For a reader of the book. The original is a system ``I -> <V | W>``, and the copier a system ``I -> <V | W x V>``
whose extra output ``v'`` always sends ``0`` and receives a teaching load. ``V = R^n`` and ``W = R^w`` with their
Euclidean inner products, so the book's ``sharp`` and ``flat`` are the identity on coordinates; the code stays on
``R^n``, as the book's ``ex.copying_pushback`` asks.

* ``pushback_copy(material, output, inclination, n, w)`` builds the book's ``copier`` from the data it asks one to
  choose: ``material`` is the integrator ``(Q, S, r, u)`` (a ``CotCoalg``: ``position_at(s, tau)`` is the look-ahead
  ``r_{s,tau}``, ``step_over(s, xi, tau)`` the update ``u_{s,tau}(xi)``), ``output(q)`` is ``o(q)``, and
  ``inclination(q, v)`` returns the pair ``(lambda_int(q, v), pi_c(q, v))`` of ``eqn.copy_general_base_field``, the
  internal inclination and the input pushback. Its configuration is ``Q x V x V*`` read out as ``(r_{s,tau}, 0, 0)``
  (``eqn.copy_general_readout``); its out-transfer sends ``(q, v', xi')`` to ``(o(q), v')``; its inclination field is
  ``(lambda_int(q, v), 0, v, pi_c(q, v))`` (``eqn.copy_general_field``); and its update adds to the inclination at
  ``q`` the teaching load ``lambda_v'`` pulled back along ``pi_c(-, v)``, differentiated at the look-ahead
  (``eqn.copy_general_update``).
* ``pushback_comparator(n, w)`` is the book's ``compare : <V | W> (x) <V | W x V> -> <V x R | W x W>``
  (``eqn.copy_general_interface``): external inputs ``(v, kappa)``, internal outputs ``(y_o, y_c, v')``; its
  in-transfer supplies ``(v + kappa v', v - kappa v')`` to the original and the copier, its out-transfer is
  ``(y_o, y_c)``, and its inclination field puts ``kappa (y_o - y_c)`` at ``y_c`` and ``0`` everywhere else
  (``eqn.copy_general_compare_field``), which is ``couple`` with the circle ``T`` of ``eqn.copy_general_transfer``
  folded in.

The composite ``compose_seq(parallel_arrangements(original, copier), pushback_comparator(n, w))`` is the book's
``compare . (original (x) copier)``, drawn in ``eqn.copy_general_wiring``. The book's small example, ``V = W = R``,
configuration ``q = (x, kappa_c)``, output ``o(q) = x`` and pushback ``pi_c(q, v) = kappa_c (x - v)``: one step of
duration 1 of the copier alone, at input ``v = 0.2`` with load ``0.1`` at its extra output, adds ``kappa_c 0.1`` at
position and ``(x - v) 0.1`` at stiffness. Run from the library's folder with it on the path, as the README's *Run it*
says (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from arrangements.coalgebra import CotCoalg
    from arrangements.functors import Phi
    from arrangements.interpretation import trivial_omega
    from arrangements.pushback_copying import pushback_copy
    material = CotCoalg(2, jnp.array([0.5, 2.0]), lambda s: s, lambda s, xi: s + xi,
                        look_ahead=lambda s, t: s, timed_step=lambda s, xi, t: s + t * xi)
    field = lambda q, v: (jnp.zeros(2), q[1:] * (q[:1] - v))     # (lambda_int, pi_c)
    copier = pushback_copy(material, lambda q: q[:1], field, 1, 1)
    _, _, s = Phi(copier).run_one((jnp.zeros(0), trivial_omega(0)),
                                  lambda y: (jnp.array([0.0, 0.1]), jnp.array([0.2])))
    print(s)    # [0.7  2.03]: 0.5 + 2 * 0.1 at position, 2 + (0.5 - 0.2) * 0.1 at stiffness

No functoriality of the construction is asserted, and the teaching inclination alone does not make the copier
converge: the tests, ``tests/test_pushback_copying.py``, check the stated step.

For agents: the head before 2026-11-13 is in ``notes/P31-B-CH3-HISTORY-2026-11-13.md``.
"""
import jax
import jax.numpy as jnp

from .arrangement import Arrangement
from .coalgebra import CotCoalg
from .wiring import trivial_parameter


def pushback_copy(material, output, inclination, input_dim, output_dim):
    """The book's ``copier : I -> <V | W x V>`` from the data it asks one to choose (``ex.copying_pushback``).

    ``material`` is the integrator (Q,S,r,u), ``output(q)`` is o(q), and ``inclination(q, v)`` is
    (lambda_int(q, v), pi_c(q, v)), with V = R^input_dim and W = R^output_dim, Euclidean. The configuration is
    Q x V x V*, the state space is S unchanged, and the last configuration covector supplies the current input v
    to the update.
    """
    if input_dim < 0 or output_dim < 0:
        raise ValueError("Input and output dimensions must be nonnegative")
    p, n = material.Q_dim, input_dim

    def read(s, duration):
        return jnp.concatenate((material.position_at(s, duration), jnp.zeros(2*n)))

    def update(s, xi, duration):
        q = material.position_at(s, duration)
        teaching, v = xi[p:p+n], xi[p+n:]          # the load at v', and the input v carried at xi'
        sensitivity = jax.jacrev(lambda qq: inclination(qq, v)[1])(q)   # T_q(v* pi_c)
        return material.step_over(s, xi[:p] + sensitivity.T @ teaching, duration)

    response = CotCoalg(
        p+2*n, material.init, lambda s: read(s, 1.),
        lambda s, xi: update(s, xi, 1.), label="pushback_copy_material",
        look_ahead=read, timed_step=update)

    def field(q, m, v):
        load_int, pi_c = inclination(q[:p], v)
        return jnp.concatenate((load_int, jnp.zeros(n), v)), jnp.zeros(0), pi_c

    return Arrangement(
        P=response, in_dim_M=0, out_dim_M=0, in_dim_N=n,
        out_dim_N=output_dim+n,
        out_f=lambda q, m: jnp.concatenate((output(q[:p]), q[p:p+n])),
        in_f=lambda q, m, x: jnp.zeros(0), omega=field,
        label="pushback_copy")


def pushback_comparator(input_dim, output_dim):
    """The book's ``compare : <V | W> (x) <V | W x V> -> <V x R | W x W>`` (``eqn.copy_general_interface``).

    The last external input is the signal kappa, switching both teaching loads. Since v' = 0 both internal inputs
    equal v, and the teaching load at v' is kappa (lambda_o - lambda_c). The shared input v receives
    lambda_o + lambda_c, so whatever supplies v and responds to that load can couple the original and the copier
    indirectly.
    """
    if input_dim < 0 or output_dim < 0:
        raise ValueError("Input and output dimensions must be nonnegative")
    n, w = input_dim, output_dim

    def inward(q, m, external):
        v, kappa, v_prime = external[:n], external[n], m[2*w:]
        return jnp.concatenate((v+kappa*v_prime, v-kappa*v_prime))

    def field(q, m, external):
        load = external[n]*(m[:w]-m[w:2*w])
        return (jnp.zeros(0), jnp.concatenate((jnp.zeros(w), load, jnp.zeros(n))),
                jnp.zeros(n+1))

    return Arrangement(
        P=trivial_parameter(), in_dim_M=2*n, out_dim_M=2*w+n,
        in_dim_N=n+1, out_dim_N=2*w,
        out_f=lambda q, m: m[:2*w], in_f=inward, omega=field,
        label="pushback_comparator")
