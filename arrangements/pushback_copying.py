"""Systems and a comparison morphism in IMC (Ch 3 ex.copying_pushback).

Cartesian configuration/input/output spaces only. The copy is specified by
its complete data; no functoriality of this construction is asserted.
The learning inclination alone does not imply convergence. See STATUS.md,
"Copying input pushback". GPT-6, fde05da3, 2026-11-11.
"""
import jax
import jax.numpy as jnp

from .arrangement import Arrangement
from .coalgebra import CotCoalg
from .wiring import trivial_parameter


def pushback_copy(material, output, inclination, input_dim, output_dim):
    """Construct I -> <V | W x V> from the data in the book.

    ``material`` is (Q,S,r,u), ``output(q)`` is o(q), and
    ``inclination(q,x)`` is (gamma(q,x), beta(q,x)). Euclidean metrics.
    Configuration is Q x V x V*, state is unchanged, and the last
    configuration covector supplies the current input to the update.
    """
    if input_dim < 0 or output_dim < 0:
        raise ValueError("Input and output dimensions must be nonnegative")
    p, n = material.Q_dim, input_dim

    def read(s, duration):
        return jnp.concatenate((material.position_at(s, duration), jnp.zeros(2*n)))

    def update(s, xi, duration):
        q = material.position_at(s, duration)
        e, x = xi[p:p+n], xi[p+n:]
        sensitivity = jax.jacrev(lambda qq: inclination(qq, x)[1])(q)
        return material.step_over(s, xi[:p] + sensitivity.T @ e, duration)

    response = CotCoalg(
        p+2*n, material.init, lambda s: read(s, 1.),
        lambda s, xi: update(s, xi, 1.), label="pushback_copy_material",
        look_ahead=read, timed_step=update)

    def field(q, m, x):
        gamma, beta = inclination(q[:p], x)
        return jnp.concatenate((gamma, jnp.zeros(n), x)), jnp.zeros(0), beta

    return Arrangement(
        P=response, in_dim_M=0, out_dim_M=0, in_dim_N=n,
        out_dim_N=output_dim+n,
        out_f=lambda q, m: jnp.concatenate((output(q[:p]), q[p:p+n])),
        in_f=lambda q, m, x: jnp.zeros(0), omega=field,
        label="pushback_copy")


def pushback_comparator(input_dim, output_dim):
    """C: <V|W> tensor <V|W x V> -> <V x R|W x W>.

    The last external input is kappa, switching both teaching loads.
    At z=0 both internal inputs equal x, and the teaching load at z is
    kappa*(a-b). The shared input returns a+b, so a responsive provider
    can indirectly couple the original and copy.
    """
    if input_dim < 0 or output_dim < 0:
        raise ValueError("Input and output dimensions must be nonnegative")
    n, w = input_dim, output_dim

    def inward(q, m, external):
        x, kappa, z = external[:n], external[n], m[2*w:]
        return jnp.concatenate((x+kappa*z, x-kappa*z))

    def field(q, m, external):
        load = external[n]*(m[:w]-m[w:2*w])
        return (jnp.zeros(0), jnp.concatenate((jnp.zeros(w), load, jnp.zeros(n))),
                jnp.zeros(n+1))

    return Arrangement(
        P=trivial_parameter(), in_dim_M=2*n, out_dim_M=2*w+n,
        in_dim_N=n+1, out_dim_N=2*w,
        out_f=lambda q, m: m[:2*w], in_f=inward, omega=field,
        label="pushback_comparator")
