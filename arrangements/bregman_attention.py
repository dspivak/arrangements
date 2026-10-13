"""Ordered Bregman attention parents (book plan P44).

The transfer from sender j to receiver i has weight exp(-D(x_j,x_i))
and signal map that weight times V_j. Sites, placement and response are
separate supplied data. The caller warrants convexity, smoothness and domain
membership; this constructor does not infer them from sample values.
Composition and dynamics use the existing attention operad and Phi.
"""
import jax
import jax.numpy as jnp

from .attention import Transfer, attention_arrangement, zero_cost, zero_transfer


def divergence(potential, sender, receiver):
    """D_phi(sender, receiver), differentiating only at the receiver.

    Both arguments must be in the potential's domain. A finite boundary value
    at the sender is allowed if the resulting pullback is smooth; supply that
    boundary formula explicitly rather than differentiating x*log(x) at zero.
    """
    return (potential(sender) - potential(receiver)
            - jnp.vdot(jax.grad(potential)(receiver), sender - receiver))


def bregman_arrangement(n, dim_H, dim_P, *, parameter, placement, endpoints,
                        potential=None, divergence_fn=None, values=None,
                        cost=zero_cost, label="Bregman-attention"):
    """Build an attention parent with ordered divergence on each transfer.

    endpoints(q, i, j) gives (sender, receiver) feature vectors for T_ij,
    indices 0..n (0 external). The book's sites/linear maps are encoded here;
    endpoints need not coincide with physical placement. Supply exactly one of
    a differentiable convex potential or an explicit divergence_fn. The latter
    supports boundary formulas such as D_entropy(0,d)=d without evaluating log(0).
    values(q) optionally returns the (n+1,dim_H,dim_H) sender maps.
    """
    if (potential is None) == (divergence_fn is None):
        raise ValueError("supply exactly one potential or explicit divergence")
    D = divergence_fn if divergence_fn is not None else lambda x, y: divergence(potential, x, y)

    def transfers(q):
        V = jnp.tile(jnp.eye(dim_H)[None], (n + 1, 1, 1)) if values is None else values(q)
        rows = []
        for i in range(n + 1):
            row = []
            for j in range(n + 1):
                if i == j:
                    row.append(zero_transfer(dim_H))
                else:
                    x, y = endpoints(q, i, j)
                    gain = jnp.exp(-D(x, y))
                    row.append(Transfer(gain, gain * V[j]))
            rows.append(tuple(row))
        return tuple(rows)

    return attention_arrangement(n, dim_H, dim_P, placement, transfers,
                                 parameter=parameter, cost=cost, label=label)
