"""Shared helpers: the harmonic particle and the chain's state shape (book sec.heat_waves).

A chain ``compose_chain([Part]*K)`` has parameter
``(P_1 (x) ... (x) P_K) (x) P_wire``, so its state is
``(nest_left([s_1, ..., s_K]), s_wire)`` with ``s_wire = R^0``; these helpers
convert between that shape and flat arrays ``q`` (and ``p``) over ``R^K``.
"""

import os

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.interpretation import trivial_omega
from arrangements.polynomial import nest_left, unnest_left
from arrangements.rvect import diagonal

IN_POS = (jnp.zeros(0), trivial_omega(0))
REGRESS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "regress")


def load_regress(name):
    """The CDLM trajectories recorded by ``regress/generate_cdlm.py``."""
    return np.load(os.path.join(REGRESS, name + ".npz"))


def harmonic_particle(parameter, m, kappa) -> Arrangement:
    """The single harmonic particle ``Part : <R^0|R^0> -> <R|R>`` with parameter
    ``parameter(Q)``, ``Q = R`` with ``sharp(p) = p/m``; ``out_f = id``, ``in_f``
    vacuous, forcing ``omega = -dk`` for the spring cost ``k(q, y) = (kappa/2)(q - y)^2``.
    The *same* particle data with ``configuration`` or ``phase`` are two different
    arrangements (def:arr): heat and wave.
    """
    return Arrangement(
        P=parameter(diagonal(jnp.array([1.0 / m]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * kappa * (q[0] - n_in[0]) ** 2),
        label="Part",
    )


def conf_state(q):
    """Flat ``q in R^K`` as the state of a chain of configuration particles."""
    K = q.shape[0]
    return (nest_left([q[i:i + 1] for i in range(K)]), jnp.zeros(0))


def phase_state(q, p):
    """Flat ``(q, p)`` as the state of a chain of phase particles."""
    K = q.shape[0]
    return (nest_left([(q[i:i + 1], p[i:i + 1]) for i in range(K)]), jnp.zeros(0))


def conf_flat(state, K):
    parts, _ = state
    return jnp.concatenate(unnest_left(parts, K))


def phase_flat(state, K):
    parts, _ = state
    pairs = unnest_left(parts, K)
    return jnp.concatenate([q for q, _ in pairs]), jnp.concatenate([p for _, p in pairs])


def flat_state(state):
    """The flat coordinates of a composite's state whose leaves are single arrays
    (configuration or K-stage-position parameters): the leaves in tensor order,
    which is the order of the presented position ``q`` (``CotCoalg.tensor``).
    Works for any association of the tensor (left-nested or balanced)."""
    return jnp.concatenate([jnp.ravel(x) for x in jax.tree_util.tree_leaves(state)])


def state_like(like, flat):
    """Inverse of ``flat_state``: the pytree of ``like``'s shape (e.g. ``arr.P.init``)
    whose leaves are the consecutive slices of ``flat``."""
    leaves, treedef = jax.tree_util.tree_flatten(like)
    pieces, i = [], 0
    for leaf in leaves:
        n = int(jnp.size(leaf))
        pieces.append(jnp.reshape(flat[i:i + n], jnp.shape(leaf)))
        i += n
    if i != flat.shape[0]:
        raise ValueError(f"state_like: {flat.shape[0]} coordinates for a state of size {i}")
    return jax.tree_util.tree_unflatten(treedef, pieces)
