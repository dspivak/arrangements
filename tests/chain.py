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

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.functors import closed_step
from arrangements.interpretation import trivial_omega
from arrangements.polynomial import nest_left, unnest_left
from arrangements.rvect import diagonal
from arrangements.wiring import boxes_and_couplers, trivial_parameter

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
    arrangements (def.arr): heat and wave.
    """
    return Arrangement(
        P=parameter(diagonal(jnp.array([1.0 / m]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * kappa * (q[0] - n_in[0]) ** 2),
        label="Part",
    )


# ---------------------------------------------------------------------------
# The chain as the book draws it (Ch 2, sec.springs): boxes, a dot, springs as couplers.
# ---------------------------------------------------------------------------


def one_port_box(parameter, response: float, init, label: str) -> Arrangement:
    """A part filling a box with one output port, as the book's tank, rod segment, mass and capacitor do.

    Its state is ``parameter``'s on ``R`` (``configuration``: the state is the position it reads out;
    ``phase``: a position and a momentum, and it reads out the position it is headed to, ``q + p/m``).
    Its response function multiplies by ``response`` (``1/C`` for a segment of heat capacity ``C``,
    ``1/m`` for a mass), it has no internal inclination beyond what ``parameter`` builds in (a mass's
    kinetic energy), and the force that comes back along its port is its state's whole responsibility.
    ``init`` is ``[q0]`` for ``configuration`` and ``([q0], [p0])`` for ``phase``.
    """
    return Arrangement(
        P=parameter(diagonal(jnp.array([response])), init=init),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label=label,
    )


def spring(kappa: float, label: str = "spring") -> Arrangement:
    """The book's spring (``eqn.spring_cost``), a joint of the rod or a resistor: a part without state
    whose two input ports receive ``x_1`` and ``x_2`` and whose pushback is minus the differential of
    the cost ``(kappa/2)(x_1 - x_2)^2``, so it returns ``kappa (x_2 - x_1)`` to the first end and
    ``kappa (x_1 - x_2)`` to the second (``eqn.spring_pushback``)."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: jnp.zeros(0),
        in_f=lambda q, e, n_in: jnp.zeros(0),
        omega=exact(lambda q, e, n_in: -0.5 * kappa * (e[0] - e[1]) ** 2),
        label=f"{label}({kappa})",
    )


def book_chain(boxes, kappas) -> Arrangement:
    """The chain of ``eqn.chain_wd``: ``K`` boxes in a row, consecutive ones joined by a spring of
    constant ``kappas[i]``, the interior boxes' readouts copied to both their springs. The ends are
    free, as in the book: the end boxes have one spring each, and the forces ``eqn.chain_forces`` sum
    to zero, so what the chain carries (heat, momentum) is conserved."""
    K = len(boxes)
    if len(kappas) != K - 1:
        raise ValueError(f"{K} boxes need {K - 1} spring constants, got {len(kappas)}")
    return boxes_and_couplers(boxes, [(i, i + 1) for i in range(K - 1)], [spring(k) for k in kappas])


def closed_run(arr: Arrangement, T: int):
    """``T`` steps of ``Phi(arr)`` for a closed arrangement. Returns the list of the ``T + 1`` positions
    ``arr.P.position(state)`` (for configuration boxes the states; for phase boxes the positions the
    masses read out, ``q + p/m``) and the list of the ``T + 1`` states, the start first."""
    from arrangements.functors import Phi
    O = Phi(arr)
    state = O.state
    positions, states = [np.asarray(arr.P.position(state))], [state]
    for _ in range(T):
        _, _, state = O.with_state(state).run_one(IN_POS, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        positions.append(np.asarray(arr.P.position(state)))
        states.append(state)
    return positions, states


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


class HeldChain:
    """A chain of ``parts`` with both ends held by boxes (``wiring.held_chain``), stepped from and to
    flat coordinates: ``step(q)`` for configuration parts, ``step(q, p)`` for phase or leapfrog parts.

    The ends are held particles at ``0`` with zero response; the right one is joined to the last part
    by a spring of constant ``kappa``. Single-stage chains step by ``functors.closed_step`` (compiled,
    checked against ``Phi``); a leapfrog chain by ``pc2.Phi2``. This replaces the pinned ends that the
    chain tests computed in code each step (``q_prev = 0`` fed in, ``xi_N = -kappa q~_K`` computed from
    the state; plan P43 (b) job 4).
    """

    def __init__(self, parts, kappa, parameter, two_stage=False):
        from arrangements.wiring import held_chain
        self.K = len(parts)
        self.arr = held_chain(parts, kappa, parameter)
        self.two_stage = two_stage
        self.phase = flat_state(self.arr.P.init).shape[0] == 2 * self.arr.P.Q_dim  # (q, p) per part
        if two_stage:
            from arrangements.pc2 import Phi2
            self.O = Phi2(self.arr)
        else:
            self._step = closed_step(self.arr)

    def state(self, q, p=None):
        z = jnp.zeros(1)
        if p is None:
            flat = jnp.concatenate([z, jnp.asarray(q, float), z])
        else:
            inner = jnp.stack([jnp.asarray(q, float), jnp.asarray(p, float)], axis=1).reshape(-1)
            flat = jnp.concatenate([z, z, inner, z, z])
        return state_like(self.arr.P.init, flat)

    def flat(self, state):
        x = flat_state(state)
        if not self.phase:
            return x[1:self.K + 1]
        qp = x[2:2 + 2 * self.K].reshape(self.K, 2)
        return qp[:, 0], qp[:, 1]

    def step(self, q, p=None):
        s = self.state(q, p)
        if self.two_stage:
            _, _, s = self.O.with_state(s).run_one(IN_POS, lambda _o: (jnp.zeros(0), jnp.zeros(0)))
        else:
            s = self._step(s)
        return self.flat(s)
