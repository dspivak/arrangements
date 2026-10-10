"""System identification: a closed system trains a predictor of itself
(CDLM rmk.graph_environment, "Predicting a closed system").

EXTENSION (beyond the paper). This reuses the dynamics functor ``Phi`` of an
arrangement with a *phase* parameter (sec.dynamics_functor, sec.integrators)
and the learner of learning.py unchanged, and the data stream is the environment
of rmk.graph_environment / ex.training_data; but the nonlinear pendulum, the
tanh-MLP predictor, and the libration-regime sampling are standard ML layered on
top, not part of the paper's formal development. It is a toy demonstration that the
primitives compose -- not an implementation of a paper result. See the "Extensions"
section of the README.

Given a closed system ``c : S -> S`` (a ``y``-coalgebra, ex.y_coalgebra_trajectory --
here ``Phi`` of a nonlinear oscillator whose parameter is ``phase(Q)``) and an
observation ``o : S -> X``, the canonical *one-step-prediction stream* is the
coalgebra with

    readout   <o, o.c> : S -> X x X      (observe now, and one step ahead)
    update    c        : S -> S          (advance)

i.e. the environment of ex.training_data, driven by the system itself. Closing
the learner with it displaces the network's weights along the forcing of the
loss (sec.neural_network) toward predicting ``o.c`` from ``o``: the system's
one-step map.

The paper's particles are all quadratic, so that map is linear; in code we put in
*any* smooth potential -- a pendulum, a double well -- and the map is genuinely
nonlinear, so the net learns a real flow map. We take ``o = id`` (full state), so
the target is a deterministic function of the input and the fit is clean.
"""

from __future__ import annotations

from typing import Callable, List, Sequence, Tuple

import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, exact
from .coalgebra import configuration, phase
from .functors import Phi
from .interpretation import trivial_omega
from .learning import parameterized_map, train
from .pc import PCMorphism
from .rvect import diagonal, euclidean

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


# ---------------------------------------------------------------------------
# The system: a closed autonomous oscillator (an arrangement with a phase parameter).
# ---------------------------------------------------------------------------


def oscillator(
    potential: Callable[[Array], Array], dim: int, m: float = 1.0, label: str = "oscillator",
    parameter=phase,
) -> Arrangement:
    """A closed autonomous oscillator: base ``Q = R^dim`` (position), sharp
    ``(1/m) I``, on-site potential (a cost) ``U(q)``, forcing ``omega = -dU``.

    The parameter is ``parameter(Q)``, by default the phase coalgebra (def:arr,
    sec.integrators); under ``Phi`` (sec.dynamics_functor) the arrangement is then
    the symplectic flow of the Hamiltonian ``H = |p|^2 / (2m) + U(q)``, with state
    ``(q, p)`` in ``R^dim x R^dim``. Quadratic ``U`` gives a linear flow; any other
    gives a nonlinear one.
    """

    return Arrangement(
        P=parameter(diagonal(jnp.full(dim, 1.0 / m))),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -potential(q)),
        label=label,
    )


# The pendulum, as (potential, dim): U(theta) = 1 - cos(theta), genuinely nonlinear,
# and bounded in the libration regime so the symplectic step stays stable. Stiffer
# potentials (e.g. a double well, force ~q^3) need a gentler step (larger m).
PENDULUM: Tuple[Callable[[Array], Array], int] = (lambda q: 1.0 - jnp.cos(q[0]), 1)


# ---------------------------------------------------------------------------
# The trajectory of the closed coalgebra, and the <o, o.c> data stream.
# ---------------------------------------------------------------------------


def trajectory(O: PCMorphism, q0: Array, p0: Array, T: int) -> List[Array]:
    """Iterate the closed coalgebra ``c = O`` from ``(q0, p0)`` for ``T`` steps
    (ex.y_coalgebra_trajectory); return the states ``s_t = (q_t, p_t)`` as flat
    ``R^{2 dim}`` arrays."""

    state = (jnp.asarray(q0, float), jnp.asarray(p0, float))
    out = [jnp.concatenate(state)]
    for _ in range(T):
        _, _, state = O.with_state(state).run_one(_IN_POS, _TRIV)
        out.append(jnp.concatenate(state))
    return out


def consecutive_pairs(states: Sequence[Array]) -> List[Tuple[Array, Array]]:
    """The ``<o, o.c>`` stream with ``o = id``: the pairs ``(s_t, s_{t+1})``."""
    return [(states[t], states[t + 1]) for t in range(len(states) - 1)]


# ---------------------------------------------------------------------------
# The predictor: a small tanh MLP, as a parameterized map (sec.neural_network).
# ---------------------------------------------------------------------------


def mlp(hidden: int, in_dim: int, out_dim: int) -> Tuple[Callable[[Array, Array], Array], int]:
    """``F(q, x) = W2 tanh(W1 x + b1) + b2`` with params flattened into ``q``;
    returns ``(F, n_params)``."""

    shapes = [(hidden, in_dim), (hidden,), (out_dim, hidden), (out_dim,)]
    sizes = [int(np.prod(s)) for s in shapes]
    n = sum(sizes)

    def unpack(q):
        out, i = [], 0
        for s, k in zip(shapes, sizes):
            out.append(q[i : i + k].reshape(s))
            i += k
        return out

    def F(q, x):
        W1, b1, W2, b2 = unpack(q)
        return W2 @ jnp.tanh(W1 @ x + b1) + b2

    return F, n


# ---------------------------------------------------------------------------
# Identify: generate trajectories, train a net to predict the next state.
# ---------------------------------------------------------------------------


def _sample_ic(rng, dim: int, ic_scale: float):
    """An initial ``(q0, p0)`` drawn uniformly from ``[-ic_scale, ic_scale]``.

    Uniform (not Gaussian) keeps the energy bounded, so a pendulum stays in the
    *libration* regime where the state is bounded and the one-step map is learnable;
    Gaussian tails would reach the rotating regime, where ``q`` grows without bound.
    """
    return (
        rng.uniform(-ic_scale, ic_scale, dim),
        rng.uniform(-ic_scale, ic_scale, dim),
    )


def identify(
    system: Tuple[Callable[[Array], Array], int] = PENDULUM,
    *,
    m: float = 1.0,
    n_traj: int = 40,
    T: int = 40,
    ic_scale: float = 1.0,
    hidden: int = 32,
    eta: float = 6e-3,
    steps: int = 8000,
    seed: int = 0,
):
    """Train a net to predict the one-step map of the oscillator ``system``.

    Returns ``(params, F, history, O, dim, m)``: the trained weights, the net,
    the per-step training loss, the system coalgebra ``O = Phi(oscillator(...))``
    (phase parameter), and the configuration dimension and mass. The learner's
    parameter is the configuration coalgebra on ``(R^n, eta * I)``.
    """

    potential, dim = system
    O = Phi(oscillator(potential, dim, m))
    d = 2 * dim
    rng = np.random.default_rng(seed)

    data: List[Tuple[Array, Array]] = []
    for _ in range(n_traj):
        q0, p0 = _sample_ic(rng, dim, ic_scale)
        data += consecutive_pairs(trajectory(O, q0, p0, T))
    rng.shuffle(data)

    F, n = mlp(hidden, d, d)
    init = 0.1 * jnp.asarray(rng.standard_normal(n))
    arr = parameterized_map(F, configuration(euclidean(n, eta)), in_dim=d, out_dim=d)
    params, history = train(arr, init, data, steps=steps)
    return params, F, history, O, dim, m


def one_step_error(
    params: Array,
    F: Callable[[Array, Array], Array],
    O: PCMorphism,
    dim: int,
    m: float,
    *,
    ic_scale: float = 1.0,
    T: int = 40,
    n_traj: int = 8,
    seed: int = 999,
) -> Tuple[float, float]:
    """Mean one-step prediction error of ``F`` on held-out trajectories, and the
    error of the naive "no change" baseline ``s_{t+1} ~ s_t`` for comparison."""

    rng = np.random.default_rng(seed)
    errs, base = [], []
    for _ in range(n_traj):
        q0, p0 = _sample_ic(rng, dim, ic_scale)
        for s, s_next in consecutive_pairs(trajectory(O, q0, p0, T)):
            errs.append(float(jnp.linalg.norm(F(params, s) - s_next)))
            base.append(float(jnp.linalg.norm(s - s_next)))
    return float(np.mean(errs)), float(np.mean(base))


# ---------------------------------------------------------------------------
# The pendulums wired to their learner (plan P43 (b) job 5): no data passed in code.
# ---------------------------------------------------------------------------
#
# A box reads out only its configuration; a phase pendulum's is its presented position
# a_t = q_t + p_t/m, which is q_{t+1}. So the learner sees what the pendulum presents, and learns
# to predict a_t from (a_{t-2}, a_{t-1}): two successive positions carry the momentum. Delay boxes
# hold the past positions; one parameter box's weights are copied to a net per pendulum, so the
# forces from all the pendulums add; each loss is one-way, so no pendulum is moved by the learner.


def observed_oscillator(potential: Callable[[Array], Array], m: float, q0: float, p0: float) -> Arrangement:
    """A one-dimensional pendulum ``<0|0> -> <0|1>`` that reads out its presented position."""
    return Arrangement(
        P=phase(diagonal(jnp.full(1, 1.0 / m)), init=(jnp.array([float(q0)]), jnp.array([float(p0)]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -potential(q)),
        label="pendulum",
    )


def delay_box(x0: float = 0.0) -> Arrangement:
    """A one-step delay ``<0|0> -> <1|1>``: presents what it heard on the previous step.

    Configuration ``Q = R^2``, ``(h, z)``: it presents ``h``; its 1-form puts ``n - z`` on ``z``, ``n``
    the value it hears, and nothing on its input (it hears without pulling); its integrator sets
    both coordinates to ``z + xi_z = n`` and ignores the push ``xi_h`` that comes back on what it
    presents, as the book's datum box ignores the force that comes back.
    """
    from .coalgebra import CotCoalg

    P = CotCoalg(
        Q_dim=2, init=jnp.array([float(x0), float(x0)]),
        position=lambda s: s,
        step=lambda s, xi: jnp.full(2, s[1] + xi[1]),
        label="delay",
    )

    def omega(q, m_out, n_in):
        return jnp.array([0.0, n_in[0] - q[1]]), jnp.zeros_like(m_out), jnp.zeros_like(n_in)

    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q[:1], in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega, label="delay",
    )


def one_way_squared_error(n: int = 1) -> Arrangement:
    """``<0|n> (x) <0|n> -> I`` returning ``l - y`` to the answer ``y`` and nothing to the label ``l``."""
    from .wiring import trivial_parameter

    def omega(q, m, nn):
        y, lab = m[:n], m[n:]
        return jnp.zeros_like(q), jnp.concatenate([lab - y, jnp.zeros(n)]), jnp.zeros_like(nn)

    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=2 * n, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, nn: jnp.zeros(0),
        omega=omega, label="one-way loss",
    )


def wired_identification(
    starts: Sequence[Tuple[float, float]],
    *,
    potential: Callable[[Array], Array] = PENDULUM[0],
    m: float = 1.0,
    hidden: int = 32,
    eta: float = 1.5e-4,
    seed: int = 0,
) -> Arrangement:
    """The closed arrangement: a pendulum per start ``(q0, p0)``, two delays each, one parameter box
    for the net's weights (response ``eta``), a copy of the net per pendulum predicting the pendulum's
    presented position from the two delayed ones, and a one-way loss per pendulum.

    Leaves in order: per pendulum ``[pendulum, delay1, delay2]``, then the weights. Its position is
    ``(a_i, h1_i, z1_i, h2_i, z2_i)`` per pendulum (a the presented position), then the weights.
    """
    from .network import map_box
    from .wiring import compose_seq, tensor_arrangements, trivial_parameter

    K = len(starts)
    F, n = mlp(hidden, 2, 1)
    rng = np.random.default_rng(seed)
    init = 0.1 * jnp.asarray(rng.standard_normal(n))
    weights = Arrangement(
        P=configuration(euclidean(n, eta), init=init), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=n,
        out_f=lambda q, m_out: q, in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (jnp.zeros_like(q), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label="weights",
    )
    leaves = []
    for q0, p0 in starts:
        leaves += [observed_oscillator(potential, m, q0, p0), delay_box(), delay_box()]
    parts = tensor_arrangements(leaves + [weights])
    # parts' outputs: per pendulum [a, d1, d2] (3K), then the weights (n); inputs: per pendulum [d1, d2]
    W = 3 * K
    out_idx = []
    for i in range(K):
        out_idx += list(range(W, W + n)) + [3 * i + 2, 3 * i + 1, 3 * i]   # weights, d2, d1, label a
    out_idx = jnp.asarray(out_idx, dtype=int)
    in_idx = jnp.asarray([j for i in range(K) for j in (3 * i, 3 * i + 1)], dtype=int)  # d1 hears a, d2 hears d1
    route = Arrangement(
        P=trivial_parameter(), in_dim_M=2 * K, out_dim_M=W + n, in_dim_N=0, out_dim_N=len(out_idx),
        out_f=lambda q, m_out: m_out[out_idx], in_f=lambda q, m_out, n_in: m_out[in_idx],
        omega=lambda q, m_out, n_in: (jnp.zeros_like(q), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label="route",
    )
    net = map_box(lambda v: F(v[:n], v[n:n + 2]), n + 2, 1, "net")
    nets = tensor_arrangements([tensor_arrangements([net, map_box(lambda v: v, 1, 1, "id")])
                                for _ in range(K)])
    losses = tensor_arrangements([one_way_squared_error(1) for _ in range(K)])
    return compose_seq(compose_seq(compose_seq(parts, route), nets), losses)


def wired_weights(arr: Arrangement, state, K: int) -> Array:
    """The weights held in the wired arrangement's state."""
    return arr.P.position(state)[5 * K:]


def wired_presented(arr: Arrangement, state, K: int) -> Array:
    """Each pendulum's presented position."""
    return arr.P.position(state)[:5 * K].reshape(K, 5)[:, 0]
