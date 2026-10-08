"""The EM coordinator (book plan P37 (b)(iii), ``AGENTDOCPLAN.md`` §8.2).

David (book repo, 2026-10-08), asked whether "the coordinator can put together the stories into a space, like
functions from the various thingies into a single space/manifold ... in such a way that the distribution of force by
the chain rule there is actually what it would give in this accounting thingy? that is, it learns how to account for
phenomena explicitly in the model and that that itself is what mathematically creates the mistake distribution as a
force-passback." This module builds that coordinator above ``coordinator.py``'s figurines and parent. Predictions
fixed before this file existed: the book repo's ``notes/runs/p37-em-coordinator-1a279037/PREDICTIONS.md``; the closed
form the tests check against is ``tests/em_coordinator_closed_form.py``; the tests ``tests/test_em_coordinator.py``.

THE CORRECTION THE BUILD RESTS ON. The chain rule of the expected complete-data cost does NOT distribute the mistake:
by Fisher's identity its gradient at the current guesses is the marginal likelihood's, which depends on the guesses
only through their sum and so is one number on every figurine of a bus (``expected_complete_cost`` and
``marginal_cost`` below; test 0). What distributes is the E-STEP'S LIFT, the posterior mean of each figurine's latent
error, the M-step's minimizer. So the coordinator sends each figurine the GAP between its guess and that lift.

THE COORDINATOR'S MODEL. Figurine ``j``'s true minutes are ``t_j = G_j + eps_j``, one LATENT ERROR per figurine,
``eps_j ~ N(0, v_j)``, ``v_j = V s_j``: the SPLIT ``s = softmax(l)``, a belief over the ``n`` figurines about where
the doubt lies (``belief.belief_coalgebra``, rate ``TAU_S``), and the DOUBT ``V = e^Lambda`` (a configuration
integrator on ``Lambda``, rate ``MU_LAMBDA``). Bus ``b`` arrives at ``t*_b = sum_{j in b} t_j + n_b``,
``n_b ~ N(0, SIGMA0^2)`` (the accident); a sighting of figurine 0 is ``o = t_0 + m``, ``m ~ N(0, SIGMA_S^2)``,
the world's input. ``G_j`` is the guess AS FIRST HEARD (``memory="record"``: a state of the coordinator with zero
response, set to the first guesses) or the CURRENT guess (``memory="none"``: stage (ii)'s depot). Observations
``y = A eps + eta``: a bus row ``y_b = t*_b - sum_{j in b} G_j``, a sighting row ``y_s = o - G_0``;
``Sigma = A diag(v) A^T + N``. Each step the coordinator computes the E-step ``eps_hat = diag(v) A^T Sigma^{-1} y``;
its own state descends the marginal ``k = y^T Sigma^{-1} y / 2 + log det Sigma / 2`` (exact, unilateral); and its
backward map sends down, as the parent's exterior message, the REPORT ``r_j = g_j - G_j - eps_hat_j``, which each
figurine reads as a push on its guess (``coordinator.figurine``: ``k_j = g_j h_j``, force ``-r_j g_j'``, response
``told`` or ``slope``). Memoryless, ``r_j = -eps_hat_j``: stage (ii)'s report with the share computed.

THE COMPOSITE is ``coordinator . parent(figurines) : I -> <E x R^{B+2} | R^0>``; the world's input is
``(t*_1..t*_B, sighting flag, o)``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement
from .attention import input_dim, make_input, message, msg_dim, spatial_arrangement, spatial_config
from .belief import belief_coalgebra, belief_of
from .coalgebra import configuration
from .coordinator import DP, bus_figurines, figurine, guesses, kind, step_fn, thetas
from .figurines import TAU
from .rvect import constant, euclidean
from .wiring import compose_seq, tensor_arrangements

EMPTY = jnp.zeros(0)
SIGMA0, SIGMA_S = 0.1, 0.5
TAU_S, MU_LAMBDA = 0.05, 0.05
V0 = 1.0


class Arm(NamedTuple):
    """What is compiled: the figurines' response, the coordinator's memory, the number of buses."""
    response: str      # "told" or "slope"
    memory: str        # "none" or "record"
    B: int

    @property
    def n(self) -> int:
        return 2 * self.B + 1


# ---------------------------------------------------------------------------
# The model: observations, the E-step, the two costs.
# ---------------------------------------------------------------------------


def incidence(B: int) -> Array:
    """``A`` with a row per bus and one sighting row (figurine 0); ``N`` the noise variances."""
    n = 2 * B + 1
    A = np.zeros((B + 1, n))
    for b in range(B):
        A[b, list(bus_figurines(b))] = 1.0
    A[B, 0] = 1.0
    N = np.array([SIGMA0 ** 2] * B + [SIGMA_S ** 2])
    return jnp.asarray(A), jnp.asarray(N)


def observations(G: Array, n_in: Array, B: int, A: Array):
    """``y`` from the record ``G`` and the world's ``(t*, flag, o)``; the sighting row is masked by the flag
    (its noise made huge when absent, so it carries nothing)."""
    t_star, flag, o = n_in[:B], n_in[B], n_in[B + 1]
    y_bus = t_star - A[:B] @ G
    y_s = flag * (o - G[0])
    return jnp.concatenate([y_bus, jnp.array([y_s])]), flag


def sigma_of(v: Array, A: Array, N: Array, flag) -> Array:
    N_eff = N.at[-1].set(jnp.where(flag > 0.5, N[-1], 1e12))
    return A @ jnp.diag(v) @ A.T + jnp.diag(N_eff)


def e_step(v: Array, y: Array, A: Array, N: Array, flag) -> Array:
    """The posterior mean of every latent error given everything seen: ``diag(v) A^T Sigma^{-1} y``."""
    Sigma = sigma_of(v, A, N, flag)
    return v * (A.T @ jnp.linalg.solve(Sigma, y))


def marginal_cost(v: Array, y: Array, A: Array, N: Array, flag) -> Array:
    """``-log p(y | v, G)`` up to a constant: the negative joint log-likelihood of every observation."""
    Sigma = sigma_of(v, A, N, flag)
    Si_y = jnp.linalg.solve(Sigma, y)
    return 0.5 * y @ Si_y + 0.5 * jnp.linalg.slogdet(Sigma)[1]


def expected_complete_cost(v_new: Array, y_new: Array, v: Array, y: Array, A: Array, N: Array, flag) -> Array:
    """EM's ``Q``: the expectation, under the posterior at ``(v, y)``, of the complete-data cost
    ``-log p(y_new, eps | v_new)``. Fisher's identity: its gradient at ``(v_new, y_new) = (v, y)`` is
    ``marginal_cost``'s (test 0)."""
    Sigma = sigma_of(v, A, N, flag)
    D = jnp.diag(v)
    K = D @ A.T @ jnp.linalg.solve(Sigma, A @ D)
    eps_hat = v * (A.T @ jnp.linalg.solve(Sigma, y))
    C = D - K                                     # the posterior covariance of the latents
    N_eff = N.at[-1].set(jnp.where(flag > 0.5, N[-1], 1e12))
    resid = y_new - A @ eps_hat
    obs = jnp.sum((resid ** 2 + jnp.diag(A @ C @ A.T)) / (2 * N_eff))
    prior = jnp.sum((eps_hat ** 2 + jnp.diag(C)) / (2 * v_new) + 0.5 * jnp.log(v_new))
    return obs + prior


# ---------------------------------------------------------------------------
# The coordinator box.
# ---------------------------------------------------------------------------


def coordinator(arm: Arm, split0: Sequence[float], G0: Sequence[float]) -> Arrangement:
    """``<E | M> -> <E x R^{B+2} | R^0>``: state the split (log-heights), the log-doubt and, with a record, the
    guesses as first heard; the backward map computes the report; the own cost is the marginal, exact."""
    n, B = arm.n, arm.B
    edim, mdim = input_dim(n, DP), msg_dim(n)
    A, N = incidence(B)
    P = belief_coalgebra(n, jnp.log(jnp.asarray(split0, float)), tau=TAU_S)
    P = P.tensor(configuration(euclidean(1, MU_LAMBDA), init=jnp.array([np.log(V0)])))
    if arm.memory == "record":
        P = P.tensor(configuration(euclidean(n, 0.0), init=jnp.asarray(G0, float)))
    qdim = n + 1 + (n if arm.memory == "record" else 0)

    def parts(q, m):
        s, Lam = q[:n], q[n]
        G = q[n + 1:] if arm.memory == "record" else m[:n]
        return s, jnp.exp(Lam) * s, G

    def report(q, m, n_in):
        _, v, G = parts(q, m)
        y, flag = observations(G, n_in, B, A)
        eps_hat = e_step(v, y, A, N, flag)
        return m[:n] - G - eps_hat

    def in_f(q, m, n_in):
        return make_input(message(report(q, m, n_in), 1.0), jnp.zeros(DP))

    def k(q, m, n_in):
        _, v, G = parts(q, m)
        y, flag = observations(G, n_in, B, A)
        return marginal_cost(v, y, A, N, flag)

    def omega(q, m, n_in):
        xi_q = jax.grad(lambda q_: -k(q_, m, n_in))(q)
        return (xi_q, jnp.zeros(mdim), jnp.zeros(B + 2))

    return Arrangement(
        P=P,
        in_dim_M=edim, out_dim_M=mdim, in_dim_N=B + 2, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0), in_f=in_f, omega=omega,
        label=f"coordinator({arm.memory},B={B})",
    )


# ---------------------------------------------------------------------------
# The composite, its state and the world.
# ---------------------------------------------------------------------------


class Built(NamedTuple):
    arr: Arrangement
    inner: Arrangement
    box: Arrangement


def coordinated(arm: Arm, theta0: Sequence[float], split0: Sequence[float], G0: Sequence[float]) -> Built:
    """``coordinator . spatial(n)(figurines) : I -> <E x R^{B+2} | R^0>``, placements frozen at 0."""
    n = arm.n
    figs = tensor_arrangements([figurine(j, n, float(theta0[j]), arm.response) for j in range(n)])
    n_Q = n * DP + n * n * n
    parent = spatial_arrangement(
        n, n, DP, TAU,
        parameter=configuration(constant(jnp.zeros((n_Q, n_Q))),
                                init=spatial_config(jnp.zeros((n, DP)), (jnp.eye(n),) * n)),
        label=f"spatial({n})")
    inner, box = compose_seq(figs, parent.arr()), coordinator(arm, split0, G0)
    return Built(compose_seq(inner, box), inner, box)


def coordinator_state(state, arm: Arm):
    """``(split, V, G)`` read off the composite's state (``G`` is ``None`` without a record)."""
    s = state[1]
    if arm.memory == "record":
        (l, Lam), G = s
    else:
        l, Lam = s
        G = None
    return belief_of(l), jnp.exp(Lam[0]), G


def exterior(B: int, t_star: Sequence[float], sighting: bool, o: float) -> Array:
    return jnp.concatenate([jnp.asarray(t_star, float), jnp.array([1.0 if sighting else 0.0, o])])


def report_fn(built: Built):
    inner, box = built.inner, built.box

    def report(s, ext):
        m = inner.out_f(inner.P.position(s[0]), EMPTY)
        return box.in_f(box.P.position(s[1]), m, ext)

    return jax.jit(report)


def run_rounds(built: Built, arm: Arm, t_star: Sequence[float], rounds: int, sighting: bool, o: float,
               state=None):
    """Run ``rounds`` outcome steps. Returns arrays over steps ``0..R`` of the figurines' states, guesses, the split,
    the doubt, the covector the cost sends on the split, and the report, and the final state."""
    arr = built.arr
    step, report = step_fn(arr), report_fn(built)
    n = arm.n
    s = arr.P.init if state is None else state
    ext = exterior(arm.B, t_star, sighting, o)
    out = {"theta": [], "g": [], "s": [], "V": [], "xi_s": [], "report": []}
    qdim = n + 1 + (n if arm.memory == "record" else 0)
    for t in range(rounds + 1):
        th = thetas(s, n)
        split, V, _ = coordinator_state(s, arm)
        s_next, xi_q = step(s, ext)          # the covector is computed at the current state
        out["theta"].append(np.asarray(th))
        out["g"].append(np.asarray(guesses(th, n)))
        out["s"].append(np.asarray(split))
        out["V"].append(float(V))
        out["xi_s"].append(np.asarray(xi_q[-qdim:][:n]))
        out["report"].append(np.asarray(report(s, ext)[:n]))
        if t == rounds:
            break
        s = s_next
    return {k: np.array(v) for k, v in out.items()}, s
