"""Science at two scales: one mind and one population as the same arrangement drawn two ways.

Predictions, fixed before this file existed, and verdicts in
``notes/SCIENCE-SCALES-PREDICTIONS-2026-10-01.md`` (tests (a)-(f)).

THE MIND (``mind_step``). ``choice.belief_arrangement``'s construction, the CLOSED version rebuilt
per datum: values ``v_i = log p(x | h_i)``, temperature ``T = 1``, ``init`` the current natural
parameter ``lam in R^{n-1}``. Its forcing is ``G t``, ``t_i = log p(x|h_i) - log p(x|h_n)``, its
response ``G^{-1}``, so one step at duration 1 is Bayes' rule, ``lam |-> lam + t`` -- computed by the
library as ``G^{-1} (G t)``, which is where floating point enters.

THE POPULATION. ``n`` holders, holder ``i`` carrying ``h_i``, state ``y_i = log N_i`` (log
abundance), Euclidean response ``eta = 1``, exact forcing ``d(log p(x|h_i) * y_i)`` -- a constant
one-form, so one step is ``y_i |-> y_i + log p(x|h_i)``: growth by the fitness ``p(x|h_i)``, the
discrete-time replicator in the log chart. Drawn two ways (clustering is free):
``population_tensor`` is the monoidal product of ``n`` one-dimensional holder boxes
(``wiring.tensor_arrangements``), ``population_single`` one ``n``-dimensional box. Both are closed
and rebuilt per datum, like the mind.

KEEPING AN IDEA (``holder_chooser``, test (b)). ``choice.py``'s simplex chooser on ``{hold, drop}``
(Shahshahani response, entropy forcing), whose cost is ``p_h (K_hold + c) + p_d K_drop`` with
``K_drop = 2 log 2L`` (a uniform default on ``[-L, L]`` for one datum from each of two sources) and
``K_hold = 1/2 S_A + 1/2 Pi_B S_B - 1/2 log Pi_B + log 2 pi``. In the learned arm the chooser is
tensored with a Euclidean box on ``ell = log Pi_B``, one closed box with one potential, so the force on
``ell`` is ``p_h`` times ``-dK_hold/d ell``. The values carry a constant ``B`` (a constant on the
simplex) that keeps ``choice.py``'s CONDITION ``0 < dt T (log Z - 1) < 2``.

HOLDERS WHO READ EACH OTHER (``account_community``, test (c)). ``m`` holders, holder ``j``'s belief
in the gauge-free log-weight chart ``y_j in R^n`` (``q_j = softmax(y_j)``; NOT the gauge-fixed
inverse-Fisher chart, whose step fails once its reference value is thin, test (a1) and
``notes/BAYES-EVERY-BELIEF.md``), Euclidean response, so the steps derived on paper under
the inverse-Fisher response (the note's §3) are translations here. Each holder's cost reads its
neighbours' accounts ``log q_k`` forward as constants (``jax.lax.stop_gradient``: nothing pushed
back), so the one-form is the block of each holder's own cost differential, a game's unilateral
forcing, not exact. ``arm="agree"``: ``alpha/2 sum_k W_jk |y_j - log q_k|^2``; ``arm="evidence"``:
``-alpha sum_k W_jk <log q_k, y_j>``; both plus the private datum ``-beta <log p(x_j|h), y_j>``. A
clock coordinate (exact forcing ``d(clock)``, so it advances by 1 per step) indexes the data.

NOTHING NEW WITHOUT VARIATION (test (d)). Grid hypotheses, mutation ``M`` (``eps/2`` to each grid
neighbour, reflecting ends, so ``M`` is symmetric). ``mutation_selection``: the population in the
abundance chart, one datum per step, ``N' = D_l (M N)``, realized as ``configuration`` with the
constant response ``diag(l)`` and the EXACT forcing ``d(1/2 N^T (M - D_l^{-1}) N)``; exact because
``M`` is symmetric, and cheap (any linear step whose ``A - I`` has real eigenvalues and is
diagonalizable has one). ``drift_filter``: the mind in the probability chart, ``p' = D_L M p / Z``,
realized directly (Euclidean response, forcing ``F(p) - p``, NOT exact; no exact one was found).

AN IDEA AGAINST A RIVAL (``rival_chooser``, test (e)). Test (b)'s construction on ``{h0, h1}``:
the Shahshahani simplex chooser tensored with Euclidean trust boxes ``ell = log Pi``, one per learned
(hypothesis, source) pair, under one potential ``U = -sum_i p_i K_i(ell) + T H(p) + B sum p`` with
TOTAL costs ``K_i = sum_X n_X/2 (e^ell S_X^i - ell + log 2 pi)``. ``rival_settle`` is one jitted
function of the data's statistics, so the arrangement rebuilt at each B count is traced, not
recompiled.

A COMMUNITY AGAINST ONE MIND (test (f)). Holders publish papers (summaries) or conclusions; the
bookkeeping of papers, held sets and replication trials is plain code, because a held set that changes
is a changeable interface, which the library does not have. The beliefs alone are the arrangement:
each step ``translate_beliefs`` moves every holder's log-weights by that step's increment ((a)'s
population drawing: Euclidean response, exact constant forcing, duration 1), one jitted traced build.

What this module does not do: make fitness anything but a likelihood. That organisms ARE
hypotheses is David's candidate reading, an analogy here (the note's §1).
"""

from __future__ import annotations

from typing import Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement
from .choice import _closed, belief_arrangement, belief_to_distribution, choice_step
from .coalgebra import configuration
from .functors import Phi
from .active_inference import run
from .replicator import shahshahani
from .rvect import constant, diagonal, euclidean
from .wiring import tensor_arrangements


def gaussian_loglik(x: float, mus: Array, sigma: float = 1.0) -> Array:
    """``log p(x | h_i)`` for ``h_i = N(mu_i, sigma^2)``, constants included."""
    mus = jnp.asarray(mus, dtype=float)
    return -0.5 * ((x - mus) / sigma) ** 2 - jnp.log(sigma) - 0.5 * jnp.log(2.0 * jnp.pi)


def _one_step(arr: Arrangement) -> Array:
    """One step of the closed ``Phi(arr)`` from its init, read out as a position (a tensor
    product's state is a nested pair; its position is the concatenation)."""
    return arr.P.position(choice_step(Phi(arr), arr.P.init))


# --- the mind ---------------------------------------------------------------


def mind_step(lam: Array, loglik: Array) -> Array:
    """One datum: the closed ``belief_arrangement`` with ``v = loglik``, ``T = 1``, ``init = lam``,
    stepped once at duration 1."""
    return _one_step(belief_arrangement(loglik, 1.0, init=lam))


# --- the population ---------------------------------------------------------


def holder(fitness_log: float, y0: float) -> Arrangement:
    """One holder: state ``y = log N``, Euclidean response, exact forcing ``d(fitness_log * y)``."""
    f = float(fitness_log)
    return _closed(configuration(euclidean(1, 1.0), init=jnp.array([y0], dtype=float)),
                   lambda y: f * y[0], "holder")


def population_tensor(loglik: Array, y: Array) -> Arrangement:
    """``n`` one-dimensional holder boxes side by side, as one arrangement."""
    return tensor_arrangements([holder(loglik[i], y[i]) for i in range(len(y))])


def population_single(loglik: Array, y: Array) -> Arrangement:
    """One ``n``-dimensional box: Euclidean response, exact forcing ``d<loglik, y>``."""
    ll = jnp.asarray(loglik, dtype=float)
    return _closed(configuration(euclidean(len(y), 1.0), init=jnp.asarray(y, dtype=float)),
                   lambda yy: jnp.dot(ll, yy), "population")


def population_step(y: Array, loglik: Array, drawn: str = "tensor") -> Array:
    """One datum on the population, drawn as ``"tensor"`` or ``"single"``."""
    build = population_tensor if drawn == "tensor" else population_single
    return _one_step(build(loglik, y))


def abundance_share(y: Array) -> Array:
    """``softmax(y)``: the finite population's shares."""
    y = jnp.asarray(y, dtype=float)
    return jnp.exp(y - jnp.max(y)) / jnp.sum(jnp.exp(y - jnp.max(y)))


def run_two_scales(xs: Sequence[float], mus: Array):
    """Run mind, tensor population and single-box population on one stream from uniform starts.

    Returns ``(lams, ys_tensor, ys_single)``, each with ``len(xs) + 1`` rows."""
    n = len(mus)
    lam, yt, ys = jnp.zeros(n - 1), jnp.zeros(n), jnp.zeros(n)
    L, YT, YS = [lam], [yt], [ys]
    for x in xs:
        ll = gaussian_loglik(float(x), mus)
        lam = mind_step(lam, ll)
        yt = population_step(yt, ll, "tensor")
        ys = population_step(ys, ll, "single")
        L.append(lam); YT.append(yt); YS.append(ys)
    return jnp.stack(L), jnp.stack(YT), jnp.stack(YS)


# --- test (b): keeping an idea ------------------------------------------------


LOG2PI = float(jnp.log(2.0 * jnp.pi))


def k_hold(S_A: float, S_B: float, ell):
    """Mean cost per datum pair of holding ``h``: ``1/2 S_A + 1/2 e^ell S_B - 1/2 ell + log 2 pi``."""
    return 0.5 * S_A + 0.5 * jnp.exp(ell) * S_B - 0.5 * ell + LOG2PI


def holder_chooser(K_hold, K_drop: float, c: float, T: float, learn: bool,
                   dt: float = 0.08, eta: float = 1.0, B: float = None) -> Arrangement:
    """The {hold, drop} chooser, closed. ``K_hold`` is a function of ``ell``; with ``learn=False``
    the state is ``p`` only and ``K_hold(0)`` is used (``Pi_B = 1``); with ``learn=True`` the state is
    ``(p, ell)`` with ``ell`` descended (Euclidean, ``eta``), starting at 0."""
    B = K_drop + 2.0 * T if B is None else B
    chooser = configuration(shahshahani(2, dt), init=jnp.full(2, 0.5))

    def U_of(p, ell):
        cost = p[0] * (K_hold(ell) + c) + p[1] * K_drop
        return -cost - T * jnp.sum(p * jnp.log(p)) + B * jnp.sum(p)

    if not learn:
        return _closed(chooser, lambda p: U_of(p, 0.0), "hold/drop fixed")
    P = chooser.tensor(configuration(euclidean(1, eta), init=jnp.zeros(1)))
    return _closed(P, lambda q: U_of(q[:2], q[2]), "hold/drop learned")


def rest(arr: Arrangement, steps: int = 3000) -> Array:
    """The position after ``steps`` jitted steps from the parameter's init."""
    return arr.P.position(run(arr, steps))


def hold_share_closed(K_drop: float, K_hold_star: float, c: float, T: float) -> float:
    """(b2)'s closed form ``1 / (1 + exp(-(K_drop - K_hold* - c)/T))``."""
    return float(1.0 / (1.0 + jnp.exp(-(K_drop - K_hold_star - c) / T)))


# --- test (c): holders who read each other -----------------------------------


def account_community(Y0: Array, W: Array, alpha: float, beta: float, loglik: Array,
                      arm: str) -> Arrangement:
    """The closed community: state ``(Y.ravel(), clock)``; ``loglik[t, j]`` is holder ``j``'s
    datum log-likelihood vector at step ``t``."""
    Y0 = jnp.asarray(Y0, dtype=float)
    m, n = Y0.shape
    W = jnp.asarray(W, dtype=float)
    LL = jnp.asarray(loglik, dtype=float)

    def potential(z):
        Y, clock = z[:-1].reshape(m, n), z[-1]
        LQ = jax.lax.stop_gradient(jax.nn.log_softmax(Y, axis=1))   # accounts, read forward
        data = jax.lax.stop_gradient(LL[jnp.asarray(clock, dtype=int)])
        if arm == "agree":
            cost = 0.5 * alpha * jnp.sum(W[:, :, None] * (Y[:, None, :] - LQ[None, :, :]) ** 2)
        elif arm == "evidence":
            cost = -alpha * jnp.sum((W @ LQ) * Y)
        else:
            raise ValueError(arm)
        cost = cost - beta * jnp.sum(data * Y)
        return -cost + clock                       # U = -cost; the clock's exact d(clock)

    grad = jax.grad(potential)
    z0 = jnp.concatenate([Y0.ravel(), jnp.zeros(1)])
    return Arrangement(
        P=configuration(euclidean(m * n + 1, 1.0), init=z0),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (grad(q), jnp.zeros(0), jnp.zeros(0)),
        label=f"community({arm}, alpha={alpha:g}, beta={beta:g})",
    )


def community_run(arr: Arrangement, steps: int, m: int, n: int):
    """``Y`` after each step, ``steps + 1`` arrays of shape ``(m, n)``."""
    _, rec = run(arr, steps, record=lambda z: z[:-1].reshape(m, n))
    return jnp.stack(rec)


def ring_W(m: int) -> Array:
    W = np.zeros((m, m))
    for j in range(m):
        W[j, (j + 1) % m] = W[j, (j - 1) % m] = 0.5
    return jnp.asarray(W)


def all_to_all_W(m: int) -> Array:
    return jnp.asarray((np.ones((m, m)) - np.eye(m)) / (m - 1))


# --- test (d): nothing new without variation ---------------------------------


def mutation_matrix(K: int, eps: float) -> Array:
    """``eps/2`` of each type to each grid neighbour; at an end the outgoing half stays."""
    M = np.eye(K) * (1.0 - eps)
    for j in range(K):
        for k in (j - 1, j + 1):
            if 0 <= k < K:
                M[k, j] += eps / 2
            else:
                M[j, j] += eps / 2
    return jnp.asarray(M)


def mutation_selection(M: Array, ell: Array, N: Array) -> Arrangement:
    """Mutate then select, one datum: response ``diag(ell)``, exact forcing
    ``d(1/2 N^T (M - D_ell^{-1}) N)``; one step lands on ``D_ell M N``."""
    ell = jnp.asarray(ell, dtype=float)
    S = jnp.asarray(M) - jnp.diag(1.0 / ell)
    return _closed(configuration(constant(jnp.diag(ell)), init=jnp.asarray(N, dtype=float)),
                   lambda n: 0.5 * n @ S @ n, "mutation-selection")


def drift_filter(M: Array, L: Array, p: Array) -> Arrangement:
    """The Bayes filter with drift ``M``, one datum, in the probability chart: Euclidean response,
    the non-exact forcing ``F(p) - p`` with ``F(p) = D_L M p / (1^T D_L M p)``."""
    M = jnp.asarray(M)
    L = jnp.asarray(L, dtype=float)

    def F(q):
        r = L * (M @ q)
        return r / jnp.sum(r)

    return Arrangement(
        P=configuration(euclidean(len(L), 1.0), init=jnp.asarray(p, dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (F(q) - q, jnp.zeros(0), jnp.zeros(0)),
        label="drift-filter",
    )


def run_variation(xs, mus, eps: float, start: Array):
    """Population and mind on one stream; returns ``(Ns, ps)``, ``len(xs) + 1`` rows each."""
    mus = jnp.asarray(mus, dtype=float)
    M = mutation_matrix(len(mus), eps)
    N = p = jnp.asarray(start, dtype=float)
    Ns, ps = [N], [p]
    for x in xs:
        L = jnp.exp(gaussian_loglik(float(x), mus))
        N = _one_step(mutation_selection(M, L / jnp.max(L), N))
        p = _one_step(drift_filter(M, L, p))
        Ns.append(N); ps.append(p)
    return jnp.stack(Ns), jnp.stack(ps)


# --- test (e): an idea against a rival ----------------------------------------


def rival_chooser(n: Array, S: Array, learned: tuple, T: float, dt, eta: Array, Bshift,
                  p0: Array, ell0: Array) -> Arrangement:
    """``n = (n_A, n_B)``; ``S[i, X]`` the mean square of source ``X`` about ``h_i``'s value;
    ``learned`` the (i, X) pairs whose trust is a descended box (the rest fixed at ``Pi = 1``)."""
    chooser = configuration(shahshahani(2, dt), init=p0)
    P = chooser.tensor(configuration(diagonal(eta), init=ell0)) if learned else chooser

    def K(i, ells):
        tot = 0.0
        for X in (0, 1):
            ell = ells[learned.index((i, X))] if (i, X) in learned else 0.0
            tot = tot + 0.5 * n[X] * (jnp.exp(ell) * S[i, X] - ell + LOG2PI)
        return tot

    def U(q):
        p, ells = q[:2], q[2:]
        return (-(p[0] * K(0, ells) + p[1] * K(1, ells))
                - T * jnp.sum(p * jnp.log(p)) + Bshift * jnp.sum(p))

    return _closed(P, U, f"rival({len(learned)} trust boxes)")


def rival_closed_K(n, S, learned) -> np.ndarray:
    """``K_i*`` with every learned trust at its rest ``1/S`` (for the gauge constant ``B`` only)."""
    out = np.zeros(2)
    for i in (0, 1):
        for X in (0, 1):
            if n[X] == 0:
                continue
            Sx = float(S[i, X])
            body = (1.0 + np.log(Sx)) if (i, X) in learned else Sx
            out[i] += 0.5 * n[X] * (body + LOG2PI)
    return out


def rival_settle_fn(learned: tuple, T: float, steps: int = None):
    """A jitted ``(state, n, S, dt, eta, Bshift[, steps]) -> state`` running ``steps`` steps of
    ``Phi(rival_chooser(...))`` from ``state = (p, ell)`` (or ``p`` with no trust boxes). With
    ``steps=None`` here, the step count is a (traced) argument."""

    def settle_(state, n, S, dt, eta, Bshift, steps_=steps):
        p0, ell0 = (state if learned else (state, jnp.zeros(0)))
        arr = rival_chooser(n, S, learned, T, dt, eta, Bshift, p0, ell0)
        O = Phi(arr)
        return jax.lax.fori_loop(0, steps_, lambda _, s: choice_step(O, s), arr.P.init)

    if steps is None:
        return jax.jit(lambda state, n, S, dt, eta, Bshift, k: settle_(state, n, S, dt, eta, Bshift, k))
    return jax.jit(lambda state, n, S, dt, eta, Bshift: settle_(state, n, S, dt, eta, Bshift))


# --- test (f): a community against one mind -------------------------------------


@jax.jit
def translate_beliefs(Y: Array, inc: Array) -> Array:
    """One step of the closed arrangement on ``Y`` (any shape): Euclidean response, exact forcing
    ``d<inc, Y>``; lands on ``Y + inc``."""
    y0 = Y.ravel()
    g = inc.ravel()
    arr = _closed(configuration(euclidean(y0.shape[0], 1.0), init=y0), lambda y: jnp.dot(g, y), "beliefs")
    return choice_step(Phi(arr), arr.P.init).reshape(Y.shape)


def stats_inc(mus: np.ndarray, N, S) -> np.ndarray:
    """Log-likelihood of ``N`` draws with sum ``S`` under ``N(mu_i, 1)``, less the part common to all
    ``i``: ``mu_i S - N mu_i^2 / 2`` (broadcasts over leading axes of ``N``, ``S``)."""
    N = np.asarray(N, float)[..., None]
    S = np.asarray(S, float)[..., None]
    return mus * S - N * mus ** 2 / 2.0


def weights(y: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
    y = np.where(mask, y, -np.inf) if mask is not None else y
    e = np.exp(y - np.max(y, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def run_summaries(draws: np.ndarray, mus: np.ndarray, A: np.ndarray, provenance: bool, extra: int):
    """Summaries arms. ``draws[t, j]``; ``A`` the 0/1 adjacency. Returns per step (after ``T`` draw
    steps and ``extra`` steps with no draws): ``Ys`` (m, n), and for provenance the paper sets, else the
    accumulated ``(N, S)``."""
    T, m = draws.shape
    Y = jnp.zeros((m, len(mus)))
    Ys, held = [], []
    if provenance:
        papers = [set() for _ in range(m)]
        for t in range(T + extra):
            inc = np.zeros((m, len(mus)))
            if t < T:
                for j in range(m):
                    papers[j].add((j, t)); inc[j] += stats_inc(mus, 1, draws[t, j])
            sent = [set(P) for P in papers]
            for j in range(m):
                for k in np.nonzero(A[j])[0]:
                    for (a, tt) in sent[k] - papers[j]:
                        papers[j].add((a, tt)); inc[j] += stats_inc(mus, 1, draws[tt, a])
            Y = translate_beliefs(Y, jnp.asarray(inc))
            Ys.append(np.asarray(Y)); held.append([frozenset(P) for P in papers])
        return Ys, held
    NS = np.zeros((m, 2))
    for t in range(T + extra):
        add = np.zeros((m, 2))
        if t < T:
            add[:, 0] += 1; add[:, 1] += draws[t]
        after_draw = NS + add
        add += A @ after_draw                       # each reader adds its neighbours' whole stock
        Y = translate_beliefs(Y, jnp.asarray(stats_inc(mus, add[:, 0], add[:, 1])))
        NS = NS + add
        Ys.append(np.asarray(Y)); held.append(NS.copy())
    return Ys, held


def run_conclusions(draws, mus, A, slices, replicate: bool, K=2, r=5, c=0.1):
    """Conclusions arms. Returns per step the masked weights (m, n) and each holder's held set."""
    T, m = draws.shape
    n = len(mus)
    Y = jnp.zeros((m, n))
    held = [list(sl) for sl in slices]
    trials = [dict() for _ in range(m)]            # h -> [best, acc, count]
    gamma = c * r
    W_out, H_out = [], []
    for t in range(T):
        ll = -0.5 * (draws[t][:, None] - mus[None, :]) ** 2          # (m, n), own draw
        Yd = np.asarray(Y) + ll
        for j in range(m):                                             # replication trials
            for h in list(trials[j]):
                best, acc, cnt = trials[j][h]
                acc += ll[j, h] - ll[j, best]; cnt += 1
                if cnt < r:
                    trials[j][h] = [best, acc, cnt]
                    continue
                del trials[j][h]
                if acc > c * r and h not in held[j]:
                    held[j].append(h)
                    if len(held[j]) > K:
                        held[j].remove(min(held[j], key=lambda i: Yd[j, i]))
        concl = [max(held[j], key=lambda i: Yd[j, i]) for j in range(m)]
        credit = np.zeros((m, n))
        for j in range(m):
            for k in np.nonzero(A[j])[0]:
                h = concl[k]
                if replicate:
                    if h not in held[j] and h not in trials[j]:
                        trials[j][h] = [concl[j], 0.0, 0]
                else:
                    credit[j, h] += gamma
                    if h not in held[j]:
                        held[j].append(h)
        Yn = Yd + credit
        if not replicate:
            for j in range(m):
                while len(held[j]) > K:
                    held[j].remove(min(held[j], key=lambda i: Yn[j, i]))
        Y = translate_beliefs(Y, jnp.asarray(ll + credit))
        mask = np.zeros((m, n), bool)
        for j in range(m):
            mask[j, held[j]] = True
        W_out.append(weights(np.asarray(Y), mask)); H_out.append([list(h) for h in held])
    return W_out, H_out


def run_bounded_mind(draws, mus, slice_):
    """One mind with ``K = len(slice_)`` fixed hypotheses and every holder's draws each step."""
    T, m = draws.shape
    y = jnp.zeros(len(mus))
    mask = np.zeros(len(mus), bool); mask[list(slice_)] = True
    out = []
    for t in range(T):
        y = translate_beliefs(y, jnp.asarray((-0.5 * (draws[t][:, None] - mus[None, :]) ** 2).sum(0)))
        out.append(weights(np.asarray(y), mask))
    return out


__all__ = [
    "translate_beliefs", "stats_inc", "weights", "run_summaries", "run_conclusions", "run_bounded_mind",
    "rival_chooser", "rival_closed_K", "rival_settle_fn",
    "mutation_matrix", "mutation_selection", "drift_filter", "run_variation",
    "account_community", "community_run", "ring_W", "all_to_all_W",
    "k_hold", "holder_chooser", "rest", "hold_share_closed",
    "gaussian_loglik", "mind_step", "holder", "population_tensor", "population_single",
    "population_step", "abundance_share", "run_two_scales", "belief_to_distribution",
]
