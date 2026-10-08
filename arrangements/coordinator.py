"""The coordinator that distributes the mistake (book plan P37 (b)(ii), ``AGENTDOCPLAN.md`` §8.2).

David (book repo, 2026-10-08), asked where a figurine's piece of the mistake comes from: "from the coordinating
box, the one they're all a part of, telling each of them what they were wrong about. it's their job to distribute
the mistake to the right people"; the evidence that feeds it, ranked: "if we saw it, that's the most reliable and
so we have a strong belief about it. if it's just what we already knew, that's something. comparing with other
plans also works too." This module builds that out of item 65's figurines (``figurines.py``), the spatial
attention parent and the listener run's belief integrator (``belief.belief_coalgebra``). Predictions fixed before
this file existed: the book repo's ``notes/runs/p37-coordinator-70fcc5f2/PREDICTIONS.md``; the closed form the
tests check against is ``tests/coordinator_closed_form.py``; the tests are ``tests/test_coordinator.py``.

Minutes throughout; a bus is scheduled at 0; a round is one outcome step (item 65's plan and margin are dropped).

THE FIGURINES. ``B`` buses share one ROUTE figurine; bus 0 has figurines 0 (its driver), 1 (its vehicle) and 2
(the route), item 65's three in its order; bus ``b >= 1`` has ``2b + 1`` (its driver), ``2b + 2`` (its vehicle)
and 2. Each is item 65's box ``I -> B``: one state ``theta_j``, a sending map that runs its mechanism
(``figurines.RUNS``) and sends its guess in slot ``j`` of ``H = R^n``, ``n = 2B + 1``. Its cost reads the REPORT
in its received slot ``j``, which carries only the exterior's slot ``j`` (no other child writes it):
``k_j = g_j h_j``, linear, the report read as a push on the guess; forcing UNILATERAL (reading returns no push),
so the force on ``theta_j`` is ``-h_j g_j'``. Two RESPONSES: ``told``, ``sharp = mu_g / g_j'(theta)^2``, so the
guess moves by ``mu_g`` times the push, to first order (a Gauss-Newton response: the figurine takes the report in
the report's units); ``slope``, item 65's Euclidean ``mu_theta`` on the parameter.

THE PARENT is the spatial attention parent over the ``n`` figurines, every placement frozen at 0 (invitation 1).

THE DEPOT is the box above the parent, the coordinator: ``<E | M> -> <E x R^{1 + B + 6B} | R^0>``. Its state is
``B`` beliefs ``y^b`` over bus ``b``'s three figurines (which one was wrong: the driver, the vehicle, the route),
one belief integrator per bus, tensored. It reads each bus's prediction ``t_hat_b = sum_{j in b} g_j`` off the
parent's message and the world's ``t*_b`` off its input, and its backward map COMPUTES THE REPORT and sends it down
as the parent's exterior message: on an outcome step (flag 1) slot ``j`` carries ``r_j = sum_{b contains j}
s^b_j e_b``, ``e_b = t_hat_b - t*_b`` the bus's mistake, the share ``s^b_j = y^b_j`` (``blame="belief"``) or ``1``
(``blame="chain"``: everyone the whole push, which the parent delivers and each figurine's run pulls back as item
65's chain rule exactly; ``test_account.py`` prediction 21's kind of report). The depot's own cost is the belief's
EVIDENCE, unilateral in the belief (the covector on the belief is the log-likelihood, so one step of the belief
integrator is Bayes' rule), gated by the flag: a SIGHTING (the world's input, a one-hot over "saw figurine ``j``
wrong" / "saw ``j`` right" for the bus's three); the PRIOR (the belief's initial state); and COMPARISON (``B = 2``):
under "the route" the other bus carries the same mistake, ``e_other ~ N(e_b, sigma^2)``; under "my driver" or "my
vehicle" the other bus is right, ``e_other ~ N(0, sigma^2)``.

THE COMPOSITE is ``depot . parent(figurines) : I -> <E x R^{1 + B + 6B} | R^0>``; the world's input is
``(flag, t*_1..t*_B, sightings)``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement
from .attention import input_dim, make_input, message, msg_dim, spatial_arrangement, spatial_config
from .belief import belief_coalgebra, belief_of
from .coalgebra import CotCoalg, configuration
from .figurines import RUNS, TAU
from .rvect import ReactiveVectorSpace, constant, euclidean
from .wiring import compose_seq, tensor_arrangements

EMPTY = jnp.zeros(0)
DP = 1
MU_G, MU_THETA = 0.1, 0.02
SIGMA = 2.0
SEE_WRONG = (0.9, 0.05)       # Pr(saw j wrong | z_j), Pr(saw j wrong | z_i), i != j
SEE_RIGHT = (0.1, 0.95)       # Pr(saw j right | z_j), Pr(saw j right | z_i), i != j
KINDS = ("driver", "vehicle", "route")


class Arm(NamedTuple):
    """What is compiled: the figurines' response, the number of buses, whose shares the depot sends."""
    response: str      # "told" or "slope"
    B: int
    blame: str         # "belief" or "chain"

    @property
    def n(self) -> int:
        return 2 * self.B + 1

    @property
    def comparison(self) -> bool:
        return self.B == 2


def kind(j: int) -> int:
    """Which mechanism figurine ``j`` is: 0 the driver, 1 the vehicle, 2 the route."""
    return 2 if j == 2 else (j if j < 2 else (j - 3) % 2)


def bus_figurines(b: int) -> Tuple[int, int, int]:
    """Bus ``b``'s (driver, vehicle, route)."""
    return (0, 1, 2) if b == 0 else (2 * b + 1, 2 * b + 2, 2)


# ---------------------------------------------------------------------------
# The figurines, with their two responses.
# ---------------------------------------------------------------------------


def told_response(run, mu_g: float) -> ReactiveVectorSpace:
    """``sharp(theta) = mu_g / g'(theta)^2`` on ``R``: the guess moves by ``mu_g`` times a push on it."""
    dg = jax.grad(run)
    return ReactiveVectorSpace(dim=1, sharp_fn=lambda th: jnp.reshape(mu_g / dg(th[0]) ** 2, (1, 1)))


def figurine(j: int, n: int, theta0: float, response: str) -> Arrangement:
    """Figurine ``j`` of ``n`` as a box ``I -> B``: state ``theta``, guess ``RUNS[kind(j)](theta)`` sent in slot
    ``j``; the report read off its received slot ``j`` as a push on the guess, unilaterally."""
    run = RUNS[kind(j)]
    slot = jnp.zeros(n).at[j].set(1.0)
    edim = input_dim(n, DP)
    Q = told_response(run, MU_G) if response == "told" else euclidean(1, MU_THETA)

    def send(theta, m_out):
        return message(run(theta[0]) * slot, 1.0)

    def omega(theta, m_out, e):
        xi_theta = jax.grad(lambda th: -run(th[0]) * e[j])(theta)
        return (xi_theta, jnp.zeros(0), jnp.zeros(edim))

    return Arrangement(
        P=configuration(Q, init=jnp.array([theta0], float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=edim, out_dim_N=msg_dim(n),
        out_f=send, in_f=lambda theta, m_out, e: jnp.zeros(0), omega=omega,
        label=f"{KINDS[kind(j)]}[{j}]({response})",
    )


# ---------------------------------------------------------------------------
# The depot: beliefs, the report, the evidence.
# ---------------------------------------------------------------------------


def sighting_nll() -> Array:
    """``N[i, o] = -log Pr(o | z_i)`` for the six sightings ``o`` of a bus's three figurines: ``o = j`` saw ``j``
    wrong, ``o = 3 + j`` saw ``j`` right."""
    N = np.zeros((3, 6))
    for i in range(3):
        for j in range(3):
            N[i, j] = -np.log(SEE_WRONG[0] if i == j else SEE_WRONG[1])
            N[i, 3 + j] = -np.log(SEE_RIGHT[0] if i == j else SEE_RIGHT[1])
    return jnp.asarray(N)


def comparison_nll(e_b, e_other) -> Array:
    """``-log`` likelihood of the other bus's mistake under bus ``b``'s three hypotheses."""
    own = e_other ** 2 / (2 * SIGMA ** 2)
    shared = (e_other - e_b) ** 2 / (2 * SIGMA ** 2)
    return jnp.stack([own, own, shared])


def depot_input_dim(B: int) -> int:
    return 1 + B + 6 * B


def depot(arm: Arm, priors: Sequence[Sequence[float]]) -> Arrangement:
    """``<E | M> -> <E x R^{1 + B + 6B} | R^0>``: state the ``B`` beliefs (log-heights), one belief integrator per
    bus, tensored; the backward map computes the report; the cost is the evidence, unilateral."""
    n, B = arm.n, arm.B
    edim, mdim = input_dim(n, DP), msg_dim(n)
    N_s = sighting_nll()
    P = belief_coalgebra(3, jnp.log(jnp.asarray(priors[0], float)))
    for b in range(1, B):
        P = P.tensor(belief_coalgebra(3, jnp.log(jnp.asarray(priors[b], float))))

    def unpack(n_in):
        flag = n_in[0]
        t_star = n_in[1:1 + B]
        sight = n_in[1 + B:].reshape(B, 6)
        return flag, t_star, sight

    def mistakes(m, n_in):
        _, t_star, _ = unpack(n_in)
        t_hat = jnp.stack([sum(m[j] for j in bus_figurines(b)) for b in range(B)])
        return t_hat - t_star

    def report(y, m, n_in):
        flag, _, _ = unpack(n_in)
        e = mistakes(m, n_in)
        r = jnp.zeros(n)
        for b in range(B):
            for i, j in enumerate(bus_figurines(b)):
                share = y[3 * b + i] if arm.blame == "belief" else 1.0
                r = r.at[j].add(share * e[b])
        return flag * r

    def in_f(y, m, n_in):
        flag, _, _ = unpack(n_in)
        return make_input(message(report(y, m, n_in), flag), jnp.zeros(DP))

    def k(y, m, n_in):
        flag, _, sight = unpack(n_in)
        e = mistakes(m, n_in)
        total = 0.0
        for b in range(B):
            yb = y[3 * b:3 * b + 3]
            total = total + jnp.dot(yb, N_s @ sight[b])
            if arm.comparison:
                total = total + jnp.dot(yb, comparison_nll(e[b], e[1 - b]))
        return flag * total

    def omega(y, m, n_in):
        xi_y = jax.grad(lambda y_: -k(y_, m, n_in))(y)
        return (xi_y, jnp.zeros(mdim), jnp.zeros(depot_input_dim(B)))

    return Arrangement(
        P=P,
        in_dim_M=edim, out_dim_M=mdim, in_dim_N=depot_input_dim(B), out_dim_N=0,
        out_f=lambda y, m: jnp.zeros(0), in_f=in_f, omega=omega,
        label=f"depot({arm.blame},B={B})",
    )


# ---------------------------------------------------------------------------
# The composite, its state and the world.
# ---------------------------------------------------------------------------


class Built(NamedTuple):
    """The composite, with the inner composite (the parent over the figurines) and the depot kept to read the
    report off: the composite's own backward map ends at the figurines' empty input."""
    arr: Arrangement
    inner: Arrangement
    depot: Arrangement


def coordinated(arm: Arm, theta0: Sequence[float], priors: Sequence[Sequence[float]]) -> Built:
    """``depot . spatial(n)(figurines) : I -> <E x R^{1 + B + 6B} | R^0>``, placements frozen at 0."""
    n = arm.n
    figs = tensor_arrangements([figurine(j, n, float(theta0[j]), arm.response) for j in range(n)])
    n_Q = n * DP + n * n * n
    parent = spatial_arrangement(
        n, n, DP, TAU,
        parameter=configuration(constant(jnp.zeros((n_Q, n_Q))),
                                init=spatial_config(jnp.zeros((n, DP)), (jnp.eye(n),) * n)),
        label=f"spatial({n})")
    inner, box = compose_seq(figs, parent.arr()), depot(arm, priors)
    return Built(compose_seq(inner, box), inner, box)


# The composite's state is ``((figurines' states, left-nested), q_parent), beliefs)``.

def thetas(state, n: int) -> Array:
    figs = state[0][0]
    out = []
    for _ in range(n - 1):
        figs, th = figs
        out.append(th)
    out.append(figs)
    return jnp.concatenate(out[::-1])


def beliefs(state, B: int) -> Array:
    """The ``B`` beliefs, ``(B, 3)``, read off the depot's state."""
    s = state[1]
    ls = []
    for _ in range(B - 1):
        s, l = s
        ls.append(l)
    ls.append(s)
    return jnp.stack([belief_of(l) for l in ls[::-1]])


def with_thetas(state, theta: Sequence[float], n: int):
    """The same composite state with the figurines' states replaced (so one compiled composite runs every
    scenario of its arm): the figurines' states are left-nested, ``((th_0, th_1), th_2), ...``."""
    nested = jnp.array([theta[0]], float)
    for j in range(1, n):
        nested = (nested, jnp.array([theta[j]], float))
    return ((nested, state[0][1]), state[1])


def guesses(theta: Array, n: int) -> Array:
    return jnp.stack([RUNS[kind(j)](theta[j]) for j in range(n)])


def exterior(B: int, t_star: Sequence[float], flag: float = 1.0, sighting=None) -> Array:
    """The world's input ``(flag, t*_1..t*_B, sightings)``; ``sighting = (bus, o)`` sets one one-hot."""
    s = jnp.zeros((B, 6))
    if sighting is not None:
        s = s.at[sighting[0], sighting[1]].set(1.0)
    return jnp.concatenate([jnp.array([flag]), jnp.asarray(t_star, float), s.reshape(-1)])


def step_fn(arr: Arrangement):
    """One step of ``Phi(arr)`` on the world's input with zero exterior covector (the form ``figurines.step_fn``
    uses)."""
    P, omega = arr.P, arr.omega

    @jax.jit
    def step(s, ext):
        q = P.position(s)
        xi_q, _, _ = omega(q, EMPTY, ext)
        return P.step(s, xi_q), xi_q

    return step


def report_fn(built: Built):
    """The parent's exterior input the depot computes at a composite state, jitted: the report in the signal
    slots, read through the inner composite's sending map and the depot's backward map."""
    inner, box = built.inner, built.depot

    def report(s, ext):
        m = inner.out_f(inner.P.position(s[0]), EMPTY)
        return box.in_f(box.P.position(s[1]), m, ext)

    return jax.jit(report)


def run_rounds(built: Built, arm: Arm, t_star: Sequence[float], rounds: int, sighting=None, state=None):
    """Run ``rounds`` outcome steps; ``sighting = (bus, o)`` arrives every round if given. Returns arrays over
    steps ``0..R`` of the figurines' states, guesses, the beliefs, each bus's prediction and mistake, and the
    report sent on each step (the last is the report that would be sent next), and the final state."""
    arr = built.arr
    step, report = step_fn(arr), report_fn(built)
    n, B = arm.n, arm.B
    s = arr.P.init if state is None else state
    ext = exterior(B, t_star, 1.0, sighting)
    out = {"theta": [], "g": [], "y": [], "t_hat": [], "e": [], "report": []}
    for t in range(rounds + 1):
        th = thetas(s, n)
        g = guesses(th, n)
        t_hat = jnp.stack([sum(g[j] for j in bus_figurines(b)) for b in range(B)])
        out["theta"].append(np.asarray(th))
        out["g"].append(np.asarray(g))
        out["y"].append(np.asarray(beliefs(s, B)))
        out["t_hat"].append(np.asarray(t_hat))
        out["e"].append(np.asarray(t_hat - jnp.asarray(t_star, float)))
        out["report"].append(np.asarray(report(s, ext)[:n]))
        if t == rounds:
            break
        s, _ = step(s, ext)
    return {k: np.array(v) for k, v in out.items()}, s
