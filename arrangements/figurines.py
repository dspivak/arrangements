"""A composed belief paid in arrears (book plan P37 (b)(i), ``AGENTDOCPLAN.md`` §8.2).

David (book repo, 2026-10-07): "each of the figurines should be given what they were wrong about ... Each
person takes accountability or responsibility for the piece of their contribution that was wrong and the
extent to which it was wrong ... But I think that's already in the math, isn't it?"; "it's paid in arrears
for making good bets, and maybe that's what we're missing." This module builds the listener's belief about
a bus out of three FIGURINE boxes that run and guess, composes them on the spatial attention parent
(``attention.spatial_arrangement``, the paper's eq:spatial-transfers), and pays the composite's bet by its
realized loss; composition in ``arr`` (``wiring.compose_seq``, pullback-and-sum) descends that payment onto
each figurine's state and placement. Predictions fixed before this file existed: the book repo's
``notes/runs/p37-composed-belief-3f16cdb9/PREDICTIONS.md``; the closed form the tests check against is
``tests/figurines_closed_form.py``; the tests are ``tests/test_figurines.py``.

Minutes throughout; the bus is scheduled at 0.

THE FIGURINES, each ``I -> B`` with one real state ``theta_j`` (a configuration integrator, Euclidean
response ``mu_theta``), whose sending map runs a small mechanism (a hand-typed loop inside the sending map, one smooth map, not a system in
the book's sense; P43 (b) job 8) from that state and reads off a guess in
minutes, sent as ``(g_j e_j, 1)`` in its own slot of ``H = R^6`` (slots 0..2 guesses, 3..5 the account's
flags, used only by the exterior):

* the DRIVER: readiness rate ``lam = e^theta``; run ``x <- x + dt lam (1 - x)`` for K = 40 steps of half a
  minute from ``x = 0``; the guess is the minutes spent not yet ready, ``sum dt (1 - x_k)``.
* the VEHICLE: fault ``phi = softplus(theta)``; run the speed ``s <- s (1 - phi)`` for 20 one-minute steps
  from ``s = 1``; the guess is the minutes lost to the sagging speed, ``sum (1 / s_k - 1)``.
* the ROUTE: congestion ``c = e^theta``; run the bus along 20 km, each taking ``0.75 + c pi_k`` minutes
  with ``pi_k = 1`` on the five downtown km; the guess is the travel time.

Each figurine's own cost reads the ACCOUNT the exterior sends, delivered by the parent tempered by the
figurine's kernel weight: ``k_j = w_j g_j^2 / 2 - g_j h_j`` on its received slots ``h_j = kappa_j f_j a_j``
and ``w_j = kappa_j f_j`` (the Gaussian likelihood's ``g``-part, linear in the message), forcing
UNILATERAL (its differential in the state only: reading returns no push), so the force on ``theta_j`` is
``-kappa_j f_j (g_j - a_j) g_j'``: Bayes-like, tempered by the invitation.

THE PARENT composes: its output message has ``h = sum_j kappa_j g_j e_j``, ``kappa_j = exp(-p_j^2/2 tau)``
the figurine's INVITATION at placement ``p_j`` (the exterior at 0); value maps frozen at the identity;
only the placements respond (mobility ``mu_p``).

THE COPY BOX, the diagonal ``M -> M (x) M (x) M`` as a static arrangement, sends the parent's message to the
bettor three times, for the plan, the bet and the observation, each plainly or through ``stop_gradient``
(ONE-WAY, the idiom of ``listener.py`` and ``account.fund_layer``), set per arm.

THE BETTOR has one state, the margin ``m`` (mobility ``mu_m``, 0 where frozen), reads the prediction
``t_hat = h_0 + h_1 + h_2`` off each copy, and carries: every step the PLAN's inconvenience
``c_inc softplus(t_pref - t_hat_plan - m)`` (the listener would rather leave later); on an outcome step,
when the exterior sets the flag and the actual arrival ``t*``, the REALIZED LOSS ``(t_hat_bet + m - t*)^2 / 2``
of the bet, and, where the arm says, the OBSERVATION's cost ``a_obs (t_hat_obs - t*)^2 / 2``.

THE WORLD is the exterior input: ``(m_0, p_0 = 0)`` for the parent, ``m_0`` the account message
``((a f, f), 1)`` on an outcome step with an account and ``0_M`` otherwise, then ``(flag, t*)`` for the
bettor. The composite is ``bettor . copy . parent(driver (x) vehicle (x) route) : I -> <E x R^2 | R^0>``.
"""

from __future__ import annotations

from typing import NamedTuple, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, zero_form
from .attention import input_dim, make_input, message, msg_dim, spatial_arrangement, spatial_config
from .coalgebra import configuration
from .rvect import constant, euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

DH, DP = 6, 1
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)      # 7, 8
EMPTY = jnp.zeros(0)
TAU = 0.8
DRIVER_DT, DRIVER_K = 0.5, 40
VEHICLE_K = 20
ROUTE_BASE, ROUTE_PROFILE = 0.75, jnp.array([1.0 if 8 <= k <= 12 else 0.0 for k in range(20)])
C_INC, T_PREF, A_OBS = 0.1, 30.0, 0.5


class Arm(NamedTuple):
    """Which copies of the prediction are one-way, whether the observation pays the belief, the margin's
    mobility."""
    plan_plain: bool
    bet_plain: bool
    observe: bool
    mu_m: float
    name: str = ""
    loss: str = "squared"     # "squared": e^2/2; "cliff": a minute late is the bus, early costs nothing here


def cliff(e, c_late: float = 5.0, w: float = 0.1):
    """A bus loss: ``c_late w softplus(e / w)``, about ``c_late`` per minute late and about ``0`` early (the
    waiting is the plan's inconvenience); smoothed over ``w`` minutes so it has a slope (P37 (b)(ii), 70fcc5f2,
    at David's "the opposite makes no sense. what's happening to cause it?")."""
    return c_late * w * jax.nn.softplus(e / w)


ARMS = {
    "A": Arm(False, True, False, 0.0, "A, one-way"),
    "B": Arm(True, True, False, 0.0, "B, exact"),
    "C": Arm(False, False, True, 0.1, "C, a margin of its own"),
    "D": Arm(True, True, True, 0.1, "D, a margin, exact"),
    "A-cliff": Arm(False, True, False, 0.0, "A, one-way, cliff loss", "cliff"),
    "B-cliff": Arm(True, True, False, 0.0, "B, exact, cliff loss", "cliff"),
    "C-cliff": Arm(False, False, True, 0.1, "C, a margin of its own, cliff loss", "cliff"),
    "D-cliff": Arm(True, True, True, 0.1, "D, a margin, exact, cliff loss", "cliff"),
}


# ---------------------------------------------------------------------------
# The figurines: boxes that run and guess.
# ---------------------------------------------------------------------------


def run_driver(theta: Array) -> Array:
    lam = jnp.exp(theta)

    def body(k, carry):
        x, acc = carry
        return x + DRIVER_DT * lam * (1.0 - x), acc + DRIVER_DT * (1.0 - x)

    _, delay = jax.lax.fori_loop(0, DRIVER_K, body, (0.0, 0.0))
    return delay


def run_vehicle(theta: Array) -> Array:
    phi = jax.nn.softplus(theta)

    def body(k, carry):
        s, acc = carry
        return s * (1.0 - phi), acc + (1.0 / s - 1.0)

    _, lost = jax.lax.fori_loop(0, VEHICLE_K, body, (1.0, 0.0))
    return lost


def run_route(theta: Array) -> Array:
    c = jnp.exp(theta)

    def body(k, acc):
        return acc + ROUTE_BASE + c * ROUTE_PROFILE[k]

    return jax.lax.fori_loop(0, 20, body, 0.0)


RUNS = (run_driver, run_vehicle, run_route)
NAMES = ("driver", "vehicle", "route")


def figurine(which: int, theta0: float, mu_theta: float) -> Arrangement:
    """Figurine ``which`` as a box ``I -> B``: state ``theta``, guess ``RUNS[which](theta)`` sent in slot
    ``which``, the account read from its received slots ``which`` and ``3 + which``, unilaterally."""
    run = RUNS[which]
    slot = jnp.zeros(DH).at[which].set(1.0)

    def send(theta, m_out):
        return message(run(theta[0]) * slot, 1.0)

    def k_account(theta, e):
        g = run(theta[0])
        return e[3 + which] * g * g / 2.0 - g * e[which]

    def omega(theta, m_out, e):
        xi_theta = jax.grad(lambda th: -k_account(th, e))(theta)
        return (xi_theta, jnp.zeros(0), jnp.zeros(EDIM))

    return Arrangement(
        P=configuration(euclidean(1, mu_theta), init=jnp.array([theta0], float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=send, in_f=lambda theta, m_out, e: jnp.zeros(0), omega=omega,
        label=NAMES[which],
    )


# ---------------------------------------------------------------------------
# The copy box and the bettor.
# ---------------------------------------------------------------------------


def copy_box(arm: Arm) -> Arrangement:
    """The diagonal ``M -> M (x) M (x) M`` as a static arrangement; the plan's and the bet's copies go
    through ``stop_gradient`` where the arm says (one-way), the observation's always plainly."""
    plain = lambda m: m
    oneway = jax.lax.stop_gradient

    def send(q, m):
        return jnp.concatenate([(plain if arm.plan_plain else oneway)(m),
                               (plain if arm.bet_plain else oneway)(m), m])

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=EDIM, out_dim_M=MDIM, in_dim_N=EDIM, out_dim_N=3 * MDIM,
        out_f=send, in_f=lambda q, m, e: e, omega=zero_form, label="copy",
    )


def prediction_of(m: Array) -> Array:
    """``t_hat = h_0 + h_1 + h_2`` off one copy of the parent's message."""
    return m[0] + m[1] + m[2]


def bettor(arm: Arm) -> Arrangement:
    """``<E | M^3> -> <E x R^2 | R^0>``: state the margin ``m``; passes the exterior's ``e`` down to the
    parent; costs the plan's inconvenience every step and, on an outcome step (flag 1), the realized loss
    of the bet and, where the arm says, the observation's cost."""

    def k(mrg, m3, n_in):
        t_plan, t_bet, t_obs = (prediction_of(m3[i * MDIM:(i + 1) * MDIM]) for i in range(3))
        flag, t_star = n_in[EDIM], n_in[EDIM + 1]
        plan = C_INC * jax.nn.softplus(T_PREF - t_plan - mrg[0])
        e_bet = t_bet + mrg[0] - t_star
        loss = cliff(e_bet) if arm.loss == "cliff" else 0.5 * e_bet ** 2
        obs = 0.5 * A_OBS * (t_obs - t_star) ** 2 if arm.observe else 0.0
        return plan + flag * (loss + obs)

    return Arrangement(
        P=configuration(euclidean(1, arm.mu_m), init=jnp.zeros(1)),
        in_dim_M=EDIM, out_dim_M=3 * MDIM, in_dim_N=EDIM + 2, out_dim_N=0,
        out_f=lambda mrg, m3: jnp.zeros(0),
        in_f=lambda mrg, m3, n_in: n_in[:EDIM],
        omega=jax.grad(lambda mrg, m3, n_in: -k(mrg, m3, n_in), argnums=(0, 1, 2)),
        label=f"bettor({arm.name})",
    )


# ---------------------------------------------------------------------------
# The composite, its state and the world.
# ---------------------------------------------------------------------------


class Model(NamedTuple):
    theta0: Tuple[float, float, float]
    p0: Tuple[float, float, float]
    mu_theta: float = 0.02
    mu_p: float = 0.0


def belief_system(model: Model, arm: Arm) -> Arrangement:
    """``bettor . copy . spatial(3)(driver, vehicle, route) : I -> <E x R^2 | R^0>``."""
    figs = tensor_arrangements([figurine(j, model.theta0[j], model.mu_theta) for j in range(3)])
    n_Q = 3 * DP + 3 * DH * DH
    diag = jnp.zeros(n_Q).at[:3].set(model.mu_p)
    parent = spatial_arrangement(
        3, DH, DP, TAU,
        parameter=configuration(constant(jnp.diag(diag)),
                                init=spatial_config(model.p0, (jnp.eye(DH),) * 3)),
        label="spatial(3)")
    composed = compose_seq(figs, parent.arr())
    return compose_seq(compose_seq(composed, copy_box(arm)), bettor(arm))


# The composite's state is ``((((th_d, th_v), th_r), q_parent), ()), m)``: the figurines' states, the
# parent's configuration, the copy's empty state, the margin.

def thetas(state) -> Array:
    (((a, b), c), _), _ = state[0]
    return jnp.concatenate([a, b, c])


def placements(state) -> Array:
    return state[0][0][1][:3]


def margin(state) -> Array:
    return state[1][0]


def exterior(flag: float = 0.0, t_star: float = 0.0, account=None) -> Array:
    """The world's input: the parent's ``(m_0, p_0 = 0)``, then ``(flag, t*)`` for the bettor. With
    ``account = (a, f)``, ``m_0 = ((a f, f), 1)``; else ``0_M``."""
    if account is None:
        m0 = jnp.zeros(MDIM)
    else:
        a, f = (jnp.asarray(x, float) for x in account)
        m0 = message(jnp.concatenate([a * f, f]), 1.0)
    return jnp.concatenate([make_input(m0, jnp.zeros(DP)), jnp.array([flag, t_star])])


def step_fn(arr: Arrangement):
    """One step of ``Phi(arr)`` on the exterior input ``ext`` with zero exterior covector, specialized to
    the configuration component of the forcing (the form ``listener.step_fn`` uses)."""
    P, omega = arr.P, arr.omega

    @jax.jit
    def step(s, ext):
        xi_q, _, _ = omega(P.position(s), EMPTY, ext)
        return P.step(s, xi_q), xi_q

    return step


def run_rounds(arr: Arrangement, truth: Sequence[float], rounds: int, steps: int = 10,
               account_flags=None, state=None):
    """Run ``rounds`` rounds of ``steps`` steps; the outcome arrives on each round's last step, with the
    account's components ``truth * account_flags`` if flags are given. Returns arrays over steps
    ``0..T`` of the figurines' states, placements, the margin, the prediction, and the final state."""
    step = step_fn(arr)
    t_star = float(sum(truth))
    s = arr.P.init if state is None else state
    T = steps * rounds
    out = {"theta": [], "p": [], "m": [], "t_hat": []}
    for t in range(T + 1):
        outcome = (t % steps == steps - 1) and t < T
        acc = (truth, account_flags) if (outcome and account_flags is not None) else None
        ext = exterior(1.0 if outcome else 0.0, t_star, acc)
        out["theta"].append(np.asarray(thetas(s)))
        out["p"].append(np.asarray(placements(s)))
        out["m"].append(float(margin(s)))
        out["t_hat"].append(float(prediction_at(s)))
        if t == T:
            break
        s, _ = step(s, ext)
    return {k: np.array(v) for k, v in out.items()}, s


def prediction_at(s) -> Array:
    """``t_hat = sum_j kappa_j g_j(theta_j)`` read off the composite's state (what the parent sends)."""
    th, p = thetas(s), placements(s)
    g = jnp.stack([RUNS[j](th[j]) for j in range(3)])
    return jnp.sum(jnp.exp(-p ** 2 / (2.0 * TAU)) * g)
