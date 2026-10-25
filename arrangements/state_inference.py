"""State inference: a belief about a hidden physical state, as a 0-ary attention arrangement with a
private prediction, wired in ``arr`` beside the oscillator it models, through a one-way sensor
(book Ch 4, predictive processing; P5 (l) stage D, ``notes/runs/p5lD-98fc949b/RESULTS.md``).

EXTENSION (beyond the attention paper): the belief's integrator carries the belief forward by the
model's own step, and its cost carries the prior term that anchors the belief to that prediction.
Without the prior term the cost's minimizer is the observation, not the posterior (stage A, A6).

The closed system is four boxes and a wiring:

* ``kicked_oscillator``  -- the hidden state ``x = (q, p)``, a ``phase`` integrator with potential
                            ``k q^2 / 2`` that hears a kick ``w`` (process noise); presents ``q + p/m``.
* ``noise_box``          -- a counter reading out pre-drawn samples ``(w_c, v_c)``; it advances at every
                            step, duration 0 included (Ch 3: a counter must), and ignores the force on
                            what it presents.
* ``sensor``             -- a latch (``system_id.delay_box``'s convention): hears the position and the
                            sensor noise, latches their sum ``y`` at a step of duration 1, holds it through
                            steps of duration 0, reads out the message ``(h, 1)``, and ignores the pushback
                            that comes back on ``h``. So the belief's pushback never reaches the oscillator.
* ``belief``             -- the 0-ary attention arrangement: configuration ``(mu, mu_p, k_hat, lambda)``,
                            cost ``1/2 e^lambda (h - H mu)^2 + 1/2 (w - 1)^2 - 1/2 lambda
                            + 1/2 (mu - F_{k_hat} mu_p)' P^{-1} (mu - F_{k_hat} mu_p)``.

Three clocks. A *frame* is one step of duration 1 (the physical step: the oscillator moves, the
sensor latches, the belief takes its last descent step on the observation it has, records that
posterior as ``mu_p`` and flows it by the model's step) followed by ``D`` steps of duration 0 (the
oscillator and the noise hold still, the sensor holds ``h``, the belief descends). The belief's
descent ``mu -> mu + G xi`` happens at every step; its parameters ``k_hat`` and ``lambda`` move by
``tau * eta * xi``, so once per frame, at the settled belief. No clock for rounds.

The steady-state Kalman filter (``kalman``, numpy) is the analytic answer run beside the arrangement
on the same observations; ``riccati`` gives the model's steady-state prior covariance ``P``, the
posterior covariance ``P_plus`` and the gain ``K``. ``P_plus H' Pi = K``: the posterior covariance used
as the belief's response makes one descent step the Kalman update exactly.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .attention import AttentionArrangement, attention_arrangement, zero_transfer
from .coalgebra import CotCoalg, phase
from .functors import closed_step
from .rvect import diagonal
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


# ---------------------------------------------------------------------------
# The model's step, and the Kalman data (ordinary calculations beside the arrangement).
# ---------------------------------------------------------------------------


def model_step(k, m: float, tau) -> Array:
    """``F_{k,tau}``: the ``phase`` integrator's duration-``tau`` step for the oscillator ``(q, p)``,
    ``(q, p) |-> (q + tau p/m, p - tau k (q + tau p/m))``; the identity at ``tau = 0``, ``F`` at ``tau = 1``."""
    return jnp.array([[1.0, tau / m], [-tau * k, 1.0 - tau * tau * k / m]])


def observation_row(m: float) -> np.ndarray:
    """``H = (1, 1/m)``: the presented position ``q + p/m`` of the stored state ``(q, p)``."""
    return np.array([1.0, 1.0 / m])


def riccati(k: float, m: float, sigma_w: float, sigma_v: float, iters: int = 20000) -> Dict[str, np.ndarray]:
    """Steady state of the Kalman recursion for ``x' = F x + e_2 w``, ``y = H x + v``.

    Returns ``F``, ``H``, the prior covariance ``P``, the posterior covariance ``P_plus``, the gain ``K``
    and ``S = H P H'``; ``P_plus = (H' Pi H + P^{-1})^{-1}`` with ``Pi = 1/sigma_v^2``, and ``P_plus H' Pi = K``."""
    F = np.asarray(model_step(k, m, 1.0), dtype=float)
    H = observation_row(m)[None, :]
    W = np.diag([0.0, sigma_w ** 2])
    R = sigma_v ** 2
    P = np.eye(2)
    for _ in range(iters):
        K = P @ H.T / (H @ P @ H.T + R)
        P = F @ (P - K @ H @ P) @ F.T + W
    K = P @ H.T / (H @ P @ H.T + R)
    P_plus = P - K @ H @ P
    return dict(F=F, H=H[0], P=P, P_plus=P_plus, K=K[:, 0], S=float((H @ P @ H.T)[0, 0]))


def innovation_variance(kd: Dict[str, np.ndarray], sigma_w: float, sigma_v: float, Pi: float) -> float:
    """``E nu^2`` of the fixed-gain filter run at the gain ``K_Pi = P H' / (S + 1/Pi)`` (``P`` the model's steady-state
    prior covariance, held fixed) on the true model: ``H Sigma H' + sigma_v^2`` with ``Sigma`` the prior error
    covariance from the discrete Lyapunov equation ``Sigma = M Sigma M' + F K K' F' sigma_v^2 + W``, ``M = F (I - K H)``."""
    from scipy.linalg import solve_discrete_lyapunov

    F, H, P, S = kd["F"], kd["H"][None, :], kd["P"], kd["S"]
    K = P @ H.T / (S + 1.0 / Pi)
    M = F @ (np.eye(2) - K @ H)
    Qn = F @ K @ K.T @ F.T * sigma_v ** 2 + np.diag([0.0, sigma_w ** 2])
    Sig = solve_discrete_lyapunov(M, Qn)
    return float((H @ Sig @ H.T)[0, 0]) + sigma_v ** 2


def precision_fixed_points(k: float, m: float, sigma_w: float, sigma_v: float) -> np.ndarray:
    """The learned precision's fixed points at the settled belief, one level: the ``Pi`` with
    ``Pi E_Pi nu^2 = (1 + Pi S)^2`` (RESULTS §2 (iv)), found on a log grid in ``[1e-2, 1e5]`` and refined by bisection.
    Empty when there is none (then ``lambda`` runs away from every start); else two, the smaller stable."""
    from scipy.optimize import brentq

    kd = riccati(k, m, sigma_w, sigma_v)
    S = kd["S"]
    g = lambda Pi: Pi * innovation_variance(kd, sigma_w, sigma_v, Pi) - (1.0 + Pi * S) ** 2
    grid = np.exp(np.linspace(np.log(1e-2), np.log(1e5), 400))
    vals = np.array([g(p) for p in grid])
    roots = []
    for i in np.where(np.diff(np.sign(vals)) != 0)[0]:
        roots.append(brentq(g, grid[i], grid[i + 1]))
    return np.array(roots)


def kalman(y: np.ndarray, mu0: np.ndarray, F: np.ndarray, H: np.ndarray, K: np.ndarray) -> np.ndarray:
    """The steady-state Kalman posterior means ``xhat_t`` for ``t = 0..T-1`` from the prior mean ``mu0``
    for ``x_0`` and the observations ``y_t``: ``xhat_t = xm_t + K (y_t - H xm_t)``, ``xm_{t+1} = F xhat_t``."""
    xm = np.asarray(mu0, dtype=float)
    out = []
    for yt in y:
        xh = xm + K * (yt - H @ xm)
        out.append(xh)
        xm = F @ xh
    return np.stack(out)


# ---------------------------------------------------------------------------
# The four boxes.
# ---------------------------------------------------------------------------


def kicked_oscillator(k: float, m: float, q0: float, p0: float) -> Arrangement:
    """``<0|0> -> <1|1>``: the oscillator with potential ``k q^2/2`` that hears a kick ``w`` on its one input
    (potential ``-k q^2/2 + w q``, so the force on the presented position is ``-k q + w``) and reads out the
    presented position ``q + p/m``. The force on its input, ``q``, goes back to whatever sends the kick."""
    P = phase(diagonal(jnp.full(1, 1.0 / m)), init=(jnp.array([float(q0)]), jnp.array([float(p0)])))
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * k * q[0] ** 2 + n_in[0] * q[0]),
        label="oscillator",
    )


def noise_box(w: np.ndarray, v: np.ndarray) -> Arrangement:
    """``<0|0> -> <0|2>``: a counter ``c`` reading out the pre-drawn samples ``(w_c, v_c)``.

    State ``c`` (an integer, a 0-dimensional manifold), configuration ``R^2``, update ``c + 1`` at every
    step whatever its duration (a counter, Ch 3 after ``def.duration_integrator``); its one-form is zero
    and its update ignores the covector it receives, as the book's datum box does."""
    samples = jnp.stack([jnp.asarray(w, dtype=float), jnp.asarray(v, dtype=float)], axis=1)
    P = CotCoalg(
        Q_dim=2, init=jnp.int32(0),
        position=lambda c: samples[c],
        step=lambda c, xi: c + 1,
        timed_step=lambda c, xi, tau: c + 1,
        label="counter",
    )
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=2,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label="noise",
    )


def sensor(h0: float) -> Arrangement:
    """``<0|0> -> <2|2>``: a latch. Hears the position and the sensor noise on two inputs, reads out the
    message ``(h, 1)``. State ``(h, z)``; one-form ``(n_1 + n_2 - z) dz`` and nothing on its inputs (it hears
    without pulling); update at duration ``tau``: ``z <- z + xi_z`` (``= n_1 + n_2``, what it hears now),
    ``h <- (1 - tau) h + tau (z + xi_z)``. So ``h`` latches the noisy position at a step of duration 1 and
    holds through steps of duration 0; the pushback ``xi_h`` on what it presents is ignored."""
    def timed_step(s, xi, tau):
        heard = s[1] + xi[1]
        return jnp.array([(1.0 - tau) * s[0] + tau * heard, heard])

    P = CotCoalg(
        Q_dim=2, init=jnp.array([float(h0), float(h0)]),
        position=lambda s: s,
        step=lambda s, xi: timed_step(s, xi, 1.0),
        timed_step=timed_step,
        label="latch",
    )

    def omega(q, m_out, n_in):
        return jnp.array([0.0, n_in[0] + n_in[1] - q[1]]), jnp.zeros_like(m_out), jnp.zeros_like(n_in)

    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=2, out_dim_N=2,
        out_f=lambda q, m_out: jnp.array([q[0], 1.0]),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega,
        label="sensor",
    )


def belief_cost(P: np.ndarray, m: float) -> Callable[[Array, Array, Array], Array]:
    """The belief's cost on ``q = (mu, mu_p, k_hat, lambda)`` and the input ``e = (h, w)``:
    ``1/2 e^lambda (h - H mu)^2 + 1/2 (w - 1)^2 - 1/2 lambda + 1/2 (mu - F_{k_hat} mu_p)' P^{-1} (mu - F_{k_hat} mu_p)``
    (``eqn.private_prediction`` with prediction ``(H mu, 1)`` in the inner product ``e^lambda (+) 1`` on the
    message space, plus terms of the configuration alone; degree two in ``e``)."""
    Pinv = jnp.asarray(np.linalg.inv(P))
    H = jnp.asarray(observation_row(m))

    def cost(q, m_out, n_in):
        mu, mu_p, k_hat, lam = q[:2], q[2:4], q[4], q[5]
        h, w = n_in[0], n_in[1]
        r = mu - model_step(k_hat, m, 1.0) @ mu_p
        return (0.5 * jnp.exp(lam) * (h - H @ mu) ** 2 + 0.5 * (w - 1.0) ** 2 - 0.5 * lam
                + 0.5 * r @ Pinv @ r)

    return cost


def belief(
    P: np.ndarray,
    *,
    m: float = 1.0,
    mu0=(0.0, 0.0),
    k_hat0: float = 0.1,
    lambda0: float = 0.0,
    G=None,
    newton: bool = False,
    eta_k: float = 0.0,
    eta_lambda: float = 0.0,
    label: str = "belief",
) -> AttentionArrangement:
    """The belief as a 0-ary attention arrangement (``def.attention_arrangement`` with ``I = 0``): no
    placement, the one transfer ``T_00 = 0``, readout the zero message, cost ``belief_cost``.

    Integrator on ``Q = S = R^6``, ``(mu, mu_p, k_hat, lambda)``, update at duration ``tau`` given ``xi``:
    ``mu+ = mu + G xi_mu`` (every step; ``G`` a 2x2 response, a scalar times the identity, or with
    ``newton`` the inverse Hessian ``(H' e^lambda H + P^{-1})^{-1}``, the posterior covariance);
    ``k_hat += tau eta_k xi_k``, ``lambda += tau eta_lambda xi_lambda`` (Euler matter: once per frame);
    ``mu <- F_{k_hat, tau} mu+`` (the model's duration-``tau`` step; identity at 0); ``mu_p <- (1 - tau) mu_p + tau mu+``.
    """
    Pinv = jnp.asarray(np.linalg.inv(P))
    H = jnp.asarray(observation_row(m))
    if G is None:
        G_mat = jnp.eye(2)
    elif jnp.ndim(G) == 0:
        G_mat = float(G) * jnp.eye(2)
    else:
        G_mat = jnp.asarray(G, dtype=float)

    def response(s):
        if newton:
            return jnp.linalg.inv(jnp.exp(s[5]) * jnp.outer(H, H) + Pinv)
        return G_mat

    def timed_step(s, xi, tau):
        mu, mu_p, k_hat, lam = s[:2], s[2:4], s[4], s[5]
        mu_plus = mu + response(s) @ xi[:2]
        k2 = k_hat + tau * eta_k * xi[4]
        lam2 = lam + tau * eta_lambda * xi[5]
        mu2 = model_step(k2, m, tau) @ mu_plus
        mu_p2 = (1.0 - tau) * mu_p + tau * mu_plus
        return jnp.concatenate([mu2, mu_p2, jnp.array([k2, lam2])])

    init = jnp.concatenate([jnp.asarray(mu0, dtype=float), jnp.asarray(mu0, dtype=float),
                            jnp.array([float(k_hat0), float(lambda0)])])
    Pc = CotCoalg(
        Q_dim=6, init=init, position=lambda s: s,
        step=lambda s, xi: timed_step(s, xi, 1.0), timed_step=timed_step, label="belief",
    )
    return attention_arrangement(
        0, 1, 0,
        placement=lambda q: jnp.zeros((0, 0)),
        transfers=lambda q: ((zero_transfer(1),),),
        cost=belief_cost(P, m), parameter=Pc, label=label,
    )


# ---------------------------------------------------------------------------
# The closed systems, and the frames.
# ---------------------------------------------------------------------------

# Configuration layout of ``closed_system``: [q~ (0); w, v (1, 2); h, z (3, 4); mu (5, 6); mu_p (7, 8); k_hat (9); lambda (10)]
I_Q, I_W, I_V, I_H, I_Z, I_MU, I_MUP, I_K, I_LAM = 0, 1, 2, 3, 4, slice(5, 7), slice(7, 9), 9, 10


def closed_system(osc: Arrangement, noise: Arrangement, sens: Arrangement, b: AttentionArrangement) -> Arrangement:
    """The closed arrangement: the four boxes in parallel, closed by a wiring that delivers ``w`` to the
    oscillator's kick, ``q~`` and ``v`` to the sensor, and the sensor's ``(h, 1)`` to the belief's external
    input, and leaves the belief's zero message dangling. Its one-form is zero."""
    parts = tensor_arrangements([osc, noise, sens, b.arr()])
    # parts' outputs: [q~ (0), w (1), v (2), h (3), 1 (4), belief's message (5, 6)]; inputs: [kick, n_1, n_2, e_h, e_w]
    idx = jnp.asarray([1, 0, 2, 3, 4], dtype=int)
    route = Arrangement(
        P=trivial_parameter(), in_dim_M=5, out_dim_M=7, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: m_out[idx],
        omega=zero_form, label="wiring",
    )
    return compose_seq(parts, route)


def open_system(osc: Arrangement, noise: Arrangement) -> Arrangement:
    """The oscillator and the noise alone, closed by the wiring that delivers ``w`` to the kick."""
    parts = tensor_arrangements([osc, noise])
    idx = jnp.asarray([1], dtype=int)
    route = Arrangement(
        P=trivial_parameter(), in_dim_M=1, out_dim_M=3, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0), in_f=lambda q, m_out, n_in: m_out[idx],
        omega=zero_form, label="wiring",
    )
    return compose_seq(parts, route)


def oscillator_state(s) -> Array:
    """The oscillator's stored ``(q, p)`` inside a state of ``closed_system`` or ``open_system``
    (states nest left: ``(((osc, noise), sensor), belief), wiring``)."""
    y = s
    while isinstance(y[0], tuple):
        y = y[0]
    return jnp.concatenate([y[0], y[1]])      # y is the phase state (q, p)


def frame_fn(arr: Arrangement, D: int, check: int = 2):
    """One frame of ``arr``: a step of duration 1 then ``D`` steps of duration 0, compiled; each step checked
    against ``Phi`` on its first ``check`` steps by ``closed_step``."""
    step1 = closed_step(arr, check=check, duration=1.0)
    step0 = closed_step(arr, check=check, duration=0.0) if D > 0 else None

    @jax.jit
    def frame(s):
        s = step1(s)
        if D > 0:
            s = jax.lax.fori_loop(0, D, lambda _i, x: step0(x), s)
        return s

    return frame


def run_frames(arr: Arrangement, D: int, T: int, check: int = 2) -> Tuple[object, Dict[str, np.ndarray]]:
    """Run ``T`` frames from ``arr.P.init``; return the final state and, per frame ``t = 0..T-1``, the
    configuration presented at the frame's step of duration 1 (``before``), the configuration after the
    frame (``after``) and the oscillator's stored state after the frame (``x_after``)."""
    frame = frame_fn(arr, D, check)
    P = arr.P

    def body(s, _):
        before = P.position_at(s, 1.0)
        s2 = frame(s)
        return s2, (before, P.position(s2), oscillator_state(s2))

    s, (before, after, x_after) = jax.lax.scan(body, P.init, None, length=T)
    return s, dict(before=np.asarray(before), after=np.asarray(after), x_after=np.asarray(x_after))


def draw_noise(sigma_w: float, sigma_v: float, n: int, seed: int) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    return sigma_w * rng.standard_normal(n), sigma_v * rng.standard_normal(n)


def build(
    *,
    k: float = 0.1, m: float = 1.0, x0=(1.0, 0.0), sigma_w: float = 0.05, sigma_v: float = 0.1,
    mu0=(0.0, 0.0), D: int = 0, T: int = 200, seed: int = 0,
    G=None, newton: bool = False, k_hat0: Optional[float] = None, lambda0: Optional[float] = None,
    eta_k: float = 0.0, eta_lambda: float = 0.0,
) -> Dict[str, object]:
    """The closed system at these settings, with its Kalman data, noise and open counterpart.
    ``k_hat0`` defaults to the true ``k``, ``lambda0`` to the true log-precision ``-2 log sigma_v``."""
    kd = riccati(k, m, sigma_w, sigma_v)
    n = T * (D + 1) + 8
    w, v = draw_noise(sigma_w, sigma_v, n, seed)
    H = observation_row(m)
    mu0 = np.asarray(mu0, dtype=float)
    osc = kicked_oscillator(k, m, x0[0], x0[1])
    noise = noise_box(w, v)
    sens = sensor(float(H @ mu0))
    b = belief(
        kd["P"], m=m, mu0=mu0, k_hat0=k if k_hat0 is None else k_hat0,
        lambda0=(-2.0 * np.log(sigma_v)) if lambda0 is None else lambda0,
        G=G, newton=newton, eta_k=eta_k, eta_lambda=eta_lambda,
    )
    return dict(
        closed=closed_system(osc, noise, sens, b), open=open_system(kicked_oscillator(k, m, x0[0], x0[1]), noise_box(w, v)),
        belief=b, kalman=kd, w=w, v=v, k=k, m=m, D=D, T=T, mu0=mu0, sigma_v=sigma_v, sigma_w=sigma_w, x0=np.asarray(x0, float),
    )


def posterior_record(rec: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """From ``run_frames``' record: ``y[t]`` the observation latched at frame ``t`` (``h`` after it),
    ``mu_p[t]`` the belief's posterior for ``x_{t-1}`` (``mu_p`` after frame ``t``, valid for ``t >= 1``),
    ``qt[t]`` the hidden presented position ``H x_t`` (presented at frame ``t``'s step of duration 1),
    ``k_hat[t]``, ``lam[t]`` after frame ``t``, and ``mu[t]`` after frame ``t``."""
    after, before = rec["after"], rec["before"]
    return dict(
        y=after[:, I_H], mu_p=after[:, I_MUP], mu=after[:, I_MU], qt=before[:, I_Q],
        k_hat=after[:, I_K], lam=after[:, I_LAM], w=before[:, I_W], v=before[:, I_V],
    )
