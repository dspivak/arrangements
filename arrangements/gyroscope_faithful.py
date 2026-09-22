"""The faithful gyroscope machine (EXTENSION, Phase 3) -- equation-faithful, in-framework.

A *supplement* to the harmonic surrogate in ``gyroscope.py``: the Bull & Achour
gyroscope-and-springs classifier built entirely from the paper's constructions, so a
reader can read the ODEs and check the factorization. Every force is one framework slot,
assembled categorically:

    spring coupling    -> the prism graph-wiring (``compose_graph``) on a hex lattice,
                          R^2 per gyro: the graph-Laplacian forcing EMERGES from
                          composition (pullback-and-sum, DESIGN.md §2; it is not a
                          hand-written omega)
    rod gravity        -> a nonlinear on-site potential (``rod_gravity``), the ``onsite``
                          of ``compose_graph``
    inertia / mass     -> the reactive sharp (1/m)
    quadratic air drag -> a nonlinear 1-form (rmk.adam: monoidal over (+), natural over
                          per-gyro O(2))
    gyroscopic torque  -> the per-gyro skew 1-form (a ``gamma`` vector)
    RK4 time-stepping  -> the ``rk4_gyro`` four-stage coalgebra as parameter,
                          ``PhiK`` = pc^(4) on the phase state
    external nudge     -> the open input port (encoder force on the input gyros)

GOAL (see GYRO_BUILD_HANDOFF.md): the **factorization** + the **springs->0 ablation**,
NOT accuracy. No external accuracy claim; synthetic generator-as-spec data
(``make_strokes``). Built small -- the factorization and the ablation do not need the
blog's ~100-gyro scale.

Like ``gyroscope.py`` this is the TRICK realization of the velocity-dependent drag and
precession 1-forms (they live in ``rk4_gyro``'s step, the forcing on ``Q`` being exact);
the DIRECT one, a non-exact forcing on ``T^*Q`` stepped by ``rk4.rk4_gyro_on_cotangent``,
is checked against ``rk4_gyro`` in tests/test_rk4_gyro.py.
"""

from __future__ import annotations

from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement
from .gyroscope import (
    GyroConfig,
    adam_init,
    adam_step,
    complex_structure,
    cross_entropy,
    hex_graph,
    rod_gravity,
)
from .interpretation import trivial_omega
from .pcK import PhiK
from .rk4 import rk4_gyro
from .rvect import diagonal
from .wiring import compose_graph

_IN_POS = (jnp.zeros(0), trivial_omega(0))
ROD_LENGTH = 3.0  # rod length L; tilts |q| < L (a rod cannot pass horizontal)


# ---------------------------------------------------------------------------
# Topology + the open-port arrangement (springs and gravity from the wiring).
# ---------------------------------------------------------------------------


def make_hex_config(
    rows: int = 3, cols: int = 3, n_classes: int = 4, settle: int = 10, force_scale: float = 0.3
) -> GyroConfig:
    """A hex-lattice gyro config: input gyros = left column, output = right column."""
    edges, in_g, out_g = hex_graph(rows, cols)
    in_idx = tuple(2 * v + c for v in in_g for c in (0, 1))
    out_idx = tuple(2 * v + c for v in out_g for c in (0, 1))
    return GyroConfig(
        rows=rows, cols=cols, edges=tuple(edges), in_idx=in_idx, out_idx=out_idx,
        J=complex_structure(rows * cols), n_classes=n_classes, settle=settle,
        force_scale=force_scale,
    )


def faithful_arrangement(
    cfg: GyroConfig, m, kappa, g_grav, L: float = ROD_LENGTH, parameter=rk4_gyro,
) -> Arrangement:
    """The open-port gyro classifier built ON ``compose_graph``.

    The closed physics -- springs (graph Laplacian) + rod gravity -- *emerges from the
    prism wiring* (``compose_graph(..., vdim=2, onsite=rod_gravity)``): its forcing
    ``omega_closed`` on ``Q = R^{2V}`` is the composite's, by pullback-and-sum. We then
    add an open input port (a drive ``-<n_in, q_in>`` in the cost, i.e. the forcing
    ``+n_in`` on the input gyros' coordinates and ``+q_in`` on the port) and an output
    readout (the output gyros' positions). The spring coupling is therefore the wired
    Laplacian, so freezing it (``kappa -> 0``) severs the input->output path -- the
    springs ablation is literally a wiring fact.

    Parameter ``parameter(Q)`` on ``Q`` with the composite's uniform sharp ``1/m``; by
    default the ``rk4_gyro`` four-stage coalgebra with all coefficients zero, and
    ``faithful_coalgebra`` passes the step, drag, per-gyro ``gamma`` and ``J``. (The
    composite's own tensored parameter is not used: the faithful machine steps the
    flat ``Q`` as one phase state, as CDLM did.)
    """
    closed = compose_graph(cfg.V, list(cfg.edges), m, kappa, vdim=2, onsite=rod_gravity(g_grav, L))
    n = closed.P.Q_dim
    Q = diagonal(jnp.full(n, 1.0 / m))  # the composite's block-diagonal sharp, uniform mass
    in_i = jnp.asarray(cfg.in_idx)
    out_i = jnp.asarray(cfg.out_idx)
    z0 = jnp.zeros(0)

    def out_f(q: Array, m_out: Array) -> Array:
        return q[out_i]  # read the output gyros' positions

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)  # closed source side (bang)

    def omega(q: Array, m_out: Array, n_in: Array):
        xi_q, _, _ = closed.omega(q, z0, z0)             # wired physics (springs + gravity)
        xi_q = xi_q.at[in_i].add(n_in)                    # plus the drive on the input gyros
        return (xi_q, jnp.zeros(0), q[in_i])              # and its port component

    return Arrangement(
        P=parameter(Q), in_dim_M=0, out_dim_M=0,
        in_dim_N=len(cfg.in_idx), out_dim_N=len(cfg.out_idx),
        out_f=out_f, in_f=in_f, omega=omega, label="faithful_gyro_net",
    )


def faithful_coalgebra(cfg: GyroConfig, p: Dict[str, Array], h: float = 0.1, L: float = ROD_LENGTH):
    """``PhiK`` of the faithful arrangement with the ``rk4_gyro`` parameter (RK4 phase,
    drag + per-gyro precession)."""
    arr = faithful_arrangement(
        cfg, jnp.exp(p["log_m"]), jnp.exp(p["log_kappa"]), jnp.exp(p["log_grav"]), L,
        parameter=lambda Q: rk4_gyro(
            Q, h, drag=jnp.exp(p["log_drag"]), gamma=jnp.repeat(p["gamma"], 2),
            J=cfg.J, gyro_block=2,
        ),
    )
    return PhiK(arr)


# ---------------------------------------------------------------------------
# Encoder / decoder (standard ML on top) + the rollout.
# ---------------------------------------------------------------------------


def init_params(cfg: GyroConfig, seed: int = 0, kappa0: float = 0.5, grav0: float = 0.2,
                drag0: float = 0.1) -> Dict[str, Array]:
    """Trainable physics (uniform log-mass / log-stiffness / log-gravity / log-drag,
    per-gyro gamma) + a linear encoder/decoder.

    Every positive physical quantity is carried in its logarithm, so training cannot flip
    its sign. Drag needs this as much as the others: the quadratic-drag 1-form is the only
    dissipative term ``rk4_gyro`` carries (unlike the ``gyro`` coalgebra it has no
    separate damping ``c``), and a negative coefficient makes it anti-damping ``~ +|v| v``
    -- energy injection growing superlinearly with speed, with nothing left to bound it.
    """
    rng = np.random.default_rng(seed)
    return {
        "log_m": jnp.array(0.0),
        "log_kappa": jnp.array(float(np.log(kappa0))),
        "log_grav": jnp.array(float(np.log(grav0))),
        "log_drag": jnp.array(float(np.log(drag0))),
        "gamma": jnp.zeros(cfg.V),
        "W_enc": jnp.asarray(0.3 * rng.standard_normal((cfg.n_in, 2))),
        "b_enc": jnp.zeros(cfg.n_in),
        "W_dec": jnp.asarray(0.1 * rng.standard_normal((cfg.n_classes, 2 * cfg.n_out))),
        "b_dec": jnp.zeros(cfg.n_classes),
    }


def _rollout_readout(O, drive_forces: Array, cfg: GyroConfig, inv_m) -> Array:
    """Run the coalgebra ``O`` forced by ``drive_forces`` (T_total, n_in); return the output
    gyros' final (positions, velocities). Only arrays cross the scan carry (no omega callables)."""
    out_idx = jnp.asarray(cfg.out_idx)
    zeros_out = jnp.zeros(cfg.n_out)

    def step(state, force):
        _outs, new_state = O.with_state(state).run_one(_IN_POS, lambda op: (zeros_out, force))
        return new_state, None

    state0 = (jnp.zeros(2 * cfg.V), jnp.zeros(2 * cfg.V))
    (q, xi), _ = jax.lax.scan(step, state0, drive_forces)
    return jnp.concatenate([q[out_idx], (inv_m * xi)[out_idx]])


def classify(p: Dict[str, Array], x_seq: Array, cfg: GyroConfig, h: float = 0.1,
             L: float = ROD_LENGTH) -> Array:
    """Logits for one stroke ``x_seq`` (T, 2): encode to input-gyro forces, run ``PhiK`` of
    the ``rk4_gyro`` machine forced for T steps then free for ``cfg.settle`` steps, decode
    the output gyros' readout."""
    O = faithful_coalgebra(cfg, p, h, L)
    forces = cfg.force_scale * (x_seq @ p["W_enc"].T + p["b_enc"])  # (T, n_in)
    forces = jnp.concatenate([forces, jnp.zeros((cfg.settle, cfg.n_in))], axis=0)
    readout = _rollout_readout(O, forces, cfg, jnp.exp(-p["log_m"]))
    return p["W_dec"] @ readout + p["b_dec"]


def output_readout(p: Dict[str, Array], drive_seq: Array, cfg: GyroConfig, h: float = 0.1,
                   L: float = ROD_LENGTH) -> Array:
    """The output gyros' (pos, vel) after being driven by ``drive_seq`` (T, n_in) -- the raw
    quantity the springs->0 ablation acts on (no encoder/decoder)."""
    O = faithful_coalgebra(cfg, p, h, L)
    forces = jnp.concatenate([drive_seq, jnp.zeros((cfg.settle, cfg.n_in))], axis=0)
    return _rollout_readout(O, forces, cfg, jnp.exp(-p["log_m"]))


def make_strokes(n: int, T: int = 16, n_classes: int = 4, seed: int = 0):
    """Synthetic 2-channel stroke series (generator-as-spec, reproducible from ``seed``).

    Class ``c`` is the documented signature: a ``(sin, cos)`` curve of frequency ``1 + c``
    and phase ``c*pi/n_classes`` over ``T`` steps, plus smooth Gaussian noise. No download,
    no provenance question -- an auditor regenerates the exact dataset from ``(seed, spec)``.
    """
    rng = np.random.default_rng(seed)
    y = rng.integers(0, n_classes, n)
    t = np.linspace(0.0, 1.0, T)
    X = np.zeros((n, T, 2))
    for i in range(n):
        c = int(y[i])
        freq, phase = 1.0 + c, c * np.pi / n_classes
        X[i, :, 0] = np.sin(2 * np.pi * freq * t + phase) + 0.1 * rng.standard_normal(T)
        X[i, :, 1] = np.cos(2 * np.pi * freq * t + phase) + 0.1 * rng.standard_normal(T)
    return jnp.asarray(X), jnp.asarray(y)


# ---------------------------------------------------------------------------
# Training (standard minibatch Adam, reusing gyroscope.py's helpers).
# ---------------------------------------------------------------------------


def batch_loss(p, xs, ys, cfg, h: float = 0.1, weight_decay: float = 1e-4):
    logits = jax.vmap(lambda x: classify(p, x, cfg, h))(xs)
    ce = jnp.mean(jax.vmap(cross_entropy)(logits, ys))
    l2 = sum(jnp.sum(p[k] ** 2) for k in ("W_enc", "W_dec"))
    return ce + weight_decay * l2


def accuracy(p, xs, ys, cfg, h: float = 0.1) -> float:
    logits = jax.vmap(lambda x: classify(p, x, cfg, h))(xs)
    return float(jnp.mean(jnp.argmax(logits, axis=1) == ys))


def train(cfg, Xtr, Ytr, *, epochs: int = 10, batch: int = 64, lr: float = 3e-3, h: float = 0.1,
          seed: int = 0, freeze: Sequence[str] = (), verbose: bool = False):
    """Minibatch Adam; ``freeze`` zeroes the gradient of the named params (for ablations,
    e.g. ``freeze=("log_kappa",)`` with tiny initial stiffness keeps the springs off)."""
    p = init_params(cfg, seed=seed)
    freeze = set(freeze)
    m, v = adam_init(p)
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]
    loss_and_grad = jax.jit(jax.value_and_grad(lambda pp, xb, yb: batch_loss(pp, xb, yb, cfg, h)))
    hist, t = [], 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        losses = []
        for i in range(0, n - batch + 1, batch):
            idx = perm[i : i + batch]
            t += 1
            loss, grads = loss_and_grad(p, Xtr[idx], Ytr[idx])
            if freeze:
                grads = {k: (jnp.zeros_like(g) if k in freeze else g) for k, g in grads.items()}
            p, m, v = adam_step(p, grads, m, v, t, lr=lr)
            losses.append(float(loss))
        hist.append(float(np.mean(losses)))
        if not np.isfinite(hist[-1]):
            # Adam carries NaN in its moments forever, so a diverged run would
            # otherwise return a silent NaN history and NaN parameters.
            raise FloatingPointError(
                f"training diverged at epoch {ep} (loss {hist[-1]}); lower lr or h"
            )
        if verbose:
            print(f"  epoch {ep:2d}  loss {hist[-1]:.4f}")
    return p, hist
