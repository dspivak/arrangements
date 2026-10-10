"""Gyroscopes-on-springs as a classifier (EXTENSION, beyond the paper).

A harmonic surrogate of the system of M. S. Bull & S. Achour, *Machine learning with
dynamics* (unconv.ai, 2026): a network of gyroscopes-on-rods coupled by springs,
trained to classify handwriting. Each force of that physical system is mapped to one
slot of the arrangement framework (def:arr) -- the forward dynamics is *exactly*
``Phi`` (functors.py) applied to a single harmonic arrangement whose parameter is the
``gyro`` coalgebra (coalgebra.py):

    gravity restoring   -> on-site forcing ``omega = -dU``   (harmonic well)
    spring coupling     -> edge forcing ``omega = -dU``      (graph Laplacian, CDLM sec.graph_laplacian)
    inertia / mass      -> the reactive sharp                (``sharpR = 1/m``)
    damping             -> the damping 1-form                (``c*zeta``; ex.damping_one_form, ``damped``)
    gyroscopic torque   -> the skew 1-form                   (``gamma*omega_gyro``; ``gyro``)
    external nudge      -> the open input port               (encoder forces drive the input gyros)

EXTENSION caveat (cf. system_id.py / pinn.py): only the *physics* is the paper's
construction. Each gyro is a 2-D harmonic particle (Q = R^2, sharp 1/m); the
network is their graph-Laplacian arrangement at R^2, with the ``gyro`` coalgebra as
parameter. The *linear encoder/decoder and the Adam loop are standard ML layered on
top*, not part of the paper's formal development. And the gyroscopic 1-form is itself
beyond the paper (see ``coalgebra.gyro``).

What factors through the framework: the only thing that moves the gyros is
the ``gyro`` coalgebra's step under ``Phi`` -- a symplectic-Euler step of the
Hamiltonian whose forcing is the spring+gravity arrangement's and whose 1-form carries
damping and precession. Training backprops through the rollout with ``jax.grad``;
because ``cot``'s backward part is reverse-mode AD (functors.cot_map), this gradient
realizes the same chain-rule pullbacks that ``cot`` encodes -- an AD-equivalence,
*not* a literal ``PCMorphism.then`` composition. (Reserve ``pc^(K)`` / rmk.multistage
for genuine multi-stage *integrators* like leapfrog/RK4; the training rollout is not
one.)

**Training through the rollout is not a run of an arrangement** (plan P43 (b) job 7, Claude Opus 5.5,
de71554a, 2026-10-10). ``train`` backpropagates a loss at the rollout's end to the physics, encoder and
decoder and updates them with Adam between rollouts: an optimization over a family of arrangements. In a
run a force reaches a box only the step it is sent, so ``online_classifier`` (below) builds what can learn
inside one arrangement: the decoder, its loss read one way from the gyros, the physics and encoder held.
Found on the way: at the initial physics (gyroscopic coefficient 0.2, damping 0.05) the undriven network
grows without bound (a 0.1 kick to 5e7 in 300 steps); the 18-step rollouts from rest never showed it.

Two realizations of the same dynamics live here (STATUS.md §2, trick vs direct):

* ``gyro_arrangement`` -- the TRICK: the forcing is exact, ``omega = -dU`` on ``Q``, and
  the velocity-dependent damping, gyroscopic and drag 1-forms are smuggled into the
  parameter's step (``coalgebra.gyro``), because CDLM's arrangement carried only a
  potential on ``Q``;
* ``gyro_arrangement_on_cotangent`` -- the DIRECT one: the base is enlarged to
  ``T^*Q`` (``coalgebra.phase_on_cotangent``) and the gyroscopic and drag terms are
  written into the forcing itself, ``omega|_Q = -dU(q) + gyro_kick(sharp(xi))``, a
  genuinely non-exact one-form. The two agree step for step
  (tests/test_gyroscope.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, exact
from .coalgebra import gyro, gyro_kick, phase_on_cotangent
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import constant

_IN_POS = (jnp.zeros(0), trivial_omega(0))


# ---------------------------------------------------------------------------
# Topology: a grid of gyros, input column on the left, output column on the right.
# ---------------------------------------------------------------------------


def grid_graph(rows: int, cols: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """A ``rows x cols`` grid of gyros with 4-neighbour springs.

    Returns ``(edges, in_gyros, out_gyros)``; the input gyros are the left column
    and the output gyros the right column, so the encoder drives the left edge and
    the decoder reads the right edge -- information must propagate across the
    springs to be classified, the coupling the blog found essential.
    """
    def idx(r, c):
        return r * cols + c

    edges: List[Tuple[int, int]] = []
    for r in range(rows):
        for c in range(cols):
            if c + 1 < cols:
                edges.append((idx(r, c), idx(r, c + 1)))
            if r + 1 < rows:
                edges.append((idx(r, c), idx(r + 1, c)))
    in_gyros = [idx(r, 0) for r in range(rows)]
    out_gyros = [idx(r, cols - 1) for r in range(rows)]
    return edges, in_gyros, out_gyros


def hex_graph(rows: int, cols: int) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
    """A ``rows x cols`` triangular/hex lattice of gyros (up to 6 neighbours interior).

    Axial offset: each ``(i, j)`` bonds east ``(i, j+1)``, south ``(i+1, j)``, and the
    hex diagonal south-west ``(i+1, j-1)`` (directed, so each undirected bond appears
    once). As in ``grid_graph`` the input gyros are the left column and the output
    gyros the right column, so information must cross the lattice -- the coupling the
    blog found essential. The blog's network is this shape; here at small scale (the
    goal is the factorization + the springs ablation, not their accuracy number).
    """
    def idx(i, j):
        return i * cols + j

    edges: List[Tuple[int, int]] = []
    for i in range(rows):
        for j in range(cols):
            if j + 1 < cols:
                edges.append((idx(i, j), idx(i, j + 1)))           # east
            if i + 1 < rows:
                edges.append((idx(i, j), idx(i + 1, j)))           # south
            if i + 1 < rows and j - 1 >= 0:
                edges.append((idx(i, j), idx(i + 1, j - 1)))       # south-west (hex diagonal)
    in_gyros = [idx(i, 0) for i in range(rows)]
    out_gyros = [idx(i, cols - 1) for i in range(rows)]
    return edges, in_gyros, out_gyros


def complex_structure(V: int) -> Array:
    """Block-diagonal 90-degree rotation ``J``, one ``[[0,-1],[1,0]]`` per 2-D gyro
    (the complex structure the gyroscopic 1-form needs; see ``coalgebra.gyro``)."""
    blk = jnp.array([[0.0, -1.0], [1.0, 0.0]])
    return jax.scipy.linalg.block_diag(*([blk] * V))


@dataclass(frozen=True)
class GyroConfig:
    """Static topology + fixed hyperparameters of the gyro classifier."""

    rows: int
    cols: int
    edges: Tuple[Tuple[int, int], ...]
    in_idx: Tuple[int, ...]   # 2-D coordinate indices of the input gyros (len 2*rows)
    out_idx: Tuple[int, ...]  # 2-D coordinate indices of the output gyros
    J: Array
    n_classes: int = 10
    settle: int = 10          # free-evolution steps after the stroke (the blog's "5 seconds")
    damping: float = 0.05     # the damping 1-form coefficient c (fixed)
    force_scale: float = 0.3  # scales encoder forces into a stable regime

    @property
    def V(self) -> int:
        return self.rows * self.cols

    @property
    def n_in(self) -> int:
        return len(self.in_idx)

    @property
    def n_out(self) -> int:
        return len(self.out_idx)


def make_config(rows: int = 3, cols: int = 4, **kw) -> GyroConfig:
    edges, in_gyros, out_gyros = grid_graph(rows, cols)
    in_idx = tuple(2 * v + c for v in in_gyros for c in (0, 1))
    out_idx = tuple(2 * v + c for v in out_gyros for c in (0, 1))
    return GyroConfig(
        rows=rows, cols=cols, edges=tuple(edges),
        in_idx=in_idx, out_idx=out_idx, J=complex_structure(rows * cols), **kw,
    )


# ---------------------------------------------------------------------------
# The physics: one 2-D harmonic arrangement with the gyro coalgebra as parameter.
# ---------------------------------------------------------------------------


def rod_gravity(g: float, L: float, margin: float = 1e-3):
    """Nonlinear rod-gravity on-site potential for a 2-D gyro (the blog's rod geometry).

    A gyro tip displaced by ``q`` (horizontal, ``r = |q|``) on a rod of length ``L`` sits
    at height ``sqrt(L^2 - r^2)``; gravity gives the on-site potential

        U_grav(q) = -g * sqrt(L^2 - |q|^2),

    a restoring well with its minimum upright (``r = 0``). It is genuinely *nonlinear*:
    the restoring force ``-dU = -g q / sqrt(L^2 - r^2)`` stiffens as the rod nears
    horizontal (``r -> L``), and its small-tilt limit is ``(g/L) q`` -- exactly the
    harmonic well the surrogate used (spring constant ``g/L``). Plug into
    ``wiring.compose_graph(..., onsite=rod_gravity(g, L))`` so each gyro carries it
    (as a cost: the vertex's forcing is ``-d`` of it).

    The formula is only defined for ``|q| < L`` -- a rod cannot tilt past horizontal --
    and a bare ``sqrt`` returns ``NaN`` beyond it, which under ``jax.grad`` poisons the
    whole rollout (and, silently, every trained parameter) the first time any gyro is
    driven that far. So outside ``r^2 <= (1 - margin) L^2`` we continue ``U`` by its
    tangent line in ``r^2``: writing ``a = |q|^2`` and ``a_clamp = min(a, (1-margin)L^2)``,

        U_grav(q) = -g * ( sqrt(L^2 - a_clamp) - (a - a_clamp) / (2 sqrt(L^2 - a_clamp)) ).

    This is exactly the rod potential wherever the rod geometry is defined; past the
    clamp it is the ``C^1`` continuation, whose force ``-g q / sqrt(margin) L`` still
    points back upright (and grows with ``|q|``), so an over-driven gyro is pulled back
    rather than producing ``NaN``. ``margin`` therefore caps the stiffness of the wall.
    """
    a_max = (1.0 - margin) * L ** 2

    def U(q: Array) -> Array:
        a = jnp.sum(q ** 2)
        a_clamp = jnp.minimum(a, a_max)          # keeps the sqrt argument >= margin * L^2
        height = jnp.sqrt(L ** 2 - a_clamp)
        return -g * (height - (a - a_clamp) / (2.0 * height))

    return U


def gyro_potential(cfg: GyroConfig, kappa: Array, grav_k: Array):
    """The gyro network's cost ``U(q, m_out, n_in) = springs + on-site gravity - drive``
    on ``Q x M^+ x N^-`` (CDLM's sign: a cost). Its forcing is ``omega = -dU``
    (``exact``); both realizations below derive theirs from it."""
    V = cfg.V
    E = jnp.asarray(cfg.edges)          # (#edges, 2)
    in_idx = jnp.asarray(cfg.in_idx)
    has_edges = E.shape[0] > 0

    def U(q, m_out, n_in):
        qv = q.reshape(V, 2)
        if has_edges:
            diff = qv[E[:, 0]] - qv[E[:, 1]]
            springs = 0.5 * jnp.sum(kappa * jnp.sum(diff ** 2, axis=1))
        else:
            springs = jnp.array(0.0)
        gravity = 0.5 * grav_k * jnp.sum(qv ** 2)
        drive = jnp.dot(n_in, q[in_idx])  # linear forcing: -dU/dq_in = +force
        return springs + gravity - drive

    return U


def _gyro_rvect(cfg: GyroConfig, inv_mass: Array):
    """``Q = R^{2V}`` with constant sharp ``1/m`` per gyro (both components share its mass)."""
    inv_diag = jnp.repeat(inv_mass, 2)
    return constant(jnp.diag(inv_diag))


def gyro_arrangement(
    cfg: GyroConfig, inv_mass: Array, kappa: Array, grav_k: Array, parameter=gyro,
) -> Arrangement:
    """The 2-D gyro network as one ``Arrangement`` (CDLM sec.graph_laplacian at R^2,
    written directly), with an open force port on the input gyros.

    Parameter ``parameter(Q)`` on ``Q = R^{2V}`` (positions of all gyros), constant
    sharp ``1/m`` per gyro; by default the ``gyro`` coalgebra with all its coefficients
    zero (plain phase dynamics), and ``coalgebra`` below passes the damping, ``gamma``
    and ``J``. Codomain box ``<R^{2 n_in} (encoder force in) | R^{2 n_out} (positions
    out)>``. Forcing ``omega = -dU`` for the cost ``U = springs + on-site gravity -
    drive`` (``gyro_potential``); its value (computed by the framework's backward pass)
    is the force the coalgebra's step integrates.

    This is the TRICK realization (STATUS.md §2, trick vs direct): the forcing is exact on ``Q`` and
    the velocity-dependent damping, gyroscopic and drag 1-forms live inside the
    parameter's step (``coalgebra.gyro``). See ``gyro_arrangement_on_cotangent`` for
    the direct one.
    """
    Q = _gyro_rvect(cfg, inv_mass)
    out_idx = jnp.asarray(cfg.out_idx)
    U = gyro_potential(cfg, kappa, grav_k)

    def out_f(q, m_out):
        return q[out_idx]

    def in_f(q, m_out, n_in):
        return jnp.zeros(0)

    return Arrangement(
        P=parameter(Q), in_dim_M=0, out_dim_M=0,
        in_dim_N=len(cfg.in_idx), out_dim_N=len(cfg.out_idx),
        out_f=out_f, in_f=in_f,
        omega=exact(lambda q, m_out, n_in: -U(q, m_out, n_in)),
        label="gyro_net",
    )


def gyro_arrangement_on_cotangent(
    cfg: GyroConfig, inv_mass: Array, kappa: Array, grav_k: Array,
    gamma=0.0, J=None, drag=0.0, gyro_block=None, parameter=phase_on_cotangent,
) -> Arrangement:
    """The same gyro network with its gyroscopic and drag terms written as a forcing.

    This is the DIRECT realization (STATUS.md §2, trick vs direct): the base is ``T^*Q = R^{4V}``,
    the parameter ``parameter(Q)`` presents ``(q~, xi)`` (``coalgebra.phase_on_cotangent``,
    which keeps the paper's damping ``c*zeta`` in its step), and the forcing on
    ``T^*Q x M^+ x N^-`` is

        omega(q, xi, m, n)|_Q   = -dU(q, m, n)|_Q + gyro_kick(sharp_q(xi)),
        omega(q, xi, m, n)|_Q*  = 0,

    with the same cost ``U`` as ``gyro_arrangement`` (``gyro_potential``) and
    ``gyro_kick = -drag |v| v - gamma (J v)`` (coalgebra.py). The kick depends on the
    momentum, so this one-form is not ``dU'`` for any ``U'`` -- it is a genuinely
    non-exact forcing, which def:arr allows and CDLM's potential-only arrangement did
    not (hence the trick). For a constant sharp it agrees with ``gyro_arrangement``
    under ``gyro(Q, c, gamma, J, drag, gyro_block)`` step for step
    (tests/test_gyroscope.py); the readout and the input port are unchanged.
    """
    Q = _gyro_rvect(cfg, inv_mass)
    n = Q.dim
    out_idx = jnp.asarray(cfg.out_idx)
    U = gyro_potential(cfg, kappa, grav_k)
    dU = exact(lambda q, m_out, n_in: -U(q, m_out, n_in))

    def out_f(qp, m_out):
        return qp[:n][out_idx]

    def in_f(qp, m_out, n_in):
        return jnp.zeros(0)

    def omega(qp, m_out, n_in):
        q, xi = qp[:n], qp[n:]
        xi_q, xi_m, xi_n = dU(q, m_out, n_in)
        kick = gyro_kick(Q.apply_sharp(q, xi), gamma, J, drag, gyro_block)
        return (jnp.concatenate([xi_q + kick, jnp.zeros(n)]), xi_m, xi_n)

    return Arrangement(
        P=parameter(Q), in_dim_M=0, out_dim_M=0,
        in_dim_N=len(cfg.in_idx), out_dim_N=len(cfg.out_idx),
        out_f=out_f, in_f=in_f, omega=omega,
        label="gyro_net on T*Q",
    )


def physics_params(cfg: GyroConfig, kappa0: float = 0.3, grav0: float = 0.1, gamma0: float = 0.2) -> Dict[str, Array]:
    """Initial trainable physics: per-gyro log inverse-mass, per-edge log stiffness,
    log gravity, and gyroscopic coefficient. Logs keep masses/stiffness/gravity > 0."""
    return {
        "log_inv_mass": jnp.zeros(cfg.V),
        "log_kappa": jnp.full(len(cfg.edges), float(np.log(kappa0))),
        "log_grav": jnp.array(float(np.log(grav0))),
        "gamma": jnp.array(float(gamma0)),
    }


def coalgebra(cfg: GyroConfig, phys: Dict[str, Array]):
    """``Phi`` of the gyro arrangement at the given physics params, with the ``gyro``
    coalgebra (damping ``cfg.damping``, precession ``phys["gamma"]``, ``cfg.J``) as
    parameter."""
    arr = gyro_arrangement(
        cfg,
        inv_mass=jnp.exp(phys["log_inv_mass"]),
        kappa=jnp.exp(phys["log_kappa"]),
        grav_k=jnp.exp(phys["log_grav"]),
        parameter=lambda Q: gyro(Q, damping=cfg.damping, gamma=phys["gamma"], J=cfg.J),
    )
    return Phi(arr)


# ---------------------------------------------------------------------------
# Encoder / decoder (standard ML, layered on top) and the classifier.
# ---------------------------------------------------------------------------


def init_params(cfg: GyroConfig, seed: int = 0, **phys_kw) -> Dict[str, Array]:
    """All trainable parameters: encoder, decoder, and the physics."""
    rng = np.random.default_rng(seed)
    p = dict(physics_params(cfg, **phys_kw))
    p["W_enc"] = jnp.asarray(0.3 * rng.standard_normal((cfg.n_in, 2)))
    p["b_enc"] = jnp.zeros(cfg.n_in)
    p["W_dec"] = jnp.asarray(0.1 * rng.standard_normal((cfg.n_classes, 2 * cfg.n_out)))
    p["b_dec"] = jnp.zeros(cfg.n_classes)
    return p


def classify(params: Dict[str, Array], x_seq: Array, cfg: GyroConfig) -> Array:
    """Logits for one stroke sequence ``x_seq`` of shape ``(T, 2)``.

    Encoder maps each stroke point to forces on the input gyros; ``Phi`` of the gyro
    arrangement runs the network forced by them for ``T`` steps, then freely for
    ``cfg.settle`` steps; the decoder reads the output gyros' positions and
    velocities into logits.
    """
    O = coalgebra(cfg, params)
    inv_mass = jnp.exp(params["log_inv_mass"])
    inv_diag = jnp.repeat(inv_mass, 2)
    out_idx = jnp.asarray(cfg.out_idx)
    zeros_out = jnp.zeros(cfg.n_out)

    def step(state, force):
        pc = O.with_state(state)
        _out_pos, _out_dir, new_state = pc.run_one(_IN_POS, lambda op: (zeros_out, force))
        return new_state, None

    # encoder forces during the stroke, then zero forcing while the network settles
    forces = cfg.force_scale * (x_seq @ params["W_enc"].T + params["b_enc"])  # (T, n_in)
    forces = jnp.concatenate([forces, jnp.zeros((cfg.settle, cfg.n_in))], axis=0)

    state0 = (jnp.zeros(2 * cfg.V), jnp.zeros(2 * cfg.V))
    (q, xi), _ = jax.lax.scan(step, state0, forces)

    pos_out = q[out_idx]
    vel_out = (inv_diag * xi)[out_idx]
    readout = jnp.concatenate([pos_out, vel_out])  # output gyros' positions + velocities
    return params["W_dec"] @ readout + params["b_dec"]


def cross_entropy(logits: Array, label: int) -> Array:
    logp = logits - jax.scipy.special.logsumexp(logits)
    return -logp[label]


def batch_loss(params: Dict[str, Array], xs: Array, ys: Array, cfg: GyroConfig, weight_decay: float = 1e-4) -> Array:
    """Mean cross-entropy over a batch ``xs:(B,T,2)``, ``ys:(B,)`` + L2 on encoder/decoder."""
    logits = jax.vmap(lambda x: classify(params, x, cfg))(xs)
    ce = jnp.mean(jax.vmap(cross_entropy)(logits, ys))
    l2 = sum(jnp.sum(params[k] ** 2) for k in ("W_enc", "W_dec"))
    return ce + weight_decay * l2


def accuracy(params: Dict[str, Array], xs: Array, ys: Array, cfg: GyroConfig) -> float:
    logits = jax.vmap(lambda x: classify(params, x, cfg))(xs)
    return float(jnp.mean(jnp.argmax(logits, axis=1) == ys))


# ---------------------------------------------------------------------------
# Training: minibatch Adam (hand-rolled; no optax dependency).
# ---------------------------------------------------------------------------


def adam_init(params):
    return {k: jnp.zeros_like(v) for k, v in params.items()}, {k: jnp.zeros_like(v) for k, v in params.items()}


def adam_step(params, grads, m, v, t, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8):
    new_p, new_m, new_v = {}, {}, {}
    for k in params:
        new_m[k] = b1 * m[k] + (1 - b1) * grads[k]
        new_v[k] = b2 * v[k] + (1 - b2) * grads[k] ** 2
        mhat = new_m[k] / (1 - b1 ** t)
        vhat = new_v[k] / (1 - b2 ** t)
        new_p[k] = params[k] - lr * mhat / (jnp.sqrt(vhat) + eps)
    return new_p, new_m, new_v


def train(
    cfg: GyroConfig,
    Xtr: Array, Ytr: Array, Xva: Array, Yva: Array,
    *, epochs: int = 30, batch: int = 128, lr: float = 3e-3, seed: int = 0, verbose: bool = True,
    freeze: Sequence[str] = (), init_kw: Dict = None,
):
    """Train the gyro classifier with minibatch Adam; return ``(params, history)``.

    ``freeze`` holds parameter keys whose gradient is zeroed (held at their init) --
    used for ablations: ``freeze=("gamma",)`` turns off gyroscopic precession,
    ``freeze=("log_kappa",)`` with tiny initial stiffness decouples the gyros (the
    blog's "Take 1": input gyros cannot reach the output gyros without springs).
    """
    params = init_params(cfg, seed=seed, **(init_kw or {}))
    freeze = set(freeze)
    m, v = adam_init(params)
    rng = np.random.default_rng(seed)
    n = Xtr.shape[0]

    loss_and_grad = jax.jit(jax.value_and_grad(lambda p, xb, yb: batch_loss(p, xb, yb, cfg)))

    history = []
    t = 0
    for ep in range(epochs):
        perm = rng.permutation(n)
        losses = []
        for i in range(0, n - batch + 1, batch):
            idx = perm[i : i + batch]
            xb, yb = Xtr[idx], Ytr[idx]
            t += 1
            loss, grads = loss_and_grad(params, xb, yb)
            if freeze:
                grads = {k: (jnp.zeros_like(g) if k in freeze else g) for k, g in grads.items()}
            params, m, v = adam_step(params, grads, m, v, t, lr=lr)
            losses.append(float(loss))
        va = accuracy(params, Xva, Yva, cfg)
        history.append({"epoch": ep, "loss": float(np.mean(losses)), "val_acc": va})
        if verbose:
            print(f"  epoch {ep:3d}  loss {np.mean(losses):.4f}  val_acc {va:.4f}")
    return params, history


# ---------------------------------------------------------------------------
# PenDigits data (the blog's dataset) + a runnable surrogate.
# ---------------------------------------------------------------------------

_UCI = "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits"


def load_pendigits(cache_dir: str = None):
    """Load UCI PenDigits, downloading/caching on first use.

    Each sample is a stroke of 8 ``(x, y)`` pen points (the dataset's resampled
    form), reshaped to ``(8, 2)`` and rescaled from ``[0,100]`` to ``~[-1,1]``.
    Returns ``(Xtr, Ytr, Xte, Yte)``; the test split is the blog's "validation" set.
    """
    import os
    import urllib.request

    cache_dir = cache_dir or os.path.join(os.path.expanduser("~"), ".cache", "arrangements")
    os.makedirs(cache_dir, exist_ok=True)

    def fetch(split):
        path = os.path.join(cache_dir, f"pendigits.{split}")
        if not os.path.exists(path):
            urllib.request.urlretrieve(f"{_UCI}.{split}", path)
        raw = np.loadtxt(path, delimiter=",")
        X = raw[:, :16].reshape(-1, 8, 2).astype(np.float64) / 50.0 - 1.0
        return jnp.asarray(X), jnp.asarray(raw[:, 16].astype(int))

    Xtr, Ytr = fetch("tra")
    Xte, Yte = fetch("tes")
    return Xtr, Ytr, Xte, Yte


def render_stroke(points: Array, width: int = 22, height: int = 11) -> str:
    """An ASCII sketch of one pen stroke: the 8 ``(x, y)`` points joined in order.

    This is what the *input* looks like -- the path the pen traced. The classifier
    never sees this picture; it only feels the points as forces on the input gyros.
    """
    P = np.asarray(points, float)
    P = (P - P.min(0)) / (np.ptp(P, axis=0) + 1e-9)
    xs = (P[:, 0] * (width - 1)).astype(int)
    ys = ((1 - P[:, 1]) * (height - 1)).astype(int)  # flip y so the digit reads upright
    grid = [[" "] * width for _ in range(height)]
    for i in range(len(P) - 1):
        for t in np.linspace(0, 1, 30):
            x = int(round(xs[i] + t * (xs[i + 1] - xs[i])))
            y = int(round(ys[i] + t * (ys[i + 1] - ys[i])))
            grid[y][x] = "#"
    return "\n".join("".join(row) for row in grid)


def show_predictions(params: Dict[str, Array], cfg: GyroConfig, X: Array, Y: Array, n: int = 6, seed: int = 7):
    """Print per-digit accuracy and a few held-out strokes with ``true -> guess``."""
    pred = np.asarray(jnp.argmax(jax.vmap(lambda x: classify(params, x, cfg))(X), axis=1))
    true = np.asarray(Y)
    print("\nper-digit test accuracy:")
    for d in range(cfg.n_classes):
        mask = true == d
        if mask.sum():
            print(f"  {d}: {100 * np.mean(pred[mask] == d):5.1f}%  ({int(mask.sum())} samples)")
    print("\nheld-out test strokes (the pen path; true digit -> network's guess):")
    rng = np.random.default_rng(seed)
    for i in rng.choice(len(X), n, replace=False):
        flag = "OK" if pred[i] == true[i] else "XX"
        print(f"\n[{flag}] true {true[i]}  ->  guessed {pred[i]}")
        print(render_stroke(X[i]))


def main(rows: int = 4, cols: int = 5, epochs: int = 60, seed: int = 0):
    """Train the harmonic gyro surrogate on PenDigits (``python -m arrangements.gyroscope``).

    Trains the gyro network (``Phi`` with the ``gyro`` coalgebra) on PenDigits and
    prints its validation accuracy. The blog's reported numbers are shown for context
    only -- this is a harmonic surrogate of a richer machine, not a reproduction of
    their result. Pass ``--show`` to also print per-digit accuracy and a few pen
    strokes with the guess.

    Usage: ``python -m arrangements.gyroscope [rows cols epochs] [--show]``.
    """
    import sys

    show = "--show" in sys.argv
    pos = [a for a in sys.argv[1:] if not a.startswith("-")]
    if pos:
        rows, cols = int(pos[0]), int(pos[1])
        if len(pos) > 2:
            epochs = int(pos[2])

    print(f"PenDigits via Phi(gyro): {rows}x{cols} gyro grid, {epochs} epochs\n")
    Xtr, Ytr, Xte, Yte = load_pendigits()
    cfg = make_config(rows=rows, cols=cols)
    params, hist = train(cfg, Xtr, Ytr, Xte, Yte, epochs=epochs, seed=seed, verbose=True)
    best = max(h["val_acc"] for h in hist)
    print(f"\nbest val_acc {best:.4f}   (harmonic surrogate; for context only -- blog's richer "
          f"machine: linear 0.562, gyroscopes 0.834, LSTM 0.896; not a reproduction)")
    if show:
        show_predictions(params, cfg, Xte, Yte)


if __name__ == "__main__":
    main()


# ---------------------------------------------------------------------------
# The classifier as one arrangement, its decoder learning online (plan P43 (b) job 7).
# ---------------------------------------------------------------------------
#
# ``train`` above backpropagates a loss at the end of each rollout through the whole rollout to the
# physics, the encoder and the decoder, and Adam updates them between rollouts: an optimization over a
# family of arrangements, not a run of one. In a run, a force reaches a box's state only at the step it
# is sent (the dynamics functor's pullback is one step), so what can learn inside the arrangement is what
# the loss reaches the step it is paid: the decoder, which reads the output gyros at that step. Here the
# gyros, a fixed encoder and the decoder's weights are boxes in one closed arrangement: a stroke box and
# a label box step through the examples (each stroke, then the settle, then a quiet gap in which the
# damping clears the network); the loss, cross-entropy weighted by a flag the label box raises at the
# readout step, pushes the decoder's weights. The physics and encoder are held at their initial values.


def _stream(xs: Array, ys: Array, cfg: GyroConfig, gap: int):
    """Per step: the stroke point (zero after the stroke), the one-hot label and the readout flag."""
    xs, ys = np.asarray(xs), np.asarray(ys)
    n, T = xs.shape[0], xs.shape[1]
    L = T + cfg.settle + gap
    X = np.zeros((n, L, 2))
    X[:, :T] = xs
    Y = np.zeros((n, L, cfg.n_classes + 1))
    Y[np.arange(n), T + cfg.settle - 1, ys] = 1.0
    Y[:, T + cfg.settle - 1, -1] = 1.0
    return X.reshape(n * L, 2), Y.reshape(n * L, cfg.n_classes + 1)


def online_classifier(cfg: GyroConfig, params: Dict[str, Array], xs: Array, ys: Array, *, lr: float,
                      gap: int = 100) -> Arrangement:
    """The closed arrangement: stroke box -> encoder (the drive lens) -> gyros -> readout -> the decoder's
    flagged cross-entropy, read one way (it pushes the weights box, not the gyros), the label box beside. Its position is the
    stroke box's place, the label box's place, the gyros' ``(q~, xi)``, then the decoder's weights."""
    from .coalgebra import configuration
    from .network import example_box, map_box
    from .rvect import euclidean
    from .wiring import compose_seq, tensor_arrangements, trivial_parameter

    X, Y = _stream(xs, ys, cfg, gap)
    n = 2 * cfg.V
    inv_diag = jnp.repeat(jnp.exp(params["log_inv_mass"]), 2)
    out_idx = jnp.asarray(cfg.out_idx)
    net = gyro_arrangement_on_cotangent(
        cfg, jnp.exp(params["log_inv_mass"]), jnp.exp(params["log_kappa"]), jnp.exp(params["log_grav"]),
        gamma=params["gamma"], J=cfg.J,
        parameter=lambda Q: phase_on_cotangent(Q, damping=cfg.damping))
    R = 2 * cfg.n_out
    net = Arrangement(P=net.P, in_dim_M=0, out_dim_M=0, in_dim_N=net.in_dim_N, out_dim_N=R,
                      out_f=lambda qp, m: jnp.concatenate([qp[:n][out_idx], (inv_diag * qp[n:])[out_idx]]),
                      in_f=net.in_f, omega=net.omega, label="gyros (positions, velocities)")
    C = cfg.n_classes
    W0 = jnp.concatenate([params["W_dec"].ravel(), params["b_dec"]])
    weights = Arrangement(
        P=configuration(euclidean(W0.shape[0], lr), init=W0), in_dim_M=0, out_dim_M=0, in_dim_N=0,
        out_dim_N=W0.shape[0], out_f=lambda q, m: q, in_f=lambda q, m, nn: jnp.zeros(0),
        omega=lambda q, m, nn: (jnp.zeros_like(q), jnp.zeros_like(m), jnp.zeros_like(nn)), label="decoder weights")
    stroke, label = example_box(X, "stroke"), example_box(Y, "label")
    parts = tensor_arrangements([net, stroke, label, weights])
    # wires out: readout (R), stroke (2), label (C + 1), weights (W); the gyros' force input comes from the encoder
    F = net.in_dim_N
    Wenc, benc, fs = params["W_enc"], params["b_enc"], cfg.force_scale
    nW = W0.shape[0]

    def drive_in(q, m, nn):
        return fs * (Wenc @ m[R:R + 2] + benc)

    drive = Arrangement(
        P=trivial_parameter(), in_dim_M=F, out_dim_M=R + 2 + C + 1 + nW, in_dim_N=0, out_dim_N=R + C + 1 + nW,
        out_f=lambda q, m: jnp.concatenate([m[:R], m[R + 2:]]), in_f=drive_in,
        omega=lambda q, m, nn: (jnp.zeros_like(q), jnp.zeros_like(m), jnp.zeros_like(nn)), label="encoder")
    # the decoder reads the gyros one way: its loss pushes the weights and nothing else (an exact read would
    # push the gyros toward a lower loss, the classifier moving the network it reads)
    def decoder_loss(q, m, nn):
        r, W, lab, flag = m[:R], m[R:R + nW], m[R + nW:R + nW + C], m[R + nW + C]

        def ce(Wv):
            lg = Wv[:C * R].reshape(C, R) @ r + Wv[C * R:]
            return flag * (-jnp.dot(lab, lg - jax.scipy.special.logsumexp(lg)))

        gW = -jax.grad(ce)(W)
        return jnp.zeros_like(q), jnp.concatenate([jnp.zeros(R), gW, jnp.zeros(C + 1)]), jnp.zeros_like(nn)

    loss = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=R + nW + C + 1, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: jnp.zeros(0), in_f=lambda q, m, nn: jnp.zeros(0), omega=decoder_loss,
                       label="decoder and its flagged cross-entropy, reading one way")
    # after the encoder: [readout R, label C+1, weights nW]; reorder to [readout, weights, label]
    order = list(range(R)) + list(range(R + C + 1, R + C + 1 + nW)) + list(range(R, R + C + 1))
    from .network import fan_box
    head = fan_box(R + C + 1 + nW, order, "order")
    return compose_seq(compose_seq(compose_seq(parts, drive), head), loss)


def online_weights(arr: Arrangement, state) -> Array:
    """The decoder's weights in the online classifier's state."""
    leaves = jax.tree_util.tree_leaves(state)
    return leaves[4]


def online_accuracy(cfg: GyroConfig, params: Dict[str, Array], W: Array, xs: Array, ys: Array,
                    gap: int = 100) -> float:
    """Run the classifier with its weights held (response 0) over ``xs`` and read, at each readout
    step, which class its decoder ranks first."""
    from .functors import closed_step

    p = dict(params)
    C, R = cfg.n_classes, 2 * cfg.n_out
    p["W_dec"], p["b_dec"] = W[:C * R].reshape(C, R), W[C * R:]
    A = online_classifier(cfg, p, xs, ys, lr=0.0, gap=gap)
    step = closed_step(A, check=0)
    L = xs.shape[1] + cfg.settle + gap
    n = 2 * cfg.V
    inv_diag = jnp.repeat(jnp.exp(params["log_inv_mass"]), 2)
    out_idx = jnp.asarray(cfg.out_idx)

    def body(s, _):
        s = step(s)
        q, xi = jax.tree_util.tree_leaves(s)[:2]
        qt = q + inv_diag * xi
        r = jnp.concatenate([qt[out_idx], (inv_diag * xi)[out_idx]])
        return s, p["W_dec"] @ r + p["b_dec"]

    _, lg = jax.lax.scan(body, A.P.init, None, length=xs.shape[0] * L)
    at = np.arange(xs.shape[0]) * L + xs.shape[1] + cfg.settle - 2
    return float(np.mean(np.argmax(np.asarray(lg)[at], axis=1) == np.asarray(ys)))
