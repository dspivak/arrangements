"""Run the worked examples and print what happens.

Usage (from ``dap-core``):

    uv run dap-demo        # or: uv run python -m dap.demo

Each example builds an arrangement (def:arr) -- the integrator is its parameter
coalgebra -- and applies the one dynamics functor ``Phi`` (sec.dynamics_functor):
``configuration`` parameters for descent (Newton, gradient descent, the heat
equation) and ``phase`` parameters for Hamilton flow (the wave equation). CDLM's
``Phiconf``/``Phiphase`` are ``Phi`` of those two arrangements.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from arrangements.interpretation import trivial_omega
from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration, damped, phase
from arrangements.functors import Phi
from arrangements.polynomial import nest_left, unnest_left
from arrangements.rvect import diagonal, euclidean, inverse_hessian
from arrangements.wiring import compose_chain
from arrangements.learning import parameterized_map, train, squared_error

_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def _harmonic(parameter, m, kappa):
    return Arrangement(
        P=parameter(diagonal(jnp.array([1.0 / m]))),
        in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -0.5 * kappa * (q[0] - n_in[0]) ** 2),
    )


def _closed(P, k):
    """A closed arrangement with parameter ``P`` and cost ``k(q)`` (forcing ``-dk``)."""
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -k(q)),
    )


def _conf_state(q):
    K = q.shape[0]
    return (nest_left([q[i:i + 1] for i in range(K)]), jnp.zeros(0))


def _phase_state(q, p):
    K = q.shape[0]
    return (nest_left([(q[i:i + 1], p[i:i + 1]) for i in range(K)]), jnp.zeros(0))


def _flat(state, K, is_phase):
    parts = unnest_left(state[0], K)
    if is_phase:
        return jnp.concatenate([s[0] for s in parts]), jnp.concatenate([s[1] for s in parts])
    return jnp.concatenate(parts)


def _laplacian_pinned(v):
    aug = np.concatenate([[0.0], np.asarray(v), [0.0]])
    return aug[:-2] - 2.0 * aug[1:-1] + aug[2:]


def demo_gradient_descent():
    m, n = 3, 2
    dim = n * m + n
    F = lambda q, x: q[: n * m].reshape(n, m) @ x + q[n * m:]
    rng = np.random.default_rng(7)
    W_true, b_true = rng.standard_normal((n, m)), rng.standard_normal(n)
    data = [(jnp.asarray(x := rng.standard_normal(m)), jnp.asarray(W_true) @ x + jnp.asarray(b_true))
            for _ in range(20)]
    arr = parameterized_map(F, configuration(euclidean(dim, eta=0.05)), m, n)
    q, hist = train(arr, jnp.zeros(dim), data, steps=4000)
    W_err = float(jnp.linalg.norm(q[: n * m].reshape(n, m) - jnp.asarray(W_true)))
    print(f"  gradient descent (conf):    loss {hist[0]:.3f} -> {hist[-1]:.1e},  weight error {W_err:.1e}")


def demo_newton():
    k = lambda q: jnp.exp(q[0]) - q[0]   # minimized at 0
    O = Phi(_closed(configuration(inverse_hessian(k, 1)), k))
    q, xs = jnp.array([2.0]), [2.0]
    for _ in range(6):
        _, _, q = O.with_state(q).run_one(_IN_POS, _TRIV)
        xs.append(float(q[0]))
    print("  newton (conf, Newton sharp): x = " + " -> ".join(f"{v:.1e}" for v in xs) + "   (-> min at 0)")


def demo_wave():
    K, m, kappa = 5, 1.5, 0.9
    O = Phi(compose_chain([_harmonic(phase, m, kappa)] * K))
    rng = np.random.default_rng(0)
    q, p = jnp.asarray(rng.standard_normal(K)), jnp.zeros(K)
    traj = [np.asarray(q)]
    for _ in range(12):
        qK = float(q[K - 1] + p[K - 1] / m)  # pinned end at the PRESENTED position q~_K
        in_dir = (jnp.array([-kappa * qK]), jnp.array([0.0]))  # the forcing of the pin
        _, _, state = O.with_state(_phase_state(q, p)).run_one(_IN_POS, lambda _o: in_dir)
        q, p = _flat(state, K, True)
        traj.append(np.asarray(q))
    a = np.stack(traj)
    # m * ddot q(t+1) = kappa * Lap q(t+1), Laplacian at the CENTER t+1.
    res = max(float(np.abs(m * (a[t + 2] - 2 * a[t + 1] + a[t]) - kappa * _laplacian_pinned(a[t + 1])).max())
              for t in range(len(a) - 2))
    print(f"  wave (phase):               discrete-wave residual {res:.1e}  (exact recurrence)")


def demo_heat():
    K, T, m, kappa = 9, 500, 1.0, 0.2
    O = Phi(compose_chain([_harmonic(configuration, m, kappa)] * K))
    rng = np.random.default_rng(4)
    q = jnp.asarray(rng.standard_normal(K))
    peak0 = float(np.max(np.abs(np.asarray(q))))
    for _ in range(T):
        in_dir = (jnp.array([-kappa * float(q[K - 1])]), jnp.array([0.0]))
        _, _, state = O.with_state(_conf_state(q)).run_one(_IN_POS, lambda _o: in_dir)
        q = _flat(state, K, False)
    print(f"  heat (conf):                peak |q| {peak0:.2f} -> {float(np.max(np.abs(np.asarray(q)))):.1e}  (dissipates to equilibrium)")


def demo_leapfrog():
    from arrangements.leapfrog import leapfrog
    from arrangements.pc2 import Phi2
    K, m, kappa = 5, 1.5, 0.9
    O = Phi2(compose_chain([_harmonic(leapfrog, m, kappa)] * K))
    rng = np.random.default_rng(0)
    q0 = jnp.asarray(rng.standard_normal(K))
    state = _phase_state(q0, jnp.zeros(K))
    peak = peak0 = float(np.max(np.abs(np.asarray(q0))))
    bdy = lambda op: (-kappa * op[0], jnp.array([0.0]))
    for _ in range(400):
        _, _, state = O.with_state(state).run_one(_IN_POS, bdy)
        peak = max(peak, float(np.max(np.abs(np.asarray(_flat(state, K, True)[0])))))
    print(f"  wave (leapfrog, pc^(2)):   peak |q| {peak0:.2f} -> max {peak:.2f} over 400 steps  (bounded -- stable)")


def demo_rk4():
    # RK4 as one pc^(4) instance: four force evaluations per macro-tick. On the
    # flow qdot = -A q the global error falls like h^4 (4th order).
    from arrangements.pcK import PhiK
    from arrangements.rk4 import rk4
    A = np.diag([1.0, 2.0, 5.0])
    k = lambda q: 0.5 * q @ (jnp.asarray(A) @ q)
    x0 = jnp.array([1.0, 1.0, 1.0])
    exact_sol = np.asarray(jax.scipy.linalg.expm(-A) @ np.asarray(x0))

    def err(N):
        O, s = PhiK(_closed(rk4(euclidean(3, 1.0), 1.0 / N), k)), x0
        for _ in range(N):
            _, s = O.with_state(s).run_one(_IN_POS, _TRIV)
        return float(np.linalg.norm(np.asarray(s) - exact_sol))

    e8, e16 = err(8), err(16)
    print(f"  rk4 (pc^(4)):              error {e8:.1e} -> {e16:.1e} as h halves  (ratio {e8 / e16:.0f} ~ 16, i.e. 4th order)")


def demo_optimizers():
    # One convex cost, three parameters: the optimizer IS the coalgebra.
    A = jnp.diag(jnp.array([1.0, 3.0, 9.0]))
    k = lambda q: 0.5 * q @ (A @ q)
    Q = euclidean(3, 0.1)
    q0 = jnp.ones(3)

    def run(P, is_phase):
        O = Phi(_closed(P, k))
        s = (q0, jnp.zeros(3)) if is_phase else q0
        for _ in range(600):
            _, _, s = O.with_state(s).run_one(_IN_POS, _TRIV)
        return float(jnp.linalg.norm(s[0] if is_phase else s))

    c, p, d = run(configuration(Q), False), run(phase(Q), True), run(damped(Q, 0.15), True)
    print(f"  optimizers (one well, 3 coalgebras):  |q| after 600 -> descent {c:.0e},  phase {p:.2f} (oscillates),  momentum {d:.0e}")


def demo_pinn():
    from arrangements.pinn import solve_deep_ritz
    u, u_star, hist, x = solve_deep_ritz(steps=3000, seed=0)
    rel = float(np.linalg.norm(np.asarray(u) - np.asarray(u_star)) / np.linalg.norm(np.asarray(u_star)))
    print(f"  pinn (deep Ritz, Poisson, conf):     energy {hist[0]:.3f} -> {hist[-1]:.3f},  rel error vs discrete solution {rel:.1%}")


def demo_system_id():
    from arrangements.system_id import identify, one_step_error, PENDULUM
    params, F, hist, O, dim, m = identify(PENDULUM, steps=2000, seed=1)
    err, base = one_step_error(params, F, O, dim, m, seed=101)
    print(f"  system id (pendulum: phase -> net):  one-step error {err:.3f}  vs no-change baseline {base:.3f}")


def demo_gyroscope():
    # A network of 2-D gyros (springs + gravity + damping + gyroscopic precession),
    # with the ``gyro`` parameter, trained to classify strokes -- a harmonic surrogate
    # inspired by Bull & Achour. Here on a fast synthetic 2-class stroke task.
    from arrangements.gyroscope import make_config, train, accuracy

    rng = np.random.default_rng(1)
    def synth(n, T=6):
        y = rng.integers(0, 2, n)
        x = 0.5 * np.where(y == 0, 1.0, -1.0)[:, None, None] + 0.25 * rng.standard_normal((n, T, 2))
        return jnp.asarray(x), jnp.asarray(y)

    cfg = make_config(rows=2, cols=3, n_classes=2, settle=6)
    Xtr, Ytr = synth(400)
    Xva, Yva = synth(200)
    _, hist = train(cfg, Xtr, Ytr, Xva, Yva, epochs=15, batch=64, lr=5e-3, verbose=False)
    print(f"  gyroscope (springs net: gyro -> logits):  val_acc {max(h['val_acc'] for h in hist):.3f}  on a 2-class stroke task")
    print("    (harmonic surrogate inspired by Bull & Achour; `python -m dap.gyroscope` -> ~0.83 at 3x4, ~0.87 at 4x5)")


def demo_faithful():
    # The faithful Bull & Achour machine (gyroscope_faithful.py): hex springs from the prism
    # wiring, nonlinear rod gravity, quadratic drag + per-gyro precession, RK4 (pc^(4)). The
    # headline is the springs->0 ablation, NOT accuracy -- a one-line categorical wiring fact.
    from arrangements.gyroscope_faithful import make_hex_config, init_params, output_readout

    cfg = make_hex_config(3, 3, n_classes=4, settle=10)
    p = init_params(cfg, seed=0)
    rng = np.random.default_rng(1)
    drive = jnp.asarray(0.2 * rng.standard_normal((8, cfg.n_in)))
    on = float(jnp.linalg.norm(output_readout(p, drive, cfg)))
    off = float(jnp.linalg.norm(output_readout({**p, "log_kappa": jnp.array(-100.0)}, drive, cfg)))
    print(f"  faithful gyro (pc^(4) RK4): output |readout| {on:.2f} with springs -> {off:.0e} with springs frozen")
    print("    (springs->0 severs input->output: the blog's 'Take 1' as a wiring fact; no accuracy claim)")


def main():
    print("dap: worked examples (one functor Phi; the integrator is the arrangement's parameter)\n")
    print("configuration parameters -- descent dynamics:")
    demo_gradient_descent()
    demo_newton()
    demo_heat()
    print("\nphase parameters -- Hamiltonian dynamics (pc):")
    demo_wave()
    print("\nMulti-stage parameters (pc^(K)):")
    demo_leapfrog()
    demo_rk4()
    print("\nExtensions (beyond the paper; the same pieces, composed):")
    demo_optimizers()
    demo_pinn()
    demo_system_id()
    demo_gyroscope()
    demo_faithful()


if __name__ == "__main__":
    main()
