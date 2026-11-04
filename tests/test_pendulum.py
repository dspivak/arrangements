"""The n-fold pendulum by nesting links (book Ch 3 ``sec.arr``; ``arrangements/pendulum.py``).

The pendulum is ``drop . link^2 . pivot``, each link a bar from its support with a bob at its bottom
(turned right side up from the first draft's ``pivot . link . end``, same costs, at David's "yes," 2026-10-14).
Predictions, fixed before the first run (book ``notes/runs/pendulum-fa95b8ae/PREDICTIONS.md``, Claude Opus
5.5, fa95b8ae, 2026-10-14), from rest at angles 1 and 1.5 rad, bobs 1 kg, bars 1 m:
1. the nested composite is the same pendulum written flat, to 1e-10;
2. as the bars stiffen through 1e4, 1e6 and steel's 2e7 N/m, the lower bob comes closer to the rigid double
   pendulum over 1 s, within 1 mm at steel; steel stretches under 10 micrometres;
4. the bar sets the step: at steel a step of 1e-3 s blows up, one of 1e-4 s keeps the energy within 0.1%.
All held (the book's RESULTS.md beside the predictions); prediction 5, chaos, is not tested here.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from arrangements.functors import closed_step
from arrangements.pendulum import bob_states, pendulum, positions, bar_cost

M, L, G = 1.0, 1.0, 9.81
ANGLES = [1.0, 1.5]


def _runner(A):
    step = closed_step(A)
    return jax.jit(lambda s, k: jax.lax.fori_loop(0, k, lambda _i, t: step(t), s), static_argnums=1)


def _bobs(s):
    b = bob_states(s)
    return [q for q, _ in b], [p for _, p in b]


def _energy(s, kappa):
    qs, ps = _bobs(s)
    E = sum(0.5 * jnp.dot(p, p) / M + M * G * q[1] for q, p in zip(qs, ps))
    return E + bar_cost(kappa, L, jnp.zeros(2), qs[0]) + bar_cost(kappa, L, qs[0], qs[1])


def _rigid(T, h):
    """Rigid double pendulum, Lagrange's equations in the two angles, RK4 (ordinary code, the reference)."""
    def lag(th, w):
        x1 = L * jnp.array([jnp.sin(th[0]), -jnp.cos(th[0])])
        x2 = x1 + L * jnp.array([jnp.sin(th[1]), -jnp.cos(th[1])])
        v1 = L * w[0] * jnp.array([jnp.cos(th[0]), jnp.sin(th[0])])
        v2 = v1 + L * w[1] * jnp.array([jnp.cos(th[1]), jnp.sin(th[1])])
        return 0.5 * M * (v1 @ v1 + v2 @ v2) - M * G * (x1[1] + x2[1])

    dw = jax.grad(lag, 1)

    def f(y):
        th, w = y[:2], y[2:]
        acc = jnp.linalg.solve(jax.jacfwd(dw, 1)(th, w), jax.grad(lag, 0)(th, w) - jax.jacfwd(dw, 0)(th, w) @ w)
        return jnp.concatenate([w, acc])

    def rk4(y):
        k1 = f(y); k2 = f(y + h / 2 * k1); k3 = f(y + h / 2 * k2); k4 = f(y + h * k3)
        return y + h / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    return jax.jit(lambda y, k: jax.lax.fori_loop(0, k, lambda _i, t: rk4(t), y), static_argnums=1)


def test_nesting_is_the_flat_pendulum():
    kappa, dt, N = 1e4, 1e-3, 500
    A = pendulum(ANGLES, M, kappa, L, G, dt)
    assert A.label == "(drop . (link . (link . pivot)))"
    s = _runner(A)(A.P.init, N)
    q = jnp.stack(positions(ANGLES, L)); p = jnp.zeros_like(q)
    cost = lambda Q: bar_cost(kappa, L, jnp.zeros(2), Q[0]) + bar_cost(kappa, L, Q[0], Q[1]) + M * G * jnp.sum(Q[:, 1])
    g = jax.jit(jax.grad(cost))
    for _ in range(N):
        q = q + dt * p / M
        p = p - dt * g(q)
    assert float(jnp.max(jnp.abs(jnp.stack(_bobs(s)[0]) - q))) < 1e-10


def test_stiff_rods_approach_rigid_ones():
    gaps = []
    for kappa, dt in [(1e4, 1e-4), (1e6, 1e-4), (2e7, 2e-5)]:
        A = pendulum(ANGLES, M, kappa, L, G, dt)
        run, every = _runner(A), int(round(1e-3 / dt))
        rigid, h = _rigid(1.0, 1e-5), 1e-5
        s, y = A.P.init, jnp.array(ANGLES + [0.0, 0.0])
        worst = stretch = 0.0
        for _ in range(1000):
            s, y = run(s, every), rigid(y, int(round(every * dt / h)))
            (q1, q2), _ = _bobs(s)
            worst = max(worst, float(jnp.linalg.norm(q2 - positions([y[0], y[1]], L)[1])))
            stretch = max(stretch, abs(float(jnp.linalg.norm(q1)) - L), abs(float(jnp.linalg.norm(q2 - q1)) - L))
        gaps.append((worst, stretch))
    assert gaps[0][0] > gaps[1][0] > gaps[2][0]
    assert gaps[2][0] < 1e-3 and gaps[2][1] < 1e-5


def test_the_rod_sets_the_step():
    kappa = 2e7
    A = pendulum(ANGLES, M, kappa, L, G, 1e-3)
    s = _runner(A)(A.P.init, 1000)
    E = float(_energy(s, kappa))
    assert not (abs(E) < 1e6)                       # blown up (or overflowed)
    A = pendulum(ANGLES, M, kappa, L, G, 1e-4)
    run, E0 = _runner(A), float(_energy(A.P.init, kappa))
    s = A.P.init
    for _ in range(20):
        s = run(s, 500)
        assert abs(float(_energy(s, kappa)) - E0) < 1e-3 * abs(E0)
