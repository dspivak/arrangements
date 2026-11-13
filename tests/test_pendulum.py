"""The n-fold pendulum, built by nesting links (book Chapter 3, ``sec.imc_arrangements``: ``eqn.link``,
``eqn.link_link``, the double pendulum and ``eqn.pendulum_cost``; linked from the footnote to the paragraph
"No real bar is perfectly stiff").

For a reader of the book. A *link* (``eqn.link``) is a bar of length ``ell`` and stiffness ``kappa`` hanging from
a support, with a bob of mass ``m`` at its bottom and gravity ``g`` pulling it down. In the library it is
``pendulum.link(m, kappa, ell, q0, g)``: its integrator is the bob (``pendulum.bob``: state ``(q, p)``, position and
momentum; told a duration ``tau`` it reads out its look-ahead ``q + tau p/m`` and, receiving a load there, adds
``tau`` times it to ``p``), its open box is the support, which reads out the point ``x`` the bar hangs from, and its
cost is the bar's ``pendulum.bar_cost(kappa, ell, x, q)``, the book's ``eqn.bar_cost``
``kappa/(8 ell^2) (|q - x|^2 - ell^2)^2``, plus gravity's ``m g q[1]``, the height being the second coordinate.
``pivot()`` is the support that never moves, reading out the point ``0``; ``drop()`` drops the last bob's position.
``pendulum(angles, m, kappa, ell, g, dt=1.0)`` is ``drop . link^n . pivot``, one link per angle, nested by
``compose_seq`` as in ``eqn.link_link``, started at rest with its bars at the given angles from straight down;
``bob_states`` reads the bobs' ``(q, p)`` off its state, top first. ``functors.closed_step(A, duration=tau)`` is one
step of the dynamics functor told the duration ``tau``.

The paragraph's numbers: two bobs of 1 kg on steel bars 1 m long, ``kappa = 2e7`` newtons per meter. Run from the
library's folder with it on the path, as the README's *Run it* says (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax
    jax.config.update("jax_enable_x64", True)
    from arrangements.functors import closed_step
    from arrangements.pendulum import pendulum, bob_states
    A = pendulum([1.0, 1.5], m=1.0, kappa=2e7, ell=1.0, g=9.81, dt=1.0)
    print(A.label)                                  # (drop . (link . (link . pivot)))
    for tau in (1e-3, 1e-4):                        # one vibration of the bars lasts about 1e-3 s
        step, s = closed_step(A, duration=tau), A.P.init
        for _ in range(round(0.1 / tau)):           # a tenth of a second
            s = step(s)
        print(tau, bob_states(s)[1][0])             # 1e-3: [nan nan], blown up; 1e-4: the lower bob at [1.82 -0.66]

What the tests check, each of which could fail. ``test_nesting_is_the_flat_pendulum``: the nested composite steps as
the cost ``eqn.pendulum_cost`` written flat, by hand, to 1e-10. ``test_stiff_bars_approach_rigid_ones``: as the bars
stiffen through 1e4, 1e6 and steel's 2e7 N/m, the lower bob comes closer to the rigid double pendulum (Lagrange's
equations, ordinary code beside the arrangement), within 1 mm at steel over 1 s, and steel stretches by a couple of
micrometers, between 1 and 5. ``test_the_rates_of_bar_and_swing``: linearized at the hanging rest, the composite's
own inclination field gives the bars' vibrations, about a thousand a second (440 and 1150), and the slow swing,
about once every two seconds (2.6 s). ``test_the_bar_sets_the_step``: one steel pendulum told a step of 1e-3 s, about
one vibration, blows up, and told 1e-4 s keeps its energy within 0.1% over 2 s; the step it can take is
``2/omega`` for its fastest vibration ``omega``; and the same step of 1e-3 s runs well on bars of ``kappa = 1e4``.

For agents: the heads before 2026-11-13 are in ``notes/P31-B-CH3-HISTORY-2026-11-13.md``.
"""

import jax

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from arrangements.functors import closed_step
from arrangements.pendulum import bob_states, pendulum, positions, bar_cost

M, L, G = 1.0, 1.0, 9.81
STEEL = 2e7
ANGLES = [1.0, 1.5]


def _runner(A, tau):
    """``k`` steps of ``A``, each told the duration ``tau``, compiled."""
    step = closed_step(A, duration=tau)
    return jax.jit(lambda s, k: jax.lax.fori_loop(0, k, lambda _i, t: step(t), s), static_argnums=1)


def _bobs(s):
    b = bob_states(s)
    return [q for q, _ in b], [p for _, p in b]


def _energy(s, kappa):
    qs, ps = _bobs(s)
    E = sum(0.5 * jnp.dot(p, p) / M + M * G * q[1] for q, p in zip(qs, ps))
    return E + bar_cost(kappa, L, jnp.zeros(2), qs[0]) + bar_cost(kappa, L, qs[0], qs[1])


def _rigid(h):
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


def _rates(kappa):
    """The angular frequencies, in radians per second, of the double pendulum's small motions about its hanging rest,
    read off the composite's own inclination field: its rest is where the field vanishes, and minus its Jacobian there,
    divided by the mass, has the squared angular frequencies as eigenvalues."""
    A = pendulum([0.0, 0.0], M, kappa, L, G, dt=1.0)
    field = lambda q: A.omega(q, jnp.zeros(0), jnp.zeros(0))[0]
    q = jnp.concatenate(positions([0.0, 0.0], L))
    for _ in range(20):    # Newton to the rest: the bars stretch under the bobs' weight (pinv: no sideways stiffness yet)
        q = q - jnp.linalg.pinv(jax.jacfwd(field)(q)) @ field(q)
    assert float(jnp.max(jnp.abs(field(q)))) < 1e-6
    return np.sqrt(np.sort(np.linalg.eigvalsh(-np.asarray(jax.jacfwd(field)(q)) / M)))


def test_nesting_is_the_flat_pendulum():
    """Could fail: nesting could wire a bar to the wrong support or drop a bob's gravity."""
    kappa, tau, N = 1e4, 1e-3, 500
    A = pendulum(ANGLES, M, kappa, L, G, dt=1.0)
    assert A.label == "(drop . (link . (link . pivot)))"
    s = _runner(A, tau)(A.P.init, N)
    q = jnp.stack(positions(ANGLES, L)); p = jnp.zeros_like(q)
    cost = lambda Q: bar_cost(kappa, L, jnp.zeros(2), Q[0]) + bar_cost(kappa, L, Q[0], Q[1]) + M * G * jnp.sum(Q[:, 1])
    g = jax.jit(jax.grad(cost))
    for _ in range(N):
        q = q + tau * p / M
        p = p - tau * g(q)
    assert float(jnp.max(jnp.abs(jnp.stack(_bobs(s)[0]) - q))) < 1e-10


def test_stiff_bars_approach_rigid_ones():
    """Could fail: stiffer bars could stray further from the rigid pendulum, or steel could stretch visibly."""
    gaps = []
    for kappa, tau in [(1e4, 1e-4), (1e6, 1e-4), (STEEL, 2e-5)]:
        A = pendulum(ANGLES, M, kappa, L, G, dt=1.0)
        run, every = _runner(A, tau), int(round(1e-3 / tau))
        h = 1e-5
        rigid = _rigid(h)
        s, y = A.P.init, jnp.array(ANGLES + [0.0, 0.0])
        worst = stretch = 0.0
        for _ in range(1000):
            s, y = run(s, every), rigid(y, int(round(every * tau / h)))
            (q1, q2), _ = _bobs(s)
            worst = max(worst, float(jnp.linalg.norm(q2 - positions([y[0], y[1]], L)[1])))
            stretch = max(stretch, abs(float(jnp.linalg.norm(q1)) - L), abs(float(jnp.linalg.norm(q2 - q1)) - L))
        print("kappa", kappa, "largest gap to rigid", worst, "largest stretch", stretch)
        gaps.append((worst, stretch))
    assert gaps[0][0] > gaps[1][0] > gaps[2][0]
    assert gaps[2][0] < 1e-3 and 1e-6 < gaps[2][1] < 5e-6


def test_the_rates_of_bar_and_swing():
    """Could fail: the bars could ring far from a thousand times a second, or the swing far from once in two seconds."""
    w = _rates(STEEL)
    print("angular frequencies (rad/s)", w, "cycles per second", w / (2 * np.pi), "periods (s)", 2 * np.pi / w)
    swing_slow, swing_fast, bar_slow, bar_fast = w
    assert 1.5 < 2 * np.pi / swing_slow < 3.0
    assert 300 < bar_slow / (2 * np.pi) < bar_fast / (2 * np.pi) < 2000


def test_the_bar_sets_the_step():
    """Could fail: steel could run at a step as long as one vibration, or the limit could be set by something other
    than the bars' fastest vibration, so that softer bars did no better."""
    A = pendulum(ANGLES, M, STEEL, L, G, dt=1.0)
    limit = 2 / _rates(STEEL)[-1]                     # symplectic Euler holds while tau omega < 2: about 2.8e-4 s
    for tau, holds in [(1e-3, False), (1.5 * limit, False), (0.75 * limit, True), (1e-4, True)]:
        s = _runner(A, tau)(A.P.init, int(round(1.0 / tau)))
        E = float(_energy(s, STEEL))
        print("steel, tau", tau, "energy after 1 s", E)
        assert (abs(E - float(_energy(A.P.init, STEEL))) < 1.0) == holds
    run, E0, s = _runner(A, 1e-4), float(_energy(A.P.init, STEEL)), A.P.init
    for _ in range(20):
        s = run(s, 1000)
        assert abs(float(_energy(s, STEEL)) - E0) < 1e-3 * abs(E0)
    soft = pendulum(ANGLES, M, 1e4, L, G, dt=1.0)
    s = _runner(soft, 1e-3)(soft.P.init, 2000)
    print("kappa 1e4, tau 1e-3, energy after 2 s", float(_energy(s, 1e4)), "from", float(_energy(soft.P.init, 1e4)))
    assert abs(float(_energy(s, 1e4)) - float(_energy(soft.P.init, 1e4))) < 1e-2 * abs(float(_energy(soft.P.init, 1e4)))
