"""A rod tied by springs to data points fits the data (book sec. 1.5, "Motion and learning
share a grammar"; David, 2026-09-27: "can we actually write this down and test it?").

The picture. Put pins at the data points (x_i, y_i). A stiff rod rides on a vertical rail at
x = 0, so it can move up and down (height h) and tilt (angle theta). At each x_i a slider
keeps a zero-length spring vertical, joining the pin to the rod's point above or below it.
The rod's point at x_i is at height h + x_i tan(theta): that is the line a + b x with
a = h, b = tan(theta), the rod's two parameters. The springs' stored energy is

    U(h, theta) = (k/2) sum_i (h + x_i tan(theta) - y_i)^2,

which is k times the squared error of the line. So "attracted to the training data" is
literal for the rod's predictions: each spring pulls the rod toward one data point.

Predictions (Claude Opus 5.5 (1M), 2026-09-27), written in the same sitting as the code and
before any run, not before the code as the library's convention asks:

(1) Syrup. Overdamped motion (each coordinate's velocity proportional to minus its force,
    with a drag for h and a different drag for theta) comes to rest on the least-squares
    line, whatever the two drags are; the drags change the path, not the rest.
(2) Air, no friction. The rod has a mass and a moment of inertia and never settles: it
    swings about the least-squares line forever and the total energy stays constant.
(3) Air with friction. It overshoots, swings, and settles on the same least-squares line:
    momentum training.
(4) Springs not kept vertical. If each zero-length spring ends on a ring that slides freely
    along the rod, the spring pulls perpendicular to the rod, and the rest is the
    total-least-squares line (through the centroid, along the data's first principal
    direction), which differs from the least-squares line on noisy data. So the sliders
    are what make the rod do ordinary regression.

**Composed since P43 (b) job 7** (Claude Opus 5.5, de71554a, 2026-10-10): every run below is ``Phi`` of
``rod.rod_arrangement``, the rod, a pin, a slider (or a free ring) and a spring per point, each a box.
It was plain numpy; those loops are kept (``_syrup``, ``_air``) and checked equal. In air the rod's
integrator is the damped phase one, which evaluates the force after the drift where the numpy loop
kicked first; the two give the same positions once the rod starts half a kick ahead, its momentum set
to the force at its start (a start state set by hand, so said), and that is how the air runs start.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements import rod as R
from arrangements.functors import closed_step

jax.config.update("jax_enable_x64", True)

K = 1.0


def _data(seed=0, n=12):
    rng = np.random.default_rng(seed)
    x = np.linspace(-2.0, 2.0, n)
    y = 0.7 + 1.3 * x + 0.5 * rng.standard_normal(n)
    return x, y


def _ols(x, y):
    b, a = np.polyfit(x, y, 1)
    return a, b


def _tls(x, y):
    xm, ym = x.mean(), y.mean()
    _, _, vt = np.linalg.svd(np.column_stack([x - xm, y - ym]))
    dx, dy = vt[0]
    b = dy / dx
    return ym - b * xm, b


def _vertical_force(x, y, h, th):
    """Minus the gradient of U in (h, theta), springs kept vertical."""
    r = h + x * np.tan(th) - y
    return -K * r.sum(), -K * (r * x).sum() / np.cos(th) ** 2


def _ring_energy(x, y, h, th):
    """Zero-length springs on free rings: each pulls along the perpendicular to the rod."""
    d = (y - h - x * np.tan(th)) * np.cos(th)
    return 0.5 * K * (d ** 2).sum()


def _ring_force(x, y, h, th, eps=1e-7):
    f = lambda hh, tt: _ring_energy(x, y, hh, tt)
    return (-(f(h + eps, th) - f(h - eps, th)) / (2 * eps),
            -(f(h, th + eps) - f(h, th - eps)) / (2 * eps))


def _syrup(force, h, th, drag_h, drag_th, dt=1e-3, steps=200_000):
    for _ in range(steps):
        fh, ft = force(h, th)
        h, th = h + dt * fh / drag_h, th + dt * ft / drag_th
    return h, th


def _air(x, y, h, th, mass, inertia, friction, dt=1e-3, steps=200_000, record=False):
    """Semi-implicit Euler (symplectic when friction = 0) on T + U, springs vertical."""
    ph = pt = 0.0
    energies = []
    for _ in range(steps):
        fh, ft = _vertical_force(x, y, h, th)
        ph += dt * (fh - friction * ph / mass)
        pt += dt * (ft - friction * pt / inertia)
        h += dt * ph / mass
        th += dt * pt / inertia
        if record:
            r = h + x * np.tan(th) - y
            energies.append(0.5 * ph ** 2 / mass + 0.5 * pt ** 2 / inertia
                            + 0.5 * K * (r ** 2).sum())
    return h, th, np.array(energies)


_Z = jnp.zeros(0)


def _run(A, s, steps, record=None):
    step = closed_step(A)
    if record is None:
        return jax.lax.fori_loop(0, steps, lambda _i, t: step(t), s)
    return jax.lax.scan(lambda t, _: (step(t), record(step(t))), s, None, length=steps)


def syrup_c(x, y, h0, th0, drag_h, drag_th, dt=1e-3, steps=200_000, rings=False):
    """``Phi`` of the rod in syrup, springs on sliders (or free rings); the rod's ``(h, theta)`` at the end."""
    A = R.rod_arrangement(x, y, R.rod_box(h0, th0, dt=dt, drags=[drag_h, drag_th], n_out=len(x)), rings=rings)
    return tuple(float(v) for v in A.P.position(_run(A, A.P.init, steps)))


def air_c(x, y, h0, th0, mass, inertia, friction, dt=1e-3, steps=200_000, record=False):
    """``Phi`` of the rod in air, started half a kick ahead: ``(h, theta)`` at the end, and with
    ``record`` the energy after each step, momentum read back to the numpy loop's convention."""
    A = R.rod_arrangement(x, y, R.rod_box(h0, th0, dt=dt, mass_inertia=[mass, inertia], friction=friction,
                                          n_out=len(x)), scale=dt)
    leaves, tdef = jax.tree_util.tree_flatten(A.P.init)
    leaves[1] = A.omega(A.P.position(A.P.init), _Z, _Z)[0]
    s0 = jax.tree_util.tree_unflatten(tdef, leaves)
    mi = jnp.array([mass, inertia])

    def energy(s):
        q, pL = jax.tree_util.tree_leaves(s)[:2]
        c = dt * friction / mi                         # the damped integrator's friction per step
        p = (pL - A.omega(q, _Z, _Z)[0]) / (1 - c)     # the numpy loop's momentum at the same step
        r = q[0] + jnp.asarray(x) * jnp.tan(q[1]) - jnp.asarray(y)
        return 0.5 * jnp.sum(p ** 2 / mi) + 0.5 * K * jnp.sum(r ** 2)

    if record:
        s, E = _run(A, s0, steps, record=energy)
        q = jax.tree_util.tree_leaves(s)[0]
        return float(q[0]), float(q[1]), np.asarray(E)
    q = jax.tree_util.tree_leaves(_run(A, s0, steps))[0]
    return float(q[0]), float(q[1]), None


def test_the_composed_rod_is_the_numpy_rod():
    """The composed rod equals the numpy loops: syrup and rings over 3000 steps, air with friction
    over 20000 and without over 5000, positions to 1e-12, and the air energies to 1e-10."""
    x, y = _data()
    for dh, dth in [(1.0, 8.0), (5.0, 0.5)]:
        np.testing.assert_allclose(syrup_c(x, y, -1.0, -0.4, dh, dth, steps=3000),
                                   _syrup(lambda h, th: _vertical_force(x, y, h, th), -1.0, -0.4, dh, dth,
                                          steps=3000), atol=1e-12)
    np.testing.assert_allclose(syrup_c(x, y, -1.0, -0.4, 1.0, 1.0, steps=3000, rings=True),
                               _syrup(lambda h, th: _ring_force(x, y, h, th), -1.0, -0.4, 1.0, 1.0,
                                      steps=3000), atol=1e-6)   # the numpy ring force is a finite difference
    a, b = _ols(x, y)
    for fr, h0, th0, n in [(0.3, -1.0, -0.4, 20000), (0.0, a - 0.5, np.arctan(b) - 0.2, 5000)]:
        hc, tc, Ec = air_c(x, y, h0, th0, 1.0, 1.0, fr, steps=n, record=True)
        h, th, E = _air(x, y, h0, th0, 1.0, 1.0, fr, steps=n, record=True)
        np.testing.assert_allclose([hc, tc], [h, th], atol=1e-12)
        np.testing.assert_allclose(Ec, E, atol=1e-10)


def test_syrup_rests_on_least_squares_whatever_the_drag():
    x, y = _data()
    a, b = _ols(x, y)
    for drag_h, drag_th in [(1.0, 1.0), (1.0, 8.0), (5.0, 0.5)]:
        h, th = syrup_c(x, y, -1.0, -0.4, drag_h, drag_th)
        np.testing.assert_allclose([h, np.tan(th)], [a, b], atol=1e-6)


def test_air_without_friction_swings_forever_and_keeps_its_energy():
    x, y = _data()
    a, b = _ols(x, y)
    # start near rest: a hard swing drives theta toward +-pi/2, where tan blows up and the
    # fixed step no longer resolves the motion (the picture itself fails there too)
    h, th, E = air_c(x, y, a - 0.5, np.arctan(b) - 0.2, mass=1.0, inertia=1.0,
                     friction=0.0, steps=100_000, record=True)
    assert np.ptp(E) / E[0] < 1e-2
    # still far from rest at the end: the swing has not died down
    r_rest = a + x * b - y
    assert E[-1] > 0.5 * K * (r_rest ** 2).sum() + 0.1 * (E[0] - 0.5 * K * (r_rest ** 2).sum())


def test_air_with_friction_overshoots_then_settles_on_least_squares():
    x, y = _data()
    a, b = _ols(x, y)
    h, th, _ = air_c(x, y, -1.0, -0.4, mass=1.0, inertia=1.0, friction=0.3)
    np.testing.assert_allclose([h, np.tan(th)], [a, b], atol=1e-6)
    # overshoot: along the way the height passes the rest value and comes back
    hs = [air_c(x, y, -1.0, -0.4, 1.0, 1.0, 0.3, steps=n)[0] for n in range(1000, 20_001, 250)]
    assert max(hs) > a + 1e-3


def test_free_rings_give_total_least_squares_not_least_squares():
    x, y = _data()
    a_ols, b_ols = _ols(x, y)
    a_tls, b_tls = _tls(x, y)
    h, th = syrup_c(x, y, -1.0, -0.4, 1.0, 1.0, steps=100_000, rings=True)
    np.testing.assert_allclose([h, np.tan(th)], [a_tls, b_tls], atol=1e-4)
    assert abs(b_tls - b_ols) > 1e-2
