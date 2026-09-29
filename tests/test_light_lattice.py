"""Light on a lattice built by composition (book plan P32 (a); `notes/P32-A-PREDICTIONS-2026-09-29.md`).

The composite ``light_lattice(N, kappa)`` (arrangements/light_lattice.py): 128 edge boxes with
the ``phase`` parameter and unit mass, 64 stateless face boxes with cost ``(kappa/2) B_f^2``, a
copy wire, on the 8 x 8 periodic grid. Its forcing ``-kappa C^T C A`` is written nowhere by hand.
The oracle is ``scripts/maxwell2d.py``, whose ``curl``/``gradU``/``div`` are copied verbatim
below (the script runs N = 200 at import, so it is not imported). The script's step ``dt`` is the
composite with ``kappa = dt^2`` and momenta ``xi = dt * x`` (the library has no dt; STATUS.md §1).

The five predictions, fixed before the code was written (the notes file quotes them in full):

1. the composite forcing equals ``-d`` of the total face energy, against two oracles, to 1e-10;
2. one ``Phi`` step equals one step of the script's update on the same data, to 1e-12;
3. the discrete divergence of the momenta (Gauss's law) is conserved to 1e-12 along a run
   started with NONZERO divergence;
4. the eigenmode ``A = C^T cos(k . r)`` obeys ``omega^2 = 4 sin^2(kx/2) + 4 sin^2(ky/2)`` in the
   small-step limit (rtol 1e-3 at dt = 0.05) and the discrete relation
   ``(2/dt)^2 sin^2(theta/2) = Omega^2`` to rtol 1e-9; the phase speed falls with ``|k|``;
5. energy stays within 3 percent over 400 steps at dt = 0.5 -- REFUTED for the naive energy
   (33 percent, a bounded oscillation of the time-staggered scheme, no secular drift); the
   staggered energy ``1/2 sum x_n^2 + 1/2 <B_n, B_{n+1}>`` is conserved to rounding.

**Every test below says which of three things it does** (the convention of test_rendezvous.py):
reproduce a CLOSED FORM, check a DISCRETE ANALOGUE, or merely ILLUSTRATE. Which could have
come out otherwise: (2), (4) and (5) could, and (5) did; (1) and (3) are what ``compose_seq`` was built to do
on a wire that is a copy rather than a permutation (STATUS.md §2, "ask which predictions could
have come out otherwise"), and (3) follows from (1) since ``div C^T = 0``.

Claude Fable 5.1, session ec5ece87, 2026-09-29.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.light_lattice import light_lattice, pack_state, unpack_state

N = 8
_IN_POS = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))
_Z0 = jnp.zeros(0)


# ---------------------------------------------------------------------------
# Oracle: copied verbatim from scripts/maxwell2d.py (plain numpy, periodic rolls).
# ---------------------------------------------------------------------------


def curl(Ax, Ay):   # face (i,j): Ax[i,j]+Ay[i+1,j]-Ax[i,j+1]-Ay[i,j]
    return Ax + np.roll(Ay, -1, 0) - np.roll(Ax, -1, 1) - Ay


def gradU(Ax, Ay):  # d/dA of 1/2 sum B^2  (transpose of curl applied to B)
    B = curl(Ax, Ay)
    gx = B - np.roll(B, 1, 1); gy = np.roll(B, 1, 0) - B
    return gx, gy


def div(xx, xy):    # at vertices: sum of incoming minus outgoing
    return (np.roll(xx, 1, 0) - xx) + (np.roll(xy, 1, 1) - xy)


def curl_T(B):
    """The transposed curl on a face field (``gradU`` with ``B`` given): a divergence-free
    edge field, the initial data of the script and the eigenmodes of ``C^T C``."""
    return B - np.roll(B, 1, 1), np.roll(B, 1, 0) - B


def script_step(Ax, Ay, xx, xy, dt):
    """One step of scripts/maxwell2d.py's loop: q' = q + dt xi; xi' = xi - dt dU(q')."""
    Ax = Ax + dt * xx; Ay = Ay + dt * xy
    gx, gy = gradU(Ax, Ay); xx = xx - dt * gx; xy = xy - dt * gy
    return Ax, Ay, xx, xy


def energy(Ax, Ay, xx, xy):
    return 0.5 * (xx ** 2 + xy ** 2).sum() + 0.5 * (curl(Ax, Ay) ** 2).sum()


def to_flat(X, Y):
    return np.concatenate([np.asarray(X).ravel(), np.asarray(Y).ravel()])


def from_flat(v):
    v = np.asarray(v)
    return v[:N * N].reshape(N, N), v[N * N:].reshape(N, N)


# ---------------------------------------------------------------------------
# The composite and its jitted step (the framework's own run_one; jit compiles only that).
# ---------------------------------------------------------------------------


@functools.lru_cache(maxsize=None)
def _system(kappa):
    arr = light_lattice(N, kappa=kappa)
    O = Phi(arr)

    def f(s):
        _, _, s2 = O.with_state(s).run_one(_IN_POS, _TRIV)
        return s2

    return arr, jax.jit(f)


def run_composite(dt, Ax, Ay, xx, xy, steps):
    """``steps`` Phi-steps of the composite at ``kappa = dt^2`` from the script's data
    ``(A, x)``; returns the trajectory in the script's variables (momenta ``x = xi/dt``)."""
    arr, step = _system(dt * dt)
    s = pack_state(arr, to_flat(Ax, Ay), dt * to_flat(xx, xy))
    traj = [(Ax, Ay, xx, xy)]
    for _ in range(steps):
        s = step(s)
        A, xi = unpack_state(s)
        traj.append(from_flat(A) + from_flat(np.asarray(xi) / dt))
    return traj


def _random_data(seed):
    rng = np.random.default_rng(seed)
    return tuple(rng.standard_normal((N, N)) for _ in range(4))


# ---------------------------------------------------------------------------
# 1. The forcing emerges.
# ---------------------------------------------------------------------------


def test_forcing_emerges_from_wiring():
    """Prediction 1. DISCRETE ANALOGUE of the graph-Laplacian pattern: the closed composite's
    forcing at kappa = 1 is ``-d sum_f 1/2 B_f^2``, against (i) ``jax.grad`` of my own
    ``1/2 sum curl(A)^2`` and (ii) the script's hand-transposed ``gradU``; the face cost is
    the only formula, the sum over faces and the transposed copy come from ``compose_seq``."""
    arr, _ = _system(1.0)
    assert (arr.out_dim_M, arr.in_dim_M, arr.out_dim_N, arr.in_dim_N) == (0, 0, 0, 0)  # closed
    assert arr.P.Q_dim == 2 * N * N
    forcing = jax.jit(lambda q: arr.omega(q, _Z0, _Z0))

    def U_mine(q):  # 1/2 sum B^2 with jnp rolls, differentiated by autodiff
        Ax, Ay = q[:N * N].reshape(N, N), q[N * N:].reshape(N, N)
        B = Ax + jnp.roll(Ay, -1, 0) - jnp.roll(Ax, -1, 1) - Ay
        return 0.5 * jnp.sum(B ** 2)

    rng = np.random.default_rng(0)
    for _ in range(3):
        q = jnp.asarray(rng.standard_normal(2 * N * N))
        xi_q, xi_m, xi_n = forcing(q)
        assert xi_m.shape == (0,) and xi_n.shape == (0,)
        want_i = -jax.grad(U_mine)(q)
        want_ii = -to_flat(*gradU(*from_flat(q)))
        np.testing.assert_allclose(np.asarray(xi_q), np.asarray(want_i), atol=1e-10)
        np.testing.assert_allclose(np.asarray(xi_q), want_ii, atol=1e-10)
        assert np.abs(want_ii).max() > 1.0  # the forcing is not trivially zero


# ---------------------------------------------------------------------------
# 2. One Phi step is the script's step.
# ---------------------------------------------------------------------------


def test_phi_step_is_the_scripts_step():
    """Prediction 2. Two independent implementations agree (STATUS.md §2): one ``Phi`` step of
    the composite at ``kappa = 0.25`` equals one step of scripts/maxwell2d.py at dt = 0.5 on
    the same random ``(A, x)``, in ``A`` and in ``x``, to 1e-12."""
    dt = 0.5
    Ax, Ay, xx, xy = _random_data(1)
    want = script_step(Ax, Ay, xx, xy, dt)
    got = run_composite(dt, Ax, Ay, xx, xy, 1)[1]
    for g, w in zip(got, want):
        np.testing.assert_allclose(np.asarray(g), w, atol=1e-12)
    # and it moved: the step is not the identity
    assert np.abs(got[0] - Ax).max() > 0.1 and np.abs(got[2] - xx).max() > 0.1


# ---------------------------------------------------------------------------
# 3. Gauss's law.
# ---------------------------------------------------------------------------


def test_gauss_law_conserved():
    """Prediction 3. DISCRETE ANALOGUE of ``d/dt div E = 0``: along 60 steps at dt = 0.5 from
    random ``A`` and random momenta with nonzero vertex divergence (so this is not 0 = 0), the
    discrete divergence of the momenta at every vertex is unchanged to 1e-12, because the
    composite's forcing lies in the image of ``C^T`` and ``div C^T = 0``."""
    dt = 0.5
    Ax, Ay, xx, xy = _random_data(2)
    g0 = div(xx, xy)
    assert np.abs(g0).max() > 0.5  # a real charge distribution, not zero
    traj = run_composite(dt, Ax, Ay, xx, xy, 60)
    worst = max(np.abs(div(x, y) - g0).max() for (_, _, x, y) in traj)
    assert worst < 1e-12, worst
    # the field itself moved a lot meanwhile (the constraint is conserved, not the state)
    assert np.abs(traj[-1][2] - xx).max() > 0.5


# ---------------------------------------------------------------------------
# 4. Dispersion: short waves run slow.
# ---------------------------------------------------------------------------


def _modal_angle(dt, a, b, steps=40):
    """Run the eigenmode ``A = C^T cos(k . r)``, ``k = 2 pi (a, b)/N``, from rest; return
    ``(theta, Omega)``: the per-step rotation angle read from the three-term recurrence
    ``c_{n+1} + c_{n-1} = 2 cos(theta) c_n`` of the modal amplitude ``c_n = <A_n, A_0>/|A_0|^2``,
    and the closed-form lattice frequency. Also asserts the mode IS an eigenmode: every
    ``A_n`` is a multiple of ``A_0`` and ``c_n = cos((n - 1/2) theta)/cos(theta/2)`` to 1e-9."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    psi = np.cos(2 * np.pi * (a * i + b * j) / N)
    Ax, Ay = curl_T(psi)
    A0 = to_flat(Ax, Ay)
    traj = run_composite(dt, Ax, Ay, np.zeros((N, N)), np.zeros((N, N)), steps)
    c = np.array([to_flat(X, Y) @ A0 / (A0 @ A0) for (X, Y, _, _) in traj])
    for n, (X, Y, _, _) in enumerate(traj):  # stays on the line through A0
        np.testing.assert_allclose(to_flat(X, Y), c[n] * A0, atol=1e-9)
    cos_theta = (c[2] + c[0]) / (2 * c[1])
    theta = np.arccos(cos_theta)
    n = np.arange(len(c))
    np.testing.assert_allclose(c, np.cos((n - 0.5) * theta) / np.cos(theta / 2), atol=1e-9)
    Omega = np.sqrt(4 * np.sin(np.pi * a / N) ** 2 + 4 * np.sin(np.pi * b / N) ** 2)
    return theta, Omega


def test_dispersion_relation():
    """Prediction 4. CLOSED FORM ``omega^2 = 4 sin^2(kx/2) + 4 sin^2(ky/2)`` in the small-step
    limit: at dt = 0.05 the measured ``omega = theta/dt`` agrees to rtol 1e-3 at
    ``k = 2 pi (1,0)/8, (1,1)/8, (2,1)/8`` (the symplectic-Euler error is about
    ``(dt Omega)^2/24 <= 5e-4`` here, always ABOVE the closed form). And the DISCRETE ANALOGUE
    ``(2/dt)^2 sin^2(theta/2) = Omega^2`` holds to rtol 1e-9."""
    dt = 0.05
    for (a, b) in [(1, 0), (1, 1), (2, 1)]:
        theta, Omega = _modal_angle(dt, a, b)
        np.testing.assert_allclose((2 / dt) ** 2 * np.sin(theta / 2) ** 2, Omega ** 2, rtol=1e-9)
        omega = theta / dt
        np.testing.assert_allclose(omega, Omega, rtol=1e-3)
        assert omega > Omega  # the discrete step runs the mode slightly fast, as the relation says


def test_short_waves_run_slow():
    """Prediction 4, second half. CLOSED FORM: the measured phase speed ``omega/|k|`` falls from
    ``k = 2 pi/8`` (a = 1) through a = 2 to a = 4 (the shortest wave, k = pi); at k = pi it is
    ``2/pi``, well below the long-wave speed 1. On the lattice, not in vacuum (the notes file's
    domain objection)."""
    dt = 0.05
    speeds = []
    for a in (1, 2, 4):
        theta, Omega = _modal_angle(dt, a, 0)
        k = 2 * np.pi * a / N
        speeds.append(theta / dt / k)
    assert speeds[0] > speeds[1] > speeds[2], speeds
    assert speeds[0] > 0.97 and abs(speeds[2] - 2 / np.pi) < 2e-3, speeds


# ---------------------------------------------------------------------------
# 5. Energy at dt = 0.5.
# ---------------------------------------------------------------------------


def staggered_energy(s, s_next):
    """The leapfrog's exactly conserved discrete energy (Yee's staggering): ``1/2 sum x_n^2 +
    1/2 <B_n, B_{n+1}>``, with ``x_n = (A_{n+1} - A_n)/dt`` the half-step velocity."""
    Ax, Ay, xx, xy = s
    Ax2, Ay2, _, _ = s_next
    return 0.5 * (xx ** 2 + xy ** 2).sum() + 0.5 * (curl(Ax, Ay) * curl(Ax2, Ay2)).sum()


def test_energy_staggered_exact_naive_oscillates():
    """Prediction 5 as written -- the naive energy ``1/2 sum x^2 + 1/2 sum B^2`` within 3 percent
    at every step of a 400-step run at dt = 0.5 -- is REFUTED: it swings by 33 percent (the
    Gaussian ``A = C^T psi`` at rest) and 119 percent (random data), the same numbers in plain
    numpy, so it is the scheme and not the library. The mechanism: ``x_n`` is the half-step
    velocity while ``B_n`` is at the integer step, so the naive sum mixes time levels and
    oscillates with amplitude about ``Omega dt / 2`` per mode (24 percent for the longest mode
    alone; up to ``sqrt 2`` for the shortest), with no secular drift. What holds, and is checked:
    (i) the STAGGERED energy ``1/2 sum x_n^2 + 1/2 <B_n, B_{n+1}>`` is conserved to 1e-12
    (a DISCRETE ANALOGUE of energy conservation, exact for the linear leapfrog); (ii) the
    naive energy's deviation is bounded (below 1.5) and its mean over the last quarter of the run
    is within 25 percent of the mean deviation of the first quarter (no secular drift; ILLUSTRATES);
    (iii) at dt = 0.1 the naive deviation falls to about 5 percent (scaling as dt; ILLUSTRATES).
    The printed drifts are the observed values. The script's own "energy drift 2.0 percent" at
    N = 200 is the END-POINT sample of the same oscillation, whose maximum over its run is
    14.7 percent (plain numpy, this run's notes)."""
    i, j = np.meshgrid(np.arange(N), np.arange(N), indexing="ij")
    psi = np.exp(-((i - N / 2) ** 2 + (j - N / 2) ** 2) / (2 * 1.5 ** 2))
    Ax, Ay = curl_T(psi)
    cases = {"gaussian": (Ax, Ay, np.zeros((N, N)), np.zeros((N, N))), "random": _random_data(3)}
    drifts = {}
    for name, data in cases.items():
        for dt in (0.5, 0.1):
            traj = run_composite(dt, *data, 400)
            E = np.array([energy(*s) for s in traj])
            S = np.array([staggered_energy(traj[n], traj[n + 1]) for n in range(len(traj) - 1)])
            np.testing.assert_allclose(S, S[0], rtol=1e-12)                       # (i)
            dev = np.abs(E - E[0]) / E[0]
            drifts[(name, dt)] = dev.max()
            assert dev.max() < 1.5, (name, dt, dev.max())                          # (ii) bounded
            q = len(dev) // 4
            first, last = dev[:q].mean(), dev[-q:].mean()  # quarter MEANS: a beating of 64 modes
            assert abs(last - first) < 0.25 * dev.mean(), (name, dt, first, last)  # no secular trend
        assert drifts[(name, 0.1)] < 0.10 and drifts[(name, 0.1)] < 0.4 * drifts[(name, 0.5)]  # (iii)
    assert drifts[("gaussian", 0.5)] > 0.03  # the prediction as written fails, and is recorded
    print("naive energy max relative deviation over 400 steps:", drifts)
