"""A weak cue over a low ridge releases a large potential: the tilted double well.

**Question.** The book's Ch 1 (``sec.check_claims``) says of cues and words that "as pushes
they are very weak", yet a few words organize a lot of activity. The model cited for it is the
attention paper's capture example, ``tests/test_cue.py``, whose preference cost
``phi_lambda(g) = (lambda/4) g^2 (1 - g)^2`` has two *equal* wells. There the energy goes the
other way: the paper's cue does work ``W = 11.6`` on the context and the composite releases
``0.38`` after it ends (the domain check of 2026-09-29, Group E2 entry 3). With equal wells it
cannot go the other way: the composite starts and ends at ``V = 0``, so what it releases after
the cue is at most what the cue put in. Is a cue that is weak *in energy*, next to what follows
it, in the model at all?

**Model.** The composite of ``test_cue.py`` unchanged -- the rendezvous parent with the context
coordinate ``g``, two subsidiaries, composed by ``compose_seq`` in the library, cost the sum of
the constituents' -- except the preference cost, which is tilted by an exact change:

    phi'(g) = A g (g - c)(g - 1),   phi(g) = A (g^4/4 - (1 + c) g^3/3 + c g^2/2).

The wells stay at ``g = 0`` (meet) and ``g = 1`` (keep the separation, the resting state) and the
ridge is at ``c``; ``phi(1) - phi(0) = A (2c - 1)/12 = D`` and
``phi(c) - phi(1) = A (1 - c)^3 (1 + c)/12 = eps``, so ``(A, c)`` is solved from ``(D, eps)``. The
resting state sits in a *shallow* well, ``eps`` below a ridge, beside a well ``D`` deeper. Since
the coupling ``K`` can be zero at every ``g``, the lowest ridge of the whole ``V`` between the two
rests is exactly ``eps`` above the resting state. The cue is the paper's shape,
``u(t) = a b_0(t/T)``; its only effect on the state is the force ``+u`` on ``g`` (checked
below), so its work is ``W = sum_k u_k (g_{k+1} - g_k)``, and what the composite releases after
the cue is ``V(T) - V(end)``. Step ``h = 0.001`` (the numbers agree to four digits at ``h = 0.002``
and ``0.0005``); all five mobilities equal ``h``, as in ``test_cue.py``.

**Prediction** (written before any run, `notes/ch1-tests-2026-09-29/tilted-cue.md` in the book
repo): the cue's work scales with the barrier ``eps``, the release with ``D``, so
``released / W`` grows like ``D / eps``; the cue only lifts the state over the ridge, and the fall
is paid by the composite's own potential.

**Result.** For the *minimal* cue -- the amplitude bisected to where capture just happens -- of
duration ``T = 0.2``, the prediction holds in direction, with two corrections. The release after
the cue is ``D + eps`` plus a small coupling excess (``4.27, 16.26, 64.25`` for ``D = 4, 16, 64`` at
``eps = 0.25``); ``W`` is between ``1.36 eps`` and ``3.0 eps`` and never below ``eps``; the ratio is
``5.7, 35, 188``. (i) At fixed ``D = 16``, ``W`` grows like ``eps^0.8`` rather than ``eps``
(``0.160, 0.464, 1.445`` for ``eps = 1/16, 1/4, 1``): a brief push dissipates, and the ridge's
distance from the resting state shrinks with ``eps``. (ii) At fixed ``eps``, ``W`` *falls* as
``D`` grows (``0.744, 0.464, 0.341``), since the ridge moves nearer the current well; so the ratio
grows faster than ``D / eps``. The briefer the cue, the more work it takes: at ``D = 16``,
``eps = 0.25`` the work done by the time ``g`` crosses the ridge is ``1.11, 0.46, 0.34, 0.30`` for
``T = 0.05, 0.2, 0.5, 1``, falling toward ``eps = 0.25``, the energy bound. The control: the
paper's untilted example gives ``W = 11.63``, released ``0.385``; its minimal ``T = 0.2`` cue
gives ``W = 2.88``, released ``1.09``: released below ``W``, as it must be.

**But that ratio is a knife-edge** (found after the prediction, when a first version of the
last test failed). The release *after* the cue is large only when the cue ends before the state
falls, and for the paper's fixed bump that holds within about ``1%`` of the minimal amplitude: at
``D = 64``, ``T = 0.2``, ``1.01`` times the minimal cue crosses the ridge while still on, keeps
pushing along the steep fall, does ``W = 1.30`` and leaves ``0.44`` after it. What is robust:
the cue's work up to the ridge crossing, between ``eps`` and ``5 eps`` for every cue tried
(``T >= 0.05``, up to ``1.5`` times minimal); the composite's own net release
``V(0) - V(end) = D`` for every capturing cue; and so the ratio of that release to *all* the
cue's work, ``D / W = 1.1, 3.5, 11`` for ``D = 4, 16, 64`` at ``1.5`` times minimal, which grows
with ``D``, sub-linearly, and exceeds ``1`` only once ``D`` is well past the few units a
fixed-shape cue spends riding the fall.

**What it does not show.** (1) A cue that stopped at the crossing would do only the work to the
ridge; the paper's bump cannot stop there, and nothing here models a cue that responds to where
the reader is. (2) The tilt, like the paper's wells, is supplied by hand: nothing here says how a
reader comes to be poised, one ``eps`` from a well ``D`` deeper. (3) Released potential is
dissipated by the composite's own descent; it is energy, not "activity" in the book's wider
sense, and nothing here measures how much the parts do beyond the positions moving. (4) The
discrete run is checked, not proved: every number here is of the explicit Euler map at the
stated step.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
from scipy.optimize import brentq

from arrangements.attention import attention_arrangement, identity_transfers, make_input, message
from arrangements.coalgebra import configuration
from arrangements.rvect import diagonal
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state
from tests.test_cue import REST, bump, phi_lam
from tests.test_rendezvous import DH, DP, MDIM, exterior, subsidiary

H = 0.001                  # the step, which is every mobility (test_cue.py's convention)
POST = 30.0                # autonomous time after the cue; V's slowest mode decays like e^{-1.17 t}


# ---------------------------------------------------------------------------
# The tilted preference cost and the composite.
# ---------------------------------------------------------------------------


def tilt_shape(D, eps):
    """``(A, c)`` with ``phi'(g) = A g (g - c)(g - 1)``, ``phi(1) - phi(0) = D`` and
    ``phi(c) - phi(1) = eps``. Dividing the two conditions gives
    ``(1 - c)^3 (1 + c)/(2c - 1) = eps/D``, whose left side falls from infinity to ``0`` on
    ``(1/2, 1)``, so it has one root; then ``A = 12 D/(2c - 1)``."""
    c = brentq(lambda c: (1 - c) ** 3 * (1 + c) / (2 * c - 1) - eps / D,
               0.5 + 1e-12, 1.0 - 1e-15, xtol=1e-15)
    return 12.0 * D / (2.0 * c - 1.0), c


def phi_tilt(g, A, c):
    """The tilted double well: wells at ``g = 0`` (depth ``0``) and ``g = 1`` (height ``D``),
    ridge at ``g = c``."""
    return A * (g ** 4 / 4.0 - (1.0 + c) * g ** 3 / 3.0 + c * g ** 2 / 2.0)


def tilted(D, eps):
    """The preference cost ``phi`` for ``(D, eps)``, with its ridge ``c``."""
    A, c = tilt_shape(D, eps)
    return (lambda g: phi_tilt(g, A, c)), c


def parent_cost(phi):
    """eq:cue-parent-cost with ``phi_lambda`` replaced by ``phi``:
    ``(1/2)(g - (h_2 - h_1))^2 + phi(g) + (1/2) u^2 - u g + 2``."""
    def cost(q, m_out, n_in):
        g, u = q[2], n_in[0]
        r = g - (m_out[MDIM] - m_out[0])
        return 0.5 * r ** 2 + phi(g) + 0.5 * u ** 2 - u * g + 2.0
    return cost


def tilted_system(phi, h=H, state=REST):
    """``a_phi(S_1, S_2) : I -> B`` by composition in ``arr``, exactly ``test_cue.cue_system``
    with the parent's cost changed; configuration ``(z_1, z_2, x_1, x_2, g)``."""
    g0, z1, z2, x1, x2 = state
    parts = [subsidiary(z1, h, "S1"), subsidiary(z2, h, "S2")]
    parent = attention_arrangement(
        2, DH, DP,
        placement=lambda q: jnp.reshape(q[:2], (2, DP)),
        transfers=identity_transfers(2, DH),
        cost=parent_cost(phi),
        parameter=configuration(diagonal(jnp.asarray([h, h, h], float)),
                                init=jnp.asarray([x1, x2, g0], float)),
        label="a_phi",
    )
    return compose_seq(tensor_arrangements(parts), parent.arr())


def V_of(phi):
    """``V = K + phi`` on ``(z_1, z_2, x_1, x_2, g)`` (eq:autonomous-cost with ``phi``)."""
    def V(y):
        z1, z2, x1, x2, g = y[0], y[1], y[2], y[3], y[4]
        r = g - (z2 - z1)
        return 0.5 * r ** 2 + 0.5 * (x1 - z1) ** 2 + 0.5 * (x2 - z2) ** 2 + phi(g)
    return V


def scan_closed(arr, us):
    """``test_rendezvous.run_closed`` as one ``lax.scan``: the closed composite's step
    ``s |-> P.step(s, omega(q(s), (), e)|_Q)`` driven by the exterior message ``(u_k, 1)`` at
    reference position ``0``. ``run_closed`` is checked there to be ``Phi``'s step."""
    P, omega, empty = arr.P, arr.omega, jnp.zeros(0)

    def step(s, u):
        xi_q, _, _ = omega(P.position(s), empty, make_input(message(u, 1.0), 0.0))
        s2 = P.step(s, xi_q)
        return s2, flat_state(s2)

    _, ys = jax.lax.scan(step, P.init, us)
    return np.asarray(jnp.concatenate([flat_state(P.init)[None], ys]))


def run(phi, amp, T, h=H, post=POST, ridge=None):
    """The run under ``u(t) = amp b_0(t/T)``, then autonomously to ``T + post``. Returns the
    cue's work ``W``, ``V`` at the start, at the cue's end and at the end, the peak of ``V``
    while the cue is on, the final ``g``, and, if ``ridge`` is given, the cue's work up to the
    step where ``g`` first falls below it."""
    t = np.arange(int(round((T + post) / h))) * h
    us = np.asarray(amp * bump(t / T))
    ys = scan_closed(tilted_system(phi, h), jnp.asarray(us))
    V = np.asarray(jax.vmap(V_of(phi))(jnp.asarray(ys)))
    g = ys[:, 4]
    k = int(round(T / h))
    out = dict(W=float(np.sum(us * np.diff(g))), V0=float(V[0]), VT=float(V[k]),
               Vend=float(V[-1]), Vpeak=float(V[:k + 1].max()), gend=float(g[-1]))
    out["released"] = out["VT"] - out["Vend"]
    if ridge is not None:
        kc = int(np.argmax(g < ridge))
        out["W_cross"] = float(np.sum(us[:kc] * np.diff(g)[:kc]))
    return out


def minimal_amplitude(phi, T, iters=16):
    """The cue amplitude at which capture (``g -> 0``) just happens, by bisection; returns an
    amplitude that captures, within ``2^-iters`` of the bracket of the threshold."""
    captures = lambda a: run(phi, a, T)["gend"] < 0.5
    hi = -1.0
    while not captures(hi):
        hi *= 2.0
    lo = hi / 2.0 if hi < -1.0 else 0.0
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if captures(mid):
            hi = mid
        else:
            lo = mid
    assert not captures(lo)
    return hi


@functools.lru_cache(maxsize=None)
def threshold(D, eps, T):
    """``minimal_amplitude`` for the tilted well ``(D, eps)``, cached across tests."""
    phi, _ = tilted(D, eps)
    return minimal_amplitude(phi, T)


def minimal_cue(D, eps, T, factor=1.0):
    """The run under ``factor`` times the minimal capturing cue of duration ``T``, with the
    work up to the ridge crossing."""
    phi, c = tilted(D, eps)
    return run(phi, factor * threshold(D, eps, T), T, ridge=c)


# ---------------------------------------------------------------------------
# The cost.
# ---------------------------------------------------------------------------


def test_the_tilted_well_has_the_stated_depths_and_the_cost_is_admissible():
    """CLOSED FORM, checked: the wells stay at ``0`` and ``1``, the ridge is at ``c``, the
    current well is ``D`` above the deep one and ``eps`` below the ridge; ``phi >= 0``; and the
    parent cost is nonnegative, as def:arrangement requires (sampled, over the sweep's
    ``(D, eps)``)."""
    dphi = jax.grad(phi_tilt)
    r = np.random.default_rng(801)
    for D, eps in [(4, 0.25), (16, 0.25), (64, 0.25), (16, 0.0625), (16, 1.0)]:
        A, c = tilt_shape(D, eps)
        assert 0.5 < c < 1.0
        np.testing.assert_allclose(float(phi_tilt(0.0, A, c)), 0.0, atol=1e-15)
        np.testing.assert_allclose(float(phi_tilt(1.0, A, c)), D, rtol=1e-12)
        np.testing.assert_allclose(float(phi_tilt(c, A, c) - phi_tilt(1.0, A, c)), eps,
                                   rtol=1e-9)
        for g in (0.0, c, 1.0):
            np.testing.assert_allclose(float(dphi(g, A, c)), 0.0, atol=1e-9)
        gs = np.linspace(-5.0, 5.0, 2001)
        assert np.all(np.asarray(phi_tilt(gs, A, c)) >= -1e-12)
        cost = parent_cost(lambda g: phi_tilt(g, A, c))
        for _ in range(100):
            q = jnp.asarray(r.standard_normal(3) * 3.0)
            m = jnp.asarray(r.standard_normal(2 * MDIM) * 3.0)
            e = jnp.asarray(r.standard_normal(3) * 5.0)
            assert float(cost(q, m, e)) >= 0.0


def test_the_cue_acts_only_as_the_force_u_on_g():
    """CLOSED FORM, checked: the composite's configuration forcing is ``-dV + u dg``, so the
    cue's power is ``u gdot`` and ``W`` above is its work. The composite is never handed
    ``V``."""
    phi, _ = tilted(16.0, 0.25)
    b = tilted_system(phi, 1.0)
    gradV = jax.grad(V_of(phi))
    r = np.random.default_rng(802)
    for u in (0.0, -2.3):
        for _ in range(4):
            q = jnp.asarray(r.standard_normal(5))
            xi_q, _, _ = b.omega(q, jnp.zeros(0), exterior(u))
            expected = -np.asarray(gradV(q))
            expected[4] += u
            np.testing.assert_allclose(np.asarray(xi_q), expected, atol=1e-10)


def test_the_shallow_well_is_held_without_a_cue():
    """The resting state ``(g, z, x) = (1, 0, 1, 0, 1)`` is a critical point of the tilted
    ``V`` too, at ``V = D``: with no cue nothing moves, so the composite is poised, not
    falling."""
    phi, _ = tilted(16.0, 0.25)
    out = run(phi, 0.0, 0.2, post=5.0)
    np.testing.assert_allclose(out["V0"], 16.0, rtol=1e-12)
    np.testing.assert_allclose(out["Vend"], 16.0, rtol=1e-12)
    np.testing.assert_allclose(out["gend"], 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# The control: equal wells.
# ---------------------------------------------------------------------------


def test_control_equal_wells_release_less_than_the_cue_did():
    """The untilted paper example reproduces the domain check's numbers: the paper's cue does
    ``W = 11.63`` and the composite releases ``0.385`` after it (``test_cue.py`` asserts the
    capture condition; ``0.3848`` is its ``V(0.2)`` at this step). Its minimal ``T = 0.2`` cue
    does ``W = 2.88`` and leaves ``1.09`` to release, just over the ridge ``lambda/64 = 1``.
    Released is below ``W`` in both, as it must be when the composite starts and ends at
    ``V = 0``."""
    paper = run(phi_lam, -12.0, 0.2)
    assert 11.0 < paper["W"] < 12.5
    np.testing.assert_allclose(paper["released"], 0.3848, atol=2e-3)
    assert paper["W"] > 20.0 * paper["released"]

    least = run(phi_lam, minimal_amplitude(phi_lam, 0.2), 0.2)
    assert least["gend"] < 1e-9
    assert 1.0 < least["released"] < 1.15          # just over the ridge
    assert 2.5 < least["W"] < 3.2
    assert least["released"] < 0.5 * least["W"]


# ---------------------------------------------------------------------------
# The tilted well: the minimal brief cue.
# ---------------------------------------------------------------------------


def _check_minimal_brief(D, eps, out):
    """Shared bounds for a minimal ``T = 0.2`` cue on ``(D, eps)``. The state is lifted just
    over the ridge (peak ``V - V(0)`` between ``eps`` and ``1.15 eps``); it is still near the
    ridge when the cue ends, so the release after it is ``D + eps`` plus the coupling's excess
    (at most ``0.12 eps`` observed, bound ``0.2 eps``); and ``W`` is at least ``eps`` (energy:
    the state must climb ``eps`` and dissipation is nonnegative) and at most ``3.5 eps``
    (``3.0 eps`` observed, at ``D = 4``)."""
    assert out["gend"] < 1e-9
    np.testing.assert_allclose(out["Vend"], 0.0, atol=1e-9)
    assert eps < out["Vpeak"] - out["V0"] < 1.15 * eps
    assert D + eps < out["released"] < D + 1.2 * eps
    assert eps < out["W"] < 3.5 * eps


def test_minimal_brief_cue_sweep_in_D():
    """At ``eps = 0.25``, ``D = 4, 16, 64``: ``W = 0.744, 0.464, 0.341``, released
    ``4.27, 16.26, 64.25``, ratio ``5.7, 35, 188``. The prediction said the ratio grows like
    ``D``; it grows faster, because ``W`` falls with ``D``: the ridge
    ``1 - c ~ (eps/2D)^(1/3)`` moves nearer the current well, so the brief push goes less far.
    Asserted: the shared bounds, ``W`` decreasing, and the ratio above ``D/(3.5 eps)`` and
    increasing by more than the factor in ``D``."""
    eps, Ds = 0.25, (4.0, 16.0, 64.0)
    outs = []
    for D in Ds:
        out = minimal_cue(D, eps, 0.2)
        _check_minimal_brief(D, eps, out)
        outs.append(out)
    Ws = [o["W"] for o in outs]
    ratios = [o["released"] / o["W"] for o in outs]
    assert Ws[0] > Ws[1] > Ws[2]
    assert all(rt > D / (3.5 * eps) for rt, D in zip(ratios, Ds))
    assert ratios[1] > 4.0 * ratios[0] and ratios[2] > 4.0 * ratios[1]


def test_minimal_brief_cue_sweep_in_eps():
    """At ``D = 16``, ``eps = 1/16, 1/4, 1``: ``W = 0.160, 0.464, 1.445``, released
    ``16.07, 16.26, 17.02``, ratio ``100, 35, 11.8``. ``W`` grows like ``eps^0.77`` then
    ``eps^0.82``, sub-linearly: a brief push dissipates more than the climb, and the more so the
    nearer the ridge. Asserted: the shared bounds, the ratio decreasing in ``eps``, and each
    local exponent ``log(W'/W)/log 4`` between ``0.7`` and ``0.9`` (four-digit agreement across
    step sizes; the band is the two observed values widened by about ``0.05``)."""
    D, epss = 16.0, (0.0625, 0.25, 1.0)
    outs = []
    for eps in epss:
        out = minimal_cue(D, eps, 0.2)
        _check_minimal_brief(D, eps, out)
        outs.append(out)
    Ws = [o["W"] for o in outs]
    ratios = [o["released"] / o["W"] for o in outs]
    assert ratios[0] > ratios[1] > ratios[2]
    for w0, w1 in zip(Ws, Ws[1:]):
        assert 0.7 < np.log(w1 / w0) / np.log(4.0) < 0.9


# ---------------------------------------------------------------------------
# Briefer cues cost more; slower ones approach the barrier, and outlast the fall.
# ---------------------------------------------------------------------------
# What is robust, and the knife-edge the ratio above stands on.
# ---------------------------------------------------------------------------


def test_minimal_work_to_the_ridge_falls_toward_eps_as_the_cue_slows():
    """At ``D = 16``, ``eps = 0.25``, the minimal cue's work up to the step where ``g`` crosses
    the ridge is ``1.11, 0.46, 0.34, 0.30`` for ``T = 0.05, 0.2, 0.5, 1``: always above ``eps``
    (the energy bound), falling toward it as the cue slows. A brief cue must be stronger and
    costs more (the domain check's "the cue must be *stronger* the briefer it is"). The
    composite's net release ``V(0) - V(end) = D`` is the same for every capturing cue."""
    D, eps = 16.0, 0.25
    outs = [minimal_cue(D, eps, T) for T in (0.05, 0.2, 0.5, 1.0)]
    Wc = [o["W_cross"] for o in outs]
    assert all(w > eps for w in Wc)
    assert Wc[0] > Wc[1] > Wc[2] > Wc[3]
    assert Wc[0] > 4.0 * eps and Wc[3] < 1.3 * eps
    for o in outs:
        np.testing.assert_allclose(o["V0"] - o["Vend"], D, atol=1e-9)


def test_the_ratio_is_a_knife_edge_and_what_survives_a_stronger_cue():
    """The release *after* the cue exceeds its work only when the cue ends before the state
    falls, which for the paper's fixed bump holds within about ``1%`` of the minimal amplitude.
    At ``D = 64``, ``T = 0.2``: the minimal cue does ``W = 0.34`` and ``64.25`` is released after
    it; ``1.01`` times it crosses the ridge while still on, pushes along the steep fall, and does
    ``W = 1.30`` with ``0.44`` left after it. At ``D = 16``, ``T = 1`` the same flip is
    ``0.30 / 15.39`` to ``1.86 / 0.20``.

    What survives ``1.5`` times the minimal cue (``T = 0.2``, ``eps = 0.25``,
    ``D = 4, 16, 64``): the work up to the ridge crossing stays within ``(eps, 5 eps)``
    (``1.15, 0.73, 0.56`` observed); total ``W = 3.54, 4.58, 5.76``, most of it spent riding
    the fall; and the composite's own net release ``D`` against all the cue's work,
    ``D / W = 1.1, 3.5, 11``, still grows with ``D`` (sub-linearly), above ``1`` once ``D`` is
    well past the few units a fixed-shape cue spends along the fall."""
    eps = 0.25
    edge = minimal_cue(64.0, eps, 0.2)
    over = minimal_cue(64.0, eps, 0.2, factor=1.01)
    assert edge["released"] > 100.0 * edge["W"]
    assert over["released"] < over["W"] and over["W"] > 3.0 * edge["W"]
    slow_edge = minimal_cue(16.0, eps, 1.0)
    slow_over = minimal_cue(16.0, eps, 1.0, factor=1.01)
    assert slow_edge["released"] > 30.0 * slow_edge["W"]
    assert slow_over["released"] < slow_over["W"] and slow_over["W"] > 4.0 * slow_edge["W"]

    Ds = (4.0, 16.0, 64.0)
    strong = [minimal_cue(D, eps, 0.2, factor=1.5) for D in Ds]
    for D, o in zip(Ds, strong):
        assert o["gend"] < 1e-9
        assert eps < o["W_cross"] < 5.0 * eps
        assert 3.0 < o["W"] < 6.5
        assert o["released"] < o["W"]                  # the fall happens under the cue
        np.testing.assert_allclose(o["V0"] - o["Vend"], D, atol=1e-9)
    amp = [D / o["W"] for D, o in zip(Ds, strong)]
    assert amp[0] < amp[1] < amp[2]
    assert amp[0] > 1.0 and amp[2] > 10.0
