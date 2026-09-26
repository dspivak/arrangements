"""The fire: a collective that reports what neither of its parts reports.

David, 2026-09-26, of the scalar-transfer suboperad (`notes/TESTS-SYNTHESIS-2026-09-26.md`
§3.6, the statement that a parent whose transfers are scalars sends only a weighted average of
what its children send): "can we make a test that handles the fire? ... i feel like this is back
to the insurance agency". The fire half is here; the holding-and-charging half is
`tests/test_insurance.py`.

The scene is folk physics. A fire is at ``X``; the wind blows at ``U`` positions per step; a
fire goes downwind, so in ``DT`` steps it will be at ``Y = X + DT U``. One scout sees the smoke
and reports where the fire *is*; another feels the wind and reports how hard it blows. Neither
reports where the fire will be. The parent must, and it is charged the square of how far its
report falls from ``Y``.

**The three predictions, fixed here before any construction code existed, and each able to
fail.**

    (1) With the value maps at the identity -- the scalar-transfer case, where every book run so
        far lives -- the parent's report is a convex combination of its scouts' reports, so it
        stays in ``[U, X]`` at every step whatever its placements do, and its distance from the
        truth cannot fall below ``Y - X``. Descent should put the attention on the smoke scout
        and push the wind scout away, approaching that floor from above without reaching it
        (reach is a rate, `notes/TURNOVER-TEST.md`).
        **Able to fail:** if the distance falls below ``Y - X``, the hull statement is false as
        stated.

    (2) With the value maps free and the placements frozen, the cost is quadratic in the two
        maps with Hessian ``a a^T``, ``a = (X, U)/2``; so the residual falls geometrically by the
        exact factor ``|1 - mu_V |a|^2|`` a step, the report leaves ``[U, X]``, and the settled
        pair is the *nearest point of the solution line to where the parent started* -- not the
        rule. In particular the ratio of the two maps is not the elapsed time ``DT``.
        **Able to fail:** if the settled ratio is ``DT``, one scene did pin the rule down and the
        note's design ("the wind's map learns the elapsed time") was right as written.

    (3) With four scenes sharing one pair of value maps, charged on three of them whose vectors
        ``(x_i)`` and ``(u_i)`` are linearly independent, the quadratic is positive definite, so
        descent settles at the unique solution: the map ratio *is* ``DT``, and the report on the
        fourth scene, which the parent is never charged for, is that scene's truth. Charged on
        one scene only, the same parent settles elsewhere on that scene's solution line and is
        wrong on the others.
        **Able to fail:** the ill-conditioned Hessian could stall the run short of the solution,
        or the one-scene fit could happen to be right on a scene it never saw.

**What this does not model, and says so rather than hiding it.** (a) *No time.* Where the fire
turns out to be enters the parent's cost as a constant target, exactly as the collective's
target ``nu`` does in `tests/test_turnover.py`; nothing here holds a report and is charged for it
later. That is the insurance construction, and David's link between the two is why it is a
separate file. (b) *The sharing is stipulated.* In the spatial family each child has its own
value map (``eq:spatial-state``), so "one rule for every scout of a kind" is a constraint imposed
on the transfers in part 4, not something the run discovers; what the run can show is whether
descent finds the rule, how many scenes pin it down, and what the parent says about a scene it
was not charged for (STATUS.md §2, "not evidence: what the construction was built to have").
(c) The scouts are frozen and cost-free: nothing here is about how a scout comes to report what
it reports.

Every docstring below says which of three things its test does (STATUS.md §3.7): reproduce a
closed form derived in the docstring, check a discrete analogue, or correct a claim made on
paper.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.attention import (
    Transfer, attention_arrangement, input_dim, make_input, mean, message, msg_dim, signal,
    spatial_arrangement, spatial_config, spatial_kappa, spatial_split, weight, zero_message,
    zero_transfer,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state
from tests.test_rendezvous import run_closed, run_phi

DH, DP = 1, 1                                   # H = W = P = R, for one scene
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)     # dim M = 2, dim E = 3
EMPTY = jnp.zeros(0)

X, U, DT = 3.0, 0.5, 4.0     # the fire's position, the wind, and the steps until it is checked
Y = X + DT * U               # where the fire will be: 5.0
TAU = 0.8                    # the parent's kernel scale
POSITIONS = (1.0, 1.0)       # where it holds the two scouts, to start
BUDGET = 20000               # the step budget every number below is read at
REGION = 12.0                # the placement region part 2's Hessian bound is taken over
MU_V = 0.2                   # the mobility of a value map in part 3

SCENES = ((3.0, 0.5), (1.0, -0.75), (-2.0, 0.25), (4.0, -1.0))   # part 4; the last is never charged
CHARGED = (0, 1, 2)
K = len(SCENES)


def truth(scene):
    """Where a fire at ``x`` in a wind ``u`` will be in ``DT`` steps."""
    x, u = scene
    return x + DT * u


def exterior(dim_H=DH):
    """The exterior input ``(0_M, 0)``, which nothing here reads."""
    return make_input(zero_message(dim_H), 0.0)


# ---------------------------------------------------------------------------
# One scene: two scouts and a spatial parent (parts 1-3).
# ---------------------------------------------------------------------------


def scout(value, label, dim_H=DH, slot=0):
    """A component ``I -> B`` (the idiom of `tests/test_member.py`, `tests/test_turnover.py`):
    its configuration is the number it reports, frozen at mobility ``0``, and it sends the
    message ``(value e_slot, 1)``, whose mean readout is ``value e_slot``. Its own forcing is
    zero -- a scout here has nothing to say about what it is asked -- so every forcing in this
    file is the parent's."""
    e = jnp.eye(dim_H)[slot]
    return Arrangement(
        P=configuration(euclidean(1, 0.0), init=jnp.array([float(value)])),
        in_dim_M=0, out_dim_M=0,
        in_dim_N=input_dim(dim_H, DP), out_dim_N=msg_dim(dim_H),
        out_f=lambda q, m_out: message(q[0] * e, 1.0),
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label=label,
    )


def report_of(q, m_out, tau=TAU):
    """The parent's report: the mean readout of the message it sends onward,

        r = sum_j kappa_j V_j h_j / sum_j kappa_j w_j,   kappa_j = exp(-||p_j||^2 / 2 tau),

    written out as `tests/test_monitor_pulling.py`'s ``outgoing`` writes ``sigma_a``, since a
    whole run's worth of configurations has to be read at once; test 1 checks it against
    ``mean(a.sigma(q, m))``. The report is the *mean*, not the weighted signal: a collective
    that doubles every weight says the same thing twice as loudly, not something else."""
    pos, V = spatial_split(q, 2, DH, DP)
    h, w = jnp.zeros(DH), jnp.asarray(0.0)
    for j in range(2):
        k = spatial_kappa(pos[j], jnp.zeros(DP), tau)
        m_j = m_out[j * MDIM:(j + 1) * MDIM]
        h = h + k * (V[j] @ signal(m_j))
        w = w + k * weight(m_j)
    return h / w


def fire_parent(mu_p, mu_V, positions=POSITIONS, values=(1.0, 1.0), target=Y):
    """A spatial-attention parent over two scouts, its cost ``(1/2)|r - target|^2`` on the report
    it sends onward. ``mu_p`` is the mobility of each placement and ``mu_V`` that of each value
    map, so parts 2 and 3 differ in one pair of numbers and nothing else."""
    cost = lambda q, m_out, n_in: 0.5 * jnp.sum((report_of(q, m_out) - target) ** 2)
    return spatial_arrangement(
        2, DH, DP, TAU, cost=cost,
        parameter=configuration(constant(jnp.diag(jnp.array([mu_p, mu_p, mu_V, mu_V]))),
                                init=spatial_config(positions, values)),
        label="fire")


def fire_system(mu_p, mu_V, scene=(X, U), **kw):
    """``a(smoke, wind) : I -> B`` by composition in ``arr``. The composite's configuration is
    ``(x, u, p_smoke, p_wind, V_smoke, V_wind)``: the children's blocks first."""
    a = fire_parent(mu_p, mu_V, **kw)
    kids = [scout(scene[0], "smoke"), scout(scene[1], "wind")]
    return kids, a, compose_seq(tensor_arrangements(kids), a.arr())


# ---------------------------------------------------------------------------
# Several scenes at once, sharing one pair of value maps (part 4).
# ---------------------------------------------------------------------------


def shared_parent(charged, mu, init=(1.0, 1.0), scenes=SCENES):
    """A ``2K``-ary attention arrangement over the scenes' scouts with *one* pair of value maps
    shared by every scout of a kind: its configuration is ``q = (v_smoke, v_wind)`` and its
    transfers are ``T_{0j} = (1, v_{kind(j)} H)``, weight multiplier one and the shared signal
    map. Scene ``i`` lives in coordinate ``i`` of ``H = R^K``, its two scouts sending in that
    coordinate alone, so the parent's report is the vector whose ``i``th entry is scene ``i``'s
    report ``(v_smoke x_i + v_wind u_i) / 2K``.

    The cost is ``(1/2) sum_{i in charged} (r_i - y_i)^2``: the parent is charged for those
    scenes and for no others. This leaves the spatial family -- there each child has its own
    value map -- and the sharing is the stipulation the module docstring names."""
    n, dim_H = 2 * len(scenes), len(scenes)
    md, Z = msg_dim(dim_H), zero_transfer(dim_H)
    ys = jnp.asarray([truth(s) for s in scenes])
    mask = jnp.asarray([1.0 if i in charged else 0.0 for i in range(len(scenes))])

    def transfers(q):
        row0 = [Z] + [Transfer(gamma=jnp.asarray(1.0), L=q[j % 2] * jnp.eye(dim_H))
                      for j in range(n)]
        return tuple([tuple(row0)] + [tuple(Z for _ in range(n + 1)) for _ in range(n)])

    def report(q, m_out):
        h, w = jnp.zeros(dim_H), jnp.asarray(0.0)
        for j in range(n):
            m_j = m_out[j * md:(j + 1) * md]
            h = h + q[j % 2] * signal(m_j)
            w = w + weight(m_j)
        return h / w

    def cost(q, m_out, n_in):
        return 0.5 * jnp.sum(mask * (report(q, m_out) - ys) ** 2)

    a = attention_arrangement(
        n, dim_H, DP,
        placement=lambda q: jnp.zeros((n, DP)),
        transfers=transfers, cost=cost,
        parameter=configuration(euclidean(2, mu), init=jnp.asarray(init, float)),
        label="shared")
    return a, report


def shared_system(charged, mu, init=(1.0, 1.0), scenes=SCENES):
    """``a(smoke_1, wind_1, ...) : I -> B``; the composite's configuration is the ``2K`` frozen
    scout values, then ``(v_smoke, v_wind)``."""
    a, report = shared_parent(charged, mu, init, scenes)
    kids = []
    for i, (x, u) in enumerate(scenes):
        kids.append(scout(x, f"smoke{i}", len(scenes), i))
        kids.append(scout(u, f"wind{i}", len(scenes), i))
    return kids, a, report, compose_seq(tensor_arrangements(kids), a.arr())


def messages(kids, values):
    """The children's messages at the configuration that gives child ``i`` the value
    ``values[i]`` (their blocks come first in the composite, one coordinate each)."""
    values = jnp.asarray(values, float)
    return jnp.concatenate([k.out_f(values[i:i + 1], EMPTY) for i, k in enumerate(kids)])


SCENE_VALUES = tuple(v for s in SCENES for v in s)


# ---------------------------------------------------------------------------
# Running.
# ---------------------------------------------------------------------------


def traj(b, steps, ext=None):
    """The whole trajectory of the composite on a constant exterior input, inside a
    ``lax.scan`` (test 0 checks it against ``Phi``). Returns ``steps`` rows."""
    P, omega = b.P, b.omega
    e = exterior() if ext is None else ext

    @jax.jit
    def step(s, _):
        xi_q, _, _ = omega(P.position(s), EMPTY, e)
        s2 = P.step(s, xi_q)
        return s2, flat_state(s2)

    return np.asarray(jax.lax.scan(step, P.init, None, length=steps)[1])


def forcing(b, q, ext=None):
    """The composite forcing's configuration component at ``q``."""
    e = exterior() if ext is None else ext
    xi_q, _, _ = b.omega(jnp.asarray(q, float), EMPTY, e)
    return np.asarray(xi_q)


def spatial_report(rows):
    """``(kappa_smoke, kappa_wind, r)`` along a trajectory of the one-scene composite, in numpy:
    licensed by test 1, and used instead of the library call so that a whole run's worth of
    configurations can be read at once."""
    p, V = rows[:, 2:4], rows[:, 4:6]
    kap = np.exp(-p ** 2 / (2.0 * TAU))
    r = (kap[:, 0] * V[:, 0] * X + kap[:, 1] * V[:, 1] * U) / kap.sum(axis=1)
    return kap, r


# ---------------------------------------------------------------------------
# Test 0: the runs are Phi's.
# ---------------------------------------------------------------------------


def test_0_the_runs_are_the_dynamics_functors():
    """LICENSES the run form used below; it establishes nothing about the predictions. For the
    one-scene composite with its placements free and with its value maps free, and for the
    four-scene one, ``Phi``, ``run_closed`` and the ``lax.scan`` of ``traj`` agree over fifty
    steps to within one unit in the last place."""
    _, _, one = fire_system(0.3, 0.0)
    _, _, two = fire_system(0.0, MU_V)
    _, _, _, three = shared_system(CHARGED, 0.5)
    for b, e in ((one, exterior()), (two, exterior()), (three, exterior(K))):
        by_phi = run_phi(b, 50, e)
        by_closed = run_closed(b, 50, lambda t: e)
        assert float(np.max(np.abs(by_phi - by_closed))) < 1e-15
        np.testing.assert_array_equal(traj(b, 50, e)[-1], by_closed[-1])


# ---------------------------------------------------------------------------
# Test 1: what the parent reports.
# ---------------------------------------------------------------------------


def test_1_the_report_is_the_mean_of_the_message_the_parent_sends():
    """REPRODUCES the closed form for ``sigma_a`` and LICENSES ``report_of`` and ``report``.

    ``sigma_a(q, m) = (sum_j kappa_j V_j h_j, sum_j kappa_j w_j)`` (eq:maps), so the mean readout
    of what the parent sends is ``report_of`` -- at random placements and value maps, not at one
    configuration. The same for the four-scene parent, whose report's ``i``th coordinate is
    ``(v_smoke x_i + v_wind u_i) / 2K``."""
    kids, a, _ = fire_system(0.3, 0.3)
    r = np.random.default_rng(601)
    m = messages(kids, (X, U))
    for _ in range(6):
        q = spatial_config(r.uniform(-2.0, 2.0, 2), r.uniform(-2.0, 2.0, 2))
        np.testing.assert_allclose(report_of(q, m), mean(a.sigma(q, m)), rtol=1e-14)

    kids4, a4, report4, _ = shared_system(CHARGED, 0.5)
    m4 = messages(kids4, SCENE_VALUES)
    for _ in range(6):
        q = jnp.asarray(r.uniform(-3.0, 3.0, 2))
        np.testing.assert_allclose(report4(q, m4), mean(a4.sigma(q, m4)), rtol=1e-14)
        by_hand = jnp.asarray([(q[0] * x + q[1] * u) / (2 * K) for x, u in SCENES])
        np.testing.assert_allclose(report4(q, m4), by_hand, rtol=1e-14)


# ---------------------------------------------------------------------------
# Test 2: identity value maps -- the report stays between the scouts.
# ---------------------------------------------------------------------------


def mobility_hull():
    """``1 / L``, ``L`` the largest ``|eigenvalue|`` of the cost's Hessian in the two placements
    over ``[-REGION, REGION]^2`` at the identity value maps: for an ``L``-smooth cost a step of
    ``1/L`` decreases it every step (the descent lemma). The region is named because the constant
    is region-dependent (STATUS.md §3.2 item 6) and the run is checked to stay inside it."""
    def k(p):
        kap = jnp.exp(-p ** 2 / (2.0 * TAU))
        return 0.5 * ((kap[0] * X + kap[1] * U) / jnp.sum(kap) - Y) ** 2

    ps = jnp.linspace(-REGION, REGION, 121)
    H = jax.vmap(lambda p1: jax.vmap(lambda p2: jax.hessian(k)(jnp.array([p1, p2])))(ps))(ps)
    return 1.0 / float(jnp.max(jnp.abs(jnp.linalg.eigvalsh(H))))


def test_2_with_identity_value_maps_the_report_stays_between_the_scouts():
    """CHECKS THE DISCRETE ANALOGUE of prediction (1), and the prediction holds.

    With scalar transfers the report is ``sum_j kappa_j w_j hbar_j / sum_j kappa_j w_j``, every
    ``kappa_j w_j > 0``: a convex combination of the scouts' reports. So it lies in
    ``[U, X] = [0.5, 3]`` whatever the placements do, and its distance from ``Y = 5`` is at least
    ``Y - X = 2``. Held at equal attention the parent says the plain average, ``1.75``. Over
    20 000 steps at mobility ``1/L = 0.005071`` (``L = 197.2`` over ``[-12, 12]^2``) the cost
    decreases every step, the run stays in the named region, and the report rises to ``2.99976``,
    a distance ``2.00024`` from the truth: the floor approached and not reached.

    What descent does with the attention, and one clause of the prediction that did not hold as
    stated. The wind scout is pushed from ``1`` out to ``3.854`` (``kappa = 9.27e-5``) and the
    smoke scout is drawn in from ``1`` to ``0.258``, so the ratio of the two attentions is
    ``1.03e4`` and rising; but the smoke scout's own ``kappa`` reaches only ``0.959``, not ``1``.
    A cost on the mean readout is blind to the total weight -- ``mean(kappa m) = mean(m)`` -- so
    once the wind scout's weight is negligible the parent has almost no reason left to hold the
    smoke scout closer, and what still moves it is only the ratio (`notes/TESTS-SYNTHESIS-2026-09-26.md`
    §2, "anything through the mean readout: expect blindness to attention"). Both placements are
    still moving at the budget, at ``7.9e-7`` and ``1.2e-5`` a step: a rate, not a boundary.

    The floor is not an artefact of where the parent started holding them: four other starts,
    near and far, symmetric and not, all end above it."""
    mu = mobility_hull()
    np.testing.assert_allclose(mu, 0.005071474, rtol=1e-6)
    kids, a, b = fire_system(mu, 0.0)
    rows = traj(b, BUDGET)
    kap, r = spatial_report(rows)
    cost = 0.5 * (r - Y) ** 2

    assert np.all(rows[:, 4:6] == 1.0)                     # the value maps are frozen
    assert np.all(np.diff(cost) <= 1e-14)                  # the cost decreases every step
    assert np.abs(rows[:, 2:4]).max() < REGION             # inside the named region
    assert np.all(r > U) and np.all(r < X)                 # never leaves the hull
    assert np.all(Y - r > Y - X - 1e-12)                   # never below the floor
    np.testing.assert_allclose((kap[0, 0] * X + kap[0, 1] * U) / kap[0].sum(), r[0], rtol=1e-14)
    np.testing.assert_allclose(r[-1], 2.99975833, rtol=1e-7)
    np.testing.assert_allclose(Y - r[-1], 2.00024167, rtol=1e-7)
    np.testing.assert_allclose(rows[-1, 2:4], [0.25781148, 3.85448905], rtol=1e-6)
    np.testing.assert_allclose(kap[-1], [0.959309309, 9.27429685e-05], rtol=1e-6)
    assert kap[-1, 0] / kap[-1, 1] > 1e4                   # the ratio of attentions, not kappa -> 1
    assert 0.95 < kap[-1, 0] < 1.0 - 1e-3                  # the smoke scout stops short of home
    assert abs(rows[-1, 3]) > abs(rows[0, 3]) + 1.0        # the wind scout is pushed out
    drift = np.abs(rows[-1, 2:4] - rows[-2, 2:4])
    assert drift.min() > 0.0                               # both still moving: a rate
    np.testing.assert_allclose(drift, [7.90e-07, 1.181e-05], rtol=1e-2)

    for start in ((0.5, 0.5), (2.0, 0.3), (0.2, 3.0), (3.0, 3.0)):
        _, _, b2 = fire_system(mu, 0.0, positions=start)
        _, r2 = spatial_report(traj(b2, BUDGET))
        assert U < r2[-1] < X
        assert Y - r2[-1] > Y - X - 1e-12


# ---------------------------------------------------------------------------
# Test 3: free value maps -- the report leaves the hull, and one scene is a line of maps.
# ---------------------------------------------------------------------------


def test_3_free_value_maps_leave_the_hull_and_one_scene_does_not_learn_the_rule():
    """REPRODUCES a closed form and CORRECTS the design in `notes/TESTS-SYNTHESIS-2026-09-26.md`
    §4, which said that with the value maps free "the wind's map learns the elapsed time".

    With the placements frozen at equal distance the two kernel values are equal, so the report
    is ``(v_smoke X + v_wind U)/2``, affine in the maps, and the cost is the quadratic
    ``(1/2)(a . v - Y)^2`` with ``a = (X, U)/2 = (1.5, 0.25)``. Its Hessian ``a a^T`` has rank
    one, eigenvalues ``|a|^2 = 2.3125`` and ``0``. So (i) the residual is multiplied by exactly
    ``1 - mu_V |a|^2 = 0.5375`` each step; (ii) the report leaves ``[0.5, 3]`` on its way to
    ``Y = 5``; and (iii) the settled pair is the orthogonal projection of the starting pair on
    the solution line,

        v* = (1, 1) + t (X, U)/2,   t = (Y - (X + U)/2) / |a|^2 = 1.405405,

    that is ``(3.108108, 1.351351)``. The rule -- the pair reporting ``x + DT u`` for *every*
    scene -- is ``(2, 2 DT) = (2, 8)``, and it is on the same line: one scene fixes a line of
    pairs, not a pair. The ratio of the settled maps is ``0.434783``, not ``DT = 4``, and a
    parent started elsewhere settles elsewhere on the line. **The design's claim is exactly what
    one scene cannot show.**"""
    a_vec = np.array([X, U]) / 2.0
    L = float(a_vec @ a_vec)
    rate = abs(1.0 - MU_V * L)
    np.testing.assert_allclose([L, rate], [2.3125, 0.5375], rtol=1e-15)

    _, _, b = fire_system(0.0, MU_V)
    rows = traj(b, 400)
    v = rows[:, 4:6]
    r = (v[:, 0] * X + v[:, 1] * U) / 2.0
    res0 = (X + U) / 2.0 - Y
    np.testing.assert_allclose(r[:20] - Y, res0 * rate ** np.arange(1, 21), rtol=1e-9)

    t = (Y - (X + U) / 2.0) / L
    v_star = np.array([1.0, 1.0]) + t * a_vec
    np.testing.assert_allclose(t, 1.4054054054054054, rtol=1e-14)
    np.testing.assert_allclose(v_star, [3.1081081081, 1.3513513514], rtol=1e-10)
    np.testing.assert_allclose(v[-1], v_star, atol=1e-12)
    np.testing.assert_allclose(r[-1], Y, atol=1e-12)
    assert np.abs(forcing(b, rows[-1])[4:6]).max() < 1e-12

    _, r_check = spatial_report(rows)
    np.testing.assert_allclose(r_check, r, rtol=1e-14)
    assert r[-1] > X                                    # the report leaves the hull
    assert np.any((r > X) & (r < Y))                    # and passes through on the way

    rule = np.array([2.0, 2.0 * DT])                    # on the same line, and not where it lands
    np.testing.assert_allclose(float(a_vec @ rule), Y, rtol=1e-14)
    np.testing.assert_allclose(v[-1, 1] / v[-1, 0], 0.4347826087, rtol=1e-9)
    assert abs(v[-1, 1] / v[-1, 0] - DT) > 3.0

    _, _, b2 = fire_system(0.0, MU_V, values=(2.5, -1.0))
    v2 = traj(b2, 400)[-1, 4:6]
    np.testing.assert_allclose(float(a_vec @ v2), Y, rtol=1e-12)
    assert np.abs(v2 - v_star).max() > 0.5


# ---------------------------------------------------------------------------
# Test 4: three scenes pin the rule down, and it holds on a scene never charged for.
# ---------------------------------------------------------------------------


def normal_matrix(charged, scenes=SCENES):
    """``sum_{i in charged} c_i c_i^T / (2K)^2``, ``c_i = (x_i, u_i)``: the Hessian of the shared
    parent's cost in ``(v_smoke, v_wind)``, positive definite exactly when the charged scenes'
    vectors ``(x_i)`` and ``(u_i)`` are linearly independent."""
    C = np.array([scenes[i] for i in sorted(charged)])
    return C.T @ C / (2.0 * len(scenes)) ** 2


def settled_maps(charged, scenes=SCENES, init=(1.0, 1.0), steps=BUDGET):
    """The value-map pair the shared parent settles at, at mobility ``1/L``."""
    mu = 1.0 / float(np.linalg.eigvalsh(normal_matrix(charged, scenes)).max())
    kids, a, report, b = shared_system(charged, mu, init, scenes)
    rows = traj(b, steps, exterior(len(scenes)))
    return rows[-1, 2 * len(scenes):2 * len(scenes) + 2], kids, report, b, rows


def test_4_three_scenes_pin_the_rule_and_it_holds_on_a_scene_never_charged_for():
    """CHECKS THE DISCRETE ANALOGUE of prediction (3), and the prediction holds.

    The parent holds four scenes' scouts and is charged for three of them. That Hessian,
    ``[[0.21875, 0.003906], [0.003906, 0.013672]]``, is positive definite (determinant
    ``0.0029755``, eigenvalues ``0.013597`` and ``0.218824``, condition number ``16.1``), so the
    quadratic has the unique minimizer ``(2K, 2K DT) = (8, 32)`` -- the rule, which reports
    ``x + DT u`` for every scene. At mobility ``1/L = 4.5699`` descent reaches it to ``1e-9``
    within the budget, **the ratio of the maps is the elapsed time ``DT = 4``** to fourteen
    digits, the report on the three charged scenes is ``(5, -2, -1)``, their truth, and the
    report on the fourth, which the parent is never charged for, is ``3.6e-15`` against its truth
    ``0``. The report leaves the hull of its scouts' reports by a plain bound: a convex
    combination of messages each carrying one coordinate has ``sum_i |r_i| <= max_j |value_j| =
    4``, and the settled report has ``sum_i |r_i| = 8``.

    Charged on scene 0 alone the Hessian is singular, and the parent settles at ``(12.8378,
    2.97297)``, the projection of its start on that scene's solution line: right on scene 0 and
    wrong on the other three by ``3.33``, ``2.12`` and ``6.05``, with a map ratio of ``0.2316``.
    Two charged scenes with independent vectors are already enough -- scenes 0 and 2 give the
    same ``(8, 32)`` -- and so is starting from a quite different pair, ``(-2, 5)``, which is
    what distinguishes a rule from a fit."""
    H = normal_matrix(CHARGED)
    ev = np.linalg.eigvalsh(H)
    assert ev.min() > 1e-3                                   # positive definite: rank 2
    np.testing.assert_allclose(ev, [0.01359750, 0.21882438], rtol=1e-6)
    np.testing.assert_allclose(np.linalg.det(H), 0.00297546387, rtol=1e-9)
    rule = np.array([2.0 * K, 2.0 * K * DT])

    v, kids, report, b, rows = settled_maps(CHARGED)
    np.testing.assert_allclose(v, rule, atol=1e-9)
    np.testing.assert_allclose(v[1] / v[0], DT, rtol=1e-10)
    assert np.abs(forcing(b, rows[-1], exterior(K))[2 * K:]).max() < 1e-9

    r_end = np.asarray(report(jnp.asarray(v), messages(kids, SCENE_VALUES)))
    np.testing.assert_allclose(r_end, [truth(s) for s in SCENES], atol=1e-9)
    assert np.abs(r_end).sum() > max(abs(w) for w in SCENE_VALUES) + 1e-9   # outside the hull

    v_other, _, _, _, _ = settled_maps(CHARGED, init=(-2.0, 5.0))
    np.testing.assert_allclose(v_other, rule, atol=1e-9)     # a rule, not a fit

    # charged on one scene: a line, not a pair
    np.testing.assert_allclose(np.linalg.det(normal_matrix((0,))), 0.0, atol=1e-18)
    v1, kids1, report1, _, _ = settled_maps((0,))
    np.testing.assert_allclose(v1, [12.83783784, 2.97297297], rtol=1e-8)
    r1 = np.asarray(report1(jnp.asarray(v1), messages(kids1, SCENE_VALUES)))
    np.testing.assert_allclose(r1[0], truth(SCENES[0]), atol=1e-9)
    wrong = np.abs(r1[1:] - np.array([truth(s) for s in SCENES[1:]]))
    assert wrong.min() > 1.0
    np.testing.assert_allclose(wrong, [3.32601351, 2.11655405, 6.04729730], rtol=1e-7)
    assert abs(v1[1] / v1[0] - DT) > 3.0

    # two independent scenes are already enough
    v2, _, _, _, _ = settled_maps((0, 2))
    np.testing.assert_allclose(v2, rule, atol=1e-9)
