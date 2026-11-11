"""One magnet, duplicated (book plan P76 (h); the predictions, written first, are in the book repo's
``notes/runs/magnets-2524b69e/RESULTS.md``).

Every test runs the closed arrangement ``plane . (bodies)`` through the dynamics functor. The numbers each is compared with are
closed formulas for a pair, written here and nowhere in the library: the library holds the field of one wire and each part's own
cost. Claude Fable 5.2, session 2524b69e, 2026-11-10.

Since book plan P76 (i) the wires carry the fields themselves (``magnets.py`` on ``smooth.py``), and the tests of P76 (h) stand as
they were but for the stepping, ``smooth.PhiK``. The tests after ``test_no_currents_no_force`` are P76 (i)'s (predictions in the book
repo's ``notes/runs/p76i-7328b8c7/RESULTS.md``): the runs against the version on ``R^n`` (``magnets_rn.py``), one arrangement for any
bodies, a body made of bodies, and the two readings of that version's ``_transfer``. Claude Fable 5.2, session 7328b8c7, 2026-11-10.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import magnets as M
from arrangements import magnets_rn as RN
from arrangements import smooth as sm
from arrangements.interpretation import trivial_omega
from arrangements.magnets import bumper, dead_loop, magnet, magnet_of_wires, system, system_of_carts, wire
from arrangements.pcK import PhiK as PhiK_rn
from arrangements.smooth import R
from arrangements.wiring import trivial_parameter

jax.config.update("jax_enable_x64", True)

HEAVY = 1e12
UNIT, PINNED, FREE = [1.0, 1.0, HEAVY], [HEAVY, HEAVY, HEAVY], [1.0, 1.0, 1.0]


def trace(arr, steps):
    O = sm.PhiK(arr)
    step = jax.jit(lambda s: O.with_state(s).run_one(sm.CLOSED_IN, sm.closed_tell)[1])
    flat = lambda s: np.concatenate([np.asarray(x, float).ravel() for x in jax.tree_util.tree_leaves(s)])
    s = O.state
    out = [flat(s)]
    for _ in range(steps):
        s = step(s)
        out.append(flat(s))
    return np.array(out)


def first_push(bodies, q0, mass):
    tau = 1e-5
    n = len(bodies)
    X = trace(system(bodies, q0, np.zeros((n, 3)), mass, tau=tau), 1)
    return (X[1, 3 * n:] / tau).reshape(n, 3)


def test_label():
    arr = system([magnet(1.0, 0.25), magnet(1.0, 0.25)], [[-1.5, 0, 0], [1.5, 0, 0]], np.zeros((2, 3)), [UNIT, UNIT])
    assert arr.label == "(plane(2) . (magnet(1,h=0.25) || magnet(1,h=0.25)))"


@pytest.mark.parametrize("cur,sign", [(1.0, +1.0), (-1.0, -1.0)])
def test_parallel_currents_attract(cur, sign):
    F = first_push([wire(1.0), wire(cur)], [[-1, 0, 0], [1, 0, 0]], [UNIT, UNIT])
    assert F[0, 0] == pytest.approx(sign / (4 * np.pi), rel=1e-6)
    assert F[1, 0] == pytest.approx(-sign / (4 * np.pi), rel=1e-6)


@pytest.mark.parametrize("q_m", [[2.0, 0.7, 0.4], [1.5, -1.0, 2.0]])
def test_a_magnet_is_two_wires(q_m):
    h = 0.25
    F = first_push([wire(1.0), magnet(1.0, h)], [[0, 0, 0], q_m], [PINNED, FREE])[1]
    e = np.array([np.cos(q_m[2]), np.sin(q_m[2])])
    ends = [np.array(q_m[:2]) + h * e, np.array(q_m[:2]) - h * e]
    Fp = first_push([wire(1.0), wire(1.0)], [[0, 0, 0], [*ends[0], 0.0]], [PINNED, FREE])[1][:2]
    Fm = first_push([wire(1.0), wire(-1.0)], [[0, 0, 0], [*ends[1], 0.0]], [PINNED, FREE])[1][:2]
    cross = lambda a, b: a[0] * b[1] - a[1] * b[0]
    assert np.allclose(F[:2], Fp + Fm, atol=1e-10)
    assert F[2] == pytest.approx(cross(h * e, Fp) + cross(-h * e, Fm), abs=1e-10)


@pytest.mark.parametrize("angle,cur,want", [(np.pi / 2, 1.0, +0.002868), (np.pi / 2, -1.0, -0.002868),
                                            (0.0, 1.0, -0.003032), (0.0, -1.0, +0.003032)])
def test_two_magnets_by_how_they_are_turned(angle, cur, want):
    """Head to tail they pull, side by side they push, and flipping one reverses each: the force on the left magnet toward the right
    one, against ``8 h^2 / (2 pi D^3 (1 + 4 h^2 / D^2))`` and ``8 h^2 / (2 pi D (D^2 - 4 h^2))`` at ``D = 3``, ``h = 1/4``."""
    bodies = [magnet(1.0, 0.25), magnet(cur, 0.25)]
    q0 = [[-1.5, 0, angle], [1.5, 0, angle]]
    F = first_push(bodies, q0, [UNIT, UNIT])
    assert F[0, 0] == pytest.approx(want, abs=5e-7)
    assert F[1, 0] == pytest.approx(-want, abs=5e-7)
    X = trace(system(bodies, q0, np.zeros((2, 3)), [UNIT, UNIT], tau=0.05), 300)
    moved = (X[-1, 3] - X[-1, 0]) - 3.0
    assert np.sign(moved) == -np.sign(want) and abs(moved) > 0.1


def test_a_magnet_turns_to_line_up():
    bodies = [magnet(1.0, 0.25), magnet(1.0, 0.25)]
    q0 = [[-1.5, 0, np.pi / 2], [1.5, 0, np.pi / 2 + 0.5]]
    X = trace(system(bodies, q0, np.zeros((2, 3)), [PINNED, [HEAVY, HEAVY, 0.01]], tau=0.05), 400)
    th = X[:, 5] - np.pi / 2
    assert th.max() <= 0.5 + 1e-9 and th.min() == pytest.approx(-0.5, abs=1e-3)
    assert th[197] == pytest.approx(0.5, abs=1e-3)          # back after one swing, time 9.85


def test_a_dead_loop_opposes_a_passing_magnet():
    bodies = [dead_loop(1.0, 0.5), magnet(1.0, 0.25)]
    X = trace(system(bodies, [[0, 0, 0], [-5, 1, 0]], [[0, 0, 0], [1, 0, 0]], [PINNED, UNIT], tau=0.01), 1000)
    speed = X[:, 9]
    assert speed.min() == pytest.approx(0.998121, abs=2e-6) and abs(X[int(np.argmin(speed)), 3]) < 0.05
    assert speed[-1] == pytest.approx(1.0, abs=1e-4)
    kinetic = 0.5 * speed ** 2 + 0.5 * X[:, 10] ** 2
    # the loop's energy is never negative, so the magnet never has more than the total: its own at the start plus the 4.1e-6 the
    # loop already holds there (the flux is not yet 0 at x = -5)
    assert kinetic.max() <= kinetic[0] + 4.1e-6


def test_a_magnet_composed_of_two_wires_is_the_magnet():
    """``pair(h) . (wire || wire)`` in place of ``magnet``: the same run, to rounding, the carts free to slide and to turn."""
    q0, p0, mass = [[-1.5, 0, 1.2], [1.5, 0.4, 2.0]], [[0.1, 0, 0], [0, 0, 0.05]], [FREE, FREE]
    one = system([magnet(1.0, 0.25), magnet(-0.7, 0.4)], q0, p0, mass, tau=0.05)
    two = system([magnet_of_wires(1.0, 0.25), magnet_of_wires(-0.7, 0.4)], q0, p0, mass, tau=0.05)
    assert two.label == "(plane(2) . ((pair(h=0.25) . (wire(1) || wire(-1))) || (pair(h=0.4) . (wire(-0.7) || wire(0.7)))))"
    A, B = trace(one, 300), trace(two, 300)
    assert np.abs(A - B).max() < 1e-11
    assert np.abs(A[-1, :6] - A[0, :6]).max() > 0.1          # and they did move


def test_independent_carts_are_the_plane():
    """Each cart a part of its own, and an arrangement with no state, against ``plane`` with all the carts in its integrator."""
    bodies = [magnet(1.0, 0.25), magnet(1.0, 0.25), wire(0.5)]
    q0, p0, mass = [[-1.5, 0, 1.2], [1.5, 0.4, 2.0], [0, 2, 0]], [[0.1, 0, 0], [0, 0, 0.05], [0, 0, 0]], [FREE] * 3
    whole = system(bodies, q0, p0, mass, tau=0.05)
    parts = system_of_carts(bodies, q0, p0, mass, tau=0.05)
    assert parts.label == "(field(3) . (((((magnet(1,h=0.25) || magnet(1,h=0.25)) || wire(0.5)) || cart) || cart) || cart))"
    A, B = trace(whole, 300), trace(parts, 300)              # A: all q then all p; B: (q, p) cart by cart
    B = np.concatenate([B[:, [0, 1, 2, 6, 7, 8, 12, 13, 14]], B[:, [3, 4, 5, 9, 10, 11, 15, 16, 17]]], axis=1)
    assert np.abs(A - B).max() < 1e-11


def _two_carts(cur, b, drag, tau, I=2.0):
    bodies = [magnet(I, 0.25), bumper(b, 0.25), bumper(b, -0.25), magnet(cur, 0.25), bumper(b, 0.25), bumper(b, -0.25)]
    return system(bodies, [[-1.5, 0, np.pi / 2], [1.5, 0, np.pi / 2]], np.zeros((2, 3)), [UNIT, UNIT], tau=tau,
                  cart_of=[0, 0, 0, 1, 1, 1], drag=drag)


def test_bumpers_stop_two_magnets_where_pull_equals_push():
    """Each cart carries a magnet and a bumper at each of its wires; with drag the carts come to rest where the magnets' pull equals
    the bumpers' push, ``D* = 0.395410``, the root of the pair formulas in the book repo's RESULTS (h9), found there by bisection."""
    X = trace(_two_carts(2.0, 1.0, 0.5, 0.01), 6000)
    r = X[:, 3] - X[:, 0]
    assert r.min() > 0.3                                     # they never pass through each other
    assert r[-1] == pytest.approx(0.395410, abs=1e-3)
    assert abs(X[-1, 6]) < 1e-4 and abs(X[-1, 9]) < 1e-4


def test_flipped_magnets_with_bumpers_part():
    X = trace(_two_carts(-2.0, 1.0, 0.5, 0.01), 3000)
    r = X[:, 3] - X[:, 0]
    assert np.all(np.diff(r) >= 0) and r[-1] > 3.5


def test_no_bumpers_no_drag_is_the_plain_run():
    plain = trace(system([magnet(1.0, 0.25), magnet(1.0, 0.25)], [[-1.5, 0, np.pi / 2], [1.5, 0, np.pi / 2]], np.zeros((2, 3)),
                         [UNIT, UNIT], tau=0.05), 300)
    assert np.abs(plain - trace(_two_carts(1.0, 0.0, 0.0, 0.05, I=1.0), 300)).max() < 1e-12


def test_no_currents_no_force():
    bodies = [magnet(0.0, 0.25), magnet(0.0, 0.25), wire(0.0)]
    X = trace(system(bodies, [[-1, 0, 0.3], [1, 0.2, 1.0], [0, 2, 0]], [[0.1, 0, 0], [0, -0.2, 0], [0, 0, 0]], [FREE] * 3), 200)
    assert np.abs(X[:, 9:] - X[0, 9:]).max() == 0.0


# ---------------------------------------------------------------------------
# Book plan P76 (i): the fields on the wires.
# ---------------------------------------------------------------------------

leaves = lambda x: np.concatenate([np.asarray(a, float).ravel() for a in jax.tree_util.tree_leaves(x)] + [np.zeros(0)])
BODY = (M.FIELDS, M.FIELDS)


def trace_rn(arr, steps):
    """A run of the version on ``R^n``, through the array core's ``PhiK``."""
    O, IN = PhiK_rn(arr), (jnp.zeros(0), trivial_omega(0))
    step = jax.jit(lambda s: O.with_state(s).run_one(IN, lambda _o: (jnp.zeros(0), jnp.zeros(0)))[1])
    s, out = O.state, [leaves(O.state)]
    for _ in range(steps):
        s = step(s)
        out.append(leaves(s))
    return np.array(out)


def both(spec, *args, **kw):
    """The same system in the two versions: ``spec`` lists each body as ``(constructor's name, its arguments)``."""
    return (M.system([getattr(M, name)(*a) for name, a in spec], *args, **kw),
            RN.system([getattr(RN, name)(*a) for name, a in spec], *args, **kw))


@pytest.mark.parametrize("spec,q0,p0,mass,kw", [
    ([("magnet", (1.0, 0.25)), ("magnet", (-0.7, 0.4))], [[-1.5, 0, 1.2], [1.5, 0.4, 2.0]], [[0.1, 0, 0], [0, 0, 0.05]],
     [FREE, FREE], {}),
    ([("magnet", (1.0, 0.25)), ("magnet", (1.0, 0.25)), ("wire", (0.5,))], [[-1.5, 0, 1.2], [1.5, 0.4, 2.0], [0, 2, 0]],
     [[0.1, 0, 0], [0, 0, 0.05], [0, 0, 0]], [FREE] * 3, {}),
    ([("dead_loop", (1.0, 0.5)), ("magnet", (1.0, 0.25))], [[0, 0, 0], [-2, 1, 0]], [[0, 0, 0], [1, 0, 0]], [PINNED, UNIT], {}),
    ([("magnet", (2.0, 0.25)), ("bumper", (1.0, 0.25)), ("bumper", (1.0, -0.25)),
      ("magnet", (2.0, 0.25)), ("bumper", (1.0, 0.25)), ("bumper", (1.0, -0.25))],
     [[-1.5, 0, np.pi / 2], [1.5, 0, np.pi / 2]], np.zeros((2, 3)), [UNIT, UNIT], dict(cart_of=[0, 0, 0, 1, 1, 1], drag=0.5)),
], ids=["two magnets", "two magnets and a wire", "a dead loop and a magnet", "bumpers and drag"])
def test_the_run_is_the_run_on_Rn(spec, q0, p0, mass, kw):
    """The field-valued composite against the composite on ``R^n``, step for step: the equality argued by hand in the book repo's
    ``notes/runs/magnets-2524b69e/RESULTS.md`` (h10)."""
    new, old = both(spec, q0, p0, mass, tau=0.05, **kw)
    assert new.label == old.label
    A, B = trace(new, 300), trace_rn(old, 300)
    assert A.shape == B.shape and np.abs(A - B).max() < 1e-11
    assert np.abs(A[-1] - A[0]).max() > 0.01


def test_one_arrangement_for_any_bodies():
    """The arrangement is built once, without the bodies; swapping a magnet of half-width 0.25 for one of 0.4 changes that part and
    nothing else. On ``R^n`` the two magnets' boxes were one arrangement and the plane was built again for each."""
    q0, p0, mass = [[-1.5, 0, 1.2], [1.5, 0.4, 2.0]], [[0.1, 0, 0], [0, 0, 0.05]], [FREE, FREE]
    the_plane, the_wire = M.plane(2, q0, p0, mass, tau=0.05), wire(0.5)
    narrow, wide = magnet(1.0, 0.25), magnet(1.0, 0.4)
    assert narrow.source == wide.source == ((), ()) and narrow.target == wide.target == BODY == the_wire.target
    x = jnp.array([0.7, 0.3])
    assert abs(float(narrow.out_f(jnp.zeros(0), ())[0](x)) - float(wide.out_f(jnp.zeros(0), ())[0](x))) > 0.01
    runs = [trace(sm.compose_seq(sm.tensor([m, the_wire]), the_plane), 300) for m in (narrow, wide)]
    assert np.abs(runs[0] - runs[1]).max() > 1e-3
    for run, h in zip(runs, (0.25, 0.4)):
        old = RN.system([RN.magnet(1.0, h), RN.wire(0.5)], q0, p0, mass, tau=0.05)
        assert np.abs(run - trace_rn(old, 300)).max() < 1e-11
    # on R^n the two magnets' parts put out the same number and have the same cost; only their labels differ
    a, b, z = RN.magnet(1.0, 0.25).box, RN.magnet(1.0, 0.4).box, jnp.zeros(0)
    n = jnp.array([0.3, -0.8])
    assert np.all(a.out_f(z, z) == b.out_f(z, z)) and np.all(leaves(a.omega(z, z, n)) == leaves(b.omega(z, z, n)))


def _received_plot(v):
    """A plot of shape ``R^3`` of ``Q x 1 x (F x F)`` for a stateless body: a pair of received fields varying with ``v``."""
    return (jnp.zeros(0), (),
            (lambda r: v[0] * jnp.sin(r[0]) + v[1] * (1.0 + r[1] ** 2) + v[2] * r[0] * r[1],
             lambda r: jnp.cos(v[2] * r[0]) + v[0] * r[1] + v[1]))


_V = jnp.array([0.4, -1.3, 0.7])
_POINTS = [jnp.array(p) for p in ([0.7, 0.3], [-1.1, 0.6], [0.2, -2.0])]


def same_body(A, B, tol):
    """Two stateless systems on a body's interface: the fields they put out at three points, and their inclination fields on a
    plot of received fields."""
    assert A.source == B.source == ((), ()) and A.target == B.target == BODY and A.P.Q_dim == B.P.Q_dim == 0
    z = jnp.zeros(0)
    for k in (M.MAGNETIC, M.CONTACT):
        for x in _POINTS:
            assert abs(float(A.out_f(z, ())[k](x)) - float(B.out_f(z, ())[k](x))) < tol
    assert np.abs(leaves(A.omega(_received_plot, _V)) - leaves(B.omega(_received_plot, _V))).max() < tol
    return float(np.abs(leaves(A.omega(_received_plot, _V))).max())


@pytest.mark.parametrize("current,h", [(1.0, 0.25), (-0.7, 0.4)])
def test_a_body_made_of_bodies_is_a_body(current, h):
    """``pair(h) . (wire || wire)`` has the magnet's interface, puts out the magnet's field, and has the magnet's inclination
    field: their costs differ by a constant."""
    made, one = magnet_of_wires(current, h), magnet(current, h)
    assert made.label == f"(pair(h={h:g}) . (wire({current:g}) || wire({-current:g})))"
    # (the costs themselves differ by the two wires' energy in each other's field, a constant; an arrangement holds the 1-form)
    assert same_body(made, one, 1e-12) > 0.1


def test_composition_is_associative_and_the_tensor_functorial_on_field_interfaces():
    """Three groupings of four wires, two pairs and the plane, every wire between them carrying fields:
    ``plane . ((pair (x) pair) . (wires (x) wires))``, ``(plane . (pair (x) pair)) . (wires (x) wires)`` and
    ``plane . ((pair . wires) (x) (pair . wires))``. The same load on the carts' configuration at random configurations, and the
    load of the two one-part magnets."""
    f1, f2 = sm.tensor([wire(1.0), wire(-1.0)]), sm.tensor([wire(-0.7), wire(0.7)])
    g1, g2 = M.pair(0.25), M.pair(0.4)
    z = np.zeros((2, 3))
    h = M.plane(2, z, z, [FREE, FREE])
    groupings = [sm.compose_seq(sm.compose_seq(sm.tensor([f1, f2]), sm.tensor([g1, g2])), h),
                 sm.compose_seq(sm.tensor([f1, f2]), sm.compose_seq(sm.tensor([g1, g2]), h)),
                 sm.compose_seq(sm.tensor([sm.compose_seq(f1, g1), sm.compose_seq(f2, g2)]), h),
                 M.system([magnet(1.0, 0.25), magnet(-0.7, 0.4)], z, z, [FREE, FREE])]
    assert all(a.source == a.target == ((), ()) and a.P.Q_dim == 6 for a in groupings)
    rng = np.random.default_rng(1)
    for _ in range(3):
        q = jnp.asarray(rng.normal(size=6) * 1.5)
        loads = [leaves(a.omega(lambda v: (v, (), ()), q)) for a in groupings]
        assert np.abs(loads[0]).max() > 1e-3
        for other in loads[1:]:
            assert np.abs(other - loads[0]).max() < 1e-12


def lens(b: RN.Body) -> sm.SmoothArrangement:
    """The static arrangement ``<(R(k)) | (R(p))> -> <FIELDS | FIELDS>`` of a body of the version on ``R^n``: out, its numbers to its
    fields (``b.field``); in, the received fields to their values at the points its cost reads (``b.reads``)."""
    k, p = len(b.reads), b.n_out

    def out_f(q, m):
        if b.field is None:
            return (M.nothing, M.nothing)
        return tuple((lambda r, col=col: b.field(m[0], r[None, :])[0, col]) for col in (M.MAGNETIC, M.CONTACT))

    def in_f(q, m, n):
        return (jnp.stack([n[which](jnp.array([x, y])) for x, y, which in b.reads]),)

    return sm.SmoothArrangement(P=trivial_parameter(), M_in=(R(k),), M_out=(R(p),), N_in=M.FIELDS, N_out=M.FIELDS,
                                out_f=out_f, in_f=in_f, omega=sm.zero_form, label="lens")


@pytest.mark.parametrize("name,args", [("wire", (1.3,)), ("magnet", (1.0, 0.25)), ("magnet", (-0.7, 0.4)),
                                       ("dead_loop", (1.0, 0.5)), ("bumper", (1.0, 0.25, 0.25))])
def test_a_body_is_its_part_on_Rn_through_its_lens(name, args):
    """``lens_b . include(B_b)`` is the field-valued body, ``B_b`` the part of the version on ``R^n``."""
    b = getattr(RN, name)(*args)
    size = same_body(sm.compose_seq(sm.include(b.box), lens(b)), getattr(M, name)(*args), 1e-12)
    assert size > 0.05


def test_the_transfer_on_Rn_is_the_plane_with_the_lenses_composed_in():
    """``plane . (lens_1 (x) lens_2 (x) lens_3 (x) lens_4)`` against ``include`` of the plane of the version on ``R^n``, which is
    built from the bodies: the same in-transfer at random configurations and numbers (both have no out-transfer and zero field)."""
    bodies = [RN.magnet(1.0, 0.25), RN.wire(0.5), RN.dead_loop(1.0, 0.5), RN.bumper(1.0, 0.1)]
    q0, mass = np.zeros((4, 3)), [FREE] * 4
    left = sm.compose_seq(sm.tensor([lens(b) for b in bodies]), M.plane(4, q0, q0, mass))
    ks, ps = [len(b.reads) for b in bodies], [b.n_out for b in bodies]
    right = sm.compose_seq(sm.join(ks, ps), sm.include(RN.plane(bodies, q0, q0, mass)))
    assert left.source == right.source == (tuple(R(k) for k in ks), tuple(R(p) for p in ps))
    assert left.target == right.target == ((), ()) and left.P.Q_dim == right.P.Q_dim == 12
    rng = np.random.default_rng(0)
    for _ in range(5):
        q = jnp.asarray(rng.normal(size=12) * 2.0)
        m = tuple(jnp.asarray(rng.normal(size=p)) for p in ps)
        got, want = leaves(left.in_f(q, m, ())), leaves(right.in_f(q, m, ()))
        assert got.shape == (sum(ks),) and np.abs(got - want).max() < 1e-12 * max(1.0, np.abs(want).max())
        assert np.abs(want).max() > 1e-3
        plot = lambda v: (q + v[:12], tuple(x + v[12 + i] for i, x in enumerate(m)), ())
        assert np.all(leaves(left.omega(plot, jnp.zeros(16))) == 0.0) and np.all(leaves(right.omega(plot, jnp.zeros(16))) == 0.0)


def test_a_cart_at_angle_zero_with_bumpers_at_its_wires():
    """With the two fields on two wires, a bumper sitting on a magnet's wire never asks for the magnet's own field there. The version
    on ``R^n`` computes both fields at each point read and loses the load to a NaN when the cart's angle is exactly 0."""
    spec = [("magnet", (2.0, 0.25)), ("bumper", (1.0, 0.25)), ("bumper", (1.0, -0.25)),
            ("magnet", (2.0, 0.25)), ("bumper", (1.0, 0.25)), ("bumper", (1.0, -0.25))]
    new, old = both(spec, [[-1.5, 0, 0.0], [1.5, 0, 0.0]], np.zeros((2, 3)), [UNIT, UNIT], tau=0.05, cart_of=[0, 0, 0, 1, 1, 1])
    A = trace(new, 20)
    assert np.all(np.isfinite(A)) and abs(A[-1, 0] - A[0, 0]) > 1e-4
    assert not np.all(np.isfinite(trace_rn(old, 2)))
