"""The core whose wires may carry a function (``arrangements/smooth.py``), and the inclusion of the array core into it, tested
as a functor (book plan P76 (i); the predictions, written first, are in the book repo's ``notes/runs/p76i-7328b8c7/RESULTS.md``).

Each law is checked on the arrangement's own data: the out-transfer and the in-transfer at a random point, and the
inclination field on a random plot, a smooth map from ``R^5`` into ``Q x M^+ x N^-``. Runs go through the dynamics functor on
each side: ``functors.Phi`` for the array core, ``smooth.PhiK`` for the new one. Claude Fable 5.2, session 7328b8c7, 2026-11-10.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import smooth as sm
from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.rvect import euclidean
from arrangements.smooth import Maps, R
from arrangements.wiring import (cap, chain_wire, compose_graph, compose_seq, graph_wire, harmonic_vertex, held_end,
                                 parallel_arrangements, tensor_arrangements, trivial_parameter)

jax.config.update("jax_enable_x64", True)

leaves = lambda x: np.concatenate([np.asarray(a, float).ravel() for a in jax.tree_util.tree_leaves(x)] + [np.zeros(0)])


def random_plot(A, seed, k=5):
    """A plot of shape ``R^k`` of ``Q x M^+ x N^-`` for an arrangement with Cartesian sides, and a point of ``R^k``."""
    rng = np.random.default_rng(seed)
    dims = [A.P.Q_dim] + [f.n for f in A.M_out] + [f.n for f in A.N_in]
    mats = [jnp.asarray(rng.normal(size=(d, k))) for d in dims]
    offs = [jnp.asarray(rng.normal(size=d)) for d in dims]
    nm = len(A.M_out)

    def plot(v):
        vals = [o + M @ v + 0.3 * jnp.sin(M @ v) for o, M in zip(offs, mats)]
        return vals[0], tuple(vals[1:1 + nm]), tuple(vals[1 + nm:])

    return plot, jnp.asarray(rng.normal(size=k))


def gap(A, B, seed=0):
    """The largest difference between two arrangements with Cartesian sides: their transfers at a random point and their fields
    on a random plot. Their interfaces and configuration spaces must be equal."""
    assert A.source == B.source and A.target == B.target and A.P.Q_dim == B.P.Q_dim
    plot, v = random_plot(A, seed)
    q, m, n = plot(v)
    d = [leaves(A.out_f(q, m)) - leaves(B.out_f(q, m)), leaves(A.in_f(q, m, n)) - leaves(B.in_f(q, m, n)),
         leaves(A.omega(plot, v)) - leaves(B.omega(plot, v))]
    return max([float(np.abs(x).max()) for x in d if x.size] + [0.0])


def gap_from_array(arr, S, seed=0):
    """``S``'s field on a random plot against the array core's ``omega``, pulled back along the plot here by ``jax.vjp``."""
    plot, v = random_plot(S, seed)
    one = lambda t: t[0] if t else jnp.zeros(0)
    flat = lambda w: (lambda q, m, n: (q, one(m), one(n)))(*plot(w))
    at, pull = jax.vjp(flat, v)
    return float(np.abs(leaves(pull(tuple(arr.omega(*at)))) - leaves(S.omega(plot, v))).max())


def array_identity(a, b):
    return Arrangement(P=trivial_parameter(), in_dim_M=a, out_dim_M=b, in_dim_N=a, out_dim_N=b,
                       out_f=lambda q, m: m, in_f=lambda q, m, n: n, omega=zero_form, label="id")


def particle(mass, kappa, x0=0.0, dt=0.1):
    """A harmonic particle ``1 -> <R|R>`` with the wave integrator, started at ``x0`` at rest."""
    return harmonic_vertex(1, 1, mass, [kappa],
                           parameter=lambda Q: phase(Q, init=(jnp.array([x0]), jnp.zeros(1)), dt=dt))


F1, F2 = particle(1.0, 2.0, 0.3), particle(1.7, 0.6, -0.2)


# ---- (i1) identities ----

def test_the_inclusion_preserves_identities():
    J, I = sm.include(array_identity(2, 3)), sm.identity((R(2),), (R(3),))
    assert gap(J, I) == 0.0
    plot, v = random_plot(I, 1)
    assert np.all(leaves(I.omega(plot, v)) == 0.0)


def test_identities_are_units():
    f = sm.include(compose_seq(tensor_arrangements([F1, F2]), chain_wire(2)))          # open: 1 -> <R|R>
    assert gap(sm.compose_seq(f, sm.identity(f.N_in, f.N_out)), f) < 1e-12
    assert gap(sm.compose_seq(sm.identity((), ()), f), f) < 1e-12


def _a_field_part():
    """A system on an interface with a function on each side: it reads out ``x |-> q_0 sin x`` and has the cost
    ``(f(0.3) - q_1)^2`` on the function ``f`` it receives."""
    P = configuration(euclidean(2, 0.1), init=jnp.array([0.4, -0.7]))
    return sm.SmoothArrangement(
        P=P, M_in=(), M_out=(), N_in=(Maps(1),), N_out=(Maps(1),),
        out_f=lambda q, m: (lambda x: q[0] * jnp.sin(x[0]),), in_f=lambda q, m, n: (),
        omega=sm.exact(lambda q, m, n: -(n[0](jnp.array([0.3])) - q[1]) ** 2), label="fieldPart")


def _field_plot(v):
    """A plot of shape ``R^3`` of ``Q x 1 x Maps(1)``."""
    return jnp.array([v[0] * v[1], jnp.sin(v[2])]), (), (lambda x: v[0] * x[0] ** 2 + jnp.cos(v[1] * x[0]) + v[2],)


def test_identities_are_units_on_a_field_interface():
    b, v = _a_field_part(), jnp.array([0.3, -1.1, 0.8])
    for c in (sm.compose_seq(b, sm.identity(b.N_in, b.N_out)), sm.compose_seq(sm.identity((), ()), b)):
        assert c.source == b.source and c.target == b.target
        x = jnp.array([0.9])
        q, _, n = _field_plot(v)
        assert float(c.out_f(q, ())[0](x)) == pytest.approx(float(b.out_f(q, ())[0](x)), abs=1e-15)
        assert np.abs(leaves(c.omega(_field_plot, v)) - leaves(b.omega(_field_plot, v))).max() < 1e-12
    # and the field is not zero there
    assert np.abs(leaves(b.omega(_field_plot, v))).max() > 0.1


# ---- (i2) a composite with two inputs ----

def test_the_inclusion_preserves_a_composite_with_two_inputs():
    f, g = tensor_arrangements([F1, F2]), chain_wire(2)
    whole, parts = sm.include(compose_seq(f, g)), sm.compose_seq(sm.include(f), sm.include(g))
    assert whole.label == parts.label == "(chain_wire(2) . (Part_v || Part_v))"
    for seed in range(3):
        assert gap(whole, parts, seed) < 1e-12
        assert gap_from_array(compose_seq(f, g), parts, seed) < 1e-12
    plot, v = random_plot(parts, 0)
    assert np.abs(leaves(parts.omega(plot, v))).max() > 0.1


# ---- (i3) a tensor ----

def test_the_inclusion_preserves_a_tensor_up_to_join():
    whole = sm.include(parallel_arrangements(F1, F2))
    parts = sm.compose_seq(sm.parallel(sm.include(F1), sm.include(F2)), sm.join([1, 1], [1, 1]))
    for seed in range(3):
        assert gap(whole, parts, seed) < 1e-12


def test_join_is_natural_for_arrangements_with_inputs():
    """``include(g1 (x) g2) . join = join . (include(g1) (x) include(g2))`` for two wirings with two and three inputs."""
    g1, g2 = chain_wire(2), chain_wire(3)
    left = sm.compose_seq(sm.join([2, 3], [2, 3]), sm.include(parallel_arrangements(g1, g2)))
    right = sm.compose_seq(sm.parallel(sm.include(g1), sm.include(g2)), sm.join([1, 1], [1, 1]))
    assert left.source == (((R(2), R(3))), (R(2), R(3))) and left.target == ((R(2),), (R(2),))
    for seed in range(3):
        assert gap(left, right, seed) < 1e-12


# ---- (i4) against array runs ----

def run_array(arr, steps):
    O, IN = Phi(arr), (jnp.zeros(0), trivial_omega(0))
    step = jax.jit(lambda s: O.with_state(s).run_one(IN, lambda _o: (jnp.zeros(0), jnp.zeros(0)))[2])
    s, out = O.state, [leaves(O.state)]
    for _ in range(steps):
        s = step(s)
        out.append(leaves(s))
    return np.array(out)


def run_smooth(arr, steps):
    O = sm.PhiK(arr)
    step = jax.jit(lambda s: O.with_state(s).run_one(sm.CLOSED_IN, sm.closed_tell)[1])
    s, out = O.state, [leaves(O.state)]
    for _ in range(steps):
        s = step(s)
        out.append(leaves(s))
    return np.array(out)


def test_a_held_chain_runs_as_in_the_array_core():
    """Three masses between two held ends, the wave integrator: built in the array core and run through ``Phi``, against the
    same chain built in the new core from the included parts (tensor, ``join``, the wiring, the cap) and run through
    ``smooth.PhiK``."""
    end = lambda kappa: held_end(lambda Q: phase(Q, dt=0.1), kappa)
    parts = [end(0.0), particle(1.0, 2.0, 0.3), particle(1.7, 0.6, -0.2), particle(0.8, 1.1, 0.5), end(1.3)]
    array = compose_seq(compose_seq(tensor_arrangements(parts), chain_wire(5)), cap())
    wired = sm.compose_seq(sm.compose_seq(sm.tensor([sm.include(p) for p in parts]), sm.join([1] * 5, [1] * 5)),
                           sm.include(chain_wire(5)))
    new = sm.compose_seq(wired, sm.include(cap()))
    assert new.label == "(cap . (chain_wire(5) . (join . ((((held || Part_v) || Part_v) || Part_v) || held))))"
    A, B = run_array(array, 50), run_smooth(new, 50)
    assert A.shape == B.shape and np.abs(A - B).max() < 1e-12
    assert np.abs(A[-1] - A[0]).max() > 0.05                  # and it moved


def test_a_triangle_runs_as_in_the_array_core():
    """``compose_graph`` on a triangle with edges both ways: the array core's composite, included, against the composite in the
    new core of the included vertices and the included prism wiring."""
    edges = [(0, 1), (1, 0), (1, 2), (2, 1), (0, 2), (2, 0)]
    start = [0.4, -0.3, 0.1]
    param = lambda x0: (lambda Q: phase(Q, init=(jnp.array([x0]), jnp.zeros(1)), dt=0.1))
    verts = [harmonic_vertex(2, 2, 1.0, [1.5, 1.5], parameter=param(x0)) for x0 in start]
    array = compose_seq(tensor_arrangements(verts), graph_wire(3, edges))
    new = sm.compose_seq(sm.compose_seq(sm.tensor([sm.include(v) for v in verts]), sm.join([2] * 3, [2] * 3)),
                         sm.include(graph_wire(3, edges)))
    A, B, C = run_array(array, 50), run_smooth(new, 50), run_smooth(sm.include(array), 50)
    assert np.abs(A - B).max() < 1e-12 and np.abs(A - C).max() < 1e-12
    assert np.abs(A[-1] - A[0]).max() > 0.05


# ---- (i5) a field is a 1-form: natural in the plot ----

def test_a_composite_field_is_natural_in_the_plot():
    C = sm.compose_seq(sm.include(tensor_arrangements([F1, F2])), sm.include(chain_wire(2)))
    plot, _ = random_plot(C, 4)
    g = lambda t: jnp.array([t[0] * t[1], jnp.sin(t[2]), t[0] - t[2] ** 2, jnp.cos(t[1]), t[0] + t[1] * t[2]])
    t = jnp.array([0.7, -0.4, 1.2])
    at, pull = jax.vjp(g, t)
    assert np.abs(leaves(C.omega(lambda s: plot(g(s)), t)) - leaves(pull(C.omega(plot, at)))).max() < 1e-10


def test_a_field_on_a_field_interface_is_natural_in_the_plot():
    b = _a_field_part()
    g = lambda t: jnp.array([t[0] * t[1], jnp.sin(t[0]) + t[1], t[1] ** 2])
    t = jnp.array([0.7, -0.4])
    at, pull = jax.vjp(g, t)
    assert np.abs(leaves(b.omega(lambda s: _field_plot(g(s)), t)) - leaves(pull(b.omega(_field_plot, at)))).max() < 1e-10


# ---- (i6) a function on a wire, and a load back along it (book Ch 4, ex.learner_currying) ----

def _learner():
    F = lambda q, x: jnp.tanh(q.reshape(2, 2) @ x)
    arr = Arrangement(P=configuration(euclidean(4, 0.1), init=jnp.array([0.5, -0.3, 0.2, 0.9])),
                      in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=2,
                      out_f=F, in_f=lambda q, m, n: jnp.zeros(0),
                      omega=exact(lambda q, m, n: -0.05 * jnp.sum(q * q)), label="learner")
    return F, arr


def test_a_learner_reads_out_its_function_and_evaluation_undoes_it():
    F, arr = _learner()
    phi = sm.include(arr)
    hat = sm.reads_out_its_function(phi)
    assert hat.source == ((), ()) and hat.target == ((R(2),), (Maps(2, 2),))
    q, x = jnp.array([0.1, 0.7, -0.4, 0.3]), jnp.array([0.6, -1.2])
    assert np.allclose(hat.out_f(q, ())[0](x), F(q, x), atol=1e-15)
    back = sm.compose_seq(sm.parallel(hat, sm.identity((), (R(2),))), sm.evaluate(2, 2))
    assert back.label == "(ev . (curry(learner) || id))"
    for seed in range(3):
        assert gap(back, phi, seed) < 1e-12


def test_a_load_passes_through_a_function_wire_inside_a_composite():
    """The curried learner, then evaluation at a datum: an output load ``lam`` reaches the configuration as ``(T_q F(-, x))^T lam``
    plus the learner's own inclination, and the datum's wire as ``(T_x F(q, -))^T lam``. The function wire here is inside the
    composite and the outer wires are Cartesian, so no covector is formed on a function wire; the two tests at the end of this
    file do that."""
    F, arr = _learner()
    hat = sm.reads_out_its_function(sm.include(arr))
    back = sm.compose_seq(sm.parallel(hat, sm.identity((), (R(2),))), sm.evaluate(2, 2))
    q, x, lam = jnp.array([0.1, 0.7, -0.4, 0.3]), jnp.array([0.6, -1.2]), jnp.array([1.0, -2.0])
    position_action, direction_action = sm.interpretation(back)(q)
    (plot_n, w), _ = position_action(sm.at((x,)), sm.zero_form)
    assert np.allclose(plot_n(w)[0], F(q, x), atol=1e-15)
    (xi_N,) = jax.vjp(plot_n, w)[1]((lam,))                  # the load, on the plot the value was sent by
    xi_Q, xi_M, in_m = direction_action(sm.at((x,)), sm.zero_form, xi_N, ())
    (want_q, want_x) = jax.vjp(F, q, x)[1](lam)
    assert np.abs(np.asarray(xi_Q) - np.asarray(want_q - 0.1 * q)).max() < 1e-12
    assert np.abs(np.asarray(xi_M[0]) - np.asarray(want_x)).max() < 1e-12
    assert in_m == ()


# ---- a load on a function wire, and the tensor of composites with states (added after the audit by session fac1f5aa, which
# ---- found that neither was run by a test; its two scratch cases are in the book repo's notes/runs/p76h-audit-fac1f5aa) ----

def _a_closing_arrangement():
    """An arrangement with state ``r`` in ``R^2`` from the interface of ``_a_field_part`` to the closed one: it hands the part a
    function made from the part's own output and ``r``, and has a cost on that output."""
    P = configuration(euclidean(2, 0.1), init=jnp.array([0.8, 0.25]))
    return sm.SmoothArrangement(
        P=P, M_in=(Maps(1),), M_out=(Maps(1),), N_in=(), N_out=(),
        out_f=lambda r, m: (), in_f=lambda r, m, n: (lambda x: r[0] * m[0](x + r[1]) + x[0] ** 2,),
        omega=sm.exact(lambda r, m, n: -0.3 * m[0](jnp.array([r[1]])) ** 2), label="closing")


def test_running_respects_composition_across_a_function_wire():
    """The two interpretations run one after the other, the load handed back on the function wire being a covector on the
    sender's plot, against the inclination field of the composite: the same covector on both configurations. This is the
    law that the dynamics functor preserves a composite, at one position, with a function on the wire between the factors."""
    f, g = _a_field_part(), _a_closing_arrangement()
    q, r = jnp.array([0.4, -0.7]), jnp.array([0.8, 0.25])
    pos_f, dir_f = sm.interpretation(f)(q)
    pos_g, dir_g = sm.interpretation(g)(r)
    shown_f, pushback_f = pos_f(sm.point(()), sm.zero_form)          # f shows a plotted function and a 1-form on functions
    shown_g, _ = pos_g(shown_f, pushback_f)
    xi_r, load_on_wire, handed_in = dir_g(shown_f, pushback_f, sm.closed_tell((shown_g, None))[0], ())
    xi_q, _, _ = dir_f(sm.point(()), sm.zero_form, load_on_wire, handed_in)
    whole = sm.compose_seq(f, g).omega(lambda v: (v, (), ()), jnp.concatenate([q, r]))
    assert np.abs(leaves(load_on_wire)).max() > 0.01              # a covector on (q, ()), the sender's plot's domain
    assert leaves(load_on_wire).shape == (2,)
    assert np.abs(np.concatenate([np.asarray(xi_q), np.asarray(xi_r)]) - np.asarray(whole)).max() < 1e-12
    assert np.abs(np.asarray(whole)).max() > 0.01


def _a_stage(r0):
    """An arrangement with state ``r`` in ``R`` from the interface with a function on each side to itself."""
    P = configuration(euclidean(1, 0.1), init=jnp.array([r0]))
    return sm.SmoothArrangement(
        P=P, M_in=(Maps(1),), M_out=(Maps(1),), N_in=(Maps(1),), N_out=(Maps(1),),
        out_f=lambda r, m: (lambda x: m[0](x) + r[0] * x[0],), in_f=lambda r, m, n: (lambda x: r[0] * n[0](x),),
        omega=sm.exact(lambda r, m, n: -0.5 * (m[0](jnp.array([0.2])) - r[0]) ** 2), label="stage")


def test_the_tensor_of_two_composites_with_states_is_the_composite_of_two_tensors():
    """``(g1 . f1) (x) (g2 . f2)`` against ``(g1 (x) g2) . (f1 (x) f2)``, every part with a state and a function on every wire.
    The two order the configuration differently, ``(f1, g1, f2, g2)`` against ``(f1, f2, g1, g2)``; up to that reordering,
    the isomorphism of integrators the book's "on isomorphism classes" is for, they are equal."""
    f1, f2, g1, g2 = _a_field_part(), _a_field_part(), _a_stage(0.3), _a_stage(-0.6)
    left = sm.parallel(sm.compose_seq(f1, g1), sm.compose_seq(f2, g2))
    right = sm.compose_seq(sm.parallel(f1, f2), sm.parallel(g1, g2))
    assert left.source == right.source and left.target == right.target and left.P.Q_dim == right.P.Q_dim == 6
    to_left = jnp.array([0, 1, 4, 2, 3, 5])                       # right's (f1, f2, g1, g2) read in left's order

    def plot_right(v):
        fn = lambda a, b: (lambda x: a * x[0] ** 2 + jnp.cos(b * x[0]) + a * b)
        return jnp.tanh(v) + jnp.array([0.4, -0.7, 0.2, 0.9, 0.3, -0.6]), (), (fn(v[0], v[1]), fn(v[2], v[5]))

    def plot_left(v):
        q, m, n = plot_right(v)
        return q[to_left], m, n

    v, x = jnp.array([0.3, -1.1, 0.8, 0.2, -0.5, 0.6]), jnp.array([0.9])
    q = plot_right(v)[0]
    for k in (0, 1):
        assert abs(float(left.out_f(q[to_left], ())[k](x)) - float(right.out_f(q, ())[k](x))) < 1e-14
    a, b = leaves(left.omega(plot_left, v)), leaves(right.omega(plot_right, v))
    assert np.abs(a - b).max() < 1e-12 and np.abs(b).max() > 0.1
    # control: without the reordering they differ
    assert np.abs(leaves(left.omega(plot_right, v)) - b).max() > 1e-3
