"""The copier and the comparison arrangement, composed and run (book Chapter 3, ``ex.copying_pushback``; linked from
its last paragraph, "The composite runs in IMC").

For a reader of the book. The arrangements are in ``arrangements/pushback_copying.py``, whose head matches the book's
names to the code's: ``pushback_copy`` is the book's ``copier``, ``pushback_comparator`` its ``compare``. Here every
step is a step of the dynamics functor ``Phi``; the updates written out by hand beside it are the book's formulas,
checked against it, not dynamics of their own. ``spring(initial, rate)`` is the book's small example: ``V = W = R``,
configuration ``q = (x, kappa_c)``, a position and a stiffness, output ``o(q) = x`` and pushback
``pi_c(q, v) = kappa_c (x - v)`` at the anchor position ``v``, with internal inclination ``(-pi_c, 0)``, the spring's
own pull; it returns the plain spring ``<V | W>`` and its copier ``<V | W x V>``. The example a reader can run is in
the module's head.

What the tests check, each of which could fail. ``test_current_input_and_nonlinear_sensitivity_at_look_ahead``: the
copier's update ``eqn.copy_general_update`` on a nonlinear pushback, differentiated at the look-ahead and at the input
the copier currently receives. ``test_composite_step_matches_both_teaching_terms_and_external_loads``: one step of
``compare . (original (x) copier)`` gives the loads of ``eqn.copy_general_step_loads``: ``kappa (y_o - y_c)`` at the
copier's ordinary output, ``kappa (lambda_o - lambda_c)`` at its extra one, pulled back to ``kappa_c`` times it at
position and ``(x - v)`` times it at stiffness, and ``lambda_o + lambda_c`` and ``0`` pushed back at the shared input
and the signal. ``test_teaching_is_descent_on_the_pushback_mismatch``: the added inclination is minus the differential
in ``q`` of ``(1/2) kappa |lambda_o - pi_c(q, v)|^2``, holding ``lambda_o`` and ``v`` fixed.
``test_zero_teaching_retains_the_prescribed_ordinary_response``: with no teaching load the copier steps as the plain
spring. ``test_no_input_recovers_the_same_integrator_under_ordinary_loads``: with ``V = 0`` the copier is its
integrator with output teaching only, as for ``ex.copying_learner``.
``test_drawn_factorization_equals_the_comparator_even_away_from_zero_readout``: the drawn wiring, the circle ``T`` and
``couple`` (``eqn.copy_general_transfer``, ``eqn.copy_general_couple``), composes to ``pushback_comparator``. These
checks establish the stated step, not convergence for every choice of copier.

For agents: the head before 2026-11-13 is in ``notes/P31-B-CH3-HISTORY-2026-11-13.md``.
"""
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.coalgebra import CotCoalg
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.pushback_copying import pushback_copy, pushback_comparator
from arrangements.wiring import compose_seq, parallel_arrangements, trivial_parameter

EMPTY = jnp.zeros(0)


def step(system, x, load, duration, state=None):
    machine = Phi(system)
    if state is not None:
        machine = machine.with_state(state)
    return machine.run_one((EMPTY, trivial_omega(0)), lambda y: (load, x), duration=duration)


def test_current_input_and_nonlinear_sensitivity_at_look_ahead():
    """A stale input, wrong derivative point, or dropped output load fails."""
    response = jnp.array([[.4, .1], [-.2, .6]])
    read = lambda s, t: s+t*jnp.array([.2, -.1])
    update = lambda s, xi, t: s+t*(response @ xi)
    material = CotCoalg(2, jnp.array([.7, -.4]), lambda s: read(s, 1.),
                        lambda s, xi: update(s, xi, 1.), look_ahead=read, timed_step=update)
    def field(q, x):
        return (jnp.array([q[1]+x[0], -q[0]+x[1]]),
                jnp.array([q[0]**2*x[0]+q[1]*x[1], jnp.sin(q[1])*x[0]-q[0]*x[1]]))
    copy = pushback_copy(material, lambda q: jnp.array([q[0]*q[1]]), field, 2, 1)
    for duration in (0., .03, .7):
        for x in (jnp.array([0., 0.]), jnp.array([.8, -.3]), jnp.array([-.5, 1.2])):
            for e in (jnp.zeros(2), jnp.array([.4, -.6])):
                q = read(material.init, duration)
                derivative = jnp.array([[2*q[0]*x[0], x[1]], [-x[1], jnp.cos(q[1])*x[0]]])
                expected = update(material.init, field(q, x)[0]+.9*jnp.array([q[1], q[0]])+derivative.T@e, duration)
                pos, _, actual = step(copy, x, jnp.concatenate((jnp.array([.9]), e)), duration)
                np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(pos[0], jnp.array([q[0]*q[1], 0., 0.]))
                np.testing.assert_allclose(pos[1](x), field(q, x)[1])


def spring(initial, learning):
    rates = jnp.array([1., learning])
    material = CotCoalg(2, jnp.array(initial), lambda s: s,
                        lambda s, xi: s+rates*xi,
                        look_ahead=lambda s, t: s,
                        timed_step=lambda s, xi, t: s+t*rates*xi)
    def field(q, x):
        b = q[1]*(q[0]-x[0])
        return jnp.array([-b, 0.]), jnp.array([b])
    base = Arrangement(material, 0, 0, 1, 1, lambda q, m: q[:1],
                       lambda q, m, x: EMPTY,
                       lambda q, m, x: (field(q, x)[0], EMPTY, field(q, x)[1]), label="spring")
    return base, pushback_copy(material, lambda q: q[:1], field, 1, 1)


def test_composite_step_matches_both_teaching_terms_and_external_loads():
    """The source model is absent from C; the copy still gets a-b now."""
    source, _ = spring((.2, 2.), 0.)
    _, copy = spring((-.3, 4.), .5)
    pair = compose_seq(parallel_arrangements(source, copy), pushback_comparator(1, 1))
    print(pair.label)
    state = pair.P.init
    for x, kappa, duration in ((.8, 1., .02), (-.4, .5, .1), (.2, 0., .2)):
        (so, sc), _ = state
        a, b = so[1]*(so[0]-x), sc[1]*(sc[0]-x)
        e = kappa*(a-b)
        load = jnp.array([.3, -.2])
        expected_o = so+duration*jnp.array([-a+load[0], 0.])
        expected_c = sc+duration*jnp.array([-b+load[1]+kappa*(so[0]-sc[0])+e*sc[1], .5*e*(sc[0]-x)])
        pos, _, state = step(pair, jnp.array([x, kappa]), load, duration, state)
        np.testing.assert_allclose(pos[1](jnp.array([x, kappa])), jnp.array([a+b, 0.]), atol=1e-12)
        np.testing.assert_allclose(state[0][0], expected_o, atol=1e-12)
        np.testing.assert_allclose(state[0][1], expected_c, atol=1e-12)


def test_teaching_is_descent_on_the_pushback_mismatch():
    """Could fail: the teaching load could be pulled back along the wrong map, or at the wrong point, so that the added
    inclination were not minus the differential of the squared mismatch of pushbacks (book ``ex.copying_pushback``)."""
    import jax

    source, _ = spring((.2, 2.), 0.)
    _, copy = spring((-.3, 4.), 1.)
    pi_c = lambda q, v: q[1]*(q[0]-v)
    for x, kappa, duration in ((.8, 1., .02), (-.4, .5, .1), (.2, .7, .3)):
        pair = compose_seq(parallel_arrangements(source, copy), pushback_comparator(1, 1))
        on = step(pair, jnp.array([x, kappa]), jnp.zeros(2), duration)[2]
        off = step(pair, jnp.array([x, 0.]), jnp.zeros(2), duration)[2]
        (so, sc), _ = pair.P.init
        lam_o = pi_c(so, x)
        mismatch = lambda q: .5*kappa*(lam_o-pi_c(q, x))**2
        added = (on[0][1]-off[0][1])/duration
        # the W-teaching kappa (y_o - y_c) lands at position only, through o(q) = x
        added_by_output = jnp.array([kappa*(so[0]-sc[0]), 0.])
        print("added inclination", added-added_by_output, "minus the differential", -jax.grad(mismatch)(sc))
        np.testing.assert_allclose(added-added_by_output, -jax.grad(mismatch)(sc), atol=1e-12)


def test_zero_teaching_retains_the_prescribed_ordinary_response():
    base, copy = spring((.4, 3.), .5)
    for duration in (0., .2, 1.):
        x = jnp.array([-.7])
        plain = step(base, x, jnp.array([.6]), duration)
        equipped = step(copy, x, jnp.array([.6, 0.]), duration)
        np.testing.assert_array_equal(plain[2], equipped[2])


def test_no_input_recovers_the_same_integrator_under_ordinary_loads():
    material = CotCoalg(1, jnp.array([.4]), lambda s: s, lambda s, xi: s+xi,
                        look_ahead=lambda s, t: s+t,
                        timed_step=lambda s, xi, t: s+t*xi)
    copy = pushback_copy(material, lambda q: q**2, lambda q, x: (q, EMPTY), 0, 1)
    for duration in (0., .2, 1.):
        q = material.position_at(material.init, duration)
        pos, _, actual = step(copy, EMPTY, jnp.array([.6]), duration)
        np.testing.assert_allclose(actual, material.step_over(material.init, q+1.2*q, duration))
        assert pos[1](EMPTY).shape == (0,)
        assert copy.P.Q_dim == material.Q_dim


def test_drawn_factorization_equals_the_comparator_even_away_from_zero_readout():
    """Filling the teacher box and composing the drawn wiring must yield C."""
    for n, w in ((2, 3), (2, 0), (0, 2), (0, 0)):
        identity = Arrangement(trivial_parameter(), 2*n, 2*w+n, 2*n, 2*w+n,
                               lambda q, m: m, lambda q, m, x: x, zero_form, label="identity_pair")
        def teacher_field(q, m, x):
            force = x[w]*(x[:w]-x[w+1:])
            return EMPTY, EMPTY, jnp.concatenate((jnp.zeros(w+1), force))
        teacher = Arrangement(trivial_parameter(), 0, 0, 2*w+1, 0,
                              lambda q, m: EMPTY, lambda q, m, x: EMPTY,
                              teacher_field, label="couple")
        def inward(q, m, x):
            plus, minus = x[:n]+x[n]*m[2*w:], x[:n]-x[n]*m[2*w:]
            return jnp.concatenate((plus, minus, m[:w], x[n:n+1], m[w:2*w]))
        wire = Arrangement(trivial_parameter(), 2*n+2*w+1, 2*w+n, n+1, 2*w,
                           lambda q, m: m[:2*w], inward, zero_form, label="drawn_T_and_splits")
        drawn = compose_seq(parallel_arrangements(identity, teacher), wire)
        c = pushback_comparator(n, w)
        for kappa in (0., .7, 1.):
            m = jnp.arange(2*w+n, dtype=float)*.2-.4
            x = jnp.concatenate((jnp.arange(n, dtype=float)-.3, jnp.array([kappa])))
            np.testing.assert_allclose(drawn.out_f(EMPTY, m), c.out_f(EMPTY, m))
            np.testing.assert_allclose(drawn.in_f(EMPTY, m, x), c.in_f(EMPTY, m, x))
            for actual, expected in zip(drawn.omega(EMPTY, m, x), c.omega(EMPTY, m, x)):
                np.testing.assert_allclose(actual, expected, atol=1e-12)
