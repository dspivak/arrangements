"""A copy and its original (book Ch 3 ``ex.copying_learner``): ``copying.copy_pair``.

Two ``osc`` of one kind (book Ch 3 ``ex.oscillator``), the original at angular frequency 0.30 and the copy at 0.22, the
copy coupled one way to the original while the external input ``K`` is 1. Each state is ``(alpha, nu)``, the angle on
the circle and the angular frequency. Each test says whether it could have come out otherwise.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.copying import copy_pair, copy_pair_q, driven_step

jax.config.update("jax_enable_x64", True)

NU_ORIG, NU_COPY, LAM, ETA = 0.30, 0.22, 0.2, 0.04


def _init(nu):
    return np.array([0.0, nu])


def _sent(s):
    """What an ``osc`` in state ``(alpha, nu)`` sends at a step of duration 1: ``sin(alpha + nu)``."""
    return np.sin(s[0] + s[1])


def _run(Ks):
    W = copy_pair(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    step = driven_step(W, check_inputs=[[1.0]] * 3)
    s, xs, ys = W.P.init, [], []
    for K in Ks:
        o, c = copy_pair_q(W, s)
        xs.append(_sent(o))
        ys.append(_sent(c))
        s = step(s, jnp.array([float(K)]))
    o, c = copy_pair_q(W, s)
    return o, c, np.array(xs), np.array(ys)


def _crossing_rate(y):
    """The angular frequency read off the zero crossings of ``y``, sent one value per unit of duration."""
    z = np.where(np.diff(np.sign(y)) != 0)[0]
    return np.pi / np.mean(np.diff(z))


def test_the_copy_takes_the_original_angular_frequency_and_keeps_it():
    """Could fail: the angular frequency could stop short, or drift once the coupling is off."""
    o, c, xs, ys = _run([1.0] * 4000 + [0.0] * 3000)
    mismatch = np.abs(xs[3800:4000] - ys[3800:4000]).max()
    print("copy nu", c[1], "by crossings after", _crossing_rate(ys[4100:]), "mismatch", mismatch)
    assert abs(c[1] - NU_ORIG) < 1e-6 and mismatch < 1e-9
    assert abs(_crossing_rate(ys[4100:]) - NU_ORIG) < 1e-3


def test_the_original_is_untouched():
    """Could fail: a coupling that pushed back would move the original's angle or angular frequency."""
    o, c, xs, _ = _run([1.0] * 4000 + [0.0] * 3000)
    free = np.sin(NU_ORIG * (np.arange(7000) + 1))
    print("original nu", o[1], "max distance from a free sine", np.abs(xs - free).max())
    assert o[1] == NU_ORIG
    assert np.abs(xs - free).max() < 1e-9


def test_uncoupled_nothing_is_copied():
    """Could fail: the copy could drift toward the original without the coupling."""
    _, c, _, ys = _run([0.0] * 3000)
    print("copy nu uncoupled", c[1], "by crossings", _crossing_rate(ys[100:]))
    assert c[1] == NU_COPY and abs(_crossing_rate(ys[100:]) - NU_COPY) < 1e-3


def test_without_the_angle_rate_nothing_locks():
    """Could fail: the angular frequency's own response could be enough to lock on. Without the angle rate ``lam`` the
    copy's angle swings around the original's like an undamped spring (book Ch 3 ``ex.copying_learner``)."""
    W = copy_pair(0.0, ETA, _init(NU_ORIG), _init(NU_COPY))
    step = driven_step(W)
    s = W.P.init
    for _ in range(4000):
        s = step(s, jnp.array([1.0]))
    _, c = copy_pair_q(W, s)
    print("lam = 0: copy nu after 4000 coupled steps", c[1])
    assert abs(c[1] - NU_ORIG) > 1e-2


def test_the_exported_copier_sends_both_values_and_steps_as_the_closed_one():
    """Could fail: exporting the values could change the step, or send something other than ``x`` and ``y``."""
    from arrangements.copying import copy_pair_out
    from arrangements.functors import Phi
    from arrangements.interpretation import trivial_omega

    W = copy_pair(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    Wo = copy_pair_out(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    assert (Wo.in_dim_M, Wo.out_dim_M, Wo.in_dim_N, Wo.out_dim_N) == (0, 0, 1, 2)
    step = driven_step(W)
    s, so = W.P.init, Wo.P.init
    for K in [1.0, 1.0, 0.5, 0.0, 1.0]:
        o, c = copy_pair_q(W, s)
        out, _, so = Phi(Wo).with_state(so).run_one(
            (jnp.zeros(0), trivial_omega(0)), lambda _o, K=K: (jnp.zeros(2), jnp.array([K])))
        s = step(s, jnp.array([K]))
        print("outputs", np.asarray(out[0]), "values", (_sent(o), _sent(c)))
        assert np.allclose(np.asarray(out[0]), [_sent(o), _sent(c)], atol=1e-12)
        assert np.allclose(np.asarray(Wo.P.position(so)), np.asarray(W.P.position(s)), atol=1e-12)


def test_the_books_copier_steps_as_copy_pair_through_the_whole_run():
    """Could fail: ``couple``'s pushback reaches the copy through the split of ``y`` and ``copier_wire``'s backward map, not
    through a contact on the wire, so a wrong order of ``couple``'s inputs, or a split that dropped or doubled the force,
    would move the copy elsewhere or touch the original (book Ch 3 ``ex.copying_learner``)."""
    from arrangements.copying import copier_q, copier_step, copy_pair_out

    Ks = [1.0] * 4000 + [0.0] * 3000
    W = copy_pair(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    Wo = copy_pair_out(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    step = driven_step(W)
    step_o = copier_step(Wo, check_inputs=[[1.0], [0.5], [0.0]])
    s, so, gap = W.P.init, Wo.P.init, 0.0
    for K in Ks:
        s, so = step(s, jnp.array([K])), step_o(so, jnp.array([K]))
        gap = max(gap, max(np.abs(a - b).max() for a, b in zip(copy_pair_q(W, s), copier_q(Wo, so))))
    o, c = copier_q(Wo, so)
    print("largest state difference over 7000 steps", gap, "copier's copy nu", c[1], "original nu", o[1])
    assert gap < 1e-12
    assert abs(c[1] - NU_ORIG) < 1e-6 and o[1] == NU_ORIG


def test_osc_has_effort_field_zero_and_steps_as_the_book_says():
    """Could fail: the turning could sit in the effort field, or the force could reach the matter other than as
    ``xi cos q`` at the look-ahead ``q = alpha + tau nu`` (book Ch 3 ``eqn.osc_step``), at durations other than 1 too."""
    from arrangements.copying import osc

    s = np.array([0.4, 0.3])
    O = osc(LAM, ETA, s)
    xi_q, _, _ = O.omega(O.P.position(jnp.asarray(s)), jnp.zeros(0), jnp.zeros(0))
    assert np.all(np.asarray(xi_q) == 0)
    alpha, nu = s
    for xi, tau in [(0.0, 1.0), (0.8, 1.0), (-1.3, 0.5), (0.4, 2.0)]:
        q = alpha + tau * nu
        assert np.allclose(np.asarray(O.P.position_at(jnp.asarray(s), tau)), [q], atol=1e-15)
        s_new = O.P.step_over(jnp.asarray(s), jnp.array([xi * np.cos(q)]), tau)
        book = np.array([q + tau * LAM * xi * np.cos(q), nu + tau * ETA * xi * np.cos(q)])
        print("tau", tau, "xi", xi, "library", np.asarray(s_new), "book", book)
        assert np.allclose(np.asarray(s_new), book, atol=1e-14)


def test_shorter_steps_land_on_the_same_wave():
    """Could fail: steps of duration 1/2 or 1/100 could land somewhere other than one step of duration 1 does, at a
    high angular frequency too (book ``exc.osc_sine`` (a); David, 2026-10-20: "a smaller duration means we're asking
    for more precision")."""
    from arrangements.copying import oscillator

    for nu in [0.3, 10.0]:
        P = oscillator(LAM, ETA, np.array([0.4, nu]))
        one = P.step_over(P.init, jnp.zeros(1), 1.0)
        for tau in [0.5, 0.01]:
            s = P.init
            for _ in range(round(1 / tau)):
                s = P.step_over(s, jnp.zeros(1), tau)
            print("nu", nu, "tau", tau, "lands at", np.asarray(s), "one step", np.asarray(one))
            assert np.allclose(np.asarray(s), np.asarray(one), atol=1e-12)


def test_angular_frequencies_two_pi_apart_differ_only_at_shorter_steps():
    """Could fail: two angular frequencies ``2 pi`` apart could send different values at steps of duration 1, or the
    same ones at duration 1/2 (book ``exc.osc_sine`` (b))."""
    from arrangements.copying import osc

    nu = 0.3
    a, b = osc(LAM, ETA, np.array([0.0, nu])), osc(LAM, ETA, np.array([0.0, nu + 2 * np.pi]))
    for tau, same in [(1.0, True), (0.5, False)]:
        sa, sb, va, vb = a.P.init, b.P.init, [], []
        for _ in range(6):
            va.append(float(a.out_f(a.P.position_at(sa, tau), None)[0]))
            vb.append(float(b.out_f(b.P.position_at(sb, tau), None)[0]))
            sa, sb = a.P.step_over(sa, jnp.zeros(1), tau), b.P.step_over(sb, jnp.zeros(1), tau)
        print("tau", tau, "values", np.round(va, 4), np.round(vb, 4))
        assert np.allclose(va, vb, atol=1e-12) == same
