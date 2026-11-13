"""One oscillator copies another (book Chapter 3, ``ex.copying_learner``, with ``ex.oscillator`` and
``exc.osc_sine``; linked from the paragraph "Run it at duration 1").

For a reader of the book. ``copying.osc(lam, eta, (alpha, nu), amp=1)`` is the book's ``osc : I -> <0 | 1>``
(``ex.oscillator``): it stores an angle ``alpha`` and an angular frequency ``nu``; told a duration ``tau`` it reads out
the angle ``q = alpha + tau nu`` and sends ``amp sin q``; receiving a load ``xi`` at its output it updates to
``(q + tau lam xi amp cos q, nu + tau eta xi amp cos q)`` (``eqn.osc_step``), ``lam`` the book's ``lambda`` and ``eta``
its ``eta``. ``copying.couple()`` is the book's ``couple : I -> <3 | 0>``: it receives the original's value ``x``, the
signal ``kappa`` and the copier's value ``y``, in that order, and pushes back ``kappa (x - y)`` at ``y`` and ``0`` at
``x`` and at ``kappa``. ``copying.copy_pair_out(lam, eta, original, copier)`` is the book's whole arrangement
``copier : I -> <1 | 2>`` of ``eqn.copier_wd``: the two ``osc`` and ``couple`` in the boxes of ``copier_wire``, the
signal ``kappa`` arriving at its input and both values ``(x, y)`` sent out. ``copier_step`` is one step of the dynamics
functor at duration 1, told ``kappa`` and the load ``0`` at the outputs; ``copier_q`` reads the two oscillators'
states ``(alpha, nu)``, the original's first.

The paragraph's run, from the library's folder with it on the path, as the README's *Run it* says
(``PYTHONPATH=. .venv/bin/python example.py``):

    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from arrangements.copying import copy_pair_out, copier_step, copier_q
    C = copy_pair_out(0.2, 0.04, [0.0, 0.3], [0.0, 0.22])
    step, s = copier_step(C), C.P.init
    for kappa in [1.0] * 4000 + [0.0] * 3000:
        s = step(s, jnp.array([kappa]))
    print(copier_q(C, s))
    # (array([2.1e+03, 3.0e-01]), array([2.1e+03, 3.0e-01])): the original turned 7000 * 0.3 = 2100 radians at
    # angular frequency 0.3, and the copier has come to the same angle and the same angular frequency

What the tests check, each of which could fail. With ``kappa = 1`` for 4000 steps the copier comes to send what the
original sends, at the original's angular frequency, and with ``kappa = 0`` for 3000 more it keeps that angular
frequency, read both from its state and from the zero crossings of what it sends; the original sends the same free
sine wave throughout; with ``kappa = 0`` from the start the copier stays at 0.22; without the angle's rate ``lam``
nothing locks. The book's ``copier`` steps as ``copying.copy_pair``, the same two ``osc`` closed by the one-way contact
with ``kappa`` as an input and nothing sent out. ``osc``'s step is ``eqn.osc_step`` at durations other than 1 too, its
inclination field is 0, shorter steps land on the same wave (``exc.osc_sine`` (a)), and angular frequencies ``2 pi``
apart send the same values at duration 1 and not at duration 1/2 (``exc.osc_sine`` (b)).

For agents: the heads before 2026-11-13 are in ``notes/P31-B-CH3-HISTORY-2026-11-13.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.copying import copier_q, copier_step, copy_pair, copy_pair_out, copy_pair_q, driven_step

jax.config.update("jax_enable_x64", True)

NU_ORIG, NU_COPY, LAM, ETA = 0.30, 0.22, 0.2, 0.04


def _init(nu):
    return np.array([0.0, nu])


def _sent(C, s):
    """The values ``(x, y)`` the book's ``copier`` sends at its two outputs at a step of duration 1 from state ``s``."""
    return np.asarray(C.out_f(C.P.position(s), jnp.zeros(0)))


def _run(kappas, lam=LAM):
    """The book's ``copier`` told the signal ``kappas[k]`` at step ``k``; returns both states and what was sent."""
    C = copy_pair_out(lam, ETA, _init(NU_ORIG), _init(NU_COPY))
    step = copier_step(C, check_inputs=[[1.0], [0.0]])
    s, sent = C.P.init, []
    for kappa in kappas:
        sent.append(_sent(C, s))
        s = step(s, jnp.array([float(kappa)]))
    o, c = copier_q(C, s)
    sent = np.array(sent)
    return o, c, sent[:, 0], sent[:, 1]


def _crossing_rate(y):
    """The angular frequency read off the zero crossings of ``y``, sent one value per unit of duration."""
    z = np.where(np.diff(np.sign(y)) != 0)[0]
    return np.pi / np.mean(np.diff(z))


def test_the_copier_takes_the_original_angular_frequency_and_keeps_it():
    """Could fail: the angular frequency could stop short, or drift once the signal is 0."""
    o, c, xs, ys = _run([1.0] * 4000 + [0.0] * 3000)
    mismatch = np.abs(xs[3800:4000] - ys[3800:4000]).max()
    print("copier nu", c[1], "by crossings after", _crossing_rate(ys[4100:]), "mismatch", mismatch)
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
    """Could fail: the copier could drift toward the original with the signal at 0."""
    _, c, _, ys = _run([0.0] * 3000)
    print("copier nu uncoupled", c[1], "by crossings", _crossing_rate(ys[100:]))
    assert c[1] == NU_COPY and abs(_crossing_rate(ys[100:]) - NU_COPY) < 1e-3


def test_without_the_angle_rate_nothing_locks():
    """Could fail: the angular frequency's own response could be enough to lock on. Without the angle's rate ``lam``
    the copier's angle swings around the original's like an undamped spring."""
    _, c, _, _ = _run([1.0] * 4000, lam=0.0)
    print("lam = 0: copier nu after 4000 steps with the signal at 1", c[1])
    assert abs(c[1] - NU_ORIG) > 1e-2


def test_the_books_copier_steps_as_copy_pair_through_the_whole_run():
    """Could fail: ``couple``'s pushback reaches the copier through the split of ``y`` and ``copier_wire``'s backward
    map, not through a contact on the wire, so a wrong order of ``couple``'s inputs, or a split that dropped or doubled
    the load, would move the copier elsewhere or touch the original."""
    Ks = [1.0] * 4000 + [0.0] * 3000
    W = copy_pair(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    C = copy_pair_out(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    step = driven_step(W, check_inputs=[[1.0]] * 3)
    step_c = copier_step(C, check_inputs=[[1.0], [0.5], [0.0]])
    s, sc, gap = W.P.init, C.P.init, 0.0
    for K in Ks:
        s, sc = step(s, jnp.array([K])), step_c(sc, jnp.array([K]))
        gap = max(gap, max(np.abs(a - b).max() for a, b in zip(copy_pair_q(W, s), copier_q(C, sc))))
    print("largest state difference over 7000 steps", gap)
    assert gap < 1e-12


def test_the_copier_sends_both_values_as_the_dynamics_functor_does():
    """Could fail: the compiled step or the outputs read here could differ from the dynamics functor's own step."""
    from arrangements.functors import Phi
    from arrangements.interpretation import trivial_omega

    C = copy_pair_out(LAM, ETA, _init(NU_ORIG), _init(NU_COPY))
    assert (C.in_dim_M, C.out_dim_M, C.in_dim_N, C.out_dim_N) == (0, 0, 1, 2)
    step = copier_step(C)
    s, s_phi = C.P.init, C.P.init
    for K in [1.0, 1.0, 0.5, 0.0, 1.0]:
        o, c = copier_q(C, s)
        out, _, s_phi = Phi(C).with_state(s_phi).run_one(
            (jnp.zeros(0), trivial_omega(0)), lambda _o, K=K: (jnp.zeros(2), jnp.array([K])))
        sent = _sent(C, s)
        s = step(s, jnp.array([K]))
        print("functor's outputs", np.asarray(out[0]), "read here", sent, "sines", (np.sin(o.sum()), np.sin(c.sum())))
        assert np.allclose(np.asarray(out[0]), sent, atol=1e-12)
        assert np.allclose(sent, [np.sin(o.sum()), np.sin(c.sum())], atol=1e-12)
        assert np.allclose(np.asarray(C.P.position(s_phi)), np.asarray(C.P.position(s)), atol=1e-12)


def test_osc_has_inclination_field_zero_and_steps_as_the_book_says():
    """Could fail: the turning could sit in the inclination field, or the load could reach the integrator other than as
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
    high angular frequency too (book ``exc.osc_sine`` (a))."""
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
