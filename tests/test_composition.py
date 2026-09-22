"""Functoriality of the dynamics functor ``Phi`` (sec.dynamics_functor; CDLM sec.spring_second_pass).

Operad-functoriality has two legs, and they are tested separately here.

* The **∘-leg** (``test_*_functoriality_*_chain``). The wave/heat chain
  ``wire_K(Part,...,Part)`` is the genuine composite
  ``compose_seq(tensor(Parts), chain_wire(K))`` in ``arr``, so

    ``Phi(compose_seq(tensor(Parts), chain_wire))``  (compose in arr, then Phi)
    ``Phi(tensor(Parts)).then(Phi(chain_wire))``     (Phi each, then compose in pc)

  must agree. This exercises ``PCMorphism.then`` on *non-identity, stateful*
  coalgebras. Since the parameter of a composite is the tensor of parameters
  (state a pair), the two sides now carry states of the *same* shape.

* The **⊗-leg** (``test_laxator_coherence_*``). ``Phi`` is only *lax* monoidal on
  objects: ``Phi(A (x) B)`` carries one one-form on the product interface, while
  ``Phi(A) (x) Phi(B)`` carries one per factor, and only the direct sum
  ``phi_laxator`` goes between them (functors.py). The coherence square

    ``(Phi(A) (x) Phi(B)).post_static(lax_N)  ==  Phi(A (x) B).pre_static(lax_M)``

  is checked on boxes whose interfaces are non-trivial on *both* sides, with
  nonlinear ``out_f``, ``in_f``, forcing, and a nonlinear incoming one-form.

* The **second pass** (``test_second_pass_*``) uses both legs at once: combine the
  ``K`` particle coalgebras by the lax monoidal structure of ``pc``, then
  post-compose the static ``Phi(wire_K)``, and compare with the first pass
  ``Phi(compose_chain(Parts))``.

* **Non-exact forcing** (``test_functoriality_non_exact_sequential``, DESIGN.md §5):
  composition in ``arr`` is pullback-and-sum, and it must agree with composition
  in ``pc`` for forcings that come from no potential -- where CDLM's "sum the
  potentials" has no meaning.
"""

from functools import reduce

import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration, phase
from arrangements.functors import Phi, phi_laxator
from arrangements.polynomial import nest_left, unnest_left
from arrangements.rvect import diagonal
from arrangements.wiring import (
    chain_wire,
    compose_chain,
    compose_seq,
    parallel_arrangements,
    tensor_arrangements,
)

from tests.chain import IN_POS, harmonic_particle, load_regress, phase_flat, conf_flat, conf_state, phase_state


def _step(O, state, in_dir, in_pos=IN_POS):
    return O.with_state(state).run_one(in_pos, lambda _o: in_dir)


def _assert_same_tree(a, b, atol=1e-10):
    """Two pytrees of arrays (nested tuples) agree leaf by leaf."""
    if isinstance(a, tuple):
        assert isinstance(b, tuple) and len(a) == len(b)
        for x, y in zip(a, b):
            _assert_same_tree(x, y, atol)
    else:
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), atol=atol)


# ---------------------------------------------------------------------------
# The compose-leg: composition in arr vs composition in pc.
# ---------------------------------------------------------------------------


def _assert_functorial(parameter, K, m, kappa, phase_):
    parts = [harmonic_particle(parameter, m, kappa) for _ in range(K)]
    first = Phi(compose_chain(parts))                                 # compose in arr, then Phi
    second = Phi(tensor_arrangements(parts)).then(Phi(chain_wire(K)))  # Phi each, compose in pc

    rng = np.random.default_rng(7)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        if phase_:
            s = phase_state(q, jnp.asarray(rng.standard_normal(K)))
        else:
            s = conf_state(q)
        in_dir = (jnp.asarray(rng.standard_normal(1)), jnp.asarray(rng.standard_normal(1)))

        (on1, om1), d1, ns1 = _step(first, s, in_dir)
        (on2, om2), d2, ns2 = _step(second, s, in_dir)

        # same output position and one-form (probe the form at a point)
        np.testing.assert_allclose(np.asarray(on1), np.asarray(on2), atol=1e-10)
        z = jnp.array([0.37])
        np.testing.assert_allclose(np.asarray(om1(z)), np.asarray(om2(z)), atol=1e-10)
        _assert_same_tree(d1, d2)   # same returned direction at the (unit) source interface
        _assert_same_tree(ns1, ns2)  # same updated state, same shape


def test_phase_functoriality_wave_chain():
    _assert_functorial(phase, K=5, m=1.7, kappa=2.3, phase_=True)


def test_conf_functoriality_heat_chain():
    _assert_functorial(configuration, K=5, m=1.5, kappa=0.9, phase_=False)


# ---------------------------------------------------------------------------
# The tensor-leg: the laxator's coherence square, on non-trivial interfaces.
# ---------------------------------------------------------------------------


def _generic_box(parameter, dim_Q, d_out_M, d_in_M, d_out_N, d_in_N, seed, data=None):
    """A box with nonlinear ``out_f``, ``in_f``, exact forcing and non-trivial
    interfaces on both sides -- so every summand of the covector triple is
    exercised. ``data`` (matrices ``A, B, c, sharp``) may be supplied, e.g. from
    the CDLM regression file; otherwise drawn from ``seed`` in CDLM's order."""
    if data is None:
        rng = np.random.default_rng(seed)
        A = jnp.asarray(rng.standard_normal((d_out_N, dim_Q + d_out_M)))
        B = jnp.asarray(rng.standard_normal((d_in_M, dim_Q + d_out_M + d_in_N)))
        c = jnp.asarray(rng.standard_normal(dim_Q + d_out_M + d_in_N))
        sharp = jnp.asarray(0.5 + rng.random(dim_Q))
    else:
        A, B, c, sharp = (jnp.asarray(data[k]) for k in ("A", "B", "c", "sharp"))

    def out_f(q, m_out):
        return jnp.tanh(A @ jnp.concatenate([q, m_out]))

    def in_f(q, m_out, n_in):
        return jnp.sin(B @ jnp.concatenate([q, m_out, n_in]))

    def U_cdlm(q, m_out, n_in):
        z = jnp.concatenate([q, m_out, n_in])
        return jnp.sum(jnp.cos(c * z)) + 0.5 * jnp.sum(z ** 2)

    return Arrangement(
        P=parameter(diagonal(sharp)),
        in_dim_M=d_in_M, out_dim_M=d_out_M, in_dim_N=d_in_N, out_dim_N=d_out_N,
        out_f=out_f, in_f=in_f, omega=exact(lambda q, m, n: -U_cdlm(q, m, n)), label=f"box{seed}",
    )


def _lax_pair(A, B):
    lax_M = phi_laxator([(P.out_dim_M, P.in_dim_M) for P in (A, B)], label="lax_M")
    lax_N = phi_laxator([(P.out_dim_N, P.in_dim_N) for P in (A, B)], label="lax_N")
    lhs = Phi(A).parallel(Phi(B)).post_static(lax_N)          # tensor in pc, then compare
    rhs = Phi(parallel_arrangements(A, B)).pre_static(lax_M)  # tensor in arr, then Phi
    return lhs, rhs


def _assert_laxator_coherence(parameter, phase_):
    A = _generic_box(parameter, 2, 2, 3, 2, 2, seed=1)
    B = _generic_box(parameter, 3, 1, 2, 3, 1, seed=2)
    lhs, rhs = _lax_pair(A, B)
    assert lhs.src_poly == rhs.src_poly and lhs.tgt_poly == rhs.tgt_poly

    d_out_N, d_in_N = A.out_dim_N + B.out_dim_N, A.in_dim_N + B.in_dim_N
    rng = np.random.default_rng(11)
    for _ in range(5):
        # a nested source position: one output value and one one-form per factor
        def field(d):
            w = jnp.asarray(rng.standard_normal(d))
            return lambda z: jnp.cos(3.0 * z) + w * z ** 2
        in_pos = ((jnp.asarray(rng.standard_normal(A.out_dim_M)), field(A.in_dim_M)),
                  (jnp.asarray(rng.standard_normal(B.out_dim_M)), field(B.in_dim_M)))
        in_dir = (jnp.asarray(rng.standard_normal(d_out_N)),
                  jnp.asarray(rng.standard_normal(d_in_N)))

        qA = jnp.asarray(rng.standard_normal(A.P.Q_dim))
        qB = jnp.asarray(rng.standard_normal(B.P.Q_dim))
        if phase_:
            pA = jnp.asarray(rng.standard_normal(A.P.Q_dim))
            pB = jnp.asarray(rng.standard_normal(B.P.Q_dim))
            s = ((qA, pA), (qB, pB))
        else:
            s = (qA, qB)

        (on1, om1), d1, ns1 = _step(lhs, s, in_dir, in_pos)
        (on2, om2), d2, ns2 = _step(rhs, s, in_dir, in_pos)

        # same output position, and the same (flat) one-form on the product
        np.testing.assert_allclose(np.asarray(on1), np.asarray(on2), atol=1e-10)
        z = jnp.asarray(rng.standard_normal(d_in_N))
        np.testing.assert_allclose(np.asarray(om1(z)), np.asarray(om2(z)), atol=1e-10)
        _assert_same_tree(d1, d2)   # same returned directions, factor by factor
        _assert_same_tree(ns1, ns2)  # same updated state (a pair on both sides)


def test_laxator_coherence_phase():
    _assert_laxator_coherence(phase, phase_=True)


def test_laxator_coherence_conf():
    _assert_laxator_coherence(configuration, phase_=False)


def test_laxator_is_not_strong():
    """The laxator is not invertible: a one-form on the product interface is
    not a tuple of one-forms on the factors (which is why ``Phi`` is only lax)."""
    lax = phi_laxator([(1, 2), (1, 2)])
    m = jnp.zeros(1)
    omega = lambda z: jnp.cos(z)
    flat, field = lax.on_position(((m, omega), (m, omega)))
    assert flat.shape == (2,)
    # the image consists of the direct sums: no cross-factor coupling survives
    z = jnp.array([0.3, 0.7, -0.2, 1.1])
    np.testing.assert_allclose(
        np.asarray(field(z)),
        np.asarray(jnp.concatenate([omega(z[:2]), omega(z[2:])])),
        atol=1e-12,
    )
    coupled = lambda z: jnp.stack([z[0] * z[3], z[1], z[2], z[3]])  # not a direct sum
    assert not np.allclose(np.asarray(coupled(z)), np.asarray(field(z)))


# ---------------------------------------------------------------------------
# The second pass (CDLM sec.spring_second_pass), both legs at once.
# ---------------------------------------------------------------------------


def _second_pass(parts):
    """``Phi(wire_K)(Phi(Part),...,Phi(Part))``: combine the ``K`` coalgebras via the
    lax monoidal structure of ``pc``, then post-compose the static ``Phi(wire_K)``."""
    K = len(parts)
    tensored = reduce(lambda a, b: a.parallel(b), [Phi(P) for P in parts])
    lax_N = phi_laxator([(P.out_dim_N, P.in_dim_N) for P in parts], label="lax_N")
    return tensored.post_static(lax_N).then(Phi(chain_wire(K)))


def _assert_second_pass(parameter, K, m, kappa, phase_):
    parts = [harmonic_particle(parameter, m, kappa) for _ in range(K)]
    first = Phi(compose_chain(parts))
    second = _second_pass(parts)

    # the source of the second pass is the K-fold tensor of the unit interface,
    # so its position is the K-fold nesting of the unit's single position
    in_pos_2 = nest_left([IN_POS] * K)
    rng = np.random.default_rng(19)
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        s = phase_state(q, jnp.asarray(rng.standard_normal(K))) if phase_ else conf_state(q)
        in_dir = (jnp.asarray(rng.standard_normal(1)), jnp.asarray(rng.standard_normal(1)))

        (on1, om1), _, ns1 = _step(first, s, in_dir)
        (on2, om2), _, ns2 = _step(second, s, in_dir, in_pos_2)

        np.testing.assert_allclose(np.asarray(on1), np.asarray(on2), atol=1e-10)
        z = jnp.array([0.41])
        np.testing.assert_allclose(np.asarray(om1(z)), np.asarray(om2(z)), atol=1e-10)
        _assert_same_tree(ns1, ns2)


def test_second_pass_wave_chain():
    _assert_second_pass(phase, K=5, m=1.7, kappa=2.3, phase_=True)


def test_second_pass_heat_chain():
    _assert_second_pass(configuration, K=5, m=1.5, kappa=0.9, phase_=False)


# ---------------------------------------------------------------------------
# Non-exact forcing: pullback-and-sum in arr agrees with composition in pc.
# ---------------------------------------------------------------------------


def _non_exact_box(parameter, dim_Q, d_out_M, d_in_M, d_out_N, d_in_N, seed):
    """Like ``_generic_box`` but with a forcing given directly by components that
    come from no potential (its Jacobian is not symmetric)."""
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((d_out_N, dim_Q + d_out_M)))
    B = jnp.asarray(rng.standard_normal((d_in_M, dim_Q + d_out_M + d_in_N)))
    d = dim_Q + d_out_M + d_in_N
    C = jnp.asarray(rng.standard_normal((d, d)))  # a generic, non-symmetric matrix

    def out_f(q, m_out):
        return jnp.tanh(A @ jnp.concatenate([q, m_out]))

    def in_f(q, m_out, n_in):
        return jnp.sin(B @ jnp.concatenate([q, m_out, n_in]))

    def omega(q, m_out, n_in):
        z = jnp.concatenate([q, m_out, n_in])
        xi = jnp.sin(C @ z) + jnp.roll(z, 1) * z  # not closed: d(omega) != 0
        return xi[:dim_Q], xi[dim_Q:dim_Q + d_out_M], xi[dim_Q + d_out_M:]

    return Arrangement(
        P=parameter(diagonal(jnp.asarray(0.5 + rng.random(dim_Q)))),
        in_dim_M=d_in_M, out_dim_M=d_out_M, in_dim_N=d_in_N, out_dim_N=d_out_N,
        out_f=out_f, in_f=in_f, omega=omega, label=f"nonexact{seed}",
    )


def _assert_non_exact_sequential(parameter, phase_):
    f = _non_exact_box(parameter, 2, 2, 3, 3, 2, seed=5)   # L = <3|2> -> M = <2|3>
    g = _non_exact_box(parameter, 3, 3, 2, 2, 1, seed=6)   # M = <2|3> -> N = <1|2>
    first = Phi(compose_seq(f, g))
    second = Phi(f).then(Phi(g))

    rng = np.random.default_rng(23)
    for _ in range(5):
        w = jnp.asarray(rng.standard_normal(f.in_dim_M))
        in_pos = (jnp.asarray(rng.standard_normal(f.out_dim_M)), lambda z, w=w: jnp.cos(3.0 * z) + w * z ** 2)
        in_dir = (jnp.asarray(rng.standard_normal(g.out_dim_N)), jnp.asarray(rng.standard_normal(g.in_dim_N)))
        qf, qg = (jnp.asarray(rng.standard_normal(P.P.Q_dim)) for P in (f, g))
        if phase_:
            s = ((qf, jnp.asarray(rng.standard_normal(f.P.Q_dim))), (qg, jnp.asarray(rng.standard_normal(g.P.Q_dim))))
        else:
            s = (qf, qg)

        (on1, om1), d1, ns1 = _step(first, s, in_dir, in_pos)
        (on2, om2), d2, ns2 = _step(second, s, in_dir, in_pos)
        np.testing.assert_allclose(np.asarray(on1), np.asarray(on2), atol=1e-10)
        z = jnp.asarray(rng.standard_normal(g.in_dim_N))
        np.testing.assert_allclose(np.asarray(om1(z)), np.asarray(om2(z)), atol=1e-10)
        _assert_same_tree(d1, d2)
        _assert_same_tree(ns1, ns2)


def test_functoriality_non_exact_sequential_conf():
    _assert_non_exact_sequential(configuration, phase_=False)


def test_functoriality_non_exact_sequential_phase():
    _assert_non_exact_sequential(phase, phase_=True)


# ---------------------------------------------------------------------------
# Regression against CDLM (DESIGN.md §3).
# ---------------------------------------------------------------------------


def test_regression_matches_cdlm():
    """``regress/composition.npz``: the first pass on CDLM's random states and the
    laxator's lhs on CDLM's boxes; xi_N and incoming one-forms negated, returned
    covectors and published one-forms negated, states identical."""
    R = load_regress("composition")

    for tag, parameter, phase_ in (("wave", phase, True), ("heat", configuration, False),
                                   ("wave2", phase, True), ("heat2", configuration, False)):
        K, m, kappa = R[f"{tag}_params"]; K = int(K)
        first = Phi(compose_chain([harmonic_particle(parameter, m, kappa) for _ in range(K)]))
        z = jnp.array([0.37]) if tag in ("wave", "heat") else jnp.array([0.41])
        for i in range(R[f"{tag}_q"].shape[0]):
            q, p = jnp.asarray(R[f"{tag}_q"][i]), jnp.asarray(R[f"{tag}_p"][i])
            s = phase_state(q, p) if phase_ else conf_state(q)
            in_dir = (-jnp.asarray(R[f"{tag}_xi"][i]), jnp.asarray(R[f"{tag}_qprev"][i]))
            (on, om), d, ns = _step(first, s, in_dir)
            np.testing.assert_allclose(on, R[f"{tag}_on"][i], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(om(z), -R[f"{tag}_omz"][i], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(d[0], -R[f"{tag}_d_xi"][i], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(d[1], R[f"{tag}_d_in"][i], rtol=1e-10, atol=1e-12)
            if phase_:
                nq, np_ = phase_flat(ns, K)
                np.testing.assert_allclose(np_, R[f"{tag}_np"][i], rtol=1e-10, atol=1e-12)
            else:
                nq = conf_flat(ns, K)
            np.testing.assert_allclose(nq, R[f"{tag}_nq"][i], rtol=1e-10, atol=1e-12)

    for tag, parameter, phase_ in (("phase", phase, True), ("conf", configuration, False)):
        A = _generic_box(parameter, 2, 2, 3, 2, 2, seed=1, data={k: R[f"lax_A_{k}"] for k in ("A", "B", "c", "sharp")})
        B = _generic_box(parameter, 3, 1, 2, 3, 1, seed=2, data={k: R[f"lax_B_{k}"] for k in ("A", "B", "c", "sharp")})
        lhs, _ = _lax_pair(A, B)
        g = lambda k: R[f"lax_{tag}_{k}"]
        for i in range(g("qA").shape[0]):
            def field(w):  # CDLM's incoming field, negated: now a forcing
                w = jnp.asarray(w)
                return lambda z: -(jnp.cos(3.0 * z) + w * z ** 2)
            in_pos = ((jnp.asarray(g("mA")[i]), field(g("wA")[i])), (jnp.asarray(g("mB")[i]), field(g("wB")[i])))
            in_dir = (-jnp.asarray(g("xi")[i]), jnp.asarray(g("inn")[i]))
            qA, qB, pA, pB = (jnp.asarray(g(k)[i]) for k in ("qA", "qB", "pA", "pB"))
            s = ((qA, pA), (qB, pB)) if phase_ else (qA, qB)
            (on, om), d, ns = _step(lhs, s, in_dir, in_pos)
            np.testing.assert_allclose(on, g("on")[i], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(om(jnp.asarray(g("z")[i])), -g("omz")[i], rtol=1e-10, atol=1e-12)
            (dA_xi, dA_in), (dB_xi, dB_in) = unnest_left(d, 2)
            np.testing.assert_allclose(dA_xi, -g("dA_xi")[i], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(dB_xi, -g("dB_xi")[i], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(dA_in, g("dA_in")[i], rtol=1e-10, atol=1e-12)
            np.testing.assert_allclose(dB_in, g("dB_in")[i], rtol=1e-10, atol=1e-12)
            if phase_:
                np.testing.assert_allclose(ns[0][0], g("nqA")[i], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(ns[0][1], g("npA")[i], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(ns[1][0], g("nqB")[i], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(ns[1][1], g("npB")[i], rtol=1e-10, atol=1e-12)
            else:
                np.testing.assert_allclose(ns[0], g("nqA")[i], rtol=1e-10, atol=1e-12)
                np.testing.assert_allclose(ns[1], g("nqB")[i], rtol=1e-10, atol=1e-12)
