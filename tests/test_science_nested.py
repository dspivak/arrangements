"""The science communities are well-factored and equal their flat forms (book task well-factored-2ac0e890,
David, 2026-11-23: "good factorizations should have 2-7 boxes inside").

``science_native.wire_boxes`` with ``groups`` puts the world (and, in test (i), the incumbent reader and
the excuse fund) beside one box of the holders, bettors or dependents, nested seven to a block
(``nest_groups``). The built composite computes by the flat composite's maps (``nesting.run_flat``);
each check here: every box holds 2 to 7 things; the configuration keeps the boxes' order (world clock
first, then each holder's state as before); the nested composite's own maps equal the built (flat) maps
near the start and over closed steps.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from arrangements import science_failure as sf  # noqa: E402
from arrangements import science_incumbent as si  # noqa: E402
from arrangements import science_native as nv  # noqa: E402
from arrangements import science_payment as sp  # noqa: E402
from arrangements.functors import closed_step  # noqa: E402
from arrangements.nesting import box_counts, carry_state  # noqa: E402
from arrangements.wiring import compose_seq  # noqa: E402

M, N, T = 10, 3, 6


def _own_maps(built):
    """The built composite with its nested construction's own maps (not the flat ones ``run_flat`` gave it)."""
    _, inner, u = built.construction
    return compose_seq(inner, u)


def _check(built, steps=8):
    counts = box_counts(built)
    assert counts and all(2 <= k <= 7 for _, k in counts), counts
    nested = _own_maps(built)
    q0 = built.P.position(built.P.init)
    rng = np.random.default_rng(7)
    e = jnp.zeros(0)
    for _ in range(3):
        q = q0 + 0.01 * jnp.asarray(rng.normal(size=q0.shape))
        for x, y in zip(built.omega(q, e, e), nested.omega(q, e, e)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-12)
    s_f, s_n = built.P.init, carry_state(built.P.init, nested.P.init)
    step_f, step_n = closed_step(built, duration=.1), closed_step(nested, duration=.1)
    for _ in range(steps):
        s_f, s_n = step_f(s_f), step_n(s_n)
    np.testing.assert_allclose(np.asarray(built.P.position(s_f)), np.asarray(nested.P.position(s_n)), atol=1e-10)
    return counts


def _ring(m):
    A = np.zeros((m, m), int)
    for j in range(m):
        A[j, (j - 1) % m] = A[j, (j + 1) % m] = 1
    return A


@pytest.mark.parametrize("arm", ["agree", "evidence"])
def test_community_native(arm):
    rng = np.random.default_rng(1)
    Y0 = jnp.asarray(rng.normal(size=(M, N)))
    X = jnp.asarray(rng.normal(size=(T, M)))
    built = nv.community_native(Y0, X, 0.5 * _ring(M), .3, 1., jnp.asarray([-1., 0., 1.]), arm)
    counts = _check(built)
    assert counts[0] == ("community", 2)                # world, holders
    np.testing.assert_allclose(np.asarray(built.P.position(built.P.init))[1:].reshape(M, N), np.asarray(Y0))


@pytest.mark.parametrize("arm", ["noprov", "lag", "journal"])
def test_summaries_native(arm):
    rng = np.random.default_rng(2)
    built = nv.summaries_native(jnp.asarray(rng.normal(size=(T, M))), _ring(M), jnp.asarray([-1., 0., 1.]), arm, 4)
    counts = _check(built)
    assert counts[0] == ("summaries", 3 if arm == "journal" else 2)


def _slices(m, n):
    return [[j % n] for j in range(m)]


def test_conclusions_native():
    rng = np.random.default_rng(3)
    built = nv.conclusions_native(jnp.asarray(rng.normal(size=(T, M))), _ring(M), jnp.asarray([-1., 0., 1.]),
                                  _slices(M, N), .05, 2, "both", "euc", .05, .5, 2)
    assert _check(built)[0] == ("conclusions", 2)


@pytest.mark.parametrize("arm", ["silent", "lower", "evidence"])
def test_failure_community(arm):
    rng = np.random.default_rng(4)
    built = sf.failure_community(jnp.asarray(rng.normal(size=(T, M))), _ring(M), jnp.asarray([-1., 0., 1.]),
                                 _slices(M, N), .05, 2, arm, .05, .5, 2)
    assert _check(built)[0] == ("failure community", 2)


def test_market_and_gated():
    rng = np.random.default_rng(5)
    Y0 = jnp.asarray(rng.normal(size=(M, sp.N_HORSES)))
    X1h = jax.nn.one_hot(jnp.asarray(rng.integers(0, sp.N_HORSES, T)), sp.N_HORSES, dtype=float)
    W = jnp.asarray(sp.ring(M))
    built = sp.market(Y0, X1h, W, .3, .2, 1., .1)
    assert _check(built)[0] == ("market", 2)
    gated = sp.market_gated(Y0, jnp.ones(M), X1h, W, .3, .2, 1., .1, .1, .01)
    assert _check(gated)[0] == ("market (gated)", 2)


def test_incumbent():
    rng = np.random.default_rng(6)
    nA, nB = 4, 10
    S0 = jnp.asarray([7.5] * 6 + [0.] * (si.N_DEP - 6))
    built = si.incumbent(jnp.asarray(rng.normal(0, 1, nA)), jnp.asarray(rng.normal(3, 1, nB)), S0, 3.,
                         jnp.asarray(.5 * np.arange(1, si.K_FIG + 1)), .5, 1., .5, 1., .1, .5, 3)
    counts = _check(built)
    assert counts[0] == ("incumbent", 4)                # world, reader, fund, dependents
    np.testing.assert_allclose(np.asarray(built.P.position(built.P.init))[-si.N_DEP:], np.asarray(S0))
