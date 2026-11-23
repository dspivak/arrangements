"""The designed models, nested by meaning, equal their flat composites (book task well-factored-2ac0e890).

David, 2026-11-23: "good factorizations should have 2-7 boxes inside ... make sure they are well-factored,
so that the parts make sense". Each model below is built as ``factor_tree(parts, wiring, tree)``; here the
nested composite, computed by its own maps (``fast=False``), is checked against the flat composite of the
same parts in the tree's order (``nesting._flat_in_order``): the forcing, the readout and the input map at
random points. The composite the library returns (``fast=True``) computes by that flat composite's maps,
so it is checked too. Every box holds 2 to 7 things.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.nesting import _flat_in_order, _leaves, factor_tree

jax.config.update("jax_enable_x64", True)


def _counts(a, out):
    """The number of things inside every box of ``a`` (a compose with a named label)."""
    if not a.construction:
        return out
    op, f, g = a.construction
    if op == "compose" and not a.label.startswith("("):
        kids = []

        def flat(x):
            if x.construction and x.construction[0] == "tensor":
                flat(x.construction[1])
                flat(x.construction[2])
            else:
                kids.append(x)
        flat(f)
        out.append(len(kids))
        for k in kids:
            _counts(k, out)
    return out


def _agree(flat, nested, seed=3, near=None):
    """At random points; ``near`` (a configuration) puts them within 5% of it, for parts whose maps
    take logarithms."""
    rng = np.random.default_rng(seed)
    for _ in range(3):
        q = jnp.asarray(rng.normal(size=flat.P.Q_dim) if near is None
                        else np.asarray(near) * (1 + .05 * rng.normal(size=flat.P.Q_dim)))
        if near is not None:
            assert np.all(np.isfinite(np.asarray(flat.omega(q, jnp.zeros(flat.out_dim_M),
                                                            jnp.zeros(flat.in_dim_N))[0])))
        m, n = jnp.asarray(rng.normal(size=flat.out_dim_M)), jnp.asarray(rng.normal(size=flat.in_dim_N))
        for x, y in zip(flat.omega(q, m, n), nested.omega(q, m, n)):
            np.testing.assert_allclose(np.asarray(x), np.asarray(y), atol=1e-12)
        np.testing.assert_allclose(np.asarray(flat.out_f(q, m)), np.asarray(nested.out_f(q, m)), atol=1e-12)
        np.testing.assert_allclose(np.asarray(flat.in_f(q, m, n)), np.asarray(nested.in_f(q, m, n)), atol=1e-12)


def _check(parts, wiring, tree, made, near=False):
    """``made`` (what the library returns) and the nested composite by its own maps both equal the flat
    composite in the tree's order; every box of the nesting holds 2 to 7 things."""
    nested = factor_tree(list(parts), wiring, tree, fast=False)
    flat = _flat_in_order(list(parts), wiring, _leaves(tree))
    at = flat.P.position(flat.P.init) if near else None
    _agree(flat, nested, near=at)
    _agree(flat, made, near=at)
    counts = _counts(nested, [])
    assert counts and all(2 <= c <= 7 for c in counts), counts
    assert _counts(made, []) == counts
    return counts


# -- the town (arrangements/town.py) ------------------------------------------------------------

@pytest.mark.parametrize("stage, options", [(2, {}), (3, dict(diligence=True)), (4, dict(diligence=True)),
                                            (4, dict(fit_coughing=True))])
def test_town_nested_equals_flat(stage, options):
    from arrangements.town import Settings, eve_town
    town = eve_town(stage, Settings(delay_steps=4), **options)
    counts = _check(town.parts, town.wiring, town.tree, town.arr)
    assert counts[0] == 4 + (stage >= 3) + (stage == 4)          # town: plant and townspeople, claims, ...


# -- the Hodgkin-Huxley membrane (arrangements/hh_atp.py) ---------------------------------------

def test_membrane_nested_equals_flat():
    from arrangements import hh_atp as hh
    made = hh.membrane()
    assert _check(hh.membrane_parts(), hh.membrane_wiring(), hh.MEMBRANE_TREE, made, near=True)[0] == 2
    # the same order as the flat composite: the same state, the same forcing
    flat = hh.fill(hh.membrane_parts(), hh.membrane_wiring())
    q = flat.P.position(flat.P.init)
    np.testing.assert_allclose(np.asarray(made.P.position(made.P.init)), np.asarray(q), rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(made.omega(q, hh.EMPTY, hh.EMPTY)[0]),
                               np.asarray(flat.omega(q, hh.EMPTY, hh.EMPTY)[0]), rtol=0, atol=0)
