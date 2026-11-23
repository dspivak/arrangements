"""The guard of ``tests/well_factored.py`` sees a box of more than seven and passes a nested one.

Eight stores on one copy wiring form a box of eight; ``nesting.nest`` regroups them into two
boxes of four, the same morphism (``tests/test_nesting.py``)."""
import jax.numpy as jnp

import well_factored
from arrangements import feeding_parts as fp
from arrangements.functors import Phi
from arrangements.nesting import box_counts, nest


def _stores(k):
    parts = [fp.rod_segment(1., float(i), f"store {i + 1}") for i in range(k)]
    wiring = fp.transfer("eight stores", 0, k, 0, k, lambda m, n: jnp.zeros(0), lambda m: m)
    return parts, wiring


def test_the_guard_sees_a_box_of_eight():
    parts, wiring = _stores(8)
    flat = fp.fill(parts, wiring)
    assert box_counts(flat) == [("eight stores", 8)]
    well_factored._seen.clear()
    well_factored._check(flat)
    assert well_factored._seen == [("eight stores", 8)]
    well_factored._seen.clear()


def test_the_guard_passes_the_nested_one():
    parts, wiring = _stores(8)
    nested = nest(parts, wiring, "stores")
    assert max(k for _, k in box_counts(nested)) <= 7
    Phi(nested)                     # watched: no failure at teardown
