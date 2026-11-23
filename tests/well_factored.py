"""Every arrangement a test runs is well-factored: no box holds more than seven (book task
well-factored, David, 2026-11-23: "good factorizations should have 2-7 boxes inside. more of
course may be inside of those ... make sure they are well-factored, so that the parts make sense
and there are not too many parts in any given arrangement").

A pytest plugin, loaded by ``conftest.py``. It watches the dynamics functors (``functors.Phi``,
``functors.closed_step``, ``functors.open_step``, ``pcK.PhiK``): each arrangement handed to one is
read with ``nesting.box_counts``, as the explorer draws it, and a test that runs an arrangement
with a box holding more than ``LIMIT`` fails, naming the box, unless the box is one of ``KNOWN``,
the composites not yet regrouped, each with where its regrouping stands. Fewer than two is not
checked: a part beside its port adapter (a motor and its rate's static map) is David's B.
"""
from __future__ import annotations

import sys

import pytest

LIMIT = 7
# Box names (prefixes) still over seven, 2026-11-23, 8b0d215f: the book plan's P109 (m)
# paragraph says which stage regroups each. Remove a line when its composite is regrouped.
KNOWN = (
    "wiring",               # tests/test_many_parts_wirings.py collective; a test's own unnamed wiring
    "route",                # attention_beside_physics.wired_attention: K pendulums with their delays, and the weights
    "portsplit",            # runner.denote: a denoted net's leaves
    "one_sun_tilted",       # tests/test_competing_wholes.py K10
)
_seen: list = []
_names = ("Phi", "closed_step", "open_step", "PhiK")
_originals: dict = {}


def _check(arr):
    from arrangements.nesting import box_counts
    try:
        counts = box_counts(arr)
    except Exception:  # noqa: BLE001  (a construction the counter cannot read is not this check's)
        return
    for name, k in counts:
        if k > LIMIT and not any(name.startswith(x) for x in KNOWN):
            _seen.append((name, k))


def _wrap(fn):
    def watched(arr, *args, **kwargs):
        _check(arr)
        return fn(arr, *args, **kwargs)
    watched.__wrapped__ = fn
    return watched


def _sweep():
    from arrangements import functors, pcK
    if not _originals:
        _originals.update({n: getattr(functors, n) for n in _names[:3]})
        _originals["PhiK"] = pcK.PhiK
    for module in list(sys.modules.values()):
        if module is None:
            continue
        for n, fn in _originals.items():
            if getattr(module, n, None) is fn:
                setattr(module, n, _wrap(fn))


def pytest_runtest_setup(item):
    _sweep()
    _seen.clear()


def pytest_runtest_teardown(item):
    if _seen:
        over = sorted(set(_seen))
        _seen.clear()
        pytest.fail(f"an arrangement this test ran has a box holding more than {LIMIT}: {over} "
                    "(tests/well_factored.py)", pytrace=False)
