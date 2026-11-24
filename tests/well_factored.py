"""Every arrangement a test runs is well-factored: no box holds more than seven (book task
well-factored, David, 2026-11-23: "good factorizations should have 2-7 boxes inside. more of
course may be inside of those ... make sure they are well-factored, so that the parts make sense
and there are not too many parts in any given arrangement").

A pytest plugin, loaded by ``conftest.py``. It watches the dynamics functors (``functors.Phi``,
``functors.closed_step``, ``functors.open_step``, ``pcK.PhiK``): each arrangement handed to one is
read with ``nesting.box_counts`` (boxes and parts only: the operations the explorer draws for a
wiring's computations are not counted, so a box can pass here and hold more than seven on the page;
for feeding, ``test_feeding_nested.py`` counts what the explorer draws), and a test that runs an arrangement
with a box holding more than ``LIMIT`` fails, naming the box, unless the box is one of ``KNOWN``,
the composites not yet regrouped, each with where its regrouping stands. Fewer than two is not
checked: a part beside its port adapter (a motor and its rate's static map) is David's B.
"""
from __future__ import annotations

import sys

import pytest

LIMIT = 7
# Box names (prefixes) still over seven, 2026-11-23, 8b0d215f: the book plan's P115 regroups
# them (feeding's are done, P109 (m)). Remove a line when its composite is regrouped.
KNOWN = (
    "route",                # attention_beside_physics.wired_attention: K pendulums with their delays, and the weights
    "portsplit",            # runner.denote: a denoted net's leaves
    "denoted net",          # runner.denote's own label, and couple's default names "denoted net a", "b"
                            # (test_market_runner A4b: 10; A2c: 8)
    "wires",                # runner.py's static route box (tests/test_market_runner.py, up to 14); was under "wire"
    # Found 2026-11-23 by 3bd2e654's guard inventory (every test file run under the guard's watch, recording):
    "burst_room",           # acceptable.py / stored_account.py: 11
    "room_",                # available.py: room_lamp_guest 9, room_with_guest 8, room_lamp 8
    "stick_slip_room",      # stored_account.py: 8
)
# Test files that run an arrangement whose top level, unnamed (drawn "(top level)"), holds more than
# seven, not yet regrouped (same inventory; largest count seen). Remove a line when it is regrouped.
KNOWN_TOP = {
    "test_addressed_hearing_teams.py": 8, "test_attention_beside_physics.py": 41, "test_attention_runner.py": 39,
    "test_cascade.py": 8, "test_competing_wholes.py": 9, "test_fire.py": 9, "test_light_sources.py": 290,
    "test_logic.py": 16, "test_positioned_people.py": 8, "test_regression_springs.py": 13,
    "test_system_id.py": 162, "test_teams_of_teams.py": 8, "test_template_copying.py": 24,
}
# Test files whose business is comparing a nested composite with the flat one it regroups: the flat one
# runs on purpose.
COMPARISONS = {"test_nesting.py", "test_well_factored_regular.py"}
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


@pytest.hookimpl(wrapper=True)
def pytest_runtest_teardown(item):
    # A wrapper, so pytest's own teardown runs first: failing before it left the test's fixtures
    # standing and the next test erred at setup ("previous item was not torn down properly").
    result = yield
    name = item.path.name
    if name in COMPARISONS:
        _seen.clear()
    elif name in KNOWN_TOP:
        _seen[:] = [s for s in _seen if s[0] != "(top level)"]
    if _seen:
        over = sorted(set(_seen))
        _seen.clear()
        pytest.fail(f"an arrangement this test ran has a box holding more than {LIMIT}: {over} "
                    "(tests/well_factored.py)", pytrace=False)
    return result
