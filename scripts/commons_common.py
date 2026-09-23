"""Shared setup for the commons scripts (``notes/COMMONS-DESIGN.md``, P8.2).

One script per wiring and variant, as the brief asks; this holds only what all of them
need -- the x64 switch, the reference parameters, and a printer. No modelling lives here.
"""

import jax

jax.config.update("jax_enable_x64", True)

import numpy as np                                                          # noqa: E402

from arrangements.commons import Params                                     # noqa: E402

PAR = Params()                       # r = 1, K = 100, p = 0.01, gamma = 1, mobility = 0.01
S_STAR = PAR.s_optimal()             # 100.0
NS = (1, 5, 20)
STEPS = 60_000

# The finite-supply regime predictions 11-13 are derived in (quasi-static: X_0 << gamma).
FIN = Params(r=0.0, mobility=0.01)
R_FIN, EPS = 5.0, 0.2


def head(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def row(*cells):
    print("  " + "".join(str(c).ljust(16) for c in cells))


def verdict(name, ok, detail=""):
    print(f"  [{'MET    ' if ok else 'NOT MET'}] {name}{'  -- ' + detail if detail else ''}")
    return bool(ok)


def close(a, b, rtol=2e-3):
    return bool(np.all(np.abs(np.asarray(a) - np.asarray(b)) <= rtol * np.abs(np.asarray(b))))
