"""Every public constructor of an arrangement declares the type of each argument (book plan P112).

A public top-level function of ``arrangements/`` whose declared return names ``Arrangement`` (``Arrangement``,
``AttentionArrangement``, ...) must declare a type for every argument, ``*args`` and ``**kwargs`` included, so that a
reader, and the explorer that asks its user for a constructor's arguments, can tell a number from an array or a
function. Static: each module is parsed, nothing is imported or run.

David, 2026-11-19, of constructors whose argument's kind could not be read off the code: "that's a bad sign. let's run
a project to fix all that." Claude Opus 5.5, 2f000a7c, 2026-11-19.
"""

import ast
from pathlib import Path

LIB = Path(__file__).resolve().parents[1] / "arrangements"

# Being rebuilt by other runs on 2026-11-19 (book plan P109 (e)); to be removed from this set when they finish.
SKIP = {"feeding", "feeding_parts", "feeding_composed"}


def undeclared_constructor_arguments():
    """``[(module, function, line, [argument, ...])]`` for each public constructor with an undeclared argument."""
    out = []
    for path in sorted(LIB.glob("*.py")):
        if path.stem in SKIP:
            continue
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            if node.returns is None or "Arrangement" not in ast.unparse(node.returns):
                continue
            a = node.args
            params = a.posonlyargs + a.args + a.kwonlyargs + [p for p in (a.vararg, a.kwarg) if p is not None]
            bare = [p.arg for p in params if p.annotation is None]
            if bare:
                out.append((path.stem, node.name, node.lineno, bare))
    return out


def test_constructors_declare_every_argument():
    bad = undeclared_constructor_arguments()
    assert not bad, "constructors with an undeclared argument:\n" + "\n".join(
        f"  {m}.{f} (l.{n}): {', '.join(args)}" for m, f, n, args in bad)


def undeclared_anywhere():
    """``[(module, function, line, [argument or 'return', ...])]`` for each public top-level function that leaves an
    argument's type or what it returns undeclared."""
    out = []
    for path in sorted(LIB.glob("*.py")):
        if path.stem in SKIP:
            continue
        for node in ast.parse(path.read_text()).body:
            if not isinstance(node, ast.FunctionDef) or node.name.startswith("_"):
                continue
            a = node.args
            params = a.posonlyargs + a.args + a.kwonlyargs + [p for p in (a.vararg, a.kwarg) if p is not None]
            bare = [p.arg for p in params if p.annotation is None] + (["return"] if node.returns is None else [])
            if bare:
                out.append((path.stem, node.name, node.lineno, bare))
    return out


def test_every_public_function_declares_its_types():
    """Since 2026-11-19 (P112, David: "please handle 1 and 2. i'm tired of puttingtings off.") every public top-level
    function declares each argument's type and what it returns."""
    bad = undeclared_anywhere()
    assert not bad, "public functions with an undeclared type:\n" + "\n".join(
        f"  {m}.{f} (l.{n}): {', '.join(args)}" for m, f, n, args in bad)
