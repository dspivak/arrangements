"""The general parts of the feeding note, each a library arrangement.

The note is the book repo's ``background/feeding-imc/feeding-imc.tex``; a name in
double backquotes after "note" below is a label there. Everything here is a morphism of
IMC's Cartesian tangent subcategory in the array core: an ``Arrangement`` whose integrator
responds to the load it receives and whose inclination field is any one-form, exact or not.

Two kinds of primitive, as in the note's ``sec.integrands`` and ``sec.push``:

* an integrand, ``rod_segment`` or ``mass``: stores a state, sends its readout through one
  output, and responds to the load that comes back along that output;
* ``push``: stores nothing; inputs ``(F, x)``, inclination field ``F dx``.

Everything else is composed from these by the library's own ``compose_seq`` and
``tensor_arrangements`` (``fill``), so each part records its factors and
``explore.inspect`` shows them:

* ``pushes``    -- a finite sum of pullbacks of ``F dx`` (note ``prop.push_generates``: the
  homomorphism of commutative monoids from the free commutative monoid on smooth maps
  ``V -> R^2`` to one-forms on ``V``, ``(c, h) |-> c dh``; the empty list goes to the zero
  form and a concatenation to the sum, which is what pullback-and-sum computes);
* ``spring``    -- note ``eqn.spring``, two pushes;
* ``dragger``   -- note ``eqn.dragger``, a memory and a spring;
* ``damper``    -- note ``eqn.damper``, a dragger and a return spring;
* ``contact``   -- a spring and a damper on one stretch, each behind a gate wire: the
  note's ground and its lever's attachment (``sec.source``, ``sec.gates``);
* ``motor``     -- note ``eqn.motor``, two pushes, told a push; ``motor_at_rate``, two pushes,
  told a rate; each reads the battery and is only as active as ``live`` of it;
* ``latch``     -- note ``eqn.contact_latch``, a store and its motor;
* ``sweep``     -- the note's sweep, two motors;
* ``send`` and ``discard`` -- a part with a constant sent in on some input wires, or with
  some outputs unread (the first paragraph of the note's ``sec.general``);
* ``identity``  -- the identity on an interface, beside the parts of an arrangement whose
  domain is not the unit.

Built 2026-11-19 for the book's P109 (e) (Claude Fable 5.2, 6a32afbb); ``contact``,
``latch`` and ``identity`` 2026-11-21 for P109 (h) (86739fa7). ``feeding.py`` is built from
these through ``feeding_composed.py``.
"""
from __future__ import annotations

from typing import Callable, Mapping, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, zero_form
from .coalgebra import CotCoalg, CotCoalgK
from .feeding import theta
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

EMPTY = jnp.zeros(0)


def nonpositive(x) -> bool:
    """Whether a constant is known to be ``<= 0``. A traced constant (the page's exporter traces
    the knobs' parameters, ``scripts/export_step_js.py``) is not known, and passes."""
    import jax
    return not isinstance(x, jax.core.Tracer) and x <= 0


def number(x) -> str:
    """A constant as it is written in a part's label; a traced constant as ``?``."""
    import jax
    return "?" if isinstance(x, jax.core.Tracer) else f"{float(x):g}"

# A term of a sum of pushes: a name, a coefficient c : V -> R and a target h : V -> R.
Term = Tuple[str, Callable[[Array], Array], Callable[[Array], Array]]


# ---------------------------------------------------------------------------
# What a part is, for drawing it (book P109 (f), act (4)): ``Arrangement.meta``.
#
# A part's ``meta`` names its input wires, says which kind of inclination field it has under
# David's rubric (book plan P109, *The rubric*: "best: 0. nothing. undrawn. 0.5: quadratic
# potential d (x-y)^2 ... drawn as a spring"), and writes its formula in the letters of its
# inputs. Nothing in it is trusted: ``term_reads`` reads which inputs a term's coefficient and
# target read off the traced code, and ``check_formulas`` compares each written formula with
# the code, its letters with those reads and its values at sampled points. ``meta`` plays no
# part in the dynamics.
#
#   inputs:    [(letter, name, width), ...], the part's input wires in order
#   terms:     [{name, c, h, c_text, h_text}, ...], the field sum_i c_i dh_i (pushes, motors)
#   constants: {letter: number}, where: [(letter, formula), ...] for the formulas
#   kind:      a key of RUBRIC; part: what the part is ('push', 'pushes', 'spring', 'motor', ...)
#   wrapper:   True for ``send`` and ``discard``, which only fix or drop wires
# ---------------------------------------------------------------------------

RUBRIC = {"zero": "0", "spring": "0.5", "stopper": "stopper", "cost": "1", "Petri transition": "2",
          "push of some inputs": "3", "push of everything": "4"}

# The functions a formula may call: those of ``viewer/explorer-forms.js`` and the page's theta.
FUNCTION_WORDS = {"θ": "θ(t) = ρ(t)/(ρ(t) + ρ(1 - t)), a smooth step, 0 for t ≤ 0 and 1 for t ≥ 1, "
                       "where ρ(t) is exp(-1/t) for t > 0 and 0 otherwise"}
FUNCTIONS = {"exp": jnp.exp, "log": jnp.log, "sqrt": jnp.sqrt, "abs": jnp.abs, "sin": jnp.sin,
             "cos": jnp.cos, "tanh": jnp.tanh, "Σ": jnp.sum, "θ": theta}


def with_meta(arr: Arrangement, **meta) -> Arrangement:
    """``arr`` with ``meta`` added to what it already records."""
    from dataclasses import replace
    return replace(arr, meta={**(arr.meta or {}), **meta})


def formula_names(text: str) -> set:
    """The names a formula uses, calls excepted."""
    import re
    return {m.group(1) for m in re.finditer(r"([^\W\d][\w]*)\s*(\()?", text) if not m.group(2)}


def spelled_out(meta: Mapping) -> str:
    """A part's ``formula``, then what each letter in it is: an input or an output by its name, a
    constant by its number and its ``words``, an abbreviation by its words. A letter with neither is
    a ``ValueError``, as it is for ``pushes``."""
    import re
    text = meta["formula"]
    wires = {letter: name for letter, name, _ in [*meta.get("inputs", ()), *meta.get("outputs", ())]}
    words, constants = meta.get("words", {}), meta.get("constants", {})
    said, seen = [], set()
    for m in re.finditer(r"(?<!\w)([^\W\d_](?:_\w+)?)(?![\w(])", text):
        n = m.group(1)
        if n in seen or n in ("a", "d") or (f"{n} = " in text and n not in words):
            continue
        seen.add(n)
        if n in constants and words.get(n):
            said.append(f"{n} = {float(constants[n]):g} is {words[n]}")
        elif words.get(n):
            said.append(f"{n} is {words[n]}")
        elif n in wires:
            said.append(f"{n} is the {wires[n]}")
        else:
            raise ValueError(f"{meta.get('part')}: no words for {n} in its formula {text}")
    said += [FUNCTION_WORDS[f] for f in FUNCTION_WORDS if f"{f}(" in text]
    return text.replace("**2", "²").replace("*", " ") + ("; here " + "; ".join(said) if said else "")


def unexplained_names(meta: Mapping) -> set:
    """The names the written formulas use that are neither an input's letter nor given ``words``."""
    where = dict(meta.get("where", ()))
    texts = [t[k] for t in meta.get("terms", ()) for k in ("c_text", "h_text") if t.get(k) is not None]
    seen, todo = set(), [n for text in texts for n in formula_names(text)]
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo.extend(formula_names(where[n]) if n in where else ())
    return seen - {letter for letter, _, _ in meta["inputs"]} - {k for k, w in meta.get("words", {}).items() if w}


def _environment(meta: Mapping, v: Array) -> dict:
    """The formulas' names at the point ``v`` of the part's inputs."""
    env, at = dict(meta.get("constants", {})), 0
    for letter, _, width in meta["inputs"]:
        env[letter] = v[at] if width == 1 else v[at:at + width]
        at += width
    for name, text in meta.get("where", ()):
        env[name] = formula_value(text, env)
    return env


def formula_value(text: str, env: Mapping) -> Array:
    """A formula's value, read with Python's precedences as ``explorer-forms.js`` reads it."""
    return eval(text, {"__builtins__": {}, **FUNCTIONS}, dict(env))


def _expanded(meta: Mapping, text: str) -> set:
    """The input letters a formula reads, through its abbreviations."""
    where, seen, todo = dict(meta.get("where", ())), set(), list(formula_names(text))
    while todo:
        n = todo.pop()
        if n not in seen:
            seen.add(n)
            todo.extend(formula_names(where[n]) if n in where else ())
    return seen & {letter for letter, _, _ in meta["inputs"]}


def _dim(meta: Mapping) -> int:
    return sum(width for _, _, width in meta["inputs"])


def term_reads(meta: Mapping, i: int) -> dict:
    """Which inputs term ``i``'s coefficient and target read, off the traced code
    (``jax.make_jaxpr``, ``explore``'s coordinate dependencies): ``{'c': letters, 'h': letters,
    'uncertain': bool}``, ``uncertain`` when the trace gives only an upper bound."""
    import numpy as np
    from ._explorer_inputs import coordinate_inputs
    term, dim = meta["terms"][i], _dim(meta)
    owner = [letter for letter, _, width in meta["inputs"] for _ in range(width)]
    v = jnp.asarray(np.linspace(.3, .9, dim))
    out, uncertain = {}, False
    for key in ("c", "h"):
        size = int(jnp.size(term[key](v)))
        (rows,) = coordinate_inputs(lambda x: jnp.ravel(term[key](x)), (v,), [size])
        read = set()
        for support, bound in rows:
            read |= {owner[j] for j in support}
            uncertain |= bound
        out[key] = [letter for letter, _, _ in meta["inputs"] if letter in read]
    out["uncertain"] = uncertain
    return out


def term_kind(meta: Mapping, i: int) -> str:
    """Term ``i``'s kind under the rubric: a cost when its coefficient reads nothing (``c dh`` is
    then ``d(c h)``); a Petri transition when its target is a fixed signed sum of the inputs (the
    same derivative at every point, and 0 at 0); else a push of some inputs or of everything."""
    import jax
    import numpy as np
    if not term_reads(meta, i)["c"]:
        return "cost"
    term, dim = meta["terms"][i], _dim(meta)
    rng = np.random.default_rng(1)
    points = [jnp.asarray(rng.normal(size=dim)) for _ in range(3)]
    slopes = [jax.jacfwd(lambda x: jnp.ravel(term["h"](x)))(x) for x in points]
    linear = (all(np.allclose(slopes[0], s, atol=1e-12) for s in slopes[1:])
              and np.allclose(term["h"](jnp.zeros(dim)), 0., atol=1e-12))
    reads = term_reads(meta, i)
    # A push on one quantity is a signed sum of one, so every push would be a transition; the
    # kind is kept for a target that moves two or more quantities at once (the run's reading).
    if linear and len(reads["h"]) > 1:
        return "Petri transition"
    return "push of everything" if set(reads["c"]) | set(reads["h"]) == {l for l, _, _ in meta["inputs"]} \
        else "push of some inputs"


def check_formulas(meta: Mapping, samples: int = 4, seed: int = 0, scale: float = 2.) -> None:
    """Raise ``ValueError`` unless every written formula of ``meta`` is the code's: the input
    letters each coefficient and target formula reads are those its traced code reads (a
    superset of them where the trace gives only an upper bound), and its value is the code's
    at ``samples`` random points of the inputs."""
    import numpy as np
    rng = np.random.default_rng(seed)
    dim = _dim(meta)
    for i, term in enumerate(meta.get("terms", ())):
        keys = [key for key in ("c", "h") if term.get(f"{key}_text") is not None]
        if not keys:
            continue
        reads = term_reads(meta, i)
        idle = {}
        for key in keys:
            written = _expanded(meta, term[f"{key}_text"])
            traced = set(reads[key])
            if written != traced and not (reads["uncertain"] and traced >= written):
                if not written >= traced:
                    raise ValueError(f"{term['name']}: its {key} is written in {sorted(written)} "
                                     f"and its code reads {sorted(traced)}")
                # The formula names more than the code reads: allowed only where the constants
                # make it independent of the rest (a collector whose clearance is 0), checked below.
                idle[key] = written - traced
        for _ in range(samples):
            v = jnp.asarray(rng.normal(scale=scale, size=dim))
            env = _environment(meta, v)
            for key in keys:
                got = jnp.ravel(jnp.asarray(formula_value(term[f"{key}_text"], env), dtype=float))
                want = jnp.ravel(term[key](v))
                got = jnp.broadcast_to(got, want.shape)
                if not np.allclose(got, want, rtol=1e-9, atol=1e-12):
                    raise ValueError(f"{term['name']}: its {key} is written {term[f'{key}_text']!r}, "
                                     f"which is {np.asarray(got)} where the code gives {np.asarray(want)}")
                for letter in idle.get(key, ()):
                    moved = dict(env, **{letter: env[letter] + 1.})
                    for name, text in meta.get("where", ()):
                        moved[name] = formula_value(text, moved)
                    if not np.allclose(formula_value(term[f"{key}_text"], moved), formula_value(term[f"{key}_text"], env),
                                       rtol=1e-9, atol=1e-12):
                        raise ValueError(f"{term['name']}: its {key} is written with {letter}, which its code does not read")


# ---------------------------------------------------------------------------
# Integrands (note sec.integrands).
# ---------------------------------------------------------------------------


def _integrator(init: Array, read: Callable[[Array], Array], response: Callable[[Array, Array], Array],
                stages: int, h: float, label: str) -> CotCoalg | CotCoalgK:
    """An integrator whose state moves at ``response(state, load)``, by one Euler round of
    length ``h`` (``stages == 1``) or the four rounds of classical Runge-Kutta (``stages == 4``).

    The same scheme as ``feeding._parameter``. A tensor of parts built with one ``stages`` and
    ``h`` steps as that scheme applied to the whole, since each round every part reads out at
    its own stage point and receives the load computed from all the readouts.
    """
    init = jnp.asarray(init, dtype=float)
    if stages == 1:
        advance = lambda s, xi, tau: s + tau * response(s, xi)
        return CotCoalg(len(read(init)), init, read, lambda s, xi: advance(s, xi, h), label,
                        timed_step=lambda s, xi, tau: advance(s, xi, h * tau))
    if stages != 4:
        raise ValueError("one Euler round or four Runge-Kutta rounds")

    def mid(s, index: int, scale: float) -> Array:
        x, ks = s
        return x + h * scale * ks[index]

    reads = (read, lambda s: read(mid(s, 0, .5)), lambda s: read(mid(s, 1, .5)), lambda s: read(mid(s, 2, 1.)))
    advances = (lambda s, xi: (s, (response(s, xi),)),
                lambda s, xi: (s[0], s[1] + (response(mid(s, 0, .5), xi),)),
                lambda s, xi: (s[0], s[1] + (response(mid(s, 1, .5), xi),)),
                lambda s, xi: s[0] + h / 6 * (s[1][0] + 2 * s[1][1] + 2 * s[1][2] + response(mid(s, 2, 1.), xi)))
    return CotCoalgK(len(read(init)), init, reads, advances, label=label,
                     durational=lambda tau: _integrator(init, read, response, 4, h * tau, label))


def rod_segment(capacity: float, initial: float | Sequence[float] | Array, label: str = "rod segment", *,
                stages: int = 1, h: float = 1.0) -> Arrangement:
    """A rod-segment integrand ``1 -> <0|n>``: stores a number, sends it out, and moves it at
    ``1/capacity`` times the load that comes back (book Ch 6 ``ex.tangent_heat``).

    ``initial`` a number gives one store (``n = 1``); an array of ``n`` numbers gives ``n``
    stores of the one capacity, side by side, drawn as one box.
    """
    if nonpositive(capacity):
        raise ValueError("a rod segment's capacity is positive")
    init = jnp.atleast_1d(jnp.asarray(initial, dtype=float))
    P = _integrator(init, lambda s: s, lambda s, xi: xi / capacity, stages, h, label)
    return Arrangement(P, 0, 0, 0, init.shape[0], lambda q, m: q, lambda q, m, n: EMPTY, zero_form, label)


def mass(m: float, position: float, momentum: float = 0.0, label: str = "mass", *,
         stages: int = 1, h: float = 1.0) -> Arrangement:
    """A mass ``1 -> <0|1>``: stores ``(q, p)``, sends out ``q`` and forgets ``p``, and
    responds to a load ``xi`` by ``(p/m, xi)`` (book Ch 6 ``exc.tangent_mass``)."""
    if nonpositive(m):
        raise ValueError("a mass is positive")
    P = _integrator(jnp.stack([jnp.asarray(position, dtype=float), jnp.asarray(momentum, dtype=float)]), lambda s: s[:1],
                    lambda s, xi: jnp.stack([s[1] / m, xi[0]]), stages, h, label)
    return Arrangement(P, 0, 0, 0, 1, lambda q, m_out: q, lambda q, m_out, n: EMPTY, zero_form, label)


# ---------------------------------------------------------------------------
# The one stateless box, and the lens that feeds parts (note sec.push).
# ---------------------------------------------------------------------------


def push(label: str = "push", width: int = 1) -> Arrangement:
    """The stateless box ``1 -> <2|0>`` with inputs ``(F, x)`` and inclination field ``F dx``:
    it pushes back ``F`` through ``x`` and ``0`` through ``F`` (note ``sec.push``).

    With ``width = n`` it is ``n`` of them side by side, drawn as one box: inputs
    ``(F_1..F_n, x_1..x_n)``, field ``sum_j F_j dx_j``.
    """
    w = width
    meta = dict(part="push", kind="push of everything", inputs=[("F", "push", w), ("x", "pushed", w)],
                terms=[dict(name="push", c=lambda v: v[:w], h=lambda v: v[w:], c_text="F", h_text="x")])
    return Arrangement(trivial_parameter(), 0, 0, 2 * width, 0, lambda q, m: EMPTY, lambda q, m, n: EMPTY,
                       lambda q, m, n: (EMPTY, EMPTY, jnp.concatenate([jnp.zeros(width), n[:width]])), label,
                       meta=meta)


def transfer(label: str, inner_in: int, inner_out: int, outer_in: int, outer_out: int,
             route: Callable[[Array, Array], Array],
             publish: Callable[[Array], Array] = lambda m: EMPTY) -> Arrangement:
    """A stateless lens with zero inclination field, ``<inner_in|inner_out> -> <outer_in|outer_out>``.

    ``route(m, n)`` computes the parts' inputs from the parts' outputs ``m`` and the outer
    inputs ``n`` (the note's circles: any smooth map); ``publish(m)`` computes the outer
    outputs from the parts' outputs. Loads come back along the transposes of their
    derivatives, which ``compose_seq`` takes; none is written here.
    """
    return Arrangement(trivial_parameter(), inner_in, inner_out, outer_in, outer_out,
                       lambda q, m: publish(m), lambda q, m, n: route(m, n), zero_form, label)


def identity(inputs: int, outputs: int, label: str = "identity") -> Arrangement:
    """The identity on the interface ``<inputs|outputs>``: every value and every load passes
    through unchanged. It stands beside the parts that fill an arrangement whose domain is
    not the unit, such as the feeding note's spatial coupling."""
    return Arrangement(trivial_parameter(), inputs, outputs, inputs, outputs,
                       lambda q, m: m, lambda q, m, n: n, zero_form, label)


def fill(parts: Sequence[Arrangement], wiring: Arrangement) -> Arrangement:
    """The parts side by side, their interfaces filled into ``wiring``: ``wiring . (part_1 (x) ...)``."""
    return compose_seq(tensor_arrangements(list(parts)), wiring)


def pushes(label: str, dim: int, terms: Sequence[Term], *, inputs: Sequence[Tuple[str, str, int]] | None = None,
           forms: Sequence[Tuple[str, str]] | None = None, constants: Mapping[str, float] | None = None,
           where: Sequence[Tuple[str, str]] = (), words: Mapping[str, str] | None = None) -> Arrangement:
    """The stateless part ``1 -> <dim|0>`` with inclination field ``sum_i c_i dh_i``.

    One ``push`` per term, fed by the circles ``c_i`` and ``h_i`` (note ``eqn.push_term``,
    ``eqn.pullback_term``). No derivative of a coefficient appears, because ``push`` is
    load-free at ``F``. With no terms it is the zero field on ``R^dim``. A target ``h_i`` with
    ``n`` components gets a ``push`` of that width, and its coefficient has ``n`` components
    too (or one, used for each).

    For drawing (``meta``): ``inputs`` names the input wires, ``(letter, name, width)`` each
    (by default ``v_1 .. v_dim``); ``forms`` writes each term's ``(c, h)`` in those letters,
    with ``constants`` and the abbreviations ``where``; ``check_formulas`` holds them to the code.
    ``words`` says what each constant and abbreviation is, for the reader of the drawing (David,
    2026-11-23: "Everything in the formula needs to be defined"): every name a formula uses is an
    input, which the drawing shows entering, or has words here, else ``ValueError``.
    """
    inputs = list(inputs) if inputs is not None else [(f"v_{i + 1}", f"input {i + 1}", 1) for i in range(dim)]
    if sum(width for _, _, width in inputs) != dim:
        raise ValueError(f"pushes: the inputs named have {sum(w for _, _, w in inputs)} numbers, not {dim}")
    if forms is not None and len(forms) != len(terms):
        raise ValueError("pushes: one (c, h) formula per term")
    meta = dict(part="pushes", kind="zero" if not terms else None, inputs=inputs, constants=dict(constants or {}),
                where=list(where), words=dict(words or {}),
                terms=[dict(name=name, c=c, h=h, c_text=forms[i][0] if forms else None,
                            h_text=forms[i][1] if forms else None) for i, (name, c, h) in enumerate(terms)])
    if not terms:
        return Arrangement(trivial_parameter(), 0, 0, dim, 0, lambda q, m: EMPTY,
                           lambda q, m, n: EMPTY, zero_form, label, meta=meta)
    unexplained = unexplained_names(meta)
    if unexplained:
        raise ValueError(f"pushes {label}: no words for {', '.join(sorted(unexplained))} in its formulas")
    widths = [int(jnp.ravel(h(jnp.zeros(dim))).shape[0]) for _, _, h in terms]
    boxes = [push(f"push: {name}", n) for (name, _, _), n in zip(terms, widths)]

    def route(m: Array, v: Array) -> Array:
        return jnp.concatenate([x for (_, c, h), n in zip(terms, widths)
                                for x in (jnp.broadcast_to(jnp.ravel(c(v)), (n,)), jnp.ravel(h(v)))])

    return with_meta(fill(boxes, transfer(label, 2 * sum(widths), 0, dim, 0, route)), **meta)


# ---------------------------------------------------------------------------
# A constant sent in on a wire; an output left unread (note sec.general, first paragraph).
# ---------------------------------------------------------------------------


def send(part: Arrangement, constants: Mapping[int, float], label: str | None = None) -> Arrangement:
    """``part`` with a constant sent in on each input wire named in ``constants`` (index -> value).

    The field is pulled back along the map that inserts the constants, so the load at those
    inputs is dropped. The remaining inputs keep their order.
    """
    d = part.in_dim_N
    if part.in_dim_M or part.out_dim_M:
        raise ValueError("send: expected a part 1 -> <in|out>")
    if any(i < 0 or i >= d for i in constants):
        raise ValueError(f"send: the part has inputs 0..{d - 1}")
    free = [i for i in range(d) if i not in constants]
    fixed = sorted(constants)
    values = jnp.stack([jnp.asarray(constants[i], dtype=float) for i in fixed]) if fixed else jnp.zeros(0)
    order = jnp.argsort(jnp.array(fixed + free)) if d else jnp.zeros(0, dtype=int)

    def route(m: Array, n: Array) -> Array:
        return jnp.concatenate([values, n])[order]

    name = label or f"{part.label}[{', '.join(f'{i}:={number(constants[i])}' for i in fixed)}]"
    return with_meta(compose_seq(part, transfer(name, d, part.out_dim_N, len(free), part.out_dim_N, route, lambda m: m)),
                     wrapper=True, inputs=_free_wires(part, constants))


def _free_wires(part: Arrangement, constants: Mapping[int, float]) -> list:
    """The input wires ``(letter, name, width)`` of ``part`` that ``send`` leaves free: a wire none
    of whose numbers is fixed stays whole, one with some fixed is split into the free numbers."""
    named = (part.meta or {}).get("inputs") or [(f"v_{i + 1}", f"input {i + 1}", 1) for i in range(part.in_dim_N)]
    kept, at = [], 0
    for letter, name, width in named:
        fixed = [j for j in range(width) if at + j in constants]
        if not fixed:
            kept.append((letter, name, width))
        elif len(fixed) < width:
            kept += [(f"{letter}_{j + 1}", f"{name} {j + 1}", 1) for j in range(width) if at + j not in constants]
        at += width
    return kept


def discard(part: Arrangement, outputs: Sequence[int], label: str | None = None) -> Arrangement:
    """``part`` with the named outputs unread: nothing reads them and their load is ``0``."""
    if part.in_dim_M or part.out_dim_M:
        raise ValueError("discard: expected a part 1 -> <in|out>")
    kept = jnp.array([i for i in range(part.out_dim_N) if i not in set(outputs)], dtype=int)
    name = label or f"{part.label}, unread"
    return with_meta(compose_seq(part, transfer(name, part.in_dim_N, part.out_dim_N, part.in_dim_N, len(kept),
                                                lambda m, n: n, lambda m: m[kept])),
                     wrapper=True, inputs=(part.meta or {}).get("inputs"))


# ---------------------------------------------------------------------------
# The general parts (note sec.general).
# ---------------------------------------------------------------------------


def spring(label: str = "spring") -> Arrangement:
    """The spring ``1 -> <2|0>`` with inputs a stiffness ``k`` and a stretch ``u`` and field
    ``-d(k u^2 / 2) = -k u du - (u^2 / 2) dk`` (note ``eqn.spring``): the pull, and the energy
    it takes to change the stiffness, which comes back through the wire that sets ``k``."""
    part = pushes(label, 2, [("pull", lambda v: -v[0] * v[1], lambda v: v[1]),
                             ("stiffening", lambda v: -0.5 * v[1] ** 2, lambda v: v[0])],
                  inputs=[("k", "stiffness", 1), ("u", "stretch", 1)],
                  forms=[("-k*u", "u"), ("-u**2/2", "k")])
    # One mark, the cost it is minus d of (David, 2026-11-19, 6a32afbb: "It's much better to
    # think of a spring as d of a cost involving only a differene"); its two pushes when opened.
    return with_meta(part, part="spring", kind="spring", cost="k/2*u**2", formula="-d(k/2*u**2)")


def dragger(beta: float, memory: float = 0.0, label: str = "dragger", *,
            stages: int = 1, h: float = 1.0) -> Arrangement:
    """The dragger ``1 -> <2|1>`` (note ``eqn.dragger``): a memory of capacity ``beta`` and a
    spring stretched between the position ``x`` it reads and the memory ``z``.

    Inputs ``(k, x)``, output ``z``. The circle computes the stretch ``x - z``; the field is
    ``-d(k (x - z)^2 / 2)`` and the memory moves at ``k (x - z) / beta``.
    """
    wiring = transfer(label, 2, 1, 2, 1, lambda m, n: jnp.stack([n[0], n[1] - m[0]]), lambda m: m)
    return with_meta(fill([rod_segment(beta, memory, "memory", stages=stages, h=h), spring()], wiring),
                     part="dragger", kind="spring", inputs=[("k", "stiffness", 1), ("x", "position", 1)],
                     outputs=[("z", "memory", 1)], constants={"β": beta}, words={"β": "the memory's capacity"},
                     formula="-d(k/2*(x - z)**2), and the memory z moves at k*(x - z)/β")


def damper(beta: float, kappa_return: float, stroke: float = 0.0, label: str = "damper", *,
           read: bool = False, stages: int = 1, h: float = 1.0) -> Arrangement:
    """The damper ``1 -> <2|0>`` (note ``eqn.damper``): a dragger whose memory, its stroke
    ``z``, a spring sent the constant stiffness ``kappa_return`` pulls back to ``0``.

    Inputs a stiffness ``k`` (``0`` out of contact) and a gap ``u``. The field is
    ``-d(k (u - z)^2 / 2 + kappa_return z^2 / 2)`` and the stroke moves at
    ``(k (u - z) - kappa_return z) / beta``. With ``read`` it is ``1 -> <2|1>``, the same
    damper sending its stroke out.
    """
    back = send(spring("return spring"), {0: kappa_return})
    wiring = transfer(label, 3, 1, 2, int(read), lambda m, n: jnp.stack([n[0], n[1], m[0]]),
                      (lambda m: m) if read else (lambda m: EMPTY))
    return with_meta(fill([dragger(beta, stroke, stages=stages, h=h), back], wiring),
                     part="damper", kind="spring", inputs=[("k", "stiffness", 1), ("u", "gap", 1)],
                     outputs=[("z", "stroke", 1)] if read else [], constants={"β": beta, "κ_r": kappa_return},
                     words={"β": "the stroke's capacity", "κ_r": "the return spring's stiffness",
                            "z": "the stroke"},
                     formula="-d(k/2*(u - z)**2 + κ_r/2*z**2), and the stroke z moves at (k*(u - z) - κ_r*z)/β")


def contact_cost(k_spring: float, k_damper: float) -> Callable[[Array, Array, Array, Array], Array]:
    """The cost of a contact's gated spring and damper, its return spring left out:
    ``(g_s, g_d, u, z) |-> k_spring g_s u^2 / 2 + k_damper g_d (u - z)^2 / 2``.

    ``contact`` is composed from ``push`` and does not use this; it is here for the circle
    that tells a motor the load a contact puts on its store (a motor must be sent the true
    other load, note ``eqn.motor``), and ``tests/test_feeding_parts.py`` checks it against
    the composed field.
    """
    return lambda g_s, g_d, u, z: 0.5 * k_spring * g_s * u ** 2 + 0.5 * k_damper * g_d * (u - z) ** 2


def contact(k_spring: float, k_damper: float, beta: float, kappa_return: float, stroke: float = 0.0,
            label: str = "contact", *, stages: int = 1, h: float = 1.0) -> Arrangement:
    """The contact ``1 -> <3|1>``: a spring and a damper on one stretch, each behind a gate wire.

    Inputs: the spring's gate ``g_s``, the damper's gate ``g_d`` and the stretch ``u``; the
    spring is sent the stiffness ``k_spring g_s`` and the damper ``k_damper g_d``. Output:
    the damper's stroke ``z``, its integrand. The field is
    ``-d(k_spring g_s u^2 / 2 + k_damper g_d (u - z)^2 / 2 + kappa_return z^2 / 2)``, so a
    gate gets back the energy it takes to stiffen its part, ``-k_spring u^2 / 2`` and
    ``-k_damper (u - z)^2 / 2``.

    The feeding note's two contacts are this one (``feeding_composed``): the ground, its
    gates ``theta(-v / eps)`` and ``theta(-v / eps_d)`` of its own stretch ``v``, the stroke
    unread; and the lever's attachment, both gates the one ``G_k``.
    """
    def route(m: Array, n: Array) -> Array:
        g_s, g_d, u = n
        return jnp.stack([k_spring * g_s, u, k_damper * g_d, u])

    return with_meta(fill([spring(f"{label} spring"),
                           damper(beta, kappa_return, stroke, f"{label} damper", read=True, stages=stages, h=h)],
                          transfer(label, 4, 1, 3, 1, route, lambda m: m)),
                     part="contact", kind="spring",
                     inputs=[("g_s", "spring gate", 1), ("g_d", "damper gate", 1), ("u", "stretch", 1)],
                     outputs=[("z", "stroke", 1)],
                     constants={"κ_s": k_spring, "κ_d": k_damper, "β": beta, "κ_r": kappa_return},
                     words={"κ_s": "the spring's stiffness where its gate is 1", "κ_r": "the return spring's stiffness",
                            "κ_d": "the damper's stiffness where its gate is 1"},
                     formula="-d(κ_s*g_s/2*u**2 + κ_d*g_d/2*(u - z)**2 + κ_r/2*z**2)")


def engagement(latch: Array) -> Array:
    """How far a latch is engaged, ``s = theta((v - .3) / .2)`` (note ``sec.gates``)."""
    return theta((latch - .3) / .2)


def latch(capacity: float, rate: float, wake: float, initial: float = 0.0, label: str = "latch", *,
          stages: int = 1, h: float = 1.0) -> Arrangement:
    """The latch ``1 -> <5|1>`` (note ``eqn.contact_latch``): a store ``v`` and its motor.

    Inputs: the battery ``b``; whether the latch is free to engage, ``U``; the targets
    ``D`` to acquire and ``T`` to retain; and the load ``P`` the rest of the arrangement
    puts on the store. Output: ``v``. With ``s = engagement(v)``, the latch tells its motor
    the rate ``rate * (U ((1 - s) D + s T) - v)``: toward ``U D`` while the latch is not
    engaged and toward ``U T`` once it is, so what it takes to engage and what it takes to
    stay engaged differ. The motor reads the battery and drives the store at that rate times
    ``live(b, wake)``, the note's rate; the battery pays ``capacity u^2 - P u`` at the rate ``u``.
    """
    def route(m: Array, n: Array) -> Array:
        v = m[0]
        b, free, acquire, retain, other = n
        s = engagement(v)
        u = rate * (free * ((1 - s) * acquire + s * retain) - v)
        return jnp.stack([b, v, u, other])

    return with_meta(fill([rod_segment(capacity, initial, f"{label} store", stages=stages, h=h),
                           motor_at_rate(capacity, wake, f"{label} motor")],
                          transfer(label, 4, 1, 5, 1, route, lambda m: m)),
                     part="latch", kind="push of some inputs",
                     inputs=[("b", "battery", 1), ("U", "free to engage", 1), ("D", "acquire", 1),
                             ("T", "retain", 1), ("P", "other load", 1)],
                     outputs=[("v", "latch", 1)], constants={"λ": rate, "C": capacity, "b_0": wake},
                     words={"λ": "how fast the latch moves to its target", "s": "how far the latch is engaged",
                            "U": "whether it is free to engage", "D": "the target until it is engaged",
                            "T": "the target once it is"},
                     formula="a motor told the rate λ*(U*((1 - s)*D + s*T) - v), s = θ((v - 0.3)/0.2)")


def live(b: Array, wake: float) -> Array:
    """How awake a motor is at the battery level ``b``: ``theta(b / wake)``, ``0`` on an empty
    battery and ``1`` from the level ``wake`` up (note ``live``)."""
    return theta(b / wake)


def _motor_words(n: int) -> dict:
    """What the names of a motor's formula are, beside the inputs the drawing shows entering."""
    return {"C": "the capacity of the store the motor turns" if n == 1
                 else f"the capacity of each of the {n} stores the motor turns",
            "g": "how awake the motor is, 0 on an empty battery and 1 from the level b_0 up",
            "b_0": "the battery level from which the motor is fully awake",
            "P": "the load on that store from everything but the motor" if n == 1
                 else "the load on each store from everything but the motor"}


def _each(text: str, n: int) -> str:
    """``text``, added over the motor's stores when there are several."""
    return text if n == 1 else f"Σ({text})"


def motor(capacity: float, wake: float, label: str = "motor", width: int = 1) -> Arrangement:
    """The motor ``1 -> <4|0>`` (note ``eqn.motor``) for a store of the given capacity, told a push.

    Inputs: the battery ``b``, the store ``q``, the push ``tau`` it is told, and the load
    ``P`` the rest of the arrangement puts on the store. It reads the battery: it applies the
    push ``g tau``, with ``g = live(b, wake)``, so on an empty battery it applies none. Field
    ``g tau dq - g tau (g tau + P) / capacity db``: the store moves at
    ``(g tau + P) / capacity``, so the battery pays the power ``g tau`` times that rate.
    Load-free at ``tau`` and ``P``. It must be sent the true ``P``.

    With ``width = n`` it drives ``n`` stores of the one capacity from the one battery:
    inputs ``(b, q_1..q_n, tau_1..tau_n, P_1..P_n)``, the field the sum of the ``n`` motors'.
    """
    if nonpositive(capacity):
        raise ValueError("the driven store's capacity is positive")
    n = width

    def applied(v: Array) -> Array:
        return live(v[0], wake) * v[1 + n:1 + 2 * n]

    part = pushes(label, 1 + 3 * n,
                  [("drive", applied, lambda v: v[1:1 + n]),
                   ("pay", lambda v: -jnp.sum(applied(v) * (applied(v) + v[1 + 2 * n:])) / capacity,
                    lambda v: v[0])],
                  inputs=[("b", "battery", 1), ("q", "store", n), ("τ", "push", n), ("P", "other load", n)],
                  forms=[("g*τ", "q"), (f"-{_each('g*τ*(g*τ + P)', n)}/C", "b")],
                  constants={"C": capacity, "b_0": wake}, where=[("g", "θ(b/b_0)")], words=_motor_words(n))
    return with_meta(part, part="motor", kind="push of some inputs", glyph="motor", turns="q", powered="b",
                     formula=f"g*τ*dq - {_each('g*τ*(g*τ + P)', n)}/C*db, g = θ(b/b_0)")


def motor_at_rate(capacity: float, wake: float, label: str = "motor at a rate", width: int = 1) -> Arrangement:
    """The motor told a rate ``u`` (note ``sec.general``, the paragraph after ``eqn.motor``):
    inputs ``(b, q, u, P)``. It reads the battery and drives its store at the rate ``g u``,
    ``g = live(b, wake)``, by pushing ``capacity g u - P``; the battery pays
    ``capacity (g u)^2 - P g u``. On an empty battery it drives at rate ``0`` and holds its
    store against ``P``. ``width`` as for ``motor``."""
    if nonpositive(capacity):
        raise ValueError("the driven store's capacity is positive")
    n = width

    def rate(v: Array) -> Array:
        return live(v[0], wake) * v[1 + n:1 + 2 * n]

    def applied(v: Array) -> Array:
        return capacity * rate(v) - v[1 + 2 * n:]

    part = pushes(label, 1 + 3 * n,
                  [("drive", applied, lambda v: v[1:1 + n]),
                   ("pay", lambda v: -jnp.sum(applied(v) * rate(v)), lambda v: v[0])],
                  inputs=[("b", "battery", 1), ("q", "store", n), ("u", "rate", n), ("P", "other load", n)],
                  forms=[("C*g*u - P", "q"), (f"-{_each('(C*g*u - P)*g*u', n)}", "b")],
                  constants={"C": capacity, "b_0": wake}, where=[("g", "θ(b/b_0)")], words=_motor_words(n))
    return with_meta(part, part="motor", kind="push of some inputs", glyph="motor", turns="q", powered="b",
                     formula=f"(C*g*u - P)*dq - {_each('(C*g*u - P)*g*u', n)}*db, g = θ(b/b_0)")


def sweep(capacity: float, wake: float, radius: float, nu: float, gamma: float,
          label: str = "sweep") -> Arrangement:
    """The sweep ``1 -> <5|0>`` (note ``sec.general``): two motors with ``P = 0`` driving two
    stores round the circle of the given radius, at ``nu`` radians per unit of duration,
    returning to that circle at the rate ``gamma``.

    Inputs: the battery ``b``, the stores ``q1, q2``, a gate ``g`` and an extra rate ``e`` on
    the first store. The motors read the battery, so with ``L = live(b, wake)`` the field is
    ``C L r1 dq1 + C L r2 dq2 - C L^2 (r1^2 + r2^2) db`` with
    ``r1 = g (nu q2 + gamma (A^2 - q1^2 - q2^2) q1) + e`` and
    ``r2 = g (-nu q1 + gamma (A^2 - q1^2 - q2^2) q2)``.
    """
    def route(m: Array, n: Array) -> Array:
        b, q1, q2, g, e = n
        ring = radius ** 2 - q1 ** 2 - q2 ** 2
        r1 = g * (nu * q2 + gamma * ring * q1) + e
        r2 = g * (-nu * q1 + gamma * ring * q2)
        zero = jnp.zeros_like(b)
        return jnp.stack([b, q1, capacity * r1, zero, b, q2, capacity * r2, zero])

    return with_meta(fill([motor(capacity, wake, "motor 1"), motor(capacity, wake, "motor 2")],
                          transfer(label, 8, 0, 5, 0, route)),
                     part="sweep", kind="push of some inputs",
                     inputs=[("b", "battery", 1), ("q_1", "store 1", 1), ("q_2", "store 2", 1),
                             ("g", "gate", 1), ("e", "extra rate", 1)],
                     constants={"C": capacity, "b_0": wake, "A": radius, "ν": nu, "γ": gamma},
                     words={"C": "the capacity of each store", "A": "the radius of the circle they go round",
                            "ν": "how many radians they go round per unit of duration",
                            "γ": "how fast they return to that circle", "q_1": "the first store",
                            "q_2": "the second", "e": "a further rate for the first store"},
                     formula="two motors told the pushes C*r_1 and C*r_2, where r_1 = g*(ν*q_2 + γ*(A**2 - q_1**2 - q_2**2)*q_1) + e "
                             "and r_2 = g*(-ν*q_1 + γ*(A**2 - q_1**2 - q_2**2)*q_2)")
