"""Arrangements whose wires may carry a function: the book's arrangements with smooth interfaces (Ch 4, ``\\smoothimc``, the
paragraph *Arrangements with smooth interfaces*), beside the array core of ``arrangement.py`` (book plan P76 (i)).

WHAT IS HERE. In the array core an interface side is ``R^d``, recorded by ``d``. Here a side is a finite list of factors, each
``R(n)`` or ``Maps(a, b)``, the smooth set of smooth maps ``R^a -> R^b`` (Ch 4 ``prop.smoothset_closed``). A point of a side
is a tuple with one entry per factor: an array of shape ``(n,)``, or a JAX-traceable Python function from arrays of shape
``(a,)`` to arrays of shape ``(b,)`` (to scalars when ``b = 1``). An arrangement ``<M^-|M^+> -> <N^-|N^+>`` is, as in the book,

    P      an integrator, exactly the array core's (``coalgebra.py``): its configuration is ``R^{P.Q_dim}``, an array
    out_f  (q, m_out)       -> n_out        the out-transfer
    in_f   (q, m_out, n_in) -> m_in         the in-transfer
    omega  an inclination field: a 1-form on ``Q x M^+ x N^-``.

HOW A 1-FORM ON A SMOOTH SET IS HELD. The book's definition (Ch 4 l.511): a 1-form on a smooth set ``X`` is a map of smooth sets
``X -> Omega``, so it sends each plot of ``X`` of shape ``U`` to a 1-form on ``U``, compatibly with reparametrizing the plot. A
*plot* here is a JAX-traceable Python function from a pytree of arrays ``u`` (a point of the Cartesian space ``U``) to a point
of ``X``. So ``omega(plot, u)`` returns the covector at ``u`` of the 1-form on ``U`` that ``omega`` assigns to ``plot``: a pytree
of arrays shaped like ``u``. For a cost function this is the differential of the cost along the plot (``exact``). A plot of
``Maps(a, b)`` of shape ``U`` is a smooth map ``U x R^a -> R^b`` (``prop.smoothset_closed``), which is what a Python function
returning a function is; JAX differentiates through it.

WHAT A LOAD ON A FIELD WIRE IS. A covector at a point of a smooth set is a family, one covector for each plot through the point,
compatible under reparametrization (the book's ``X-technical-holding.tex``, ``def.cot``). A part that sends a value forward
presents it by a plot, and what comes back is that family's component on that plot: a covector on the plot's domain. That is
all the part needs, by ``def.cot``'s formula for the backward map of ``cot(f)`` (the family at ``f . p`` restricted). On a
Cartesian wire the array core's covector in ``R^n`` is the component on the identity plot. Every plot's domain is built from
configurations and Cartesian values, so what reaches an integrator is an array on its configuration: THE DYNAMICS FUNCTOR
STILL ONLY STEPS ARRAYS, by ``pcK.rounds``, the code the array core's ``PhiK`` runs.

COMPOSITION TAKES NO DERIVATIVE. Pullback of a 1-form is precomposition on plots and addition is plotwise (Ch 4 l.511), so the
composite's field on a plot ``c`` is ``omega_g(A . c) + omega_f(B . c)``, with ``A`` and ``B`` the locating maps of
``wiring.compose_seq``. The ``jax.vjp`` of the array core's ``compose_seq`` is not here. Derivatives are taken where a cost
function gives its 1-form, and where an array arrangement's field is brought in by ``include``.

THE INCLUSION. ``include`` sends an array ``Arrangement`` to a ``SmoothArrangement``: ``<R^a|R^b>`` to ``<(R(a))|(R(b))>`` (``R^0``
to the empty side), the same integrator and transfers, and the field pulled back along each plot. It preserves composites and
identities on the nose and tensors up to ``join``, the static arrangement between ``(R(a), R(b))`` and ``(R(a+b))``;
``tests/test_smooth.py`` checks each law against the array core.

WHAT THIS DOES NOT COVER. Not every smooth set: the sides are products of Cartesian spaces and of mapping spaces between
Cartesian spaces, and a map of smooth sets is given by a function on points. That is enough for a smooth set that is
*concrete* (its plots are functions into its points; Ch 4 l.584) and whose points Python can hold as arrays, functions of
arrays and tuples of those; mapping spaces of mapping spaces would need only a new factor type. A smooth set that is not
concrete, such as ``Omega`` itself, has plots that are not functions into points, and would need maps given on plots. And an
integrator's configuration and states stay ``R^d``, as in ``\\smoothimc``; the further extension of the book's
``rmk.smooth_set_arrangements`` is not built.

Claude Fable 5.2, session 7328b8c7, 2026-11-10.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Sequence, Tuple, Union

import jax
import jax.numpy as jnp
from jax import tree_util as jtu

from .arrangement import Arrangement
from .coalgebra import CotCoalgK, iterate
from .pcK import PCMorphismK, rounds
from .wiring import trivial_parameter


# ---------------------------------------------------------------------------
# Sides: finite products of Cartesian spaces and mapping spaces.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class R:
    """The Cartesian space ``R^n``. A point is an array of shape ``(n,)``."""
    n: int

    def __repr__(self):
        return f"R({self.n})"


@dataclass(frozen=True)
class Maps:
    """The smooth set of smooth maps ``R^a -> R^b``. A point is a JAX-traceable function from arrays of shape ``(a,)`` to arrays of
    shape ``(b,)`` (to scalars when ``b = 1``)."""
    a: int
    b: int = 1

    def __repr__(self):
        return f"Maps({self.a},{self.b})"


Side = Tuple[Union[R, Maps], ...]


def cartesian(d: int) -> Side:
    """The side the array core's ``R^d`` goes to: ``(R(d),)``, and the empty side for ``d = 0``."""
    return (R(d),) if d else ()


def _add(a, b):
    return jtu.tree_map(lambda x, y: x + y, a, b)


def _zero(u):
    return jtu.tree_map(jnp.zeros_like, u)


# ---------------------------------------------------------------------------
# Arrangements and 1-forms.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SmoothArrangement:
    """An arrangement ``<M_in|M_out> -> <N_in|N_out>`` with smooth interfaces (module docstring). ``omega(plot, u)``: ``plot`` sends
    a pytree of arrays to a triple ``(q, m_out, n_in)``, and the result is a covector at ``u``, a pytree shaped like ``u``."""

    P: Any
    M_in: Side
    M_out: Side
    N_in: Side
    N_out: Side
    out_f: Callable   # (q, m_out)        -> n_out
    in_f: Callable    # (q, m_out, n_in)  -> m_in
    omega: Callable   # (plot, u)         -> covector at u
    label: str = ""

    @property
    def source(self) -> Tuple[Side, Side]:
        return (self.M_in, self.M_out)

    @property
    def target(self) -> Tuple[Side, Side]:
        return (self.N_in, self.N_out)


def exact(U: Callable) -> Callable:
    """The 1-form ``dU`` of a function ``U(q, m_out, n_in) -> R`` on ``Q x M^+ x N^-``: on a plot, the differential of ``U`` along it.
    A cost function ``k`` has ``U = -k``, as in ``arrangement.exact``."""

    def omega(plot, u):
        return jax.grad(lambda v: jnp.asarray(U(*plot(v)), float))(u)

    return omega


def zero_form(plot, u):
    """The zero 1-form."""
    return _zero(u)


def identity(side_in: Side, side_out: Side) -> SmoothArrangement:
    """The identity on ``<side_in|side_out>``: the point integrator, identity transfers, the zero field."""
    return SmoothArrangement(P=trivial_parameter(), M_in=side_in, M_out=side_out, N_in=side_in, N_out=side_out,
                             out_f=lambda q, m: m, in_f=lambda q, m, n: n, omega=zero_form, label="id")


def compose_seq(f: SmoothArrangement, g: SmoothArrangement) -> SmoothArrangement:
    """``g . f``: the integrators tensored, the transfers composed as lenses, and the two fields precomposed with the maps that
    locate each factor's point, then added (``wiring.compose_seq``, with precomposition on plots where that has ``jax.vjp``)."""
    if f.target != g.source:
        raise ValueError(f"compose_seq: codomain of f <{f.N_in}|{f.N_out}> != domain of g <{g.M_in}|{g.M_out}>")
    nf = f.P.Q_dim

    def out_f(q, l_out):
        return g.out_f(q[nf:], f.out_f(q[:nf], l_out))

    def in_f(q, l_out, n_in):
        return f.in_f(q[:nf], l_out, g.in_f(q[nf:], f.out_f(q[:nf], l_out), n_in))

    def locate_g(q, l_out, n_in):
        return q[nf:], f.out_f(q[:nf], l_out), n_in

    def locate_f(q, l_out, n_in):
        return q[:nf], l_out, g.in_f(q[nf:], f.out_f(q[:nf], l_out), n_in)

    def omega(plot, u):
        return _add(g.omega(lambda v: locate_g(*plot(v)), u), f.omega(lambda v: locate_f(*plot(v)), u))

    return SmoothArrangement(P=f.P.tensor(g.P), M_in=f.M_in, M_out=f.M_out, N_in=g.N_in, N_out=g.N_out,
                             out_f=out_f, in_f=in_f, omega=omega, label=f"({g.label} . {f.label})")


def parallel(p1: SmoothArrangement, p2: SmoothArrangement) -> SmoothArrangement:
    """The monoidal product: sides concatenated, integrators tensored, each field precomposed with its projection, then added."""
    n1, a, b = p1.P.Q_dim, len(p1.M_out), len(p1.N_in)

    def first(q, m_out, n_in):
        return q[:n1], m_out[:a], n_in[:b]

    def second(q, m_out, n_in):
        return q[n1:], m_out[a:], n_in[b:]

    def out_f(q, m_out):
        return tuple(p1.out_f(q[:n1], m_out[:a])) + tuple(p2.out_f(q[n1:], m_out[a:]))

    def in_f(q, m_out, n_in):
        return tuple(p1.in_f(*first(q, m_out, n_in))) + tuple(p2.in_f(*second(q, m_out, n_in)))

    def omega(plot, u):
        return _add(p1.omega(lambda v: first(*plot(v)), u), p2.omega(lambda v: second(*plot(v)), u))

    return SmoothArrangement(P=p1.P.tensor(p2.P), M_in=p1.M_in + p2.M_in, M_out=p1.M_out + p2.M_out,
                             N_in=p1.N_in + p2.N_in, N_out=p1.N_out + p2.N_out,
                             out_f=out_f, in_f=in_f, omega=omega, label=f"({p1.label} || {p2.label})")


def tensor(parts: Sequence[SmoothArrangement]) -> SmoothArrangement:
    """The fold of ``parallel`` (state left-nested, as ``wiring.tensor_arrangements``)."""
    if len(parts) < 1:
        raise ValueError("tensor: need at least one factor")
    acc = parts[0]
    for p in parts[1:]:
        acc = parallel(acc, p)
    return acc


# ---------------------------------------------------------------------------
# The inclusion of the array core.
# ---------------------------------------------------------------------------


def _pack(d: int, x):
    return (x,) if d else ()


def _unpack(d: int, v):
    return v[0] if d else jnp.zeros(0)


def include(arr: Arrangement) -> SmoothArrangement:
    """The inclusion of the array core: ``<R^a|R^b>`` to ``<(R(a))|(R(b))>``, the same integrator and transfers, and the field
    ``(plot, u) |-> (T_u plot)^T omega(plot(u))``, the pullback of the array core's 1-form along the plot."""
    aM, bM, aN, bN = arr.in_dim_M, arr.out_dim_M, arr.in_dim_N, arr.out_dim_N

    def flat(q, m_out, n_in):
        return q, _unpack(bM, m_out), _unpack(aN, n_in)

    def omega(plot, u):
        at, pull = jax.vjp(lambda v: flat(*plot(v)), u)
        (xi,) = pull(tuple(arr.omega(*at)))
        return xi

    return SmoothArrangement(
        P=arr.P, M_in=cartesian(aM), M_out=cartesian(bM), N_in=cartesian(aN), N_out=cartesian(bN),
        out_f=lambda q, m_out: _pack(bN, arr.out_f(q, _unpack(bM, m_out))),
        in_f=lambda q, m_out, n_in: _pack(aM, arr.in_f(*flat(q, m_out, n_in))),
        omega=omega, label=arr.label)


def join(ins: Sequence[int], outs: Sequence[int]) -> SmoothArrangement:
    """The comparison of the inclusion on a tensor: the static arrangement from the side-by-side interface
    ``<(R(a_1), ..., R(a_k)) | (R(b_1), ..., R(b_l))>`` to ``<(R(a_1 + ... + a_k)) | (R(b_1 + ... + b_l))>``. It concatenates what
    goes out and splits what comes in; dimensions 0 are skipped. An isomorphism, with zero field."""
    ins, outs = [a for a in ins if a], [b for b in outs if b]
    cuts = [sum(ins[:k]) for k in range(len(ins) + 1)]

    def out_f(q, m_out):
        return _pack(sum(outs), jnp.concatenate(m_out) if m_out else jnp.zeros(0))

    def in_f(q, m_out, n_in):
        return tuple(n_in[0][cuts[k]:cuts[k + 1]] for k in range(len(ins)))

    return SmoothArrangement(P=trivial_parameter(), M_in=tuple(R(a) for a in ins), M_out=tuple(R(b) for b in outs),
                             N_in=cartesian(sum(ins)), N_out=cartesian(sum(outs)),
                             out_f=out_f, in_f=in_f, omega=zero_form, label="join")


# ---------------------------------------------------------------------------
# A function read out, and evaluation (Ch 4 thm.smooth_arr_closed, in the case of ex.learner_currying).
# ---------------------------------------------------------------------------


def evaluate(a: int, b: int) -> SmoothArrangement:
    """Evaluation ``<(R(a))|(Maps(a, b))> (x) <()|(R(a))> -> <()|(R(b))>``: it transfers out the function's value at the datum, and
    transfers the datum in to whatever reads out the function. The point integrator, zero field (Ch 4
    ``eqn.smooth_arr_evaluation``, with ``A = 1``, ``C = 1``)."""
    return SmoothArrangement(P=trivial_parameter(), M_in=(R(a),), M_out=(Maps(a, b), R(a)), N_in=(), N_out=(R(b),),
                             out_f=lambda q, m: (jnp.reshape(m[0](m[1]), (b,)),), in_f=lambda q, m, n: (m[1],),
                             omega=zero_form, label="ev")


def reads_out_its_function(phi: SmoothArrangement) -> SmoothArrangement:
    """The curry of ``phi : <()|(R(a))> -> <()|(R(b))>`` (Ch 4 ``ex.learner_currying``): the system ``1 -> <(R(a))|(Maps(a, b))>`` with
    the same integrator that reads out the function ``x |-> out_f(q, x)``; its field is ``phi``'s with the datum moved from the
    internal output to the external input (``eqn.smooth_arr_curry``'s ``sigma``)."""
    if phi.M_in or phi.N_in or len(phi.M_out) != 1 or len(phi.N_out) != 1:
        raise ValueError("reads_out_its_function: expected <()|(R(a))> -> <()|(R(b))>")
    (A,), (B,) = phi.M_out, phi.N_out

    def sigma(q, m_out, n_in):
        return q, n_in, ()

    return SmoothArrangement(P=phi.P, M_in=(), M_out=(), N_in=(A,), N_out=(Maps(A.n, B.n),),
                             out_f=lambda q, m: (lambda x: phi.out_f(q, (x,))[0],), in_f=lambda q, m, n: (),
                             omega=lambda plot, u: phi.omega(lambda v: sigma(*plot(v)), u), label=f"curry({phi.label})")


# ---------------------------------------------------------------------------
# The dynamics functor.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BoxPoly:
    """The polynomial of an interface ``<inn|out>``: a position is a plotted point of ``out`` with a 1-form on ``inn``; a direction
    is a covector on that plot's domain with a point of ``inn``."""
    out: Side
    inn: Side


def point(value) -> Tuple[Callable, Any]:
    """A point as a plotted point: the constant plot, of shape ``R^0``."""
    return (lambda u: value, ())


def at(value) -> Tuple[Callable, Any]:
    """A point of a Cartesian side (a tuple of arrays) as a plotted point: the identity plot. The covector that comes back on it
    is the array core's."""
    return (lambda u: u, value)


def interpretation(arr: SmoothArrangement) -> Callable:
    """``q |-> (position_action, direction_action)``, as ``interpretation.interpretation`` for the array core.

    A position ``(out_m, omega_M)`` of the source: ``out_m = (p, u)``, a plot of ``M^+`` and a point of its domain, and ``omega_M``
    a 1-form on ``M^-`` (a function of a plot and a point, as everywhere here). The position it gives of the target is the
    value ``out_f(q, p(u))`` presented by the plot ``(q', u') |-> out_f(q', p(u'))`` at ``(q, u)``, with the 1-form on ``N^-``

        omega_N = (in_f(q, m, -))^* omega_M + omega(q, m, -),

    both pullbacks being precompositions. A direction ``(xi_N, in_n)`` of the target: ``xi_N`` a covector on ``(q, u)``, the
    component on that plot of a covector at the value, and ``in_n`` a point of ``N^-``. The parameter covector is the ``q`` part of

        xi_N + (in_f(-, -, in_n))^* omega_M + omega(-, -, in_n),

    all three covectors on ``(q, u)``; its ``u`` part goes back to the source, with ``in_m = in_f(q, p(u), in_n)``."""

    def at_position(q):
        def position_action(out_m, omega_M):
            p, u = out_m
            m = p(u)

            def omega_N(c, v):
                return _add(omega_M(lambda x: arr.in_f(q, m, c(x)), v), arr.omega(lambda x: (q, m, c(x)), v))

            return (lambda w: arr.out_f(w[0], p(w[1])), (q, u)), omega_N

        def direction_action(out_m, omega_M, xi_N, in_n):
            p, u = out_m
            w = (q, u)
            xi = _add(xi_N, omega_M(lambda x: arr.in_f(x[0], p(x[1]), in_n), w))
            xi = _add(xi, arr.omega(lambda x: (x[0], p(x[1]), in_n), w))
            return xi[0], xi[1], arr.in_f(q, p(u), in_n)

        return position_action, direction_action

    return at_position


def PhiK(arr: SmoothArrangement) -> PCMorphismK:
    """The dynamics functor on ``arr``: its interpretation, stepped by ``pcK.rounds``, the array core's own rounds. A one-stage
    parameter is taken as its one-round lift."""
    P = arr.P if isinstance(arr.P, CotCoalgK) else iterate(arr.P, 1)
    return rounds(P, interpretation(arr), BoxPoly(arr.M_out, arr.M_in), BoxPoly(arr.N_out, arr.N_in))


# What a closed arrangement is shown and told: nothing.
CLOSED_IN = (point(()), zero_form)


def closed_tell(out_pos):
    """The direction a closed arrangement is told: the zero covector on its position's plot, and the empty point."""
    (_plot, w), _omega = out_pos
    return _zero(w), ()
