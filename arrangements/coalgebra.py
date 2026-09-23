"""``cot(Q)``-coalgebras: the parameters of arrangements (sec.integrators).

An object of ``cotcoalg`` is a triple ``(Q, S, u)`` of smooth sets ``Q`` (the
base) and ``S`` (the state) with a ``cot(Q)``-coalgebra

    u : S -> cot(Q)(S),    s |-> (q(s), xi |-> s'),

i.e. each state presents a base position ``q(s)`` and, on receiving a covector
``xi in T^*_{q(s)} Q``, moves to a new state. This is the book's *integrator*
("u for update", NOTATION.md): in def:arr the coalgebra ``u``, and hence the
integrator, is part of an arrangement -- there is no separate integrator
semantics to choose later, as there was in CDLM (``Phiconf``/``Phiphase``).
The response ``sharp`` is not a separate field either; it lives inside ``step``.

In v1 the smooth sets are Cartesian spaces ``R^d`` (the admissible subcategory
of STATUS.md §3.1), recorded by their dimension; a state ``s in S`` is
whatever pytree the constructor chose (an array for ``configuration``, a pair
for ``phase``, a pair of states for ``tensor``).

The standard coalgebras take a reactive vector space ``(Q, sharp_Q)`` as input
(rvect.py):

* ``configuration(Q)`` -- state ``S = Q``, position the identity, step
  ``s + sharp_s(xi)`` (sec.displacement): the integrator *advances* along the
  forcing, whose sign (``omega = dU = -dk``) already makes it descend a cost.
* ``phase(Q)`` -- state ``S = T^*Q = Q x Q^*``, position the presented position
  ``q + sharp_q(xi)``, step one symplectic-Euler step of Hamilton's equations.

The remaining constructors are EXTENSIONS (beyond the paper, notes/DESIGN.md §4), ported
from CDLM's ``integrator.py`` with their status kept verbatim:

* ``damped(Q, c)`` -- the phase coalgebra with the damping one-form ``c*zeta``;
* ``gyro(Q, ...)`` -- the phase coalgebra with damping, a gyroscopic (skew) term
  and a quadratic-drag term smuggled into the step (the *trick* realization; the
  direct realization is a non-exact forcing on the enlarged base ``T^*Q``, see
  ``phase_on_cotangent`` and tests/test_drag_one_form.py);
* ``CotCoalg2`` / ``CotCoalgK`` -- two- and ``K``-stage coalgebras
  ``S -> cot(Q)^{<|K}(S)`` (CDLM rmk.multistage), with ``leapfrog``, ``rk4``,
  ``rk4_gyro`` as instances; their dynamics land in ``pc^(K)`` (pc2.py, pcK.py).
  Whether ``arr -> pc^(K)`` is a lax monoidal *functor* is conjectural (the
  monoidality is open); ``tensor`` is provided so wirings compose, nothing more.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Tuple

import jax.numpy as jnp
from jax import Array

from .rvect import ReactiveVectorSpace


@dataclass(frozen=True)
class CotCoalg:
    """A ``cot(Q)``-coalgebra ``(Q, S, u)`` (sec.integrators), with an initial state.

    * ``Q_dim``    -- the base ``Q = R^{Q_dim}``.
    * ``init``     -- the initial state ``s_0 in S`` (the state ``Phi`` starts from).
    * ``position`` -- ``S -> Q``, the base position ``q(s)`` the state presents.
    * ``step``     -- ``(S, xi_Q) -> S``, the new state after receiving the
                      covector ``xi_Q in T^*_{q(s)} Q``.

    Together ``position`` and ``step`` are ``u : S -> cot(Q)(S)``; the response
    ``sharp`` is inside ``step``.
    """

    Q_dim: int
    init: Any
    position: Callable[[Any], Array]
    step: Callable[[Any, Array], Any]
    label: str = ""

    def tensor(self, other: "CotCoalg") -> "CotCoalg":
        """The parallel product ``(Q, S, u) (x) (Q', S', u')`` (sec.integrators).

        Base ``Q x Q'``, state ``S x S'`` (a pair), and the product coalgebra
        induced by ``cot(Q x Q') ~= cot(Q) (x) cot(Q')``: the position is the
        concatenation, and a covector on ``Q x Q'`` is split and handed to each
        factor's ``step``.
        """
        n = self.Q_dim

        def position(s):
            s1, s2 = s
            return jnp.concatenate([self.position(s1), other.position(s2)])

        def step(s, xi_Q):
            s1, s2 = s
            return (self.step(s1, xi_Q[:n]), other.step(s2, xi_Q[n:]))

        return CotCoalg(
            Q_dim=n + other.Q_dim,
            init=(self.init, other.init),
            position=position,
            step=step,
            label=f"({self.label} (x) {other.label})",
        )


# ---------------------------------------------------------------------------
# The standard coalgebras.
# ---------------------------------------------------------------------------


def configuration(Q: ReactiveVectorSpace, init: Array = None) -> CotCoalg:
    """The configuration coalgebra on ``(Q, sharp_Q)``: ``S = Q``, one displacement per step.

    The state is the position, ``q(s) = s``, and an incoming covector displaces it
    *along* the forcing (sec.displacement, STATUS.md §1):

        s |-> s + sharp_s(xi_Q).

    With ``omega = dU = -dk`` this descends the cost ``k``; CDLM's
    ``q - sharp(dk)`` is the same update after translating conventions.
    """

    s0 = jnp.zeros(Q.dim) if init is None else jnp.asarray(init, dtype=float)
    return CotCoalg(
        Q_dim=Q.dim,
        init=s0,
        position=lambda s: s,
        step=lambda s, xi_Q: s + Q.apply_sharp(s, xi_Q),
        label="conf",
    )


def phase(Q: ReactiveVectorSpace, init=None) -> CotCoalg:
    """The phase coalgebra on ``(Q, sharp_Q)``: ``S = T^*Q = Q (+) Q^*`` in coordinates.

    The state is ``s = (q, xi)``, position and momentum. The presented position is

        q~ = q + sharp_q(xi),

    the position the step moves to; forces are evaluated there, which is what
    makes the step symplectic (semi-implicit Euler) rather than explicit. One
    step of Hamilton's equations, with the velocity read off the stored momentum
    and the covector *added* (the book's sign, NOTATION.md "Displacement map"):

        (q, xi) |-> (q + sharp_q(xi), xi + xi_Q),   xi_Q computed at q~.

    CDLM's phase integrator subtracts its ``xi_Q``; the two agree once
    ``U_cdlm`` becomes ``omega = -dU_cdlm``.
    """

    s0 = (jnp.zeros(Q.dim), jnp.zeros(Q.dim)) if init is None else init
    return CotCoalg(
        Q_dim=Q.dim,
        init=s0,
        position=lambda s: s[0] + Q.apply_sharp(s[0], s[1]),
        step=lambda s, xi_Q: (s[0] + Q.apply_sharp(s[0], s[1]), s[1] + xi_Q),
        label="phase",
    )


# ---------------------------------------------------------------------------
# EXTENSIONS (beyond the paper): damping, gyroscopic and drag terms, ported from
# CDLM's ``integrator.py`` with the sign of STATUS.md §1 (the covector is added).
# ---------------------------------------------------------------------------


def damped(Q: ReactiveVectorSpace, damping: float = 0.0, init=None) -> CotCoalg:
    """The phase coalgebra with the paper's *damping* 1-form added.

    The damping 1-form ``ex.damping_one_form`` -- ``zeta(q, xi) = (xi, 0)`` -- is
    *present* in the paper (where it "plays no further role"); that example already
    records that for a coefficient ``c in [0,1]`` the scalar multiple ``c*zeta``
    "removes a fraction c of the momentum at each step". This integrator is that
    line put to work: the only step beyond the paper's formal development is wiring
    the 1-form into an integrator to make an optimizer (README "Extensions").

    The phase readout combines the kinetic 1-form ``beta`` with the *damping*
    1-form ``c * zeta``, ``zeta(q, xi) = (xi, 0)`` (a monoidal 1-form by
    prop.one_forms_vector_space). Fed through the symplectic sharp it adds
    ``-c * xi`` to the momentum, so the Hamilton step (eqn.phase_update) gains a
    friction term (the book's sign: the covector is *added*):

        (q, xi) |-> (q + sharp_q(xi),  (1 - c) * xi + xi_Q),   xi_Q at q~.

    At ``c = 0`` this is the conservative phase integrator (oscillates forever);
    at ``0 < c < 1`` it dissipates a fraction ``c`` of the momentum each step --
    heavy-ball *momentum* that converges; as ``c -> 1`` it collapses toward plain
    descent (the configuration integrator). The integrator alone interpolates
    between the optimizers, exactly as it does between wave and heat dynamics.
    """

    c = float(damping)
    s0 = (jnp.zeros(Q.dim), jnp.zeros(Q.dim)) if init is None else init
    return CotCoalg(
        Q_dim=Q.dim,
        init=s0,
        position=lambda s: s[0] + Q.apply_sharp(s[0], s[1]),
        step=lambda s, xi_Q: (s[0] + Q.apply_sharp(s[0], s[1]), (1.0 - c) * s[1] + xi_Q),
        label=f"damped({c:g})",
    )


def quadratic_drag_kick(v, gyro_block: int):
    """Per-gyro quadratic-drag kick ``|v_i| v_i`` in coordinates (block Euclidean norm).

    The blog's air drag is a force ``~ -|v| v`` (opposes velocity, magnitude growing
    with speed squared). A 1-form on ``T^*Q`` takes values in ``Q^* (+) Q``, and it is
    the ``Q^*``-component that drives momentum, so the drag 1-form is

        omega_drag(q, xi) = (|v| flat(v),  0) : Q^* (+) Q,   v = sharp_q(xi),

    with ``|v|`` taken *per gyro* (the Euclidean norm of each ``gyro_block``-dimensional
    block) and ``flat : Q -> Q^*`` the Euclidean flat -- ``|v| v`` on its own is a
    *vector*, not the covector the momentum slot takes. This function returns the kick
    in coordinates, i.e. with ``flat`` read as the identity matrix; against the flat
    ``sharp^{-1} = m`` that the kinetic term uses, that is off by the constant factor
    ``m``, which the trainable coefficient ``drag`` multiplying it absorbs.

    Naturality grounding (rmk.adam -- the monoidality is non-negotiable, the
    naturality may be restricted):

    * MONOIDAL over ``(+)``: each gyro drags only on its own velocity, so
      ``omega_drag`` is block-diagonal, ``omega_{Q(+)W} = omega_Q (+) omega_W``.
      rmk.adam: "The monoidal requirement of integrators cannot be negotiated ...
      each box can compute its own update." Quadratic drag passes this.
    * NATURAL only over a *subcategory*: both the norm ``|v_i|`` and the flat come from
      the Euclidean structure of each gyro block, so both are equivariant under per-gyro
      orthogonal maps (``O(2)`` per 2-D gyro) but not under a general sharp-equivariant
      ``rvect``-isomorphism. rmk.adam explicitly allows this -- such a section "is not
      outside the framework; it just lives over a smaller ``Q``" -- here the
      subcategory of Euclidean gyro-spaces with block-orthogonal isomorphisms, the
      *top* of the paper's chain ``orthogonal > signed perms > coord perms`` (drag is
      thus *more* natural than Adam, which the paper places over signed permutations).
      Restricting ``Q`` keeps thm.dynamics_functor, so this remains a genuine integrator.
    """
    vg = v.reshape(-1, gyro_block)
    sq = jnp.sum(vg * vg, axis=1, keepdims=True)  # |v_i|^2 per gyro
    # Double-``where`` so the gradient of ``|v_i|`` is finite (its true value 0) at rest
    # ``v_i = 0``, without perturbing the exact value ``|v_i| v_i`` anywhere else: a
    # quadratic drag IS differentiable at zero velocity, but a naive ``sqrt`` gives 0/0.
    speed = jnp.where(sq > 0.0, jnp.sqrt(jnp.where(sq > 0.0, sq, 1.0)), 0.0)
    return (speed * vg).reshape(v.shape)


def gyro_kick(v, gamma=0.0, J=None, drag: float = 0.0, gyro_block=None):
    """The momentum kick of the gyroscopic and drag 1-forms at velocity ``v``:
    ``-drag * |v| v  -  gamma * (J v)`` (zero when neither is present). Shared by the
    trick realization (``gyro``, ``rk4_gyro``) and the direct one (``phase_on_cotangent``)."""
    kick = jnp.zeros_like(v)
    if gyro_block is not None:
        kick = kick - drag * quadratic_drag_kick(v, gyro_block)  # quadratic air drag
    if J is not None:
        kick = kick - gamma * (jnp.asarray(J, float) @ v)       # gyroscopic precession (skew, no work)
    return kick


def gyro(
    Q: ReactiveVectorSpace, damping: float = 0.0, gamma: float = 0.0, J=None,
    drag: float = 0.0, gyro_block=None, init=None,
) -> CotCoalg:
    """The phase coalgebra with a *damping* and a *gyroscopic* 1-form added.

    EXTENSION (beyond the paper). Like ``damped`` this lives in the readout slot
    ``nu_{sigma^1, omega}`` (sec.phase_integrators), but adds, to the kinetic
    1-form ``beta`` and the damping 1-form ``c*zeta``, a third, *skew* 1-form --
    the gyroscopic term of a spinning rotor. With a complex structure ``J`` (a
    block-diagonal 90-degree rotation, one ``[[0,-1],[1,0]]`` block per 2-D gyro),
    define

        omega_gyro(q, xi) = (gamma * flat(J @ sharp_q(xi)),  0) : Q^* (+) Q,

    where ``flat : Q -> Q^*`` is the Euclidean flat: ``J @ sharp_q(xi)`` is a vector
    in ``Q``, and the momentum slot of a 1-form takes a covector. As with the drag term
    below, the code computes this in coordinates (``flat`` read as the identity), which
    rescales it by a constant absorbed into ``gamma``.

    Fed through the symplectic sharp (which sends ``(a, b) |-> (b, -a)``,
    eqn.canonical_sharp) it adds ``-gamma * J @ sharp_q(xi)`` to the momentum: a
    force perpendicular to the velocity (``(J v) . v = 0``, rmk.symplectic_perpendicular).
    In continuous time such a skew force does no work -- gyroscopic precession rotates
    the velocity without changing its length -- and, unlike the potential, it is
    velocity-dependent, so it cannot be a ``dU``. Beware: this *explicit* discretization
    of the skew term is not symplectic; it mildly injects energy each step and is meant
    to be run with the damping ``c`` (as in ``dap/gyroscope.py``), which keeps it
    bounded. Combined with damping the Hamilton step (eqn.phase_update) becomes, in the
    book's sign (``xi_Q`` added),

        (q, xi) |-> ( q + sharp_q(xi),
                      (1 - c) * xi + xi_Q - gamma * J @ sharp_q(xi) ),   xi_Q at q~.

    Caveat (why it is beyond the paper): ``omega_gyro`` is a monoidal 1-form only over
    a subcategory of ``rvect``, and it is the flat that costs the naturality, not ``J``
    alone: the isomorphisms must be per-gyro isometries (to be compatible with ``flat``)
    *and* commute with ``J``, i.e. per-gyro ``SO(2)`` -- one step below the drag term's
    ``O(2)``, since a reflection anticommutes with ``J``. The paper's own 1-forms
    (``beta``, ``zeta``) are written with no such identification and keep the full
    reactive symmetry. It is exactly the minimal example that lives
    in ``rvect`` but outside the *harmonic* (symmetric-sharp) regime of
    def.arrangement_terminology. (The same rmk.adam principle that justifies the
    quadratic-drag term below applies: ``omega_gyro`` "lives over a smaller ``Q``".)

    With ``gyro_block`` set, a fourth, *quadratic-drag* 1-form is added (the blog's
    air drag): ``omega_drag(q, xi) = (|v| v, 0)``, ``v = sharp_q(xi)``, with ``|v|``
    the per-gyro Euclidean norm (see ``quadratic_drag_kick``). It is monoidal over
    ``(+)`` but natural only over per-gyro orthogonal maps -- a "smaller ``Q``" in the
    precise sense rmk.adam sanctions (the chain ``orthogonal > signed perms > coord
    perms``; drag sits at the top, Adam at signed perms). ``drag`` may be trained;
    ``gyro_block`` (its presence and the gyro dimension) is the Python-time choice.

    This is the TRICK realization: the velocity-dependent one-forms are smuggled
    into the coalgebra's step because CDLM's arrangement carried only a potential
    on ``Q``. The book's direct realization puts them where they live, as a
    (non-exact) forcing on the base ``T^*Q``; see ``phase_on_cotangent``.
    """

    # damping ``c``, gyroscopic ``gamma`` and ``drag`` may be traced (they are
    # trainable in the gyroscope emulation), so we do not coerce them to ``float``
    # here; only the *presence* of ``J`` and of ``gyro_block`` are static
    # (Python-time) choices. ``gamma`` may be a scalar or a per-component vector
    # (length ``Q.dim``, e.g. ``repeat(gamma_per_gyro, 2)``) -- the latter, broadcast
    # against ``Jm @ v``, gives each gyro its own precession rate.
    c = damping

    def step(s, xi_Q):
        q, xi = s
        v = Q.apply_sharp(q, xi)  # velocity sharp_q(xi)
        new_xi = (1.0 - c) * xi + xi_Q + gyro_kick(v, gamma, J, drag, gyro_block)
        return (q + v, new_xi)

    s0 = (jnp.zeros(Q.dim), jnp.zeros(Q.dim)) if init is None else init
    return CotCoalg(
        Q_dim=Q.dim,
        init=s0,
        position=lambda s: s[0] + Q.apply_sharp(s[0], s[1]),
        step=step,
        label="gyro",
    )


def phase_on_cotangent(Q: ReactiveVectorSpace, damping: float = 0.0, init=None) -> CotCoalg:
    """The (damped) phase coalgebra presenting its whole state: base ``T^*Q``.

    The DIRECT realization of velocity-dependent forcings (notes/DESIGN.md §6(c)): base
    ``Q' = Q x Q^* = R^{2 dim}``, state ``(q, xi)``, presented position
    ``(q~, xi)`` with ``q~ = q + sharp_q(xi)``, and step the damped Hamilton step
    driven by the ``Q``-component of the incoming covector on ``Q'`` (its ``Q^*``-
    component, a force on momentum, is discarded: the state's momentum slot is
    driven by ``Q``-covectors only, exactly as in ``damped``):

        (q, xi) |-> (q + sharp_q(xi), (1 - c) * xi + xi_Q).

    An arrangement with this parameter carries its gyroscopic and drag terms as a
    genuine forcing ``omega(q', m, n)`` on ``Q' x M^+ x N^-`` whose ``Q``-component
    is ``gyro_kick(sharp(xi))`` -- velocity-dependent, hence not ``dU`` for any
    ``U`` -- instead of inside the step as ``gyro`` does. For a constant sharp the
    two agree step for step (tests/test_drag_one_form.py, tests/test_gyroscope.py).
    """

    c = float(damping)
    n = Q.dim

    def position(s):
        q, xi = s
        return jnp.concatenate([q + Q.apply_sharp(q, xi), xi])

    def step(s, xi_Qp):
        q, xi = s
        return (q + Q.apply_sharp(q, xi), (1.0 - c) * xi + xi_Qp[:n])

    s0 = (jnp.zeros(n), jnp.zeros(n)) if init is None else init
    return CotCoalg(Q_dim=2 * n, init=s0, position=position, step=step, label=f"phase_on_T*Q({c:g})")


# ---------------------------------------------------------------------------
# EXTENSIONS: multi-stage coalgebras ``S -> cot(Q)^{<|K}(S)`` (CDLM rmk.multistage).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CotCoalg2:
    """A *two-stage* ``cot(Q)``-coalgebra: operationally ``Store∘S ⇒ cot^{◁2}`` (rmk.multistage).

    The multi-stage form the remark proposes (``upd: Store∘S ⇒ p^{◁2}`` with
    ``p = cot``): two emit/receive rounds before the state updates. As callables
    (the reactive vector space is bound at construction, as for ``CotCoalg``):

    * ``init``                    -- the initial state.
    * ``read1(state)``            -- parameter position emitted in round 1.
    * ``advance(state, xi_Q1)``   -- consume the round-1 covector; return the
                                     intermediate ("inner coalgebra state").
    * ``read2(mid)``              -- parameter position emitted in round 2.
    * ``finish(state, mid, xi_Q2)`` -- consume the round-2 covector; new state.

    ``pc2.Phi2`` turns an arrangement with such a parameter into a ``pc^(2)``
    morphism, exactly as ``functors.Phi`` does for a 1-stage ``CotCoalg``.
    EXTENSION: whether this is a lax monoidal functor is conjectural (the
    monoidality is open).
    """

    Q_dim: int
    init: Any
    read1: Callable[[Any], Array]
    advance: Callable[[Any, Array], Any]
    read2: Callable[[Any], Array]
    finish: Callable[[Any, Any, Array], Any]
    label: str = ""

    def tensor(self, other) -> "CotCoalg2":
        """Parallel product; a single-stage ``other`` is lifted by ``iterate``."""
        other = other if isinstance(other, CotCoalg2) else as_two_stage(other)
        n = self.Q_dim

        def read1(s):
            return jnp.concatenate([self.read1(s[0]), other.read1(s[1])])

        def advance(s, xi):
            return (self.advance(s[0], xi[:n]), other.advance(s[1], xi[n:]))

        def read2(mid):
            return jnp.concatenate([self.read2(mid[0]), other.read2(mid[1])])

        def finish(s, mid, xi):
            return (self.finish(s[0], mid[0], xi[:n]), other.finish(s[1], mid[1], xi[n:]))

        return CotCoalg2(
            Q_dim=n + other.Q_dim, init=(self.init, other.init),
            read1=read1, advance=advance, read2=read2, finish=finish,
            label=f"({self.label} (x) {other.label})",
        )


@dataclass(frozen=True)
class CotCoalgK:
    """A *K-stage* ``cot(Q)``-coalgebra: operationally ``Store∘S ⇒ cot^{◁K}`` (rmk.multistage).

    The general ``ell``-stage form the remark proposes (``upd: Store∘S ⇒ p^{◁ell}``
    with ``p = cot``): ``K`` emit/receive rounds before the macro-state updates. It
    is ``CotCoalg2`` (read1/advance/read2/finish) one dimension up -- ``K`` reads
    and ``K`` advances, the last advance being the ``finish`` that lands in a new
    macro-state. As callables:

    * ``init``                  -- the initial macro-state ``s_0``.
    * ``reads[i](m)``           -- parameter position emitted in round ``i``
                                   (``m`` is the round-``i`` intermediate, ``m = s_0``
                                   in round 0).
    * ``advances[i](m, xi_Q)``  -- consume the round-``i`` covector; return the
                                   next intermediate. For ``i = K-1`` this is the
                                   ``finish``: its result is the new macro-state.

    ``pcK.PhiK`` turns an arrangement with such a parameter into a ``pc^(K)``
    morphism, exactly as ``pc2.Phi2`` does for the two-stage case. The lists
    ``reads`` and ``advances`` must have equal length ``K >= 1``; ``K = 1`` recovers
    the single-stage ``CotCoalg`` (one read, one step). EXTENSION: functoriality
    and monoidality of ``arr -> pc^(K)`` are open.
    """

    Q_dim: int
    init: Any
    reads: Tuple[Callable[[Any], Array], ...]
    advances: Tuple[Callable[[Any, Array], Any], ...]
    label: str = ""

    @property
    def K(self) -> int:
        return len(self.reads)

    def tensor(self, other) -> "CotCoalgK":
        """Parallel product; a single-stage ``other`` is lifted by ``iterate``."""
        other = other if isinstance(other, CotCoalgK) else iterate(other, self.K)
        if other.K != self.K:
            raise ValueError(f"tensor needs equal stage counts, got {self.K} and {other.K}")
        n = self.Q_dim

        def read(i):
            return lambda m: jnp.concatenate([self.reads[i](m[0]), other.reads[i](m[1])])

        def advance(i):
            return lambda m, xi: (self.advances[i](m[0], xi[:n]), other.advances[i](m[1], xi[n:]))

        return CotCoalgK(
            Q_dim=n + other.Q_dim, init=(self.init, other.init),
            reads=tuple(read(i) for i in range(self.K)),
            advances=tuple(advance(i) for i in range(self.K)),
            label=f"({self.label} (x) {other.label})",
        )


def iterate(P: CotCoalg, K: int) -> CotCoalgK:
    """The canonical lift of a 1-stage coalgebra to ``K`` stages: ``K`` steps per
    macro-tick, ``u_K = cot^{<|(K-1)}(u) . ... . cot(u) . u``. ``K = 1`` is ``P`` itself;
    on the trivial parameter ``R^0`` (a wiring's) it is how wirings tensor with
    multi-stage parts."""
    return CotCoalgK(
        Q_dim=P.Q_dim, init=P.init,
        reads=(P.position,) * K, advances=(P.step,) * K,
        label=f"{P.label}-K{K}",
    )


def as_two_stage(P: CotCoalg) -> CotCoalg2:
    """``iterate(P, 2)`` in the ``CotCoalg2`` shape."""
    return CotCoalg2(
        Q_dim=P.Q_dim, init=P.init,
        read1=P.position, advance=P.step, read2=P.position,
        finish=lambda s, mid, xi: P.step(mid, xi),
        label=f"{P.label}-2",
    )
