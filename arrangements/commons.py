"""A commons: three wirings on one cost (``notes/COMMONS-DESIGN.md`` in the book repo).

The design note is P8.1, ``articulating-forms-that-matter/notes/COMMONS-DESIGN.md``,
and every section reference below (§1, §2, ...) is to it. Nothing here retunes the
cost between wirings: ``r, K, p, gamma`` and the three potential terms are fixed
once in :class:`Params`, the governed wiring *adds* one coupler carrying one new
parameter ``s``, and the insulated wiring *replaces* each catch coupler's one-form
by a non-exact one carrying one new parameter ``Rhat``.

The boxes and couplers (§1). Writing ``X = sum_i x_i``:

===========  =======  ===========================  ==============================
box          state    interface                    its own forcing (``omega = dU``)
===========  =======  ===========================  ==============================
``Res``      ``R>0``  ``<R^0|R^0> -> <R^0|R>``     ``d(rR - (r/2K)R^2)``
``Ext_i``    ``x_i``  ``<R^0|R^0> -> <R^0|R>``     ``d((gamma/2p) x_i^2)``
``C_i``      none     ``<R^0|R^2> -> <R^0|R^0>``   ``d(-x_i R)``
``S``        none     ``<R^0|R^N> -> <R^0|R^0>``   ``d((s/2) X^2)``
===========  =======  ===========================  ==============================

with responses ``sharp_R(xi) = mu R xi`` (multiplicative, keeps ``R > 0``) and
``sharp_x(xi) = -mu p xi`` (constant and *negative*: §3's oppositely signed
responses, the whole content of "the externality is the sign, not a missing wire").
``mu`` is the mobility, i.e. the discrete step (STATUS.md §3.2 item 1); it scales
both responses, so it moves the transient and the stability bound and leaves every
settled state -- the zeros of ``omega`` -- exactly where they were.

**Everything is built by the library's own composition.** A wiring is

    compose_seq(compose_seq(tensor_arrangements(boxes), fan_wire(...)),
                tensor_arrangements(couplers))

so the catch's two faces, and the fact that the resource's port receives
``sum_i x_i`` while each extractor's port receives no ``x_j`` at all (§3), come out
of ``compose_seq``'s pullback-and-sum -- the transpose of the fan-out copy is the
sum (KCL) -- and are not written by hand anywhere in this file.

The closed forms of §§4-6 are *not* used to build anything; they are here only as
the values the runs are checked against (``tests/test_commons.py``), exactly as
``eq:meeting-cost`` is used in ``tests/test_rendezvous.py``.

The entry model of §12b (O1, O2) is in :mod:`arrangements.commons_entry`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import List, Optional, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import configuration
from .rvect import ReactiveVectorSpace, constant
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

EMPTY = jnp.zeros(0)


# ---------------------------------------------------------------------------
# The one cost, written once (§2).
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Params:
    """``r, K, p, gamma``: fixed before any run and shared by all three wirings (§2).

    ``mobility`` is the integrator's, not the cost's: it is the discrete step, and
    ``settled`` below does not mention it.
    """

    r: float = 1.0
    K: float = 100.0
    p: float = 0.01
    gamma: float = 1.0
    mobility: float = 0.01

    # ---- the design's beta parametrization (§4, §5) ----

    def beta(self, N: int) -> float:
        """``beta = gamma r / (N K p)``, so ``R*_oa = K beta/(1 + beta)``."""
        return self.gamma * self.r / (N * self.K * self.p)

    def beta_gov(self, N: int, s: float) -> float:
        """``beta' = beta + r s / K`` (§9 prediction 3); ``s = 0`` gives ``beta``."""
        return self.beta(N) + self.r * s / self.K

    # ---- the closed forms the runs are checked against ----

    def R_open_access(self, N: int) -> float:
        """``R* = r gamma K / (r gamma + N p K)`` (§5)."""
        r, K, p, g = self.r, self.K, self.p, self.gamma
        return r * g * K / (r * g + N * p * K)

    def R_governed(self, N: int, s: float) -> float:
        """``R_gov = K beta'/(1 + beta')`` (§4)."""
        b = self.beta_gov(N, s)
        return self.K * b / (1.0 + b)

    def R_cooperative(self, N: int) -> float:
        """``R_coop = K(1 + beta)/(2 + beta)``, the planner's symmetric optimum (§4)."""
        b = self.beta(N)
        return self.K * (1.0 + b) / (2.0 + b)

    def s_optimal(self) -> float:
        """``s = K/r``: the sanction at which ``R_gov = R_coop`` for every ``N, p, gamma`` (§4)."""
        return self.K / self.r

    def X_of_R(self, R: float) -> float:
        """``X = r(1 - R/K)``, the total effort a settled stock ``R`` supports."""
        return self.r * (1.0 - R / self.K)

    def harvest(self, R: float, X: float) -> float:
        """``E = X R``: dissipation, in the one sense §7 allows."""
        return X * R

    def rent(self, R: float, N: int) -> float:
        """``Pi_oa = N p^2 R^2/(2 gamma)``, §5's closed form -- **open access only**.

        It assumes ``x_i = pR/gamma``, which is the extractors' settled effort when nothing
        else reaches their ports. Under the sanction ``x_i = pR/(gamma + Nps)`` and this
        formula is wrong; use :meth:`rent_from`. (Applying it to the governed wiring was a
        mistake worth recording: it is where the note's "rent ``-> pKr/4``" appeared to fail.)
        """
        return N * self.p ** 2 * R ** 2 / (2.0 * self.gamma)

    def rent_from(self, R: float, x) -> float:
        """The rent actually collected, ``Pi = p R sum_i x_i - (gamma/2) sum_i x_i^2``.

        The social objective §4 compares across wirings, with the sanction excluded from it
        -- the *transfer* reading, an economics assumption imported from outside the
        formalism and declared as strain S5.
        """
        x = jnp.asarray(x)
        return float(self.p * R * jnp.sum(x) - 0.5 * self.gamma * jnp.sum(x ** 2))

    def rent_cooperative(self, N: int) -> float:
        """The planner's rent at ``R_coop``: ``Pi = p R X - (gamma/2N) X^2``, ``X = r(1-R/K)``."""
        R = self.R_cooperative(N)
        X = self.X_of_R(R)
        return self.p * R * X - 0.5 * self.gamma * X ** 2 / N


# ---------------------------------------------------------------------------
# The boxes (§1).
# ---------------------------------------------------------------------------


def resource(par: Params, R0: float, label: str = "Res") -> Arrangement:
    """``Res : <R^0|R^0> -> <R^0|R>``, a *driven* box: it carries its own regeneration.

    State ``R``, response ``sharp_R(xi) = mu R xi`` (multiplicative, Shahshahani-type,
    so ``R > 0`` is invariant), output the published stock, forcing
    ``omega_reg = d(rR - (r/2K)R^2)``. With ``r = 0`` this is the finite-supply
    variant of §7: an undriven capacitance.
    """
    mu = par.mobility
    Q = ReactiveVectorSpace(dim=1, sharp_fn=lambda R: jnp.array([[mu * R[0]]]))
    U = lambda R, m_out, n_in: par.r * R[0] - (par.r / (2.0 * par.K)) * R[0] ** 2
    return Arrangement(
        P=configuration(Q, init=jnp.array([float(R0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda R, m_out: R,
        in_f=lambda R, m_out, n_in: EMPTY,
        omega=exact(U),
        label=label,
    )


def extractor(par: Params, x0: float, label: str = "Ext") -> Arrangement:
    """``Ext_i : <R^0|R^0> -> <R^0|R>``: state ``x_i``, response ``sharp_x = -mu p``.

    The response is *negative* -- the extractor moves against the shared potential
    in its own coordinate (§3) -- and the box's own forcing is
    ``omega_eff = d((gamma/2p) x_i^2)``, a function of its own state alone (§1's
    congruence rule).
    """
    mu = par.mobility
    Q = constant(jnp.array([[-mu * par.p]]))
    U = lambda x, m_out, n_in: (par.gamma / (2.0 * par.p)) * x[0] ** 2
    return Arrangement(
        P=configuration(Q, init=jnp.array([float(x0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda x, m_out: x,
        in_f=lambda x, m_out, n_in: EMPTY,
        omega=exact(U),
        label=label,
    )


# ---------------------------------------------------------------------------
# The couplers (§1, §4, §6) -- stateless, in the idiom of tests/test_two_tanks.py.
# ---------------------------------------------------------------------------


def stateless_coupler(arity: int, omega, label: str) -> Arrangement:
    """A stateless ``arity``-port coupler ``<R^0|R^arity> -> <R^0|R^0>``.

    ``omega(q, e, n_in)`` returns the covector triple; its ``M^+`` component is the
    returned covector ``F(e)`` of the reciprocity lemma.
    """
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=arity, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: EMPTY,
        in_f=lambda q, e, n_in: EMPTY,
        omega=omega,
        label=label,
    )


def catch_coupler(label: str = "C") -> Arrangement:
    """``C_i``, ports ``(R, x_i)``, exact with ``U_catch = -x_i R`` (§1).

    One term with two faces (§2): it returns ``-x_i`` to the resource port and
    ``-R`` to the effort port, and ``d^2 U/dx dR = -1`` is one number.
    """
    return stateless_coupler(2, exact(lambda q, e, n: -e[0] * e[1]), label)


def catch_coupler_insulated(Rhat: float, label: str = "C_ins") -> Arrangement:
    """``C_i^ins``, ports ``(R, x_i)``, carrying ``omega = -x_i dR - Rhat dx_i`` (§6).

    Not exact, and §6 proves it cannot be made exact: ``d omega = dR ^ dx_i``, with
    the ``(R, x_i)`` component equal to ``1`` independent of ``Rhat``. Written as a
    one-form directly -- there is no potential to differentiate.
    """
    Rhat = float(Rhat)

    def omega(q, e, n_in):
        return (jnp.zeros_like(q), jnp.array([-e[1], -Rhat]), jnp.zeros_like(n_in))

    return stateless_coupler(2, omega, label)


def sanction_coupler(s: float, N: int, label: str = "S") -> Arrangement:
    """``S``, ports ``(x_1, ..., x_N)``, exact with ``U_sanc = (s/2) X^2`` (§4).

    Returns ``s X`` to *every* effort port and nothing to the resource: the private
    effort cost taken on the collective's total. Its returned-covector Jacobian is
    ``s`` times the all-ones matrix, symmetric, so the coupler is exact and the
    restraint is mutual.
    """
    return stateless_coupler(
        N, exact(lambda q, e, n: 0.5 * s * jnp.sum(e) ** 2), label
    )


def unmoved_monitor(s: float, N: int, label: str = "S_unmoved") -> Arrangement:
    """§4's external enforcer *as the note words it*: returns ``sX`` to every user port and
    ``0`` to its own, on ports ``(x_1, ..., x_N, y)``.

    **This coupler is exact**, and finding that out is a correction to the note. Its
    one-form is ``d((s/2)X^2)`` with a zero component on ``y``; being unmoved costs it
    nothing, because ``X`` does not mention ``y`` and so the mixed partials it would have to
    break are all zero. Non-exactness needs the sanction to depend on the monitor's *own*
    state -- see :func:`external_enforcer`. Kept so the distinction is in the record.
    """

    def omega(q, e, n_in):
        X = jnp.sum(e[:N])
        F = jnp.concatenate([jnp.full(N, s * X), jnp.zeros(1)])
        return (jnp.zeros_like(q), F, jnp.zeros_like(n_in))

    return stateless_coupler(N + 1, omega, label)


def external_enforcer(N: int, label: str = "S_ext") -> Arrangement:
    """The *unilateral* coupler §4 means: a monitor whose own state sets the sanction.

    Ports ``(x_1, ..., x_N, y)``: the monitor holds a sanction level ``y``, returns ``y X``
    to every user port and ``0`` to its own. Then ``dF_i/dy = X`` while ``dF_y/dx_i = 0``,
    so the antisymmetric part of ``DF`` is nonzero and ``d omega != 0`` -- the formalism's
    waterfall (``tests/test_two_tanks.py``), one-way influence. Among the *users* it is
    still symmetric (``dF_i/dx_j = y = dF_j/dx_i``); what breaks exactness is the monitor,
    not the graduation.
    """

    def omega(q, e, n_in):
        X, y = jnp.sum(e[:N]), e[N]
        F = jnp.concatenate([jnp.full(N, y * X), jnp.zeros(1)])
        return (jnp.zeros_like(q), F, jnp.zeros_like(n_in))

    return stateless_coupler(N + 1, omega, label)


# ---------------------------------------------------------------------------
# The fan-out wiring: one R copied to N extractors (§3).
# ---------------------------------------------------------------------------


def fan_wire(src_dim: int, picks: Sequence[int], label: str = "fan") -> Arrangement:
    """A static wiring ``<R^0|R^src_dim> -> <R^0|R^len(picks)>``, ``m |-> m[picks]``.

    Trivial parameter, zero forcing. Copying is what makes the arrangement
    asymmetric: the transpose of a copy is a sum, so the resource's single port
    receives every extractor's term while no extractor's port mentions another's
    (§3). That fact is produced by ``compose_seq``'s pullback, not asserted here.
    """
    idx = jnp.asarray(list(picks), dtype=int)
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=src_dim, in_dim_N=0, out_dim_N=len(idx),
        out_f=lambda q, m_out: m_out[idx],
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form,
        label=label,
    )


# ---------------------------------------------------------------------------
# The three wirings (§§4-6), each a genuine composite in arr.
# ---------------------------------------------------------------------------


def commons(
    par: Params,
    N: int,
    R0: float = 90.0,
    x0: float = 0.1,
    s: Optional[float] = None,
    Rhat: Optional[float] = None,
    label: Optional[str] = None,
) -> Arrangement:
    """The closed arrangement ``<R^0|R^0> -> <R^0|R^0>`` of one of the three wirings.

    * ``s is None``, ``Rhat is None``  -- OPEN ACCESS (§5): boxes, fan-out, ``N`` catch couplers.
    * ``s`` given                      -- GOVERNED (§4): the same, **plus** one ``N``-ary sanction.
    * ``Rhat`` given                   -- INSULATED (§6): the same boxes and wiring, each catch
      coupler's one-form replaced by the non-exact ``-x_i dR - Rhat dx_i``.

    Nothing else differs between the three. The composite's configuration is
    ``q = (R, x_1, ..., x_N)``.
    """
    if s is not None and Rhat is not None:
        raise ValueError("governed and insulated are different wirings; pass one")
    boxes = [resource(par, R0)] + [extractor(par, x0, f"Ext{i + 1}") for i in range(N)]
    # ports: (R, x_1), (R, x_2), ..., (R, x_N), then (x_1, ..., x_N) if governed.
    picks: List[int] = []
    for i in range(N):
        picks += [0, 1 + i]
    if s is not None:
        picks += [1 + i for i in range(N)]
    if Rhat is None:
        couplers = [catch_coupler(f"C{i + 1}") for i in range(N)]
    else:
        couplers = [catch_coupler_insulated(Rhat, f"C{i + 1}^ins") for i in range(N)]
    if s is not None:
        couplers.append(sanction_coupler(s, N))
    wired = compose_seq(
        compose_seq(tensor_arrangements(boxes), fan_wire(1 + N, picks)),
        tensor_arrangements(couplers),
    )
    kind = "gov" if s is not None else ("ins" if Rhat is not None else "oa")
    return replace(wired, label=label or f"commons[{kind}](N={N})")


def open_access(par: Params, N: int, **kw) -> Arrangement:
    """Wiring 2 (§5)."""
    return commons(par, N, **kw)


def governed(par: Params, N: int, s: float, **kw) -> Arrangement:
    """Wiring 1 (§4): open access **plus** one exact ``N``-ary coupler."""
    return commons(par, N, s=s, **kw)


def insulated(par: Params, N: int, Rhat: float, **kw) -> Arrangement:
    """Wiring 3 (§6): open access with each catch coupler's one-form made stale."""
    return commons(par, N, Rhat=Rhat, **kw)


# ---------------------------------------------------------------------------
# Running a closed arrangement.
# ---------------------------------------------------------------------------


def closed_step(arr: Arrangement):
    """One step of ``Phi(arr)`` for a closed arrangement, as in ``tests/test_rendezvous.py``.

    For ``in_dim_M = out_dim_M = 0`` and a zero exterior covector the interpretation's
    triple collapses to the forcing's own ``Q``-component, so the step is
    ``s |-> u(s, omega(q(s), (), ())|_Q)``. ``test_the_closed_step_is_Phi`` checks that
    against ``Phi`` rather than assuming it.
    """
    P, omega = arr.P, arr.omega

    @jax.jit
    def step(s):
        xi_q, _, _ = omega(P.position(s), EMPTY, EMPTY)
        return P.step(s, xi_q)

    return step


def run(arr: Arrangement, steps: int, every: int = 1) -> Array:
    """The configuration trajectory ``q_t = (R, x_1, ..., x_N)``, sampled every ``every``.

    The loop is a ``lax.scan`` so that a long run is one compiled kernel, not ``steps``
    dispatches; the sampling is a stride on the result.
    """
    step, P = closed_step(arr), arr.P

    def body(s, _):
        s2 = step(s)
        return s2, P.position(s2)

    _, qs = jax.lax.scan(body, P.init, None, length=steps)
    return jnp.concatenate([P.position(P.init)[None, :], qs])[::every]


def settle(arr: Arrangement, steps: int) -> Array:
    """The configuration after ``steps`` steps, with the loop inside ``jax``."""
    step, P = closed_step(arr), arr.P
    s = jax.lax.fori_loop(0, steps, lambda _i, st: step(st), P.init)
    return P.position(s)


def split(q: Array) -> Tuple[float, Array]:
    """``q = (R, x_1, ..., x_N) |-> (R, x)``."""
    return float(q[0]), q[1:]


# ---------------------------------------------------------------------------
# Exactness bookkeeping (§9 predictions 7, 8).
# ---------------------------------------------------------------------------


def composite_port_form(arr: Arrangement):
    """The composite forcing as a one-form on the configuration ``Q = R^{1+N}``.

    For a closed arrangement the configuration coordinates *are* the ports
    ``(R, x_1, ..., x_N)`` -- every ``out_f`` is the identity -- so the composite
    one-form ``omega_oa``, ``omega_gov``, ``omega_ins`` of §§4-6 is exactly
    ``q |-> omega(q, (), ())|_Q``, and its exterior derivative is the antisymmetric
    part of that map's Jacobian (the reciprocity lemma, ``tests/test_two_tanks.py``).
    """
    return lambda q: arr.omega(q, EMPTY, EMPTY)[0]


def d_omega(form):
    """``(d omega)_{ij} = d_i F_j - d_j F_i``, the antisymmetric part of ``DF``."""

    def at(q):
        J = jax.jacobian(form)(q)   # J[j, i] = d F_j / d q_i
        return J.T - J

    return at


def returned_covector(cpl: Arrangement):
    """``F(e)``: a stateless coupler's forcing restricted to its ports."""
    return lambda e: cpl.omega(EMPTY, e, EMPTY)[1]
