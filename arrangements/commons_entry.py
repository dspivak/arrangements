"""Entry into a commons by placement: §12b's two open items, O2 then O1.

The design note is ``articulating-forms-that-matter/notes/COMMONS-DESIGN.md``; §12b
lists two things it assumes and does not establish, and asks P8.2 to **test** them.
This module is the test bed. The plain three-wiring model is
:mod:`arrangements.commons`; nothing here retunes its cost.

**O2 -- what message does the kernel weigh, and how does the catch ride on it.**
The interface is the attention paper's ``B = (E, M)``, ``M = H x W``, ``E = M x P``
(``arrangements.attention``), with ``dim H = dim P = 1``, and the reading is:

* the **sender is the resource**, and the only sender. Its message is
  ``m_Res = (h, w) = (R, 1)``: the *signal* carries the stock on offer and the
  *weight* carries the stock's being one resource, counted once.
* the **extractors are receivers** and emit the zero message ``0_M = (0, 0)``, which
  is admissible. So extractor ``i``'s received input is
  ``rho_a(q, m, e)_i = ((kappa_i R, kappa_i), l_i(q))`` with
  ``kappa_i = exp(-||l_i - l_Res||^2 / 2 tau)`` -- the paper's own kernel, nothing added.
* the **catch rides on the weighted signal**, not on the mean readout: the term the
  extractor carries is ``-x_i h(e_i) = -x_i kappa_i R``, bilinear in effort and
  received signal, which keeps the cost in the quadratic family (``attention.py``
  warns that the mean readout would not).

That last point is the content of O2 and it is forced, not chosen:
``mean(kappa_i R, kappa_i) = R`` for every ``kappa_i > 0``, so **the mean readout is
proximity-blind** and cannot carry a commons. Being near the fishery must change what
you can take; it must not change what you believe the stock to be.

At ``kappa_i = 1`` for all ``i`` the composite is *exactly* §2's ``U``, so the
placement model is a deformation of the fixed cost along the kernel and not a
different model (``tests/test_commons_entry.py``).

**O1 -- what makes entry stop.** Each extractor also carries a *cost of proximity*
``a_i kappa(l_i)``: the outside option it forgoes in proportion to how far in it is.
Its placement then feels ``(a_i - x_i R) d kappa``, so the stationary condition is
"marginal rent equals the marginal cost of being close". What the runs show
(``tests/test_commons_entry.py``, ``notes/COMMONS-RESULTS.md``):

* entry **does** stop, at an interior placement, and the stopping condition is
  ``x_i R = a_i`` to machine precision. So §12b's "placement settles where marginal
  rent equals the marginal cost of being close" is confirmed. It is stable only
  because the *stock* responds: at fixed ``R`` the net benefit is convex in ``kappa``
  and the same point is a repeller, so the stabilizer is the resource, not the cost.
* that point is **not** the zero-rent condition. At it the rent collected is exactly
  half the proximity cost paid (:func:`marginal_threshold` versus
  :func:`zero_rent_threshold`), because rent is quadratic in the insideness ``kappa``
  and the proximity cost is linear. Every participant settles at a *negative* net
  benefit ``V = -a kappa/2`` -- worse than never having come -- and no local move
  gets it out.
* ``kappa = 0`` is a fixed point with a vanishing gradient, so a participant placed
  far out never arrives: entry has to be seeded inside the kernel's pull region.

Signs. The extractor's response is negative in *every* coordinate it owns, effort and
placement alike (``sharp_x = -mu p``, ``sharp_l = -mu_l``), which is §3's oppositely
signed responses. The placement coordinate lives in the *arrangement's* configuration,
not the extractor's, so that pairing is a modeling choice the formalism does not force;
it is recorded in ``notes/COMMONS-RESULTS.md`` as a strain.
"""

from __future__ import annotations

from dataclasses import replace
from typing import List, Optional, Sequence

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact
from .attention import (
    input_dim, make_input, message, msg_dim, spatial_arrangement, spatial_config,
    spatial_kappa, zero_cost, zero_message,
)
from .coalgebra import configuration
from .commons import Params
from .rvect import ReactiveVectorSpace, constant
from .wiring import compose_seq, tensor_arrangements

DH, DP = 1, 1
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)
EMPTY = jnp.zeros(0)
EXTERIOR = make_input(zero_message(DH), 0.0)   # the exterior sends nothing, from position 0


# ---------------------------------------------------------------------------
# The two components (I -> B).
# ---------------------------------------------------------------------------


def resource_component(par: Params, R0: float, label: str = "Res") -> Arrangement:
    """``Res : I -> B``, the sole sender: it emits ``m_Res = (R, 1)``.

    Same state, same response ``sharp_R = mu R`` and same own forcing
    ``d(rR - (r/2K)R^2)`` as ``commons.resource``; only the output map differs, and it
    differs by putting the stock in the *signal* slot and ``1`` in the *weight* slot.
    """
    mu = par.mobility
    Q = ReactiveVectorSpace(dim=1, sharp_fn=lambda R: jnp.array([[mu * R[0]]]))
    return Arrangement(
        P=configuration(Q, init=jnp.array([float(R0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda R, m_out: message(R[0], 1.0),
        in_f=lambda R, m_out, n_in: EMPTY,
        omega=exact(lambda R, m_out, n_in:
                    par.r * R[0] - (par.r / (2.0 * par.K)) * R[0] ** 2),
        label=label,
    )


def extractor_component(
    par: Params, x0: float, a: float, tau: float, power: float = 1.0, label: str = "Ext",
) -> Arrangement:
    """``Ext_i : I -> B``, a pure receiver emitting ``0_M``, with potential

        U_i(x_i, e_i) = (gamma/2p) x_i^2  -  x_i h(e_i)  +  a kappa(p(e_i))^power,

    where ``e_i = (m, p)`` is the input it receives: ``h(e_i)`` is the signal of the
    weighted message (the stock as it reaches this extractor) and ``p(e_i)`` is the
    position it is placed at. The first two terms are §2's effort and catch terms; the
    third is the cost of proximity that O1 asks for, ``a`` the outside option forgone.
    ``power`` is the shape of that cost in the insideness ``kappa``; ``1`` is the
    default reading (the outside option forgone in proportion to how far in one is).

    The response is ``-mu p`` on effort (§3). The placement's response lives in the
    parent's integrator (see :func:`entry_system`).
    """
    mu = par.mobility
    Q = constant(jnp.array([[-mu * par.p]]))

    def U(x, m_out, n_in):
        h = n_in[0]                       # signal of the received message
        pos = n_in[MDIM]                  # position half of the received input
        kappa = jnp.exp(-pos ** 2 / (2.0 * tau))
        return ((par.gamma / (2.0 * par.p)) * x[0] ** 2 - x[0] * h
                + a * kappa ** power)

    return Arrangement(
        P=configuration(Q, init=jnp.array([float(x0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda x, m_out: zero_message(DH),
        in_f=lambda x, m_out, n_in: EMPTY,
        omega=exact(U),
        label=label,
    )


# ---------------------------------------------------------------------------
# The arrangement: one spatial-attention operation with N + 1 children.
# ---------------------------------------------------------------------------


def entry_system(
    par: Params,
    placements: Sequence[float],
    outside: Sequence[float],
    tau: float = 2.0,
    R0: float = 90.0,
    x0: float = 0.01,
    place_mobility: float = 0.002,
    power: float = 1.0,
    label: str = "entry",
) -> Arrangement:
    """``a(Res, Ext_1, ..., Ext_N) : I -> B``, by composition in ``arr``.

    ``placements[i]`` and ``outside[i]`` are extractor ``i``'s initial placement and its
    outside option ``a_i``. The resource sits at ``l_Res = 0`` with zero mobility; the
    value maps are frozen at the identity (zero mobility), as in
    ``tests/test_placement_wiring.py``, so the only thing the kernel sees move is the
    extractors' placements.

    Composite configuration, in the order ``compose_seq`` produces:
    ``(R, x_1..x_N, l_Res, l_1..l_N, V_Res, V_1..V_N)``.
    """
    N = len(placements)
    if len(outside) != N:
        raise ValueError("one outside option per extractor")
    n = N + 1
    children = [resource_component(par, R0)] + [
        extractor_component(par, x0, float(outside[i]), tau, power, f"Ext{i + 1}")
        for i in range(N)
    ]
    # placements: the resource frozen, the extractors' response NEGATIVE (§3).
    mob = jnp.asarray([0.0] + [-float(place_mobility)] * N + [0.0] * n)
    parent = spatial_arrangement(
        n, DH, DP, tau, cost=zero_cost,
        parameter=configuration(
            constant(jnp.diag(mob)),
            init=spatial_config([0.0] + [float(v) for v in placements], [1.0] * n),
        ),
        label=f"spatial({n})",
    )
    composite = compose_seq(tensor_arrangements(children), parent.arr())
    return replace(composite, label=f"{label}(N={N})")


def unpack(q: Array, N: int):
    """``q |-> (R, x, l)`` with ``l`` the ``N`` extractor placements (the resource's is ``q[1+N]``)."""
    q = jnp.asarray(q)
    return q[0], q[1:1 + N], q[2 + N:2 + 2 * N]


def kappas(q: Array, N: int, tau: float) -> Array:
    """``kappa_i = exp(-(l_i - l_Res)^2 / 2 tau)``: extractor ``i``'s coupling to the resource."""
    q = jnp.asarray(q)
    l_res = q[1 + N]
    l = q[2 + N:2 + 2 * N]
    return jnp.exp(-(l - l_res) ** 2 / (2.0 * tau))


def effective_N(q: Array, N: int, tau: float) -> float:
    """``N_eff = sum_i kappa_i^2``: the ``N`` the settled stock actually sees.

    At each extractor's own settled effort ``x_i = p kappa_i R / gamma`` the harvest is
    ``sum_i kappa_i x_i R = (p R^2/gamma) sum_i kappa_i^2``, so the open-access formula
    ``R* = K beta/(1+beta)`` holds with ``beta = gamma r/(N_eff K p)``. This is the sense
    in which effective ``N`` rises during a run inside a fixed syntax (§12b, S1).
    """
    return float(jnp.sum(kappas(q, N, tau) ** 2))


# ---------------------------------------------------------------------------
# Running.
# ---------------------------------------------------------------------------


def closed_step(arr: Arrangement, e=EXTERIOR):
    """One step of a system closed by the fixed exterior input ``e`` (``tests/test_rendezvous.py``)."""
    P, omega = arr.P, arr.omega

    @jax.jit
    def step(s):
        xi_q, _, _ = omega(P.position(s), EMPTY, e)
        return P.step(s, xi_q)

    return step


def run(arr: Arrangement, steps: int, every: int = 1, e=EXTERIOR) -> Array:
    """The configuration trajectory, as one compiled ``lax.scan`` sampled at stride ``every``."""
    step, P = closed_step(arr, e), arr.P

    def body(s, _):
        s2 = step(s)
        return s2, P.position(s2)

    _, qs = jax.lax.scan(body, P.init, None, length=steps)
    return jnp.concatenate([P.position(P.init)[None, :], qs])[::every]


def settle(arr: Arrangement, steps: int, e=EXTERIOR) -> Array:
    step, P = closed_step(arr, e), arr.P
    s = jax.lax.fori_loop(0, steps, lambda _i, st: step(st), P.init)
    return P.position(s)


# ---------------------------------------------------------------------------
# The closed forms O1 is tested against.
# ---------------------------------------------------------------------------


def rent_per_extractor(par: Params, R: float, kappa: float) -> float:
    """``pi_i = p^2 kappa^2 R^2 / (2 gamma)``: extractor ``i``'s rent at its settled effort.

    At ``x_i = p kappa R/gamma`` the profit flow ``p kappa R x_i - (gamma/2)x_i^2`` is this.
    """
    return par.p ** 2 * kappa ** 2 * R ** 2 / (2.0 * par.gamma)


def net_benefit(par: Params, R: float, kappa: float, a: float, power: float = 1.0) -> float:
    """``V(kappa) = p kappa^2 R^2/(2 gamma) - a kappa^power``: what the placement ascends.

    The potential the extractor's coordinates move on is in effort units (``p`` is scaled
    into the response, §1), so ``V = pi/p - a kappa^power``, with ``V(0) = 0`` the value of
    staying out entirely. The runs settle at ``V'(kappa) = 0``; they do **not** settle at
    ``V = 0``, and §12b's O1 asked which.
    """
    return par.p * kappa ** 2 * R ** 2 / (2.0 * par.gamma) - a * kappa ** power


def marginal_threshold(par: Params, R: float, kappa: float = 1.0, power: float = 1.0) -> float:
    """``a`` at which ``dV/dkappa = 0``: ``a = p kappa^{2-power} R^2 / (power gamma)``.

    The *marginal* condition -- marginal rent equals the marginal cost of being close --
    which is what a gradient flow on the placement can see.
    """
    return par.p * kappa ** (2.0 - power) * R ** 2 / (power * par.gamma)


def zero_rent_threshold(par: Params, R: float, kappa: float = 1.0, power: float = 1.0) -> float:
    """``a`` at which ``V(kappa) = 0``: ``a = p kappa^{2-power} R^2 / (2 gamma)``.

    Gordon's condition -- entry until the rent is fully dissipated against the outside
    option. It differs from :func:`marginal_threshold` by the factor ``power/2``, and
    coincides with it only for ``power = 2``, where both are degenerate.
    """
    return par.p * kappa ** (2.0 - power) * R ** 2 / (2.0 * par.gamma)


def entry_count(par: Params, a: float, N_max: int, threshold=marginal_threshold) -> int:
    """The largest ``n <= N_max`` that ``threshold`` still admits, with the stock settling.

    ``n`` fully-placed entrants settle the stock at ``R*(n) = K beta_n/(1+beta_n)``; the
    ``(n+1)``-st is admitted iff ``a < threshold(par, R*(n+1))``. With
    :func:`marginal_threshold` this is the count the placement dynamics actually
    produces; with :func:`zero_rent_threshold` it is Gordon's, and the two differ.
    """
    n = 0
    while n < N_max and threshold(par, par.R_open_access(n + 1)) > a:
        n += 1
    return n
