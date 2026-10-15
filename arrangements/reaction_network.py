"""Mass-action reaction networks as ``Phi`` of configuration arrangements.

A reversible reaction network on species ``1..n`` is a list of reactions
``r`` with reactant/product stoichiometric vectors ``alpha_r, beta_r in N^n``
and rate constants ``kf_r, kb_r > 0``; mass-action kinetics is the ODE

    x' = sum_r zeta_r * v_r(x),   zeta_r = beta_r - alpha_r,
    v_r(x) = kf_r * x^{alpha_r} - kb_r * x^{beta_r}          (net flux).

The network is *detailed balanced* if some ``x_eq > 0`` equalizes every
reaction's forward and backward flux: ``kf_r x_eq^{alpha_r} = kb_r
x_eq^{beta_r} =: w_r``. This module settles which of these ODEs are the
configuration dynamics -- ``Phi`` of an arrangement whose parameter is the
configuration coalgebra ``configuration(Q)`` (def.arr, sec.matter) -- and
in which encoding.

VACUITY GUARD. ``def.rvect`` puts no symmetry or definiteness condition on a
sharp, so *every* smooth ODE ``x' = V(x)`` is trivially the configuration
dynamics of a one-box arrangement: take ``omega = dx_1`` (the exact forcing of
``U(x) = x_1``) and the rank-one sharp ``sharpR_x = dt * V(x) (x) e_1``. That
realization has no content -- the "potential" certifies nothing and descends
nothing. The claims below are therefore about *structured* realizations: a symmetric
positive-semidefinite sharp together with a potential that the flow genuinely
descends (an Onsager gradient structure), which is what every dissipative
example in the paper and in this repo has.

THE VERIFIED FRONTIER.

1. MONOLITHIC (one box), any molecularity: every reversible detailed-balanced
   mass-action network is ``Phi`` of the closed arrangement
   (``crn_arrangement``) whose parameter is the configuration coalgebra on
   ``R^n`` with the state-dependent Mielke--Onsager sharp [Mielke 2011, "A
   gradient structure for reaction-diffusion systems and for
   energy-drift-diffusion systems", Nonlinearity 24, thm 3.1 specialized to
   pure reaction]

       sharpR_x = dt * K(x),
       K(x)     = sum_r Lambda(f_r(x), b_r(x)) * zeta_r zeta_r^T,

   with ``f_r = kf_r x^{alpha_r}``, ``b_r = kb_r x^{beta_r}`` the one-way
   fluxes and ``Lambda(u, v) = (u - v)/(log u - log v)`` the logarithmic mean
   (``Lambda(u, u) = u``), and whose forcing is exact, ``omega = -dF`` for the
   free energy

       F(x) = sum_i x_i (log(x_i / x_eq_i) - 1).

   The realization is EXACT, not asymptotic: since ``dF = log(x/x_eq)`` and
   detailed balance gives ``zeta_r . dF = log(b_r/f_r)``, the identity
   ``Lambda(f, b) log(b/f) = b - f`` makes one configuration-coalgebra step
   ``x -> x + sharpR_x(omega)``, ``omega = -dF`` (sec.differential_equations) literally
   the forward-Euler mass-action step ``x + dt * sum_r zeta_r v_r(x)``, to
   machine precision. ``K(x)`` is symmetric PSD on ``x > 0`` (``Lambda > 0``),
   so ``F`` decreases along the flow and the equilibrium in each stoichiometric
   compatibility class is its minimum. ``dt`` rides on the sharp exactly as in
   ``replicator.shahshahani``. Positivity of ``x`` is preserved by the flow
   but only approximately by Euler steps (keep ``dt`` small); ``F``, ``K``
   need ``x > 0`` and reversibility ``kb_r > 0`` (a one-way flux 0 puts
   ``Lambda`` on the boundary and the log-mean form degenerates).

   Not covered: networks WITHOUT detailed balance. Irreversible reactions
   break ``Lambda`` outright; reversible-but-Wegscheider-violating cycles
   (e.g. ``A<->B<->C<->A`` with cycle affinity != 0) admit no ``x_eq`` and
   ``crn_arrangement`` rejects them. Complex-balanced networks still
   dissipate this ``F`` but the flow has a non-gradient part; no structured
   realization is claimed or built here.

2. COMPOSITIONAL, unimolecular reversible networks (``A_i <-> A_j`` edges,
   detailed balanced): realized by genuine composition in ``arr``
   (``unimolecular_arrangement``), on the kuramoto/graph-Laplacian pattern:
   one species box per species (parameter the configuration coalgebra on
   ``R`` holding ``x_i``, CONSTANT scalar sharp ``dt * x_eq_i``, readout the
   activity ``a_i = x_i / x_eq_i`` broadcast to its reaction ports), one
   stateless reaction box per edge with forcing ``-d`` of the cost
   ``(w_r / 2)(a_i - a_j)^2``, and a routing wire; the composite is
   ``compose_seq(tensor(boxes), wire)`` and its total forcing ``-dU``,

       U(x) = sum_r (w_r / 2) (x_i/x_eq_i - x_j/x_eq_j)^2,

   emerges from pullback-and-sum (sec.forcing, notes/DESIGN.md §2), written nowhere
   by hand. One ``Phi`` step is again EXACTLY the forward-Euler mass-action
   step: the covector reaction ``r`` returns to species ``i`` is ``-+ w_r (a_i
   - a_j) / x_eq_i`` (the ``1/x_eq_i`` from the readout's pullback), and the
   sharp ``dt * x_eq_i`` cancels the weight, leaving ``-+ dt * w_r (a_i -
   a_j) = dt * zeta_{r,i} v_r(x)``. Here the linear vector field is
   ``-K0 * diag(1/x_eq) x`` with CONSTANT ``K0 = sum_r w_r zeta_r zeta_r^T``;
   the constant-sharp/quadratic-potential pair is the same gradient flow as
   the Mielke pair of 1 (both equal ``sum_r zeta_r v_r`` on the nose), so the
   monolithic and compositional encodings agree step-for-step.

3. COMPOSITIONAL OBSTRUCTION, bimolecular: the species-box encoding STOPS at
   unimolecular. Precisely: in any composite built from species boxes with
   one-dimensional state (storing any diffeomorphic reparameterization of its
   own concentration -- log-coordinates included), stateless reaction boxes,
   and wiring, the composite sharp is the DIRECT SUM of the species sharps
   (``rvect.direct_sum``, via ``compose_seq``/``parallel_arrangements``), each
   block a function of its own species' state only. The realizable dynamics
   are therefore exactly

       x_i' = -g_i(x_i) * dU/dx_i (x),        g_i a function of x_i alone,

   (reaction data in the shared potential, per-species conformal factors in
   the sharp). Closedness of ``dU`` forces, for all ``i != j`` and all ``x``,

       dV_j/dx_i (x) / dV_i/dx_j (x) = g_j(x_j) / g_i(x_i),

   i.e. the ratio of opposite Jacobian entries of the vector field ``V`` must
   be SEPARABLE -- independent of every other species. For ``A + B <-> C``
   (``V = zeta * (kf x_A x_B - kb x_C)``, ``zeta = (-1,-1,+1)``) the (A, C)
   ratio is ``(dV_C/dx_A)/(dV_A/dx_C) = (kf/kb) * x_B``, which varies with the
   third species ``B``: NO choice of ``g_A, g_B, g_C`` and ``U`` works, in any
   per-species coordinates (a reparameterization ``x_i = phi_i(y_i)`` only
   changes ``g_i``, never the ratio). Irreversibly, ``A + B -> C`` is worse:
   ``dV_A/dx_C = 0`` while ``dV_C/dx_A = kf x_B != 0``, so ``C`` would need a
   potential term the closedness of ``dU`` forbids -- the covector a potential
   returns to a species is the potential's OWN derivative in that species, so
   production stoichiometry cannot be decoupled from flux dependence. This is
   also why the paper's Lotka--Volterra pattern (log states, monomial
   potential, oppositely-SIGNED sharps) works for predation yet never yields a
   produced third species: its interaction monomial couples exactly the
   species that appear in it, generating the Kolmogorov class
   ``x_i' = x_i * (sign_i) dU/d(log x_i)`` -- interaction matrix = signed
   diagonal times symmetric -- and ``A + B <-> C`` is outside that class by
   the same ratio test (``separability_ratio``).

   Scope of the no-go: species boxes with hidden state (dim > 1) and stateful
   reaction boxes are not ruled out. In particular a SINGLE reversible
   reaction is realizable in its extent coordinate ``s`` (state ``x_0 + zeta
   s``), but that is a one-box arrangement -- monolithic in disguise; with two
   reactions sharing a species each extent box's sharp would need the other
   box's state, which ``direct_sum`` forbids, and the obstruction returns.
   Item 4 occupies exactly this gap.

   The no-go is about EXACT forcings (a shared potential ``U``): closedness of
   ``dU`` is what forces the separability. The book's forcing need not be
   exact (def.arr), and item 6 is the realization the obstruction does not
   reach -- at the price of certifying nothing.

4. COMPOSITIONAL, ARBITRARY Petri net (the paper's "Petri nets" item): put the
   state in the TRANSITIONS, doubled, and the species in the WIRING. A Petri
   net ``(S, T, m, n, r, x0)`` has arc multiplicities ``m, n : T x S -> N``,
   rate constants ``r : T -> R_{>=0}`` and initial concentrations ``x0``; each
   transition is one-way and fires at ``f_t(x) = r(t) prod_s x_s^{m_ts}`` (a
   reversible reaction is two transitions). Transition ``t`` is a box
   ``<R^{S_t}|R> `` whose parameter is the configuration coalgebra on
   ``logic.py``'s transistor pair

       (R^2, sharpR),   sharpR(xi, xi') = (xi', 0),

   holding ``(eps_t, eps'_t)``, reporting ``eps_t`` (the extent of reaction),
   receiving the concentrations at the species ``S_t`` it touches, with
   forcing ``-dU_t`` for

       U_t((eps_t, eps'_t), x) = -eps'_t * f_t(x).

   The species are the static wiring: an affine input map sending the reported
   extents to ``x_s = x0_s + sum_t (n_ts - m_ts) eps_t`` (affine wiring has
   precedent in ``amplifier.py``'s feedback stage). Since ``U`` is LINEAR in
   ``eps'_t``, ``-dU/deps'_t = f_t(x)`` is the rate itself rather than a
   derivative of it, and the nilpotent sharp delivers that to ``eps_t`` while
   discarding ``dU/deps_t`` (the rate sensitivities, which are what a genuine
   descent would follow). One ``Phi`` tick is then EXACTLY the forward-Euler
   mass-action step, at any molecularity, irreversibly, with no detailed-balance
   hypothesis; ``eps'_t`` never moves and its value never matters.

   WHAT THIS DOES AND DOESN'T BUY. The sharp is nilpotent, not symmetric PSD,
   and ``U_t`` is a rate, not an energy: nothing descends, so by the vacuity
   guard above this realization certifies no thermodynamics. What it does have,
   which the one-box vacuous realization does not, is LOCALITY: each box carries
   only its own rate constant and arc multiplicities and sees only its own
   species, and the total forcing is assembled by ``compose_seq``. Moiety
   conservation is also structural rather than numerical: the state IS the
   extent vector and ``x`` is recomputed as ``x0 + Z^T eps`` at every tick, so
   any ``c`` with ``Z c = 0`` has ``c . x = c . x0`` at every step without
   rounding accumulating in ``x`` (Euler stepped on ``x`` preserves such laws
   in exact arithmetic too, but drifts by accumulated float error).

5. COMPOSITIONAL, ARBITRARY Petri net IN LOG COORDINATES: positivity by
   construction. Dual to 4: the state moves into doubled SPECIES boxes and the
   transitions become stateless. Species ``s`` is a box ``<R^0|R^0> ->
   <R^0 | R^{2 deg(s)}>`` with the configuration coalgebra on the transistor
   pair

       (R^2, sharpR),   sharpR(xi, xi') = (dt * xi', 0),

   holding ``(y_s, y'_s)``, broadcasting the pair ``(x_s, y'_s)``,
   ``x_s = exp(y_s)``, to each incident transition; transition ``t`` is a
   STATELESS box receiving those pairs at the species it touches, with forcing
   ``-dV_t`` for

       V_t = -f_t(x) * sum_s zeta_ts * y'_s / x_s,   zeta = n - m;

   the wiring is an honest prism (a port bijection -- no affine map, and
   ``x0`` is genuinely the initial state rather than wiring data). Since
   ``-dV_t/dy'_s = zeta_ts f_t(x)/x_s`` is ``y'``-free, and the exp Jacobian
   of the readout lands only in the discarded ``xi``-slot, one ``Phi``
   tick is EXACTLY

       y_s -> y_s + dt * sum_t zeta_ts f_t(x)/x_s,

   the forward-Euler step of mass action written in log-concentration
   coordinates ``y = log x`` (equivalently ``x_s -> x_s * exp(dt *
   (Z^T f)_s / x_s)``, a multiplicative Euler step). Concentrations are
   POSITIVE BY CONSTRUCTION at every tick and any ``dt``, where encoding 4
   can overshoot negative; the price is exactly dual: 4 preserves every
   moiety conservation law structurally and positivity only approximately,
   5 preserves positivity structurally and conservation laws only
   approximately (``O(dt^2)`` drift per tick). ``y'_s`` is frozen and its
   value never matters, as in 4. POSITIVITY IS NOT STABILITY: near the
   orthant boundary the exponent ``(Z^T f)_s/x_s`` is unbounded, so at large
   ``dt`` the iteration can still blow up (in floats, overflowing to
   ``inf``/``0``); what the construction removes is only the orthant-exit
   failure mode, and ``dt`` must still be small for stability and accuracy.
   The clean win is stiff pure decay (``A -> 0`` fast): Euler exits the
   orthant at ``dt > 1/rate`` while the log tick ``x e^{-rate dt}`` is
   positive, monotone, and stable at every ``dt``.

6. DIRECT, any network (the book's non-exact forcing, def.arr, sec.forcing;
   STATUS.md §2, trick vs direct). An arrangement's forcing need not be exact, so the
   mass-action field itself is admissible as a one-form.
   ``mass_action_arrangement`` is the closed one-box arrangement whose
   parameter is the PLAIN configuration coalgebra on ``R^n`` with the
   Euclidean sharp ``dt * I`` and whose forcing is written by hand, with no
   potential,

       omega(x) = V(x)^flat = sum_r zeta_r v_r(x)     (flat the identity in coordinates).

   One ``Phi`` step is ``x + dt * V(x)``, the forward-Euler mass-action step,
   at any molecularity, reversible or not, with no detailed-balance
   hypothesis: it runs the networks 1 excludes (Wegscheider-violating cycles,
   irreversible reactions, one-way Petri nets), and, being one box, is
   untouched by the obstruction of 3, which is about shared potentials and
   direct-sum sharps. On a detailed-balanced network it agrees with 1 step for
   step -- both are ``x + dt V(x)`` on the nose.

   TRICK versus DIRECT. Item 1 is the TRICK realization: CDLM's arrangement
   carried only a potential, so a field that is no Euclidean gradient had to be
   manufactured as ``sharp(-dF)`` by engineering a state-dependent sharp
   ``K(x)`` around the free energy. Item 6 is the DIRECT realization: the field
   is the forcing, and the sharp is plain. What the direct realization does NOT
   buy is exactly what the vacuity guard warns about: ``omega = V^flat`` is
   exact iff ``DV`` is symmetric, which fails already for ``A + B <-> C``
   (``dV_C/dx_A = kf x_B`` against ``dV_A/dx_C = kb``), so nothing is descended
   and no thermodynamics is certified; the monotone free energy of 1 comes from
   the Onsager pair ``(K, F)`` and only from there. Both are "the configuration
   dynamics of an arrangement"; 1 says WHY the flow dissipates, 6 only THAT it
   is this flow. Read against the vacuity guard, the book shortens the vacuous
   realization by one step too -- ``omega = V^flat`` with the plain sharp needs
   no rank-one trick -- which is the same warning: admissibility as a forcing
   is cheap, structure is not.

In short: detailed-balanced mass-action = configuration dynamics with an
Onsager (symmetric-PSD, free-energy-descending) structure, monolithically,
exactly; compositionally with SPECIES boxes exactly the unimolecular ones, the
first bimolecular reaction already unwireable because the cross-species Onsager
coupling ``zeta_r zeta_r^T`` cannot arise from a direct-sum sharp; and
compositionally with doubled TRANSITION boxes every Petri net, exactly, at the
price of a nilpotent sharp and a potential that is not an energy -- or with
doubled SPECIES boxes in log coordinates, exactly Euler-in-log, positive by
construction, trading structural conservation for structural positivity; and
directly, as the book allows, every network as a non-exact forcing on a plain
configuration coalgebra, certifying nothing -- in one box (6), or composed with a
box per chemical holding its amount and a box per reaction (7).

7. COMPOSED DIRECT, any network (book Ch 1 ``sec.same_calculus``, *Chemical
   reactions*; plan P43 (b) job 2, David's choice of 2026-10-10). Encoding 6
   taken apart: a CHEMICAL box per species (``chemical_box``), the
   configuration coalgebra on ``R`` holding its amount ``x_s`` with response
   ``dt``, its amount broadcast to each reaction that touches it; a stateless
   REACTION box per reaction (``reaction_box``), hearing the amounts of the
   chemicals it touches and returning to each of them the 1-form component
   ``zeta_rs v_r(x)`` -- its net rate times that chemical's stoichiometric
   number -- which is not exact (its Jacobian is not symmetric once a reaction
   consumes two chemicals, as the waterfall's coupling is not); and a routing
   wire (``reaction_wire``). ``composed_mass_action_arrangement`` is
   ``compose_seq(tensor(chemicals, reactions), wire)``; pullback-and-sum
   gathers at each chemical the sum over reactions of ``zeta_rs v_r``, so one
   ``Phi`` step is the Euler mass-action step at any molecularity, reversible
   or not. Item 3's no-go does not reach it, being about exact forcings. As in
   6, nothing is descended: on a detailed-balanced network the free energy
   ``F`` falls along the run because the mass-action flow dissipates it (a
   measured fact, at small ``dt``), not because the arrangement's forcing is
   ``-dF``.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Any, Callable, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .pc import PCMorphism
from .rvect import ReactiveVectorSpace, constant, diagonal, euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter


# ---------------------------------------------------------------------------
# Mass-action ground truth (used by the tests as the reference; the dynamics
# of encodings 1-5 never call it -- they arise from Phi. Encoding 6 is it.)
# ---------------------------------------------------------------------------


def one_way_fluxes(
    alpha: Array, beta: Array, kf: Array, kb: Array
) -> Callable[[Array], Tuple[Array, Array]]:
    """``x |-> (f, b)`` with ``f_r = kf_r x^{alpha_r}``, ``b_r = kb_r x^{beta_r}``."""
    alpha = jnp.asarray(alpha, float)
    beta = jnp.asarray(beta, float)
    kf = jnp.asarray(kf, float)
    kb = jnp.asarray(kb, float)

    def fluxes(x: Array) -> Tuple[Array, Array]:
        f = kf * jnp.prod(x[None, :] ** alpha, axis=1)
        b = kb * jnp.prod(x[None, :] ** beta, axis=1)
        return f, b

    return fluxes


def mass_action_rhs(
    alpha: Array, beta: Array, kf: Array, kb: Array
) -> Callable[[Array], Array]:
    """The mass-action vector field ``x |-> sum_r zeta_r v_r(x)``."""
    Z = jnp.asarray(beta, float) - jnp.asarray(alpha, float)
    fluxes = one_way_fluxes(alpha, beta, kf, kb)

    def rhs(x: Array) -> Array:
        f, b = fluxes(x)
        return Z.T @ (f - b)

    return rhs


def check_detailed_balance(
    alpha: Array, beta: Array, kf: Array, kb: Array, x_eq: Array, tol: float = 1e-9
) -> Array:
    """Verify ``kf_r x_eq^{alpha_r} = kb_r x_eq^{beta_r}`` for every ``r``.

    Returns the equilibrium fluxes ``w_r`` (the common value); raises
    ``ValueError`` if ``x_eq`` is not a detailed-balance point (relative
    residual above ``tol``), or if positivity/reversibility fails.
    """
    x_eq = jnp.asarray(x_eq, float)
    if not bool(jnp.all(x_eq > 0)):
        raise ValueError("check_detailed_balance: x_eq must be strictly positive")
    if not (bool(jnp.all(jnp.asarray(kf, float) > 0)) and bool(jnp.all(jnp.asarray(kb, float) > 0))):
        raise ValueError("check_detailed_balance: need kf, kb > 0 (reversibility)")
    wf, wb = one_way_fluxes(alpha, beta, kf, kb)(x_eq)
    resid = jnp.abs(wf - wb) / jnp.maximum(jnp.maximum(wf, wb), 1e-300)
    if not bool(jnp.all(resid <= tol)):
        raise ValueError(
            f"check_detailed_balance: x_eq is not a detailed-balance point "
            f"(max relative flux residual {float(jnp.max(resid)):.3e} > {tol:g})"
        )
    return wf


# ---------------------------------------------------------------------------
# Closed systems: the shared step, and the shape of a composite's state.
# ---------------------------------------------------------------------------


_IN_POS = (jnp.zeros(0), trivial_omega(0))
_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def flat_state(state: Any) -> Array:
    """The flat coordinate vector of a configuration state.

    A composite ``compose_seq(tensor(boxes), wire)`` has parameter
    ``(P_1 (x) ... (x) P_K) (x) P_wire`` (``wiring.compose_seq``), so its state
    is the nested tuple ``(nest_left([s_1, ..., s_K]), s_wire)`` with
    ``s_wire = R^0`` (a stateless box's ``s_k`` is ``R^0`` too). CDLM's
    ``direct_sum`` flattened all of this into one array, which is the vector
    the closed-system helpers below take and return; this concatenates the
    leaves in that order (for configuration coalgebras it equals the base
    position ``P.position(state)``). ``state_from_flat`` is its inverse.
    """
    return jnp.concatenate([jnp.asarray(leaf).reshape(-1) for leaf in jax.tree_util.tree_leaves(state)])


def state_from_flat(template: Any, flat: Array) -> Any:
    """The state shaped like ``template`` (e.g. ``O.state``, the coalgebra's
    ``init``) whose leaves are the consecutive slices of ``flat``; inverse of
    ``flat_state``."""
    leaves, treedef = jax.tree_util.tree_flatten(template)
    flat = jnp.asarray(flat, float)
    out, i = [], 0
    for leaf in leaves:
        k = int(jnp.size(leaf))
        out.append(flat[i:i + k].reshape(jnp.shape(leaf)))
        i += k
    return jax.tree_util.tree_unflatten(treedef, out)


def crn_step(O: PCMorphism, x: Array) -> Array:
    """One ``Phi`` Euler step of a closed system from the flat coordinates ``x``
    (the concentrations for the monolithic encodings 1 and 6; the composite's
    flat state, ``flat_state``, for encoding 2). The coalgebra's own state is
    ``state_from_flat(O.state, x)``; the result is flattened again."""
    _, _, new_state = O.with_state(state_from_flat(O.state, x)).run_one(_IN_POS, lambda _o: _IN_DIR)
    return flat_state(new_state)


def step_matrix(P: CotCoalg, flat: Array) -> Array:
    """The matrix ``xi |-> flat(P.step(s, xi)) - flat(s)`` at the state ``s``
    with flat coordinates ``flat``: for a configuration coalgebra with constant
    sharp this IS the composite sharp (``rvect.direct_sum`` of the boxes'),
    which the book stores inside ``step`` rather than as a field
    (sec.matter)."""
    s = state_from_flat(P.init, flat)
    base = flat_state(s)
    return jax.jacobian(lambda xi: flat_state(P.step(s, xi)) - base)(jnp.zeros(P.Q_dim))


# ---------------------------------------------------------------------------
# 1. The monolithic encoding: Mielke--Onsager sharp + free-energy forcing.
# ---------------------------------------------------------------------------


def log_mean(u: Array, v: Array) -> Array:
    """The logarithmic mean ``Lambda(u, v) = (u - v)/(log u - log v)``, ``Lambda(u,u) = u``.

    Guarded at ``u ~ v``: when ``|log(u/v)| < 1e-10`` the midpoint ``(u+v)/2``
    is used; the two branches then differ by relative ``O(log(u/v)^2) < 1e-20``,
    below rounding, so the guard costs no exactness.
    """
    r = jnp.log(u / v)
    close = jnp.abs(r) < 1e-10
    safe_r = jnp.where(close, 1.0, r)
    return jnp.where(close, 0.5 * (u + v), (u - v) / safe_r)


def onsager_rvect(
    alpha: Array, beta: Array, kf: Array, kb: Array, dt: float = 1.0
) -> ReactiveVectorSpace:
    """The Mielke--Onsager reactive vector space ``(R^n, dt * K(x))`` (module docstring, 1).

    ``K(x) = sum_r Lambda(f_r, b_r) zeta_r zeta_r^T`` is symmetric PSD on
    ``x > 0``; state-dependent, like ``replicator.shahshahani``. ``dt`` is the
    Euler step, carried by the sharp.
    """
    Z = jnp.asarray(beta, float) - jnp.asarray(alpha, float)
    n = int(Z.shape[1])
    fluxes = one_way_fluxes(alpha, beta, kf, kb)

    def sharp_fn(x: Array) -> Array:
        f, b = fluxes(x)
        lam = log_mean(f, b)
        return dt * (Z.T * lam) @ Z

    return ReactiveVectorSpace(dim=n, sharp_fn=sharp_fn)


def free_energy(x_eq: Array) -> Callable[[Array], Array]:
    """The free energy ``F(x) = sum_i x_i (log(x_i/x_eq_i) - 1)`` relative to ``x_eq``."""
    x_eq = jnp.asarray(x_eq, float)

    def F(x: Array) -> Array:
        return jnp.sum(x * (jnp.log(x / x_eq) - 1.0))

    return F


def crn_arrangement(
    alpha: Array, beta: Array, kf: Array, kb: Array, x_eq: Array, dt: float = 1.0,
    parameter=configuration,
) -> Arrangement:
    """The closed arrangement ``(parameter(R^n, dt*K(x)), !, !, -dF) : I -> I`` (module docstring, 1).

    The TRICK realization (module docstring, 6): the forcing is exact,
    ``omega = -dF``, and the mass-action field is manufactured by the
    state-dependent Onsager sharp. Verifies detailed balance at ``x_eq``
    (raising ``ValueError`` otherwise), since without it the Onsager pair
    ``(K, F)`` does not reproduce the mass-action field and the construction
    would silently change the kinetics.
    """
    check_detailed_balance(alpha, beta, kf, kb, x_eq)
    F = free_energy(x_eq)
    Q = onsager_rvect(alpha, beta, kf, kb, dt)

    return Arrangement(
        P=parameter(Q),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -F(q)),
        label="crn",
    )


def crn_dynamics(
    alpha: Array, beta: Array, kf: Array, kb: Array, x_eq: Array, dt: float = 1.0,
    parameter=configuration,
) -> PCMorphism:
    """``Phi`` of the monolithic reaction-network arrangement."""
    return Phi(crn_arrangement(alpha, beta, kf, kb, x_eq, dt, parameter))


# ---------------------------------------------------------------------------
# 2. The compositional encoding for unimolecular reversible networks
#    (kuramoto pattern: species boxes + reaction boxes + routing wire).
# ---------------------------------------------------------------------------


def unimolecular_stoichiometry(
    n_species: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[Array, Array]:
    """``(alpha, beta)`` for the network with one reaction ``A_i <-> A_j`` per edge."""
    R = len(edges)
    alpha = jnp.zeros((R, n_species))
    beta = jnp.zeros((R, n_species))
    for r, (i, j) in enumerate(edges):
        alpha = alpha.at[r, i].set(1.0)
        beta = beta.at[r, j].set(1.0)
    return alpha, beta


def _incidences(
    n_species: int, edges: Sequence[Tuple[int, int]]
) -> Tuple[List[int], List[int]]:
    """Species degrees and the routing bijection (cf. ``kuramoto._incidences``).

    Species ``s`` gets one out-port per reaction it takes part in (out-ports
    species-major); reaction box ``r = (i, j)`` has in-port 0 for ``a_i`` and
    in-port 1 for ``a_j`` (in-ports reaction-major). Both sides have ``2R``
    ports; ``perm`` satisfies ``in_f(m_out)[p] = m_out[perm[p]]``.
    """
    for r, (i, j) in enumerate(edges):
        if i == j:
            raise ValueError(f"reaction {r} = ({i}, {j}) is a self-loop; not allowed")
        if not (0 <= i < n_species and 0 <= j < n_species):
            raise ValueError(f"reaction {r} = ({i}, {j}) out of range")

    out_port: Dict[Tuple[int, int], int] = {}
    degrees: List[int] = []
    idx = 0
    for s in range(n_species):
        deg = 0
        for r, (i, j) in enumerate(edges):
            if s in (i, j):
                out_port[(s, r)] = idx
                idx += 1
                deg += 1
        degrees.append(deg)

    perm: List[int] = []
    for r, (i, j) in enumerate(edges):
        perm.append(out_port[(i, r)])
        perm.append(out_port[(j, r)])
    return degrees, perm


def species_box(x_eq_i: float, degree: int, dt: float = 1.0, parameter=configuration) -> Arrangement:
    """The species box ``Sp_i : <R^0|R^0> -> <R^0 | R^{degree}>``.

    Parameter ``parameter(Q_i)``, ``Q_i = R`` holding the concentration
    ``x_i``, with CONSTANT scalar sharp ``dt * x_eq_i`` (the species' own
    equilibrium weight -- the only equilibrium datum the species carries);
    readout the activity ``a_i = x_i / x_eq_i`` broadcast to the ``degree``
    incident reaction ports; no inputs, zero forcing.
    """
    x_eq_i = float(x_eq_i)
    if x_eq_i <= 0:
        raise ValueError("species_box: x_eq_i must be positive")

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(q / x_eq_i, degree)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    return Arrangement(
        P=parameter(diagonal(jnp.full(1, dt * x_eq_i))),
        in_dim_M=0, out_dim_M=0,
        in_dim_N=0, out_dim_N=degree,
        out_f=out_f, in_f=in_f, omega=zero_form,
        label=f"Sp(xeq={x_eq_i:g})",
    )


def unimolecular_reaction_box(w: float) -> Arrangement:
    """The stateless reaction box ``Rxn_r : <R^0|R^0> -> <R^2 | R^0>``.

    Trivial parameter; two input ports carrying the endpoint activities
    ``(a_i, a_j)``; forcing ``-d`` of the cost ``(w/2)(a_i - a_j)^2``, ``w``
    the reaction's equilibrium flux (its only datum). All its dynamics flows
    back through the wiring as the cotangent pullback of this forcing to the
    two species' parameters (cf. ``kuramoto.kuramoto_coupling``).
    """
    w = float(w)

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return 0.5 * w * (n_in[0] - n_in[1]) ** 2

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=0,
        in_dim_N=2, out_dim_N=0,
        out_f=out_f, in_f=in_f, omega=exact(lambda q, m_out, n_in: -U(q, m_out, n_in)),
        label=f"Rxn(w={w:g})",
    )


def unimolecular_wire(
    n_species: int, edges: Sequence[Tuple[int, int]]
) -> Arrangement:
    """The routing wire (image of a finite-set lens under ``R^-``, lem.lens_pow).

    Source ``<R^{2R} | R^{2R}>`` (all boxes' ports), target the unit
    ``<R^0|R^0>``, trivial parameter, zero forcing; routes each reaction box's
    two in-ports from its endpoint species' broadcast out-ports by the
    bijection ``perm`` of ``_incidences`` (cf. ``kuramoto.kuramoto_wire``).
    """
    R = len(edges)
    _, perm = _incidences(n_species, edges)
    perm = jnp.asarray(perm, dtype=int)

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return m_out[perm]

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=2 * R, out_dim_M=2 * R,
        in_dim_N=0, out_dim_N=0,
        out_f=out_f, in_f=in_f, omega=zero_form,
        label=f"uni_wire(n={n_species}, R={R})",
    )


def unimolecular_arrangement(
    n_species: int,
    edges: Sequence[Tuple[int, int]],
    kf: Sequence[float],
    kb: Sequence[float],
    x_eq: Array,
    dt: float = 1.0,
    parameter=configuration,
) -> Arrangement:
    """The closed unimolecular network, by genuine composition in ``arr``:

        compose_seq( tensor(Sp_1, ..., Sp_n, Rxn_1, ..., Rxn_R),
                     unimolecular_wire(n, edges) )  :  <R^0|R^0> -> <R^0|R^0>,

    with parameter base ``R^n`` (the concentrations, species order; the
    stateless reaction boxes contribute nothing) and CONSTANT diagonal sharp
    ``dt * diag(x_eq)``. The total forcing ``-d sum_r (w_r/2)(a_i - a_j)^2``
    emerges from ``compose_seq``'s pullback-and-sum -- it is written nowhere
    by hand. Detailed balance ``kf_r x_eq_i = kb_r x_eq_j`` is verified per
    reaction (this fixes ``w_r``); with cycles present this is exactly the
    Wegscheider condition, and violating rates are rejected. The composite's
    state is nested (``flat_state``); its flat coordinates are ``x``.
    """
    x_eq = jnp.asarray(x_eq, float)
    alpha, beta = unimolecular_stoichiometry(n_species, edges)
    w = check_detailed_balance(alpha, beta, kf, kb, x_eq)

    degrees, _ = _incidences(n_species, edges)
    boxes = [
        species_box(float(x_eq[s]), degrees[s], dt, parameter) for s in range(n_species)
    ] + [unimolecular_reaction_box(float(w[r])) for r in range(len(edges))]
    wired = compose_seq(tensor_arrangements(boxes), unimolecular_wire(n_species, edges))
    return replace(wired, label=f"uni_crn(n={n_species}, R={len(edges)})")


def unimolecular_dynamics(
    n_species: int,
    edges: Sequence[Tuple[int, int]],
    kf: Sequence[float],
    kb: Sequence[float],
    x_eq: Array,
    dt: float = 1.0,
    parameter=configuration,
) -> PCMorphism:
    """``Phi`` of the compositional unimolecular arrangement."""
    return Phi(unimolecular_arrangement(n_species, edges, kf, kb, x_eq, dt, parameter))


# ---------------------------------------------------------------------------
# 3. The obstruction certificate (module docstring, 3).
# ---------------------------------------------------------------------------


def separability_ratio(
    rhs: Callable[[Array], Array], i: int, j: int
) -> Callable[[Array], Array]:
    """``x |-> (dV_j/dx_i)(x) / (dV_i/dx_j)(x)`` for the vector field ``V = rhs``.

    In any species-box realization ``x_k' = -g_k(x_k) dU/dx_k`` (the general
    compositional form: direct-sum sharp, shared potential) the closedness of
    ``dU`` forces this ratio to equal ``g_j(x_j)/g_i(x_i)`` -- a separable
    function, independent of every OTHER coordinate, in every per-species
    coordinate system. A network whose ratio varies with a third species
    therefore has NO species-box realization; ``A + B <-> C`` fails this test
    (the (A, C) ratio is ``(kf/kb) x_B``). Only meaningful where the
    denominator entry is nonzero; for the irreversible failure mode (a zero
    denominator against a nonzero numerator) compare the Jacobian entries
    directly.
    """

    def ratio(x: Array) -> Array:
        J = jax.jacobian(rhs)(x)
        return J[j, i] / J[i, j]

    return ratio


# ---------------------------------------------------------------------------
# 4. Arbitrary Petri nets: doubled transition boxes, species as wiring
#    (module docstring, 4; the paper's "Petri nets" item).
# ---------------------------------------------------------------------------


# The transistor pair of logic.py: sharpR(xi, xi') = (xi', 0). Repeated here
# rather than imported so that this module does not depend on an extension.
_TRANSISTOR_SHARP = jnp.array([[0.0, 1.0], [0.0, 0.0]])


def petri_check(m: Array, n: Array, r: Array, x0: Array) -> Tuple[Array, Array, Array, Array]:
    """Validate a Petri net ``(m, n, r, x0)`` and return it as float arrays."""
    m = jnp.asarray(m, float)
    n = jnp.asarray(n, float)
    r = jnp.asarray(r, float)
    x0 = jnp.asarray(x0, float)
    if m.shape != n.shape or m.ndim != 2:
        raise ValueError("petri_check: m, n must be (T, S) of the same shape")
    if r.shape != (m.shape[0],) or x0.shape != (m.shape[1],):
        raise ValueError("petri_check: r must be (T,) and x0 must be (S,)")
    if not bool(jnp.all(m >= 0) and jnp.all(n >= 0)):
        raise ValueError("petri_check: arc multiplicities must be nonnegative")
    if not bool(jnp.all(m == jnp.round(m)) and jnp.all(n == jnp.round(n))):
        raise ValueError("petri_check: arc multiplicities must be integers")
    if not bool(jnp.all(r >= 0)):
        raise ValueError("petri_check: rate constants must be nonnegative")
    if not bool(jnp.all(x0 >= 0)):
        raise ValueError("petri_check: initial concentrations must be nonnegative")
    return m, n, r, x0


def petri_incidences(m: Array, n: Array) -> List[List[int]]:
    """``S_t``, the species each transition touches (as reactant or product)."""
    m, n = jnp.asarray(m, float), jnp.asarray(n, float)
    return [
        [s for s in range(m.shape[1]) if bool(m[t, s] > 0 or n[t, s] > 0)]
        for t in range(m.shape[0])
    ]


def petri_transition_box(
    m_local: Array, r_t: float, dt: float = 1.0, label: str = "Trans", parameter=configuration,
) -> Arrangement:
    """The doubled transition box ``<R^0|R^0> -> <R^{S_t} | R>`` (module docstring, 4).

    Parameter ``parameter(R^2, sharpR)`` with the transistor sharp ``(xi, xi')
    |-> (dt * xi', 0)``, state ``(eps_t, eps'_t)``; readout the extent
    ``eps_t``; inputs the concentrations at the ``len(m_local)`` species the
    transition touches, in the order given by ``petri_incidences``; forcing
    ``-dU_t`` for

        U_t((eps_t, eps'_t), x_local) = -eps'_t * r_t * prod_s x_s^{m_local_s}.

    ``m_local`` is the transition's REACTANT multiplicities at those species
    (its product multiplicities live in the wiring, not here). ``dt`` rides on
    the sharp, as in ``onsager_rvect``.
    """
    m_local = jnp.asarray(m_local, float)
    r_t = float(r_t)
    if r_t < 0:
        raise ValueError("petri_transition_box: rate constant must be nonnegative")
    k = int(m_local.shape[0])

    def out_f(q: Array, m_out: Array) -> Array:
        return q[:1]  # report the extent eps_t

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def U(q: Array, m_out: Array, n_in: Array) -> Array:
        return -q[1] * r_t * jnp.prod(n_in ** m_local)

    return Arrangement(
        P=parameter(constant(dt * _TRANSISTOR_SHARP)),
        in_dim_M=0, out_dim_M=0,
        in_dim_N=k, out_dim_N=1,
        out_f=out_f, in_f=in_f, omega=exact(lambda q, m_out, n_in: -U(q, m_out, n_in)),
        label=f"{label}(r={r_t:g})",
    )


def petri_wire(m: Array, n: Array, x0: Array) -> Arrangement:
    """The species wiring ``(x)_t <R^{S_t}|R> -> <R^0|R^0>`` (module docstring, 4).

    Static (trivial parameter, zero forcing) but not a prism: its input map is
    the AFFINE map carrying the transitions' reported extents to the
    concentrations

        x_s = x0_s + sum_t (n_ts - m_ts) eps_t,

    gathered to each transition's in-ports (cf. the feedback stage of
    ``amplifier.py``, whose input map is likewise affine rather than a
    permutation).
    """
    m, n, _, x0 = petri_check(m, n, jnp.zeros(jnp.asarray(m).shape[0]), x0)
    Z = n - m
    idx = jnp.asarray([s for St in petri_incidences(m, n) for s in St], dtype=int)

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return (x0 + Z.T @ m_out)[idx]  # m_out = the reported extents

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=int(idx.shape[0]), out_dim_M=int(Z.shape[0]),
        in_dim_N=0, out_dim_N=0,
        out_f=out_f, in_f=in_f, omega=zero_form,
        label=f"species_wire(S={int(Z.shape[1])}, T={int(Z.shape[0])})",
    )


def petri_arrangement(
    m: Array, n: Array, r: Array, x0: Array, dt: float = 1.0, parameter=configuration,
) -> Arrangement:
    """The closed Petri net, by genuine composition in ``arr``:

        compose_seq( tensor(Trans_t for t in T), petri_wire(m, n, x0) ),

    a ``<R^0|R^0> -> <R^0|R^0>`` arrangement with parameter base ``R^{2T}``
    holding ``(eps_t, eps'_t)_t`` and block-diagonal nilpotent sharp. The total
    forcing ``-d(-sum_t eps'_t f_t(x))`` emerges from ``compose_seq``'s
    pullback-and-sum; it is written nowhere by hand.
    """
    m, n, r, x0 = petri_check(m, n, r, x0)
    St = petri_incidences(m, n)
    boxes = [
        petri_transition_box(m[t, jnp.asarray(St[t], dtype=int)], float(r[t]), dt, parameter=parameter)
        for t in range(int(m.shape[0]))
    ]
    wired = compose_seq(tensor_arrangements(boxes), petri_wire(m, n, x0))
    return replace(wired, label=f"petri(S={int(m.shape[1])}, T={int(m.shape[0])})")


def petri_dynamics(
    m: Array, n: Array, r: Array, x0: Array, dt: float = 1.0, parameter=configuration,
) -> PCMorphism:
    """``Phi`` of the Petri-net arrangement."""
    return Phi(petri_arrangement(m, n, r, x0, dt, parameter))


def petri_initial_state(num_transitions: int, eps_prime: float = 1.0) -> Array:
    """The state ``(eps_t, eps'_t)_t = (0, eps_prime)_t`` at time zero, in flat
    coordinates (``flat_state``; one ``R^2`` per transition box).

    ``eps_prime`` is arbitrary and never changes: it only scales ``U``, whose
    ``eps'``-derivative -- the only component the sharp reads -- is independent
    of it.
    """
    return jnp.tile(jnp.array([0.0, float(eps_prime)]), num_transitions)


def petri_extents(state: Array) -> Array:
    """The extents ``(eps_t)_t`` of a flat Petri-net state (its even coordinates)."""
    return jnp.asarray(state)[0::2]


def petri_concentrations(m: Array, n: Array, x0: Array, state: Array) -> Array:
    """``x = x0 + Z^T eps`` read off a flat Petri-net state."""
    m, n = jnp.asarray(m, float), jnp.asarray(n, float)
    return jnp.asarray(x0, float) + (n - m).T @ petri_extents(state)


def petri_step(O: PCMorphism, state: Array) -> Array:
    """One ``Phi`` tick of a closed Petri-net coalgebra, on flat coordinates
    (``crn_step``)."""
    return crn_step(O, state)


# ---------------------------------------------------------------------------
# 5. Arbitrary Petri nets in log coordinates: doubled species boxes,
#    stateless transitions, prism wiring; positive by construction
#    (module docstring, 5).
# ---------------------------------------------------------------------------


def petri_log_incidences(m: Array, n: Array) -> Tuple[List[int], List[int]]:
    """Species degrees and the routing bijection for the log encoding.

    Species ``s`` gets one PAIR of out-ports ``(x_s, y'_s)`` per transition it
    takes part in (out-ports species-major, ``x`` before ``y'``); transition
    ``t`` expects its touched species' pairs in ``petri_incidences`` order
    (in-ports transition-major). ``perm`` satisfies
    ``in_f(m_out)[p] = m_out[perm[p]]`` and is a bijection: the wire is a
    prism, unlike the affine ``petri_wire``.
    """
    St = petri_incidences(m, n)
    T, S = jnp.asarray(m).shape

    out_port: Dict[Tuple[int, int], int] = {}
    degrees: List[int] = []
    idx = 0
    for s in range(int(S)):
        deg = 0
        for t in range(int(T)):
            if s in St[t]:
                out_port[(s, t)] = idx
                idx += 2
                deg += 1
        degrees.append(deg)

    perm: List[int] = []
    for t in range(int(T)):
        for s in St[t]:
            perm.append(out_port[(s, t)])
            perm.append(out_port[(s, t)] + 1)
    return degrees, perm


def petri_log_species_box(degree: int, dt: float = 1.0, parameter=configuration) -> Arrangement:
    """The doubled species box ``<R^0|R^0> -> <R^0 | R^{2 degree}>``.

    Parameter ``parameter(R^2, sharpR)`` with the transistor sharp ``(xi, xi')
    |-> (dt * xi', 0)``, state ``(y_s, y'_s)``; readout the pair ``(x_s, y'_s)``,
    ``x_s = exp(y_s)``, broadcast to the ``degree`` incident transitions; no
    inputs, zero forcing. The exp Jacobian of the readout pulls incoming
    ``x``-port covectors back into the ``xi``-slot, which the sharp discards;
    the ``y'``-port covectors land untouched in the ``xi'``-slot, which drives
    ``y_s``. The box carries NO reaction data -- not even its own initial
    concentration, which is the initial state, not a structural constant.
    """

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.tile(jnp.stack([jnp.exp(q[0]), q[1]]), degree)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    return Arrangement(
        P=parameter(constant(dt * _TRANSISTOR_SHARP)),
        in_dim_M=0, out_dim_M=0,
        in_dim_N=0, out_dim_N=2 * degree,
        out_f=out_f, in_f=in_f, omega=zero_form,
        label=f"LogSp(deg={degree})",
    )


def petri_log_transition_box(
    m_local: Array, zeta_local: Array, r_t: float
) -> Arrangement:
    """The STATELESS transition box ``<R^0|R^0> -> <R^0 | R^{2k}>``-dual,
    ``k = len(m_local)``: trivial parameter, in-ports the pairs
    ``(x_s, y'_s)`` at the ``k`` species it touches, forcing ``-dV_t`` for

        V_t = -f_t(x) * sum_s zeta_local_s * y'_s / x_s,
        f_t(x) = r_t * prod_s x_s^{m_local_s}.

    ``-dV_t/dy'_s = zeta_local_s f_t(x)/x_s`` is the (signed, per-species)
    log-rate itself, independent of ``y'``; all ``x``-derivatives flow to
    discarded slots. Unlike ``petri_transition_box``, the box carries its
    PRODUCT multiplicities too (inside ``zeta_local``), since delivery no
    longer routes through affine wiring.
    """
    m_local = jnp.asarray(m_local, float)
    zeta_local = jnp.asarray(zeta_local, float)
    r_t = float(r_t)
    if r_t < 0:
        raise ValueError("petri_log_transition_box: rate constant must be nonnegative")
    k = int(m_local.shape[0])

    def out_f(q: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q: Array, m_out: Array, n_in: Array) -> Array:
        return jnp.zeros(0)

    def V(q: Array, m_out: Array, n_in: Array) -> Array:
        x, yp = n_in[0::2], n_in[1::2]
        f = r_t * jnp.prod(x ** m_local)
        return -f * jnp.sum(zeta_local * yp / x)

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=0,
        in_dim_N=2 * k, out_dim_N=0,
        out_f=out_f, in_f=in_f, omega=exact(lambda q, m_out, n_in: -V(q, m_out, n_in)),
        label=f"LogTrans(r={r_t:g})",
    )


def petri_log_wire(m: Array, n: Array) -> Arrangement:
    """The routing prism: a port BIJECTION, cf. ``unimolecular_wire``.

    No affine map and no ``x0``: species compute their own concentrations,
    so the wire merely routes each species' broadcast ``(x_s, y'_s)`` pair to
    the transitions touching it.
    """
    _, perm = petri_log_incidences(m, n)
    perm_arr = jnp.asarray(perm, dtype=int)
    P = int(perm_arr.shape[0])

    def out_f(q_wire: Array, m_out: Array) -> Array:
        return jnp.zeros(0)

    def in_f(q_wire: Array, m_out: Array, n_in: Array) -> Array:
        return m_out[perm_arr]

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=P, out_dim_M=P,
        in_dim_N=0, out_dim_N=0,
        out_f=out_f, in_f=in_f, omega=zero_form,
        label=f"log_wire(P={P})",
    )


def petri_log_arrangement(
    m: Array, n: Array, r: Array, dt: float = 1.0, parameter=configuration,
) -> Arrangement:
    """The closed log-coordinate Petri net, by genuine composition in ``arr``:

        compose_seq( tensor(LogSp_s for s in S, LogTrans_t for t in T),
                     petri_log_wire(m, n) )  :  <R^0|R^0> -> <R^0|R^0>,

    with parameter base ``R^{2S}`` holding ``(y_s, y'_s)_s`` (the stateless
    transition boxes contribute nothing) and block-diagonal nilpotent sharp.
    The total forcing ``-d(-sum_t f_t(x) sum_s zeta_ts y'_s/x_s)`` emerges from
    ``compose_seq``'s pullback-and-sum. Note ``x0`` is NOT an argument: initial
    concentrations enter through ``petri_log_initial_state``.
    """
    m, n, r, _ = petri_check(m, n, r, jnp.zeros(jnp.asarray(m).shape[1]))
    Z = n - m
    St = petri_incidences(m, n)
    degrees, _ = petri_log_incidences(m, n)
    boxes = [petri_log_species_box(degrees[s], dt, parameter) for s in range(int(m.shape[1]))] + [
        petri_log_transition_box(
            m[t, jnp.asarray(St[t], dtype=int)],
            Z[t, jnp.asarray(St[t], dtype=int)],
            float(r[t]),
        )
        for t in range(int(m.shape[0]))
    ]
    wired = compose_seq(tensor_arrangements(boxes), petri_log_wire(m, n))
    return replace(wired, label=f"petri_log(S={int(m.shape[1])}, T={int(m.shape[0])})")


def petri_log_dynamics(
    m: Array, n: Array, r: Array, dt: float = 1.0, parameter=configuration,
) -> PCMorphism:
    """``Phi`` of the log-coordinate Petri-net arrangement."""
    return Phi(petri_log_arrangement(m, n, r, dt, parameter))


def petri_log_initial_state(x0: Array, y_prime: float = 1.0) -> Array:
    """The state ``(y_s, y'_s)_s = (log x0_s, y_prime)_s`` at time zero, in flat
    coordinates (``flat_state``; one ``R^2`` per species box, the stateless
    transitions contributing nothing).

    Requires ``x0 > 0`` strictly: the chart is ``y = log x``.  ``y_prime`` is
    arbitrary and never changes, as in ``petri_initial_state``.
    """
    x0 = jnp.asarray(x0, float)
    if not bool(jnp.all(x0 > 0)):
        raise ValueError("petri_log_initial_state: initial concentrations must be strictly positive")
    return jnp.stack(
        [jnp.log(x0), jnp.full(x0.shape[0], float(y_prime))], axis=1
    ).reshape(-1)


def petri_log_concentrations(state: Array) -> Array:
    """``x = exp(y)`` read off a flat log-Petri state (its even coordinates)."""
    return jnp.exp(jnp.asarray(state)[0::2])


def log_mass_action_step(
    m: Array, n: Array, r: Array, dt: float = 1.0
) -> Callable[[Array], Array]:
    """Ground truth for the tests: ``x |-> x * exp(dt * (Z^T f)(x) / x)``,
    the forward-Euler step of mass action in log coordinates."""
    m = jnp.asarray(m, float)
    Z = jnp.asarray(n, float) - m
    r = jnp.asarray(r, float)

    def step(x: Array) -> Array:
        f = r * jnp.prod(x[None, :] ** m, axis=1)
        return x * jnp.exp(dt * (Z.T @ f) / x)

    return step


# ---------------------------------------------------------------------------
# 6. The DIRECT encoding: the mass-action field as a non-exact forcing on the
#    plain configuration coalgebra (module docstring, 6; STATUS.md §2, trick vs direct).
# ---------------------------------------------------------------------------


def mass_action_arrangement(
    alpha: Array, beta: Array, kf: Array, kb: Array, dt: float = 1.0, parameter=configuration,
) -> Arrangement:
    """The closed arrangement ``(parameter(R^n, dt * I), !, !, V^flat) : I -> I`` (module docstring, 6).

    The DIRECT realization: parameter the plain configuration coalgebra on
    ``R^n`` with Euclidean sharp ``dt * I`` (``rvect.euclidean``), forcing the
    mass-action vector field written as a covector by hand,

        omega(x, -, -) = (sum_r zeta_r v_r(x), 0, 0),

    with NO potential: it is exact iff ``DV`` is symmetric, which already
    ``A + B <-> C`` violates (module docstring, 3 and 6), and in general it is
    a genuinely non-exact forcing (def.arr, sec.forcing). One ``Phi`` step is
    ``x + dt * V(x)`` for ANY network -- no detailed balance, reversibility or
    molecularity hypothesis, so it runs what ``crn_arrangement`` (the TRICK,
    module docstring 1) rejects; on a detailed-balanced network the two agree
    step for step. Nothing is descended and nothing is certified: see the
    vacuity guard.
    """
    V = mass_action_rhs(alpha, beta, kf, kb)
    n = int(jnp.asarray(alpha).shape[1])

    def omega(q: Array, m_out: Array, n_in: Array):
        return V(q), jnp.zeros_like(m_out), jnp.zeros_like(n_in)

    return Arrangement(
        P=parameter(euclidean(n, dt)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega,
        label="mass_action",
    )


def mass_action_dynamics(
    alpha: Array, beta: Array, kf: Array, kb: Array, dt: float = 1.0, parameter=configuration,
) -> PCMorphism:
    """``Phi`` of the direct mass-action arrangement."""
    return Phi(mass_action_arrangement(alpha, beta, kf, kb, dt, parameter))


# ---------------------------------------------------------------------------
# 7. COMPOSED DIRECT: a box per chemical, a box per reaction (module docstring, 7).
# ---------------------------------------------------------------------------


def reaction_incidences(alpha: Array, beta: Array) -> Tuple[List[List[int]], List[int], List[int]]:
    """Which chemicals each reaction touches, each chemical's degree, and the routing bijection.

    Reaction ``r`` touches ``S_r = {s : alpha_rs > 0 or beta_rs > 0}`` (in species order) and has
    one in-port per chemical in ``S_r`` (in-ports reaction-major); chemical ``s`` has one
    out-port per reaction touching it (out-ports species-major, reactions in order). ``perm``
    satisfies ``in_f(m_out)[p] = m_out[perm[p]]``.
    """
    alpha = jnp.asarray(alpha, float)
    beta = jnp.asarray(beta, float)
    R, n = alpha.shape
    touched = [[s for s in range(n) if float(alpha[r, s]) > 0 or float(beta[r, s]) > 0]
               for r in range(R)]
    out_port: Dict[Tuple[int, int], int] = {}
    degrees: List[int] = []
    idx = 0
    for s in range(n):
        deg = 0
        for r in range(R):
            if s in touched[r]:
                out_port[(s, r)] = idx
                idx += 1
                deg += 1
        degrees.append(deg)
    perm = [out_port[(s, r)] for r in range(R) for s in touched[r]]
    return touched, degrees, perm


def chemical_box(x0: float, degree: int, dt: float = 1.0) -> Arrangement:
    """A chemical's box ``<R^0|R^0> -> <R^0 | R^{degree}>``: its state the amount ``x_s``.

    Configuration coalgebra on ``R`` with response ``dt``, initial amount ``x0``; reads out the
    amount to each of the ``degree`` reactions that touch it; no inputs, zero forcing.
    """
    return Arrangement(
        P=configuration(euclidean(1, dt), init=jnp.array([x0], dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=degree,
        out_f=lambda q, m_out: jnp.tile(q, degree),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form,
        label="chemical",
    )


def reaction_box(alpha_r: Array, beta_r: Array, kf: float, kb: float) -> Arrangement:
    """A reaction's stateless box ``<R^0|R^0> -> <R^k | R^0>``, ``k`` the chemicals it touches.

    ``alpha_r``, ``beta_r`` are its reactant and product numbers on those ``k`` chemicals. It
    hears their amounts ``x`` and returns to chemical ``s`` the component ``zeta_s v(x)``,
    ``zeta = beta_r - alpha_r``, ``v(x) = kf x^alpha_r - kb x^beta_r``: a 1-form on its input
    side written directly, with no potential (``kb = 0`` for a one-way reaction).
    """
    a = jnp.asarray(alpha_r, float)
    b = jnp.asarray(beta_r, float)
    zeta = b - a
    k = int(a.shape[0])

    def omega(q: Array, m_out: Array, n_in: Array):
        v = kf * jnp.prod(n_in ** a) - kb * jnp.prod(n_in ** b)
        return jnp.zeros_like(q), jnp.zeros_like(m_out), zeta * v

    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=0, in_dim_N=k, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=omega,
        label="reaction",
    )


def reaction_wire(alpha: Array, beta: Array) -> Arrangement:
    """The routing wire ``<R^P|R^P> -> <R^0|R^0>``: each reaction's in-ports from its chemicals'
    out-ports, by the bijection of ``reaction_incidences``; trivial parameter, zero forcing."""
    _, _, perm = reaction_incidences(alpha, beta)
    P = len(perm)
    perm = jnp.asarray(perm, dtype=int)
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=P, out_dim_M=P, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: m_out[perm],
        omega=zero_form,
        label="reaction_wire",
    )


def composed_mass_action_arrangement(
    alpha: Array, beta: Array, kf: Array, kb: Array, x0: Array, dt: float = 1.0,
) -> Arrangement:
    """The closed network ``compose_seq(tensor(chemicals, reactions), reaction_wire)`` (docstring, 7).

    Its parameter's position is the amounts ``x`` in species order (the reaction boxes and the
    wire are stateless), starting at ``x0``. The total forcing at chemical ``s``,
    ``sum_r zeta_rs v_r(x)``, is gathered by ``compose_seq``'s pullback-and-sum and written
    nowhere whole. No detailed-balance, reversibility or molecularity hypothesis.
    """
    alpha = jnp.asarray(alpha, float)
    beta = jnp.asarray(beta, float)
    kf = jnp.asarray(kf, float)
    kb = jnp.asarray(kb, float)
    x0 = jnp.asarray(x0, float)
    touched, degrees, _ = reaction_incidences(alpha, beta)
    chemicals = [chemical_box(float(x0[s]), degrees[s], dt) for s in range(len(degrees))]
    reactions = [reaction_box(alpha[r, jnp.asarray(t)], beta[r, jnp.asarray(t)], float(kf[r]), float(kb[r]))
                 for r, t in enumerate(touched)]
    wired = compose_seq(tensor_arrangements(chemicals + reactions), reaction_wire(alpha, beta))
    return replace(wired, label=f"composed_mass_action(n={len(degrees)}, R={len(touched)})")
