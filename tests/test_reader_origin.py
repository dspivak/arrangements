"""A candidate beginning of a reader, rebuilt as a composite arrangement.

The candidate is Codex's (the book repository's ``notes/reader-origin-model.md``, checked by its
``notes/check-reader-origin.py``): a selective catalytic apparatus ``q_1`` whose activity on fuel
rebuilds it and makes a protective boundary ``q_2``; in a retaining pore it establishes from a
trace; in the same arrangement with the pore box replaced by one publishing ``0``, started at that
high state, the boundary keeps the high state up. Two arrangements, each run as itself: the *pore*
arrangement and the *exposed* one. "Release" and "released" below name the exposed arrangement
started at the pore arrangement's end state; nothing in any run leaves anything, and the
replacement is the experimenter's (STATUS.md §2, "Not evidence of an event"). There it is one box
whose forcing is the ODE's own velocity, which by STATUS.md §2 certifies nothing beyond the ODE. The design here, and why it
is worth building, is the book repository's ``notes/reader-origin-arr.md`` (proposed; David,
2026-09-26, "both", i.e. build stages A and B). This module holds both stages.

**The field** (Codex's §3, dimensionless; ``kappa`` the match, 1 in stage A):

    V_1 = a F + k_r kappa F q_1 - (1 + l / (1 + q_2)) q_1 - q_1^2,
    V_2 = k_b kappa F q_1 - q_2,

with Codex's numbers ``F = 1``, ``k_r = 3``, ``k_b = 10``, leakage ``l = 0.2`` in the pore and
``4`` exposed, assembly ``a = 1e-6`` while establishing and ``0`` after.

**The predictions, fixed here before any construction code exists.** Each says whether it is a
*reproduction* (the construction was built to have it; STATUS.md §2, last paragraph) or a
*result* (it could have come out otherwise), and how it could fail.

(T0, reproduction) The direct composite -- species boxes on the plain configuration coalgebra,
held supplies in the drive idiom, one stateless transition box per term of ``V``, a routing
wire -- ticks as ``q + h V(q)`` to machine precision. Its runs recover Codex's table: established
in the pore at the exact pore equilibrium ``q_1 = (19 + sqrt 433)/20 = 1.990433``; exposed with
the same assembly, below ``1e-6``; released with boundary and supply, the exact exposed high
equilibrium ``(1.788153, 17.881527)``; the four controls (no fuel, no match, no rebuild, no
boundary production) and the unseeded pore, below ``1e-8``. Fails only by a construction error.

(T1a, reproduction of a hand computation) The composite's ``d omega``, by autodiff through
``compose_seq``, equals Codex's
``(k_b kappa F - l q_1 / (1 + q_2)^2) dq_1 ^ dq_2``, and it is the sum of exactly two children's
contributions, boundary production and the protected leak; every other child contributes zero.

(T1b, result, derived by hand before this module) On the open quadrant ``V = -G grad U`` with
the diagonal response ``G = h diag(q_1, (k_b kappa F / l)(1 + q_2)^2)`` and

    U = -a F log q_1 - (k_r kappa F - 1) q_1 + q_1^2 / 2 + l q_1 / (1 + q_2)
        + (l / (k_b kappa F)) (log(1 + q_2) + 1 / (1 + q_2)).

So a *structured* composite exists: species boxes with those state-dependent sharps, and one
exact child per term of ``U``. The two curl sources of T1a are then one child, the cost
``l q_1 / (1 + q_2)``, which gives both the protected leak and the boundary production. Its
ticks equal T0's to machine precision, and ``U`` decreases at every tick of the release run.
The response ratio ``g_2 / g_1`` is forced up to a constant (the separability condition of
``reaction_network.py`` item 3), so the fuel, the match and the pore's leakage cannot all be kept
out of the response. Able to fail: an error in the derivation makes the ticks disagree.

(T2a, reproduction) The exposed arrangement is the pore arrangement with the pore child
substituted by the drive publishing ``0`` -- a different inner box, not a changed coefficient --
and, started at the pore arrangement's end state, gives T0's release outcomes.

(T2b, result) Protection by mass-action binding instead of Codex's law: free apparatus ``A``,
free boundary ``B``, complex ``C = AB`` that does not leak, with ``q_1 = A + C``,
``q_2 = B + C``, binding ``A + B <-> C`` at ``k_on``, ``k_off = k_on / K``. The other terms are
Codex's, apportioned so that ``q_1`` and ``q_2`` obey his equations with the leak ``l A`` in
place of ``l q_1 / (1 + q_2)``: damage takes ``A`` to nothing and ``C`` to ``B``; the saturating
loss takes ``A`` and ``C`` in proportion; boundary loss takes ``B`` to nothing and ``C`` to
``A``. Predictions:
  (i) fast binding (``k_on = 20``, ``K = 1``): establishes in the pore arrangement, and the
      exposed arrangement keeps the high state, its ``q_1`` within 10% of Codex's 1.788153;
  (ii) there is a binding constant ``K_*`` below which the exposed arrangement has no high
      state at fast binding, and ``K_* < 1``: Codex's law is not at the edge;
  (iii) slow binding (``k_on = 0.1``, ``K = 1``): the exposed arrangement falls to zero, because
      the complex forms slower than the apparatus turns over;
  (iv) no species-box structured realization: ``separability_ratio`` for the pair ``(A, C)``
      varies with ``B`` at fixed ``A, C``. T1b's potential then belongs to Codex's
      two-coordinate reduction, not to this chemistry.
Able to fail: (i) and (iii) could each go the other way; (ii) could put ``K_*`` above 1; (iv)
could be separable once the other reactions are added.

Time is discrete (STATUS.md §1): ``h = 0.01`` for T0--T2a and ``1e-3`` for T2b, whose binding
is stiffer. Fixed points of a tick are the zeros of ``V`` and do not depend on ``h``.

**Stage B, the predictions, fixed here before its construction** (David, 2026-09-26, "yes").

(T3) *The match as a distance.* The substrate is a box given a response and a constant forcing
from one more child, so it drifts at speed ``v = 0.005`` away from the site by the dynamics,
exposed, from the established state. Exposed,
``q_2 = 10 kappa q_1`` and ``10 kappa q_1^2 - (10 kappa r - 1) q_1 + (4 - r) = 0`` with
``r = 3 kappa - 1``. The high state folds where the discriminant vanishes:
``kappa_* = 0.770433``, ``d_* = sqrt(2 tau log(1/kappa_*)) = 0.722223``, ``q_1 = 0.590750`` at
the fold.
  (a) *Result, closed-form check.* Before ``d_*`` the run tracks the upper root within 0.02.
      It collapses (``q_1 < 0.05``) at a distance in ``[d_*, d_* + 0.1]``; the lag is a
      slow-passage effect of the finite ``v``. Able to fail: the fold could sit elsewhere, or
      the boundary coordinate could lag enough to move it.
  (b) *Forced, a reproduction.* A second run, the drift child's forcing reversed and started at
      the collapsed state, is still at zero when the substrate is back at distance 0. Exposed, the
      zero state's growth coefficient ``3 kappa - 5`` is negative for every ``kappa``, and with
      no assembly an exact zero stays zero. Said so that it is not read as a finding.

(T4) *Two variants, one substrate.* Apparatus ``A_1`` with its site at the substrate
(``kappa_1 = 1``) and ``A_2`` at distance ``0.459044`` (``kappa_2 = 0.9``). They share the
boundary ``B``, whose production and protection are Codex's with ``A_1 + A_2`` in place of
``q_1``. Exposed, no assembly, both started at 1 with ``B = 20``. **The saturating loss is a
choice the predictions depend on**, so it is run both ways: *crowding*, ``A_i (A_1 + A_2)``, a
shared space; and *self-limiting*, ``A_i^2``.
  (a) *Crowding, held fuel: exclusion at a closed-form rate (a reproduction of a derivation).*
      Every term but the rebuild is the same per unit of ``A_i``, so
      ``d/dt log(A_2/A_1) = -k_r F (kappa_1 - kappa_2) = -0.3`` exactly in continuous time. The
      measured slope is within 2% of that, and ``A_2 < 1e-6`` at the end.
  (b) *Self-limiting, held fuel: coexistence, and each helps the other (a result).* Both end
      above 0.1, and ``A_1`` ends higher than it does alone, because the only coupling is the
      shared boundary. Able to fail: the partner's boundary may not move ``A_1`` at all.
  (c) *Self-limiting, fuel a species (a result).* The fuel is a species fed at rate ``J`` and
      washed out at rate 1; rebuild and boundary production each consume one unit of fuel per
      event. Predictions:
      - as ``J`` falls, ``A_2`` is lost first, at ``J_2 > J_1``;
      - at a ``J`` where both persist, ``A_1`` ends *lower* with ``A_2`` present than alone:
        the competition for fuel outweighs the shared boundary. This one is a guess.
      Able to fail: the order could reverse, or the partner could help.
"""

from dataclasses import dataclass, replace
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.reaction_network import separability_ratio
from arrangements.rvect import ReactiveVectorSpace, constant, euclidean
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

from tests.chain import flat_state

EMPTY = jnp.zeros(0)
H = 0.01            # the tick, T0--T2a
H_BIND = 1e-3       # the tick for T2b's stiffer binding
TAU = 1.0           # the kernel scale of the match (stage B moves placements)
L_OPEN = 4.0        # leakage with no pore; the pore publishes rho = 3.8, leaving 0.2
RHO_PORE = 3.8
SMALL = 1e-8


@dataclass(frozen=True)
class Conditions:
    fuel: float = 1.0
    site: float = 0.0       # the apparatus's placement
    substrate: float = 0.0  # the substrate's placement; kappa = exp(-(site - substrate)^2/2 tau)
    rho: float = 0.0        # what the pore child publishes; 0 is no pore
    assembly: float = 0.0
    rebuild: float = 3.0
    boundary: float = 10.0


EXPOSED = Conditions()
PORE = replace(EXPOSED, rho=RHO_PORE, assembly=1e-6)


def kappa(site, substrate):
    return jnp.exp(-(site - substrate) ** 2 / (2.0 * TAU))


def codex_field(q, c):
    """Codex's ``V`` (``notes/check-reader-origin.py``'s ``rates``), the ground truth."""
    q1, q2 = q[0], q[1]
    act = c.fuel * kappa(c.site, c.substrate) * q1
    leak = L_OPEN - c.rho
    return jnp.array([c.assembly * c.fuel + c.rebuild * act - (1.0 + leak / (1.0 + q2)) * q1
                      - q1 ** 2,
                      c.boundary * act - q2])


# ---------------------------------------------------------------------------
# A reaction network by composition: species and held supplies broadcast their values;
# each transition box reads some of them and returns a covector on what it reads; a
# routing wire connects them (the pattern of ``reaction_network.unimolecular_arrangement``).
# ---------------------------------------------------------------------------


def box(x0, degree, rv, label):
    """A species or a held supply: configuration coalgebra on ``rv``, value broadcast."""
    return Arrangement(
        P=configuration(rv, init=jnp.array([float(x0)])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=degree,
        out_f=lambda q, m_out: jnp.tile(q, degree),
        in_f=lambda q, m_out, n_in: EMPTY,
        omega=zero_form, label=label,
    )


HELD = constant(jnp.zeros((1, 1)))    # the drive idiom: zero response, holds its value


@dataclass(frozen=True)
class Transition:
    label: str
    reads: tuple
    covector: object = None   # direct: n -> returned covector on the reads
    cost: object = None       # structured: n -> scalar cost; the forcing is -d cost


def transition_box(t):
    k = len(t.reads)
    if t.cost is not None:
        omega = exact(lambda q, m_out, n_in: -t.cost(n_in))
    else:
        omega = lambda q, m_out, n_in: (EMPTY, EMPTY, t.covector(n_in))
    return Arrangement(
        P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=k, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY, in_f=lambda q, m_out, n_in: EMPTY,
        omega=omega, label=t.label,
    )


def network(boxes, transitions):
    """``compose_seq(tensor(boxes..., transitions...), wire)``, closed. ``boxes`` is a list
    of ``(name, x0, rv)``; the composite's configuration is their values in that order."""
    names = [b[0] for b in boxes]
    degree = {n: sum(r == n for t in transitions for r in t.reads) for n in names}
    offset, acc = {}, 0
    for n in names:
        offset[n], acc = acc, acc + degree[n]
    used, perm = {n: 0 for n in names}, []
    for t in transitions:
        for r in t.reads:
            perm.append(offset[r] + used[r])
            used[r] += 1
    perm = jnp.asarray(perm, dtype=int)
    ports = len(perm)
    wire = Arrangement(
        P=trivial_parameter(), in_dim_M=ports, out_dim_M=ports, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: EMPTY, in_f=lambda q, m_out, n_in: m_out[perm],
        omega=zero_form, label="wire",
    )
    parts = [box(x0, degree[n], rv, n) for n, x0, rv in boxes]
    parts += [transition_box(t) for t in transitions]
    return compose_seq(tensor_arrangements(parts), wire)


def settle(arr, steps):
    P, omega = arr.P, arr.omega

    def step(_i, s):
        return P.step(s, omega(P.position(s), EMPTY, EMPTY)[0])

    return np.asarray(flat_state(jax.jit(
        lambda s0: jax.lax.fori_loop(0, steps, step, s0))(P.init)))


def forcing(arr, q):
    return arr.omega(jnp.asarray(q, float), EMPTY, EMPTY)[0]


# ---------------------------------------------------------------------------
# The direct composite (T0, T1a, T2a).
# ---------------------------------------------------------------------------

SUPPLIES = ("F", "site", "S", "pore")


def direct_transitions(c):
    def kap(n_site, n_sub):
        return kappa(n_site, n_sub)
    return [
        Transition("assembly", ("A", "F"),
                   covector=lambda n: jnp.array([c.assembly * n[1], 0.0])),
        Transition("rebuild", ("A", "F", "site", "S"),
                   covector=lambda n: jnp.array(
                       [c.rebuild * kap(n[2], n[3]) * n[1] * n[0], 0.0, 0.0, 0.0])),
        Transition("boundary production", ("A", "B", "F", "site", "S"),
                   covector=lambda n: jnp.array(
                       [0.0, c.boundary * kap(n[3], n[4]) * n[2] * n[0], 0.0, 0.0, 0.0])),
        Transition("damage", ("A",), covector=lambda n: jnp.array([-n[0]])),
        Transition("saturating loss", ("A",), covector=lambda n: jnp.array([-n[0] ** 2])),
        Transition("boundary loss", ("B",), covector=lambda n: jnp.array([-n[0]])),
        Transition("protected leak", ("A", "B", "pore"),
                   covector=lambda n: jnp.array(
                       [-(L_OPEN - n[2]) * n[0] / (1.0 + n[1]), 0.0, 0.0])),
    ]


def supplies(c):
    return [("F", c.fuel, HELD), ("site", c.site, HELD), ("S", c.substrate, HELD),
            ("pore", c.rho, HELD)]


def direct(q, c, only=None, h=H):
    ts = direct_transitions(c)
    if only is not None:
        ts = [t for t in ts if t.label == only]
    boxes = [("A", q[0], euclidean(1, h)), ("B", q[1], euclidean(1, h))] + supplies(c)
    return network(boxes, ts)


def run(q, c, duration, h=H):
    return settle(direct(q, c, h=h), int(round(duration / h)))[:2]


def exact_equilibria(leak):
    """Codex's §4: with ``q_2 = 10 q_1``, ``10 q_1^2 - 19 q_1 + (leak - 2) = 0``."""
    disc = math.sqrt(441.0 - 40.0 * leak)
    return [np.array([x, 10.0 * x]) for x in ((19.0 - disc) / 20.0, (19.0 + disc) / 20.0)]


def test_T0_the_composite_ticks_as_codex_field_and_recovers_his_table():
    rng = np.random.default_rng(0)
    for c in (EXPOSED, PORE, replace(EXPOSED, fuel=0.7, rebuild=2.0)):
        for _ in range(5):
            q = rng.uniform(0.0, 5.0, 2)
            tick = settle(direct(q, c), 1)[:2]
            assert np.allclose(tick, q + H * np.asarray(codex_field(jnp.asarray(q), c)),
                               atol=1e-14, rtol=0.0)

    established = run([0.0, 0.0], PORE, 30.0)
    pore_high = exact_equilibria(0.2)[1]
    assert np.allclose(established, pore_high, atol=1e-6), established
    assert abs(established[0] - 1.990433) < 1e-6

    assert run([0.0, 0.0], replace(EXPOSED, assembly=1e-6), 30.0)[0] < 1e-6

    retained = run(established, EXPOSED, 60.0)
    assert np.allclose(retained, exact_equilibria(4.0)[1], atol=1e-8)

    controls = {
        "no fuel": replace(EXPOSED, fuel=0.0),
        "no match": replace(EXPOSED, substrate=50.0),
        "no rebuild": replace(EXPOSED, rebuild=0.0),
        "no boundary production": replace(EXPOSED, boundary=0.0),
    }
    for name, c in controls.items():
        assert run(established, c, 60.0)[0] < SMALL, name
    assert run([0.0, 0.0], replace(PORE, assembly=0.0), 60.0)[0] == 0.0


def curl(arr, q, idx=(0, 1)):
    J = jax.jacobian(lambda x: forcing(arr, x))(jnp.asarray(q, float))
    i, j = idx
    return float(J[j, i] - J[i, j])     # (d omega)_{12} = d_1 omega_2 - d_2 omega_1


def test_T1a_the_curl_is_codex_and_lives_in_two_children():
    rng = np.random.default_rng(1)
    labels = [t.label for t in direct_transitions(EXPOSED)]
    for c in (EXPOSED, PORE):
        leak = L_OPEN - c.rho
        for _ in range(5):
            q = np.concatenate([rng.uniform(0.1, 5.0, 2), [c.fuel, c.site, c.substrate, c.rho]])
            total = curl(direct(q[:2], c), q)
            codex = c.boundary * c.fuel - leak * q[0] / (1.0 + q[1]) ** 2
            assert abs(total - codex) < 1e-12
            parts = {lab: curl(direct(q[:2], c, only=lab), q) for lab in labels}
            nonzero = {lab for lab, v in parts.items() if abs(v) > 1e-14}
            assert nonzero == {"boundary production", "protected leak"}
            assert abs(sum(parts.values()) - total) < 1e-12


def test_T2a_release_is_substituting_the_pore_child():
    """The exposed composite differs from the pore composite in one inner box: the pore
    child publishing ``rho = 3.8`` is replaced by one publishing ``0``. Every other child and
    the wire are the same objects' construction, so its run from the pore composite's end
    state is T0's release: two arrangements compared, no event in either."""
    established = run([0.0, 0.0], PORE, 30.0)
    with_pore = direct(established, replace(PORE, assembly=0.0))
    without = direct(established, EXPOSED)
    kept = settle(with_pore, 6000)[:2]
    released = settle(without, 6000)[:2]
    assert np.allclose(kept, exact_equilibria(0.2)[1], atol=1e-8)
    assert np.allclose(released, exact_equilibria(4.0)[1], atol=1e-8)
    # The one differing coordinate is the pore child's published value.
    assert settle(with_pore, 0)[5] == RHO_PORE and settle(without, 0)[5] == 0.0


# ---------------------------------------------------------------------------
# The structured composite (T1b).
# ---------------------------------------------------------------------------


def potential_terms(c):
    """The exact children of T1b: each a cost on what it reads; the forcing is ``-d cost``."""
    return [
        Transition("assembly", ("A", "F"), cost=lambda n: -c.assembly * n[1] * jnp.log(n[0])),
        Transition("rebuild", ("A", "F", "site", "S"),
                   cost=lambda n: -c.rebuild * kappa(n[2], n[3]) * n[1] * n[0]),
        Transition("damage", ("A",), cost=lambda n: n[0]),
        Transition("saturating loss", ("A",), cost=lambda n: 0.5 * n[0] ** 2),
        Transition("protection", ("A", "B", "pore"),
                   cost=lambda n: (L_OPEN - n[2]) * n[0] / (1.0 + n[1])),
        Transition("boundary loss", ("B", "F", "site", "S", "pore"),
                   cost=lambda n: (L_OPEN - n[4]) / (c.boundary * kappa(n[2], n[3]) * n[1])
                   * (jnp.log(1.0 + n[0]) + 1.0 / (1.0 + n[0]))),
    ]


def U(q, c):
    return sum(t.cost(jnp.array([{"A": q[0], "B": q[1], "F": c.fuel, "site": c.site,
                                  "S": c.substrate, "pore": c.rho}[r] for r in t.reads]))
               for t in potential_terms(c))


def structured(q, c, h=H):
    """Species boxes whose responses are T1b's ``G``: ``h q_1`` for the apparatus and
    ``h (k_b kappa F / l)(1 + q_2)^2`` for the boundary. The boundary's response carries the
    fuel, the match and the leakage as constants of its own matter."""
    leak = L_OPEN - c.rho
    g2 = c.boundary * float(kappa(c.site, c.substrate)) * c.fuel / leak
    A = ReactiveVectorSpace(1, lambda x: jnp.array([[h * x[0]]]))
    B = ReactiveVectorSpace(1, lambda x: jnp.array([[h * g2 * (1.0 + x[0]) ** 2]]))
    return network([("A", q[0], A), ("B", q[1], B)] + supplies(c), potential_terms(c))


def test_T1b_codex_field_is_a_descent_under_a_diagonal_response():
    rng = np.random.default_rng(2)
    for c in (EXPOSED, PORE, replace(EXPOSED, fuel=0.7, rebuild=2.0, substrate=0.4)):
        for _ in range(5):
            q = rng.uniform(0.05, 5.0, 2)
            tick = settle(structured(q, c), 1)[:2]
            direct_tick = settle(direct(q, c), 1)[:2]
            assert np.allclose(tick, direct_tick, atol=1e-13, rtol=0.0), (tick, direct_tick)
            # The response ratio is forced: g_2/g_1 = J_21/J_12 of V (item 3's condition).
            J = jax.jacobian(lambda x: codex_field(x, c))(jnp.asarray(q))
            g1 = q[0]
            g2 = c.boundary * float(kappa(c.site, c.substrate)) * c.fuel \
                / (L_OPEN - c.rho) * (1 + q[1]) ** 2
            assert abs(float(J[1, 0] / J[0, 1]) - g2 / g1) < 1e-9 * g2 / g1
    # The protection child alone carries both of T1a's curl sources, now as one exact cost.
    q = np.array([0.7, 3.0])
    only = network([("A", q[0], euclidean(1, 1.0)), ("B", q[1], euclidean(1, 1.0))]
                   + supplies(EXPOSED), [potential_terms(EXPOSED)[4]])
    x = np.concatenate([q, [1.0, 0.0, 0.0, 0.0]])
    assert abs(curl(only, x)) < 1e-14
    # U decreases at every tick of the release run.
    established = run([0.0, 0.0], PORE, 30.0)
    arr = structured(established, EXPOSED)
    P, omega = arr.P, arr.omega
    step = jax.jit(lambda s: P.step(s, omega(P.position(s), EMPTY, EMPTY)[0]))
    s, last = P.init, float(U(established, EXPOSED))
    drops = []
    for _ in range(6000):
        s = step(s)
        now = float(U(np.asarray(flat_state(s))[:2], EXPOSED))
        drops.append(last - now)
        last = now
    assert min(drops) >= -1e-13, min(drops)      # rounding once converged, |U| ~ 10
    assert np.allclose(np.asarray(flat_state(s))[:2], exact_equilibria(4.0)[1], atol=1e-8)


# ---------------------------------------------------------------------------
# T2b: protection by mass-action binding.
# ---------------------------------------------------------------------------


def binding_transitions(c, k_on, K):
    k_off = k_on / K

    def kap(n_site, n_sub):
        return kappa(n_site, n_sub)

    return [
        Transition("assembly", ("A", "F"),
                   covector=lambda n: jnp.array([c.assembly * n[1], 0.0])),
        Transition("rebuild", ("A", "C", "F", "site", "S"),
                   covector=lambda n: jnp.array(
                       [c.rebuild * kap(n[3], n[4]) * n[2] * (n[0] + n[1]), 0.0, 0.0, 0.0, 0.0])),
        Transition("boundary production", ("A", "B", "C", "F", "site", "S"),
                   covector=lambda n: jnp.array(
                       [0.0, c.boundary * kap(n[4], n[5]) * n[3] * (n[0] + n[2]),
                        0.0, 0.0, 0.0, 0.0])),
        Transition("damage", ("A", "B", "C"),
                   covector=lambda n: jnp.array([-n[0], n[2], -n[2]])),
        Transition("saturating loss", ("A", "B", "C"),
                   covector=lambda n: (n[0] + n[2]) * jnp.array([-n[0], n[2], -n[2]])),
        Transition("boundary loss", ("A", "B", "C"),
                   covector=lambda n: jnp.array([n[2], -n[1], -n[2]])),
        Transition("leak", ("A", "pore"),
                   covector=lambda n: jnp.array([-(L_OPEN - n[1]) * n[0], 0.0])),
        Transition("binding", ("A", "B", "C"),
                   covector=lambda n: (k_on * n[0] * n[1] - k_off * n[2])
                   * jnp.array([-1.0, -1.0, 1.0])),
    ]


def binding(x, c, k_on, K, h=H_BIND):
    boxes = [("A", x[0], euclidean(1, h)), ("B", x[1], euclidean(1, h)),
             ("C", x[2], euclidean(1, h))] + supplies(c)
    return network(boxes, binding_transitions(c, k_on, K))


def binding_run(x, c, k_on, K, duration, h=H_BIND):
    return settle(binding(x, c, k_on, K, h), int(round(duration / h)))[:3]


def totals(x):
    return np.array([x[0] + x[2], x[1] + x[2]])


def released(k_on, K):
    """Run the pore composite from nothing, then the exposed composite (the pore child
    substituted by hand) from that end state; return the totals ``(q_1, q_2)`` at the end of
    each run."""
    est = binding_run([0.0, 0.0, 0.0], PORE, k_on, K, 40.0)
    rel = binding_run(est, EXPOSED, k_on, K, 60.0)
    return totals(est), totals(rel)


def test_T2b_the_totals_obey_codex_equations_with_the_true_free_amount():
    """Construction check: in ``(q_1, q_2)`` the binding composite's tick is Codex's with the
    leak ``l A`` in place of ``l q_1 / (1 + q_2)``."""
    rng = np.random.default_rng(3)
    for c in (EXPOSED, PORE):
        for _ in range(5):
            x = rng.uniform(0.0, 3.0, 3)
            q = totals(x)
            tick = totals(settle(binding(x, c, 20.0, 1.0), 1)[:3])
            leak = L_OPEN - c.rho
            v = np.array(codex_field(jnp.asarray(q), c))
            v[0] += leak * q[0] / (1.0 + q[1]) - leak * x[0]
            assert np.allclose(tick, q + H_BIND * v, atol=1e-13, rtol=0.0)


BINDING_REPORT = {}


def test_T2b_i_fast_binding_establishes_and_persists():
    est, rel = released(20.0, 1.0)
    BINDING_REPORT["fast"] = (est, rel)
    print("fast binding: established", est, "released", rel)
    assert rel[0] > 0.5
    assert abs(rel[0] - 1.788153) < 0.1 * 1.788153, rel


def test_T2b_ii_a_binding_threshold_below_one():
    ok = lambda K: released(20.0, K)[1][0] > 0.5
    assert ok(1.0)
    lo, hi = 1e-3, 1.0
    assert not ok(lo)
    for _ in range(12):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if ok(mid) else (mid, hi)
    print("K_* in", (lo, hi))
    BINDING_REPORT["K_star"] = (lo, hi)
    assert hi < 1.0


def test_T2b_iii_slow_binding_fails_on_release():
    est, rel = released(0.1, 1.0)
    print("slow binding: established", est, "released", rel)
    assert est[0] > 0.5            # the pore still establishes it
    assert rel[0] < SMALL, rel


def test_T2b_iv_the_binding_chemistry_has_no_species_box_potential():
    c = PORE

    def rhs(x):
        n = {"A": x[0], "B": x[1], "C": x[2], "F": c.fuel, "site": c.site,
             "S": c.substrate, "pore": c.rho}
        v = jnp.zeros(3)
        idx = {"A": 0, "B": 1, "C": 2}
        for t in binding_transitions(c, 20.0, 1.0):
            cov = t.covector(jnp.array([n[r] for r in t.reads]))
            for r, val in zip(t.reads, cov):
                if r in idx:
                    v = v.at[idx[r]].add(val)
        return v

    ratio = separability_ratio(rhs, 0, 2)          # (dV_C/dA)/(dV_A/dC)
    r1 = float(ratio(jnp.array([0.3, 1.0, 0.5])))
    r2 = float(ratio(jnp.array([0.3, 4.0, 0.5])))
    print("(A, C) ratio at B = 1, 4:", r1, r2)
    assert abs(r1 - r2) > 1e-3 * max(abs(r1), abs(r2))


# ---------------------------------------------------------------------------
# Stage B. T3: the substrate drifts away and back.
# ---------------------------------------------------------------------------

V_DRIFT = 0.005
KAPPA_STAR, D_STAR, Q_FOLD = 0.770433, 0.722223, 0.590750


def fold():
    """The closed form of T3: the discriminant of the exposed equilibrium quadratic in
    ``q_1`` vanishes at ``kappa_*``."""
    from scipy.optimize import brentq

    def disc(k):
        r = 3.0 * k - 1.0
        return (10.0 * k * r - 1.0) ** 2 - 40.0 * k * (4.0 - r)

    k = brentq(disc, 0.5, 1.0)
    r = 3.0 * k - 1.0
    return k, math.sqrt(2.0 * TAU * math.log(1.0 / k)), (10.0 * k * r - 1.0) / (20.0 * k)


def upper_root(k):
    r = 3.0 * k - 1.0
    b = 10.0 * k * r - 1.0
    disc = b ** 2 - 40.0 * k * (4.0 - r)
    return (b + math.sqrt(disc)) / (20.0 * k) if disc >= 0 else float("nan")


def drifting(q, c, direction, h=H):
    """T0's direct composite with the substrate box given the response ``h v`` and a
    constant forcing ``direction`` from one more child: it drifts at speed ``v``."""
    ts = direct_transitions(c) + [
        Transition("drift", ("S",), covector=lambda n: jnp.array([direction]))]
    boxes = [("A", q[0], euclidean(1, h)), ("B", q[1], euclidean(1, h)),
             ("F", c.fuel, HELD), ("site", c.site, HELD),
             ("S", c.substrate, constant(jnp.array([[h * V_DRIFT]]))),
             ("pore", c.rho, HELD)]
    return network(boxes, ts)


def trajectory(arr, steps, every):
    P, omega = arr.P, arr.omega

    def tick(s, _):
        return P.step(s, omega(P.position(s), EMPTY, EMPTY)[0]), None

    def chunk(s, _):
        s, _ = jax.lax.scan(tick, s, None, length=every)
        return s, flat_state(s)

    _, out = jax.jit(lambda s0: jax.lax.scan(chunk, s0, None, length=steps // every))(P.init)
    return np.asarray(out)


def drift_away():
    return trajectory(drifting(exact_equilibria(4.0)[1], EXPOSED, +1.0), 24000, 10)  # to d = 1.2


def tracking_error(out):
    q1, dist = out[:, 0], out[:, 4]
    before = np.where(dist < D_STAR - 0.05)[0]
    return np.array([q1[i] - upper_root(float(kappa(0.0, dist[i]))) for i in before]), dist[before]





@pytest.mark.xfail(strict=True, reason="refuted 2026-09-26: the lag reaches 0.042 at d_* - 0.05")
def test_T3a_tracking_within_the_predicted_bound():
    """Prediction (a)'s first half, kept on the record: refuted. The run lags *above* the
    upper root by an amount that grows toward the fold (0.0014 at d = 0.1, 0.012 at 0.5,
    0.021 at 0.6, 0.042 at 0.672): the relaxation rate vanishes at a fold, so a fixed drift
    speed outruns it. The bound 0.02 ignored that. Not tuned."""
    err, _ = tracking_error(drift_away())
    assert np.max(np.abs(err)) < 0.02


def test_T3_the_match_as_a_distance_folds_where_the_closed_form_says():
    k, d, qf = fold()
    assert abs(k - KAPPA_STAR) < 1e-6 and abs(d - D_STAR) < 1e-6 and abs(qf - Q_FOLD) < 1e-6
    out = drift_away()
    q1, dist = out[:, 0], out[:, 4]
    err, where = tracking_error(out)
    # What was observed in place of the refuted bound: a lag, positive, growing toward the fold.
    assert np.all(err > -1e-6) and err[-1] == np.max(err) and err[-1] < 0.05
    collapse = float(dist[np.argmax(q1 < 0.05)])
    print("T3: collapse at d =", collapse, "against d_* =", D_STAR, "; last lag", err[-1])
    assert D_STAR <= collapse <= D_STAR + 0.1

    back = trajectory(drifting(out[-1][:2], replace(EXPOSED, substrate=float(dist[-1])), -1.0),
                      24000, 10)
    print("T3: after drifting back to d =", back[-1, 4], "q_1 =", back[-1, 0])
    assert abs(back[-1, 4]) < 1e-9 and back[-1, 0] < SMALL


# ---------------------------------------------------------------------------
# T4: two variants on one substrate.
# ---------------------------------------------------------------------------

D2 = math.sqrt(2.0 * TAU * math.log(1.0 / 0.9))     # kappa_2 = 0.9


def variants_transitions(c, crowding, feed=None):
    """Two apparatus variants sharing the boundary. With ``feed = J`` the fuel is a species
    fed at ``J`` and washed out at rate 1, and every rebuild and boundary-production event
    consumes one unit of it; otherwise the fuel is held and the consumption is returned to a
    box that does not respond."""
    use = 0.0 if feed is None else 1.0

    def rebuild(i):
        return Transition(f"rebuild {i}", (f"A{i}", "F", f"site{i}", "S"),
                          covector=lambda n: c.rebuild * kappa(n[2], n[3]) * n[1] * n[0]
                          * jnp.array([1.0, -use, 0.0, 0.0]))

    def loss(i):
        if crowding:
            return Transition(f"loss {i}", ("A1", "A2"),
                              covector=lambda n: -(n[0] + n[1]) * n[i - 1]
                              * jnp.eye(2)[i - 1])
        return Transition(f"loss {i}", (f"A{i}",), covector=lambda n: jnp.array([-n[0] ** 2]))

    def leak(i):
        return Transition(f"leak {i}", (f"A{i}", "B", "pore"),
                          covector=lambda n: jnp.array(
                              [-(L_OPEN - n[2]) * n[0] / (1.0 + n[1]), 0.0, 0.0]))

    def production(n):
        rate = c.boundary * n[3] * (kappa(n[4], n[6]) * n[0] + kappa(n[5], n[6]) * n[1])
        return jnp.array([0.0, 0.0, rate, -use * rate, 0.0, 0.0, 0.0])

    ts = [rebuild(1), rebuild(2), loss(1), loss(2), leak(1), leak(2),
          Transition("damage", ("A1", "A2"), covector=lambda n: -n),
          Transition("boundary production", ("A1", "A2", "B", "F", "site1", "site2", "S"),
                     covector=production),
          Transition("boundary loss", ("B",), covector=lambda n: -n)]
    if feed is not None:
        ts.append(Transition("feed", ("F",), covector=lambda n: jnp.array([feed - n[0]])))
    return ts


def variants(x, c, crowding, feed=None, h=H):
    """Configuration ``(A_1, A_2, B, F, ...)``; ``A_2 = 0`` exactly is the variant absent."""
    fuel_rv = HELD if feed is None else euclidean(1, h)
    boxes = [("A1", x[0], euclidean(1, h)), ("A2", x[1], euclidean(1, h)),
             ("B", x[2], euclidean(1, h)), ("F", x[3], fuel_rv),
             ("site1", 0.0, HELD), ("site2", D2, HELD), ("S", 0.0, HELD), ("pore", c.rho, HELD)]
    return network(boxes, variants_transitions(c, crowding, feed))


START = [1.0, 1.0, 20.0, 1.0]
ALONE = [1.0, 0.0, 20.0, 1.0]


def test_T4a_crowding_excludes_the_worse_match_at_the_closed_form_rate():
    out = trajectory(variants(START, EXPOSED, crowding=True), 20000, 100)   # 200 time units
    t = np.arange(1, out.shape[0] + 1) * 100 * H
    late = t >= 100.0
    slope = np.polyfit(t[late], np.log(out[late, 1] / out[late, 0]), 1)[0]
    print("T4a: slope of log(A2/A1)", slope, "against", -3.0 * (1.0 - 0.9))
    assert abs(slope - (-0.3)) < 0.02 * 0.3
    assert out[-1, 1] < 1e-6


def ends(x, crowding, feed=None, duration=300.0):
    return settle(variants(x, EXPOSED, crowding, feed), int(round(duration / H)))


def test_T4b_self_limiting_on_held_fuel_coexists_and_each_helps():
    both, alone = ends(START, False), ends(ALONE, False)
    print("T4b: both", both[:3], "; A1 alone", alone[:3])
    assert both[0] > 0.1 and both[1] > 0.1
    assert both[0] > alone[0] + 1e-6


def survives(J, i, x=START):
    return ends(x, False, feed=J)[i] > 1e-3


def threshold(i, lo, hi, x=START):
    assert survives(hi, i, x) and not survives(lo, i, x)
    for _ in range(14):
        mid = math.sqrt(lo * hi)
        lo, hi = (lo, mid) if survives(mid, i, x) else (mid, hi)
    return hi


def test_T4c_on_a_fed_fuel_the_worse_match_is_lost_first():
    J1, J2 = threshold(0, 1.0, 400.0), threshold(1, 1.0, 400.0)
    J1_alone = threshold(0, 1.0, 400.0, ALONE)
    print("T4c: J_1 =", J1, " J_2 =", J2, " J_1 alone =", J1_alone)
    assert J2 > J1
    J = 2.0 * J2
    both, alone = ends(START, False, feed=J), ends(ALONE, False, feed=J)
    print(f"T4c at J = {J:.3f}: both", both[:4], "; A1 alone", alone[:4])
    assert both[1] > 1e-3
    assert both[0] < alone[0]
