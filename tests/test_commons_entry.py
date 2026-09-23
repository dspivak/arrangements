"""Entry into a commons by placement: §12b's O2 and O1, tested rather than assumed.

The design is ``articulating-forms-that-matter/notes/COMMONS-DESIGN.md`` §12b, which lists
two things the note assumes and does not establish and asks P8.2 to settle **first**:

* **O2.** What message does the attention kernel weigh in a commons, what do its signal and
  weight carry, and how does the catch ride on the weighted input?
* **O1.** Arrival by placement works. What makes entry *stop*? Add a cost of proximity so
  placement settles where marginal rent equals the marginal cost of being close, and test
  whether that settling point is the zero-rent condition.

The interface is written in ``arrangements/commons_entry.py``; what the tests below add is
the part that could have come out otherwise. **Every docstring says which of three things
its test does** -- reproduce a closed form, check a discrete-time analogue, or illustrate.
Four of them record outcomes the design note did *not* predict, and say so.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.attention import (
    input_dim, make_input, mean, message, msg_dim, spatial_kappa, zero_message,
)
from arrangements.commons import Params, composite_port_form, open_access
from arrangements.commons_entry import (
    DH, DP, EDIM, EXTERIOR, MDIM, effective_N, entry_count, entry_system, kappas,
    marginal_threshold, net_benefit, run, settle, unpack, zero_rent_threshold,
)

TAU = 2.0
PAR = Params(mobility=0.1)          # mobility is the step; see commons.py
PM = 0.02                           # placement mobility: slow, so the stock is quasi-static
STEPS = 300_000


def settled(placements, outside, power=1.0, R0=90.0, steps=STEPS):
    N = len(placements)
    arr = entry_system(PAR, placements, outside, tau=TAU, R0=R0, x0=0.9,
                       place_mobility=PM, power=power)
    q = settle(arr, steps)
    R, x, _ = unpack(q, N)
    return float(R), np.asarray(x), np.asarray(kappas(q, N, TAU)), q


# ---------------------------------------------------------------------------
# O2: the interface. What the kernel weighs, and how the catch rides on it.
# ---------------------------------------------------------------------------


def test_O2_the_received_input_is_the_stock_weighted_by_proximity():
    """O2, REPRODUCING the attention paper's own ``rho_a`` (eq:maps) at a configuration.

    The resource is the sole sender and emits ``m_Res = (R, 1)``: *signal* the stock on
    offer, *weight* the resource counted once. Extractors emit ``0_M`` and are pure
    receivers. Then extractor ``i``'s received input is exactly
    ``((kappa_i R, kappa_i), l_i)`` -- the stock scaled by proximity in both coordinates,
    with the extractors contributing nothing to one another. Nothing is asserted about the
    kernel here that the paper does not already supply."""
    from arrangements.attention import spatial_arrangement, spatial_config, zero_cost
    N, R = 3, 37.0
    n = N + 1
    ells = [0.0, 0.3, 1.1, 2.5]
    a = spatial_arrangement(n, DH, DP, TAU, cost=zero_cost)
    q = spatial_config(ells, [1.0] * n)
    m_out = jnp.concatenate([message(R, 1.0)] + [zero_message(DH)] * N)
    for i in range(1, N + 1):
        e_i = a.received(q, m_out, EXTERIOR, i + 1)          # child i+1 is extractor i
        kappa = float(spatial_kappa(jnp.array([ells[i]]), jnp.array([ells[0]]), TAU))
        np.testing.assert_allclose(float(e_i[0]), kappa * R, rtol=1e-12)   # signal
        np.testing.assert_allclose(float(e_i[1]), kappa, rtol=1e-12)       # weight
        np.testing.assert_allclose(float(e_i[MDIM]), ells[i], rtol=1e-12)  # position


def test_O2_the_mean_readout_is_proximity_blind_so_the_catch_cannot_ride_on_it():
    """O2's content, and it is FORCED, not chosen. The mean readout of the received
    message is ``mean(kappa_i R, kappa_i) = R`` for *every* ``kappa_i > 0``: what an
    extractor believes the stock to be does not depend on how near it is. So the catch
    must ride on the **weighted signal** ``h = kappa_i R`` and not on the mean, or
    proximity would change nothing at all. That is also what keeps the cost quadratic
    (``attention.py``: the mean readout would leave the quadratic family)."""
    R = 37.0
    for kappa in (1.0, 0.5, 1e-3, 1e-8):
        m = message(kappa * R, kappa)
        np.testing.assert_allclose(float(mean(m)[0]), R, rtol=1e-9)
        np.testing.assert_allclose(float(m[0]), kappa * R, rtol=1e-12)     # the signal does move


def test_O2_at_full_proximity_the_composite_is_exactly_the_fixed_cost_of_section_2():
    """O2, REPRODUCING §2's ``U`` -- the check that the placement model is a *deformation*
    of the fixed cost and not a second model (§10's F2). With every ``kappa_i = 1`` and no
    proximity cost, the entry system's forcing on ``(R, x_1, ..., x_N)`` equals the plain
    open-access composite's forcing, coefficient for coefficient, and it settles at
    ``R* = r gamma K/(r gamma + NpK)``."""
    N = 3
    arr = entry_system(PAR, [0.0] * N, [0.0] * N, tau=TAU, R0=90.0, x0=0.9)
    plain = open_access(PAR, N)
    F_plain = composite_port_form(plain)
    for R, xs in ((37.0, [0.11, 0.23, 0.07]), (5.0, [0.9, 0.0, 0.4])):
        q_plain = jnp.asarray([R] + xs)
        q_entry = jnp.asarray([R] + xs + [0.0] * (N + 1) + [1.0] * (N + 1))
        xi, _, _ = arr.omega(q_entry, jnp.zeros(0), EXTERIOR)
        np.testing.assert_allclose(np.asarray(xi)[:1 + N], np.asarray(F_plain(q_plain)),
                                   atol=1e-12)
    R, x, k, _ = settled([0.0] * N, [0.0] * N, steps=100_000)
    np.testing.assert_allclose(k, 1.0, atol=1e-12)
    np.testing.assert_allclose(R, PAR.R_open_access(N), rtol=1e-6)


# ---------------------------------------------------------------------------
# O1: what makes entry stop.
# ---------------------------------------------------------------------------


def test_O1_placement_settles_where_marginal_rent_equals_the_marginal_cost():
    """O1's first half, CONFIRMED and CHECKING THE DISCRETE ANALOGUE. With a cost of
    proximity ``a kappa`` the placements do settle, at an *interior* point, and the settled
    condition is the marginal one, ``x_i R = a``, to nine digits. Four identical
    participants starting together settle at ``kappa = 0.900545``, the root of
    ``20 kappa = (1 + 4 kappa^2)^2`` that ``x_i R = a`` and ``R = K/(1 + 4 kappa^2)`` give;
    the effective ``N`` is ``sum kappa_i^2 = 3.244``, strictly between 3 and 4."""
    N, a = 4, 5.0
    R, x, k, q = settled([0.5] * N, [a] * N)
    np.testing.assert_allclose(x * R, a, rtol=1e-8)                 # the marginal condition
    np.testing.assert_allclose(k, 0.900545, rtol=1e-5)
    np.testing.assert_allclose(R, 100.0 / (1.0 + 4.0 * k[0] ** 2), rtol=1e-6)
    np.testing.assert_allclose(effective_N(q, N, TAU), float(np.sum(k ** 2)), rtol=1e-12)
    assert 3.0 < effective_N(q, N, TAU) < 4.0
    # it is interior: neither at the resource nor unwired
    assert np.all(k > 0.1) and np.all(k < 0.99)


def test_O1_the_stabilizer_is_the_stock_not_the_cost():
    """O1, ILLUSTRATING *why* the previous test settles, which the note does not anticipate.
    At a **fixed** stock the net benefit ``V(kappa) = p kappa^2 R^2/(2 gamma) - a kappa`` is
    *convex*, so its interior stationary point is a minimum and the placement flow runs away
    from it: the corners, not the interior, would be the attractors. What makes the interior
    point attracting is that the resource responds -- ``R`` falls as ``kappa`` rises fast
    enough to reverse the slope. The stabilizer of entry is the stock itself."""
    a, R_fixed = 5.0, 23.563093
    k0 = 0.900545
    # at fixed R, V is convex and k0 is a minimum: V(k0) < V(k0 +- h) for both signs
    V = lambda kk: net_benefit(PAR, R_fixed, kk, a)
    assert V(k0) < V(k0 + 0.05) and V(k0) < V(k0 - 0.05)
    # with R responding, the same point is a maximum of the *flow*'s driving quantity
    drive = lambda kk: PAR.p * kk * (100.0 / (1.0 + 4.0 * kk ** 2)) ** 2 / PAR.gamma - a
    assert drive(k0 - 0.05) > 0 > drive(k0 + 0.05)      # pushes in below, out above


def test_O1_the_settling_point_is_NOT_the_zero_rent_condition():
    """O1's second half, and the answer is **no** -- the note treated this as a prediction
    and it does not hold. REPRODUCES the closed form the runs land on: at the settled
    placement the rent collected is exactly ``power/2`` times the proximity cost paid, so
    for the default linear cost it is **half** of it, and each participant's net benefit is
    ``V = -a kappa/2 < 0`` -- strictly worse than never having come. Gordon's zero-rent
    condition would be ``V = 0``. The two coincide only at ``power = 2``, where both are
    degenerate. Checked at ``power = 1`` and ``power = 3``, where the ratio is ``3/2``."""
    N, a = 4, 5.0
    for power, ratio in ((1.0, 0.5), (3.0, 1.5)):
        R, x, k, _ = settled([0.5] * N, [a] * N, power=power)
        # the marginal condition: x_i R = power * a * kappa^{power-1}
        np.testing.assert_allclose(x * R, power * a * k ** (power - 1.0), rtol=1e-6)
        rent_over_p = PAR.p * k ** 2 * R ** 2 / (2.0 * PAR.gamma)
        cost = a * k ** power
        np.testing.assert_allclose(rent_over_p / cost, ratio, rtol=1e-5)
        V = net_benefit(PAR, R, k[0], a, power)
        np.testing.assert_allclose(V, (ratio - 1.0) * cost[0], rtol=1e-5)
        assert (V < 0) == (power < 2.0)
        # and the two thresholds differ by the same factor at the settled configuration
        np.testing.assert_allclose(
            marginal_threshold(PAR, R, k[0], power) / zero_rent_threshold(PAR, R, k[0], power),
            2.0 / power, rtol=1e-9)


def test_O1_heterogeneous_outside_options_give_different_placements_and_a_sorted_split():
    """O1's failure criterion, NOT met: the note says it "fails visibly if every participant
    ends at the same placement". With identical participants they do -- symmetry forces it
    -- but with different outside options ``a_i`` they end at genuinely different placements,
    sorted by ``a_i``: the two cheapest outside options go all the way in and the two dearest
    go all the way out, and the stock settles at ``R*(2) = 33.33``, the open-access value for
    the number that entered. Effective ``N`` is 2 inside a syntax with ``N = 4``."""
    a = [1.0, 4.0, 8.0, 20.0]
    R, x, k, q = settled([0.5] * 4, a)
    assert k[0] > 0.99 and k[1] > 0.99 and k[2] < 1e-4 and k[3] < 1e-4
    assert k[0] >= k[1] > k[2] >= k[3]                    # sorted by outside option
    np.testing.assert_allclose(R, PAR.R_open_access(2), rtol=1e-6)
    np.testing.assert_allclose(effective_N(q, 4, TAU), 2.0, rtol=1e-6)
    # the count the placement dynamics produces is the *marginal* count, not Gordon's
    assert entry_count(PAR, 8.0, 4, marginal_threshold) == 2
    assert entry_count(PAR, 8.0, 4, zero_rent_threshold) == 1


def test_O1_the_count_that_enters_is_the_marginal_count_not_gordons():
    """O1, the discrepancy made numerical. With four identical participants at ``a = 5`` and
    one of them started far out, three enter fully and the fourth stays out, at
    ``R = R*(3) = 25`` -- which is the count ``marginal_threshold`` predicts. Gordon's
    zero-rent count is 2. The three that entered are each at ``V = -1.875 < 0``: the gradient
    equilibrium admits one participant more than free entry with full information would, and
    every insider is worse off than it would be outside."""
    a = 5.0
    R, x, k, q = settled([0.5, 0.5, 0.5, 6.0], [a] * 4)
    assert np.sum(k > 0.99) == 3 and k[3] < 1e-4
    np.testing.assert_allclose(R, PAR.R_open_access(3), rtol=1e-6)
    np.testing.assert_allclose(x[:3] * R, 6.25, rtol=1e-6)         # > a = 5, so they stay
    assert entry_count(PAR, a, 4, marginal_threshold) == 3
    assert entry_count(PAR, a, 4, zero_rent_threshold) == 2
    for i in range(3):
        assert net_benefit(PAR, R, k[i], a) == pytest.approx(-1.875, abs=1e-4)


def test_O1_the_stopping_point_is_not_unique():
    """O1, ILLUSTRATING a limit the note does not mention: the same parameters admit several
    stable stopping points, selected by where the participants started. Four identical
    participants at ``a = 5`` settle at four interior placements (``N_eff = 3.24``) when they
    start together, and at three-in-one-out (``N_eff = 3``) when one starts far out. Gordon's
    ``n*`` is a number; this model's is a set."""
    a = 5.0
    _, _, k1, q1 = settled([0.5] * 4, [a] * 4)
    _, _, k2, q2 = settled([0.5, 0.5, 0.5, 6.0], [a] * 4)
    assert not np.allclose(np.sort(k1), np.sort(k2), atol=1e-3)
    assert effective_N(q1, 4, TAU) > effective_N(q2, 4, TAU)


def test_O1_a_participant_placed_far_out_never_arrives():
    """O1, ILLUSTRATING the sharpest limit the runs found, and it bears directly on the
    corrected S1's "positive rent is exactly what makes proximity attractive". The kernel's
    gradient is ``kappa'(l) = -kappa l/tau``, which vanishes as ``kappa -> 0``; with any
    positive cost of proximity the placement force is ``-mu(a - x_i R) kappa'(l)``, so far
    out -- where ``x_i R -> 0`` -- it points **away**. ``kappa = 0`` is a locally attracting
    fixed point. Entry is therefore not spontaneous: it has to be seeded inside the pull
    region, and how far in "inside" is depends on ``tau``, which nothing in the model sets.
    With ``a = 0`` the force points inward at every finite placement, but its magnitude
    decays like ``kappa l``, so arrival from far out is exponentially slow."""
    a, N, R0 = 5.0, 1, 90.0
    # each participant starts at *its own* settled effort for the placement it is at,
    # x = p kappa R/gamma; otherwise the comparison is with an effort it has no reason to hold.
    for l0 in (4.0, 6.0, 8.0):
        x0 = PAR.p * float(np.exp(-l0 ** 2 / (2 * TAU))) * R0 / PAR.gamma
        arr = entry_system(PAR, [l0], [a], tau=TAU, R0=R0, x0=x0, place_mobility=PM)
        tr = np.asarray(run(arr, 40_000, every=20_000))
        ells = tr[:, 2 + N]
        assert ells[-1] > ells[0]                    # moves further out, not in
        assert float(np.exp(-ells[-1] ** 2 / (2 * TAU))) < float(np.exp(-l0 ** 2 / (2 * TAU)))
    # the separatrix is x_i R = a, i.e. kappa = a gamma/(p R^2): inside it, the same
    # participant does come in.
    l_c = float(np.sqrt(-2 * TAU * np.log(a * PAR.gamma / (PAR.p * R0 ** 2))))
    assert 3.0 < l_c < 4.0
    x0 = PAR.p * float(np.exp(-(0.8 * l_c) ** 2 / (2 * TAU))) * R0 / PAR.gamma
    arr = entry_system(PAR, [0.8 * l_c], [a], tau=TAU, R0=R0, x0=x0, place_mobility=PM)
    tr = np.asarray(run(arr, 40_000, every=20_000))
    assert tr[-1, 2 + N] < tr[0, 2 + N]
    # and with no cost of proximity even the far start moves in
    x0 = PAR.p * float(np.exp(-16.0 / (2 * TAU))) * R0 / PAR.gamma
    arr = entry_system(PAR, [4.0], [0.0], tau=TAU, R0=R0, x0=x0, place_mobility=PM)
    tr = np.asarray(run(arr, 40_000, every=20_000))
    assert tr[-1, 2 + N] < tr[0, 2 + N]
