"""Placement changes the effective wiring (the book's reclustering point).

The mathematics is ``background/affine_quadratic_operad.tex`` in the book repository
(eq:spatial-transfers, eq:maps) read as the book, Ch. 5 reads it. **Every docstring
below says which of three things its test does** -- reproduce a closed form the paper
proves, check a discrete-time analogue the book still has to prove (STATUS.md §3.2
item 1), or merely illustrate (STATUS.md §3.2).

No test in the paper isolates this. The paper supplies the kernel
``kappa_ij = exp(-||l_i - l_j||^2 / 2 tau)``; §5.4 draws the consequence: "coupling
strength depends on placement through the kernel, and placement is a descended
coordinate, so the *effective* wiring changes during a run within a fixed all-to-all
syntax; two boxes pulled far apart are effectively unwired."

**What stays fixed, and it is the whole point.** The number of boxes and their port types
do not change: the arrangement is the same arity-two operation of ``A_B`` throughout, with
the same ``B^(x)2 -> B``, the same ``in_dim/out_dim`` on every port, and the same transfer
family ``T``. Only ``q`` moves, and ``q`` is a coordinate the dynamics descends. Changeable
interfaces would be a different construction (arXiv 2602.17917 §9.5); this is not one.

The setup: a two-ary spatial arrangement with the placements ``p_1, p_2`` mobile and the
value maps frozen at the identity, and two components ``I -> B`` that predict the position
they receive (eq:subsidiary-cost, imported from tests/test_rendezvous.py) with fixed
expectations ``-R`` and ``+R``. The composite drives the two children apart, and the
kernel does the rest.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.attention import (
    identity_transfer, input_dim, make_input, message, msg_dim, spatial_arrangement,
    spatial_config, spatial_kappa, weight, zero_cost, zero_message,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import constant
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_rendezvous import run_closed, subsidiary

DH, DP = 1, 1
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)
TAU = 0.8
R = 3.0                                      # the expectation each component holds
EXT = make_input(zero_message(DH), 0.0)      # exterior input (0_M, 0)


def spatial_parent(mobility, start=(0.0, 0.0)):
    """The arity-two spatial arrangement of eq:spatial-transfers over
    ``Q_2 = P^2 x End(H)^2``, zero cost, value maps held at the identity (mobility ``0``)
    and placements mobile."""
    return spatial_arrangement(
        2, DH, DP, TAU, cost=zero_cost,
        parameter=configuration(
            constant(jnp.diag(jnp.asarray([mobility, mobility, 0.0, 0.0]))),
            init=spatial_config(start, (1.0, 1.0))),
        label="spatial(2)")


def separating_system(mobility=0.2, start=(0.0, 0.0)):
    """``a(c_1, c_2) : I -> B``: the spatial arrangement with two components whose costs
    ``(1/2)(p - z_i)^2`` pull their placements toward ``z_1 = -R`` and ``z_2 = +R``.

    The components' own expectations are frozen (mobility ``0`` on ``z_i``), so they are
    fixed private predictions; what moves is the arrangement's placement, and it moves
    because the components' costs are pulled back to it by substitution -- the paper's
    "Even a placement arrangement with zero cost can receive position covectors from the
    components placed inside it", here with genuine transfers as well."""
    parts = [subsidiary(-R, 0.0, "c1"), subsidiary(R, 0.0, "c2")]
    return compose_seq(tensor_arrangements(parts), spatial_parent(mobility, start).arr())


def placements(q):
    """``(l_1(q), l_2(q))`` from a composite configuration ``(z_1, z_2, p_1, p_2, V_1, V_2)``."""
    return np.asarray(q)[2:4]


def kappa_12(q):
    """``kappa_12 = exp(-(l_1 - l_2)^2 / 2 tau)`` at a composite configuration."""
    p = placements(q)
    return float(spatial_kappa(jnp.array([p[0]]), jnp.array([p[1]]), TAU))


def separation_for(eps):
    """The separation at which eq:spatial-transfers' weight multiplier drops below
    ``eps``: ``kappa < eps`` iff ``||l_1 - l_2|| > sqrt(-2 tau log eps)``."""
    return float(np.sqrt(-2.0 * TAU * np.log(eps)))


# ---------------------------------------------------------------------------


def test_the_weight_multiplier_is_below_any_epsilon_past_a_separation():
    """REPRODUCES A CLOSED FORM the paper states (eq:spatial-transfers): the transfer
    ``T_ij(q)(h, w) = (kappa_ij V_j h, kappa_ij w)`` has weight multiplier
    ``kappa_ij = exp(-||l_i - l_j||^2 / 2 tau)``, so for any ``eps > 0`` a separation
    beyond ``sqrt(-2 tau log eps)`` makes the transferred weight less than ``eps`` times
    the sender's. The "any ``eps``" is the content: this is not a threshold in the kernel,
    it is the kernel's tail."""
    a = spatial_parent(0.0)
    for eps in (1e-3, 1e-6, 1e-9):
        s = separation_for(eps)
        q = spatial_config((0.0, s * 1.001), (1.0, 1.0))
        T = a.transfers(q)
        m = message(1.0, 1.0)
        assert float(weight(m)) == 1.0
        assert float(T[1][2].gamma) < eps
        assert float(weight(jnp.asarray(T[1][2].gamma) * m)) < eps
        # and just inside the separation it is not below eps
        q_near = spatial_config((0.0, s * 0.999), (1.0, 1.0))
        assert float(a.transfers(q_near)[1][2].gamma) > eps


def test_a_run_drives_the_children_apart_and_unwires_them():
    """CHECKS THE DISCRETE ANALOGUE the book still has to prove (STATUS.md §3.2
    item 1) of the book's own reclustering point (§5.4(i)), which the paper does not state:
    the placements are descended coordinates, so the effective coupling changes *during a
    run*. Starting with both children at the same place -- ``kappa_12 = 1``, a fully wired
    pair -- the pulled-back component costs separate them, and ``kappa_12`` falls
    monotonically below ``1e-6``: the weight child ``1`` receives from child ``2`` is a
    millionth of what child ``2`` sends -- it settles at ``1.69e-10``, the kernel at the
    separation ``2R = 6``. Nothing about the syntax changed to make this happen."""
    b = separating_system(mobility=0.2, start=(0.0, 0.0))
    tr = run_closed(b, 200, lambda t: EXT)

    ks = np.array([kappa_12(q) for q in tr])
    np.testing.assert_allclose(ks[0], 1.0, atol=1e-14)
    assert np.all(np.diff(ks) <= 0.0)                    # never increases
    assert np.all(np.diff(ks[:20]) < 0.0)                # strictly, until it settles
    assert ks[-1] < 1e-6
    np.testing.assert_allclose(ks[-1], np.exp(-(2 * R) ** 2 / (2 * TAU)), rtol=1e-8)
    np.testing.assert_allclose(placements(tr[-1]), [-R, R], atol=1e-9)
    assert abs(placements(tr[-1])[1] - placements(tr[-1])[0]) > separation_for(1e-6)


def test_the_syntax_is_fixed_while_the_effective_wiring_changes():
    """ILLUSTRATES what §5.4 says remains outside the mathematics: "the number of boxes
    and their port types are fixed". The composite is one arrangement object for the whole
    run -- same arity, same four port dimensions, same transfer family, same parameter --
    and the transfers are the *identity* on the signal at every configuration visited
    (value maps frozen), so nothing but the scalar ``kappa_12`` differs between the wired
    and the unwired end of the run. The change is in ``q``, which the dynamics descends,
    and it is a change of a coefficient in a fixed operation, not a change of operation."""
    b = separating_system(mobility=0.2, start=(0.0, 0.0))
    a = spatial_parent(0.2)
    dims = (b.in_dim_M, b.out_dim_M, b.in_dim_N, b.out_dim_N)
    assert dims == (0, 0, EDIM, MDIM)
    assert (a.n, a.dim_H, a.dim_P) == (2, DH, DP)

    tr = run_closed(b, 200, lambda t: EXT)
    I = identity_transfer(DH)
    for q in (tr[0], tr[100], tr[-1]):
        T = a.transfers(jnp.asarray(q)[2:])
        for i in range(3):
            for j in range(3):
                if i != j:
                    # T_ij = kappa_ij * (identity on H, identity on W): only the scalar moves
                    np.testing.assert_allclose(T[i][j].L, T[i][j].gamma * I.L, atol=1e-14)
    assert dims == (b.in_dim_M, b.out_dim_M, b.in_dim_N, b.out_dim_N)


def test_an_unwired_child_stops_hearing_the_other():
    """CHECKS THE DISCRETE ANALOGUE (§5.9 item 1) of the same point, read off the
    receiving map rather than the kernel, which is what "effectively unwired" has to mean:
    perturbing child ``2``'s message changes child ``1``'s received input ``rho_a(q, m, e)_1``
    (eq:maps) by the full perturbation at the start of the run and by less than ``1e-6`` of
    it at the end. The position half of the input is untouched by the perturbation at
    either end -- placement is what carries the message, not the other way about."""
    b = separating_system(mobility=0.2, start=(0.0, 0.0))
    a = spatial_parent(0.2)
    tr = run_closed(b, 200, lambda t: EXT)

    m = jnp.concatenate([message(0.4, 1.0), message(-1.3, 2.0)])
    bump = jnp.concatenate([zero_message(DH), message(1.0, 1.0)])
    sens = []
    for q in (tr[0], tr[-1]):
        qa = jnp.asarray(q)[2:]
        base = a.received(qa, m, EXT, 1)
        moved = a.received(qa, m + bump, EXT, 1)
        sens.append(float(jnp.max(jnp.abs(moved - base))))
        np.testing.assert_allclose(moved[MDIM:], base[MDIM:], atol=1e-14)  # position half
    assert sens[0] > 0.9
    assert sens[1] < 1e-6 * sens[0]
