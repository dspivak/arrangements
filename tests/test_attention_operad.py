"""The attention operad ``A_B``: its algebra, and substitution as composition in ``arr``.

The mathematics is ``background/affine_quadratic_operad.tex`` in the book
repository, cited by its own labels. There is no old attention code and hence no
regression ``.npz`` for this module: acceptance is against the paper's closed
forms. **Every docstring below says which of three things its test does** --
reproduce a closed form the paper proves, check a discrete-time analogue the book
must still prove, or merely illustrate (STATUS.md §3.2).

The test that earns the stage is ``test_substitution_agrees_with_composition_*``:
the paper gives substitution twice over, explicitly
(eq:composite-state/-transfers/-cost) and abstractly (composition of lenses), and
prop:operad says the two agree. Here ``substitute`` is the former, written with no
reference to ``wiring.py``, and ``compose_seq(tensor_arrangements([...]), a)`` is
the latter.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.attention import (
    Transfer, add_transfers, apply_transfer, assert_admissible, attention_arrangement,
    compose_transfers, fixed_prediction, forget_to_placement, identity_transfer,
    identity_transfers, input_dim, is_admissible, is_pure_placement, make_input, mean,
    message, msg_dim, placement_arrangement, prediction_cost, spatial_Q_dim,
    spatial_arrangement, spatial_kappa, spatial_mean, spatial_split, split_messages,
    substitute, symmetric_action, transfer, unit_arrangement, zero_message,
    zero_transfer,
)
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

DH, DP, TAU = 2, 2, 0.8     # signal, position dimension; the Gaussian scale
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)


# ---------------------------------------------------------------------------
# Fixtures: spatial arrangements with private prediction costs.
# ---------------------------------------------------------------------------


def spatial_part(n, seed, mobility=0.05, label=None):
    """An ``n``-ary spatial arrangement (eq:spatial-transfers) whose cost is a
    private prediction error (eq:prediction) with a random predictor, affine in the
    child messages and smoothly configuration-dependent."""
    r = np.random.default_rng(seed)
    init = jnp.asarray(r.standard_normal(spatial_Q_dim(n, DH, DP)) * 0.6)
    A = jnp.asarray(r.standard_normal((EDIM, n * MDIM)) * 0.3)
    c = jnp.asarray(r.standard_normal(EDIM) * 0.3)

    def predictor(q, m_out):
        return A @ m_out + c + 0.2 * jnp.sum(q)

    return spatial_arrangement(
        n, DH, DP, TAU, cost=prediction_cost(predictor), init=init,
        mobility=mobility, label=label or f"s{seed}/{n}",
    )


def random_interaction(a, seed):
    """A random point ``(q, m, e)`` of ``Q x D_n`` for the arrangement ``a``."""
    r = np.random.default_rng(seed)
    return (
        jnp.asarray(r.standard_normal(a.P.Q_dim) * 0.5),
        jnp.asarray(r.standard_normal(a.n * MDIM)),
        jnp.asarray(r.standard_normal(EDIM)),
    )


def permute_blocks(q, sizes, order):
    """``q`` cut into consecutive blocks of the given ``sizes`` and reassembled in
    ``order`` -- the coherence isomorphism reordering a product of configuration
    spaces (eq:integrators)."""
    cut = [sum(sizes[:i]) for i in range(len(sizes) + 1)]
    return jnp.concatenate([q[cut[i]:cut[i + 1]] for i in order])


def assert_same_operation(x, y, q_x, q_y, seed, atol=1e-11):
    """Two attention arrangements agree as operations: same placement, transfers,
    sending, receiving and cost, at configurations ``q_x``, ``q_y`` that correspond
    under the reordering of configuration factors."""
    assert x.n == y.n
    np.testing.assert_allclose(x.placement(q_x), y.placement(q_y), atol=atol)
    Tx, Ty = x.transfers(q_x), y.transfers(q_y)
    for i in range(x.n + 1):
        for j in range(x.n + 1):
            np.testing.assert_allclose(Tx[i][j].gamma, Ty[i][j].gamma, atol=atol)
            np.testing.assert_allclose(Tx[i][j].L, Ty[i][j].L, atol=atol)
    r = np.random.default_rng(seed)
    for _ in range(4):
        m = jnp.asarray(r.standard_normal(x.n * MDIM))
        e = jnp.asarray(r.standard_normal(EDIM))
        np.testing.assert_allclose(x.sigma(q_x, m), y.sigma(q_y, m), atol=atol)
        np.testing.assert_allclose(x.rho(q_x, m, e), y.rho(q_y, m, e), atol=atol)
        np.testing.assert_allclose(x.cost(q_x, m, e), y.cost(q_y, m, e), atol=atol)


def trajectory(arr, steps, seed=0):
    """``Phi(arr)`` run for ``steps`` steps on fixed exterior data: the child
    messages, the incoming one-form on ``E^n``, and the exterior ``(xi_N, e)``.
    Returns the flattened states."""
    r = np.random.default_rng(seed)
    m_out = jnp.asarray(r.standard_normal(arr.out_dim_M) * 0.5)
    B = jnp.asarray(r.standard_normal((arr.in_dim_M, arr.in_dim_M)) * 0.05)
    c = jnp.asarray(r.standard_normal(arr.in_dim_M) * 0.05)
    xi_N = jnp.asarray(r.standard_normal(arr.out_dim_N) * 0.1)
    e = jnp.asarray(r.standard_normal(arr.in_dim_N) * 0.5)
    in_pos = (m_out, lambda x: B @ x + c)

    O = Phi(arr)
    state, out = O.state, [flat_state(O.state)]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one(in_pos, lambda _o: (xi_N, e))
        out.append(flat_state(state))
    return out


def fixed_transfers(n, gammas, mats=None):
    """A configuration-independent transfer family from a ``(n+1, n+1)`` matrix of
    weight multipliers (zero entries meaning the zero transfer) and optional signal
    maps; ``L_ij = gamma_ij * mats[i][j]``, the identity by default."""
    G = jnp.asarray(gammas, float)

    def T(q):
        rows = []
        for i in range(n + 1):
            row = []
            for j in range(n + 1):
                g = G[i, j]
                if float(g) == 0.0:
                    row.append(zero_transfer(DH))
                else:
                    M = jnp.eye(DH) if mats is None else jnp.asarray(mats[i][j], float)
                    row.append(Transfer(gamma=g, L=g * M))
            rows.append(tuple(row))
        return tuple(rows)

    return T


# ---------------------------------------------------------------------------
# Messages, transfers, and the mean readout (notes/ATTENTION-STATUS.md §1).
# ---------------------------------------------------------------------------


def test_transfer_algebra_adds_and_composes():
    """REPRODUCES A CLOSED FORM the paper proves (eq:transfer-algebra):
    ``T + T' = T_{gamma+gamma', L+L'}`` and ``T . T' = T_{gamma gamma', L L'}``,
    checked against the action on messages. Signal maps need not commute, so the
    composite in the other order differs."""
    r = np.random.default_rng(11)
    T1 = transfer(1.7, r.standard_normal((DH, DH)))
    T2 = transfer(0.4, r.standard_normal((DH, DH)))
    m = message(r.standard_normal(DH), 2.5)

    np.testing.assert_allclose(
        apply_transfer(add_transfers(T1, T2), m),
        apply_transfer(T1, m) + apply_transfer(T2, m), atol=1e-12)
    np.testing.assert_allclose(
        apply_transfer(compose_transfers(T1, T2), m),
        apply_transfer(T1, apply_transfer(T2, m)), atol=1e-12)
    np.testing.assert_allclose(compose_transfers(T1, T2).gamma, T1.gamma * T2.gamma, atol=1e-12)
    assert not np.allclose(compose_transfers(T1, T2).L, compose_transfers(T2, T1).L)

    I = identity_transfer(DH)
    np.testing.assert_allclose(apply_transfer(I, m), m, atol=1e-12)
    np.testing.assert_allclose(apply_transfer(zero_transfer(DH), m), zero_message(DH), atol=1e-12)


def test_transfers_preserve_admissible_messages():
    """REPRODUCES A CLOSED FORM the paper proves (eq:transfer): a message transfer
    preserves ``M_adm = {0_M} u (H x R_{>0})``, because ``gamma > 0``. The last two
    assertions ILLUSTRATE the guard of notes/ATTENTION-STATUS.md §1: ``R^{dim_H+1}`` does
    not enforce admissibility, so a negative weight is caught rather than ignored."""
    r = np.random.default_rng(12)
    T = transfer(0.6, r.standard_normal((DH, DH)))
    for m in [message(r.standard_normal(DH), 3.0), zero_message(DH)]:
        assert is_admissible(m)
        assert is_admissible(apply_transfer(T, m))

    assert not is_admissible(message(jnp.ones(DH), -1.0))
    assert not is_admissible(message(jnp.ones(DH), 0.0))   # nonzero signal, no weight
    with pytest.raises(ValueError):
        assert_admissible(message(jnp.ones(DH), -1.0))


def test_mean_readout_is_the_weighted_average_and_needs_positive_weight():
    """REPRODUCES A CLOSED FORM the paper proves (eq:mean and the line after it):
    messages ``m_j = (w_j h_j, w_j)`` have ``mean(sum_j m_j) = sum_j w_j h_j / sum_j w_j``.
    The failure case ILLUSTRATES the §3 guard: ``mean`` is defined only on
    ``M_{>0}``, and is a separate operation, never folded into a cost."""
    r = np.random.default_rng(13)
    h = [jnp.asarray(r.standard_normal(DH)) for _ in range(3)]
    w = [0.5, 2.0, 1.25]
    total = zero_message(DH)
    for hj, wj in zip(h, w):
        total = total + message(wj * hj, wj)
    np.testing.assert_allclose(
        mean(total), sum(wj * hj for hj, wj in zip(h, w)) / sum(w), atol=1e-12)

    with pytest.raises(ValueError):
        mean(message(jnp.ones(DH), 0.0))
    with pytest.raises(ValueError):
        mean(message(jnp.ones(DH), -2.0))


def test_spatial_mean_is_the_mean_of_the_received_message():
    """REPRODUCES A CLOSED FORM the paper proves (eq:spatial-mean): the weighted
    spatial softmax

        hbar_i = sum_{j in J_i} kappa_ij w_j V_j h_j / sum_{j in J_i} kappa_ij w_j

    is exactly the mean readout of the message ``rho_a(q, m, e)_i`` that the
    receiving map (eq:maps) delivers -- the normalization is a readout, and the
    unnormalized total and weight remain on the interface."""
    n = 3
    r = np.random.default_rng(14)
    q = jnp.asarray(r.standard_normal(spatial_Q_dim(n, DH, DP)) * 0.7)
    a = spatial_arrangement(n, DH, DP, TAU, init=q)
    msgs = jnp.stack([message(r.standard_normal(DH), 0.5 + abs(r.standard_normal()))
                      for _ in range(n + 1)])
    m_out = jnp.ravel(msgs[1:])
    e = make_input(msgs[0], r.standard_normal(DP))

    for i in range(1, n + 1):
        received = a.received(q, m_out, e, i)[:MDIM]     # the message half of e_i
        np.testing.assert_allclose(
            spatial_mean(n, DH, DP, TAU, q, msgs, i), mean(received), atol=1e-12)
    # receiver 0 is the exterior: its message is sigma_a, summed over J_0 = {1..n}
    np.testing.assert_allclose(
        spatial_mean(n, DH, DP, TAU, q, msgs, 0), mean(a.sigma(q, m_out)), atol=1e-12)


def test_prediction_cost_of_a_fixed_prediction_and_its_forcing():
    """REPRODUCES A CLOSED FORM (eq:prediction, differentiated): with the private
    prediction ``ehat`` held fixed, ``k = (1/2)||e - ehat||^2`` in the product
    Euclidean norm on ``E = M x P``, so ``omega = -dk`` is ``(ehat - e) de`` -- a
    covector with a message half and a position half -- and vanishes on ``Q`` and on
    the child messages. The prediction concerns the weighted input itself; no mean
    readout appears inside the cost."""
    r = np.random.default_rng(15)
    e_hat = jnp.asarray(r.standard_normal(EDIM))
    a = attention_arrangement(
        2, DH, DP, lambda q: jnp.zeros((2, DP)), identity_transfers(2, DH),
        cost=prediction_cost(fixed_prediction(e_hat)), label="predictor")
    q = jnp.zeros(0)
    m = jnp.asarray(r.standard_normal(2 * MDIM))
    e = jnp.asarray(r.standard_normal(EDIM))

    np.testing.assert_allclose(a.cost(q, m, e), 0.5 * float(jnp.sum((e - e_hat) ** 2)), atol=1e-12)
    xi_q, xi_m, xi_e = a.arr().omega(q, m, e)
    assert xi_q.shape == (0,)
    np.testing.assert_allclose(xi_m, jnp.zeros(2 * MDIM), atol=1e-14)
    np.testing.assert_allclose(xi_e, e_hat - e, atol=1e-12)


def test_forcing_is_the_full_differential_on_configuration_and_interaction():
    """ILLUSTRATES eq:forcing, which the paper is explicit about: ``omega_a = -dk_a``
    on ``Q x D_n`` is the full differential, "not just a differential in ``q``". So
    all three components of the arrangement's one-form -- on ``Q``, on ``M^n`` and
    on ``E`` -- are generally nonzero for a prediction cost."""
    a = spatial_part(2, 21)
    q, m, e = random_interaction(a, 22)
    xi_q, xi_m, xi_e = a.arr().omega(q, m, e)
    assert float(jnp.max(jnp.abs(xi_q))) > 1e-6
    assert float(jnp.max(jnp.abs(xi_m))) > 1e-6
    assert float(jnp.max(jnp.abs(xi_e))) > 1e-6


# ---------------------------------------------------------------------------
# The theorem: substitution is composition in arr (prop:operad).
# ---------------------------------------------------------------------------


def _agrees_with_composition(a, parts, seed, steps=40):
    """``substitute`` (eq:composite-*) against ``compose_seq(tensor(parts), a)``:
    equal ``sigma``, equal ``rho``, equal forcing in all three components, and equal
    ``Phi`` trajectories."""
    b = substitute(a, parts)
    explicit = b.arr()
    composite = compose_seq(tensor_arrangements([p.arr() for p in parts]), a.arr())

    assert (explicit.in_dim_M, explicit.out_dim_M) == (composite.in_dim_M, composite.out_dim_M)
    assert (explicit.in_dim_N, explicit.out_dim_N) == (composite.in_dim_N, composite.out_dim_N)
    assert explicit.P.Q_dim == composite.P.Q_dim

    r = np.random.default_rng(seed)
    for _ in range(5):
        q = jnp.asarray(r.standard_normal(explicit.P.Q_dim) * 0.5)
        m = jnp.asarray(r.standard_normal(explicit.out_dim_M))
        e = jnp.asarray(r.standard_normal(explicit.in_dim_N))
        np.testing.assert_allclose(explicit.out_f(q, m), composite.out_f(q, m), atol=1e-12)
        np.testing.assert_allclose(explicit.in_f(q, m, e), composite.in_f(q, m, e), atol=1e-12)
        for x, y in zip(explicit.omega(q, m, e), composite.omega(q, m, e)):
            np.testing.assert_allclose(x, y, atol=1e-11)

    t1 = trajectory(explicit, steps, seed=seed)
    t2 = trajectory(composite, steps, seed=seed)
    assert len(t1) == steps + 1
    for x, y in zip(t1, t2):
        np.testing.assert_allclose(x, y, atol=1e-11)
    assert float(jnp.max(jnp.abs(t1[-1] - t1[0]))) > 1e-3   # the run is not stationary
    return b


def test_substitution_agrees_with_composition_two_level_nest():
    """REPRODUCES WHAT THE PAPER PROVES (prop:operad), for the concrete data of
    eq:composite-state, eq:composite-transfers, eq:composite-cost: the explicit
    composite equals the lens composite ``compose_seq(tensor(a_1, a_2), a)`` in
    ``sigma``, ``rho`` and the whole forcing (the chain rule eq:chain-rule).

    The trajectory comparison is the book's DISCRETE reading, which the book still
    has to state (STATUS.md §3.2): here it follows from the equality of
    the arrangement data, since both composites carry the same parameter, so it
    checks the port rather than the mathematics."""
    a = spatial_part(2, 1)
    _agrees_with_composition(a, [spatial_part(2, 2), spatial_part(2, 3)], seed=101)


def test_substitution_agrees_with_composition_uneven_nest():
    """REPRODUCES WHAT THE PAPER PROVES (prop:operad) for an uneven nest, arities
    ``1, 2, 0``. The ``0``-ary component exercises the empty sum: its sending map is
    the empty sum ``0_M`` and it has no children to place, while it still carries a
    cost on its own received input."""
    a = spatial_part(3, 4)
    parts = [spatial_part(1, 5), spatial_part(2, 6), spatial_part(0, 7)]
    b = _agrees_with_composition(a, parts, seed=102)
    assert b.n == 3


def test_empty_sum_is_the_zero_message():
    """REPRODUCES A CLOSED FORM (the paper's "Empty sums equal ``0_M``", with
    eq:maps): a ``0``-ary arrangement sends ``0_M`` whatever its configuration, and
    receives nothing; its cost still depends on the exterior input, so it is not a
    trivial component."""
    z = spatial_part(0, 7)
    q, m, e = random_interaction(z, 71)
    np.testing.assert_allclose(z.sigma(q, m), zero_message(DH), atol=1e-14)
    assert z.rho(q, m, e).shape == (0,)
    _, _, xi_e = z.arr().omega(q, m, e)
    assert float(jnp.max(jnp.abs(xi_e))) > 1e-6


def test_substitution_is_associative_on_a_three_level_nest():
    """REPRODUCES WHAT THE PAPER PROVES (prop:operad: "Associativity of linear-map
    composition makes this independent of parenthesization ... position addition and
    the coherent integrator product are associative as well"). The two bracketings
    of a three-level nest are compared as operations -- placement, transfers,
    sending, receiving, cost, and the forcing -- up to the reordering of
    configuration factors, which is the coherence isomorphism of eq:integrators and
    is exhibited explicitly as ``permute_blocks``."""
    a = spatial_part(2, 31)
    a1, a2 = spatial_part(2, 32), spatial_part(1, 33)
    g = [spatial_part(1, 34), spatial_part(2, 35), spatial_part(1, 36)]

    left = substitute(substitute(a, [a1, a2]), g)          # Q: g0 g1 g2 a1 a2 a
    right = substitute(a, [substitute(a1, g[:2]), substitute(a2, g[2:])])   # g0 g1 a1 g2 a2 a
    sizes = [p.P.Q_dim for p in g] + [a1.P.Q_dim, a2.P.Q_dim, a.P.Q_dim]

    r = np.random.default_rng(103)
    q_left = jnp.asarray(r.standard_normal(left.P.Q_dim) * 0.5)
    q_right = permute_blocks(q_left, sizes, [0, 1, 3, 2, 4, 5])
    assert_same_operation(left, right, q_left, q_right, seed=104)

    # the forcings agree too, with the configuration covector reordered the same way
    m = jnp.asarray(r.standard_normal(left.n * MDIM))
    e = jnp.asarray(r.standard_normal(EDIM))
    xl = left.arr().omega(q_left, m, e)
    xr = right.arr().omega(q_right, m, e)
    np.testing.assert_allclose(permute_blocks(xl[0], sizes, [0, 1, 3, 2, 4, 5]), xr[0], atol=1e-11)
    np.testing.assert_allclose(xl[1], xr[1], atol=1e-11)
    np.testing.assert_allclose(xl[2], xr[2], atol=1e-11)


def test_symmetric_action_relabels_slots():
    """REPRODUCES A CLOSED FORM the paper states (prop:operad's ``tau`` formulas):
    ``l_{a.tau,i} = l_{a,tau(i)}``, ``T^{a.tau}_{ij} = T^a_{tau(i)tau(j)}``, and the
    cost precomposed with the relabeling. Consequently
    ``rho_{a.tau}(q, m, e)_i = rho_a(q, m', e)_{tau(i)}`` with ``m'_{tau(i)} = m_i``:
    the permutation relabels child indices and their message slots, fixing the
    exterior index ``0``."""
    a = spatial_part(3, 41)
    tau = (2, 0, 1)
    at = symmetric_action(a, tau)
    q, m, e = random_interaction(a, 42)
    ms = split_messages(m, 3, DH)
    m_rel = jnp.concatenate([ms[tau.index(k)] for k in range(3)])   # m'_{tau(i)} = m_i

    np.testing.assert_allclose(at.sigma(q, m), a.sigma(q, m_rel), atol=1e-12)
    rt = at.rho(q, m, e).reshape(3, EDIM)
    ra = a.rho(q, m_rel, e).reshape(3, EDIM)
    for i in range(3):
        np.testing.assert_allclose(rt[i], ra[tau[i]], atol=1e-12)
    np.testing.assert_allclose(at.cost(q, m, e), a.cost(q, m_rel, e), atol=1e-12)


def test_symmetric_action_commutes_with_substitution():
    """REPRODUCES WHAT THE PAPER STATES (prop:operad: "these relabelings respect
    substitution"). For ``tau in S_n``,

        (a.tau)(a_{tau(1)}, ..., a_{tau(n)}) = (a(a_1, ..., a_n)).phi,

    ``phi`` the induced block permutation of the ``N`` composite slots, again up to
    the reordering of configuration factors."""
    a = spatial_part(3, 51)
    parts = [spatial_part(2, 52), spatial_part(1, 53), spatial_part(2, 54)]
    tau = (2, 0, 1)

    b = substitute(a, parts)
    left = substitute(symmetric_action(a, tau), [parts[tau[i]] for i in range(3)])

    starts_b = [sum(p.n for p in parts[:i]) for i in range(4)]
    phi = [starts_b[tau[i]] + j for i in range(3) for j in range(parts[tau[i]].n)]
    right = symmetric_action(b, phi)

    sizes = [p.P.Q_dim for p in parts] + [a.P.Q_dim]
    r = np.random.default_rng(105)
    q_b = jnp.asarray(r.standard_normal(b.P.Q_dim) * 0.5)
    q_left = permute_blocks(q_b, sizes, list(tau) + [3])
    assert_same_operation(left, right, q_left, q_b, seed=106)


def test_unit_acts_as_an_identity_for_substitution():
    """REPRODUCES WHAT THE PAPER STATES (prop:operad): the unit has arity one,
    one-point configuration, placement zero, cost zero and ``T_{01} = T_{10} = M``,
    and its sending and receiving maps are identities. Both unit laws are checked,
    ``a(1, ..., 1) = a`` and ``1(a) = a``; the configuration spaces differ only by
    ``R^0`` factors, so the same ``q`` serves for all three."""
    one = unit_arrangement(DH, DP)
    q0 = jnp.zeros(0)
    np.testing.assert_allclose(one.sigma(q0, message(jnp.arange(DH, dtype=float), 2.0)),
                               message(jnp.arange(DH, dtype=float), 2.0), atol=1e-14)
    e = make_input(message(jnp.ones(DH), 1.5), jnp.arange(DP, dtype=float))
    np.testing.assert_allclose(one.rho(q0, zero_message(DH), e), e, atol=1e-14)

    a = spatial_part(3, 61)
    q, _, _ = random_interaction(a, 62)
    right = substitute(a, [one, one, one])
    left = substitute(one, [a])
    assert right.P.Q_dim == a.P.Q_dim and left.P.Q_dim == a.P.Q_dim
    assert_same_operation(right, a, q, q, seed=63)
    assert_same_operation(left, a, q, q, seed=64)


def test_zero_diagonal_keeps_a_groups_message_out_of_that_group():
    """REPRODUCES WHAT THE PAPER PROVES (prop:operad: "The zero diagonal of the
    outer arrangement ensures that a group's outgoing message is not returned to
    that same group"). Inside group ``1`` the transfer from child ``2`` to child
    ``1`` is set to the zero map; in the composite, child ``(1,1)`` then receives
    *nothing* from child ``(1,2)``, because the only other route -- out of group 1
    and back in -- would run through ``T^a_{11} = 0``. Across groups the same child
    does receive, so the isolation is the diagonal's doing and not a dead arrangement."""
    outer = attention_arrangement(
        2, DH, DP, lambda q: jnp.zeros((2, DP)),
        fixed_transfers(2, [[0, 1, 1], [1, 0, 1], [1, 1, 0]]), label="outer")
    inner1 = attention_arrangement(
        2, DH, DP, lambda q: jnp.zeros((2, DP)),
        fixed_transfers(2, [[0, 1, 1], [1, 0, 0], [1, 1, 0]]), label="in1")  # T_12 = 0
    inner2 = attention_arrangement(
        2, DH, DP, lambda q: jnp.zeros((2, DP)),
        fixed_transfers(2, [[0, 1, 1], [1, 0, 1], [1, 1, 0]]), label="in2")

    b = substitute(outer, [inner1, inner2])
    T = b.transfers(jnp.zeros(0))
    assert float(T[1][2].gamma) == 0.0 and float(jnp.max(jnp.abs(T[1][2].L))) == 0.0
    assert float(T[1][3].gamma) > 0.0        # from group 2 it does hear

    r = np.random.default_rng(71)
    m = jnp.asarray(r.standard_normal(4 * MDIM))
    e = jnp.asarray(r.standard_normal(EDIM))
    ms = split_messages(m, 4, DH)
    bumped_same = jnp.ravel(ms.at[1].add(jnp.ones(MDIM)))     # child (1,2)
    bumped_other = jnp.ravel(ms.at[2].add(jnp.ones(MDIM)))    # child (2,1)
    base = b.received(jnp.zeros(0), m, e, 1)
    np.testing.assert_allclose(b.received(jnp.zeros(0), bumped_same, e, 1), base, atol=1e-14)
    assert float(jnp.max(jnp.abs(b.received(jnp.zeros(0), bumped_other, e, 1) - base))) > 0.1


# ---------------------------------------------------------------------------
# The pure placement suboperad P_B (the paper's "Pure placement" paragraph).
# ---------------------------------------------------------------------------


def placement_part(n, seed, mobility=0.1):
    """An operation of ``P_B``: identity transfers, zero cost, placement the
    configuration itself (``Q = P^n``)."""
    r = np.random.default_rng(seed)
    init = jnp.asarray(r.standard_normal(n * DP) * 0.5)
    return placement_arrangement(
        n, DH, DP, lambda q: jnp.reshape(q, (n, DP)),
        parameter=configuration(euclidean(n * DP, mobility), init=init), label=f"p{seed}")


def test_pure_placement_is_closed_under_substitution():
    """REPRODUCES WHAT THE PAPER PROVES: "eq:composite-transfers preserves these
    identity transfers, and the cost remains zero", so ``P_B`` is a suboperad.
    Placements compose by eq:composite-state, ``l_{b,(i,j)} = l_{a,i} + l_{a_i,j}``,
    which is checked here as well."""
    a, p1, p2 = placement_part(2, 81), placement_part(2, 82), placement_part(3, 83)
    b = substitute(a, [p1, p2])

    r = np.random.default_rng(84)
    q = jnp.asarray(r.standard_normal(b.P.Q_dim) * 0.5)
    inter = [(jnp.asarray(r.standard_normal(b.n * MDIM)), jnp.asarray(r.standard_normal(EDIM)))
             for _ in range(3)]
    assert is_pure_placement(b, q, inter)

    q1, q2, qa = q[:p1.P.Q_dim], q[p1.P.Q_dim:p1.P.Q_dim + p2.P.Q_dim], q[-a.P.Q_dim:]
    expected = jnp.concatenate([
        a.placement(qa)[0] + p1.placement(q1),
        a.placement(qa)[1] + p2.placement(q2)])
    np.testing.assert_allclose(b.placement(q), expected, atol=1e-12)


def test_placement_retraction_is_the_identity_on_pure_placements():
    """REPRODUCES WHAT THE PAPER STATES: forgetting transfers and costs and then
    supplying identity transfers and zero cost is a retraction
    ``P_B -> A_B -> P_B``, whose remaining data are the integrator and the placement
    map. It is the identity on ``P_B``, and it sends a general attention arrangement
    into ``P_B`` keeping its placement."""
    p = placement_part(3, 85)
    q = jnp.asarray(np.random.default_rng(86).standard_normal(p.P.Q_dim) * 0.5)
    assert_same_operation(forget_to_placement(p), p, q, q, seed=87)

    a = spatial_part(3, 88)          # an arrangement with genuine transfers and cost
    qa, m, e = random_interaction(a, 89)
    fa = forget_to_placement(a)
    assert is_pure_placement(fa, qa, [(m, e)])
    assert fa.P is a.P
    np.testing.assert_allclose(fa.placement(qa), a.placement(qa), atol=1e-14)


def test_zero_cost_placement_still_receives_position_covectors():
    """REPRODUCES WHAT THE PAPER STATES ("Even a placement arrangement with zero
    cost can receive position covectors from the components placed inside it"): the
    parent's own cost is zero, but the children's costs are evaluated at inputs
    whose position coordinate is ``p_0 + l_i(q_a)`` (eq:maps), so ``-dk_b`` has a
    nonzero component on the parent's placement coordinates."""
    parent = placement_part(2, 91)
    children = [spatial_part(2, 92), spatial_part(1, 93)]
    b = substitute(parent, children)

    q, m, e = random_interaction(b, 94)
    np.testing.assert_allclose(parent.cost(q[-parent.P.Q_dim:], m[:2 * MDIM], e), 0.0, atol=1e-14)
    xi_q, _, _ = b.arr().omega(q, m, e)
    xi_parent = xi_q[-parent.P.Q_dim:]
    assert float(jnp.max(jnp.abs(xi_parent))) > 1e-6


# ---------------------------------------------------------------------------
# The Gaussian family is not closed under substitution (the paper's caveat).
# ---------------------------------------------------------------------------


def test_spatial_family_is_not_closed_under_substitution():
    """REPRODUCES THE PAPER'S OWN CAVEAT, with a witness: composing spatial
    arrangements "generally has different effective value maps for different
    sender-receiver pairs and is not a single Gaussian kernel on the exposed
    positions". For one sender ``(r,s)``, a receiver inside group ``r`` sees the
    transfer ``kappa V^{a_r}_s`` of eq:spatial-transfers, while a receiver in
    another group sees ``gamma * V^a_r V^{a_r}_s`` -- a different effective value
    map for the same sender -- and that ``gamma`` is a product of three Gaussians in
    the intervening configurations, not the Gaussian of the two exposed placements.
    The family supplies elementary operations in a closed operad; it is not itself
    closed."""
    a, a1, a2 = spatial_part(2, 111), spatial_part(2, 112), spatial_part(2, 113)
    b = substitute(a, [a1, a2])

    r = np.random.default_rng(114)
    q = jnp.asarray(r.standard_normal(b.P.Q_dim) * 0.6)
    T = b.transfers(q)
    sender = 3          # child (2,1), i.e. group 2's first child
    inside, outside = 4, 1    # (2,2) in the same group; (1,1) in the other

    V_in = T[inside][sender].L / T[inside][sender].gamma
    V_out = T[outside][sender].L / T[outside][sender].gamma
    assert not np.allclose(V_in, V_out, atol=1e-6)

    # inside the group the sender's own value map survives, exactly as in eq:spatial-transfers
    _, V2 = spatial_split(q[a1.P.Q_dim:a1.P.Q_dim + a2.P.Q_dim], 2, DH, DP)
    np.testing.assert_allclose(V_in, V2[0], atol=1e-12)
    # across groups it is V^a_2 V^{a_2}_1, which is not any single sender's value map
    _, Va = spatial_split(q[-a.P.Q_dim:], 2, DH, DP)
    np.testing.assert_allclose(V_out, Va[1] @ V2[0], atol=1e-12)

    # nor is the weight multiplier the Gaussian kernel of the exposed placements
    ell = b.placement(q)
    assert abs(float(T[outside][sender].gamma)
               - float(spatial_kappa(ell[outside - 1], ell[sender - 1], TAU))) > 1e-3
