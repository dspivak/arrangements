"""A prediction error moves a reader toward a source (the attention paper).

The mathematics is ``background/affine_quadratic_operad.tex`` in the book repository,
cited by its own labels (eq:spatial-transfers, eq:prediction, eq:reader-cost,
eq:reader-force). **Every docstring below says which of three things its test does** --
reproduce a closed form the paper proves, check a discrete-time analogue the book still
has to prove (STATUS.md §3.2), or merely illustrate (STATUS.md §3.2).

**This example is a local response calculation, not a convergence claim, and the reader's
private prediction is held fixed throughout.** The paper says so in as many words ("This
is a local response calculation, not a convergence claim"), and the book, Ch. 5 says
not to restate it as convergence or as "the reader moves". Accordingly no test here runs
to a limit and none is named for convergence; the one-step tests check the *initial*
response at the stated configuration and nothing beyond it.

The arrangement is ``H = W = P = R``, three children, eq:spatial-transfers with identity
value maps, zero arrangement cost. The children are components ``I -> B`` (plain
``Arrangement``s on the monoidal unit interface, as in tests/test_rendezvous.py): a reader
sending ``0_M`` and carrying the private input prediction cost of eq:prediction, and two
sources sending ``(1, 1)`` and ``(0, 1)`` with zero cost. eq:reader-cost is then what
substitution -- composition in ``arr`` -- produces; it is written here only as the closed
form the composite is checked against.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact
from arrangements.attention import (
    fixed_prediction, input_dim, make_input, message, msg_dim, prediction_cost,
    spatial_arrangement, spatial_config, spatial_kappa, zero_cost,
)
from arrangements.coalgebra import configuration
from arrangements.rvect import constant, trivial
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_rendezvous import run_phi

DH, DP = 1, 1
MDIM, EDIM = msg_dim(DH), input_dim(DH, DP)
TAU = 0.8                                       # the Gaussian scale of eq:spatial-transfers
D = float(np.sqrt(2.0 * TAU * np.log(2.0)))     # the paper's d = sqrt(2 tau log 2)
E_HAT = jnp.array([1.0, 1.0, 0.0])              # ehat_1 = ((1, 1), 0), held fixed
EXT = make_input(message(0.0, 0.0), 0.0)        # the exterior input, fixed to (0_M, 0)


# ---------------------------------------------------------------------------
# The three components and the spatial arrangement.
# ---------------------------------------------------------------------------


def component(m, cost, label):
    """A component ``I -> B``: it sends the fixed message ``m in M``, receives an input
    ``e in E``, and carries the cost ``cost(e)``; no configuration of its own."""
    return Arrangement(
        P=configuration(trivial()),
        in_dim_M=0, out_dim_M=0, in_dim_N=EDIM, out_dim_N=MDIM,
        out_f=lambda q, m_out: m,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -cost(n_in)),
        label=label,
    )


def reader():
    """The reader: message ``0_M``, cost ``k_R(e_1) = (1/2)||e_1 - ehat_1||^2``
    (eq:prediction) with the private prediction ``ehat_1 = ((1, 1), 0)`` *held fixed*."""
    k = prediction_cost(fixed_prediction(E_HAT))
    return component(message(0.0, 0.0),
                     lambda e: k(jnp.zeros(0), jnp.zeros(0), e), "reader")


def source(h, label):
    """A source: message ``(h, 1)``, zero cost."""
    return component(message(h, 1.0), lambda e: jnp.asarray(0.0), label)


def reader_system(mobilities, positions=(0.0, D, -D), values=(1.0, 1.0, 1.0)):
    """``a(reader, source_1, source_0) : I -> B``, by composition in ``arr``.

    ``a`` is the three-ary spatial arrangement of eq:spatial-transfers with zero cost; its
    configuration is ``q = (p_1, p_2, p_3, V_1, V_2, V_3)`` (eq:spatial-state), the value
    maps being ``1 x 1`` matrices here. ``mobilities`` is the diagonal of the response on
    that ``Q``: the paper's "positive scalar Euclidean mobility on each position
    coordinate", with the value maps given mobility ``0`` because the paper's calculation
    holds them at the identity. See ``test_the_forcing_also_pushes_the_value_maps``.
    """
    a = spatial_arrangement(
        3, DH, DP, TAU, cost=zero_cost,
        parameter=configuration(constant(jnp.diag(jnp.asarray(mobilities, float))),
                                init=spatial_config(positions, values)),
        label="spatial(3)")
    return compose_seq(
        tensor_arrangements([reader(), source(1.0, "src_1"), source(0.0, "src_0")]),
        a.arr())


def k_comp(p1, p2, p3):
    """eq:reader-cost, ``(1/2)[(g_2 - 1)^2 + (g_2 + g_3 - 1)^2 + p_1^2]``, the closed form
    the composite is checked against. Nothing builds the composite from it."""
    g2 = jnp.exp(-(p1 - p2) ** 2 / (2.0 * TAU))
    g3 = jnp.exp(-(p1 - p3) ** 2 / (2.0 * TAU))
    return 0.5 * ((g2 - 1.0) ** 2 + (g2 + g3 - 1.0) ** 2 + p1 ** 2)


def positions_of(q):
    return np.asarray(q)[:3]


# ---------------------------------------------------------------------------
# What the reader receives, and the cost substitution produces.
# ---------------------------------------------------------------------------


def test_the_reader_receives_the_papers_input():
    """REPRODUCES A CLOSED FORM the paper states: with identity value maps, the exterior
    input ``(0_M, 0)`` and source messages ``m_2 = (1, 1)``, ``m_3 = (0, 1)``, the
    receiving map of eq:maps under eq:spatial-transfers delivers to the reader

        e_1 = ((g_2, g_2 + g_3), p_1),     g_j = exp(-(p_1 - p_j)^2 / 2 tau).

    The signal is the weighted total ``g_2 * 1 + g_3 * 0`` and the weight is the total
    ``g_2 + g_3``; both stay on the interface, no mean readout is taken (eq:mean is a
    separate operation, and the paper notes that folding it into a cost would leave the
    quadratic family)."""
    a = spatial_arrangement(3, DH, DP, TAU, cost=zero_cost,
                            init=spatial_config((0.0, D, -D), (1.0, 1.0, 1.0)))
    q = spatial_config((0.0, D, -D), (1.0, 1.0, 1.0))
    m = jnp.concatenate([message(0.0, 0.0), message(1.0, 1.0), message(0.0, 1.0)])
    e1 = a.received(q, m, EXT, 1)

    g2 = float(spatial_kappa(jnp.array([0.0]), jnp.array([D]), TAU))
    g3 = float(spatial_kappa(jnp.array([0.0]), jnp.array([-D]), TAU))
    np.testing.assert_allclose([g2, g3], [0.5, 0.5], atol=1e-14)
    np.testing.assert_allclose(e1, make_input(message(g2, g2 + g3), 0.0), atol=1e-14)


def test_reader_cost_emerges_from_substitution():
    """REPRODUCES A CLOSED FORM the paper proves (eq:reader-cost, via eq:composite-cost):
    "Substitution pulls the reader's cost back to the following cost on the arrangement's
    positions",

        k_comp(p_1, p_2, p_3) = (1/2)[(g_2 - 1)^2 + (g_2 + g_3 - 1)^2 + p_1^2].

    The composite is ``compose_seq(tensor(reader, src, src), a.arr())`` and its forcing is
    pullback-and-sum; ``k_comp`` is only the closed form it is compared with. The
    positional components of ``-dk_comp`` are checked at random configurations, so the
    identity is the whole function and not one value."""
    b = reader_system([1.0] * 3 + [0.0] * 3)
    grad_k = jax.grad(k_comp, argnums=(0, 1, 2))
    r = np.random.default_rng(301)
    for _ in range(6):
        p = r.standard_normal(3) * 0.8
        q = spatial_config(p, (1.0, 1.0, 1.0))
        xi_q, _, _ = b.omega(q, jnp.zeros(0), EXT)
        expected = [-float(t) for t in grad_k(*[float(v) for v in p])]
        np.testing.assert_allclose(np.asarray(xi_q)[:3], expected, atol=1e-13)


def test_positional_forcing_at_the_stated_configuration():
    """REPRODUCES A CLOSED FORM the paper proves (eq:reader-force): with
    ``d = sqrt(2 tau log 2)``, at ``(p_1, p_2, p_3) = (0, d, -d)`` both ``g_2`` and
    ``g_3`` equal ``1/2`` and

        -d k_comp |_(0, d, -d) = (d / 4 tau) (d p_1 - d p_2).

    In particular the force on the third child -- the source of signal ``0`` -- is exactly
    zero at this configuration. This is a statement about one configuration: a local
    response, not a convergence claim, with the reader's prediction held fixed."""
    b = reader_system([1.0] * 3 + [0.0] * 3)
    q = spatial_config((0.0, D, -D), (1.0, 1.0, 1.0))
    xi_q, _, xi_e = b.omega(q, jnp.zeros(0), EXT)

    c = D / (4.0 * TAU)
    np.testing.assert_allclose(np.asarray(xi_q)[:3], [c, -c, 0.0], atol=1e-14)
    assert abs(float(xi_q[2])) < 1e-15          # exactly zero on the other source


def test_one_step_brings_the_reader_and_the_source_together():
    """CHECKS THE DISCRETE ANALOGUE (STATUS.md §3.2) of the paper's next
    sentence: "With positive scalar Euclidean mobility on each position coordinate, this
    force initially moves the reader and the source of signal ``1`` toward one another
    ... The reader's attention to the source of signal ``1`` therefore increases."

    ONE step, because the claim is about the initial response. ``|p_1 - p_2|`` decreases
    and ``kappa_12`` increases; the third child does not move at all. Nothing is asserted
    about where the run goes afterwards, and the reader's prediction ``ehat_1`` stays
    fixed."""
    mu = 0.05
    b = reader_system([mu] * 3 + [0.0] * 3)
    tr = run_phi(b, 1, EXT)
    p0, p1 = positions_of(tr[0]), positions_of(tr[1])

    assert abs(p1[0] - p1[1]) < abs(p0[0] - p0[1])
    k0 = float(spatial_kappa(jnp.array([p0[0]]), jnp.array([p0[1]]), TAU))
    k1 = float(spatial_kappa(jnp.array([p1[0]]), jnp.array([p1[1]]), TAU))
    assert k1 > k0
    np.testing.assert_allclose(p1[2], p0[2], atol=1e-15)
    np.testing.assert_allclose(p1[0] - p0[0], mu * D / (4.0 * TAU), atol=1e-13)
    np.testing.assert_allclose(p1[1] - p0[1], -mu * D / (4.0 * TAU), atol=1e-13)


def test_sources_with_zero_response_leave_the_reader_all_the_moving():
    """CHECKS THE DISCRETE ANALOGUE of the book's own observation, which the paper does
    not make (the book, Ch. 5: "a source with zero response does not move, so the
    reader does all the moving; that is a book"). The forcing is the same one-form -- the
    sources still receive their covectors -- and only the integrator differs: mobility
    ``0`` on ``p_2`` and ``p_3``. The reader's own displacement is unchanged by the
    change, since the forcing at the initial configuration is unchanged, and the two
    sources stay exactly where they were."""
    mu = 0.05
    free = run_phi(reader_system([mu] * 3 + [0.0] * 3), 1, EXT)
    frozen = run_phi(reader_system([mu, 0.0, 0.0] + [0.0] * 3), 1, EXT)

    np.testing.assert_allclose(positions_of(frozen[1])[1:], positions_of(frozen[0])[1:],
                               atol=1e-15)
    np.testing.assert_allclose(positions_of(frozen[1])[0], positions_of(free[1])[0],
                               atol=1e-15)
    assert not np.allclose(positions_of(free[1])[1], positions_of(free[0])[1])


def test_the_forcing_also_pushes_the_value_maps():
    """ILLUSTRATES a place the executable arrangement is larger than the paper's display,
    which the book will have to say when it states this example. eq:reader-cost is a cost
    "on the arrangement's positions", but the configuration of eq:spatial-state is
    ``Q_3 = P^3 x End(H)^3``, and the reader's received signal is ``kappa_12 V_2 h_2``, so
    ``-dk_comp`` has a nonzero component on ``V_2`` as well: at ``(0, d, -d)`` it is
    ``(1 - g_2) g_2 = 1/4``. The paper's calculation holds the value maps at the identity;
    here that is a choice of integrator (mobility ``0`` on the ``End(H)`` block), not an
    absence of force. The components on ``V_1`` and ``V_3`` do vanish -- the reader sends
    ``0_M`` and the second source's signal is ``0``."""
    b = reader_system([1.0] * 3 + [0.0] * 3)
    q = spatial_config((0.0, D, -D), (1.0, 1.0, 1.0))
    xi_q, _, _ = b.omega(q, jnp.zeros(0), EXT)
    xi_V = np.asarray(xi_q)[3:]

    np.testing.assert_allclose(xi_V, [0.0, 0.25, 0.0], atol=1e-14)
    assert float(abs(xi_V[1])) > 1e-6

    # released, the value map does move, so it is the integrator that freezes it
    moved = run_phi(reader_system([0.0] * 3 + [0.05] * 3), 1, EXT)
    assert abs(float(moved[1][4] - moved[0][4])) > 1e-6
