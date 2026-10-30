"""Teams of teams, run in time (book plan P87; Ch 4 ``sec.teams_learn``, the hearing-teams suboperad).

**The question** (David, 2026-10-30, ccb6b0fc, from an outside review: "an example where the compositional formulation
reveals something surprising that would be difficult to see from the standard equations"). When does a team of teams,
each group learning whom to trust on its own residual, come to rest where the flat team does?

**The arrangements.** Four members, each a box filled by a clock part that reads out ``msg(y_a[floor t])``, that day's
report (as in ``test_trust_in_time``). Two groups, ``A = {1, 2}`` and ``B = {3, 4}``: each a hearing team
``R, R -> R`` that hears its members at ``exp(-q^2/2)``, delivers nothing to them, is delivered what comes in at
``beta = 1``, and has the one residual ``m_0 - sum_j delta_j m_j`` (``eqn.trust_day_cost`` with two members). The
team hears its groups at ``exp(-q^2/2)`` and delivers what comes in to each at ``beta = 1``, with the same residual.
The nested team is ``team . (A (x) B)`` in arr, its configuration the product, its forcing the sum of the three costs'
pulled back (``eqn.predictive_composite``): a group's trust-levels feel the team's residual through the force that
comes back on the group's readout, and their own. The flat team is one hearing team over the four members. Each is
closed by a world whose clock delivers ``msg(z[floor t])``, and run by ``closed_step``. A day lasts duration 1.

**Predictions, written before the run** (from ``notes/runs/p87-bb8976b6/`` in the book repo, the exact second-moment
rest of these same arrangements; errors of mean square ``Sigma``, ``z ~ N(0, 1)``):
(1) uncorrelated, precisions ``(4, 1 | 1, 1/4)``: the nested team comes to rest where the flat team does, proportions
    ``(0.64, 0.16, 0.16, 0.04)``, trust ``team (0.815, 0.259)``, ``A (0.727, 0.182)``, ``B (0.571, 0.143)``.
(2) members 1 and 3 (one in each group) share an error of mean square ``1/2``: the flat team rests at
    ``(0.427, 0.373, 0.107, 0.093)``, the nested team at ``(0.437, 0.357, 0.147, 0.059)``, trust ``team (0.810, 0.268)``,
    ``A (0.453, 0.370)``, ``B (0.462, 0.185)``: group B, which cannot see that member 3's error is member 1's, trusts
    member 3 more than the flat team does, and the nested readout's value has mean squared error ``0.381`` against
    the flat team's ``0.373``.
Each can fail; (2)'s error gap is near the sampling noise of 10000 days, so the proportions are the test.

**What happened** (2026-10-30, 6 s, passes; seed 0, duration 1/4). Both held. (1) Nested trust over the last half
``team (0.816, 0.258)``, ``A (0.719, 0.189)``, ``B (0.570, 0.143)``; proportions nested ``(0.634, 0.167, 0.159,
0.040)``, flat ``(0.635, 0.167, 0.159, 0.040)``; readout squared error ``0.165`` each. (2) Nested ``(0.432, 0.370,
0.140, 0.058)``, flat ``(0.421, 0.386, 0.102, 0.091)``; readout squared error nested ``0.390``, flat ``0.382``.
Reasonable and weird (the agent's judgment, not evidence): reasonable for desks that each weight their own analysts
under a head who weights the desks, two analysts on different desks reading the same newsletter; weird that the gap in
error is small at these numbers, and that the groups also feel the team's residual, so what is lost is a compromise.
Recorded in the book repo's ``notes/runs/p87-bb8976b6/RESULTS.md``.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.attention import (
    attention_arrangement, identity_transfer, input_dim, msg_dim, transfer, zero_transfer,
)
from arrangements.coalgebra import configuration
from arrangements.functors import closed_step
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_trust_in_time import EMPTY, member, world

jax.config.update("jax_enable_x64", True)

DAYS = 20000
MU = 0.01
VARS = np.array([0.25, 1.0, 1.0, 4.0])


def days(shared=0.0, n=DAYS, seed=0):
    """``z ~ N(0, 1)``; ``y_a = z + eps_a`` with independent parts of mean square ``VARS``, and, when ``shared > 0``, an
    error of mean square ``shared`` common to members 1 and 3."""
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n)
    eps = np.sqrt(VARS)[:, None] * rng.standard_normal((4, n))
    eta = np.sqrt(shared) * rng.standard_normal(n)
    eps[0] += eta
    eps[2] += eta
    return z, z[None, :] + eps


def residual_cost(n):
    """Half the squared norm of ``m_0 - sum_j exp(-q_j^2/2) m_j``, ``m_0`` the message delivered to the team."""
    def cost(q, m_out, n_in):
        pred = (jnp.exp(-q ** 2 / 2)[:, None] * m_out.reshape(n, 2)).sum(0)
        d = n_in[:2] - pred
        return 0.5 * d @ d
    return cost


def hearing_team(n, label, init):
    """A hearing team ``R, ..., R -> R``: hears box ``i`` at ``exp(-q_i^2/2) C(id)``, delivers what comes in to each
    box at ``beta = 1``, nothing else."""
    Z, I = zero_transfer(1), identity_transfer(1)

    def T(q):
        g = jnp.exp(-q ** 2 / 2)
        row0 = [Z] + [transfer(g[i], g[i] * jnp.eye(1)) for i in range(n)]
        return tuple([tuple(row0)] + [tuple([I] + [Z] * n) for _ in range(n)])

    return attention_arrangement(
        n, 1, 1, placement=lambda q: jnp.zeros((n, 1)), transfers=T, cost=residual_cost(n),
        parameter=configuration(euclidean(n, MU), init=jnp.asarray(init, float)), label=label)


def nested(Y):
    ms = [member(Y[a], f"member{a + 1}") for a in range(4)]
    A = compose_seq(tensor_arrangements(ms[:2]), hearing_team(2, "A", [1.0, 1.0]).arr())
    B = compose_seq(tensor_arrangements(ms[2:]), hearing_team(2, "B", [1.0, 1.0]).arr())
    return compose_seq(tensor_arrangements([A, B]), hearing_team(2, "team", [1.0, 1.0]).arr())


def flat(Y):
    ms = [member(Y[a], f"member{a + 1}") for a in range(4)]
    return compose_seq(tensor_arrangements(ms), hearing_team(4, "flat", [1.0] * 4).arr())


def run(team, z, duration=0.25):
    system = compose_seq(team, world(z, team))
    step = closed_step(system, duration=duration)
    per_day = int(round(1 / duration))
    k = team.P.Q_dim
    readout = jax.jit(lambda q: team.out_f(q[:k], EMPTY))

    @jax.jit
    def one_day(s, _):
        q = system.P.position(s)
        r = readout(q)
        s = jax.lax.fori_loop(0, per_day, lambda _i, t: step(t), s)
        return s, (q, r)

    _, (qs, rs) = jax.lax.scan(one_day, system.P.init, None, length=DAYS)
    return system, np.asarray(qs), np.asarray(rs)


def trust(q):
    return np.exp(-np.asarray(q) ** 2 / 2)


def report(name, z, team, expect_props, expect_trust=None):
    system, qs, rs = run(team, z)
    print(f"\n{name}: {system.label}, configuration dimension {system.P.Q_dim}")
    last = slice(DAYS // 2, None)
    q_mean = qs[last].mean(0)
    clocks = q_mean[[0, 1, 4, 5]] if expect_trust is not None else q_mean[:4]
    if expect_trust is not None:
        # configuration order: A's members' clocks, A's q (2), B's clocks, B's q (2), team's q (2), world's clock
        gA, gB, gT = trust(q_mean[2:4]), trust(q_mean[6:8]), trust(q_mean[8:10])
        props = np.concatenate([gT[0] * gA, gT[1] * gB])
        print(f"  trust over the last half: team {np.round(gT, 3)}, A {np.round(gA, 3)}, B {np.round(gB, 3)}; "
              f"predicted {expect_trust}")
    else:
        g = trust(q_mean[4:8])
        props = g
        print(f"  trust over the last half {np.round(g, 3)}")
    props = props / props.sum()
    value = rs[:, 0] / rs[:, 1]
    err = np.mean((value[last] - z[last]) ** 2)
    print(f"  proportions {np.round(props, 3)}, predicted {expect_props}; readout value squared error {err:.4f}")
    assert np.allclose(clocks, np.arange(DAYS)[last].mean(), atol=1), "the members' clocks keep the days"
    assert np.allclose(props, expect_props, atol=0.03), name
    return props, err


def test_teams_of_teams_in_time():
    z, Y = days(0.0)
    p_n, e_n = report("(1) nested, uncorrelated", z, nested(Y), [0.64, 0.16, 0.16, 0.04],
                      "team (0.815, 0.259), A (0.727, 0.182), B (0.571, 0.143)")
    p_f, e_f = report("(1) flat, uncorrelated", z, flat(Y), [0.64, 0.16, 0.16, 0.04])
    assert np.allclose(p_n, p_f, atol=0.02)
    z, Y = days(0.5)
    p_n, e_n = report("(2) nested, members 1 and 3 share an error", z, nested(Y), [0.437, 0.357, 0.147, 0.059],
                      "team (0.810, 0.268), A (0.453, 0.370), B (0.462, 0.185)")
    p_f, e_f = report("(2) flat, members 1 and 3 share an error", z, flat(Y), [0.427, 0.373, 0.107, 0.093])
    print(f"\n(2) member 3's share: nested {p_n[2]:.3f}, flat {p_f[2]:.3f}; readout squared error nested {e_n:.4f}, "
          f"flat {e_f:.4f}")
    assert p_n[2] > p_f[2] + 0.02 and p_n[3] < p_f[3] - 0.02
