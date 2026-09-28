"""A jointed arm pulled at its tip by springs trains itself (book sec. 1.5; David, 2026-09-27,
of the pinball picture: "it would be even more interesting if it was just *attracted* to that
or forced into that tuning somehow").

The picture. A planar arm of two stiff links, lengths L1 and L2, with joint angles q1, q2:
those are the parameters. An input x turns the base, so the tip sits at
p(q, x) = L1 u(x + q1) + L2 u(x + q1 + q2), u(a) = (cos a, sin a). For each training example
(x_j, t_j) there is one copy of the arm, its base turned by x_j, and all the copies' joints are
geared together, so they share q. A zero-length spring ties each copy's tip to its target t_j.
The joints sit in syrup: each joint turns at a rate proportional to the torque on it.

Nobody computes a gradient. The torque a force F at the tip puts on the joints is J^T F, where
J = dp/dq (virtual work), and the spring's force is F = -k (p - t) = -grad of (k/2)|p - t|^2.
So the torque is minus the gradient of the squared miss, carried back by the transpose, which
is what backpropagation computes.

Caveat found after the run (Claude, 2026-09-27): an input that only turns the base makes every
copy a rotated copy of the same arm, so the three examples below carry the information of one
(each miss is |p(q) - R(-x_j) t_j|^2). Tests (1) and (3) do not depend on this; (2) recovers
the true angles because two knobs meet two numbers. The 100-joint tests use an input that bends
every joint instead.

Predictions (Claude Opus 5.5 (1M), 2026-09-27), written with the code, before any run:

(1) The torque the springs put on each geared joint, computed as forces transmitted through
    the links (J^T F, J by finite differences of the tip position), equals minus the gradient
    of the total squared miss (by autodiff), to within finite-difference error.
(2) In syrup, starting away from it, the arm settles on joint angles that put every tip on
    its target when the targets were made by a true setting q*: the total miss goes to ~0.
(3) With targets no single setting can reach, it settles where the spring forces balance at
    the joints, which is the least total squared miss among nearby settings, not zero.

The 100-joint arm (David, 2026-09-27: "could you try a 100 joint arm or would it not work?").
The same arm with 100 links of length 0.02 (total length 2), so 100 parameters; targets made
by a true setting q* of small random bends. Predictions, written before this code and run:

(4) The torque through the links still equals minus the gradient (it is a theorem at any size).
(5) With 3 examples (6 numbers to hit, 100 knobs) the arm reaches every target, the miss near
    zero, but ends far from q*: many settings hit the targets, and it settles on one of them.
(6) With 60 examples (120 numbers, more than 100 knobs) the settings that hit every target
    are few, and descent may stall short of them in a wrong pose; I predict the miss falls by
    at least a factor of 100 from the straight start and do not predict that it reaches zero.

Verdicts (first run with the base-turning input, then rerun with the input bending each joint
by x s_k, s fixed and small, and the step set to 1/(largest curvature)). (4) holds, to 1e-10.
(5) holds: the miss falls about 1e5-fold (0.48 to 4e-6 in 100,000 steps, still creeping) and
the arm ends as far from q* as q* is from straight. (6) refuted in its number: in 100,000 steps
the miss fell about 11-fold from straight and about 63-fold from a slightly bent start, still
falling slowly, while the arm drifted away from q*. The straight start is slow for a reason a
reader can see: pulling a straight arm along its own length puts no twist on any joint. The
slow crawl is ill-conditioning: the step is set by the stiffest direction, and the softest
ones barely move, which is the trouble momentum is used for (not run here).

Random starts and momentum (David, 2026-09-27: "we should start the arms in random position.
and let's put in momentum"). Each start draws every joint angle uniformly from [-pi, pi], a
crumpled arm; 60 examples; 20,000 steps each, so every run feels the same number of pulls.
Syrup steps at 1/(largest curvature at the start). Momentum is the arm in air with friction
(unit mass, semi-implicit Euler at step 1/sqrt(largest curvature)), keeping a fraction beta
of its velocity each step. Predictions, written before this code and run:

(7) In syrup from a crumpled start the miss falls but crawls, as from near straight.
(8) With momentum at beta = 0.9 the final miss is at least 10 times lower than syrup's from
    the same start, on every one of three starts.
(9) At beta = 0.99 it is lower than syrup's too; I do not predict whether lower than 0.9's.
(10) The three starts end in three different poses, none of them the setting that made the
    targets.

Verdicts. The first run, with the step fixed from the start's stiffness, refuted (8): momentum
ended worse than syrup. The fault was the construction's, not momentum's: as a crumpled arm
unfolds it stiffens (curvature 870 against 155 at the start), the fixed step became too long
for the stiffest direction, and the miss rose even in syrup, which true syrup never does. With
the step reset to half of 1/(current stiffness) every 500 steps, all four hold, on each start:
syrup 64 -> 2.5 and still crawling; beta 0.9 about 50 times lower; beta 0.99 about 1,800 times
lower than syrup (it is also lower than 0.9's); three different final poses.
Weird, and an arm's alone: at beta 0.99 some joints whirl through many full turns on the way
(18 on one start) before settling, since a joint angle is periodic and momentum carries it
round; a network's parameters are not periodic, so this has no counterpart there.
"""

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

L1, L2, K = 1.0, 0.7, 1.0


def _tip(q, x):
    a1 = x + q[0]
    a2 = a1 + q[1]
    return jnp.array([L1 * jnp.cos(a1) + L2 * jnp.cos(a2), L1 * jnp.sin(a1) + L2 * jnp.sin(a2)])


def _miss(q, xs, ts):
    return sum(0.5 * K * jnp.sum((_tip(q, x) - t) ** 2) for x, t in zip(xs, ts))


def _torque_through_links(q, xs, ts, eps=1e-6):
    """Sum over the geared copies of J^T F, J from the tip's motion alone."""
    q = np.asarray(q, float)
    total = np.zeros(2)
    for x, t in zip(xs, ts):
        p = np.asarray(_tip(q, x))
        J = np.column_stack([
            (np.asarray(_tip(q + eps * e, x)) - np.asarray(_tip(q - eps * e, x))) / (2 * eps)
            for e in np.eye(2)])
        F = -K * (p - np.asarray(t))
        total += J.T @ F
    return total


def _examples(q_true, xs):
    return [np.asarray(_tip(jnp.asarray(q_true), x)) for x in xs]


XS = [0.0, 0.8, -0.5]


def test_torque_through_the_links_is_minus_the_gradient():
    rng = np.random.default_rng(1)
    ts = _examples([0.4, 1.1], XS)
    grad = jax.grad(_miss)
    for _ in range(5):
        q = rng.uniform(-2, 2, size=2)
        np.testing.assert_allclose(_torque_through_links(q, XS, ts),
                                   -np.asarray(grad(jnp.asarray(q), XS, ts)), atol=1e-7)


def test_in_syrup_the_arm_pulls_itself_onto_reachable_targets():
    q_true = np.array([0.4, 1.1])
    ts = _examples(q_true, XS)
    q, dt = np.array([-0.3, 0.2]), 0.05
    for _ in range(4000):
        q = q + dt * _torque_through_links(q, XS, ts)
    assert float(_miss(jnp.asarray(q), XS, ts)) < 1e-12
    np.testing.assert_allclose(q, q_true, atol=1e-5)


def test_unreachable_targets_settle_where_the_forces_balance():
    ts = [np.array([1.5, 0.2]), np.array([0.1, 1.2]), np.array([-0.4, -0.9])]
    q, dt = np.array([-0.3, 0.2]), 0.05
    for _ in range(4000):
        q = q + dt * _torque_through_links(q, XS, ts)
    np.testing.assert_allclose(_torque_through_links(q, XS, ts), 0.0, atol=1e-8)
    m = float(_miss(jnp.asarray(q), XS, ts))
    assert m > 1e-3
    rng = np.random.default_rng(2)
    for _ in range(20):
        dq = 1e-2 * rng.standard_normal(2)
        assert float(_miss(jnp.asarray(q + dq), XS, ts)) >= m - 1e-12


N_LINKS = 100
LINK = 2.0 / N_LINKS


S_BEND = jnp.asarray(0.05 * np.random.default_rng(0).standard_normal(N_LINKS))


def _tip_n(q, x):
    a = jnp.cumsum(q + x * S_BEND)
    return LINK * jnp.array([jnp.sum(jnp.cos(a)), jnp.sum(jnp.sin(a))])


def _miss_n(q, xs, ts):
    ps = jax.vmap(lambda x: _tip_n(q, x))(xs)
    return 0.5 * K * jnp.sum((ps - ts) ** 2)


def _torque_n(q, xs, ts):
    """Sum over the geared copies of J^T F, J the tip's motion under each joint."""
    def one(x, t):
        J = jax.jacfwd(_tip_n)(q, x)
        return J.T @ (-K * (_tip_n(q, x) - t))
    return jnp.sum(jax.vmap(one)(xs, ts), axis=0)


def _setup_n(n_examples, seed):
    rng = np.random.default_rng(seed)
    q_true = jnp.asarray(0.1 * rng.standard_normal(N_LINKS))
    xs = jnp.asarray(np.linspace(-3.0, 3.0, n_examples))
    ts = jax.vmap(lambda x: _tip_n(q_true, x))(xs)
    return q_true, xs, ts


def _syrup_n(q, xs, ts, steps):
    lmax = float(jnp.max(jnp.linalg.eigvalsh(jax.hessian(_miss_n)(q, xs, ts))))
    dt = 1.0 / lmax
    body = lambda _, qq: qq + dt * _torque_n(qq, xs, ts)
    return jax.jit(lambda q0: jax.lax.fori_loop(0, steps, body, q0))(q)


def test_hundred_joints_torque_is_minus_the_gradient():
    q_true, xs, ts = _setup_n(60, 3)
    q = q_true + 0.3
    np.testing.assert_allclose(np.asarray(_torque_n(q, xs, ts)),
                               -np.asarray(jax.grad(_miss_n)(q, xs, ts)), atol=1e-10)


def test_hundred_joints_three_examples_reach_but_not_the_true_setting():
    q_true, xs, ts = _setup_n(3, 4)
    q0 = jnp.zeros(N_LINKS)
    q = _syrup_n(q0, xs, ts, 20_000)
    miss0, miss = float(_miss_n(q0, xs, ts)), float(_miss_n(q, xs, ts))
    print(f"\n3 examples: miss {miss0:.3e} -> {miss:.3e}; |q - q*| = "
          f"{float(jnp.linalg.norm(q - q_true)):.3f}, |q*| = {float(jnp.linalg.norm(q_true)):.3f}")
    assert miss < miss0 * 1e-3
    assert float(jnp.linalg.norm(q - q_true)) > 0.5 * float(jnp.linalg.norm(q_true))


def test_hundred_joints_sixty_examples_miss_falls_slowly():
    q_true, xs, ts = _setup_n(60, 5)
    q0 = jnp.asarray(0.02 * np.random.default_rng(9).standard_normal(N_LINKS))
    q = _syrup_n(q0, xs, ts, 20_000)
    miss0, miss = float(_miss_n(q0, xs, ts)), float(_miss_n(q, xs, ts))
    print(f"\n60 examples: miss {miss0:.3e} -> {miss:.3e}; |q - q*| = "
          f"{float(jnp.linalg.norm(q - q_true)):.3f}, |q*| = {float(jnp.linalg.norm(q_true)):.3f}")
    assert miss < miss0 / 10


_STIFF = jax.jit(lambda q, xs, ts: jnp.max(jnp.abs(jnp.linalg.eigvalsh(
    jax.hessian(_miss_n)(q, xs, ts)))))


def _momentum_body(xs, ts):
    def body(_, state):
        q, v, lr, beta = state
        v = beta * v + lr * _torque_n(q, xs, ts)
        return q + v, v, lr, beta
    return body


def _run_adaptive(q, xs, ts, beta, chunks=40, per_chunk=500):
    """beta = 0: syrup; beta > 0: in air with friction keeping beta of the velocity per step.
    The step is half of 1/(current stiffness), reset every chunk, so the discrete run stays
    faithful to the continuous one as the arm stiffens."""
    run = jax.jit(lambda st: jax.lax.fori_loop(0, per_chunk, _momentum_body(xs, ts), st))
    v = jnp.zeros_like(q)
    for _ in range(chunks):
        lr = 0.5 / float(_STIFF(q, xs, ts))
        q, v, _, _ = run((q, v, lr, beta))
    return q


def test_hundred_joints_random_starts_with_and_without_momentum():
    q_true, xs, ts = _setup_n(60, 5)
    finals = []
    for seed in (11, 12, 13):
        q0 = jnp.asarray(np.random.default_rng(seed).uniform(-np.pi, np.pi, N_LINKS))
        m0 = float(_miss_n(q0, xs, ts))
        q_s, q_9, q_99 = (_run_adaptive(q0, xs, ts, b) for b in (0.0, 0.9, 0.99))
        ms, m9, m99 = (float(_miss_n(q, xs, ts)) for q in (q_s, q_9, q_99))
        print(f"\nstart {seed}: miss {m0:.3e}; syrup {ms:.3e}; beta 0.9 {m9:.3e}; "
              f"beta 0.99 {m99:.3e}; |q99 - q*| = {float(jnp.linalg.norm(q_99 - q_true)):.2f}")
        finals.append(q_99)
        assert ms < m0
        assert m9 < ms / 10
        assert m99 < ms
    # a joint angle is only defined up to a full turn, so poses are compared mod 2 pi
    pose = lambda a, b: float(jnp.linalg.norm(jnp.angle(jnp.exp(1j * (a - b)))))
    for i in range(3):
        print(f"pose {i}: {pose(finals[i], q_true):.2f} from q*; "
              f"{float(jnp.max(jnp.abs(finals[i] - q_true)) / (2 * np.pi)):.1f} turns at most")
        assert pose(finals[i], q_true) > 0.1
        for j in range(i):
            assert pose(finals[i], finals[j]) > 0.1
