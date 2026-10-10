"""A scene of two figurines, the coordinator a market over the one-part-altered scenes (book plan P37 (b)(iii′)
stage (B); ``arrangements/market_scene.py``, stage (B) section).

Predictions fixed before any code for the stage, in the book repo's
``notes/runs/p37-market-of-scenes-3819a169/PREDICTIONS.md``, stage (B) (Claude Fable 5.1, 3819a169, 2026-10-08),
their numbers from ``B_by_hand.log``; results in the same folder's ``RESULTS.md``. The closed form the nets are
checked against is ``market_scene.scene_closed_form`` (the scene's recurrence and the trust's Bayes steps in numpy,
the pushes by ``jax.grad`` of the recurrence).

B1. Nested equals flat: the outer trust times the inner equals the flat market's trust at every round, to 1e-12,
    for the three evidence kinds (the compositor's equation of the prediction market dynamic operad).
B2. Stage (ii)'s kinds return with nothing supplied: trust in the cat's family after rounds 1–6 is
    (0.590, 0.620, 0.629, 0.617, 0.628, 0.695) with the outcome alone, (0.869, 0.992, 0.998, 1.000, 1.000, 1.000)
    with the sighting, (0.677, 0.805, 0.878, 0.921, 0.954, 0.979) with comparison; sighting > comparison > outcome.
B3. The flat direction: the two copies of the base scene keep their prior ratio 1 to 1e-12 at every round.
B4. The push routed by trust: bare, theta_s moves -0.2325 and theta_c +0.1258; routed, -0.1335 and +0.0895
    (predicted with the copies un-lagged; the closed form was lagged to match the relays before the test ran, and
    gives -0.1359 and +0.0955: see RESULTS); the right, sensitive part moves about 0.57 of the bare move.
B5. The runner decodes to ``ref_step`` to 1e-10; the library's denotation agrees with ``ref_step`` on a reduced
    instance (T = 2, two values a family).

Run with ``PYTEST_ADDOPTS=-s`` to see the numbers.
"""
import jax
import numpy as np
import pytest

from arrangements import market_scene as S
from arrangements import runner as R

jax.config.update("jax_enable_x64", True)

B_TO_A = [0, 1, 2, 3, 4]


def _softmax(l):
    l = np.asarray(l, float)
    q = np.exp(l - l.max())
    return q / q.sum()


def _run_arm(key, rounds=7):
    """The arm's net coupled to the world by hand; the flat trust and the base log-parameters after each step."""
    arm = S.SCENE_ARMS[key]
    net, wires = S.scene_market_net(arm)
    world = S.scene_world_net()
    mk = [w for w, nd in enumerate(net.nodes) if nd[0] == "market"][0]
    sa, sb = net.init_state(), world.init_state()
    trust, theta = [], []
    for _ in range(rounds):
        sa, sb = R.couple_by_hand(net, world, [], B_TO_A, sa, sb)
        trust.append(_softmax(sa[mk]))
        if arm.push != "none":
            theta.append(np.array([sa[wires["th_s"]], sa[wires["th_c"]]]))
    return dict(net=net, wires=wires, world=world, trust=trust, theta=theta, closed=S.scene_closed_form(arm))


@pytest.fixture(scope="module")
def arms():
    return {key: _run_arm(key) for key in S.SCENE_ARMS}


def test_B0_every_arm_is_its_closed_form(arms):
    """The nets, stepped by hand, equal the by-hand model: trust to 1e-13, parameters to 1e-12."""
    for key, r in arms.items():
        worst_q = max(np.abs(q - c).max() for q, c in zip(r["trust"], r["closed"]["trust"]))
        assert worst_q < 1e-13, key
        if r["theta"]:
            worst_t = max(np.abs(t - c).max() for t, c in zip(r["theta"], r["closed"]["theta"]))
            assert worst_t < 1e-12, key
        print(f"\nB0 {key}: {len(r['net'].nodes)} nodes; net vs closed form, trust {worst_q:.1e}")


def test_B1_nested_equals_flat(arms):
    """The nested market, its likelihoods made by boxes (``nested_market_composed``: the rounds' d_0, the world
    and a likelihood box per scene, wired into the beliefs and the paying box; P43 (b) job 7, de71554a), equals
    the flat closed form to 1e-12, and equals ``nested_market`` fed the numpy likelihoods it replaced."""
    worst = {}
    for key in ("O", "S", "C"):
        nested = S.nested_market_composed(S.SCENE_ARMS[key])
        fed = S.nested_market(S.scene_likelihoods(S.SCENE_ARMS[key]))
        assert max(np.abs(np.asarray(a) - np.asarray(b)).max() for a, b in zip(nested, fed)) < 1e-12
        worst[key] = max(np.abs(np.asarray(a) - b).max() for a, b in zip(nested, arms[key]["closed"]["trust"]))
        assert worst[key] < 1e-12
    print(f"\nB1 nested (library, three belief integrators, the outer paid by aggregates) vs flat: {worst}")


def test_B2_the_three_kinds_return_with_nothing_supplied(arms):
    want = {"O": [0.590, 0.620, 0.629, 0.617, 0.628, 0.695],
            "S": [0.869, 0.992, 0.998, 1.000, 1.000, 1.000],
            "C": [0.677, 0.805, 0.878, 0.921, 0.954, 0.979]}
    got = {}
    for key in ("O", "S", "C"):
        cat = [float(q[3:].sum()) for q in arms[key]["trust"][1:]]
        got[key] = cat
        assert np.round(cat, 3).tolist() == pytest.approx(want[key], abs=0.0015), key
    for t in range(6):
        assert got["S"][t] > got["C"][t] > got["O"][t]
    truth = {key: float(arms[key]["trust"][-1][4]) for key in ("O", "S", "C")}   # the scene (a0, b' = 0.4)
    shown = {k: np.round(v, 3).tolist() for k, v in got.items()}
    print(f"\nB2 trust in the cat's family after rounds 1-6: {shown}")
    print(f"B2 the truth's scene (the cat at 0.4) holds, of the whole trust, at the end: "
          f"{ {k: round(v, 3) for k, v in truth.items()} }")
    assert abs(truth["O"] - 0.679) < 0.002 and abs(truth["S"] - 0.995) < 0.002 and abs(truth["C"] - 0.956) < 0.002


def test_B3_the_flat_direction_is_the_scenes_with_equal_outcomes(arms):
    for key in ("O", "S", "C", "O-bare", "O-routed"):
        ratios = [q[2] / q[3] for q in arms[key]["trust"]]
        assert max(abs(x - 1.0) for x in ratios) < 1e-12, key
    print("\nB3 the base scene's two copies (one in each family) keep their prior ratio 1 at every round, every arm")


def test_B4_the_push_routed_by_trust(arms):
    th0 = arms["O-bare"]["theta"][0]
    bare = arms["O-bare"]["theta"][-1] - th0
    rout = arms["O-routed"]["theta"][-1] - th0
    print(f"\nB4 bare chain rule moved (theta_s, theta_c) by {np.round(bare, 5)}, a,b -> {np.round(np.exp(th0 + bare), 4)}")
    print(f"B4 routed by trust moved them by {np.round(rout, 5)}, a,b -> {np.round(np.exp(th0 + rout), 4)}")
    print(f"B4 ratios routed/bare: snake {rout[0] / bare[0]:.4f}, cat {rout[1] / bare[1]:.4f}; "
          f"the bare share taken by the snake (right) {abs(bare[0]) / (abs(bare[0]) + abs(bare[1])):.4f}")
    assert np.round(bare, 4).tolist() == pytest.approx([-0.2325, 0.1258], abs=1e-4)
    assert abs(rout[0]) < abs(bare[0]) and abs(rout[1]) < abs(bare[1])
    assert 0.5 < rout[0] / bare[0] < 0.6
    assert np.round(rout, 4).tolist() == pytest.approx([-0.1359, 0.0955], abs=1e-4)


def test_B5_the_runner_runs_the_scene(arms):
    r = arms["O"]
    c = R.couple_net(r["net"], r["world"], [], B_TO_A)
    mk = [w for w, nd in enumerate(c.nodes) if nd[0] == "market"][0]
    prog = R.compile_net(c, zero=0.0, one=1.0)
    print(f"\nB5 the translation: {prog.L} cells, {prog.Lp} runner steps a round, {prog.n_slp} instructions")
    sc = c.init_state()
    data, a = prog.encode(sc), 0.0
    worst = 0.0
    for _ in range(7):
        sc = R.ref_step(c, sc)
        a = R.machine_run(prog, data, a, 1)
        worst = max(worst, np.abs(np.array(prog.decode(data)[mk]) - np.array(sc[mk])).max())
    print(f"B5 runner vs ref_step heights {worst:.1e}")
    assert worst < 1e-10


def test_B5b_the_reduced_instance_through_the_library():
    net, _ = S.scene_market_net(S.SceneArm(), A=(0.2, 0.3), B=(0.2, 0.4), T=2)
    world = S.scene_world_net()
    c = R.couple_net(net, world, [], B_TO_A)
    lib = R.library_run_jit(c, 4, check=0, every=1)
    sc = c.init_state()
    worst = 0.0
    for k in range(4):
        sc = R.ref_step(c, sc)
        worst = max(worst, np.abs(np.array(lib[k]) - np.array(R.flat_ref(c, sc))).max())
    print(f"\nB5b the library's denotation of the reduced instance ({len(c.nodes)} nodes) vs ref_step: {worst:.1e}")
    assert worst < 1e-12
