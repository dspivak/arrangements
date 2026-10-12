"""P34(k): assembled attention kit, force correspondence, and full runner passes.

Reference: old kit/independent Fraction stepper, not a second spelling of the
attention construction. Kernels: ordered entropy Bregman divergence, including
its forces and state response. No claim that weighted routing preserves an
arbitrary program or that attention learns a runner. See STATUS.md.
"""
from fractions import Fraction as F
import math

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import attention_runner as A, runner as R
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import interpretation, trivial_omega
from arrangements.rvect import ReactiveVectorSpace


CASES = [
    ("const", (-3.5,), [], [2.0]),
    ("register", (1.25,), [], [-0.7]),
    ("drum", ([1.0, -2.0, 3.0],), [], [0.6]),
    ("add", (), [2.0, -3.0], [1.7]),
    ("mul", (), [2.0, -3.0], [1.7]),
    ("push", (), [2.0, 3.0], []),
    ("cost", (), [-5.0], []),
    ("fan", (3, [2, 0, 2]), [1.0, 2.0, 3.0], [0.5, -2.0, 4.0]),
    ("fan", (2, []), [2.0, 3.0], []),
    ("fan", (0, []), [], []),
]


@pytest.mark.parametrize("name,args,values,pushes", CASES)
def test_kit_forward_backward_and_update(name, args, values, pushes):
    new, old = A.kit()[name](*args), R._kit()[name](*args)
    z, values, pushes = jnp.zeros(0), jnp.array(values), jnp.array(pushes)
    answers = []
    for arr in (new, old):
        q = arr.P.position(arr.P.init)
        positions, directions = interpretation(arr)(q)
        output, _ = positions(values, trivial_omega(0))
        xi_q, xi_m, in_m = directions(values, trivial_omega(0), pushes, z)
        machine = Phi(arr)
        _, _, state = machine.run_one((values, trivial_omega(0)), lambda _: (pushes, z))
        answers.append((output, xi_q, xi_m, in_m, R.flat_state(state)))
    for new_value, old_value in zip(*answers):
        np.testing.assert_allclose(new_value, old_value, atol=1e-13, rtol=1e-13)


def test_bregman_weight_value_force_and_response():
    # Strictly convex entropy on positive sites; log coordinates cover all R².
    potential = lambda p: jnp.sum(p*jnp.log(p)-p)
    gain = lambda q: A.bregman_gain(jnp.exp(q[:1]), jnp.exp(q[1:]), potential)
    q = jnp.log(jnp.array([2.0, 3.0]))
    matter = configuration(ReactiveVectorSpace(2, lambda q: jnp.eye(2)), init=q)
    route = A.linear_attention([[1.0]], parameter=matter, gain=gain,
                               label="entropy-Bregman-attention")
    g = math.exp(-(2*math.log(2/3)-2+3))
    reverse = float(A.bregman_gain(jnp.array([3.0]), jnp.array([2.0]), potential))
    assert g != pytest.approx(reverse, abs=1e-4)
    pos, dirs = interpretation(route)(q)
    output, _ = pos(jnp.array([5.0]), trivial_omega(0))
    xi_q, xi_m, _ = dirs(jnp.array([5.0]), trivial_omega(0), jnp.array([4.0]), jnp.zeros(0))
    expected_force = -20*g*np.array([2*math.log(2/3), 3-2])
    np.testing.assert_allclose(output, [5*g], atol=1e-13, rtol=0)
    np.testing.assert_allclose(xi_m, [4*g], atol=1e-13, rtol=0)
    np.testing.assert_allclose(xi_q, expected_force, atol=1e-13, rtol=0)
    _, _, state = Phi(route).run_one((jnp.array([5.0]), trivial_omega(0)),
                                    lambda _: (jnp.array([4.0]), jnp.zeros(0)))
    np.testing.assert_allclose(R.flat_state(state), np.asarray(q)+expected_force, atol=1e-13, rtol=0)
    # No cancellation: positive attenuation actually changes what is received.
    assert float(output[0]) < 5


@pytest.mark.parametrize("make", [R.two_tanks, R.turner, R.crossed])
def test_guest_terms_through_attention(make):
    net = make()
    arr = A.denote(R.to_term(net))
    assert "attention-" in arr.label and "unit-source" in arr.label
    machine, state = Phi(arr), arr.P.init
    expected, z = net.init_state(), jnp.zeros(0)
    for _ in range(3):
        expected = R.ref_step(net, expected)
        _, _, state = machine.with_state(state).run_one((z, trivial_omega(0)), lambda _: (z, z))
        np.testing.assert_allclose(R.flat_state(state),
                                   [float(x) for x in R.flat_ref(net, expected)], atol=1e-12, rtol=0)


@pytest.mark.parametrize("make", [R.two_tanks, R.turner, R.crossed])
def test_runner_full_passes(make):
    net = make()
    program = R.compile_net(net, zero=F(0), one=F(1))
    expected = net.init_state()
    runner = R.machine_net(program.flags(), program.encode(expected), a0=F(0), minus_one=F(-1))
    arr = A.denote(R.to_term(runner))
    print(f"\n{make.__name__}: runner steps/pass={program.Lp}; state cells={arr.P.Q_dim}")
    print("composite label:", arr.label)
    # JIT of this composite's own step, checked against Phi's first two steps.
    trajectory = R.arrangement_run_jit(arr, 3*program.Lp, check=2, every=program.Lp)
    data, accumulator = program.encode(expected), F(0)
    max_error = 0.0
    for flat in trajectory:
        expected = R.ref_step(net, expected)
        accumulator = R.machine_run(program, data, accumulator, 1)
        np.testing.assert_allclose(flat[:4*program.Lp],
                                   [float(x) for f in program.flags() for x in f], atol=1e-13, rtol=0)
        actual_data = flat[4*program.Lp:4*program.Lp+program.L]
        max_error = max(max_error, float(np.max(np.abs(np.array(actual_data)-np.array(data, dtype=float)))))
        np.testing.assert_allclose(actual_data, [float(x) for x in data], atol=2e-12, rtol=0)
        assert flat[-1] == pytest.approx(float(accumulator), abs=2e-12, rel=0)
        decoded = program.decode(actual_data)
        np.testing.assert_allclose([decoded[w] for w in sorted(decoded)],
                                   [float(expected[w]) for w in sorted(expected)], atol=2e-12, rtol=0)
    print(f"maximum absolute data/work-cell error over three passes: {max_error:.3g}")


def test_invalid_constructions_do_not_fall_back_to_old_kit():
    with pytest.raises(ValueError, match="original"):
        A.denote(("exp",))
    with pytest.raises(ValueError, match="nonempty"):
        A.drum([])
    with pytest.raises(ValueError, match="indices"):
        A.fan(1, [1])
    with pytest.raises(ValueError, match="temperature"):
        A.bregman_gain(jnp.ones(1), jnp.ones(1), lambda q: jnp.vdot(q, q), 0)
