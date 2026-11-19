"""Independent open components and their composition; live browser parity past the PDF episode."""
import json
from pathlib import Path
import shutil
import subprocess

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import hh_atp as hh
from arrangements.functors import open_step


def test_voltage_driven_gates_run_alone_and_in_channels():
    duration = 1e-6  # This 65 mV voltage jump stays inside each store's linear step at 1 µs.
    standalone = []
    for name in "mhn":
        gate = hh.gate(name)
        x = gate.P.position(gate.P.init)[0]
        rate = gate.omega(gate.P.position(gate.P.init), hh.EMPTY, jnp.array([0.]))[0][0]
        step = open_step(gate, jnp.array([0.]), duration=duration)
        new = gate.P.position(step(gate.P.init))[0]
        expected = x + duration * rate
        np.testing.assert_allclose(new, expected, rtol=1e-12)
        assert np.sign(new - x) == (-1 if name == "h" else 1)
        standalone.append(new)
    for channel, mu, expected in [(hh.sodium_channel(), -5000., standalone[:2]),
                                  (hh.potassium_channel(), 7500., standalone[2:])]:
        step = open_step(channel, jnp.array([0., mu]), duration=duration)
        np.testing.assert_allclose(channel.P.position(step(channel.P.init)), expected, rtol=1e-12)


def test_one_pump_cycle_reaches_four_stores_and_unchanged_gates():
    active, silent = hh.membrane(), hh.membrane(capacity=0)
    q = active.P.position(active.P.init)
    pump = hh.sodium_potassium_pump()
    port_load = pump.omega(hh.EMPTY, hh.EMPTY, q[:4])[2]
    outward_rate = -port_load[3]
    assert outward_rate > 0
    np.testing.assert_allclose(port_load, outward_rate*jnp.array([-hh.F, -3, 2, -1]))
    difference = active.omega(q, hh.EMPTY, hh.EMPTY)[0] - silent.omega(q, hh.EMPTY, hh.EMPTY)[0]
    np.testing.assert_allclose(difference, jnp.concatenate([port_load, jnp.zeros(3)]), atol=1e-20)


@pytest.mark.skipif(shutil.which("node") is None, reason="Browser export parity uses Node")
def test_offline_step_and_port_observations_match_library():
    module = Path(__file__).resolve().parents[1] / "viewer" / "hh-atp-dynamics.js"
    state, step, observe, _ = hh.live_membrane()
    steps = [0, 230, 1200, 2000, 10000, 100000]
    code = """
      require(process.argv[1]);
      let state=MembraneDynamics.initial, i=0;
      const data=JSON.parse(process.argv[2]).map(end=>{
        while(i<end){state=MembraneDynamics.step(state); i++;}
        return {state,...MembraneDynamics.observe(state)};
      });
      process.stdout.write(JSON.stringify(data));
    """
    frames = json.loads(subprocess.check_output(["node", "-e", code, str(module), json.dumps(steps)], text=True))
    advance = jax.jit(lambda s, count: jax.lax.fori_loop(0, count, lambda _, x: step(x), s))
    observe = jax.jit(observe)
    previous = 0
    for i, frame in zip(steps, frames, strict=True):
        state = advance(state, i - previous)
        q, values, loads, total = observe(state)
        for key, actual in dict(state=state, q=q, values=values, loads=loads, total=total).items():
            absolute = np.full(actual.shape, {"state": 1e-20, "q": 1e-12, "values": 1e-12,
                                               "loads": 1e-19, "total": 1e-19}[key])
            # The large reciprocal gate rates cancel at equilibrium, to roundoff in J/s.
            if key == "loads": absolute[[8, 10, 12]] = 1e-10
            if key == "total": absolute[4:] = 1e-10
            error = np.abs(np.asarray(frame[key]) - actual)
            assert np.all(error <= absolute + 2e-8 * np.abs(actual)), (key, i, error)
        previous = i
