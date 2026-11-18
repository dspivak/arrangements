"""Check the feeding page against the composed IMC receiver, through whole search trips."""
import json
import subprocess
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.feeding import feeding, pack_state, unpack_state
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK


def test_search_receiver_and_browser_agree():
    jax.config.update('jax_enable_x64', True)
    viewer = Path(__file__).resolve().parents[1] / 'viewer'
    browser = json.loads(subprocess.check_output(
        ['node', 'food-learning.test.js', '--json'], cwd=viewer, text=True))
    empty = jnp.zeros(0)
    inputs = (empty, trivial_omega(0))
    loads = lambda _: (jnp.zeros(2), empty)
    dynamics = Phi(feeding((4., 8., -5.), (10., 10., .05), learning=True))
    duration = 1e-5

    @jax.jit
    def step(state):
        _, fiber = dynamics.step(state, duration)
        return fiber(inputs)[1](loads(None))[1]

    for sample in browser['samples']:
        s = np.asarray(sample['state'])
        result = unpack_state(step(pack_state(s, 3)), 3)
        np.testing.assert_allclose(result, s + duration*np.asarray(sample['rate']), rtol=0, atol=1e-12)

    fixed = PhiK(feeding([4.], [.05], learning=True, stages=4, h=.004, parameters=browser['fixed']))

    @jax.jit
    def revisit(state):
        def advance(_, s):
            return fixed.with_state(s).run_one(inputs, loads)[1]
        return jax.lax.fori_loop(0, 250, advance, state)

    for visit in browser['revisits']:
        result = unpack_state(revisit(pack_state(visit['start'], 1)), 1)
        np.testing.assert_allclose(result, visit['end'], rtol=0, atol=1e-10)

    for record in browser['records']:
        arr = feeding(record['places'], record['foods'], learning=True, stages=4, h=.004,
                      cart=record['cart'], memory=record['memory'],
                      parameters={'searchSeed': record['seed']})
        dynamics4 = PhiK(arr)

        def advance(_, s):
            return dynamics4.with_state(s).run_one(inputs, loads)[1]

        end = jax.lax.fori_loop(0, round(record['time']/.004), advance, arr.P.init)
        np.testing.assert_allclose(unpack_state(end, len(record['places'])), record['end'], atol=3e-7)
