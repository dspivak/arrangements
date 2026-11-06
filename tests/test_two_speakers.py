"""Ch 4 ex.two_speakers: one listener hearing two speakers through the network (p, 1-p).

The static arrangement whose out-transfer is the value action of the network P_p = (p, 1-p) from two speakers to
one listener, run through Phi with a load at the listener's value. Expected (book, prop.network_actions):
the listener receives p v1 + (1-p) v2; the load l comes back to the speakers as (p l, (1-p) l); at p = 0 speaker 1
is silenced both ways. The load on p along the simplex direction (1, -1) is l (v1 - v2), read off a one-dimensional
configuration integrator whose readout is p itself (the same arrangement with p as its configuration).
Runtime: a few seconds.
"""
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.rvect import euclidean

ETA = 1.0


def listener(p0: float) -> Arrangement:
    """Two speakers' outputs in, one heard value out; configuration p, read out unchanged, learning rate ETA."""
    return Arrangement(
        P=configuration(euclidean(1, ETA), init=jnp.array([p0], dtype=float)),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=1,
        out_f=lambda q, m: q[0] * m[0:1] + (1 - q[0]) * m[1:2],
        in_f=lambda q, m, n: jnp.zeros(0),
        omega=zero_form,
        label="listener(p)",
    )


def run(p, v1, v2, load):
    arr = listener(p)
    in_pos = (jnp.array([v1, v2]), trivial_omega(0))
    tell = lambda out_pos: (jnp.array([load]), jnp.zeros(0))
    out_pos, (xi_M, _in_m), new_state = Phi(arr).with_state(arr.P.init).run_one(in_pos, tell)
    return float(out_pos[0][0]), np.asarray(xi_M), (float(new_state[0]) - p) / ETA


def test_values_and_loads():
    v1, v2, load = 2.0, -1.0, 0.7
    for p in [0.0, 0.3, 1.0]:
        heard, loads, load_on_p = run(p, v1, v2, load)
        np.testing.assert_allclose(heard, p * v1 + (1 - p) * v2, atol=1e-6)
        np.testing.assert_allclose(loads, [p * load, (1 - p) * load], atol=1e-6)
        np.testing.assert_allclose(load_on_p, load * (v1 - v2), atol=1e-6)


def test_silenced_speaker_gets_no_load():
    heard, loads, _ = run(0.0, 2.0, -1.0, 0.7)
    assert heard == -1.0 and loads[0] == 0.0
