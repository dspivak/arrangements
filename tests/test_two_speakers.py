"""Ch 4 ex.two_speakers: one listener hearing two speakers through the network (p, 1-p).

The static arrangement whose out-transfer is the value action of the network P_p = (p, 1-p) from two speakers to
one listener, run through Phi with a load at the listener's value. Expected (book, prop.network_actions):
the listener receives p v1 + (1-p) v2; the load l comes back to the speakers as (p l, (1-p) l); at p = 0 speaker 1
is silenced both ways. The load on p along the simplex direction (1, -1) is l (v1 - v2), read off a one-dimensional
configuration integrator whose readout is p itself (the same arrangement with p as its configuration).
Runtime: a few seconds.

Ch 4 ``ex.load_two_speakers`` (Claude Opus 5.5, 00f2d232, 2026-11-12): two listeners, the network
``T(q1, q2) = ((q1, 1 - q1), (q2, 1 - q2))`` with configuration ``q`` in ``(0, 1)^2``, and an inclination field of its
own, ``-dc`` for ``c(q) = |q - (1/2, 1/2)|^2 / 2``. Prediction before the code (``eqn.load_two_speakers``): told the
loads ``l1, l2`` at the two heard values, the total inclination at ``q`` is ``-dc_q + l1 (v1 - v2) dq1 + l2 (v1 - v2) dq2``,
read off the step of the configuration integrator at durations ``1/4`` and ``1``; the load-induced part is the same at
every ``q``.
"""
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
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


def two_listeners(q0) -> Arrangement:
    """``ex.load_two_speakers``: two speakers' values in, two heard values out, through ``T(q)``."""
    return Arrangement(
        P=configuration(euclidean(2, ETA), init=jnp.asarray(q0, dtype=float)),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=2,
        out_f=lambda q, m: jnp.stack([q[0] * m[0] + (1 - q[0]) * m[1], q[1] * m[0] + (1 - q[1]) * m[1]]),
        in_f=lambda q, m, n: jnp.zeros(0),
        omega=exact(lambda q, m, n: -0.5 * jnp.sum((q - 0.5) ** 2)),
        label="two_listeners(q)",
    )


def test_load_two_speakers():
    v1, v2, l1, l2 = 2.0, -1.0, 0.7, -0.4
    induced = np.array([l1 * (v1 - v2), l2 * (v1 - v2)])
    for q0 in ([0.5, 0.25], [0.9, 0.1]):
        arr = two_listeners(q0)
        in_pos = (jnp.array([v1, v2]), trivial_omega(0))
        tell = lambda out_pos: (jnp.array([l1, l2]), jnp.zeros(0))
        for tau in (0.25, 1.0):
            out_pos, (xi_M, _), s1 = Phi(arr).with_state(arr.P.init).run_one(in_pos, tell, tau)
            q = np.asarray(q0)
            np.testing.assert_allclose(out_pos[0], [q[0] * v1 + (1 - q[0]) * v2, q[1] * v1 + (1 - q[1]) * v2], atol=1e-12)
            np.testing.assert_allclose(xi_M, [q[0] * l1 + q[1] * l2, (1 - q[0]) * l1 + (1 - q[1]) * l2], atol=1e-12)
            total = (np.asarray(s1) - q) / (tau * ETA)
            np.testing.assert_allclose(total, -(q - 0.5) + induced, atol=1e-12)
