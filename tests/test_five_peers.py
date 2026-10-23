"""Five peers who move to hear one another better (the book's Ch 4 ``ex.five_peers``, plan P5 (k)).

The construction is ``arrangements/five_peers.py``: an attention arrangement whose matter is the five places,
its internal boxes filled by five persons who report the number plus a fresh error each round, composed in
``arr`` and run through ``Phi`` (``open_step``, which checks its compiled step against ``Phi`` on the first two
steps). The predictions were written before any code, in the book repository,
``notes/runs/p5k-b5cf8328/PREDICTIONS.md``; each test below is one of them, and its docstring says whether it
checks a closed form or illustrates what the book says it does not prove (STATUS.md §2).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements import five_peers as fp
from arrangements.attention import mean
from arrangements.functors import open_step

jax.config.update("jax_enable_x64", True)

SIGMA2S = (1.5, 3.7, 6.0, 10.0)


def runner(a, x, chunk):
    """``chunk`` steps of ``Phi(a)`` on the external input ``msg(x)``, compiled once; returns the end state and
    the places after each step."""
    step = open_step(a, fp.exterior(x))

    @jax.jit
    def many(s):
        def body(s, _):
            s2 = step(s)
            return s2, fp.places_of(s2)
        return jax.lax.scan(body, s, None, length=chunk)

    return many


def with_places(state, places):
    """The composite state with the five places replaced (the persons' errors and the sampler kept)."""
    (persons, _), key = state
    return (persons, jnp.asarray(places, float)), key


def test_the_cost_reads_what_the_delivery_hands_each_person():
    """CLOSED FORM. At random places and reports, each sent with weight 1, person ``j``'s heard value in the
    realized cost, ``sum_i P_ji(q) h_i``, is the value of the message the delivery hands person ``j``; and the
    message from outside reaches no person, so the number enters only the cost function."""
    rng = np.random.default_rng(0)
    a = fp.parent(fp.realized_cost, np.zeros(5), 1.0)
    for _ in range(5):
        q = jnp.asarray(rng.uniform(-2, 2, 5))
        h = jnp.asarray(rng.normal(size=5))
        m_out = jnp.concatenate([jnp.array([hi, 1.0]) for hi in h])
        heard = fp.weights(q) @ h
        for x in (7.0, -3.0):
            e = fp.exterior(x)
            for j in range(5):
                delivered = a.received(q, m_out, e, j + 1)
                assert abs(float(mean(delivered[:2])[0]) - float(heard[j])) < 1e-12
                assert abs(float(delivered[2]) - float(q[j])) < 1e-12        # placed at its own place


def test_the_expected_value_of_the_realized_cost_is_the_expected_cost():
    """CLOSED FORM (prediction 1). The realized cost at reports ``x + sigma_i z_i`` and number ``x`` is quadratic in
    the seed ``z``, so its expected value is exactly ``f(0) + (1/2) tr d^2f/dz^2``; it equals ``sum_j k_j(q)``, and
    the same holds for its derivative in the places. This is the book's "over time these moves add up to the ones
    that lower the expected value": the expected force on the places is the expected cost's force."""
    rng = np.random.default_rng(1)
    x = 7.0
    for s2 in (1.5, 6.0):
        sd = jnp.sqrt(fp.variances(s2))

        def f(z, q):
            m_out = jnp.concatenate([jnp.array([x + sd[i] * z[i], 1.0]) for i in range(5)])
            return fp.realized_cost(q, m_out, fp.exterior(x))

        z0 = jnp.zeros(5)
        for _ in range(5):
            q = jnp.asarray(rng.uniform(-2, 2, 5))
            Ef = f(z0, q) + 0.5 * jnp.trace(jax.hessian(f)(z0, q))
            assert abs(float(Ef) - float(jnp.sum(fp.person_costs(q, s2)))) < 1e-10
            g = jax.grad(f, argnums=1)
            Eg = g(z0, q) + 0.5 * jnp.trace(jax.jacfwd(jax.jacfwd(g, argnums=0), argnums=0)(z0, q), axis1=1, axis2=2)
            dk = jax.grad(lambda q: jnp.sum(fp.person_costs(q, s2)))(q)
            assert float(jnp.max(jnp.abs(Eg - dk))) < 1e-10


def test_descent_on_the_expected_cost_rests_where_the_book_says():
    """ILLUSTRATES what the book says it does not prove (prediction 2). For each ``sigma^2`` in (1.5, 3.7, 6, 10),
    20 starts uniform on ``[-2, 2]^5``, response ``eta = 1``: every run comes to rest (a step moves no place by more
    than 1e-9) with persons 1-4 within 1e-5 of one another and person 5 at ``sqrt(2 ln sigma^2)`` from them; there
    person 1's weights are ``(0, s, s, s, 1) / (3 s + 1)`` for ``s = sigma^2``, persons 1-4 have cost
    ``s / (3 s + 1)`` and person 5 has cost 1/4. The starts are bounded: from places far apart the softmin hears
    only the nearest, and the forces between groups are of order ``exp(-distance^2 / 2)``."""
    rng = np.random.default_rng(2)
    for s2 in SIGMA2S:
        a = fp.five_peers(s2, np.zeros(5), realized=False, eta=1.0)
        many = runner(a, 7.0, 5000)
        r = fp.rest_distance(s2)
        for _ in range(20):
            s = with_places(a.P.init, rng.uniform(-2, 2, 5))
            for _chunk in range(40):
                s, trace = many(s)
                if float(jnp.max(jnp.abs(trace[-1] - trace[-2]))) < 1e-9:
                    break
            else:
                raise AssertionError(f"no rest at sigma^2={s2}")
            q = fp.places_of(s)
            four = q[:4]
            assert float(jnp.max(four) - jnp.min(four)) < 1e-5
            assert abs(abs(float(q[4] - jnp.mean(four))) - r) < 1e-5
            P1 = fp.weights(q)[0]
            assert float(jnp.max(jnp.abs(P1 - jnp.array([0, s2, s2, s2, 1]) / (3 * s2 + 1)))) < 1e-6
            k = fp.person_costs(q, s2)
            assert float(jnp.max(jnp.abs(k[:4] - s2 / (3 * s2 + 1)))) < 1e-8
            assert abs(float(k[4]) - 0.25) < 1e-8


def test_the_rest_is_strict_up_to_translation():
    """CLOSED FORM, at the rest the book names (prediction 3). There the force on the places is zero and the
    Hessian of the expected total cost has exactly one zero eigenvalue, moving all five together, and four
    positive ones (smallest 0.167, at ``sigma^2 = 1.5``), so no other direction is flat."""
    for s2 in SIGMA2S:
        q = jnp.array([0.0, 0.0, 0.0, 0.0, fp.rest_distance(s2)])
        total = lambda q: jnp.sum(fp.person_costs(q, s2))
        assert float(jnp.max(jnp.abs(jax.grad(total)(q)))) < 1e-12
        ev = np.linalg.eigvalsh(np.asarray(jax.hessian(total)(q)))
        assert abs(ev[0]) < 1e-12
        assert ev[1] > 0.1


def test_descent_on_each_rounds_squared_error_jitters_about_the_same_rest():
    """ILLUSTRATES (prediction 4). The realized cost, which uses no variance, with the number ``x = 7`` revealed
    from outside after each round and every error drawn fresh; ``eta = 0.01``, 40,000 rounds, three seeds for each
    ``sigma^2`` in (4, 3.7, 6, 10), from starts uniform on ``[-2, 2]^5``. Over the last 20,000 rounds the four's
    spread averages below 0.2 (seen: 0.10-0.11) and person 5's distance from the four's mean averages within 10%
    of ``sqrt(2 ln sigma^2)`` (seen: 1-3% above it, the step's bias, the same sign at every ``sigma^2``)."""
    rng = np.random.default_rng(3)
    for s2 in (4.0, 3.7, 6.0, 10.0):
        r = fp.rest_distance(s2)
        for sd in range(3):
            a = fp.five_peers(s2, rng.uniform(-2, 2, 5), realized=True, eta=0.01, sampler_seed=sd)
            _, trace = runner(a, 7.0, 40000)(a.P.init)
            tail = np.asarray(trace)[20000:]
            four = tail[:, :4]
            assert (four.max(1) - four.min(1)).mean() < 0.2
            assert abs(np.abs(tail[:, 4] - four.mean(1)).mean() / r - 1) < 0.1


def test_the_number_cancels():
    """CLOSED FORM (prediction 5). Each heard value minus the number is the weighted sum of the errors, so the
    realized run with the number 7 and with the number -3, on one seed, moves the places identically."""
    q0 = [0.3, -0.5, 0.9, 0.1, -1.2]
    runs = []
    for x in (7.0, -3.0):
        a = fp.five_peers(4.0, q0, realized=True, x=x, eta=0.01, sampler_seed=5)
        runs.append(np.asarray(runner(a, x, 2000)(a.P.init)[1]))
    assert np.max(np.abs(runs[0] - runs[1])) < 1e-10
