"""The feeding note's general parts as library arrangements, each against its equation.

The note is the book repo's ``background/feeding-imc/feeding-imc.tex`` (``sec.push``,
``sec.general``). What runs in IMC here: every part is built by ``compose_seq`` and
``tensor_arrangements`` from integrands and ``push``; the fields are read off the composites'
own ``omega``; the motor and sweep runs go through ``functors.open_step``, which is checked
against ``Phi``. The one hand-typed loop, the dragger told a moving position, uses the
composite's own field and integrator and says so.
"""
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree

from arrangements import explore
from arrangements import feeding_parts as fp
from arrangements.functors import open_step

jax.config.update("jax_enable_x64", True)
EMPTY = jnp.zeros(0)
RNG = np.random.default_rng(0)


def field(part, q, n):
    """The composite's own inclination field at configuration ``q`` and inputs ``n``."""
    xi_q, _, xi_n = part.omega(jnp.asarray(q, float), EMPTY, jnp.asarray(n, float))
    return np.asarray(xi_q), np.asarray(xi_n)


def leaves(record):
    kids = record["children"]
    return [record["label"]] if not kids else [x for k in kids for x in leaves(k)]


def test_push_is_F_dx():
    for F, x in RNG.normal(size=(5, 2)):
        _, xi = field(fp.push(), [], [F, x])
        np.testing.assert_allclose(xi, [0.0, F])


def test_pushes_is_a_monoid_homomorphism():
    """Unit and a two-term sum: the empty list is the zero form, a concatenation the sum,
    and one term ``(c, h)`` is ``c dh`` with no derivative of ``c``."""
    a = ("a", lambda v: jnp.sin(v[0]) * v[2], lambda v: v[1] ** 2 + v[0])
    b = ("b", lambda v: v[1] * v[2], lambda v: jnp.cos(v[2]) * v[0])
    for v in RNG.normal(size=(5, 3)):
        v = jnp.asarray(v)
        _, zero = field(fp.pushes("none", 3, []), [], v)
        _, xa = field(fp.pushes("a", 3, [a]), [], v)
        _, xb = field(fp.pushes("b", 3, [b]), [], v)
        _, xab = field(fp.pushes("a+b", 3, [a, b]), [], v)
        np.testing.assert_allclose(zero, 0.0)
        np.testing.assert_allclose(xa, a[1](v) * jax.grad(a[2])(v), atol=1e-12)
        np.testing.assert_allclose(xb, b[1](v) * jax.grad(b[2])(v), atol=1e-12)
        np.testing.assert_allclose(xab, xa + xb, atol=1e-12)


def test_integrands_respond_as_the_note_says():
    tau = 0.01
    rod = fp.rod_segment(0.4, 1.5)
    s = open_step(rod, EMPTY, [2.0], duration=tau)(rod.P.init)
    np.testing.assert_allclose(s, [1.5 + tau * 2.0 / 0.4])
    m = fp.mass(0.25, 1.0, 0.5)
    s = open_step(m, EMPTY, [2.0], duration=tau)(m.P.init)
    np.testing.assert_allclose(s, [1.0 + tau * 0.5 / 0.25, 0.5 + tau * 2.0])


def test_spring_is_minus_d_half_k_u_squared():
    for k, u in RNG.normal(size=(5, 2)):
        _, xi = field(fp.spring(), [], [k, u])
        np.testing.assert_allclose(xi, [-0.5 * u * u, -k * u], atol=1e-12)


def test_send_and_discard():
    """A constant stiffness sent in drops the stiffening load; an unread output gets load 0."""
    held = fp.send(fp.spring(), {0: 7.0})
    assert (held.in_dim_N, held.out_dim_N) == (1, 0)
    _, xi = field(held, [], [0.3])
    np.testing.assert_allclose(xi, [-7.0 * 0.3])
    air = fp.discard(fp.send(fp.dragger(0.2, memory=0.1), {0: 40.0}), [0])
    assert (air.in_dim_N, air.out_dim_N) == (1, 0)
    xi_z, xi_x = field(air, [0.1], [0.6])
    np.testing.assert_allclose(xi_z, [40.0 * 0.5])
    np.testing.assert_allclose(xi_x, [-40.0 * 0.5])


def test_dragger_field_and_drag():
    beta = 0.4
    for k, x, z in RNG.normal(size=(5, 3)):
        xi_z, xi_n = field(fp.dragger(beta), [z], [k, x])
        d = x - z
        np.testing.assert_allclose(xi_z, [k * d], atol=1e-12)                 # memory moves at k d / beta
        np.testing.assert_allclose(xi_n, [-0.5 * d * d, -k * d], atol=1e-12)  # loads at k and x
    # Told a position moving at velocity w, the stretch settles to beta w / kappa and the load
    # on the position to -beta w, whatever kappa is. A hand loop over the composite's own
    # field and integrator, since its input changes each step.
    w, dt = 0.5, 0.002
    for kappa in (40.0, 160.0):
        part = fp.send(fp.dragger(beta), {0: kappa})

        @jax.jit
        def advance(s, x):
            xi_q, _, xi_x = part.omega(part.P.position(s), EMPTY, x)
            return part.P.step_over(s, xi_q, dt), xi_x

        s = part.P.init
        for i in range(4000):
            s, xi_x = advance(s, jnp.array([w * i * dt]))
        np.testing.assert_allclose(xi_x, [-beta * w], rtol=1e-3)


def test_damper_field_and_return():
    beta, kr = 40.0, 10.0
    for k, u, z in RNG.normal(size=(5, 3)):
        xi_z, xi_n = field(fp.damper(beta, kr), [z], [k, u])
        d = u - z
        np.testing.assert_allclose(xi_z, [k * d - kr * z], atol=1e-12)
        np.testing.assert_allclose(xi_n, [-0.5 * d * d, -k * d], atol=1e-12)
    # Out of contact, k = 0: nothing acts on the gap and the stroke returns at kr / beta.
    part = fp.damper(beta, kr, stroke=0.3)
    xi_z, xi_n = field(part, [0.3], [0.0, 5.0])
    np.testing.assert_allclose(xi_n[1], 0.0)
    s = ravel_pytree(open_step(part, [0.0, 5.0], duration=0.01)(part.P.init))[0]
    np.testing.assert_allclose(s, [0.3 - 0.01 * kr * 0.3 / beta])


def test_contact_field_and_gates():
    """A spring and a damper on one stretch, each behind a gate wire: the field is minus the
    differential of ``contact_cost`` plus the return spring's, and each gate gets back the
    energy it takes to stiffen its part."""
    ks, kd, beta, kr = 100.0, 4000.0, 40.0, 10.0
    part = fp.contact(ks, kd, beta, kr)
    assert (part.in_dim_N, part.out_dim_N, part.P.Q_dim) == (3, 1, 1)
    cost = fp.contact_cost(ks, kd)
    whole = lambda z, gs, gd, u: cost(gs, gd, u, z) + 0.5 * kr * z * z
    for gs, gd, u, z in RNG.normal(size=(8, 4)) * np.array([1.0, 1.0, 0.3, 0.05]):
        xi_z, xi_n = field(part, [z], [gs, gd, u])
        d = u - z
        np.testing.assert_allclose(xi_z, [kd * gd * d - kr * z], atol=1e-10)
        np.testing.assert_allclose(xi_n, [-0.5 * ks * u * u, -0.5 * kd * d * d, -ks * gs * u - kd * gd * d],
                                   atol=1e-10)
        grads = jax.grad(whole, argnums=(0, 1, 2, 3))(z, gs, gd, u)
        np.testing.assert_allclose(np.concatenate([xi_z, xi_n]), -np.asarray(grads), atol=1e-10)
        np.testing.assert_allclose(part.out_f(jnp.array([z]), EMPTY), [z])       # it sends its stroke out
    # Both gates shut: nothing acts on the stretch, and the stroke returns at kr / beta.
    xi_z, xi_n = field(part, [0.3], [0.0, 0.0, 5.0])
    np.testing.assert_allclose(xi_n[2], 0.0)
    np.testing.assert_allclose(xi_z, [-kr * 0.3])
    # The spring's gate alone: no damper, the stroke still returns.
    xi_z, xi_n = field(part, [0.3], [1.0, 0.0, 0.2])
    np.testing.assert_allclose(xi_n[2], -ks * 0.2)
    np.testing.assert_allclose(xi_z, [-kr * 0.3])


def test_latch_law_and_payment():
    """The latch's store moves at the rate of the note's ``eqn.contact_latch``, whatever load
    is on it, when its motor is told that load; the battery pays ``C u^2 - P u``."""
    C, rate, wake = 1e-4, 40.0, 1.0
    theta = fp.theta

    def law(v, b, free, acquire, retain):
        s = theta((v - .3) / .2)
        return rate * theta(b / wake) * (free * ((1 - s) * acquire + s * retain) - v)

    part = fp.latch(C, rate, wake)
    assert (part.in_dim_N, part.out_dim_N, part.P.Q_dim) == (5, 1, 1)
    for v, b, free, acquire, retain, P in RNG.uniform(size=(8, 6)) * np.array([1.2, 2.0, 1.0, 1.0, 1.0, 0.02]):
        u = law(v, b, free, acquire, retain)
        xi_v, xi_n = field(part, [v], [b, free, acquire, retain, P])
        np.testing.assert_allclose(xi_v, [C * u - P], atol=1e-12)
        np.testing.assert_allclose(xi_n, [-(C * u - P) * u, 0.0, 0.0, 0.0, 0.0], atol=1e-10)
        # A load P comes back along its output; told P, the store moves at the law's rate.
        tau = 1e-3
        s = ravel_pytree(open_step(fp.latch(C, rate, wake, initial=v), [b, free, acquire, retain, P], [P],
                                   duration=tau)(fp.latch(C, rate, wake, initial=v).P.init))[0]
        np.testing.assert_allclose(s, [v + tau * u], atol=1e-9)
    # What it takes to engage and what it takes to stay engaged differ: with nothing to
    # acquire and something to retain, a latch at 0 stays at 0 and a latch at 1 stays at 1.
    np.testing.assert_allclose(law(0.0, 5.0, 1.0, 0.0, 1.0), 0.0)
    np.testing.assert_allclose(law(1.0, 5.0, 1.0, 0.0, 1.0), 0.0)
    # Not free (another latch is engaged): it relaxes toward 0 from anywhere.
    assert law(1.0, 5.0, 0.0, 1.0, 1.0) < 0 and law(0.2, 5.0, 0.0, 1.0, 1.0) < 0
    # A flat battery: it does not move.
    np.testing.assert_allclose(law(0.7, 0.0, 1.0, 1.0, 1.0), 0.0)


def test_identity_passes_values_and_loads():
    part = fp.identity(2, 3)
    m, n = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0])
    np.testing.assert_allclose(part.out_f(EMPTY, m), m)
    np.testing.assert_allclose(part.in_f(EMPTY, m, n), n)
    assert all(not np.any(x) for x in part.omega(EMPTY, m, n))


def test_motor_field():
    """Told a push t, the motor applies live(b) t: the whole push from the level ``wake`` up,
    none on an empty battery, part in between."""
    C, wake = 0.02, 1.0
    for b, q, t, P in RNG.normal(size=(8, 4)) + np.array([0.6, 0.0, 0.0, 0.0]):
        g = fp.theta(b / wake)
        _, xi = field(fp.motor(C, wake), [], [b, q, t, P])
        np.testing.assert_allclose(xi, [-g * t * (g * t + P) / C, g * t, 0.0, 0.0], atol=1e-12)
    _, xi = field(fp.motor(C, wake), [], [0.0, 0.3, 2.0, -1.0])
    np.testing.assert_allclose(xi, [0.0, 0.0, 0.0, 0.0])


def test_motor_at_a_rate_against_a_spring():
    """A battery, a store, a spring on the store, and a motor told the spring's load: the
    store moves at the chosen rate times live(b), and the battery pays the store's friction
    C u^2 plus the work done against the spring, at that rate. On an empty battery the store
    stays where it is and nothing is paid."""
    C, u, kappa, q0, tau, wake = 0.02, 0.7, 3.0, 0.3, 1e-3, 1.0
    for b0 in [5.0, 0.5, 0.0]:
        g = float(fp.theta(b0 / wake))
        parts = [fp.rod_segment(1.0, b0, "battery"), fp.rod_segment(C, q0, "store"),
                 fp.motor_at_rate(C, wake), fp.send(fp.spring(), {0: kappa})]
        wiring = fp.transfer("driven store", 5, 2, 0, 0,
                             lambda m, n: jnp.stack([m[0], m[1], jnp.asarray(u), -kappa * m[1], m[1]]))
        system = fp.fill(parts, wiring)
        b, q = ravel_pytree(open_step(system, EMPTY, duration=tau)(system.P.init))[0]
        np.testing.assert_allclose(q, q0 + tau * g * u, atol=1e-14)
        np.testing.assert_allclose(b, b0 - tau * (C * (g * u) ** 2 + kappa * q0 * g * u), atol=1e-14)
    assert 0.0 < float(fp.theta(0.5 / wake)) < 1.0


def test_sweep_field_and_circle():
    C, A, nu, gamma, wake = 0.002, 6.0, 0.6, 0.05, 1.0
    for b, q1, q2, g, e in RNG.normal(size=(8, 5)) + np.array([0.6, 0.0, 0.0, 0.0, 0.0]):
        _, xi = field(fp.sweep(C, wake, A, nu, gamma), [], [b, q1, q2, g, e])
        L = fp.theta(b / wake)
        ring = A * A - q1 * q1 - q2 * q2
        r1 = g * (nu * q2 + gamma * ring * q1) + e
        r2 = g * (-nu * q1 + gamma * ring * q2)
        np.testing.assert_allclose(xi, [-C * L * L * (r1 * r1 + r2 * r2), C * L * r1, C * L * r2, 0.0, 0.0],
                                   atol=1e-12)
    # Gate open, no extra rate: the stores go round the circle and the battery pays C nu^2 A^2
    # per unit of duration; gate shut: they stop.
    def run(gate, steps, tau=0.01):
        parts = [fp.rod_segment(1.0, 10.0, "battery"), fp.rod_segment(C, A, "look"),
                 fp.rod_segment(C, 0.0, "scan"), fp.sweep(C, wake, A, nu, gamma)]
        wiring = fp.transfer("eye", 5, 3, 0, 0,
                             lambda m, n: jnp.stack([m[0], m[1], m[2], jnp.asarray(gate), jnp.asarray(0.0)]))
        system = fp.fill(parts, wiring)
        step = open_step(system, EMPTY, duration=tau)
        s = system.P.init
        for _ in range(steps):
            s = step(s)
        return np.asarray(ravel_pytree(s)[0])
    b, q1, q2 = run(1.0, 2000)
    np.testing.assert_allclose(np.hypot(q1, q2), A, rtol=0.02)
    np.testing.assert_allclose(10.0 - b, C * nu * nu * A * A * 20.0, rtol=0.05)
    np.testing.assert_allclose(run(0.0, 20), [10.0, A, 0.0])


def test_the_explorer_shows_the_parts():
    """``explore.inspect`` reads the damper's factors off the library's own record."""
    k, u, z = 4000.0, 0.02, 0.005
    part = fp.damper(40.0, 10.0, stroke=z)
    record = explore.inspect(part, inputs=[k, u])
    names = leaves(record)
    assert names.count("memory") == 1
    assert names.count("push: pull") == 2 and names.count("push: stiffening") == 2
    d = u - z
    np.testing.assert_allclose(record["inclination"], [k * d - 10.0 * z], atol=1e-9)
    np.testing.assert_allclose(record["n_in_load"], [-0.5 * d * d, -k * d], atol=1e-9)
    explore.scene(record)
