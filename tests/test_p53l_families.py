"""P53 (l): families closed under nesting in arr, each checked on a branching composite.

Book repo ``notes/runs/p53l-2e8eeac1/RESULTS.md`` (Claude Opus 5.5, 2e8eeac1, 2026-10-21) has the proofs; these
tests check them on data. Every composite is built with ``compose_seq`` and ``parallel_arrangements`` (the book's
``eqn.arr_compose``, ``eqn.lens_composite_forcing``, ``eqn.arr_tensor``) and every step is taken by ``closed_step``,
the dynamics functor's step. The branching composite throughout is ``b . (a1 (x) a2)``: two systems filling the two
internal boxes of ``b``, whose delivery copies the external input to both.

- potential-free: arrangements whose stateless arrangement is static (effort field 0) stay so;
- linear: linear readout, delivery and effort field stay linear, and a closed linear composite steps linearly;
  exact (symmetric) ones stay symmetric; the rest relation of the composite system is Lagrangian;
- reciprocal: the rest set of a nonlinear system with closed effort field and critical matter is isotropic, and the
  identity B^*(-theta_M + theta_N) = pi^* omega - <xi_q, dq> behind it holds on an arrangement with all four ports;
- mirror: a Bayes step is the exact flow (steps compose as durations add); two beliefs whose cost is affine in each
  but couples them are not;
- quadratic: the composite's tangent response is the block-diagonal constant symmetric sharp;
- (l2) open reaction networks glued along species by a junction (P54 (a) Prop. C) read as the glued network, with
  stoichiometry and two ports on one junction, and closed they step by Baez--Pollard's rate equation; resistor
  circuits glued at nodes rest in Baez--Fong's black box;
- (l3) learners: a branching composite of learners closed by a datum and a loss takes the gradient step of the
  composite loss; an orthogonal change of parameters gives an isomorphic arrangement, q = 2p does not;
- (l4) the prediction market's trust boxes, nested, step as the compositor's flat trust at unit duration only, and
  the Hebbian draft's operation steps as the draft says, its compositor W + W' commuting with unit steps without
  being a morphism of arrangements.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement
from arrangements.coalgebra import CotCoalg, configuration, phase
from arrangements.functors import closed_step
from arrangements.rvect import constant
from arrangements.wiring import compose_seq, parallel_arrangements, trivial_parameter

RNG = np.random.default_rng(53)
Z0 = jnp.zeros(0)


def rod(C):
    return configuration(constant(jnp.array([[1.0 / C]])))


def mass(m):
    return phase(constant(jnp.array([[1.0 / m]])))


def system(P, out_f, omega, in_N=1, out_N=1, label="sys"):
    """A system 1 -> <R^in_N | R^out_N> with matter P: readout out_f(q), effort field omega(q, n) -> (xi_q, xi_n)."""
    return Arrangement(
        P=P, in_dim_M=0, out_dim_M=0, in_dim_N=in_N, out_dim_N=out_N,
        out_f=lambda q, m: out_f(q),
        in_f=lambda q, m, n: Z0,
        omega=lambda q, m, n: (*omega(q, n)[:1], Z0, omega(q, n)[1]),
        label=label,
    )


def outer(P, out_f, in_f, omega, label="b"):
    """b : <R|R> (x) <R|R> -> <R|R>, matter P; out_f(q, m), in_f(q, m, n) -> R^2, omega(q, m, n)."""
    return Arrangement(P=P, in_dim_M=2, out_dim_M=2, in_dim_N=1, out_dim_N=1,
                       out_f=out_f, in_f=in_f, omega=omega, label=label)


def cap():
    """The static closing <R|R> -> 1 that delivers the box's output back to its input (a loop)."""
    return Arrangement(P=trivial_parameter(), in_dim_M=1, out_dim_M=1, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: Z0, in_f=lambda q, m, n: m,
                       omega=lambda q, m, n: (Z0, jnp.zeros(1), Z0), label="cap")


def branching(a1, a2, b):
    return compose_seq(parallel_arrangements(a1, a2), b)


def flat(arr):
    """The effort field of arr as one vector function of the concatenated point (q, m_out, n_in)."""
    dq, dm, dn = arr.P.Q_dim, arr.out_dim_M, arr.in_dim_N

    def w(x):
        q, m, n = x[:dq], x[dq:dq + dm], x[dq + dm:]
        return jnp.concatenate(arr.omega(q, m, n))

    return w, dq + dm + dn


def is_linear(f, d, trials=4, atol=1e-11):
    for _ in range(trials):
        x, y = jnp.array(RNG.normal(size=d)), jnp.array(RNG.normal(size=d))
        a, b = RNG.normal(size=2)
        if not jnp.allclose(f(a * x + b * y), a * f(x) + b * f(y), atol=atol):
            return False
    return True


# ---------------------------------------------------------------------------
# Potential-free: the stateless arrangement static.
# ---------------------------------------------------------------------------


def potential_free_pieces():
    zero = lambda q, n: (jnp.zeros_like(q), jnp.zeros_like(n))
    a1 = system(rod(2.0), lambda q: q, zero, label="rod1")
    a2 = system(mass(3.0), lambda q: q ** 3, zero, label="mass2")
    b = outer(rod(1.5),
              out_f=lambda q, m: jnp.array([m[0] * m[1] + jnp.sin(q[0])]),
              in_f=lambda q, m, n: jnp.array([n[0] + m[1], n[0] * m[0] + q[0]]),
              omega=lambda q, m, n: (jnp.zeros(1), jnp.zeros(2), jnp.zeros(1)))
    return a1, a2, b


def test_potential_free_branching_composite_has_zero_effort_field():
    c = branching(*potential_free_pieces())
    w, d = flat(c)
    for _ in range(5):
        assert jnp.allclose(w(jnp.array(RNG.normal(size=d))), 0.0)
    # closed by the loop, every part coasts: a step with no force
    z = compose_seq(c, cap())
    tree = jax.tree_util.tree_structure(z.P.init)
    s0 = jax.tree_util.tree_unflatten(tree, [jnp.array([v]) for v in (0.3, 0.2, 0.7, -0.4)] + [Z0])
    step = closed_step(z, check=0, duration=0.1)
    r1, q2, p2, r3, _ = jax.tree_util.tree_leaves(step(s0))
    assert jnp.allclose(r1, 0.3) and jnp.allclose(r3, -0.4)          # the rods stay put
    assert jnp.allclose(p2, 0.7) and jnp.allclose(q2, 0.2 + 0.1 * 0.7 / 3.0)  # the mass coasts


# ---------------------------------------------------------------------------
# Linear, exact and not.
# ---------------------------------------------------------------------------


def linear_pieces(symmetric):
    def sym(A):
        return (A + A.T) / 2 if symmetric else A

    A1, A2 = sym(jnp.array(RNG.normal(size=(2, 2)))), sym(jnp.array(RNG.normal(size=(2, 2))))
    Ab = sym(jnp.array(RNG.normal(size=(4, 4))))
    D = jnp.array(RNG.normal(size=(2, 4)))
    D = D.at[:, 3].set(1.0)                                  # the external input is copied to both boxes
    cb = jnp.array(RNG.normal(size=3))

    def lin(A):
        return lambda q, n: tuple(jnp.split(A @ jnp.concatenate([q, n]), [q.shape[0]]))

    a1 = system(rod(2.0), lambda q: 1.7 * q, lin(A1), label="rod1")
    a2 = system(mass(3.0), lambda q: -0.6 * q, lin(A2), label="mass2")

    def omega_b(q, m, n):
        v = Ab @ jnp.concatenate([q, m, n])
        return v[:1], v[1:3], v[3:]

    b = outer(rod(1.5),
              out_f=lambda q, m: jnp.array([cb[0] * m[0] + cb[1] * m[1] + cb[2] * q[0]]),
              in_f=lambda q, m, n: D @ jnp.concatenate([m, q, n]),
              omega=omega_b)
    return a1, a2, b


def test_linear_branching_composite_is_linear_and_steps_linearly():
    for symmetric in (True, False):
        c = branching(*linear_pieces(symmetric))
        dq = c.P.Q_dim
        assert is_linear(lambda x: c.out_f(x[:dq], Z0), dq)
        assert is_linear(lambda x: c.in_f(x[:dq], Z0, x[dq:]), dq + 1)
        w, d = flat(c)
        assert is_linear(w, d)
        J = jax.jacfwd(w)(jnp.zeros(d))
        assert bool(jnp.allclose(J, J.T, atol=1e-12)) == symmetric   # exact iff symmetric, kept by nesting
        z = compose_seq(c, cap())
        step = closed_step(z, check=0, duration=0.25)
        leaves, tree = jax.tree_util.tree_flatten(z.P.init)
        sizes = [l.shape[0] for l in leaves]

        def vec_step(x):
            parts = jnp.split(x, np.cumsum(sizes)[:-1])
            return jnp.concatenate(jax.tree_util.tree_leaves(step(jax.tree_util.tree_unflatten(tree, parts))))

        assert is_linear(vec_step, sum(sizes))


def rest_relation_basis(arr):
    """The rest relation of a linear system 1 -> N with critical matter: B(ker balance), as a matrix of columns.

    Parameters e = (q, n_in, xi_nout); port data (n_in, n_out, xi_nin, xi_nout), forces recorded outward
    (Ch 5 def.rest_relation): xi_nin = omega_n, xi_q = omega_q - (d_q out)^T xi_nout.
    """
    dq, di, do = arr.P.Q_dim, arr.in_dim_N, arr.out_dim_N

    def parts(e):
        q, n, xo = e[:dq], e[dq:dq + di], e[dq + di:]
        xq, _, xn = arr.omega(q, Z0, n)
        _, vjp = jax.vjp(lambda qq: arr.out_f(qq, Z0), q)
        balance = xq - vjp(xo)[0]
        port = jnp.concatenate([n, arr.out_f(q, Z0), xn, xo])
        return balance, port

    d = dq + di + do
    Bal = jax.jacfwd(lambda e: parts(e)[0])(jnp.zeros(d))
    Bnd = jax.jacfwd(lambda e: parts(e)[1])(jnp.zeros(d))
    _, s, vt = np.linalg.svd(np.asarray(Bal))
    rank = int((s > 1e-10).sum())
    kernel = vt[rank:].T
    image = np.asarray(Bnd) @ kernel
    u, s2, _ = np.linalg.svd(image)
    return u[:, : int((s2 > 1e-10).sum())], di + do


def symplectic(V, k):
    """The port pairing (v, xi), (v', xi') |-> xi'(v) - xi(v') on columns of V, for k values and k forces."""
    J = np.block([[np.zeros((k, k)), -np.eye(k)], [np.eye(k), np.zeros((k, k))]])
    return V.T @ J @ V


def test_linear_exact_critical_composite_rests_in_a_lagrangian_relation():
    for symmetric in (True, False):
        c = branching(*linear_pieces(symmetric))
        R, k = rest_relation_basis(c)
        assert R.shape[1] == k                                    # half the dimension of the port data
        lagrangian = np.allclose(symplectic(R, k), 0.0, atol=1e-10)
        assert lagrangian == symmetric


# ---------------------------------------------------------------------------
# Reciprocal: closed effort fields, nonlinear.
# ---------------------------------------------------------------------------


def nonlinear_system(closed):
    """1 -> <R|R> with two critical stocks, readout q0 + sin q1, cost k(q, n); a waterfall term breaks closedness."""
    def k(q, n):
        return 0.5 * q[0] ** 2 * q[1] + jnp.cos(q[0] * n[0]) + 0.25 * q[1] ** 4 + n[0] * q[1] + 0.5 * q[0] ** 2

    def omega(q, n):
        gq, gn = jax.grad(k, argnums=(0, 1))(q, n)
        xn = -gn + (0.0 if closed else 0.8 * q[0])
        return -gq, xn

    P = configuration(constant(jnp.eye(2)))
    return system(P, lambda q: jnp.array([q[0] + jnp.sin(q[1])]), omega, label="nonlin")


def test_nonlinear_closed_rest_set_is_isotropic():
    for closed in (True, False):
        a = nonlinear_system(closed)

        def balance(e):
            q, n, xo = e[:2], e[2:3], e[3:4]
            xq, _, _ = a.omega(q, Z0, n)
            _, vjp = jax.vjp(lambda qq: a.out_f(qq, Z0), q)
            return xq - vjp(xo)[0]

        def port(e):
            q, n, xo = e[:2], e[2:3], e[3:4]
            return jnp.concatenate([n, a.out_f(q, Z0), a.omega(q, Z0, n)[2], xo])

        worst, found = 0.0, 0
        while found < 4:
            e = jnp.array(RNG.normal(size=4))
            for _ in range(40):                                   # Newton on (n, xi_out), with q held
                F = balance(e)
                Ju = jax.jacfwd(balance)(e)[:, 2:]
                e = e.at[2:].add(-jnp.linalg.solve(Ju, F))
            if not jnp.allclose(balance(e), 0.0, atol=1e-10):
                continue                                          # no rest point from this start
            found += 1
            _, s, vt = np.linalg.svd(np.asarray(jax.jacfwd(balance)(e)))
            T = vt[2:].T                                          # the tangent plane of the rest set Z
            V = np.asarray(jax.jacfwd(port)(e)) @ T
            worst = max(worst, float(np.abs(symplectic(V, 2)).max()))
        assert (worst < 1e-9) == closed


def test_port_one_form_identity_on_an_arrangement_with_four_ports():
    """B^*(-theta_M + theta_N) = pi^* omega - <xi_q, dq>, the identity behind isotropy (RESULTS.md, reciprocal)."""
    def k(q, m, n):
        return q[0] * m[0] * n[0] + jnp.sin(q[0] + m[0]) + m[0] ** 2 * n[0] + 0.3 * q[0] ** 3

    a = Arrangement(P=rod(1.0), in_dim_M=1, out_dim_M=1, in_dim_N=1, out_dim_N=1,
                    out_f=lambda q, m: jnp.array([q[0] * m[0] + m[0] ** 2]),
                    in_f=lambda q, m, n: jnp.array([jnp.tanh(q[0] + n[0] * m[0])]),
                    omega=lambda q, m, n: tuple(-g for g in jax.grad(k, argnums=(0, 1, 2))(q, m, n)))

    def unpack(e):
        return e[0:1], e[1:2], e[2:3], e[3:4], e[4:5]            # m_out, q, n_in, xi_min, xi_nout

    def data(e):
        m, q, n, xmi, xno = unpack(e)
        wq, wm, wn = a.omega(q, m, n)
        _, vin = jax.vjp(lambda qq, mm, nn: a.in_f(qq, mm, nn), q, m, n)
        _, vout = jax.vjp(lambda qq, mm: a.out_f(qq, mm), q, m)
        iq, im, inn = vin(xmi)
        oq, om = vout(xno)
        xi_nin = wn + inn
        xi_mout = -(wm + im - om)
        xi_q = wq + iq - oq
        return dict(m_in=a.in_f(q, m, n), m_out=m, n_in=n, n_out=a.out_f(q, m),
                    xi_min=xmi, xi_mout=xi_mout, xi_nin=xi_nin, xi_nout=xno, xi_q=xi_q, w=(wm, wq, wn))

    for _ in range(4):
        e = jnp.array(RNG.normal(size=5))
        D = data(e)
        jac = {key: jax.jacfwd(lambda ee, key=key: data(ee)[key])(e) for key in ("m_in", "m_out", "n_in", "n_out")}
        lhs = (-jac["m_in"].T @ D["xi_min"] - jac["m_out"].T @ D["xi_mout"]
               + jac["n_in"].T @ D["xi_nin"] + jac["n_out"].T @ D["xi_nout"])
        wm, wq, wn = D["w"]
        rhs = jnp.concatenate([wm, wq - D["xi_q"], wn, jnp.zeros(2)])
        assert jnp.allclose(lhs, rhs, atol=1e-12)


# ---------------------------------------------------------------------------
# Mirror matter: state s, readout dF(s), update s + tau xi.
# ---------------------------------------------------------------------------


def mirror(n, F, init):
    dF = jax.grad(F)
    return CotCoalg(Q_dim=n, init=jnp.asarray(init), position=dF, step=lambda s, xi: s + xi,
                    timed_step=lambda s, xi, tau: s + tau * xi, label="mirror")


def lse(s):
    return jax.scipy.special.logsumexp(s)


def closed_from_cost(P, k, label):
    """The closed system whose matter P reads out to a stateless box with cost k(y)."""
    return Arrangement(P=P, in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: Z0, in_f=lambda q, m, n: Z0,
                       omega=lambda q, m, n: (-jax.grad(k)(q), Z0, Z0), label=label)


def composes_as_durations_add(z, t1, t2):
    s = z.P.init
    two = closed_step(z, check=0, duration=t2)(closed_step(z, check=0, duration=t1)(s))
    one = closed_step(z, check=1, duration=t1 + t2)(s)
    return bool(jnp.allclose(two, one, atol=1e-12))


def test_bayes_mirror_step_is_exact_and_coupled_beliefs_are_not():
    ell = jnp.log(jnp.array([0.2, 0.5, 0.3]))                    # log Pr(x | z_i)
    prior = jnp.log(jnp.array([0.5, 0.25, 0.25]))
    bayes = closed_from_cost(mirror(3, lse, prior), lambda y: -y @ ell, "bayes")
    post = jax.nn.softmax(closed_step(bayes, duration=1.0)(bayes.P.init))
    assert jnp.allclose(post, jnp.exp(prior + ell) / jnp.exp(prior + ell).sum())
    assert composes_as_durations_add(bayes, 0.3, 0.7)

    # two beliefs, cost affine in each and coupling them: a branching composite of mirror matter
    P2 = mirror(3, lse, prior).tensor(mirror(3, lse, jnp.zeros(3)))
    C = jnp.array(RNG.normal(size=(3, 3)))
    def k2(y):
        y1, y2 = y[:3], y[3:]
        return -y1 @ ell - y2 @ ell[::-1] + y1 @ C @ y2
    two = closed_from_cost(P2, k2, "two beliefs")
    s = two.P.init
    a = closed_step(two, check=0, duration=0.7)(closed_step(two, check=0, duration=0.3)(s))
    b = closed_step(two, check=0, duration=1.0)(s)
    assert not all(bool(jnp.allclose(x, y, atol=1e-6)) for x, y in zip(a, b))


# ---------------------------------------------------------------------------
# Quadratic matter: Euler integrators of constant symmetric responses.
# ---------------------------------------------------------------------------


def test_quadratic_matter_tangent_response_is_constant_and_symmetric():
    S1 = jnp.array([[2.0, 0.5], [0.5, 1.0]])
    P = configuration(constant(S1)).tensor(configuration(constant(jnp.array([[1 / 1.5]]))))
    sharp = jax.scipy.linalg.block_diag(S1, jnp.array([[1 / 1.5]]))
    for _ in range(3):
        s = (jnp.array(RNG.normal(size=2)), jnp.array(RNG.normal(size=1)))
        xi = jnp.array(RNG.normal(size=3))
        v = jax.jacfwd(lambda t: jnp.concatenate(P.step_over(s, xi, t)))(0.0)
        assert jnp.allclose(v, sharp @ xi)
    # a state change that is not linear destroys constancy: g(s) = s^3 + s on one rod
    g = lambda s: s ** 3 + s
    ginv_newton = lambda y: jax.lax.fori_loop(0, 60, lambda i, s: s - (g(s) - y) / (3 * s ** 2 + 1), y)
    def u_new(sp, xi, t):                                         # g(u(g^{-1}(sp), (Dg)^T xi))
        s = ginv_newton(sp)
        return g(s + t * (1 / 1.5) * (3 * s ** 2 + 1) * xi)
    vs = [jax.jacfwd(lambda t: u_new(sp, 1.0, t))(0.0) for sp in (0.0, 2.0)]
    assert not jnp.allclose(vs[0], vs[1])


# ---------------------------------------------------------------------------
# (l2) Open languages through the junction dictionary (P54 (a) Prop. C).
# ---------------------------------------------------------------------------


def stocks(n):
    return configuration(constant(jnp.eye(n))) if n else trivial_parameter()


def reaction_piece(owned, boundary, transitions):
    """An open reaction network as a system 1 -> <R^boundary | 1>: owned species are unit-mobility stocks, boundary
    species external inputs. A transition (s, t, r), s and t dicts species -> count, returns at every species
    sigma the force r c^s (t - s)_sigma (Baez--Pollard's reaction velocity), a stock-flow forcing."""
    names = list(owned) + list(boundary)

    def omega(q, m, n):
        c = dict(zip(names, jnp.concatenate([q, n])))
        xi = {x: 0.0 for x in names}
        for s, t, r in transitions:
            flux = r * jnp.prod(jnp.array([c[x] ** k for x, k in s.items()] or [1.0]))
            for x in names:
                xi[x] = xi[x] + flux * (t.get(x, 0) - s.get(x, 0))
        v = jnp.array([xi[x] for x in names])
        return v[: len(owned)], Z0, v[len(owned):]

    return Arrangement(P=stocks(len(owned)), in_dim_M=0, out_dim_M=0, in_dim_N=len(boundary), out_dim_N=0,
                       out_f=lambda q, m: Z0, in_f=lambda q, m, n: Z0, omega=omega,
                       label=f"rx[{','.join(owned)}|{','.join(boundary)}]")


def junction(inner_ports, outer_ports, pi):
    """P54 (a)'s u_Phi: inner ports (a list of names, each box's in order) to junctions by pi; junctions not named by
    an outer port are stocks; each inner port is delivered its junction's level or the outer port's value."""
    interior = sorted({pi[p] for p in inner_ports} - {pi[p] for p in outer_ports})
    outer_of = {pi[p]: i for i, p in enumerate(outer_ports)}

    def in_f(q, m, n):
        lv = {j: q[i] for i, j in enumerate(interior)}
        return jnp.array([lv[pi[p]] if pi[p] in lv else n[outer_of[pi[p]]] for p in inner_ports])

    return Arrangement(P=stocks(len(interior)), in_dim_M=len(inner_ports), out_dim_M=0,
                       in_dim_N=len(outer_ports), out_dim_N=0, out_f=lambda q, m: Z0, in_f=in_f,
                       omega=lambda q, m, n: (jnp.zeros(len(interior)), Z0, jnp.zeros(len(outer_ports))),
                       label="junction"), interior


def rate_equation(species, transitions):
    """Baez--Pollard's v^R(c) = sum_tau r (t - s) c^s (arXiv:1704.02051, Sec. 5, Def. 13 and the equation after it)."""
    def v(c):
        cc = dict(zip(species, c))
        out = jnp.zeros(len(species))
        for s, t, r in transitions:
            flux = r * jnp.prod(jnp.array([cc[x] ** k for x, k in s.items()] or [1.0]))
            out = out + flux * jnp.array([t.get(x, 0) - s.get(x, 0) for x in species])
        return out
    return v


T1 = [({"A": 2}, {"B": 1}, 1.3), ({"A": 1, "X": 1}, {"X": 2}, 0.7)]
T2 = [({"B": 1, "C": 1}, {"C": 2}, 0.5), ({"C": 1}, {}, 0.4), ({"B2": 1}, {"C": 1}, 0.9)]


def test_reaction_networks_glue_as_nesting_in_a_junction():
    p1 = reaction_piece(["X"], ["A", "B"], T1)
    p2 = reaction_piece(["C"], ["B", "B2"], T2)                   # B and B2 will be one species: two ports, one junction
    inner = ["1A", "1B", "2B", "2B2"]
    pi = {"1A": "A", "1B": "B", "2B": "B", "2B2": "B"}

    # closed: every species owned, by its piece or by the junction
    u, interior = junction(inner, [], pi)
    whole = compose_seq(parallel_arrangements(p1, p2), u)
    glued = [(s, t, r) for s, t, r in T1] + [({"B" if x == "B2" else x: k for x, k in s.items()},
                                             {"B" if x == "B2" else x: k for x, k in t.items()}, r) for s, t, r in T2]
    order = ["X", "C"] + interior                                  # the composite's matter: piece 1, piece 2, junction
    v = rate_equation(order, glued)
    for _ in range(4):
        c = jnp.array(RNG.uniform(0.1, 2.0, size=len(order)))
        xi_q, _, _ = whole.omega(c, Z0, Z0)
        assert jnp.allclose(xi_q, v(c), atol=1e-12)              # unit mobilities: tangent dynamics is v^R

    # open: species B stays on the boundary, read as the glued network with boundary {B}
    u_open, interior_open = junction(inner, ["0B"], {**pi, "0B": "B"})
    composite = compose_seq(parallel_arrangements(p1, p2), u_open)
    reading = reaction_piece(["X", "C"] + interior_open, ["B"], glued)
    for _ in range(4):
        q = jnp.array(RNG.uniform(0.1, 2.0, size=composite.P.Q_dim))
        n = jnp.array(RNG.uniform(0.1, 2.0, size=1))
        for x, y in zip(composite.omega(q, Z0, n), reading.omega(q, Z0, n)):
            assert jnp.allclose(x, y, atol=1e-12)


def resistor_piece(interior, boundary, edges):
    """A resistor circuit with boundary (Baez--Fong Def. 2.2, arXiv:1504.05625) as a system 1 -> <R^boundary | 1>: a stock at each
    interior node, cost the extended power functional P = 1/2 sum (1/r)(phi_t - phi_s)^2 (their Def. 2.3)."""
    names = list(interior) + list(boundary)

    def P(q, n):
        phi = dict(zip(names, jnp.concatenate([q, n])))
        return 0.5 * sum((phi[t] - phi[s]) ** 2 / r for s, t, r in edges)

    def omega(q, n):
        gq, gn = jax.grad(P, argnums=(0, 1))(q, n)
        return -gq, -gn

    return system(stocks(len(interior)), lambda q: Z0, omega, in_N=len(boundary), out_N=0, label="circuit"), P


def black_box(names_int, names_bd, P):
    """{(psi, -dQ_psi)}: Baez--Fong's Graph(dP) restricted by Kirchhoff's current law inside, with the book's sign
    (outward force, the negated boundary current); a basis of columns."""
    ni, nb = len(names_int), len(names_bd)
    H = jax.hessian(lambda x: P(x[:ni], x[ni:]))(jnp.zeros(ni + nb))
    Hii, Hib, Hbb = H[:ni, :ni], H[:ni, ni:], H[ni:, ni:]
    K = Hbb - Hib.T @ jnp.linalg.solve(Hii, Hib)                  # the Schur complement: dQ = K psi
    return np.asarray(jnp.concatenate([jnp.eye(nb), -K]))


def same_subspace(A, B):
    return np.linalg.matrix_rank(np.hstack([A, B]), tol=1e-9) == np.linalg.matrix_rank(A, tol=1e-9) == \
        np.linalg.matrix_rank(B, tol=1e-9)


def test_resistor_circuits_rest_as_their_black_box():
    e1 = [("a", "m", 2.0), ("m", "b", 1.0), ("m", "c", 3.0), ("a", "b", 4.0)]
    e2 = [("b", "n", 0.5), ("n", "d", 1.5), ("c", "n", 2.5)]
    c1, P1 = resistor_piece(["m"], ["a", "b", "c"], e1)
    c2, P2 = resistor_piece(["n"], ["b", "c", "d"], e2)
    inner = ["1a", "1b", "1c", "2b", "2c", "2d"]
    pi = {"1a": "a", "1b": "b", "1c": "c", "2b": "b", "2c": "c", "2d": "d", "0a": "a", "0d": "d"}
    u, interior = junction(inner, ["0a", "0d"], pi)               # glue along b and c; keep a and d as terminals
    composite = compose_seq(parallel_arrangements(c1, c2), u)
    R, k = rest_relation_basis(composite)
    assert R.shape[1] == k == 2
    whole_int = ["m", "n"] + interior
    _, P = resistor_piece(whole_int, ["a", "d"], e1 + e2)
    assert same_subspace(R, black_box(whole_int, ["a", "d"], P))
    assert np.allclose(symplectic(R, 2), 0.0, atol=1e-10)


# ---------------------------------------------------------------------------
# (l3) Learners: Para over Euclidean spaces with isometric reparametrization, into arr.
# ---------------------------------------------------------------------------


def learner(dP, dA, dB, I, eta):
    """G_eta(P, I): <1|R^dA> -> <1|R^dB>, matter the gradient integrator p |-> p + tau eta xi on P = R^dP,
    stateless arrangement static with readout I(p, a)."""
    return Arrangement(P=configuration(constant(eta * jnp.eye(dP))), in_dim_M=0, out_dim_M=dA, in_dim_N=0,
                       out_dim_N=dB, out_f=lambda q, m: I(q, m), in_f=lambda q, m, n: Z0,
                       omega=lambda q, m, n: (jnp.zeros(dP), jnp.zeros(dA), Z0), label="learner")


def data_and_loss(x, ell, dB):
    """The datum x as a stateless system 1 -> <1|R^dA>, and the loss box <1|R^dB> -> 1 with cost 1/2 |y - ell|^2."""
    datum = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=x.shape[0],
                        out_f=lambda q, m: x, in_f=lambda q, m, n: Z0,
                        omega=lambda q, m, n: (Z0, Z0, Z0), label="datum")
    loss = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=dB, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: Z0, in_f=lambda q, m, n: Z0,
                       omega=lambda q, m, n: (Z0, -(m - ell), Z0), label="loss")
    return datum, loss


def test_learners_compose_and_descend_the_composite_loss():
    eta, tau = 0.3, 0.5
    I1 = lambda p, a: jnp.tanh(p[:2] * a[0] + p[2:])               # R -> R^2, parameters R^4
    I2 = lambda p, a: jnp.array([p[0] * a[0] ** 2 + jnp.sin(p[1])])  # R -> R, parameters R^2
    I3 = lambda p, a: jnp.array([p[0] * a[0] + p[1] * a[1] * a[2] + p[2]])  # R^3 -> R, parameters R^3
    l1, l2, l3 = learner(4, 1, 2, I1, eta), learner(2, 1, 1, I2, eta), learner(3, 3, 1, I3, eta)
    copy = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=1, in_dim_N=0, out_dim_N=2,
                       out_f=lambda q, m: jnp.concatenate([m, m]), in_f=lambda q, m, n: Z0,
                       omega=lambda q, m, n: (Z0, jnp.zeros(1), Z0), label="copy")
    net = compose_seq(compose_seq(copy, parallel_arrangements(l1, l2)), l3)   # branching: x feeds both l1 and l2
    x, ell = jnp.array([0.7]), jnp.array([0.2])
    datum, loss = data_and_loss(x, ell, 1)
    closed = compose_seq(compose_seq(datum, net), loss)
    p0 = jnp.array(RNG.normal(size=9))

    def composite_loss(p):
        h = jnp.concatenate([I1(p[:4], x), I2(p[4:6], x)])
        return 0.5 * jnp.sum((I3(p[6:], h) - ell) ** 2)

    leaves, tree = jax.tree_util.tree_flatten(closed.P.init)
    sizes = [l.shape[0] for l in leaves]
    s0 = jax.tree_util.tree_unflatten(tree, list(jnp.split(p0, np.cumsum(sizes)[:-1])))
    step = closed_step(closed, check=0, duration=tau)
    p1 = jnp.concatenate(jax.tree_util.tree_leaves(step(s0)))
    assert jnp.allclose(p1, p0 - tau * eta * jax.grad(composite_loss)(p0), atol=1e-12)


def test_learner_reparametrization_isometric_or_not():
    """An orthogonal change of parameters gives an isomorphic arrangement; q = 2p (P53 (j) Sec. 6) does not."""
    eta, tau = 0.3, 0.5
    I = lambda p, a: jnp.array([p[0] * a[0] + jnp.sin(p[1]) * a[0] ** 2])
    x, ell = jnp.array([0.9]), jnp.array([-0.4])
    datum, loss = data_and_loss(x, ell, 1)

    def stepped(I_, p):
        closed = compose_seq(compose_seq(datum, learner(2, 1, 1, I_, eta)), loss)
        leaves, tree = jax.tree_util.tree_flatten(closed.P.init)
        sizes = [l.shape[0] for l in leaves]
        s = jax.tree_util.tree_unflatten(tree, list(jnp.split(p, np.cumsum(sizes)[:-1])))
        return jnp.concatenate(jax.tree_util.tree_leaves(closed_step(closed, check=0, duration=tau)(s)))

    p = jnp.array([0.3, -1.1])
    c, s_ = np.cos(0.8), np.sin(0.8)
    O = jnp.array([[c, -s_], [s_, c]])
    I_rot = lambda q, a: I(O.T @ q, a)                             # the same learner in rotated parameters q = O p
    assert jnp.allclose(stepped(I_rot, O @ p), O @ stepped(I, p), atol=1e-12)
    I_dbl = lambda q, a: I(q / 2, a)                               # q = 2p
    assert not jnp.allclose(stepped(I_dbl, 2 * p), 2 * stepped(I, p), atol=1e-6)


# ---------------------------------------------------------------------------
# (l4) The prediction market and Hebbian learners: single operations are arrangements; their compositors are
# maps of unit steps, not morphisms of arrangements.
# ---------------------------------------------------------------------------


def multiplicative_weights(n, init):
    """State and readout a trust split mu in the open simplex; update mu * exp(tau xi), normalized. Well defined on
    covectors of the simplex (adding a constant to xi changes nothing); at tau = 1 with xi_i = log g_i it is Bayes."""
    def upd(mu, xi, tau):
        w = mu * jnp.exp(tau * (xi - xi.max()))
        return w / w.sum()
    return CotCoalg(Q_dim=n, init=jnp.asarray(init), position=lambda s: s, step=lambda s, xi: upd(s, xi, 1.0),
                    timed_step=upd, label="trust")


def trust_box(n, X, init):
    """The market's operation of arity n as an arrangement <X|Bet>^n -> <X|Bet> (adaptives.tex sec.kelley): readout
    the trust-weighted guess, delivery the outcome copied to every member, effort field log gamma_i(x) on the trust
    only (not exact: members are not pushed). Outcomes are one-hot values here, standing in for the discrete port."""
    def out_f(mu, g):
        return (mu[:, None] * g.reshape(n, X)).sum(0)

    def in_f(mu, g, x):
        return jnp.tile(x, n)

    def omega(mu, g, x):
        lik = g.reshape(n, X) @ x
        return jnp.log(lik), jnp.zeros(n * X), jnp.zeros(X)

    return Arrangement(P=multiplicative_weights(n, init), in_dim_M=n * X, out_dim_M=n * X, in_dim_N=X, out_dim_N=X,
                       out_f=out_f, in_f=in_f, omega=omega, label=f"trust{n}")


def guesses(G):
    """Fixed guesses (rows of G) as a stateless system 1 -> <X|Bet>^k."""
    k, X = G.shape
    return Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=k * X, out_dim_N=k * X,
                       out_f=lambda q, m: G.reshape(-1), in_f=lambda q, m, n: Z0,
                       omega=lambda q, m, n: (Z0, Z0, jnp.zeros(k * X)), label="guesses")


def outcome(x):
    """Closes <X|Bet>: delivers the outcome x, and pushes nothing."""
    X = x.shape[0]
    return Arrangement(P=trivial_parameter(), in_dim_M=X, out_dim_M=X, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: Z0, in_f=lambda q, m, n: x,
                       omega=lambda q, m, n: (Z0, jnp.zeros(X), Z0), label="outcome")


def test_market_composite_runs_the_compositor_at_unit_duration_only():
    X = 3
    G = jax.nn.softmax(jnp.array(RNG.normal(size=(4, X))), axis=1)   # four guessers, two under each inner trust
    mu, nu1, nu2 = jnp.array([0.6, 0.4]), jnp.array([0.3, 0.7]), jnp.array([0.5, 0.5])
    x = jnp.eye(X)[1]
    inner = parallel_arrangements(trust_box(2, X, nu1), trust_box(2, X, nu2))
    nested = compose_seq(compose_seq(compose_seq(guesses(G), inner), trust_box(2, X, mu)), outcome(x))
    flat_mu = jnp.concatenate([mu[0] * nu1, mu[1] * nu2])           # the compositor mu o nu
    flat = compose_seq(compose_seq(guesses(G), trust_box(4, X, flat_mu)), outcome(x))
    def trusts(arr, tau):
        return [l for l in jax.tree_util.tree_leaves(closed_step(arr, duration=tau)(arr.P.init)) if l.shape[0]]

    for tau, agree in ((1.0, True), (0.5, False)):
        n1, n2, m = trusts(nested, tau)
        (f,) = trusts(flat, tau)
        assert bool(jnp.allclose(jnp.concatenate([m[0] * n1, m[1] * n2]), f, atol=1e-12)) == agree
    (f,) = trusts(flat, 1.0)
    assert jnp.allclose(f, flat_mu * (G @ x) / (flat_mu @ (G @ x)))   # the market's update is Bayes on trust


def hebbian_box(N, alpha, beta, W0):
    """Libkind--Spivak's Hebbian operation (draft, Hebbian-Learners/02) as an arrangement <R^N|R^2N> -> <R^N|R^2N>:
    matter the interaction pattern W (W |-> W + tau xi), readout the outputs passed on, delivery I + W O_0, and the
    effort field on W that makes the unit step beta (W + alpha W . (O_0 O_1^T)); it is not closed."""
    P = configuration(constant(jnp.eye(N * N)), init=W0.reshape(-1))

    def omega(w, o, i):
        W = w.reshape(N, N)
        xi = (beta - 1.0) * W + beta * alpha * W * jnp.outer(o[:N], o[N:])
        return xi.reshape(-1), jnp.zeros(2 * N), jnp.zeros(N)

    return Arrangement(P=P, in_dim_M=N, out_dim_M=2 * N, in_dim_N=N, out_dim_N=2 * N,
                       out_f=lambda w, o: o, in_f=lambda w, o, i: i + w.reshape(N, N) @ o[:N],
                       omega=omega, label="hebb")


def test_hebbian_steps_match_the_draft_and_its_compositor_is_not_a_morphism():
    N, alpha, beta = 2, 0.3, 0.9
    W1, W2 = jnp.array(RNG.normal(size=(N, N))), jnp.array(RNG.normal(size=(N, N)))
    O = jnp.array(RNG.normal(size=2 * N))
    src = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=N, out_dim_N=2 * N,
                      out_f=lambda q, m: O, in_f=lambda q, m, n: Z0,
                      omega=lambda q, m, n: (Z0, Z0, jnp.zeros(N)), label="channels")
    sink = Arrangement(P=trivial_parameter(), in_dim_M=N, out_dim_M=2 * N, in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m: Z0, in_f=lambda q, m, n: jnp.zeros(N),
                       omega=lambda q, m, n: (Z0, jnp.zeros(2 * N), Z0), label="sink")
    two = compose_seq(compose_seq(compose_seq(src, hebbian_box(N, alpha, beta, W1)), hebbian_box(N, alpha, beta, W2)), sink)
    upd = lambda W: beta * (W + alpha * W * jnp.outer(O[:N], O[N:]))
    leaves = jax.tree_util.tree_leaves(closed_step(two, duration=1.0)(two.P.init))
    w1, w2 = [l for l in leaves if l.shape[0] == N * N]
    assert jnp.allclose(w1.reshape(N, N), upd(W1)) and jnp.allclose(w2.reshape(N, N), upd(W2))
    assert jnp.allclose(w1 + w2, upd(W1 + W2).reshape(-1))          # the compositor W + W' commutes with unit steps
    # but it is not a morphism of arrangements: pulled back along (W, W') |-> W + W', the single box's effort field
    # puts the same covector on both factors, while the composite's puts each factor's own
    h = hebbian_box(N, alpha, beta, W1 + W2)
    xi_single = h.omega((W1 + W2).reshape(-1), O, jnp.zeros(N))[0]
    xi_1 = hebbian_box(N, alpha, beta, W1).omega(W1.reshape(-1), O, jnp.zeros(N))[0]
    assert not jnp.allclose(xi_single, xi_1)
    J = jax.jacfwd(lambda o: hebbian_box(N, alpha, beta, W1).omega(W1.reshape(-1), o, jnp.zeros(N))[0])(O)
    assert not jnp.allclose(J, 0.0)                                  # the force on W depends on the outputs: d omega != 0
