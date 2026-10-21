"""Exponential-family operads as data, and Gaussian evidence trees (book P53 (h)).

Predictions, fixed before ``arrangements/evidence.py`` and ``arrangements/expfamily.py`` existed, and verdicts: the
book's ``notes/runs/p53h-82601336/RESULTS.md``. Each docstring says which of three things its test does: reproduce
an operad law (substitution against lens composition), check a closed form, or check against a reference computed
beside the arrangements (the joint Gaussian of a tree model, by linear algebra).
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.active_inference import run
from arrangements.attention import (
    Transfer, attention_arrangement, compose_transfers, substitute as att_substitute, zero_cost, zero_transfer,
    identity_transfer,
)
from arrangements.coalgebra import configuration
from arrangements.evidence import (
    SLOT, Link, Node, corolla, datum, disagreement, down, graft, identity, log_normalizer, prior_cap, up,
)
from arrangements.expfamily import (
    ExpFamilyOperation, cholesky_precision, gaussian_term, substitute as ef_substitute,
)
from arrangements.functors import Phi
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.chain import flat_state

jax.config.update("jax_enable_x64", True)


def rot(b):
    c, s = jnp.cos(b), jnp.sin(b)
    return jnp.array([[c, -s], [s, c]])


def positive_messages(r, n, rng):
    h = rng.standard_normal((n, r))
    w = np.exp(rng.standard_normal((n, 1)) * 0.5)
    return jnp.asarray(np.concatenate([h, w], axis=1).ravel())


def trajectory(arr, steps, rng, positive=True, r=2):
    """``Phi(arr)`` for ``steps`` steps on fixed exterior data; positive messages where the family needs them."""
    k = r + 1
    m_out = positive_messages(r, arr.out_dim_M // k, rng) if positive else jnp.asarray(rng.standard_normal(arr.out_dim_M))
    B = jnp.asarray(rng.standard_normal((arr.in_dim_M, arr.in_dim_M)) * 0.05)
    c = jnp.asarray(rng.standard_normal(arr.in_dim_M) * 0.05)
    xi_N = jnp.asarray(rng.standard_normal(arr.out_dim_N) * 0.1)
    e = positive_messages(r, 1, rng) if positive else jnp.asarray(rng.standard_normal(arr.in_dim_N))
    O = Phi(arr)
    state, out = O.state, [flat_state(O.state)]
    for _ in range(steps):
        _, _, state = O.with_state(state).run_one((m_out, lambda x: B @ x + c), lambda _o: (xi_N, e))
        out.append(flat_state(state))
    return out


def agree(x, y, rng, positive=True, r=2, steps=40, atol=1e-11):
    """Two arrangements with the same ports and parameter agree in sending, receiving, the forcing and ``Phi``."""
    assert (x.in_dim_M, x.out_dim_M, x.in_dim_N, x.out_dim_N) == (y.in_dim_M, y.out_dim_M, y.in_dim_N, y.out_dim_N)
    assert x.P.Q_dim == y.P.Q_dim
    k = r + 1
    worst = 0.0
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(x.P.Q_dim) * 0.5)
        m = positive_messages(r, x.out_dim_M // k, rng) if positive else jnp.asarray(rng.standard_normal(x.out_dim_M))
        e = positive_messages(r, 1, rng) if positive else jnp.asarray(rng.standard_normal(x.in_dim_N))
        pairs = [(x.out_f(q, m), y.out_f(q, m)), (x.in_f(q, m, e), y.in_f(q, m, e))]
        pairs += list(zip(x.omega(q, m, e), y.omega(q, m, e)))
        for u, v in pairs:
            worst = max(worst, float(jnp.max(jnp.abs(u - v))) if u.size else 0.0)
    assert worst < atol, worst
    t1, t2 = trajectory(x, steps, np.random.default_rng(0), positive, r), trajectory(y, steps, np.random.default_rng(0), positive, r)
    tw = max(float(jnp.max(jnp.abs(u - v))) for u, v in zip(t1, t2))
    assert tw < atol, tw
    assert float(jnp.max(jnp.abs(t1[-1] - t1[0]))) > 1e-3   # the run moves
    return worst, tw


# ---------------------------------------------------------------------------
# Prediction 1: the Gaussian with a precision function, 2 x 2.
# ---------------------------------------------------------------------------

R1, DP1 = 2, 2
I1, Z1 = identity_transfer(R1), zero_transfer(R1)


def _transfers(n):
    if n == 0:
        return lambda q: ((Z1,),)
    return lambda q: ((Z1, I1), (I1, Z1))


def g_outer():
    """1-ary: configuration ``(mu, chol)``; places its child at ``mu``; terms: the exterior signal predicted by ``mu``,
    and its child's message signal predicted by ``mu / 2``, both with precision ``Pi_a(q)``."""
    Pi = cholesky_precision(R1, 2)
    base = attention_arrangement(1, R1, DP1, placement=lambda q: q[:2][None, :], transfers=_transfers(1),
                                 cost=zero_cost, parameter=configuration(euclidean(5, 0.01)), label="outer")
    t1 = gaussian_term(R1, lambda q, m, e: q[:2], Pi, lambda q, m, e: e[:2])
    t2 = gaussian_term(R1, lambda q, m, e: 0.5 * q[:2], Pi, lambda q, m, e: m[:2])
    return ExpFamilyOperation(base, [t1, t2])


def g_middle(arity):
    """``arity``-ary: configuration ``(nu, chol)``; term: its received position predicted by ``nu``; places a child at
    ``nu`` when it has one."""
    Pi = cholesky_precision(R1, 2)
    place = (lambda q: q[:2][None, :]) if arity else (lambda q: jnp.zeros((0, DP1)))
    base = attention_arrangement(arity, R1, DP1, placement=place, transfers=_transfers(arity), cost=zero_cost,
                                 parameter=configuration(euclidean(5, 0.01)), label=f"middle{arity}")
    return ExpFamilyOperation(base, [gaussian_term(R1, lambda q, m, e: q[:2], Pi, lambda q, m, e: e[3:5])])


def g_inner():
    """0-ary: configuration ``chol``; term: its received position predicted by the fixed point ``(0.3, -0.2)``."""
    Pi = cholesky_precision(R1, 0)
    base = attention_arrangement(0, R1, DP1, placement=lambda q: jnp.zeros((0, DP1)), transfers=_transfers(0),
                                 cost=zero_cost, parameter=configuration(euclidean(3, 0.01)), label="inner")
    eta = jnp.array([0.3, -0.2])
    return ExpFamilyOperation(base, [gaussian_term(R1, lambda q, m, e: eta, Pi, lambda q, m, e: e[3:5])])


def test_gaussian_precision_terms_compose():
    """REPRODUCES the operad law for term lists, and CHECKS A CLOSED FORM: the term-list substitution's cost equals
    the attention operad's own composite cost (computed by ``attention.substitute``), the hand formula
    ``sum 1/2 (y - mu)^T Pi (y - mu) - 1/2 log det Pi``, and the lens composite's forcing; the force on the observed
    exterior signal is ``Pi_a (mu - y)``."""
    a, c = g_outer(), g_middle(0)
    b = ef_substitute(a, [c])
    via_attention = att_substitute(a.attention(), [c.attention()])
    lens = compose_seq(c.attention().arr(), a.attention().arr())
    Pa, P1 = cholesky_precision(R1, 2), cholesky_precision(R1, 2)
    rng = np.random.default_rng(11)
    worst = {"att": 0.0, "hand": 0.0, "force": 0.0}
    for _ in range(6):
        q = jnp.asarray(rng.standard_normal(10) * 0.5)          # (nu, chol_1) then (mu, chol_a)
        e = jnp.asarray(rng.standard_normal(5))
        m = jnp.zeros(0)
        k = b.cost(q, m, e)
        worst["att"] = max(worst["att"], abs(float(k - via_attention.cost(q, m, e))))
        qc, qa = q[:5], q[5:]
        mu, nu, h0, p0 = qa[:2], qc[:2], e[:2], e[3:5]
        Pi_a, Pi_1 = Pa(qa), P1(qc)
        hand = (0.5 * (h0 - mu) @ Pi_a @ (h0 - mu) + 0.5 * (0.5 * mu) @ Pi_a @ (0.5 * mu) - jnp.linalg.slogdet(Pi_a)[1]
                + 0.5 * (p0 + mu - nu) @ Pi_1 @ (p0 + mu - nu) - 0.5 * jnp.linalg.slogdet(Pi_1)[1])
        worst["hand"] = max(worst["hand"], abs(float(k - hand)))
        xi = lens.omega(q, m, e)
        worst["force"] = max(worst["force"], float(jnp.max(jnp.abs(xi[2][:2] - Pi_a @ (mu - h0)))))
    print("prediction 1, two levels:", worst)
    assert worst["att"] < 1e-12 and worst["hand"] < 1e-12 and worst["force"] < 1e-12
    print("prediction 1, b against the lens composite:", agree(b.attention().arr(), lens, rng, positive=False))


def test_gaussian_precision_terms_associate():
    """REPRODUCES unary-chain cost associativity, not ordered-list equality on branching trees. Terms are taken modulo permutations (P53(j), b2b2dfd4): placement,
    sending, receiving, cost and forcing."""
    a, c, d = g_outer(), g_middle(1), g_inner()
    left = ef_substitute(ef_substitute(a, [c]), [d])
    right = ef_substitute(a, [ef_substitute(c, [d])])
    rng = np.random.default_rng(12)
    worst = 0.0
    for _ in range(6):
        q = jnp.asarray(rng.standard_normal(13) * 0.5)
        e = jnp.asarray(rng.standard_normal(5))
        m = jnp.zeros(0)
        worst = max(worst, abs(float(left.cost(q, m, e) - right.cost(q, m, e))))
        for u, v in zip(left.attention().arr().omega(q, m, e), right.attention().arr().omega(q, m, e)):
            worst = max(worst, float(jnp.max(jnp.abs(u - v))) if u.size else 0.0)
    print("prediction 1, associativity:", worst)
    assert worst < 1e-12


# ---------------------------------------------------------------------------
# Evidence trees: the nodes of the examples.
# ---------------------------------------------------------------------------

R = 2
K3 = R + 1


def _link(q, i):
    a, b, c = q[3 * i], q[3 * i + 1], q[3 * i + 2]
    return Link(jnp.exp(a), rot(b), jnp.exp(c))


def leaf_node(y, name):
    """A datum ``y`` seen with precision ``e^{q_0}``; no children."""
    y = jnp.asarray(y, float)
    return Node(configuration(euclidean(1, 0.02)), lambda q: datum(y, jnp.exp(q[0]))[None, :], lambda q: [], name)


def mid_node(n, y=None, name="mid"):
    """``n`` children with links read from the configuration, three numbers each (``log kappa``, angle, ``log s``),
    then, if ``y`` is given, a datum seen with precision ``e^{q_{3n}}``."""
    dim = 3 * n + (1 if y is not None else 0)
    if y is None:
        own = lambda q: jnp.zeros((0, K3))
    else:
        y = jnp.asarray(y, float)
        own = lambda q: datum(y, jnp.exp(q[3 * n]))[None, :]
    return Node(configuration(euclidean(dim, 0.02)), own, lambda q: [_link(q, i) for i in range(n)], name)


DATA = {f"l{j}": np.random.default_rng(100 + j).standard_normal(R) for j in range(1, 8)}
DATA["A"] = np.random.default_rng(99).standard_normal(R)
PRIOR = datum(jnp.array([0.2, -0.1]), 0.5)


def nodes():
    out = {k: leaf_node(v, k) for k, v in DATA.items() if k.startswith("l")}
    out["A"] = mid_node(3, DATA["A"], "A")
    out["B"] = mid_node(4, None, "B")
    out["R"] = mid_node(2, None, "R")
    return out


SPEC = ("R", [("A", ["l1", "l2", "l3"]), ("B", ["l4", "l5", "l6", "l7"])])


def build(spec, hole=None):
    """The tree of ``spec`` with the subtree named ``hole`` replaced by ``SLOT``; and its nodes in configuration
    order (children's subtrees first, then the node)."""
    N = nodes()

    def go(s):
        name, kids = (s, []) if isinstance(s, str) else s
        if name == hole:
            return SLOT, []
        built, order = [], []
        for k in kids:
            t, o = go(k)
            built.append(t)
            order += o
        return graft(corolla(R, N[name], len(kids)), built) if kids else corolla(R, N[name], 0), order + [name]

    return go(spec)


def config_of(order, qd):
    return jnp.concatenate([qd[n] for n in order]) if order else jnp.zeros(0)


def random_config(rng, scale=0.5):
    N = nodes()
    return {k: jnp.asarray(rng.standard_normal(v.P.Q_dim) * scale) for k, v in N.items()}


# ---------------------------------------------------------------------------
# Prediction 2: the operad laws.
# ---------------------------------------------------------------------------


def test_grafting_is_lens_composition():
    """REPRODUCES the operad morphism: a grafted tree evaluated node by node equals the lens composite of its parts,
    ``compose_seq(tensor(A', B'), R)``, in sending, receiving, the three forcing components and 40 steps of ``Phi``;
    links with ``kappa != 1``, a rotation and ``s > 0``, all read from the configuration."""
    N = nodes()
    A_open = graft(corolla(R, N["A"], 3), [corolla(R, N["l1"], 0), SLOT, corolla(R, N["l3"], 0)])
    B_open = graft(corolla(R, N["B"], 4), [SLOT, corolla(R, N["l5"], 0), corolla(R, N["l6"], 0), corolla(R, N["l7"], 0)])
    Rc = corolla(R, N["R"], 2)
    explicit = graft(Rc, [A_open, B_open]).arr()
    lens = compose_seq(tensor_arrangements([A_open.arr(), B_open.arr()]), Rc.arr())
    print("prediction 2, grafting against compose_seq (worst, trajectory):", agree(explicit, lens, np.random.default_rng(21)))
    # with a bare slot grafted: the identity
    explicit2 = graft(Rc, [A_open, SLOT]).arr()
    lens2 = compose_seq(tensor_arrangements([A_open.arr(), identity(R)]), Rc.arr())
    print("prediction 2, with the bare slot:", agree(explicit2, lens2, np.random.default_rng(22)))


def test_grafting_associates_and_has_a_unit():
    """REPRODUCES associativity and the unit law: the two bracketings of a three-level graft, and grafting bare slots,
    give the same sending, receiving and cost."""
    N = nodes()
    Rc, Ac, Bc = corolla(R, N["R"], 2), corolla(R, N["A"], 3), corolla(R, N["B"], 4)
    leaves = [corolla(R, N[f"l{j}"], 0) for j in range(1, 8)]
    left = graft(graft(Rc, [Ac, Bc]), leaves[:2] + [SLOT] + leaves[3:6] + [SLOT])
    right = graft(Rc, [graft(Ac, leaves[:2] + [SLOT]), graft(Bc, leaves[3:6] + [SLOT])])
    unit = graft(right, [SLOT, SLOT])
    rng = np.random.default_rng(23)
    worst = 0.0
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(right.Q_dim) * 0.5)
        m, e = positive_messages(R, 2, rng), positive_messages(R, 1, rng)
        for t in (left, unit):
            for u, v in [(t.sigma(q, m), right.sigma(q, m)), (t.rho(q, m, e), right.rho(q, m, e)),
                         (t.cost(q, m), right.cost(q, m))]:
                worst = max(worst, float(jnp.max(jnp.abs(u - v))))
    print("prediction 2, associativity and unit:", worst)
    assert worst < 1e-12


# ---------------------------------------------------------------------------
# The reference beside the arrangements: the joint Gaussian of the tree model.
# ---------------------------------------------------------------------------


def joint(spec, qd, prior):
    """The linear-Gaussian tree model of ``spec`` at the configurations ``qd``: latents ``z_v`` in node order,
    ``z_root ~ N(prior mean, 1/prior weight)``, ``z_c = kappa U z_parent + N(0, s)``, each own datum
    ``y = z_v + N(0, 1/pi)``. Returns the data's mean, covariance and values, and each latent's mean and covariance
    with the latents-data cross covariance."""
    N = nodes()
    names, parent, link = [], {}, {}

    def go(s, par, lk):
        name, kids = (s, []) if isinstance(s, str) else s
        names.append(name)
        parent[name], link[name] = par, lk
        links = N[name].links(qd[name])
        for i, k in enumerate(kids):
            go(k, name, links[i])

    go(spec, None, None)
    V = len(names)
    idx = {n: i for i, n in enumerate(names)}
    G = jnp.zeros((V * R, V * R))
    D = jnp.zeros((V * R, V * R))
    b = jnp.zeros(V * R)
    for n in names:
        i = idx[n]
        sl = slice(i * R, (i + 1) * R)
        if parent[n] is None:
            b = b.at[sl].set(prior[:R] / prior[R])
            D = D.at[sl, sl].set(jnp.eye(R) / prior[R])
        else:
            L, j = link[n], idx[parent[n]]
            G = G.at[sl, j * R:(j + 1) * R].set(L.kappa * L.U)
            D = D.at[sl, sl].set(jnp.eye(R) * L.s)
    Minv = jnp.linalg.inv(jnp.eye(V * R) - G)
    mz, Cz = Minv @ b, Minv @ D @ Minv.T
    rows, ys, noise = [], [], []
    for n in names:
        own = N[n].own(qd[n])
        for k in range(own.shape[0]):
            S = jnp.zeros((R, V * R)).at[:, idx[n] * R:(idx[n] + 1) * R].set(jnp.eye(R))
            rows.append(S)
            ys.append(own[k, :R] / own[k, R])
            noise.append(jnp.full(R, 1.0 / own[k, R]))
    S = jnp.concatenate(rows)
    y = jnp.concatenate(ys)
    Cy = S @ Cz @ S.T + jnp.diag(jnp.concatenate(noise))
    return dict(names=names, idx=idx, y=y, my=S @ mz, Cy=Cy, mz=mz, Cz=Cz, Czy=Cz @ S.T)


def neg_log_evidence(J):
    d = J["y"] - J["my"]
    return 0.5 * d @ jnp.linalg.solve(J["Cy"], d) + 0.5 * jnp.linalg.slogdet(2 * jnp.pi * J["Cy"])[1]


# ---------------------------------------------------------------------------
# Predictions 3 and 4: evidence nests; receiving is the downward pass.
# ---------------------------------------------------------------------------


def test_cost_of_the_nest_is_minus_its_log_evidence():
    """CHECKS AGAINST A REFERENCE: the closed tree (eight data, a prior cap) has total cost
    ``-log p(data) - (r/2) N log 2 pi``, ``p`` from the joint covariance, at five configurations; and the closed
    composite's forcing on the configuration is the gradient of ``log p(data)``."""
    T, order = build(SPEC)
    closed = compose_seq(T.arr(), prior_cap(PRIOR))
    rng = np.random.default_rng(31)
    diffs, gerr = [], 0.0
    for _ in range(5):
        qd = random_config(rng)
        q = config_of(order, qd)
        total = T.cost(q, jnp.zeros(0)) + disagreement([PRIOR, T.sigma(q, jnp.zeros(0))])
        diffs.append(float(total - neg_log_evidence(joint(SPEC, qd, PRIOR))))

        def logp(qflat):
            parts, c = {}, 0
            for n in order:
                parts[n] = qflat[c:c + qd[n].shape[0]]
                c += qd[n].shape[0]
            return -neg_log_evidence(joint(SPEC, parts, PRIOR))

        xi_q = closed.omega(q, jnp.zeros(0), jnp.zeros(0))[0]
        gerr = max(gerr, float(jnp.max(jnp.abs(xi_q - jax.grad(logp)(q)))))
    const = -0.5 * R * 8 * np.log(2 * np.pi)
    print("prediction 3: total cost + log p(data) at five configurations:", diffs, "predicted", const,
          "; forcing against grad log p:", gerr)
    assert max(abs(d - const) for d in diffs) < 1e-10
    assert gerr < 1e-10


def test_receiving_is_the_downward_pass():
    """CHECKS AGAINST A REFERENCE: at every node ``v``, the evidence sent up by ``v``'s subtree plus the message the
    rest of the closed tree sends down to it, both read off the wire between the two arrangements, is the exact
    posterior of ``v``'s latent: mean and precision from conditioning the joint Gaussian."""
    rng = np.random.default_rng(41)
    worst = 0.0
    for _ in range(3):
        qd = random_config(rng)
        J = joint(SPEC, qd, PRIOR)
        post_C = J["Cz"] - J["Czy"] @ jnp.linalg.solve(J["Cy"], J["Czy"].T)
        post_m = J["mz"] + J["Czy"] @ jnp.linalg.solve(J["Cy"], J["y"] - J["my"])
        for name in J["names"]:
            sub = None

            def find(s):
                nm, kids = (s, []) if isinstance(s, str) else s
                if nm == name:
                    return s
                for k in kids:
                    f = find(k)
                    if f is not None:
                        return f
                return None

            sub = find(SPEC)
            Tv, order_v = build(sub)
            sigma_v = Tv.sigma(config_of(order_v, qd), jnp.zeros(0))
            if name == "R":
                rho_v = PRIOR
            else:
                Tc, order_c = build(SPEC, hole=name)
                rho_v = Tc.rho(config_of(order_c, qd), sigma_v, PRIOR)
            m = sigma_v + rho_v
            i = J["idx"][name]
            Cv = post_C[i * R:(i + 1) * R, i * R:(i + 1) * R]
            worst = max(worst, float(jnp.max(jnp.abs(m[:R] / m[R] - post_m[i * R:(i + 1) * R]))),
                        float(jnp.max(jnp.abs(Cv - jnp.eye(R) / m[R]))))
    print("prediction 4: posterior from the wires against conditioning, worst:", worst)
    assert worst < 1e-10


# ---------------------------------------------------------------------------
# Prediction 5: precisions learned by descent, no funnel.
# ---------------------------------------------------------------------------

J5, K5 = 4, 8


def p5_data():
    r = np.random.default_rng(0)
    return np.asarray([-3.0, -1.0, 1.0, 3.0])[:, None] + r.standard_normal((J5, K5))


def p5_tree(x, eta, init_groups, init_root):
    """Root: one log-precision ``l2``; links ``kappa = 1``, ``s = e^{-l2}`` to ``J`` groups. Group ``j``: one
    log-precision ``l1_j``, its ``K`` samples as own evidence with precision ``e^{l1_j}``. Scalar signals."""
    groups = []
    for j in range(J5):
        xs = jnp.asarray(x[j])
        node = Node(configuration(euclidean(1, eta), init=jnp.full(1, init_groups)),
                    lambda q, xs=xs: jnp.stack([jnp.array([jnp.exp(q[0]) * v, jnp.exp(q[0])]) for v in xs]),
                    lambda q: [], f"g{j}")
        groups.append(corolla(1, node, 0))
    root = Node(configuration(euclidean(1, eta), init=jnp.full(1, init_root)), lambda q: jnp.zeros((0, 2)),
                lambda q: [Link(jnp.asarray(1.0), jnp.eye(1), jnp.exp(-q[0])) for _ in range(J5)], "root")
    return graft(corolla(1, root, J5), groups)


def p5_neg_log_evidence(x, l, prior):
    """``-log p(x)`` for ``x_jk = mu_j + N(0, e^{-l1_j})``, ``mu_j = nu + N(0, e^{-l2})``, ``nu ~`` the prior."""
    l1, l2 = l[:J5], l[J5]
    v0 = 1.0 / prior[1]
    C = v0 * jnp.ones((J5 * K5, J5 * K5))
    blocks = jnp.kron(jnp.eye(J5), jnp.ones((K5, K5))) * jnp.exp(-l2)
    noise = jnp.repeat(jnp.exp(-l1), K5)
    C = C + blocks + jnp.diag(noise)
    d = jnp.ravel(jnp.asarray(x)) - prior[0] / prior[1]
    return 0.5 * d @ jnp.linalg.solve(C, d) + 0.5 * jnp.linalg.slogdet(2 * jnp.pi * C)[1]


def test_precisions_are_learned_without_a_funnel():
    """CHECKS AGAINST A REFERENCE: the closed tree run by ``Phi`` from rest, and from a start with the upper precision
    40, reaches the maximizer of ``log p(data)`` found by Newton's method beside it."""
    x = p5_data()
    prior = jnp.array([0.0, 1e-2])
    f = lambda l: p5_neg_log_evidence(x, l, prior)
    l = jnp.zeros(J5 + 1)
    for _ in range(60):
        l = l - jnp.linalg.solve(jax.hessian(f)(l), jax.grad(f)(l))
    assert float(jnp.max(jnp.abs(jax.grad(f)(l)))) < 1e-10
    for init_root in (0.0, float(np.log(40.0))):
        T = p5_tree(x, 0.05, 0.0, init_root)
        closed = compose_seq(T.arr(), prior_cap(prior))
        s, rec = run(closed, 6000, record=flat_state)
        got = rec[-1]   # (l1_1..l1_J, l2) then the cap's empty state
        gap = float(jnp.max(jnp.abs(got[:J5 + 1] - l)))
        lp = [float(-f(r[:J5 + 1])) for r in rec[::1000]]
        print(f"prediction 5, start l2={init_root:.3f}: reached", np.asarray(got[:J5 + 1]), "Newton", np.asarray(l),
              "gap", gap, "; log p along the run", lp)
        assert gap < 1e-5
        assert all(b >= a - 1e-9 for a, b in zip(lp, lp[1:]))   # log evidence never falls


# ---------------------------------------------------------------------------
# Prediction 6: at s = 0 a tree is one attention arrangement.
# ---------------------------------------------------------------------------


def as_transfer_up(L):
    return Transfer(gamma=L.kappa ** 2, L=L.kappa * L.U.T)


def as_transfer_down(L):
    return Transfer(gamma=1.0 / L.kappa ** 2, L=L.U / L.kappa)


def s0_node(n, name):
    """``n`` children with ``s = 0``: links ``(e^{a}, rot(b), 0)``, two numbers each; no own evidence."""
    links = lambda q: [Link(jnp.exp(q[2 * i]), rot(q[2 * i + 1]), jnp.asarray(0.0)) for i in range(n)]
    return Node(configuration(euclidean(2 * n, 0.02)), lambda q: jnp.zeros((0, K3)), links, name)


def s0_attention(node, n):
    """The node as one attention arrangement: ``T_0j = up_j``, ``T_i0 = down_i``, ``T_ij = down_i up_j``."""
    Z = zero_transfer(R)

    def transfers(q):
        Ls = node.links(q)
        T = [[Z] * (n + 1) for _ in range(n + 1)]
        for j in range(1, n + 1):
            T[0][j] = as_transfer_up(Ls[j - 1])
            T[j][0] = as_transfer_down(Ls[j - 1])
            for i in range(1, n + 1):
                if i != j:
                    T[i][j] = compose_transfers(as_transfer_down(Ls[i - 1]), as_transfer_up(Ls[j - 1]))
        return tuple(tuple(row) for row in T)

    return attention_arrangement(n, R, 0, placement=lambda q: jnp.zeros((n, 0)), transfers=transfers,
                                 cost=zero_cost, parameter=node.P, label=node.label)


def test_at_s_zero_a_tree_is_one_attention_arrangement():
    """CHECKS A CLOSED FORM: with every ``s = 0`` and no own evidence, the two-level tree's sending and receiving
    maps are those of the attention operad's composite (``attention.substitute``) of its nodes read as attention
    arrangements, ``T_ij = down_i up_j``: the paper's composite transfers route as belief propagation does. With one
    ``s > 0`` the sending map is not additive in the messages, so no attention arrangement has it."""
    Rn, An, Bn = s0_node(2, "R"), s0_node(2, "A"), s0_node(3, "B")
    tree = graft(corolla(R, Rn, 2), [corolla(R, An, 2), corolla(R, Bn, 3)])
    att = att_substitute(s0_attention(Rn, 2), [s0_attention(An, 2), s0_attention(Bn, 3)])
    rng = np.random.default_rng(61)
    worst = 0.0
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(tree.Q_dim) * 0.5)
        m, e = positive_messages(R, 5, rng), positive_messages(R, 1, rng)
        worst = max(worst, float(jnp.max(jnp.abs(tree.sigma(q, m) - att.sigma(q, m)))),
                    float(jnp.max(jnp.abs(tree.rho(q, m, e) - att.rho(q, m, e)))))
    print("prediction 6, s = 0 tree against attention.substitute:", worst)
    assert worst < 1e-12
    # one s > 0: not additive
    node = mid_node(2, None, "noisy")
    t = corolla(R, node, 2)
    q = jnp.asarray(rng.standard_normal(6) * 0.5)
    m1, m2 = positive_messages(R, 2, rng), positive_messages(R, 2, rng)
    gap = float(jnp.max(jnp.abs(t.sigma(q, m1 + m2) - t.sigma(q, m1) - t.sigma(q, m2))))
    print("prediction 6, s > 0: sigma(m + m') - sigma(m) - sigma(m'):", gap)
    assert gap > 1e-3
