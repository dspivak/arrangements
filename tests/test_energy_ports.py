"""Energy through ports (book Ch 5 ``thm.energy_ports``, plan §5.9 item 92; run notes ``notes/runs/p76e-5c9967d2/`` in the book).

On a linear exact-effort arrangement whose integrator is a tensor of masses, the kept energy

    E(t) = sum p_i^2 / (2 m_i) - (1/2) (A y_{t-1}) (y_t)

changes at each round by the work at the external interface less the work at the internal one, where the work is the inward loads
paired with the central difference of the values; and it adds over composites. So a closed arrangement keeps it, at every duration.

Every test runs a composed arrangement through ``PhiK`` and reads its label. Every test says which of three things it does (the
convention of test_rendezvous.py): reproduce a CLOSED FORM, check a DISCRETE ANALOGUE, or merely ILLUSTRATE. All of these could have
come out otherwise: the balance is an identity in the step, and the tests would fail if the book's step, or its signs, were not the
ones the proof uses.

Claude Opus 5.5, session 5c9967d2, 2026-11-05.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import CotCoalgK
from arrangements.hopfield import _tensor_balanced
from arrangements.interpretation import trivial_omega
from arrangements.pcK import PhiK
from arrangements.wiring import compose_seq, trivial_parameter

jax.config.update("jax_enable_x64", True)

_Z = jnp.zeros(0)
_IN = (jnp.zeros(0), trivial_omega(0))
_TRIV = lambda _o: (jnp.zeros(0), jnp.zeros(0))


def masses(q0, p0, m, tau, label="masses"):
    """A tensor of masses: look-ahead q + tau p / m, update (q + tau p / m, p + tau load)."""
    q0, p0, m = jnp.asarray(q0, float), jnp.asarray(p0, float), jnp.asarray(m, float)
    d = q0.shape[0]
    return CotCoalgK(Q_dim=d, init=jnp.concatenate([q0, p0]), reads=(lambda s: s[:d] + tau * s[d:] / m,),
                     advances=(lambda s, xi: jnp.concatenate([s[:d] + tau * s[d:] / m, s[d:] + tau * xi]),), label=label)


def run_states(arr, steps):
    """The states of a closed arrangement at rounds 0, ..., steps, flat."""
    O = PhiK(arr)
    f = jax.jit(lambda s: O.with_state(s).run_one(_IN, _TRIV)[1])
    s = O.state
    out = []
    for _ in range(steps + 1):
        out.append(np.concatenate([np.ravel(np.asarray(x, float)) for x in jax.tree_util.tree_leaves(s) if np.asarray(x).size]))
        s = f(s)
    return np.array(out)


def test_block_keeps_one_half():
    """CLOSED FORM. The block on a spring of Ch 2 exc.waves_one_spring: six states, the Hamiltonian swinging and the kept energy 1/2."""
    tau = 1.0
    box = Arrangement(P=masses([1.0], [0.0], [1.0], tau), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
                      out_f=lambda q, m: _Z, in_f=lambda q, m, n: _Z,
                      omega=exact(lambda q, m, n: -0.5 * q @ q), label="block")
    st = run_states(box, 6)
    want = [(1, 0), (1, -1), (0, -1), (-1, 0), (-1, 1), (0, 1), (1, 0)]
    np.testing.assert_allclose(st, np.array(want, float), atol=1e-12)
    q, p = st[:, 0], st[:, 1]
    kept = 0.5 * p ** 2 + 0.5 * q * (q + tau * p)
    np.testing.assert_allclose(kept, 0.5, atol=1e-12)
    np.testing.assert_allclose(0.5 * p ** 2 + 0.5 * q ** 2, [0.5, 1, 0.5, 0.5, 1, 0.5, 0.5], atol=1e-12)


def test_kept_past_the_bound():
    """CLOSED FORM. At duration 3 (duration^2 kappa / m = 9, past 4), the same block grows and still keeps 1/2."""
    tau = 3.0
    box = Arrangement(P=masses([1.0], [0.0], [1.0], tau), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=0,
                      out_f=lambda q, m: _Z, in_f=lambda q, m, n: _Z,
                      omega=exact(lambda q, m, n: -0.5 * q @ q), label="block")
    st = run_states(box, 2)
    np.testing.assert_allclose(st, [[1, 0], [1, -3], [-8, 21]], atol=1e-12)
    q, p = st[:, 0], st[:, 1]
    np.testing.assert_allclose(0.5 * p ** 2 + 0.5 * q * (q + tau * p), 0.5, atol=1e-12)


def _bead(q0, p0, m, kappa, tau):
    return Arrangement(P=masses([q0], [p0], [m], tau, "bead"), in_dim_M=0, out_dim_M=0, in_dim_N=1, out_dim_N=1,
                       out_f=lambda q, m_: q, in_f=lambda q, m_, n: _Z,
                       omega=exact(lambda q, m_, n: -0.5 * kappa * (q[0] - n[0]) ** 2), label="bead")


def test_bead_chain_numbers_and_balance():
    """CLOSED FORM. Ch 3 ex.bead_chain from (8, 0, 0) at rest: the kept energy is 8 at each round, the Hamiltonian 8, 12, 10.25; and each
    bead's balance holds, its work the inward loads at its two ports against the central differences."""
    tau, k1, k2 = 1.0, 0.25, 0.25
    first = Arrangement(P=masses([8.0], [0.0], [1.0], tau, "mass"), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
                        out_f=lambda q, m_: q, in_f=lambda q, m_, n: _Z, omega=zero_form, label="mass")
    parts = [first, _bead(0.0, 0.0, 1.0, k1, tau), _bead(0.0, 0.0, 1.0, k2, tau)]
    route = jnp.asarray([0, 1], dtype=int)  # the first bead hears the mass, the second hears the first; the second's output is dropped
    wire = Arrangement(P=trivial_parameter(), in_dim_M=2, out_dim_M=3, in_dim_N=0, out_dim_N=0, out_f=lambda q, m_: _Z,
                       in_f=lambda q, m_, n: m_[route], omega=zero_form, label="row")
    arr = compose_seq(_tensor_balanced(parts), wire)
    assert arr.label.count("bead") == 2 and "mass" in arr.label
    st = run_states(arr, 8)
    q, p = st[:, [0, 2, 4]], st[:, [1, 3, 5]]
    np.testing.assert_allclose(q[:4], [[8, 0, 0], [8, 0, 0], [6, 2, 0], [3, 4.5, 0.5]], atol=1e-12)
    x = q + tau * p
    kept = 0.5 * (p ** 2).sum(1) + k1 / 2 * (q[:, 0] - q[:, 1]) * (x[:, 0] - x[:, 1]) + k2 / 2 * (q[:, 1] - q[:, 2]) * (x[:, 1] - x[:, 2])
    np.testing.assert_allclose(kept, 8.0, atol=1e-12)
    ham = 0.5 * (p ** 2).sum(1) + k1 / 2 * (q[:, 0] - q[:, 1]) ** 2 + k2 / 2 * (q[:, 1] - q[:, 2]) ** 2
    np.testing.assert_allclose(ham[:3], [8, 12, 10.25], atol=1e-12)
    # The first bead (index 1): input the mass's x_0, position x_1; received load at its output, the second bead's pushback.
    for t in range(1, 6):
        def E(t_):
            return 0.5 * p[t_, 1] ** 2 + k1 / 2 * (q[t_, 0] - q[t_, 1]) * (x[t_, 0] - x[t_, 1])

        inward_in = k1 * (x[t, 0] - x[t, 1])
        received = k2 * (x[t, 2] - x[t, 1])
        work = inward_in * (x[t + 1, 0] - x[t - 1, 0]) / 2 + received * (x[t + 1, 1] - x[t - 1, 1]) / 2
        assert abs(E(t + 1) - E(t) - work) < 1e-12


def test_two_input_composite():
    """DISCRETE ANALOGUE. Three random boxes inside a stateless exact top box: each box's balance, the top's, and the sum, at a duration
    inside the bound and at one far past it, each relative to the terms' size at its round."""
    rng = np.random.default_rng(18)

    def psd(n):
        X = rng.normal(size=(n, n))
        return X @ X.T / n

    for tau in (0.1, 2.0):
        boxes = []
        for _ in range(3):
            n_out = int(rng.integers(1, 3))
            boxes.append(dict(m=rng.uniform(0.5, 2.0, size=2), H=psd(2), L=rng.normal(size=(n_out, 2)), q=rng.normal(size=2), p=rng.normal(size=2)))
        total = sum(b["L"].shape[0] for b in boxes)
        Htop = psd(total)

        def box_arr(b):
            H, L = jnp.asarray(b["H"]), jnp.asarray(b["L"])
            return Arrangement(P=masses(b["q"], b["p"], b["m"], tau, "box"), in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=L.shape[0],
                               out_f=lambda q, m_: L @ q, in_f=lambda q, m_, n: _Z,
                               omega=exact(lambda q, m_, n: -0.5 * q @ (H @ q)), label="box")

        Ht = jnp.asarray(Htop)
        top = Arrangement(P=trivial_parameter(), in_dim_M=0, out_dim_M=0, in_dim_N=total, out_dim_N=0, out_f=lambda q, m_: _Z,
                          in_f=lambda q, m_, n: _Z, omega=exact(lambda q, m_, n: -0.5 * n @ (Ht @ n)), label="top")
        idx = jnp.arange(total, dtype=int)
        wire = Arrangement(P=trivial_parameter(), in_dim_M=total, out_dim_M=total, in_dim_N=0, out_dim_N=0, out_f=lambda q, m_: _Z,
                           in_f=lambda q, m_, n: m_[idx], omega=zero_form, label="wire")
        arr = compose_seq(_tensor_balanced([box_arr(b) for b in boxes] + [top]), wire)
        assert arr.label == "(wire . ((box || box) || (box || top)))"
        T = 40
        st = run_states(arr, T - 1).reshape(T, 3, 4)
        q = [st[:, i, :2] for i in range(3)]
        p = [st[:, i, 2:] for i in range(3)]
        x_i = [q[i][1:] for i in range(3)]
        r = [x_i[i] @ boxes[i]["L"].T for i in range(3)]
        x = np.concatenate(r, axis=1)
        G = x @ Htop
        lam, c = [], 0
        for i in range(3):
            d = boxes[i]["L"].shape[0]
            lam.append(-G[:, c:c + d])
            c += d

        def Ebox(i, t):
            b = boxes[i]
            return 0.5 * p[i][t] @ (p[i][t] / b["m"]) + 0.5 * q[i][t] @ (b["H"] @ q[i][t + 1])

        def Etop(t):
            return 0.5 * x[t - 1] @ (Htop @ x[t])

        first = None
        for t in range(1, T - 3):
            size = sum(p[i][t] @ (p[i][t] / boxes[i]["m"]) + np.abs(boxes[i]["H"]).max() * (q[i][t] @ q[i][t] + q[i][t + 1] @ q[i][t + 1]) for i in range(3))
            size += np.abs(Htop).max() * (x[t - 1] @ x[t - 1] + x[t] @ x[t])
            works = 0.0
            for i in range(3):
                w = lam[i][t] @ ((r[i][t + 1] - r[i][t - 1]) / 2)
                assert abs(Ebox(i, t + 1) - Ebox(i, t) - w) / size < 1e-12
                works += w
            assert abs(Etop(t + 1) - Etop(t) + works) / size < 1e-12
            tot = sum(Ebox(i, t) for i in range(3)) + Etop(t)
            first = tot if first is None else first
            assert abs(tot - first) / size < 1e-12


def test_light_regions():
    """DISCRETE ANALOGUE. Book Ch 6 ex.light_energy: the lattice on the 3-torus with four cells a side, pair (1/h, h), as two regions, each
    one box with its cut wires as ports, joined by one wire. Equal to the plain step; each region's balance; the two works add to 0 and
    are not 0."""
    N = 4
    h = 2 * np.pi / N
    mu = 0.5
    tau = mu * h
    kappa, mass = 1 / h, h
    SIGN = np.array([1.0, 1.0, -1.0, -1.0])

    def eid(f, i, j, k):
        return f * N ** 3 + (i % N) * N * N + (j % N) * N + (k % N)

    faces = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                faces.append((i, (eid(0, i, j, k), eid(1, i + 1, j, k), eid(0, i, j + 1, k), eid(1, i, j, k))))
                faces.append((i, (eid(1, i, j, k), eid(2, i, j + 1, k), eid(1, i, j, k + 1), eid(2, i, j, k))))
                faces.append((i, (eid(2, i, j, k), eid(0, i, j, k + 1), eid(2, i + 1, j, k), eid(0, i, j, k))))
    owner = lambda e: (e % N ** 3) // (N * N)
    pieces = []
    for reg in ((0, 1), (2, 3)):
        ex = [e for e in range(3 * N ** 3) if owner(e) in reg]
        fx = [f for f in faces if f[0] in reg]
        fo = [f for f in faces if f[0] not in reg]
        pieces.append(dict(ex=ex, fx=fx, ext=sorted({e for f in fx for e in f[1]} - set(ex)), outs=sorted(set(ex) & {e for f in fo for e in f[1]})))
    rng = np.random.default_rng(19)
    A0, x0 = rng.normal(size=3 * N ** 3), rng.normal(size=3 * N ** 3)

    def piece_arr(P_):
        ex, ext, outs = P_["ex"], P_["ext"], P_["outs"]
        d = len(ex)
        pos = {e: n for n, e in enumerate(ex + ext)}
        C = np.zeros((len(P_["fx"]), d + len(ext)))
        for r_, f in enumerate(P_["fx"]):
            for s_, e in zip(SIGN, f[1]):
                C[r_, pos[e]] += s_
        C = jnp.asarray(C)
        oidx = jnp.asarray([ex.index(e) for e in outs], dtype=int)

        def J(q, m_, n):
            B = C @ jnp.concatenate([q, n])
            return 0.5 * kappa * B @ B

        return Arrangement(P=masses(A0[ex], x0[ex], np.full(d, mass), tau, "region"), in_dim_M=0, out_dim_M=0, in_dim_N=len(ext), out_dim_N=len(outs),
                           out_f=lambda q, m_: q[oidx], in_f=lambda q, m_, n: _Z, omega=exact(lambda q, m_, n: -J(q, m_, n)), label="region")

    all_outs = pieces[0]["outs"] + pieces[1]["outs"]
    route = jnp.asarray([all_outs.index(e) for e in pieces[0]["ext"] + pieces[1]["ext"]], dtype=int)
    wire = Arrangement(P=trivial_parameter(), in_dim_M=int(route.shape[0]), out_dim_M=len(all_outs), in_dim_N=0, out_dim_N=0, out_f=lambda q, m_: _Z,
                       in_f=lambda q, m_, n: m_[route], omega=zero_form, label="cut_wires")
    arr = compose_seq(_tensor_balanced([piece_arr(pieces[0]), piece_arr(pieces[1])]), wire)
    assert arr.label == "(cut_wires . (region || region))"
    T = 14
    st = run_states(arr, T - 1)
    d0 = len(pieces[0]["ex"])
    A = np.zeros((T, 3 * N ** 3))
    X = np.zeros((T, 3 * N ** 3))
    for t in range(T):
        for c, P_ in enumerate(pieces):
            off = 0 if c == 0 else 2 * d0
            n_ = len(P_["ex"])
            A[t, P_["ex"]] = st[t, off:off + n_]
            X[t, P_["ex"]] = st[t, off + n_:off + 2 * n_]

    def B_of(a, f_):
        return sum(s_ * a[e] for s_, e in zip(SIGN, f_[1]))

    # The plain step: A' = A + (tau/h) xi, xi' = xi - (tau/h) d^T d A'.
    for t in range(T - 1):
        g = np.zeros(3 * N ** 3)
        for f_ in faces:
            b = B_of(A[t + 1], f_)
            for s_, e in zip(SIGN, f_[1]):
                g[e] += s_ * b
        np.testing.assert_allclose(A[t + 1], A[t] + mu * X[t], atol=1e-10)
        np.testing.assert_allclose(X[t + 1], X[t] - mu * g, atol=1e-10)

    def kept(P_, t):
        return (X[t][P_["ex"]] @ X[t][P_["ex"]]) / (2 * mass) + kappa / 2 * sum(B_of(A[t], f_) * B_of(A[t + 1], f_) for f_ in P_["fx"])

    def work(P_, t):
        cd = (A[t + 2] - A[t]) / 2
        mine, ext_set, ex_set = {id(f_) for f_ in P_["fx"]}, set(P_["ext"]), set(P_["ex"])
        out = 0.0
        for f_ in faces:
            b = B_of(A[t + 1], f_)
            inside = id(f_) in mine
            for s_, e in zip(SIGN, f_[1]):
                if inside and e in ext_set:
                    out += kappa * s_ * b * cd[e]
                if not inside and e in ex_set:
                    out += -kappa * s_ * b * cd[e]
        return out

    for t in range(0, 8):
        W = [work(P_, t) for P_ in pieces]
        assert abs(W[0]) > 1e-3
        assert abs(W[0] + W[1]) < 1e-10
        for c, P_ in enumerate(pieces):
            assert abs(kept(P_, t + 1) - kept(P_, t) - W[c]) < 1e-10


def test_driven_region_with_a_clock():
    """DISCRETE ANALOGUE. A region of the lattice driven by an exact solution of Maxwell's equations, through a part with a clock: the clock
    supplies the region's input values and returns the loads at its output ports that the faces it does not own would return at exact
    values. Equal to the driven loop, which the run notes F20 (the book's plan §5.9 item 93) then compare with the exact solution."""
    N = 4
    h = 2 * np.pi / N
    mu = 0.5
    tau = mu * h
    kappa, mass = 1 / h, h
    SIGN = np.array([1.0, 1.0, -1.0, -1.0])
    MODES = [((1, 0, 0), (0.0, 1.0, 0.5)), ((1, 1, 1), (1.0, -1.0, 0.0)), ((0, 1, -1), (0.5, 1.0, 1.0))]

    def eid(f, i, j, k):
        return f * N ** 3 + (i % N) * N * N + (j % N) * N + (k % N)

    faces = []
    for i in range(N):
        for j in range(N):
            for k in range(N):
                faces.append((i, (eid(0, i, j, k), eid(1, i + 1, j, k), eid(0, i, j + 1, k), eid(1, i, j, k))))
                faces.append((i, (eid(1, i, j, k), eid(2, i, j + 1, k), eid(1, i, j, k + 1), eid(2, i, j, k))))
                faces.append((i, (eid(2, i, j, k), eid(0, i, j, k + 1), eid(2, i + 1, j, k), eid(0, i, j, k))))
    ne = 3 * N ** 3
    D = np.zeros((len(faces), ne))
    for r_, f_ in enumerate(faces):
        for s_, e in zip(SIGN, f_[1]):
            D[r_, e] += s_
    owned = np.array([f_[0] in (0, 1) for f_ in faces])
    Down, Dout = D[owned], D[~owned]

    # Each mode's sampled circulations and dual fluxes, as vectors on edges: a_i h s_i cos(k . mid), a_i h^2 s_j s_l cos(k . mid).
    circ, flux, freq = [], [], []
    for k_, a_ in MODES:
        k_, a_ = np.array(k_, float), np.array(a_, float)
        s = np.sinc(k_ * h / 2 / np.pi)
        c, x = np.zeros(ne), np.zeros(ne)
        for f in range(3):
            sp = np.prod(np.delete(s, f))
            for i in range(N):
                for j in range(N):
                    for l in range(N):
                        mid = (np.array([i, j, l], float) + 0.5 * np.eye(3)[f]) * h
                        ph = np.cos(k_ @ mid)
                        c[eid(f, i, j, l)] = a_[f] * h * s[f] * ph
                        x[eid(f, i, j, l)] = a_[f] * h * h * sp * ph
        circ.append(c)
        flux.append(x)
        freq.append(np.linalg.norm(k_))
    circ, flux, freq = np.array(circ), np.array(flux), np.array(freq)

    A_exact_ = lambda t: (np.cos(freq * t) @ circ)
    X_exact_ = lambda t: ((-freq * np.sin(freq * t)) @ flux)

    ex = [e for e in range(ne) if (e % N ** 3) // (N * N) in (0, 1)]
    used = sorted(set(np.nonzero(Down)[1].tolist()))
    ext = sorted(set(used) - set(ex))
    read_out = sorted(set(ex) & set(np.nonzero(Dout)[1].tolist()))
    pos = {e: n for n, e in enumerate(ex + ext)}
    C = np.zeros((Down.shape[0], len(ex) + len(ext)))
    for e, n_ in pos.items():
        C[:, n_] = Down[:, e]
    C_j = jnp.asarray(C)
    oidx = jnp.asarray([ex.index(e) for e in read_out], dtype=int)

    def J(q, m_, n):
        B = C_j @ jnp.concatenate([q, n])
        return 0.5 * kappa * B @ B

    region = Arrangement(P=masses(A_exact_(0.0)[ex], X_exact_(tau / 2)[ex], np.full(len(ex), mass), tau, "region"),
                         in_dim_M=0, out_dim_M=0, in_dim_N=len(ext), out_dim_N=len(read_out), out_f=lambda q, m_: q[oidx],
                         in_f=lambda q, m_, n: _Z, omega=exact(lambda q, m_, n: -J(q, m_, n)), label="region")

    circ_j, freq_j = jnp.asarray(circ), jnp.asarray(freq)
    ext_j, out_j = jnp.asarray(ext, dtype=int), jnp.asarray(read_out, dtype=int)
    G_out = jnp.asarray(Dout.T @ Dout)

    def A_of(t):
        return jnp.cos(freq_j * t) @ circ_j

    clock_P = CotCoalgK(Q_dim=1, init=jnp.array([0.0]), reads=(lambda t: t + tau,), advances=(lambda t, xi: t + tau,), label="clock")

    def clock_omega(q, m_, n):
        return (jnp.zeros(1), _Z, -kappa * (G_out @ A_of(q[0]))[out_j])

    clock = Arrangement(P=clock_P, in_dim_M=0, out_dim_M=0, in_dim_N=len(read_out), out_dim_N=len(ext),
                        out_f=lambda q, m_: A_of(q[0])[ext_j], in_f=lambda q, m_, n: _Z, omega=clock_omega, label="clock")
    # The clock's outputs come first among the outputs, its inputs first among the inputs.
    route = jnp.asarray([len(ext) + i for i in range(len(read_out))] + list(range(len(ext))), dtype=int)
    wire = Arrangement(P=trivial_parameter(), in_dim_M=int(route.shape[0]), out_dim_M=int(route.shape[0]), in_dim_N=0, out_dim_N=0,
                       out_f=lambda q, m_: _Z, in_f=lambda q, m_, n: m_[route], omega=zero_form, label="ports")
    arr = compose_seq(_tensor_balanced([clock, region]), wire)
    assert arr.label == "(ports . (clock || region))"
    T = 12
    st = run_states(arr, T)
    t_ = st[:, 0]
    np.testing.assert_allclose(t_, np.arange(T + 1) * tau, atol=1e-11)
    d = len(ex)
    q_arr, p_arr = st[:, 1:1 + d], st[:, 1 + d:1 + 2 * d]

    # The driven loop.
    A = A_exact_(0.0).copy()
    X = X_exact_(tau / 2).copy()
    mask = np.zeros(ne, bool)
    mask[ex] = True
    for n in range(T):
        t = (n + 1) * tau
        A_new = np.where(mask, A + mu * X, A_exact_(t))
        g = Down.T @ (Down @ A_new) + Dout.T @ (Dout @ A_exact_(t))
        X = np.where(mask, X - mu * g, X_exact_(t + tau / 2))
        A = A_new
        np.testing.assert_allclose(q_arr[n + 1], A[ex], atol=1e-11)
        np.testing.assert_allclose(p_arr[n + 1], X[ex], atol=1e-11)
