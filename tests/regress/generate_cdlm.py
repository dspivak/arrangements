"""Record CDLM (``dap-core-old``) trajectories for the regression tests (STATUS.md §1).

Run from the OLD package so that ``import dap`` is CDLM's:

    cd ../dynamic-algebra-potentials/dap-core-old && PYTHONPATH=. uv run --no-sync python ../../arrangements/tests/regress/generate_cdlm.py

(``PYTHONPATH=.`` because the old ``.venv`` was made when that directory was still
called ``dap-core``; the new package is now ``arrangements``, so nothing shadows it.)

Writes one ``.npz`` per example into this directory. Each file records the
*inputs* the old test drew (seeds are replayed here, so the new tests need not
reproduce the draw order) and the *outputs* the old package produced, all in the
old sign convention: environment covectors ``xi_N`` are CDLM's (the integrator
subtracts them), published fields are ``pullback + dU``. The new tests translate
per STATUS.md §1: ``xi_N_new = -xi_N_old``, fields ``omega_N_new = -omega_N_old``,
states identical.
"""

import os
import sys

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from dap.arrangement import SmoothArrangement
from dap.functors import Phiconf, Phiphase, phi_laxator
from dap.interpretation import trivial_omega
from dap.learning import parameterized_map, squared_error, squared_error_grad, train, forward_backward
from dap.polynomial import nest_left, unnest_left
from dap.rvect import diagonal, euclidean
from dap.wiring import chain_wire, compose_chain, parallel_arrangements, tensor_arrangements

assert "dap-core-old" in os.path.abspath(sys.modules["dap"].__file__), \
    "run this from dap-core-old with PYTHONPATH=. so that `dap` is the old package"

HERE = os.path.dirname(os.path.abspath(__file__))
_IN_POS = (jnp.zeros(0), trivial_omega(0))


def _harmonic_particle(m, kappa):
    return SmoothArrangement(
        Q=diagonal(jnp.array([1.0 / m])), out_dim_M=0, in_dim_M=0, out_dim_N=1, in_dim_N=1,
        out_f=lambda q, m_out: q,
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        U=lambda q, m_out, n_in: 0.5 * kappa * (q[0] - n_in[0]) ** 2,
        label="Part",
    )


def _save(name, **arrays):
    path = os.path.join(HERE, name + ".npz")
    np.savez(path, **{k: np.asarray(v) for k, v in arrays.items()})
    print("wrote", path, sorted(arrays))


# ---------------------------------------------------------------------------
# Heat (test_heat_equation.py): pinned-end Phiconf trajectories.
# ---------------------------------------------------------------------------


def _heat_step(O, q, kappa):
    K = q.shape[0]
    xi_N = jnp.array([kappa * float(q[K - 1])])
    _, _, new_q = O.with_state(q).run_one(_IN_POS, lambda _o: (xi_N, jnp.array([0.0])))
    return new_q


def heat():
    out = {}
    for tag, (K, T, m, kappa, seed) in {"a": (5, 10, 1.5, 0.9, 3), "b": (9, 500, 1.0, 0.2, 4)}.items():
        O = Phiconf(compose_chain([_harmonic_particle(m, kappa)] * K))
        q = jnp.asarray(np.random.default_rng(seed).standard_normal(K))
        traj = [q]
        for _ in range(T):
            q = _heat_step(O, q, kappa)
            traj.append(q)
        out[f"traj_{tag}"] = np.stack([np.asarray(x) for x in traj])
        out[f"params_{tag}"] = np.array([K, T, m, kappa, seed])
    _save("heat", **out)


# ---------------------------------------------------------------------------
# Wave (test_wave_equation.py): composite_final samples, the published field, pinned trajectory.
# ---------------------------------------------------------------------------


def wave():
    out = {}
    K, m, kappa = 5, 1.7, 2.3
    O = Phiphase(compose_chain([_harmonic_particle(m, kappa)] * K))
    rng = np.random.default_rng(20260507)
    qs, ps, qprevs, xiNs, nqs, nps = [], [], [], [], [], []
    for _ in range(5):
        q = jnp.asarray(rng.standard_normal(K))
        p = jnp.asarray(rng.standard_normal(K))
        q_prev = float(rng.standard_normal())
        xi_N = float(rng.standard_normal())
        in_dir = (jnp.array([xi_N]), jnp.array([q_prev]))
        _, _, (new_q, new_p) = O.with_state((q, p)).run_one(_IN_POS, lambda _o: in_dir)
        qs.append(q); ps.append(p); qprevs.append(q_prev); xiNs.append(xi_N); nqs.append(new_q); nps.append(new_p)
    out.update(cf_q=np.stack(qs), cf_p=np.stack(ps), cf_qprev=np.array(qprevs), cf_xiN=np.array(xiNs),
               cf_new_q=np.stack(nqs), cf_new_p=np.stack(nps), cf_params=np.array([K, m, kappa]))

    # published field (m=1, kappa=0.7)
    K2, m2, kappa2 = 5, 1.0, 0.7
    O2 = Phiphase(compose_chain([_harmonic_particle(m2, kappa2)] * K2))
    q = jnp.array([0.4, -0.2, 1.1, 0.0, 2.3])
    _, fiber = O2.step((q, jnp.zeros(K2)))
    (out_n, omega_N), _ = fiber(_IN_POS)
    out.update(field_q=q, field_out_n=out_n, field_at_0=omega_N(jnp.zeros(1)), field_at_1=omega_N(jnp.ones(1)),
               field_params=np.array([K2, m2, kappa2]))

    # pinned trajectory (m=1.5, kappa=0.9, seed 42, T+2 = 12 steps)
    K3, T3, m3, kappa3 = 5, 10, 1.5, 0.9
    O3 = Phiphase(compose_chain([_harmonic_particle(m3, kappa3)] * K3))
    q0 = jnp.asarray(np.random.default_rng(42).standard_normal(K3))
    state = (q0, jnp.zeros(K3))
    q_traj, p_traj = [state[0]], [state[1]]
    for _ in range(T3 + 2):
        q, p = state
        xi_N = jnp.array([kappa3 * float(q[K3 - 1] + p[K3 - 1] / m3)])
        _, _, state = O3.with_state(state).run_one(_IN_POS, lambda _o: (xi_N, jnp.array([0.0])))
        q_traj.append(state[0]); p_traj.append(state[1])
    out.update(pin_q=np.stack(q_traj), pin_p=np.stack(p_traj), pin_params=np.array([K3, T3, m3, kappa3]))
    _save("wave", **out)


# ---------------------------------------------------------------------------
# Functors (test_functors.py): single-particle updates and the shared readout.
# ---------------------------------------------------------------------------


def functors():
    m, kappa = 2.0, 3.0
    P = _harmonic_particle(m, kappa)
    q0, p0 = jnp.array([0.7]), jnp.array([1.1])
    xi_N, q_prev = jnp.array([0.4]), jnp.array([0.55])
    _, _, (nq, np_) = Phiphase(P).with_state((q0, p0)).run_one(_IN_POS, lambda _o: (xi_N, q_prev))
    _, _, cq = Phiconf(P).with_state(q0).run_one(_IN_POS, lambda _o: (xi_N, q_prev))

    m2, kappa2 = 1.3, 0.8
    P2 = _harmonic_particle(m2, kappa2)
    q, p = jnp.array([0.3]), jnp.array([0.5])
    in_dir = (jnp.array([0.1]), jnp.array([0.05]))
    pp, dp, _ = Phiphase(P2).with_state((q, p)).run_one(_IN_POS, lambda _o: in_dir)
    z = jnp.array([0.42])
    _save("functors", phase_new_q=nq, phase_new_p=np_, conf_new_q=cq,
          share_out_n=pp[0], share_omega_z=pp[1](z), share_xi_M=dp[0])


# ---------------------------------------------------------------------------
# Composition (test_composition.py): the first pass on random states, and the laxator square.
# ---------------------------------------------------------------------------


def _generic_box(dim_Q, d_out_M, d_in_M, d_out_N, d_in_N, seed):
    rng = np.random.default_rng(seed)
    A = jnp.asarray(rng.standard_normal((d_out_N, dim_Q + d_out_M)))
    B = jnp.asarray(rng.standard_normal((d_in_M, dim_Q + d_out_M + d_in_N)))
    c = jnp.asarray(rng.standard_normal(dim_Q + d_out_M + d_in_N))
    sharp = jnp.asarray(0.5 + rng.random(dim_Q))

    def out_f(q, m_out):
        return jnp.tanh(A @ jnp.concatenate([q, m_out]))

    def in_f(q, m_out, n_in):
        return jnp.sin(B @ jnp.concatenate([q, m_out, n_in]))

    def U(q, m_out, n_in):
        z = jnp.concatenate([q, m_out, n_in])
        return jnp.sum(jnp.cos(c * z)) + 0.5 * jnp.sum(z ** 2)

    arr = SmoothArrangement(diagonal(sharp), d_out_M, d_in_M, d_out_N, d_in_N,
                            out_f=out_f, in_f=in_f, U=U, label=f"box{seed}")
    return arr, dict(A=A, B=B, c=c, sharp=sharp)


def composition():
    out = {}
    # first pass, wave chain and heat chain
    for tag, (Phi, K, m, kappa, phase, seed) in {
        "wave": (Phiphase, 5, 1.7, 2.3, True, 7), "heat": (Phiconf, 5, 1.5, 0.9, False, 7),
        "wave2": (Phiphase, 5, 1.7, 2.3, True, 19), "heat2": (Phiconf, 5, 1.5, 0.9, False, 19),
    }.items():
        first = Phi(compose_chain([_harmonic_particle(m, kappa) for _ in range(K)]))
        rng = np.random.default_rng(seed)
        rows = {k: [] for k in ("q", "p", "xi", "qprev", "on", "omz", "d_xi", "d_in", "nq", "np")}
        for _ in range(5):
            q = jnp.asarray(rng.standard_normal(K))
            p = jnp.asarray(rng.standard_normal(K)) if phase else jnp.zeros(K)
            in_dir = (jnp.asarray(rng.standard_normal(1)), jnp.asarray(rng.standard_normal(1)))
            s1 = (q, p) if phase else q
            (on1, om1), d1, ns1 = first.with_state(s1).run_one(_IN_POS, lambda _o: in_dir)
            z = jnp.array([0.37]) if seed == 7 else jnp.array([0.41])
            rows["q"].append(q); rows["p"].append(p); rows["xi"].append(in_dir[0]); rows["qprev"].append(in_dir[1])
            rows["on"].append(on1); rows["omz"].append(om1(z)); rows["d_xi"].append(d1[0]); rows["d_in"].append(d1[1])
            rows["nq"].append(ns1[0] if phase else ns1); rows["np"].append(ns1[1] if phase else jnp.zeros(K))
        for k, v in rows.items():
            out[f"{tag}_{k}"] = np.stack([np.asarray(x) for x in v])
        out[f"{tag}_params"] = np.array([K, m, kappa])

    # laxator coherence: record the box data and lhs outputs
    A, dA = _generic_box(2, 2, 3, 2, 2, seed=1)
    B, dB = _generic_box(3, 1, 2, 3, 1, seed=2)
    for name, d in (("A", dA), ("B", dB)):
        for k, v in d.items():
            out[f"lax_{name}_{k}"] = v
    for tag, (Phi, phase) in {"phase": (Phiphase, True), "conf": (Phiconf, False)}.items():
        lax_N = phi_laxator([(P.out_dim_N, P.in_dim_N) for P in (A, B)], label="lax_N")
        lhs = Phi(A).parallel(Phi(B)).post_static(lax_N)
        d_out_N, d_in_N = A.out_dim_N + B.out_dim_N, A.in_dim_N + B.in_dim_N
        rng = np.random.default_rng(11)
        rows = {k: [] for k in ("mA", "wA", "mB", "wB", "xi", "inn", "qA", "qB", "pA", "pB", "z",
                                "on", "omz", "dA_xi", "dA_in", "dB_xi", "dB_in", "nqA", "npA", "nqB", "npB")}
        for _ in range(5):
            def field(seed, d):
                w = jnp.asarray(rng.standard_normal(d))
                return (lambda z: jnp.cos(3.0 * z) + w * z ** 2), w
            mA = jnp.asarray(rng.standard_normal(A.out_dim_M)); fA, wA = field(0, A.in_dim_M)
            mB = jnp.asarray(rng.standard_normal(B.out_dim_M)); fB, wB = field(1, B.in_dim_M)
            in_pos = ((mA, fA), (mB, fB))
            in_dir = (jnp.asarray(rng.standard_normal(d_out_N)), jnp.asarray(rng.standard_normal(d_in_N)))
            qA = jnp.asarray(rng.standard_normal(A.Q.dim)); qB = jnp.asarray(rng.standard_normal(B.Q.dim))
            if phase:
                pA = jnp.asarray(rng.standard_normal(A.Q.dim)); pB = jnp.asarray(rng.standard_normal(B.Q.dim))
                s = ((qA, pA), (qB, pB))
            else:
                pA, pB = jnp.zeros(A.Q.dim), jnp.zeros(B.Q.dim)
                s = (qA, qB)
            (on1, om1), d1, ns1 = lhs.with_state(s).run_one(in_pos, lambda _o: in_dir)
            z = jnp.asarray(rng.standard_normal(d_in_N))
            (dA_xi, dA_in), (dB_xi, dB_in) = unnest_left(d1, 2)
            rows["mA"].append(mA); rows["wA"].append(wA); rows["mB"].append(mB); rows["wB"].append(wB)
            rows["xi"].append(in_dir[0]); rows["inn"].append(in_dir[1])
            rows["qA"].append(qA); rows["qB"].append(qB); rows["pA"].append(pA); rows["pB"].append(pB); rows["z"].append(z)
            rows["on"].append(on1); rows["omz"].append(om1(z))
            rows["dA_xi"].append(dA_xi); rows["dA_in"].append(dA_in); rows["dB_xi"].append(dB_xi); rows["dB_in"].append(dB_in)
            if phase:
                rows["nqA"].append(ns1[0][0]); rows["npA"].append(ns1[0][1]); rows["nqB"].append(ns1[1][0]); rows["npB"].append(ns1[1][1])
            else:
                rows["nqA"].append(ns1[0]); rows["npA"].append(pA); rows["nqB"].append(ns1[1]); rows["npB"].append(pB)
        for k, v in rows.items():
            out[f"lax_{tag}_{k}"] = np.stack([np.asarray(x) for x in v])
    _save("composition", **out)


# ---------------------------------------------------------------------------
# Gradient descent (test_gradient_descent.py).
# ---------------------------------------------------------------------------


def _two_layer(h, m, n):
    shapes = [(h, m), (h,), (n, h), (n,)]
    sizes = [int(np.prod(s)) for s in shapes]
    dim = sum(sizes)

    def unpack(q):
        out, i = [], 0
        for s, k in zip(shapes, sizes):
            out.append(q[i:i + k].reshape(s))
            i += k
        return out

    def F(q, x):
        W1, b1, W2, b2 = unpack(q)
        return W2 @ jnp.tanh(W1 @ x + b1) + b2

    return F, dim


def gradient_descent():
    out = {}
    F, dim = _two_layer(4, 3, 2)
    arr = parameterized_map(F, euclidean(dim), in_dim=3, out_dim=2)
    rng = np.random.default_rng(0)
    q = jnp.asarray(rng.standard_normal(dim)); x = jnp.asarray(rng.standard_normal(3)); lam = jnp.asarray(rng.standard_normal(2))
    y, xi_Q, xi_M = forward_backward(arr, q, x, squared_error_grad(F(q, x), lam))
    out.update(bp_q=q, bp_x=x, bp_lam=lam, bp_y=y, bp_xi_Q=xi_Q, bp_xi_M=xi_M)

    eta = 0.05
    arr2 = parameterized_map(F, euclidean(dim, eta), in_dim=3, out_dim=2)
    rng = np.random.default_rng(1)
    q = jnp.asarray(rng.standard_normal(dim)); x = jnp.asarray(rng.standard_normal(3)); lam = jnp.asarray(rng.standard_normal(2))
    _, _, new_q = Phiconf(arr2).with_state(q).run_one(
        (x, trivial_omega(0)), lambda op: (squared_error_grad(op[0], lam), jnp.zeros(0)))
    out.update(step_q=q, step_x=x, step_lam=lam, step_new_q=new_q, step_eta=np.array(eta))

    m, n = 3, 2
    dim3 = n * m + n

    def F3(q, x):
        return q[: n * m].reshape(n, m) @ x + q[n * m:]

    rng = np.random.default_rng(2)
    W_true = jnp.asarray(rng.standard_normal((n, m))); b_true = jnp.asarray(rng.standard_normal(n))
    xs, lams = [], []
    for _ in range(20):
        x = jnp.asarray(rng.standard_normal(m)); xs.append(x); lams.append(W_true @ x + b_true)
    data = list(zip(xs, lams))
    arr3 = parameterized_map(F3, euclidean(dim3, eta=0.05), in_dim=m, out_dim=n)
    q, history = train(arr3, jnp.zeros(dim3), data, steps=5000)
    out.update(train_xs=np.stack(xs), train_lams=np.stack(lams), train_q=q, train_history=np.array(history))
    _save("gradient_descent", **out)


if __name__ == "__main__":
    heat(); wave(); functors(); composition(); gradient_descent()
