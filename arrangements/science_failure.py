"""Science at two scales, test (k): publishing failed replications, natively.

Predictions, fixed before this file existed, in ``notes/SCIENCE-FAILURE-REPORTS-PREDICTIONS-2026-10-01.md``
§2 (with "Numbers fixed before code" at its end); verdicts in its §3. Quoted:

  (k1) Lower-the-seed: 0 of 20 seeds end with any holder on a wrong hypothesis (silent: 0); the truth reaches
  every holder in at least as many seeds as silent (19 of 20 ring, 20 of 20 all-to-all); arrival time per ring
  hop within 10% of silent's; the seed mass planted on hypotheses later rejected is lower than silent's in at
  least 15 of 20 seeds.
  (k2) Counted as evidence: at beta = 0.1 within 2 seeds of silent; as beta grows, the count of seeds in which
  the truth never reaches some holder rises, above zero at beta = 10. A true hypothesis can be reported dead by
  bad luck and the reports keep it dead: a negative echo chamber.
  (k3) Mechanism: in (k2)'s failures, the first report against the truth comes from a holder whose first few
  local draws favoured a neighbouring hypothesis; the run reports, per failing seed, whether that holds.
  (k4) Same split as stage 2: a report used to lower a seed is safe, one counted as evidence is not.

THE CONSTRUCTION. Stage 2's run-2 holder (``science_native.conclusion_holder``, ``frame="euc"``, seed arm),
with two more state blocks, all Euclidean with response ``dt``: state ``(p, z, a)``.
* ``p``: the simplex weights; data forcing the replicator field ``p * (g - <g, p>)`` (NOT exact, as in stage 2),
  seed forcing exact (below).
* ``z``: forcing ``-p * (f - <f, p>)``, own draws only: the net weight its own data have taken off each
  hypothesis. Zero exactly for a hypothesis never received.
* ``a``: forcing ``eps * c~``, the planted inflow; an observation register that feeds nothing back.
Out-port ``(c, R)``: the sharpened conclusion ``c = p^{1/tau}/sum`` and the FAILURE REPORT
``R = exp(-z0 / z)`` for ``z > 0``, else 0 (C-infinity, not analytic; nonnegative, below 1). A reader averages
its neighbours' conclusions into ``c̄`` and their reports into ``R̄``.
Arms: ``"silent"`` (reports not read; stage 2's seed arm); ``"lower"`` (``c~ = (1 - R̄) c̄``, ``S = sum c~``,
seed forcing ``eps (c~ - S p)``, the differential of ``eps(<c~, p> - S|p|^2/2)``, exact, summing to zero);
``"evidence"`` (fitness ``f - beta R̄`` in the replicator field, seed as silent). Every forcing is zero on the
in-ports, so nothing is pushed back: a report, like a conclusion, is a value sent forward.
CORRECTION (§4 of the note, ``form="shortfall"``). §3's report (``form="tally"``) is a lifetime tally: a holder at
0.99 on the truth still reported it failed. The corrected report reads the CURRENT shortfall ``x = b - p``, where
``b`` is a fourth state block with the seed's own forcing (``eps (c̄ - b)``, or ``eps (c~ - S b)`` when lowering):
the weight the seed alone would now have planted. ``R = report_shortfall(x)``, ``exp(-x0/x)`` for ``x > 0``, its
derivative given by a custom JVP that evaluates it without ``0 * inf``: exactly C-infinity, no cut.
Predictions (§4): (k5) lower-the-seed: §3's (k1) counts within 1 seed each way. (k6) counted as evidence: the
truth, once reached, is lost in 0 of 20 seeds at every beta on both networks. (k7) the never-reached counts stay
within 1 of §3's.
NOT EXACT OR NOT SMOOTH: the replicator data forcing (as stage 2), the ``z`` forcing (minus one block of it),
the world's table lookup (``draw_source``, as stage 1), and ``R``, which is smooth but not analytic at 0.
"""

from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement
from .coalgebra import configuration
from .functors import Phi
from .rvect import euclidean
from .science_native import _IN_DIR, _IN_POS, draw_source, neighbours, wire_boxes

Z0 = 0.5


def report(z):
    """The failure report ``exp(-z0/z)`` for ``z > 0``, else 0 (safe under autodiff). Cut at
    ``z = z0/700``, where the report is below ``1e-304``: without the cut, a tiny positive ``z`` gives
    ``0 * inf = NaN`` in the readout's derivative, which ``compose_seq``'s pullback evaluates (times a zero
    cotangent), and the first recorded run was NaN on the ring (notes, §3)."""
    pos = z > Z0 / 700.0
    return jnp.where(pos, jnp.exp(-Z0 / jnp.where(pos, z, 1.0)), 0.0)


@jax.custom_jvp
def report_shortfall(x):
    """§4's report, ``exp(-x0/x)`` for ``x > 0``, else 0: exactly C-infinity, no cut."""
    pos = x > 0
    return jnp.where(pos, jnp.exp(-Z0 / jnp.where(pos, x, 1.0)), 0.0)


@report_shortfall.defjvp
def _report_shortfall_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    pos = x > 0
    xs = jnp.where(pos, x, 1.0)
    d = jnp.where(pos, jnp.exp(jnp.log(Z0) - Z0 / xs - 2.0 * jnp.log(xs)), 0.0)
    return report_shortfall(x), d * t


def failure_holder(p0, wj, mus, dt: float, arm: str, eps: float, beta: float, tau_inv: int,
                   label: str = "holder", form: str = "tally") -> Arrangement:
    """A holder ``I -> <d 2n + 1 | 2n>``: state ``(p, z, a)``, out-port ``(c, R)``, in-ports its ``d``
    neighbours' ``(c, R)`` and its draw."""
    n = p0.shape[0]
    d = wj.shape[0]

    def inputs(n_in):
        nb = n_in[: d * 2 * n].reshape(d, 2 * n)
        cbar, rbar = wj @ nb[:, :n], wj @ nb[:, n:]
        ll = -0.5 * (n_in[d * 2 * n] - mus) ** 2
        return cbar, rbar, ll - jnp.min(ll) + 1.0

    if arm == "lower":
        def V_seed(p, n_in):
            cbar, rbar, _ = inputs(n_in)
            ct = (1.0 - rbar) * cbar
            return eps * (jnp.dot(ct, p) - 0.5 * jnp.sum(ct) * jnp.dot(p, p))
    else:
        def V_seed(p, n_in):
            cbar, _, _ = inputs(n_in)
            return eps * (jnp.dot(cbar, p) - 0.5 * jnp.dot(p, p))
    dV = jax.grad(V_seed, argnums=0)

    def force(q, n_in):
        p = q[:n]
        cbar, rbar, f = inputs(n_in)
        g = f - beta * rbar if arm == "evidence" else f
        xi_p = p * (g - jnp.dot(g, p)) + dV(p, n_in)
        xi_z = -p * (f - jnp.dot(f, p))
        ct = (1.0 - rbar) * cbar if arm == "lower" else cbar
        b = q[3 * n:]
        xi_b = eps * (ct - jnp.sum(ct) * b) if arm == "lower" else eps * (cbar - b)
        return jnp.concatenate([xi_p, xi_z, eps * ct, xi_b])

    def readout(q):
        r = q[:n] ** tau_inv
        if form == "tally":
            R = report(q[n:2 * n])
        elif form == "shortfall":
            R = report_shortfall(q[3 * n:] - q[:n])
        else:
            raise ValueError(form)
        return jnp.concatenate([r / jnp.sum(r), R])

    init = jnp.concatenate([p0, jnp.zeros(3 * n)])
    return Arrangement(
        P=configuration(euclidean(4 * n, dt), init=init),
        in_dim_M=0, out_dim_M=0, in_dim_N=d * 2 * n + 1, out_dim_N=2 * n,
        out_f=lambda q, m_out: readout(q),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=lambda q, m_out, n_in: (force(q, n_in), jnp.zeros_like(m_out), jnp.zeros_like(n_in)),
        label=label,
    )


def failure_community(draws, A, mus, slices, dt, hold, arm, eps, beta, tau_inv, form="tally") -> Arrangement:
    """World plus ``m`` failure holders wired by adjacency ``A`` (each reads all ``2n`` coordinates of each
    neighbour's out-port), closed by ``wire_boxes``."""
    T, m = draws.shape
    n = mus.shape[0]
    nb = neighbours(A)
    boxes, feeds = [draw_source(draws, hold)], [[]]
    for j in range(m):
        p0 = np.zeros(n); p0[list(slices[j])] = 1.0 / len(slices[j])
        wj = jnp.full(len(nb[j]), 1.0 / len(nb[j]))
        boxes.append(failure_holder(jnp.asarray(p0), wj, mus, dt, arm, eps, beta, tau_inv, f"holder {j}",
                                    form))
        feeds.append([(k + 1, c) for k in nb[j] for c in range(2 * n)] + [(0, j)])
    return wire_boxes(boxes, feeds)


@partial(jax.jit, static_argnames=("A_key", "slices", "hold", "arm", "eps", "beta", "tau_inv", "form"))
def _failure_run(draws, mus, dt, A_key, slices, hold, arm, eps, beta, tau_inv, form):
    A = np.asarray(A_key)
    T, m = draws.shape
    n = mus.shape[0]
    arr = failure_community(draws, A, mus, slices, dt, hold, arm, eps, beta, tau_inv, form)
    O = Phi(arr)

    def one(carry, _):
        s, lo, dev = carry
        _, _, s2 = O.with_state(s).run_one(_IN_POS, lambda _o: _IN_DIR)
        P = arr.P.position(s2)[1:].reshape(m, 4 * n)[:, :n]
        return (s2, jnp.minimum(lo, jnp.min(P)), jnp.maximum(dev, jnp.max(jnp.abs(P.sum(1) - 1.0)))), None

    def per_draw(carry, _):
        carry, _ = jax.lax.scan(one, carry, None, length=hold)
        return carry, arr.P.position(carry[0])[1:].reshape(m, 4 * n)

    (_, lo, dev), traj = jax.lax.scan(per_draw, (arr.P.init, jnp.inf, 0.0), None, length=T)
    return traj, lo, dev


def failure_run(draws, A, mus, slices, dt, hold, arm, eps, beta, tau, form="tally"):
    """Over seeds (``draws`` of shape ``(seeds, T, m)``): ``p``, ``z``, ``a``, ``b`` after each draw, each
    ``(seeds, T, m, n)``; the minimum weight and ``max |sum p - 1|`` over every step of ``Phi``."""
    A_key = tuple(tuple(int(v) for v in row) for row in np.asarray(A))
    sl = tuple(tuple(int(i) for i in s) for s in slices)
    n = len(mus)
    f = lambda dr: _failure_run(dr, jnp.asarray(mus, float), dt, A_key, sl, int(hold), arm, float(eps),
                                float(beta), int(round(1.0 / tau)), form)
    traj, lo, dev = jax.vmap(f)(jnp.asarray(draws, float))
    traj = np.asarray(traj)
    return (traj[..., :n], traj[..., n:2 * n], traj[..., 2 * n:3 * n], traj[..., 3 * n:], np.asarray(lo),
            np.asarray(dev))
