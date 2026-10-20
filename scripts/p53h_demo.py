"""Predictive coding against the precision sent forward (book plan P53 (h'), demo).

Four runs of ``Phi`` on the separated data of ``tests/test_nested_coding.py``, at one step size ``eta``:
P5 (i)'s two-level predictive coding (``level1`` around ``level2``, closed by ``data_box``) and the precision sent
forward (``evidence_hierarchy``, ``nu`` integrated), each from rest and from every belief at the grand mean with the
upper precision 40 and the lower precision ``1/(W+B)``. Plots the upper precision ``e^l2`` against step, with the
marginal-likelihood precision as a line, and prints each run's end.

    uv run python scripts/p53h_demo.py OUTDIR [eta] [steps]
"""

import sys

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.optimize import minimize

from arrangements.active_inference import run
from arrangements.nested_coding import evidence_hierarchy, free_energy
from tests.chain import flat_state
from tests.test_nested_coding import (
    J, K, SEPARATED, data_first, neg_log_marginal, spreads, stationary, unpack_df,
)

out = sys.argv[1]
eta = float(sys.argv[2]) if len(sys.argv) > 2 else 0.0005
steps = int(sys.argv[3]) if len(sys.argv) > 3 else 120000

x = SEPARATED
W, B = spreads(x)
g = float(x.mean())
l1_start, l2_start = np.log(1 / (W + B)), np.log(40.0)

ml = min((minimize(neg_log_marginal, t0, args=(x,), method="Nelder-Mead",
                   options=dict(xatol=1e-12, fatol=1e-14, maxiter=20000))
          for t0 in ([0, 0], [0, 3], [-1, -3])), key=lambda r: r.fun)
P1_ml, P2_ml = 1 / np.exp(ml.x)
P1_F, P2_F, _ = stationary(x)[0]

runs = {}
# predictive coding: flat state (nu, l2, mu_1..mu_J, l1)
for name, i1, i2 in (("predictive coding, from rest", None, None),
                     ("predictive coding, from the grand mean, upper precision 40",
                      jnp.concatenate([jnp.full(J, g), jnp.array([l1_start])]), jnp.array([g, l2_start]))):
    _, rec = run(data_first(eta, x, i1, i2)[1], steps, record=flat_state)
    R = np.array(rec)
    runs[name] = dict(P2=np.exp(R[:, 1]), P1=np.exp(R[:, 2 + J]), last=R[-1],
                      F=float(free_energy(x, *unpack_df(jnp.asarray(R[-1])))))
# precision sent forward: flat state (l1, nu, l2)
for name, i1, i2 in (("precision sent forward, from rest", 0.0, None),
                     ("precision sent forward, from the grand mean, upper precision 40",
                      float(l1_start), jnp.array([g, l2_start]))):
    _, rec = run(evidence_hierarchy(x, eta, True, i1, i2), steps, record=flat_state)
    R = np.array(rec)
    runs[name] = dict(P2=np.exp(R[:, 2]), P1=np.exp(R[:, 0]), last=R[-1])

lines = [f"data: SEPARATED (J={J} groups of K={K}, centers -3,-1,1,3, seed 0); W={W:.6f}, B={B:.6f}, grand mean {g:.6f}",
         f"eta={eta}, steps={steps}; funnel start: every belief at {g:.6f}, Pi1=1/(W+B)={1/(W+B):.6f}, Pi2=40",
         f"marginal-likelihood precisions: Pi1={P1_ml:.6f}, Pi2={P2_ml:.6f}",
         f"plain F's interior minimum:     Pi1={P1_F:.6f}, Pi2={P2_F:.6f}",
         f"predictive coding's stability edge in Pi2, 2/(eta (J+1)) = {2 / (eta * (J + 1)):.3f}"]
for name, r in runs.items():
    tail = r["P2"][-steps // 10:]
    lines.append(f"{name}: final Pi1={r['P1'][-1]:.6f}, Pi2={r['P2'][-1]:.6f}; "
                 f"last tenth Pi2 in [{tail.min():.6f}, {tail.max():.6f}], max over run {r['P2'].max():.6f}"
                 + (f"; F={r['F']:.4f}" if "F" in r else ""))
print("\n".join(lines))
open(f"{out}/demo-eta{eta}.log", "w").write("\n".join(lines) + "\n")

fig, ax = plt.subplots(figsize=(8, 5))
style = {"predictive coding, from rest": ("#c0392b", "-"),
         "predictive coding, from the grand mean, upper precision 40": ("#c0392b", "--"),
         "precision sent forward, from rest": ("#1f5fa8", "-"),
         "precision sent forward, from the grand mean, upper precision 40": ("#1f5fa8", "--")}
n = np.arange(steps + 1)
for name, r in runs.items():
    c, ls = style[name]
    ax.plot(n, r["P2"], color=c, ls=ls, lw=1.8, label=name)
ax.axhline(P2_ml, color="0.2", lw=1, ls=":", label=f"marginal-likelihood precision {P2_ml:.4f}")
ax.axhline(2 / (eta * (J + 1)), color="#c0392b", lw=0.8, ls=":", alpha=0.6,
           label=f"predictive coding's stability edge, 2/(η(J+1)) = {2 / (eta * (J + 1)):.0f}")
ax.set_yscale("log")
ax.set_xlabel("step of Φ")
ax.set_ylabel("upper precision  $e^{l_2}$")
ax.set_title(f"Upper precision, separated data, step size η = {eta}")
ax.legend(fontsize=8, loc="best", frameon=False)
fig.tight_layout()
fig.savefig(f"{out}/upper-precision-eta{eta}.pdf")
fig.savefig(f"{out}/upper-precision-eta{eta}.png", dpi=150)
