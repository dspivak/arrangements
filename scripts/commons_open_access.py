"""Wiring 2, OPEN ACCESS (`notes/COMMONS-DESIGN.md` §5): predictions 1, 2, 7 and 9.

    .venv/bin/python scripts/commons_open_access.py
"""

from commons_common import NS, PAR, STEPS, close, head, row, verdict

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.commons import (
    composite_port_form, d_omega, open_access, run, settle, split,
)


def main():
    head("WIRING 2 -- OPEN ACCESS.  boxes + fan-out + N catch couplers, nothing else.")
    met = []

    head("Prediction 1: converges to R* = r gamma K/(r gamma + N p K) from every interior start")
    row("N", "R* (run)", "R* (closed)", "x_i (run)", "x_i (closed)")
    ok = True
    for N in NS:
        for R0, x0 in ((90.0, 0.01), (5.0, 2.0), (1.0, 0.001)):
            R, x = split(settle(open_access(PAR, N, R0=R0, x0=x0), STEPS))
            ok &= close(R, PAR.R_open_access(N))
            ok &= close(float(x[0]), PAR.p * PAR.R_open_access(N) / PAR.gamma)
        row(N, round(R, 5), round(PAR.R_open_access(N), 5), round(float(x[0]), 6),
            round(PAR.p * PAR.R_open_access(N) / PAR.gamma, 6))
    met.append(verdict("1  open access converges, no cycle", ok))

    head("Prediction 2: rent dissipation")
    row("N", "R*", "E* = X R", "Pi = Np^2R^2/2g", "gamma r^2/2N")
    ok = True
    for N in (1, 5, 20, 100, 400):
        R, x = split(settle(open_access(PAR, N), STEPS))
        E, Pi = PAR.harvest(R, float(jnp.sum(x))), PAR.rent_from(R, x)
        row(N, round(R, 5), round(E, 5), round(Pi, 6), round(PAR.gamma * PAR.r ** 2 / (2 * N), 6))
        if N >= 100:
            ok &= close(Pi, PAR.gamma * PAR.r ** 2 / (2 * N), rtol=0.02)
    met.append(verdict("2  R* and rent fall to zero; E* maximal at beta = 1 (N = 1)", ok))

    head("Prediction 7 (open-access half): the composite one-form is exact")
    rng = np.random.default_rng(3)
    worst = 0.0
    for N in (1, 3, 6):
        f = composite_port_form(open_access(PAR, N))
        for _ in range(4):
            q = jnp.asarray(np.concatenate([[2.0 + abs(rng.normal())], rng.standard_normal(N)]))
            worst = max(worst, float(np.max(np.abs(d_omega(f)(q)))))
    row("max |d omega|", f"{worst:.3e}")
    met.append(verdict("7a d omega_oa = 0 to machine precision", worst < 1e-11))

    head("Prediction 9: the settled state is mobility-free; the stability bound shrinks in N")
    row("mobility", "R* (N=5)")
    ok = True
    for mu in (0.02, 0.005, 0.001):
        from arrangements.commons import Params
        R, _ = split(settle(open_access(Params(mobility=mu), 5), int(600 / mu)))
        row(mu, round(R, 6))
        ok &= close(R, PAR.R_open_access(5))
    row("N", "mu_max")
    from arrangements.commons import Params
    bounds = []
    for N in NS:
        par = Params(mobility=1.0)
        R = PAR.R_open_access(N)
        q = jnp.asarray(np.concatenate([[R], np.full(N, PAR.p * R / PAR.gamma)]))
        sharp = jnp.diag(jnp.concatenate([q[:1], jnp.full(N, -par.p)]))
        ev = np.linalg.eigvals(np.asarray(
            sharp @ jax.jacobian(composite_port_form(open_access(par, N)))(q)))
        ev = ev[ev.real < -1e-12]
        bounds.append(float(np.min(-2.0 * ev.real / np.abs(ev) ** 2)))
        row(N, round(bounds[-1], 6))
    ok &= bounds[0] > bounds[1] > bounds[2]
    met.append(verdict("9  settled state mobility-free; mu_max shrinks (1.500 -> 1.048)", ok))
    print("\n  NOTE: under open access mu_max only falls from 1.5 to 1.0 and never further,")
    print("  because N does not enter the effort block's rate at all -- the extractors are")
    print("  mutually invisible (§3). Under the sanction gamma' = gamma + Nps does enter and")
    print("  the bound collapses like 1/N (see commons_governed.py).")

    head(f"OPEN ACCESS: {sum(met)}/{len(met)} met")


if __name__ == "__main__":
    main()
