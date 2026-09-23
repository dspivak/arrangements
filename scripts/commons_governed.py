"""Wiring 1, GOVERNED (`notes/COMMONS-DESIGN.md` §4): predictions 3, 4, 5, 7, 8, 15.

Open access **plus one exact N-ary coupler**. Nothing else changes.

    .venv/bin/python scripts/commons_governed.py
"""

from commons_common import NS, PAR, S_STAR, STEPS, close, head, row, verdict

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.commons import (
    Params, composite_port_form, d_omega, external_enforcer, governed, open_access,
    returned_covector, sanction_coupler, settle, split, unmoved_monitor,
)


def settle_gov(N, s):
    """The mobility is the integrator's: the sanction raises the effort rate to
    gamma' = gamma + Nps, so the step must respect gamma'. The cost is untouched."""
    par = Params(mobility=min(0.01, 0.5 / (PAR.gamma + N * PAR.p * s)))
    return split(settle(governed(par, N, s=s), int(600 / par.mobility)))


def main():
    head("WIRING 1 -- GOVERNED.  open access + one exact N-ary sanction coupler.")
    met = []

    head("Prediction 3: any s > 0 settles strictly above open access; s = 0 reproduces it")
    row("N", "s", "R_gov (run)", "R_gov (closed)", "R_oa")
    ok = True
    for N in NS:
        prev = None
        for s in (0.0, 1.0, 10.0, 100.0, 1000.0):
            R, _ = settle_gov(N, s)
            ok &= close(R, PAR.R_governed(N, s))
            if prev is not None:
                ok &= R > prev
            prev = R
            row(N, s, round(R, 5), round(PAR.R_governed(N, s), 5),
                round(PAR.R_open_access(N), 5))
    met.append(verdict("3  R_gov = K beta'/(1+beta'), monotone in s, s=0 is open access", ok))

    head("Prediction 4: s = K/r = 100 hits the cooperative optimum, for every N, p, gamma")
    row("N", "p", "gamma", "R_gov (run)", "R_coop")
    ok = True
    for N in NS:
        R, _ = settle_gov(N, S_STAR)
        ok &= close(R, PAR.R_cooperative(N))
        row(N, PAR.p, PAR.gamma, round(R, 5), round(PAR.R_cooperative(N), 5))
    for p, g in ((0.02, 1.0), (0.01, 4.0), (0.05, 0.5)):
        par = Params(p=p, gamma=g, mobility=0.002)
        for N in (2, 7):
            R, _ = split(settle(governed(par, N, s=par.s_optimal()), 300_000))
            ok &= close(R, par.R_cooperative(N), rtol=1e-3)
            row(N, p, g, round(R, 5), round(par.R_cooperative(N), 5))
    met.append(verdict("4  R_gov = R_coop iff s = K/r, independent of N, p, gamma", ok))

    head("Prediction 5: as N grows the governed state tends to maximum sustainable yield")
    N = 200
    par = Params(mobility=0.5 / (PAR.gamma + N * PAR.p * S_STAR))
    R, x = split(settle(governed(par, N, s=S_STAR), 400_000))
    row("", "R", "E = XR", "rent")
    row("governed N=200", round(R, 4), round(PAR.harvest(R, float(jnp.sum(x))), 4),
        round(PAR.rent_from(R, x), 5))
    row("MSY limit", "K/2 = 50", "rK/4 = 25", "pKr/4 = 0.25")
    R_oa, x_oa = split(settle(open_access(PAR, N), STEPS))
    row("open access", round(R_oa, 4), round(PAR.harvest(R_oa, float(jnp.sum(x_oa))), 4),
        round(PAR.rent_from(R_oa, x_oa), 6))
    ok = (close(R, 50.0, 2e-2) and close(PAR.harvest(R, float(jnp.sum(x))), 25.0, 4e-2)
          and close(PAR.rent_from(R, x), 0.25, 5e-2) and R_oa < 0.6)
    met.append(verdict("5  governed -> (50, 25, 0.25); open access -> (0, 0, 0)", ok))
    print("\n  NOTE: §9's 'rent -> pKr/4' is the rent p R X - (gamma/2) sum x_i^2, not §5's")
    print("  Pi_oa = N p^2 R^2/(2 gamma), which assumes x_i = pR/gamma and is false under the")
    print("  sanction. Using the open-access formula here makes the prediction appear to fail.")

    head("Prediction 7 (governed half): the composite one-form is still exact")
    rng = np.random.default_rng(4)
    worst = 0.0
    for N in (1, 3, 6):
        f = composite_port_form(governed(PAR, N, s=0.37))
        for _ in range(4):
            q = jnp.asarray(np.concatenate([[2.0 + abs(rng.normal())], rng.standard_normal(N)]))
            worst = max(worst, float(np.max(np.abs(d_omega(f)(q)))))
    row("max |d omega|", f"{worst:.3e}")
    met.append(verdict("7b d omega_gov = 0: adding the sanction keeps it exact", worst < 1e-11))

    head("Prediction 8: the sanction is reciprocal; the note's external enforcer is NOT")
    N, s = 4, 0.6
    F = returned_covector(sanction_coupler(s, N))
    e = jnp.asarray(np.random.default_rng(5).standard_normal(N))
    J = np.asarray(jax.jacobian(F)(e))
    row("DF (sanction)", "all entries = s?", bool(np.allclose(J, s)))
    row("d omega", f"{float(np.max(np.abs(d_omega(F)(e)))):.3e}")
    flat = returned_covector(unmoved_monitor(s, N))
    e2 = jnp.asarray(np.random.default_rng(6).standard_normal(N + 1))
    dflat = float(np.max(np.abs(d_omega(flat)(e2))))
    G = returned_covector(external_enforcer(N))
    dg = float(np.max(np.abs(d_omega(G)(e2))))
    row("d omega (monitor returning sX, unmoved)", f"{dflat:.3e}", "-> EXACT")
    row("d omega (monitor whose own state y sets the sanction)", f"{dg:.3e}", "-> not exact")
    met.append(verdict("8  mutual sanction exact; unilateral enforcer non-exact",
                       float(np.max(np.abs(d_omega(F)(e)))) < 1e-12 and dg > 1e-3,
                       "but the note's own form of the enforcer is exact -- see below"))
    print("\n  CORRECTION TO §4. A monitor that returns sX to every user and is unmoved by")
    print("  them is EXACT: its one-form is d((s/2)X^2) with a zero component on the monitor,")
    print("  and X does not mention the monitor's coordinate, so there is no mixed partial to")
    print("  break. Non-exactness needs the sanction to depend on the monitor's own state.")

    head("Prediction 9 (governed half): the stability bound collapses like 1/N")
    row("N", "mu_max", "2/(gamma + Nps)")
    for N in NS:
        par = Params(mobility=1.0)
        R = PAR.R_governed(N, S_STAR)
        x = PAR.p * R / (PAR.gamma + N * PAR.p * S_STAR)
        q = jnp.asarray(np.concatenate([[R], np.full(N, x)]))
        sharp = jnp.diag(jnp.concatenate([q[:1], jnp.full(N, -par.p)]))
        ev = np.linalg.eigvals(np.asarray(
            sharp @ jax.jacobian(composite_port_form(governed(par, N, s=S_STAR)))(q)))
        ev = ev[ev.real < -1e-12]
        row(N, round(float(np.min(-2.0 * ev.real / np.abs(ev) ** 2)), 6),
            round(2.0 / (PAR.gamma + N * PAR.p * S_STAR), 6))

    head("Prediction 15: forgone rent at the regenerating settled state (NOT dissipation)")
    row("N", "Pi_coop", "Pi_oa", "gap")
    for N in (1, 5, 20, 400):
        g = PAR.rent_cooperative(N) - PAR.rent(PAR.R_open_access(N), N)
        row(N, round(PAR.rent_cooperative(N), 6), round(PAR.rent(PAR.R_open_access(N), N), 6),
            round(g, 6))
    ok = close(PAR.rent_cooperative(400) - PAR.rent(PAR.R_open_access(400), 400),
               PAR.p * PAR.K * PAR.r / 4.0, 0.02)
    met.append(verdict("15 forgone rent -> pKr/4 = 0.25", ok))

    head(f"GOVERNED: {sum(met)}/{len(met)} met")


if __name__ == "__main__":
    main()
