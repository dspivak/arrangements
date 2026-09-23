"""Wiring 3, INSULATED (`notes/COMMONS-DESIGN.md` §6): predictions 6 and 7.

The same boxes and the same fan-out; each catch coupler's one-form replaced by the stale
``-x_i dR - Rhat dx_i``. This is the note's highest-stakes check (§10's F3): the
composite cannot be exact, and the failure of exactness is the error in the account.

    .venv/bin/python scripts/commons_insulated.py
"""

from commons_common import PAR, STEPS, close, head, row, verdict

import jax.numpy as jnp
import numpy as np

from arrangements.commons import (
    composite_port_form, d_omega, insulated, open_access, settle, split,
)


def main():
    head("WIRING 3 -- INSULATED.  same boxes, same wiring, a stale catch coupler.")
    met = []

    head("Prediction 6: effort is deaf to the stock; the collapse threshold is N p Rhat/gamma = r")
    N = 5
    row("Rhat", "N p Rhat/g", "x_i (run)", "p Rhat/g", "R (run)", "K(1 - NpRhat/gr)")
    ok = True
    for Rhat in (4.0, 10.0, 19.0, 20.0, 50.0):
        R, x = split(settle(insulated(PAR, N, Rhat=Rhat, R0=90.0), STEPS))
        X = N * PAR.p * Rhat / PAR.gamma
        pred = max(0.0, PAR.K * (1 - X / PAR.r))
        row(Rhat, round(X, 3), round(float(x[0]), 6),
            round(PAR.p * Rhat / PAR.gamma, 6), round(R, 6), round(pred, 6))
        ok &= close(float(x[0]), PAR.p * Rhat / PAR.gamma, 1e-5)
        if X < PAR.r - 1e-9:
            ok &= abs(R - pred) < 1e-2 * pred
        elif X > PAR.r + 1e-9:
            ok &= R < 1e-5                      # exponential collapse
        else:
            # exactly at the threshold Rdot = -rR^2/K, so R ~ K/(r t): algebraic, not
            # exponential. 600 time units gives K/(r T) = 0.167, which is what the run
            # shows. The note's dichotomy is right; the rate at the knife edge is not in it.
            ok &= abs(R - PAR.K / (PAR.r * STEPS * PAR.mobility)) < 1e-3
    # deafness: the settled effort does not move with the stock's initial condition
    for R0 in (5.0, 99.0):
        _, x2 = split(settle(insulated(PAR, N, Rhat=10.0, R0=R0), STEPS))
        ok &= close(float(x2[0]), PAR.p * 10.0 / PAR.gamma, 1e-5)
    met.append(verdict("6  x_i -> p Rhat/gamma, deaf; threshold N p Rhat/gamma = r exact", ok))

    head("Prediction 7 (the F3 check): the composite one-form is NOT closed, and cannot be")
    row("N", "Rhat", "(d omega)_{R,x_i}", "off-diagonal", "omega_ins - omega_oa")
    ok = True
    rng = np.random.default_rng(9)
    for N in (1, 3, 6):
        f_oa = composite_port_form(open_access(PAR, N))
        for Rhat in (0.0, 10.0, 1e4):
            f = composite_port_form(insulated(PAR, N, Rhat=Rhat))
            q = jnp.asarray(np.concatenate([[2.0 + abs(rng.normal())], rng.standard_normal(N)]))
            dw = np.asarray(d_omega(f)(q))
            eta = np.asarray(f(q)) - np.asarray(f_oa(q))
            row(N, Rhat, round(float(dw[0, 1]), 12), f"{np.max(np.abs(dw[1:, 1:])):.1e}",
                f"(R - Rhat) = {float(q[0]) - Rhat:.4f} vs {eta[1]:.4f}")
            ok &= abs(dw[0, 1] - 1.0) < 1e-11
            ok &= np.max(np.abs(dw[1:, 1:])) < 1e-11
            ok &= abs(eta[0]) < 1e-10 and np.max(np.abs(eta[1:] - (float(q[0]) - Rhat))) < 1e-9
    met.append(verdict("7c d omega_ins has (R, x_i) component exactly 1 for every Rhat; "
                       "eta = sum_i (R - Rhat) dx_i", ok))
    print("\n  This is the note's claim (i) and it is a theorem, not an interpretation: within")
    print("  one potential the extractor's hearing of R and the resource's being harvested by")
    print("  x_i are the same mixed partial, so cutting the hearing cuts the harvest -- but")
    print("  the insulated extractor does take x_i R. No potential gives this wiring. F3 is")
    print("  not met: the wiring is not expressible as an exact form.")

    head(f"INSULATED: {sum(met)}/{len(met)} met")


if __name__ == "__main__":
    main()
