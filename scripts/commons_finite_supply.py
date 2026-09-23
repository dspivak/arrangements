"""The FINITE-SUPPLY and FRUSTRATION variants (`notes/COMMONS-DESIGN.md` §7):
predictions 10-14.

    .venv/bin/python scripts/commons_finite_supply.py
"""

from commons_common import EPS, FIN, PAR, R_FIN, close, head, row, verdict

import jax.numpy as jnp
import numpy as np

from arrangements.commons import governed, insulated, open_access, run, settle, split


def T_eps(arr, R0=R_FIN, eps=EPS, steps=60_000, every=2):
    tr = np.asarray(run(arr, steps, every=every))
    hit = np.nonzero(tr[:, 0] <= eps * R0)[0]
    return None if len(hit) == 0 else float(hit[0] * every * FIN.mobility)


def main():
    head("FINITE SUPPLY (r = 0).  The only settled state is exhaustion.")
    met = []

    head("Prediction 10: R_0 - R(T) = int E dt, hence T >= (R_0 - R(T))/max E")
    R0, N = 100.0, 5
    tr = np.asarray(run(open_access(FIN, N, R0=R0, x0=FIN.p * R0), 200_000, every=1))
    R, X = tr[:, 0], tr[:, 1:].sum(axis=1)
    E = R * X
    consumed, T = R0 - R[-1], (len(tr) - 1) * FIN.mobility
    row("consumed", round(consumed, 6), "mu * sum E", round(FIN.mobility * E[:-1].sum(), 6))
    row("T", round(T, 3), "bound (R0-R)/max E", round(consumed / E.max(), 5))
    row("max E / final E", f"{E.max() / E[-1]:.3e}")
    met.append(verdict("10 identity and bound hold on the run", close(consumed,
                       FIN.mobility * E[:-1].sum(), 2e-3) and T >= consumed / E.max(),
                       "the equality case T = R_0/E is an idealization, not a run: E falls "
                       "by 3+ orders of magnitude"))

    head("Predictions 11-13 in the quasi-static regime their derivations assume")
    print(f"  (R_0 = {R_FIN}, eps = {EPS}, so X_0 = {FIN.p * R_FIN:.3f} N << gamma = 1)")
    row("N", "T_oa (run)", "T_oa (closed)", "T_gov (run)", "T_gov (closed)", "T_ins (run)")
    s = 100.0
    ok11 = ok12 = ok13 = True
    Toa, Tgov = [], []
    for N in (1, 2, 4, 8, 16):
        t_oa = T_eps(open_access(FIN, N, R0=R_FIN, x0=FIN.p * R_FIN))
        c_oa = PAR.gamma / (N * PAR.p * R_FIN) * (1 / EPS - 1)
        t_g = T_eps(governed(FIN, N, s=s, R0=R_FIN,
                             x0=FIN.p * R_FIN / (PAR.gamma + N * PAR.p * s)), steps=30_000)
        c_g = (PAR.gamma + N * PAR.p * s) / (N * PAR.p * R_FIN) * (1 / EPS - 1)
        t_i = T_eps(insulated(FIN, N, Rhat=R_FIN, R0=R_FIN, x0=FIN.p * R_FIN))
        c_i = PAR.gamma / (N * PAR.p * R_FIN) * np.log(1 / EPS)
        row(N, round(t_oa, 2), round(c_oa, 2), round(t_g, 2), round(c_g, 2), round(t_i, 2))
        if N <= 4:                 # X_0/gamma = 0.05 N: quasi-static only while that is small
            ok11 &= close(t_oa, c_oa, 0.10)
        ok12 &= close(t_g, c_g, 0.02)
        ok13 &= close(t_i, c_i, 0.02)
        Toa.append(t_oa)
        Tgov.append(t_g)
    row("ratio N=1 -> 16", "open access", round(Toa[0] / Toa[-1], 2),
        "governed", round(Tgov[0] / Tgov[-1], 2), f"limit {s / R_FIN * (1 / EPS - 1):.0f}")
    met.append(verdict("11 T_oa propto 1/N (N <= 4; X_0/gamma <= 0.2)",
                       ok11 and close(Toa[0] / Toa[2], 4.0, 0.08),
                       "to 2%, 4%, 9% at N = 1, 2, 4 -- the residual grows with N as the "
                       "quasi-static hypothesis degrades, and by N = 8, 16 the run runs "
                       "ahead of the closed form outright (8.1 vs 10, 3.1 vs 5)"))
    met.append(verdict("12 T_gov flattens; an N-independent limit", ok12 and Tgov[0] / Tgov[-1] < 2))
    met.append(verdict("13 T_ins = (gamma/NpR_0) ln(1/eps), and T_ins < T_oa < T_gov",
                       ok13 and Toa[0] > T_eps(insulated(FIN, 1, Rhat=R_FIN, R0=R_FIN,
                                                         x0=FIN.p * R_FIN))))

    head("Prediction 11 OUTSIDE that regime -- at the note's own reference start it FAILS")
    print("  R_0 = 100 gives X_0 = N, so the stock is the FAST variable and the effort is")
    print("  frozen through the collapse. The 1/N law is a slow-stock statement; §7 does not")
    print("  say so.")
    row("N", "T_oa (run)", "T_oa (closed)", "ratio")
    bad = []
    for N in (2, 4, 8):
        t = T_eps(open_access(FIN, N, R0=100.0, x0=FIN.p * 100.0), R0=100.0, eps=0.01,
                  steps=400_000, every=5)
        c = PAR.gamma / (N * PAR.p * 100.0) * 99.0
        bad.append(t)
        row(N, round(t, 3), round(c, 3), round(t / c, 4))
    row("T(2)/T(4)", round(bad[0] / bad[1], 2), "predicted", 2.0)
    verdict("11 at the reference start", False, "measured ratio "
            f"{bad[0] / bad[1]:.1f} against a predicted 2.0")

    head("Prediction 14: at the finite-supply settled state dissipation is zero, all wirings")
    row("N", "wiring", "R", "E = XR")
    ok = True
    for N in (2, 8):
        for name, arr in (("open access", open_access(FIN, N, R0=R_FIN, x0=FIN.p * R_FIN)),
                          ("governed", governed(FIN, N, s=100.0, R0=R_FIN, x0=FIN.p * R_FIN)),
                          ("insulated", insulated(FIN, N, Rhat=R_FIN, R0=R_FIN,
                                                  x0=FIN.p * R_FIN))):
            R, x = split(settle(arr, 600_000))
            E = PAR.harvest(R, float(jnp.sum(x)))
            row(N, name, f"{R:.3e}", f"{E:.3e}")
            ok &= E < 1e-4
    met.append(verdict("14 E -> 0 at exhaustion, every wiring and every N", ok))

    head("FRUSTRATION, corrected (§7, strain S8): the sanction bounds total spend in N")
    row("N", "X_oa(R) = NpR/g", "X_gov(R) = NpR/(g+Nps)", "R/s")
    R, s = 40.0, 20.0
    for N in (1, 4, 16, 64, 256):
        row(N, round(N * PAR.p * R / PAR.gamma, 4),
            round(N * PAR.p * R / (PAR.gamma + N * PAR.p * s), 4), round(R / s, 4))
    print("\n  The plan's wording -- 'N competitors keep E high where one extractor would let")
    print("  it fall with R' -- is FALSE in this model: with quadratic effort cost every")
    print("  extractor's effort is proportional to R, so E falls with R at every N, and the")
    print("  1/N lifetime is N mouths. The statement with content is the table above.")

    head(f"FINITE SUPPLY: {sum(met)}/{len(met)} met (plus one recorded failure outside regime)")


if __name__ == "__main__":
    main()
