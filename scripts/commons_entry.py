"""STAGE A: §12b's two open items, O2 then O1 (`notes/COMMONS-DESIGN.md` §12b).

    .venv/bin/python scripts/commons_entry.py
"""

from commons_common import head, row, verdict

import jax.numpy as jnp
import numpy as np

from arrangements.attention import (
    mean, message, spatial_arrangement, spatial_config, spatial_kappa, zero_cost,
    zero_message,
)
from arrangements.commons import Params, composite_port_form, open_access
from arrangements.commons_entry import (
    DH, DP, EXTERIOR, MDIM, effective_N, entry_count, entry_system, kappas,
    marginal_threshold, net_benefit, run, settle, unpack, zero_rent_threshold,
)

TAU = 2.0
PAR = Params(mobility=0.1)
PM = 0.02
STEPS = 300_000


def settled(placements, outside, power=1.0, R0=90.0, x0=0.9, steps=STEPS):
    N = len(placements)
    q = settle(entry_system(PAR, placements, outside, tau=TAU, R0=R0, x0=x0,
                            place_mobility=PM, power=power), steps)
    R, x, _ = unpack(q, N)
    return float(R), np.asarray(x), np.asarray(kappas(q, N, TAU)), q


def main():
    head("O2 -- what the kernel weighs, and how the catch rides on it")
    print("""
  The resource is the SOLE SENDER and emits m_Res = (h, w) = (R, 1): the signal carries
  the stock on offer, the weight carries the resource counted once. Extractors emit the
  zero message 0_M and are pure receivers. Extractor i's received input is then

      rho_a(q, m, e)_i = ((kappa_i R, kappa_i), l_i),   kappa_i = exp(-|l_i - l_Res|^2/2tau)

  and the term it carries is  -x_i h(e_i) = -x_i kappa_i R,  bilinear in effort and
  received signal, so the cost stays in the quadratic family.""")
    n, N, R = 4, 3, 37.0
    ells = [0.0, 0.3, 1.1, 2.5]
    a = spatial_arrangement(n, DH, DP, TAU, cost=zero_cost)
    q = spatial_config(ells, [1.0] * n)
    m_out = jnp.concatenate([message(R, 1.0)] + [zero_message(DH)] * N)
    row("i", "l_i", "kappa_i", "signal h", "weight w", "mean h/w")
    ok = True
    for i in range(1, N + 1):
        e = a.received(q, m_out, EXTERIOR, i + 1)
        k = float(spatial_kappa(jnp.array([ells[i]]), jnp.array([ells[0]]), TAU))
        row(i, ells[i], round(k, 6), round(float(e[0]), 6), round(float(e[1]), 6),
            round(float(mean(e[:MDIM])[0]), 6))
        ok &= abs(float(e[0]) - k * R) < 1e-12 and abs(float(mean(e[:MDIM])[0]) - R) < 1e-9
    verdict("O2 the interface is writable", ok)
    print("""
  THE MEAN READOUT IS PROXIMITY-BLIND: mean(kappa R, kappa) = R for every kappa > 0. What
  an extractor BELIEVES the stock to be does not depend on how near it is, so the catch
  cannot ride on the mean; it must ride on the weighted signal. That is forced by the
  paper's own kernel, not chosen -- and it is O2's whole content.""")

    head("O2 -- at kappa = 1 the composite is exactly §2's fixed cost (no second model)")
    arr = entry_system(PAR, [0.0] * N, [0.0] * N, tau=TAU, R0=90.0, x0=0.9)
    f = composite_port_form(open_access(PAR, N))
    qe = jnp.asarray([37.0, 0.11, 0.23, 0.07] + [0.0] * (N + 1) + [1.0] * (N + 1))
    xi, _, _ = arr.omega(qe, jnp.zeros(0), EXTERIOR)
    row("entry forcing", np.round(np.asarray(xi)[:1 + N], 8))
    row("plain forcing", np.round(np.asarray(f(jnp.asarray([37.0, 0.11, 0.23, 0.07]))), 8))
    R0, _, k0, _ = settled([0.0] * N, [0.0] * N, steps=100_000)
    row("settled R", round(R0, 6), "R*(3)", round(PAR.R_open_access(3), 6))
    verdict("O2 deformation, not a second model",
            np.allclose(np.asarray(xi)[:1 + N],
                        np.asarray(f(jnp.asarray([37.0, 0.11, 0.23, 0.07]))), atol=1e-12))

    head("O1 -- does entry stop, and is the stopping point the zero-rent condition?")
    a_val = 5.0
    row("case", "R", "kappa", "x_i R", "a", "N_eff")
    R, x, k, q = settled([0.5] * 4, [a_val] * 4)
    row("4 identical, together", round(R, 5), round(k[0], 6), round(x[0] * R, 6), a_val,
        round(effective_N(q, 4, TAU), 5))
    print("\n  Entry DOES stop, at an INTERIOR placement, with x_i R = a to nine digits:")
    print("  §12b's 'placement settles where marginal rent equals the marginal cost of being")
    print("  close' is CONFIRMED. But it is stable only because the STOCK responds -- at")
    print("  fixed R the net benefit is convex in kappa and the same point is a repeller.")
    row("", "rent/p", "cost a kappa", "ratio", "V = net benefit")
    for power in (1.0, 3.0):
        R, x, k, _ = settled([0.5] * 4, [a_val] * 4, power=power)
        rent = PAR.p * k[0] ** 2 * R ** 2 / (2 * PAR.gamma)
        cost = a_val * k[0] ** power
        row(f"power = {power:g}", round(rent, 6), round(cost, 6), round(rent / cost, 6),
            round(net_benefit(PAR, R, k[0], a_val, power), 6))
    print("""
  IT IS NOT THE ZERO-RENT CONDITION. At the settled placement the rent collected is
  exactly (power/2) times the proximity cost paid: for the default linear cost, HALF of
  it, so every participant sits at V = -a kappa/2 < 0 -- strictly worse than never having
  come, and no local move gets it out. Gordon's condition is V = 0. The two coincide only
  at power = 2, where both are degenerate. Rent is quadratic in the insideness and the
  cost is linear; a gradient flow sees only the marginal balance.""")

    head("O1 -- the count that enters, and the failure criterion §12b names")
    row("a", "marginal count", "Gordon (zero-rent) count")
    for av in (1.0, 5.0, 8.0, 20.0):
        row(av, entry_count(PAR, av, 8, marginal_threshold),
            entry_count(PAR, av, 8, zero_rent_threshold))
    row("case", "kappas", "R", "N_eff")
    for tag, pl, out in (("4 identical, together", [0.5] * 4, [5.0] * 4),
                         ("4 identical, one far out", [0.5, 0.5, 0.5, 6.0], [5.0] * 4),
                         ("a = 1, 4, 8, 20", [0.5] * 4, [1.0, 4.0, 8.0, 20.0])):
        R, x, k, q = settled(pl, out)
        row(tag, np.round(k, 5), round(R, 5), round(effective_N(q, 4, TAU), 5))
    print("""
  §12b: "O1 fails visibly if every participant ends at the same placement." With IDENTICAL
  participants they do -- symmetry forces it -- but with different outside options they end
  at genuinely different placements, sorted by a_i, and the stock settles at the open-access
  value for the number that entered. Effective N rises and falls inside a fixed syntax.

  TWO THINGS THE NOTE DOES NOT ANTICIPATE:
   (i) the stopping point is NOT UNIQUE. The same parameters give N_eff = 3.24 when four
       identical participants start together and N_eff = 3 when one starts far out.
       Gordon's n* is a number; this model's is a set, selected by the initial placements.
  (ii) kappa = 0 is a fixed point with a VANISHING GRADIENT, so a participant placed far
       out never arrives. With any a > 0 the placement force points outward wherever
       x_i R < a, and far out x_i R -> 0. Entry is not spontaneous: it has to be seeded
       inside the kernel's pull region, whose radius is set by tau, which nothing fixes.""")
    row("l_0", "kappa_0", "l_final", "moved")
    R0 = 90.0
    for l0 in (2.5, 4.0, 6.0, 8.0):
        k0 = float(np.exp(-l0 ** 2 / (2 * TAU)))
        x0 = PAR.p * k0 * R0 / PAR.gamma
        tr = np.asarray(run(entry_system(PAR, [l0], [5.0], tau=TAU, R0=R0, x0=x0,
                                         place_mobility=PM), 40_000, every=20_000))
        row(l0, f"{k0:.3e}", round(float(tr[-1, 3]), 5),
            "IN" if tr[-1, 3] < tr[0, 3] else "OUT")
    print(f"\n  separatrix at x_i R = a, i.e. l_c = "
          f"{np.sqrt(-2 * TAU * np.log(5.0 * PAR.gamma / (PAR.p * R0 ** 2))):.4f}")


if __name__ == "__main__":
    main()
