# Convection loop: results as they landed (Claude Opus 5.5 (1M), 4c44c6b5, 2026-10-01)

Book question (David, 2026-10-01): is Ch 2's convection roll (`sec.driven`) worth it, and does it work? It was never
run (plan §5.6). Model: four heat-holding cells in a loop (bottom, right, top, left), each a tank of heat with
temperature as output; two zero-response reservoirs (hot under the bottom cell, cold over the top); one stateless
flow box receiving all six temperatures (since 2026-10-02, Claude Opus 5.5, 9dd117f1: split into local boxes, Heat's
joints, ring joints that also carry the flow, a buoyancy box reading left and right; the same equations, tested), flow Q = g (T_right - T_left) set at each moment by the temperature
difference (no inertia, David's correction), returning conduction D, plate exchange k, and heat carried by the flow.
Parameters D = 0.1, k = 1, g = 1, step s = 0.05; plates at +-Delta/2.

## Piece 1 (composite through Phi)
One and five Phi steps of the composite equal the hand Euler step T' = T + s F(T) to 0.0; the reservoirs do not move.
About 25 ms per Phi step, so long runs use the hand step (equal to Phi's exactly).

## Piece 2 (upwind heat carrying, max(Q,0) terms)
Onset exactly at the predicted Delta_c = 2D(k+2D)/(gk) = 0.24: no flow at 0.5, 0.9, 0.95 Delta_c; flow at 1.05 and up.
Heat through the loop / conduction alone: 1.04, 1.07, 1.72, 2.60, 3.40 at 1.05, 1.1, 2, 4, 8 Delta_c. Withdrawing the
difference: flow ~1e-88. Opposite stir: opposite roll. Flow grows linearly from onset, not as a square
root. First filed here as "an artifact ... replaced by centered carrying"; **that was wrong** (corrected 2026-10-02,
same run, after David asked whether the swap was a hack). Upwind carrying is exactly a well-mixed tank:
dT_i = Q (T_up - T_i), inflow at the upstream temperature, outflow at the tank's own. Algebraically upwind = centered
+ (|Q|/2)(T_up + T_down - 2 T_i): the stirring inside each tank is extra mixing proportional to the flow's speed. At
steady state it raises the pinned interior difference to g(T_bottom - T_top) = 2D + |Q|, and the |Q| term (quadratic in
the amplitude) beats the cubic saturation near onset, so |Q| grows linearly. So the linear onset is what a ring of
stirred tanks really does; the square root is a continuous fluid's (no stirred cells). Centered carrying does not
describe tanks (outflow at the average of two cells), so it does not match the tank picture.

## Piece 2c (centered heat carrying, Q (T_up - T_down)/2) -- the continuum-like comparison, not a tank model
Same onset. Flow Q = 0.110, 0.155, 0.490, 0.849, 1.296 at 1.05, 1.1, 2, 4, 8 Delta_c, i.e. Q^2 = Delta - Delta_c here;
heat ratio 1.24, 1.46, 3.50, 4.75, 5.38. Withdrawal kills it; opposite stir gives the mirror roll.
Closed form (derived by hand, then checked): above onset, T_bottom - T_top = 2D/g (pinned at its threshold value; all
extra drive goes into the flow), Q^2 = gk (Delta - Delta_c), heat flux J = k(Delta/2 - D/g) against conduction's
kD Delta/(k+2D); J exceeds conduction exactly when Delta > Delta_c.

## Piece 3 (2026-10-02): N stirred tanks around a loop (David: "if we had enough tanks, it would be like they were unstirred")
Script `notes/convection_many_tanks.py`. Setup changed to the continuum-friendly loop: N cells at angles 2 pi i/N, every
cell exchanging (rate k = 1) with a wall at (Delta/2) cos(angle), conduction kappa = 0.1 per unit angle, flow
Q = g (2/N) sum T_i sin(angle_i), g = 1; steady states by Newton (residual <= 1e-13). Continuum: Q^2 = (gk/2)(Delta -
Delta_c), Delta_c = 2(kappa+k)^2/(gk) = 2.42. Distance above each N's own onset eps = 0.01, 0.04, 0.16, 1.
- Onset / continuum onset: 1.52 (N=4), 1.10 (8), 1.024 (16), 1.0015 (64), 1.0001 (256).
- Stirred flow at eps = .01: 0.0084, 0.0144, 0.0267, 0.0686, 0.0973 against the continuum's 0.1100.
- Local exponent of flow against eps (0.5 is a square root, 1 a straight line), eps .01 to .04: 0.99, 0.97, 0.90, 0.67,
  0.54. David's intuition holds: the straight-line stretch shrinks as the tanks shrink, and the square root returns.
- Unstirred (centered) carrying gives the continuum's square root exactly at every N in this setup.
