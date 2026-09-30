# More than two oscillators: predictions, fixed before any code

David, 2026-09-29, on §1.5's Synchronization item, when the domain check proposed narrowing "the
oscillators would fall into step" to "two oscillators" because only two were run: "Are you saying it
doesn't work for more than two? If the test hasn't been run, just run it and then be willing to say
more than two." Claude Opus 5.5 (1M), session 16311c9d.

The runs use the library's own composite, `kuramoto_arrangement` (oscillator boxes, edge boxes, the
routing wire, composed; the forcing emerges from the wiring), stepped by `Phi`. Coupling is written per
edge; all-to-all uses `kappa / N` per edge, so the dynamics is the classical
`theta_i' = omega_i + (kappa / N) sum_j sin(theta_j - theta_i)`. Natural frequencies are evenly spaced
over `[-0.5, 0.5]`; starting angles are drawn at random (seed fixed). "In step" means every
oscillator advances at the same average rate over the last window.

1. **All-to-all, above threshold.** For N = 3, 5, 10 at kappa = 2 (well above the locking coupling
   for this spread, about 0.64 in the large-N limit and 1 for N = 2), from random angles, every
   oscillator ends up advancing at the mean rate: the spread of the average rates over the last
   window is below 1e-6. Fails if any N leaves a slipping oscillator.
2. **All-to-all, below threshold.** At kappa = 0.3 the rates stay apart: spread above 0.05 for each N.
3. **A ring can stay out of step.** Ten identical oscillators on a ring (each coupled to its two
   neighbours, K = 1), started in the twisted state `theta_i = 2 pi i / 10` and nudged by 0.01 at
   random, return to the twist: the phase differences stay within 0.02 of `2 pi / 10` and the
   oscillators never gather (order parameter below 0.05). This is why the book's sentence names
   all-to-all coupling rather than any coupling.
4. **With inertia and friction, many keep in step.** The swing arrangement with damping
   (`coalgebra.damped`, m = 20, c = 0.05), all-to-all N = 5 at kappa = 2 from rest: the average
   rates agree to 1e-6 over the last window.

## Results (same day, `tests/test_kuramoto_many.py`, 4 passed in 5 s)

All four held. (1) kappa = 2: the average rates agree exactly (spread 0.0 at float precision) for
N = 3, 5, 10, order parameter 0.978, 0.984, 0.987. (2) kappa = 0.3: rate spreads 0.944, 0.928, 0.966.
(3) The twisted ring is an exact fixed point of the discrete map and returns after a 0.01 nudge; the
order parameter stays below 0.05 throughout. (4) The damped swing, N = 5, locks from random angles.
**Reasonable**: all four are the textbook behaviour (Kuramoto's all-to-all locking; stable twisted
states on a ring). **Weird**: nothing new; frequencies evenly spaced rather than random, and N stops at
10, so "any number" is the classical theorem, not this run.
