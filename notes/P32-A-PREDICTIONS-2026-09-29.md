# P32 (a): light in the library -- predictions fixed before code

Claude Fable 5.1, session ec5ece87, 2026-09-29, 14:15 CDT. Capped run (about 180k tokens,
35 minutes). Source of the predictions: the book repo's
`notes/SPRINGS-TO-TOWNS-AUDIT-2026-09-29.md`, "The smallest test for each thin spot", item (a).
Oracle: `scripts/maxwell2d.py` (plain numpy, A on edges, B = curl A on faces, U = 1/2 sum B^2,
phase rule q' = q + dt xi; xi' = xi - dt dU(q')).

## The model as an arrangement

N x N periodic grid, N = 8. Each of the 2N^2 = 128 EDGES is a box holding one number A_e:
parameter `phase(diagonal([1.0]))` (unit mass), no in-ports, one out-port emitting A_e, zero
forcing. Each of the N^2 = 64 FACES is a stateless box (trivial parameter) with four in-ports and
cost k_f = (kappa/2)(n_0 + n_1 - n_2 - n_3)^2 = (kappa/2) B_f^2, forcing omega = -dk_f. A
routing wire copies each edge's out-port to the in-ports of the two faces it borders (a copy,
not a permutation). The composite is `compose_seq(tensor(edges + faces), wire)`, closed.

**Where the script's dt lives.** The library's step has no dt (STATUS.md section 1, "time is
discrete by design"). With unit mass the phase step is A' = A + xi, so the library momentum is
xi = dt * x_script, and the script's second half-step xi' = xi - dt^2 C^T C A' is the library's
forcing at kappa = dt^2. So: the script at step dt equals the composite with unit mass and face
constant kappa = dt^2, momenta rescaled by dt. Energy in the script's terms is
E = 1/2 sum (xi/dt)^2 + 1/2 sum B^2; the library's Hamiltonian is dt^2 E, same relative drift.

## The five predictions, each able to fail

1. **Forcing emerges.** At kappa = 1 the composite forcing's Q-component at a random A equals
   -grad of sum_f 1/2 B_f^2 computed by (i) my own `jax.grad` of 1/2 sum B^2 and (ii) the
   script's `gradU`, to atol 1e-10 (rounding; the values are O(1)). The face cost is written
   only inside the face box; the sum over faces and the transposed copy are produced by
   `compose_seq`'s pullback-and-sum. Fails if any component differs by more than 1e-10, or if
   the composite is not closed.
2. **One Phi step is the script's step.** From random (A, x) at dt = 0.5, one `Phi` step of the
   composite (kappa = 0.25, xi = 0.5 x) equals one script step (A + 0.5 x, then x - 0.5 gradU) to
   atol 1e-12 in A and in xi/dt. Fails otherwise. (I reimplement the script's `curl`/`gradU`
   on 8 x 8 by importing nothing from it: the script runs N = 200 at import, so its two
   functions are copied verbatim into the test and marked as such.)
3. **Gauss's law.** From random A and random momenta with NONZERO vertex divergence, run 60
   steps at dt = 0.5; the discrete divergence of the momenta at every vertex changes by less
   than 1e-12 (absolute; the momenta are O(1)). Starting from zero momenta would make this
   0 = 0, so the initial divergence is random. Fails if any vertex's divergence drifts.
4. **Dispersion.** For the eigenmode A = C^T cos(k . r) with k = (2 pi a/N, 2 pi b/N), the
   step at dt = 0.05 rotates the modal amplitude by an angle theta per step; check
   (i) the discrete relation (2/dt)^2 sin^2(theta/2) = Omega^2 := 4 sin^2(kx/2) + 4 sin^2(ky/2)
   to rtol 1e-9 (a discrete analogue; exact for a linear eigenmode), (ii) the closed form
   omega = theta/dt agrees with Omega to rtol 1e-3 (the symplectic-Euler error is about
   (dt Omega)^2/24 <= 5e-4 at the modes checked; the closed form is the small-step limit), at
   k = (1, 0), (1, 1) and (2, 1) in units of 2 pi/8, and (iii) the phase speed Omega/|k|
   DECREASES from k = (1,0) through (2,0) to (4,0): short waves run slow. Fails if any of
   (i)-(iii) fails, or if the cosine mode is not an eigenmode (amplitude not cos(n theta)
   to 1e-9 over 40 steps).
5. **Energy at dt = 0.5.** From the script's kind of initial data (A = C^T psi for a Gaussian
   psi, zero momenta, plus a second run from random data), 400 steps at dt = 0.5 (time 200):
   the relative deviation |E(t) - E(0)|/E(0) stays below 3 percent at every step. I will
   report the largest drift seen. Fails above 3 percent. (CFL for unit spacing in 2D is
   dt <= 1/sqrt 2 = 0.707; at 0.5 the symplectic step is stable and its energy oscillates
   without secular drift.)

## What could have come out otherwise

Guaranteed by construction, given the library's `compose_seq` is right: (1) is the
pullback-and-sum doing what its docstring says on a copy wire (a new case: the graph wire is a
permutation; this wire is not); (3) follows from (1), since div C^T = 0 identically. Could have
gone otherwise: (2) is a match of two independently written implementations (the script's
update is not derived from the composite); (4) is a closed form the composite is checked
against, and the eigenmode could have failed to be one if the routing put a sign wrong;
(5) is a numerical fact about the step at dt = 0.5, not implied by the construction (at dt = 1.5
it would blow up).

## The domain objection (Yee / FDTD)

Someone who knows computational electromagnetics would say: "This is Yee's TE grid in the
temporal gauge, A on edges and B through faces, leapfrogged in time; the two Maxwell equations
it 'proves' (div B = 0, Faraday) hold by construction because B is defined as dA, as in every
discrete-exterior-calculus scheme, and Gauss's law is conserved rather than enforced, also as in
Yee. Fine. But prediction (4) is not a property of light: Omega^2 = 4 sin^2(kx/2) + 4 sin^2(ky/2)
is the lattice's numerical dispersion, anisotropic and slow at short wavelengths, and vacuum light
is dispersionless and isotropic. 'Short waves run slow' is an artefact of the grid, and a book
that says it of light is wrong." The model survives as lattice Maxwell: the conserved
quantities, the wave speed at long wavelength (Omega -> |k|), and the energy are right. The
sentence "short waves run slow" does not survive as a statement about light in vacuum; it is
true of the lattice, of a spring chain, and of light in a periodic medium, and the book may
say it only of those. Stability at dt = 0.5 also needs the CFL bound (dt < 1/sqrt 2), which the
model meets; the FDTD person would ask that it be stated.

## Progress
