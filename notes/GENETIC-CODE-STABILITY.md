# Does the standard genetic code stay put? (predictions fixed before any code)

Claude Opus 5.5 (1M), session 2141582f, 2026-09-28. David's question, for the book's §1.5
inventory entry on the genetic code: "starting near there, you don't descend to something else."

## Model (the book's Ch 4.6 design, plan §4.5; weights from memory, to be checked against sources)

- State: for each of the 61 sense codons c, odds p(·|c) over the 20 amino acids (a product of
  simplexes). Integrator: rowwise entropic mirror descent (log-weights, renormalized).
- Slip cost R: over ordered pairs of sense codons c, c' one base apart, weight w(c,c') times
  E[(x_a − x_a')²], a ~ p(·|c), a' ~ p(·|c'), with x the polar requirement (Woese; values from
  memory). w: Freeland–Hurst-style position/type weights from memory: position 1 transition 1,
  transversion 0.5; position 2 transition 0.5, transversion 0.1; position 3 both 1.
- Repertoire cost, weight γ: −Σ_a log u_a, u_a the mean over codons of p(a|c).
- Definiteness cost, weight β: Σ_c H(p(·|c)), entropy of each codon's odds.
- Start: the standard code softened, own amino acid 1 − ε, the rest ε/19, ε = 0.02.
- Comparison: codes made by Freeland–Hurst's shuffle (permute which amino acid each of the
  standard code's 20 synonymous blocks gets), same weights, same start.

## Predictions

1. With β > 0, every hard code is a local rest (entropy has infinite slope at the simplex's
   boundary), so the standard code stays, and so does every shuffled code, for starts close
   enough. Staying then says nothing about the standard code in particular.
2. With β = 0 and γ = 0, the standard code slides: codons move toward amino acids like their
   neighbours', and the code loses amino acids (Met, Trp first).
3. With β = 0 and moderate γ, a few codons reassign (fewer than 10 of 61), Met and Trp kept.
4. At the same weights, the standard code reassigns fewer codons than most shuffled codes
   (the Freeland–Hurst direction), but is not a strict rest for every γ.

## Verdicts (after `scripts/genetic_code_stability.py`, plain numpy, seconds per run)

Descent with γ = 3, ε = 0.02, 8000 steps; "moved" counts codons whose most likely amino acid
changed; shuffles are 20 Freeland–Hurst block shuffles (10 for the ε checks).

| β | standard code moved | shuffles moved, median (min) |
|---|---|---|
| 0.5 | 13 (loses Asp, Glu) | 18 (10) |
| 1 | 2 (loses Asp) | 13 (8) |
| 2 | 0 | 7 (0) |
| 4 | 0 | 2 (0) |

The same pattern at ε = 0.05 and 0.1 (β = 2: standard 2, shuffles median 8–9; β = 4: 0 vs 3–4).
Linearized at the hard code with β = 0 and γ = 0, 36 of 61 standard codons would rather be
another amino acid, against a median of 55 for shuffles, and none of 2000 shuffles has as few.

1. **Refuted as stated.** Not every hard code stays for a close start: at these weights the
   definiteness term holds the standard code from β ≈ 2, while most shuffled codes still slide
   there; only at β = 4 do some shuffles stay. So staying *does* distinguish the standard code.
2. **Held.** With neither definiteness nor repertoire, the code collapses (57 codons move, 19
   amino acids lost).
3. **Refuted.** Repertoire alone does not hold it: 33–40 codons move and Asp, Glu, Phe are lost;
   the repertoire gradient pulls every codon toward the rarest amino acids (Met, Trp).
4. **Held.** The standard code moves fewer codons than the median shuffle at every weight tried,
   and is not a rest for β ≤ 1.

**Reasonable** (Claude's judgment): the standard code resists sliding better than codes with
its block structure but other assignments, the Freeland–Hurst direction. **Weird**: Asp goes
first, because its polar requirement (13.0) is the extreme and the slip cost counts any
difference as damage, while a cell needs Asp regardless; that the answer hangs on β, a weight
with no measured value; the polar requirements and slip weights are from memory, unchecked;
and the descent is plain numpy, not the library's integrators, with 20 shuffles only.
