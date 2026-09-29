"""The genetic-code descent rerun with SOURCED inputs (notes/P32-D-SOURCES-2026-09-29.md).

P32 stage (d) piece 1. `scripts/genetic_code_stability.py` says at its head that its polar
requirements and slip weights were written FROM MEMORY. This script does not edit it: it imports
it, rebuilds the two inputs from the sources below, ASSERTS that the rebuilt inputs agree with the
from-memory ones, and reruns. Every other line of the model -- the cost, the mirror descent, the
shuffle -- is the original's, reached by import.

INPUT 1, polar requirement, Woese et al. 1966. Read from Buhrman, van der Gulik, Kelk, Koolen,
Stougie, "Some mathematical refinements concerning error minimization in the genetic code",
arXiv:0909.1442v2, TABLE I (p. 3), which prints the value beside each codon and attributes them
(p. 2) to its reference [2], Woese, Dugre, Dugre, Kondo, Saxinger, Cold Spring Harbor Symp. Quant.
Biol. 31, 723-736 (1966). The 1966 primary could not be read here and is not cited as read.

INPUT 2, mistranslation weights. Read from Buhrman, van der Gulik, Klau, Schaffner, Speijer,
Stougie, "A Realistic Model under which the Genetic Code is Optimal", arXiv:1309.4589v1, section 2
"Methods" (p. 8), quoting "the following set of weights introduced by Freeland and Hurst [21]":
alpha = 0.5 for a transversion in the first position or a transition in the second; 0.1 for a
transversion in the second; 1 otherwise. [21] is Freeland and Hurst, J. Mol. Evol. 47, 238-248
(1998), which is paywalled and was not read.

SENSITIVITY ONLY, the recomputed scale of Mathew and Luthey-Schulten (2008), read from
arXiv:1309.4589v1 Table 1 (p. 3). It is a different measurement, not a correction to the input
Freeland and Hurst used, so it is run beside the 1966 scale and never in place of it.

PREDICTION, fixed before running (notes/P32-D-SOURCES-2026-09-29.md): the ordering survives -- the
standard code moves fewer codons than the median block shuffle at every beta tried -- and the beta
threshold at which the standard code stays may move.

Claude Fable 5.1, session ec5ece87, 2026-09-29.
"""
import importlib.util
import itertools
import pathlib
import sys

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gcs", _HERE / "genetic_code_stability.py")
gcs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(gcs)          # the original, unedited, its __main__ not run

AAS, B, N, K = gcs.AAS, gcs.B, gcs.N, gcs.K
sense, idx, std = gcs.sense, gcs.idx, gcs.std

# --- input 1, rebuilt from arXiv:0909.1442 TABLE I -------------------------------------------
PR_1966 = dict(A=7.0, R=9.1, N=10.0, D=13.0, C=4.8, Q=8.6, E=12.5, G=7.9, H=8.4, I=4.9,
               L=4.9, K=10.1, M=5.3, F=5.0, P=6.6, S=7.5, T=6.6, W=5.2, Y=5.4, V=5.6)
# --- the recomputed scale, arXiv:1309.4589 Table 1 (Mathew & Luthey-Schulten 2008) ------------
PR_2008 = dict(F=4.5, L=4.4, I=5.0, M=5.0, V=6.2, S=7.5, P=6.1, T=6.2, A=6.5, Y=7.7,
               H=7.9, Q=8.9, N=9.6, K=10.2, D=12.2, E=13.6, C=4.3, W=4.9, R=8.6, G=9.0)

PURINE = set("AG")


def alpha(pos, b1, b2):
    """Freeland-Hurst weights exactly as arXiv:1309.4589 section 2 states them.

    Written from the source's own three clauses rather than from the original script's table, so
    that the assert below is a real comparison and not a copy checking itself.
    """
    transition = (b1 in PURINE) == (b2 in PURINE)
    if pos == 0 and not transition:
        return 0.5
    if pos == 1 and transition:
        return 0.5
    if pos == 1 and not transition:
        return 0.1
    return 1.0


def slip_weights():
    W = np.zeros((N, N))
    for c in sense:
        for pos in range(3):
            for b in B:
                if b == c[pos]:
                    continue
                c2 = c[:pos] + b + c[pos + 1:]
                if c2 in idx:
                    W[idx[c], idx[c2]] = alpha(pos, c[pos], b)
    return W / (W.sum() / N)


def distances(pr):
    x = np.array([pr[a] for a in AAS])
    return (x[:, None] - x[None, :]) ** 2


W_SRC, D_1966, D_2008 = slip_weights(), distances(PR_1966), distances(PR_2008)


def check_against_memory():
    dw = np.abs(W_SRC - gcs.W).max()
    dd = np.abs(D_1966 - gcs.D).max()
    print(f"sourced vs from-memory: max |dW| = {dw:.3e}, max |dD| = {dd:.3e}")
    assert dw == 0.0 and dd == 0.0, "sourced inputs differ from the from-memory ones"
    print("  -> the two inputs are IDENTICAL; the 1966 rerun is a reproduction, not a new number.")


def use(D):
    gcs.D = D
    gcs.W = W_SRC


def grid(label, D, betas, gammas=(0, 3), nshuf=20, steps=8000, seed=0):
    use(D)
    rng = np.random.default_rng(seed)
    std_s = {c: std[c] for c in sense}
    print(f"\n### {label}: descent, {nshuf} block shuffles, {steps} steps, eps=0.02")
    print("beta gamma | standard moved (lost) | shuffles moved median (min)")
    for beta in betas:
        for gamma in gammas:
            m, _, lost, _ = gcs.run(std_s, beta, gamma, steps=steps)
            sh = [gcs.run(gcs.shuffled(rng), beta, gamma, steps=steps)[0] for _ in range(nshuf)]
            print(f"{beta:>4} {gamma:>5} | {m:>3} ({''.join(lost) or '-'})"
                  f"{'':>12} | {int(np.median(sh)):>3} ({min(sh)})"
                  f"   standard below median: {m < np.median(sh)}", flush=True)


def hard_code_linearization(label, D, gammas=(0, 3, 10, 30), nshuf=2000, seed=0):
    use(D)
    rng = np.random.default_rng(seed)
    std_s = {c: std[c] for c in sense}
    print(f"\n### {label}: best-response unstable codons at the HARD code, beta=0")
    for gamma in gammas:
        counts = [len(gcs.best_response_unstable(gcs.shuffled(rng), gamma)) for _ in range(nshuf)]
        s = len(gcs.best_response_unstable(std_s, gamma))
        frac = float(np.mean(np.array(counts) <= s))
        print(f"  gamma={gamma:>3}: standard {s:>2}, {nshuf} shuffles median "
              f"{int(np.median(counts))}, fraction with <= standard {frac:.4f}", flush=True)


if __name__ == "__main__":
    check_against_memory()
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if what in ("all", "hard"):
        hard_code_linearization("1966 scale (sourced)", D_1966)
    if what in ("all", "grid"):
        grid("1966 scale (sourced)", D_1966, betas=(0.5, 1, 2, 4))
    if what in ("all", "thresh"):
        grid("1966 scale, beta threshold scan (gamma=3)", D_1966,
             betas=(1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0), gammas=(3,))
    if what in ("all", "y2008"):
        hard_code_linearization("2008 recomputed scale (sensitivity)", D_2008)
        grid("2008 recomputed scale (sensitivity)", D_2008, betas=(0.5, 1, 2, 4))
        grid("2008 scale, beta threshold scan (gamma=3)", D_2008,
             betas=(1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0), gammas=(3,))
