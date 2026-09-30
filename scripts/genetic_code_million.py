"""Is the code cells use one in a million? A million block shuffles, on two measures.

David, 2026-09-29, on §1.5's genetic-code item ("less than 1% of the 2,000 shuffles we tried do as
well"): "Somehow this does not seem like the 99.9999 percentile thing"; then "Sure, let's do that" to
running a million. Two thousand shuffles can show at best about one in two thousand; this script
runs a million, reusing `scripts/genetic_code_stability_sourced.py`'s sourced inputs (polar
requirement after Woese et al. 1966, read from Buhrman et al. arXiv:0909.1442v2 Table I p. 3; slip
weights after Freeland and Hurst 1998, read from Buhrman et al. arXiv:1309.4589v1 §2 p. 8) and
`scripts/genetic_code_stability.py`'s shuffle (the 20 amino acids permuted over the standard code's
blocks, stop codons fixed).

TWO MEASURES, both at the hard code (every codon assigned to one amino acid):
(a) THE SLIP COST ITSELF, sum over codon pairs one letter apart of weight times squared difference
    in polar requirement between the amino acids they are assigned to. This is the model's slip
    cost at a hard code, and it has the form of Freeland and Hurst's weighted error measure.
(b) THE COUNT the book's item already uses (`best_response_unstable`, gamma = 3): how many codons
    would rather be assigned to another amino acid under the slip and repertoire costs.
For each: the fraction of shuffles doing at least as well as the real code (cost <=, count <=).

PREDICTIONS, fixed before the run (Claude Opus 5.5 (1M), session 16311c9d, 2026-09-29):
(1) (a) on the 1966 scale: the fraction is between 1e-6 and 1e-4 (Freeland and Hurst's published
    "one in a million" is from memory, their paper paywalled; it fails if the fraction is above 1e-4).
(2) (a) on the 2008 scale: the fraction is larger than on the 1966 scale and below 1e-2.
(3) (b) on the 1966 scale: below 1e-3 (0 of 2,000 before); on the 2008 scale: between 1e-3 and 1e-2
    (2 to 10 of 2,000 before).
Results are printed per 100,000 shuffles and appended to notes/GENETIC-CODE-MILLION-2026-09-29.md.
"""
import importlib.util
import pathlib
import sys
import time

import numpy as np

_HERE = pathlib.Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location("gcs_src", _HERE / "genetic_code_stability_sourced.py")
src = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(src)            # its __main__ not run
gcs = src.gcs
AAS, N, K, sense, idx, std = gcs.AAS, gcs.N, gcs.K, gcs.sense, gcs.idx, gcs.std
OUT = _HERE.parent / "notes" / "GENETIC-CODE-MILLION-2026-09-29.md"

a0 = np.array([AAS.index(std[c]) for c in sense])          # each sense codon's amino acid, standard
BLOCK = np.zeros((N, K))
BLOCK[np.arange(N), a0] = 1                                   # codon -> standard block
SIZE = BLOCK.sum(0)                                           # codons per standard block


def measures(perms, W, D, gamma=3.0):
    """For a batch of permutations (B x K; perm[b] = the amino acid block b is assigned to):
    (a) slip cost at the hard code, (b) best-response-unstable count, as in the scripts."""
    Wb = W @ BLOCK                                            # N x K: weight from codon c into block b
    M = BLOCK.T @ W @ BLOCK                                   # K x K: weight between blocks
    Dp = D[perms[:, :, None], perms[:, None, :]]              # B x K x K: D between assigned acids
    cost = np.einsum("ab,nab->n", M, Dp)
    # gradient at the hard code: g[c, x] = 2 sum_b Wb[c, b] D[x, perm b] - gamma / (N u_x)
    Dx = D[:, perms].transpose(1, 0, 2)                       # B x K(x) x K(b): D[x, perm b]
    g = 2 * np.einsum("cb,nxb->ncx", Wb, Dx)
    u = np.zeros((perms.shape[0], K))
    np.put_along_axis(u, perms, SIZE[None, :] / N, axis=1)    # u_x = share of codons on acid x
    g = g - gamma / (N * np.maximum(u, 1e-12))[:, None, :]
    own = perms[:, a0]                                        # B x N: each codon's assigned acid
    g_own = np.take_along_axis(g, own[:, :, None], axis=2)[:, :, 0]
    count = (g_own > g.min(2) + 1e-9).sum(1)
    return cost, count


def check(W, D):
    """The batched measures agree with the scripts' own functions on the standard code and 20 shuffles."""
    rng = np.random.default_rng(7)
    perms = np.stack([np.arange(K)] + [rng.permutation(K) for _ in range(20)])
    cost, count = measures(perms, W, D)
    for p, cst, cnt in zip(perms, cost, count):
        assign = {c: AAS[p[AAS.index(std[c])]] for c in sense}
        P = gcs.code_matrix(assign)
        assert abs(cst - np.sum(W * (P @ D @ P.T))) < 1e-9 * max(1, cst), (cst,)
        assert cnt == len(gcs.best_response_unstable(assign, 3.0)), (cnt,)


def run(label, D, total=1_000_000, chunk=25_000, report=100_000, seed=0):
    src.use(D)
    W = gcs.W
    check(W, D)
    ident = np.arange(K)[None, :]
    c0, n0 = (x[0] for x in measures(ident, W, D))
    rng = np.random.default_rng(seed)
    le_cost = le_count = done = 0
    t = time.time()
    head = f"\n### {label}: standard code slip cost {c0:.4f}, count {n0}"
    print(head, flush=True)
    with open(OUT, "a") as f:
        f.write(head + "\n")
    while done < total:
        perms = np.argsort(rng.random((chunk, K)), axis=1)
        cost, count = measures(perms, W, D)
        le_cost += int((cost <= c0 + 1e-12).sum())
        le_count += int((count <= n0).sum())
        done += chunk
        if done % report == 0 or done >= total:
            line = (f"- {done:>9,} shuffles: slip cost <= standard {le_cost} ({le_cost/done:.2e}); "
                    f"count <= standard {le_count} ({le_count/done:.2e}); {time.time()-t:.0f} s")
            print(line, flush=True)
            with open(OUT, "a") as f:
                f.write(line + "\n")
    return le_cost, le_count, done


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "1966"
    total = int(sys.argv[2]) if len(sys.argv) > 2 else 1_000_000
    D = src.D_1966 if which == "1966" else src.D_2008
    run(f"{which} scale", D, total=total)
