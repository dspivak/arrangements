"""Does the standard genetic code stay put under descent? (notes/GENETIC-CODE-STABILITY.md)

Plain numpy, rowwise entropic mirror descent; not the library's integrators.
Polar requirements and slip weights are from memory (Woese; Freeland-Hurst), to be checked.
Claude Opus 5.5 (1M), 2141582f, 2026-09-28.
"""
import itertools, sys
import numpy as np

B = "TCAG"
TABLE = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
AAS = "ARNDCQEGHILKMFPSTWYV"
PR = dict(A=7.0, R=9.1, N=10.0, D=13.0, C=4.8, Q=8.6, E=12.5, G=7.9, H=8.4, I=4.9,
          L=4.9, K=10.1, M=5.3, F=5.0, P=6.6, S=7.5, T=6.6, W=5.2, Y=5.4, V=5.6)
codons = ["".join(t) for t in itertools.product(B, repeat=3)]
std = {c: a for c, a in zip(codons, TABLE)}
sense = [c for c in codons if std[c] != "*"]
N, K = len(sense), 20
idx = {c: i for i, c in enumerate(sense)}
x = np.array([PR[a] for a in AAS])
D = (x[:, None] - x[None, :]) ** 2
purine = set("AG")

def weight(pos, b1, b2):
    ts = (b1 in purine) == (b2 in purine)  # transition
    return [(1.0, 0.5), (0.5, 0.1), (1.0, 1.0)][pos][0 if ts else 1]

W = np.zeros((N, N))
for c in sense:
    for pos in range(3):
        for b in B:
            if b == c[pos]:
                continue
            c2 = c[:pos] + b + c[pos + 1:]
            if c2 in idx:
                W[idx[c], idx[c2]] = weight(pos, c[pos], b)
W /= W.sum() / N  # per-codon scale ~ one neighbour-average

def code_matrix(assign):
    P = np.zeros((N, K))
    for c in sense:
        P[idx[c], AAS.index(assign[c])] = 1
    return P

def grad(P, beta, gamma):
    g = 2 * W @ P @ D
    if gamma:
        g += -gamma / (N * P.mean(0))[None, :]
    if beta:
        g += -beta * (np.log(P + 1e-300) + 1)
    return g

def run(assign, beta, gamma, eps=0.02, eta=0.02, steps=20000):
    P0 = code_matrix(assign)
    P = (1 - eps) * P0 + eps / (K - 1) * (1 - P0)
    th = np.log(P)
    for t in range(steps):
        g = grad(P, beta, gamma)
        th -= eta * g
        th -= th.max(1, keepdims=True)
        P = np.exp(th); P /= P.sum(1, keepdims=True)
    moved = int((P.argmax(1) != P0.argmax(1)).sum())
    lost = [AAS[a] for a in range(K) if P.argmax(1).tolist().count(a) == 0]
    return moved, P.max(1).min(), lost, P

def best_response_unstable(assign, gamma):
    """At the hard code (beta=0): codons whose own amino acid is not the argmin of the gradient."""
    P = code_matrix(assign)
    g = 2 * W @ P @ D - gamma / (N * np.maximum(P.mean(0), 1e-12))[None, :]
    own = P.argmax(1)
    return [sense[i] for i in range(N) if g[i, own[i]] > g[i].min() + 1e-9]  # ties count as stable

def shuffled(rng):
    perm = rng.permutation(K)
    m = {AAS[i]: AAS[perm[i]] for i in range(K)}
    return {c: m[std[c]] for c in sense}

if __name__ == "__main__":
    rng = np.random.default_rng(0)
    std_s = {c: std[c] for c in sense}
    mode = sys.argv[1] if len(sys.argv) > 1 else "all"
    print("best-response unstable codons at the hard standard code, beta=0:")
    for gamma in [0, 1, 3, 10, 30, 100]:
        u = best_response_unstable(std_s, gamma)
        print(f"  gamma={gamma:>4}: {len(u):2d} {' '.join(c+'('+std[c]+')' for c in u)}")
    print("shuffled codes, count unstable at the hard code:")
    for gamma in [0, 3, 10, 30]:
        counts = [len(best_response_unstable(shuffled(rng), gamma)) for _ in range(2000)]
        s = len(best_response_unstable(std_s, gamma))
        frac = np.mean(np.array(counts) <= s)
        print(f"  gamma={gamma:>4}: standard {s}, shuffled median {int(np.median(counts))}, "
              f"fraction of shuffles with <= standard's count {frac:.4f}")
    if mode == "grid":
        for beta in [0.5, 1, 2, 4]:
            for gamma in [0, 3]:
                m, mn, lost, _ = run(std_s, beta, gamma, steps=8000)
                sh = [run(shuffled(rng), beta, gamma, steps=8000)[0] for _ in range(20)]
                print(f"beta={beta} gamma={gamma}: standard moved {m} lost {''.join(lost)}; "
                      f"20 shuffles moved median {int(np.median(sh))}, min {min(sh)}", flush=True)
    if mode == "all":
        for beta, gamma in [(0, 0), (0, 3), (0, 10), (0.5, 0), (0.5, 3)]:
            m, mn, lost, _ = run(std_s, beta, gamma)
            print(f"descent beta={beta} gamma={gamma}: moved {m}, min max-prob {mn:.3f}, lost {lost}")
