"""The genetic code's descent as an arrangement (notes/P32-D-PREDICTIONS-2026-09-29.md).

P32 stage (d) piece 2. `scripts/genetic_code_stability.py` runs the model as plain numpy rowwise
entropic mirror descent on one explicit gradient, `2 W P D` plus a repertoire and a definiteness
term. Here the same dynamics is an arrangement, and the point of doing it again is that
**nothing sums over a codon's neighbours**: what is written is one weight per wired pair and one
20x20 amino-acid distance matrix, and the total force on a codon is produced by the wiring's
pullback-and-sum.

THE BOXES.

* `codon_box(dt)` -- base `Q = R^K`, the log-odds `y = log p(.|c)` over the K amino acids;
  response `replicator.log_shahshahani(K, dt)`, the Shahshahani sharp of the K-simplex pushed
  along `y = log x`; integrator `coalgebra.configuration`. Its forward port carries the odds
  `x = exp(y)`. **Its forcing is zero**: a codon box has no cost of its own, so every force in
  the composite arrived through the wiring.
* `slip_box(w, D)` -- stateless, two ports of K, cost `k = 2 w (x)^T D (x')`, forcing `-dk`. One
  per UNORDERED pair of sense codons one base apart; the `2` is what makes the pair count once
  in each direction, as the script's symmetric `W` does.
* `definiteness_box(beta)` -- stateless, one port, cost `beta * H(x)`, `H` the entropy.
* `repertoire_box(gamma, n_codons)` -- stateless, `n_codons` ports, cost
  `-gamma * sum_a log u_a` with `u_a` the mean of `x^c_a` over codons.

THE WIRING is `fanout`, a static arrangement with trivial parameter and ZERO forcing whose
forward map gathers each cost box's ports out of the codon boxes' outputs. A codon feeding nine
slip boxes, its definiteness box and the repertoire box is copied eleven times by that gather,
and the eleven covectors come back summed because `wiring.compose_seq` pulls a forcing back with
`jax.vjp` and the adjoint of a gather is a scatter-add. That is the library's
"Kirchhoff as the transpose of copying" (README) doing the work a hand-written `W @ P @ D` does
in the script.

WHAT ONE STEP IS. With `omega = -dk` the composite's forcing in the odds chart is `-g`, `g` the
script's gradient; pulled back along `x = exp(y)` it is `x * (-g)`; and the log-chart sharp
`dt (diag(1/x) - 1 1^T)` sends that to `-dt g + dt <x, g>`. So

    x'_i = x_i exp(-dt g_i) * exp(dt <x, g>),

which is the script's `p_i exp(-eta g_i)` times ONE SCALAR PER CODON. The scalar is the log
chart's documented O(dt^2) departure from `sum x = 1` (`replicator.py`, "LOG COORDINATES"); it is
uniform in the amino acid, hence pure gauge on the simplex, so renormalizing each row recovers
the script's step exactly. `phi_step` does the renormalization and says so.

Claude Fable 5.1, session ec5ece87, 2026-09-29.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List, Sequence, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .replicator import log_shahshahani
from .wiring import trivial_parameter

# ---------------------------------------------------------------------------
# The standard code and its sourced inputs (notes/P32-D-SOURCES-2026-09-29.md).
# ---------------------------------------------------------------------------

BASES = "TCAG"
#: NCBI translation table 1, codons in `BASES` product order.
TABLE = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
AAS = "ARNDCQEGHILKMFPSTWYV"
#: Polar requirement, Woese et al. 1966, read from arXiv:0909.1442v2 TABLE I (p. 3).
POLAR_REQUIREMENT = dict(A=7.0, R=9.1, N=10.0, D=13.0, C=4.8, Q=8.6, E=12.5, G=7.9, H=8.4,
                         I=4.9, L=4.9, K=10.1, M=5.3, F=5.0, P=6.6, S=7.5, T=6.6, W=5.2,
                         Y=5.4, V=5.6)
PURINE = set("AG")


def fh_weight(pos: int, b1: str, b2: str) -> float:
    """Freeland-Hurst mistranslation weight, arXiv:1309.4589v1 section 2 (p. 8), quoting
    Freeland and Hurst, J. Mol. Evol. 47, 238-248 (1998): 0.5 for a transversion in the first
    position or a transition in the second, 0.1 for a transversion in the second, 1 otherwise."""
    transition = (b1 in PURINE) == (b2 in PURINE)
    if pos == 0 and not transition:
        return 0.5
    if pos == 1:
        return 0.5 if transition else 0.1
    return 1.0


def standard_code() -> Tuple[List[str], Dict[str, str]]:
    """The 61 sense codons and the standard assignment, `BASES` product order."""
    codons = ["".join(t) for t in itertools.product(BASES, repeat=3)]
    std = dict(zip(codons, TABLE))
    return [c for c in codons if std[c] != "*"], std


def neighbour_pairs(sense: Sequence[str], weight=fh_weight) -> List[Tuple[int, int, float]]:
    """Unordered pairs of sense codons one base apart, with their weight.

    The ONLY place the adjacency of the code is written. No matrix is formed: the caller gets
    263 triples, one per wired pair, and the sum over a codon's neighbours is left to the wiring.
    """
    idx = {c: i for i, c in enumerate(sense)}
    out = []
    for c in sense:
        for pos in range(3):
            for b in BASES:
                if b == c[pos]:
                    continue
                c2 = c[:pos] + b + c[pos + 1:]
                if c2 in idx and idx[c] < idx[c2]:
                    out.append((idx[c], idx[c2], weight(pos, c[pos], b)))
    return out


def scaled_pairs(pairs: Sequence[Tuple[int, int, float]], n_codons: int):
    """The script's per-codon normalization `W /= W.sum() / N`, applied pairwise.

    `W.sum()` over the script's symmetric matrix is twice the sum of the unordered weights.
    """
    total = 2.0 * sum(w for _, _, w in pairs)
    s = total / n_codons
    return [(i, j, w / s) for i, j, w in pairs]


def aa_distance(pr: Dict[str, float] = None, aas: str = AAS) -> Array:
    """`D_{a,a'} = (PR_a - PR_{a'})^2`, the slip box's only knowledge of amino acids."""
    pr = POLAR_REQUIREMENT if pr is None else pr
    x = jnp.array([pr[a] for a in aas])
    return (x[:, None] - x[None, :]) ** 2


# ---------------------------------------------------------------------------
# The boxes.
# ---------------------------------------------------------------------------


def codon_box(K: int, dt: float, y0: Array, label: str = "codon") -> Arrangement:
    """`<0|0> -> <0|R^K>`: state the log-odds, port the odds, forcing ZERO."""
    return Arrangement(
        P=configuration(log_shahshahani(K, dt), init=jnp.asarray(y0, dtype=float)),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=K,
        out_f=lambda y, m_out: jnp.exp(y),
        in_f=lambda y, m_out, n_in: jnp.zeros(0),
        omega=zero_form, label=label,
    )


def _cost_box(port_dim: int, k: Callable[[Array], Array], label: str) -> Arrangement:
    """A stateless `<0|R^port_dim> -> <0|0>` with forcing `omega = -dk` (STATUS.md section 1)."""
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=port_dim, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m_out: jnp.zeros(0),
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=exact(lambda q, m_out, n_in: -k(m_out)), label=label,
    )


def slip_box(w: float, D: Array, label: str = "slip") -> Arrangement:
    """Two codons one base apart: `k = 2 w (x)^T D (x')`. Sees neither codon's identity."""
    K = D.shape[0]
    return _cost_box(2 * K, lambda e: 2.0 * w * (e[:K] @ (D @ e[K:])), label)


def definiteness_box(beta: float, K: int, label: str = "definiteness") -> Arrangement:
    """`k = beta * H(x)`, the entropy of one codon's odds. Minimizing it makes the codon definite."""
    return _cost_box(K, lambda x: beta * -jnp.sum(x * jnp.log(x + 1e-300)), label)


def repertoire_box(gamma: float, n_codons: int, K: int, label: str = "repertoire") -> Arrangement:
    """`k = -gamma sum_a log u_a`, `u_a` the mean over codons of `x^c_a`."""
    def k(flat: Array) -> Array:
        u = flat.reshape(n_codons, K).mean(0)
        return -gamma * jnp.sum(jnp.log(u + 1e-300))
    return _cost_box(n_codons * K, k, label)


# ---------------------------------------------------------------------------
# The wiring: a fan-out whose adjoint is the scatter-add.
# ---------------------------------------------------------------------------


def fanout(source_dim: int, gather: Sequence[int], label: str = "fanout") -> Arrangement:
    """`<0|R^source_dim> -> <0|R^len(gather)>`, forward `m_out[gather]`, forcing ZERO.

    A static wiring. It copies; `compose_seq`'s `jax.vjp` pullback supplies the transpose, which
    for a gather is the scatter-add, and that sum is the only place a codon's several forces meet.
    """
    g = jnp.asarray(list(gather), dtype=int)
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=source_dim, in_dim_N=0, out_dim_N=len(gather),
        out_f=lambda q, m_out: m_out[g],
        in_f=lambda q, m_out, n_in: jnp.zeros(0),
        omega=zero_form, label=label,
    )


# ---------------------------------------------------------------------------
# The n-ary monoidal product, flat (prediction 6).
# ---------------------------------------------------------------------------


def tensor_all(parts: Sequence[Arrangement], label: str = "tensor_all") -> Arrangement:
    """The direct sum of `parts`, computed in one pass instead of by folding
    `wiring.parallel_arrangements`. Same object; the fold nests the state as deep as there are
    boxes, and there are 386 here. Checked against the library's fold in the tests."""
    n = len(parts)
    qd = [p.P.Q_dim for p in parts]
    mo, mi = [p.out_dim_M for p in parts], [p.in_dim_M for p in parts]
    no, ni = [p.out_dim_N for p in parts], [p.in_dim_N for p in parts]
    cq, cmo, cni = np.cumsum([0] + qd), np.cumsum([0] + mo), np.cumsum([0] + ni)

    def pieces(q, m_out, n_in):
        return [(q[cq[i]:cq[i + 1]], m_out[cmo[i]:cmo[i + 1]], n_in[cni[i]:cni[i + 1]])
                for i in range(n)]

    P = CotCoalg(
        Q_dim=int(cq[-1]),
        init=tuple(p.P.init for p in parts),
        position=lambda s: jnp.concatenate([parts[i].P.position(s[i]) for i in range(n)]),
        step=lambda s, xi: tuple(parts[i].P.step(s[i], xi[cq[i]:cq[i + 1]]) for i in range(n)),
        label=label,
    )

    def omega(q, m_out, n_in):
        cols = [parts[i].omega(*pieces(q, m_out, n_in)[i]) for i in range(n)]
        return tuple(jnp.concatenate([c[j] for c in cols]) for j in range(3))

    return Arrangement(
        P=P,
        in_dim_M=int(sum(mi)), out_dim_M=int(sum(mo)),
        in_dim_N=int(sum(ni)), out_dim_N=int(sum(no)),
        out_f=lambda q, m: jnp.concatenate(
            [parts[i].out_f(q[cq[i]:cq[i + 1]], m[cmo[i]:cmo[i + 1]]) for i in range(n)]),
        in_f=lambda q, m, nn: jnp.concatenate(
            [parts[i].in_f(*pieces(q, m, nn)[i]) for i in range(n)]),
        omega=omega, label=label,
    )


# ---------------------------------------------------------------------------
# The whole code, wired.
# ---------------------------------------------------------------------------


def code_arrangement(y0: Array, pairs, D: Array, beta: float, gamma: float,
                     dt: float, slip: bool = True, repertoire: bool = True,
                     definiteness: bool = True) -> Arrangement:
    """The closed arrangement `I -> I` of the whole code. `y0` is `(n_codons, K)` log-odds.

    `pairs` is the list of `(i, j, w)` triples from `scaled_pairs`; nothing else knows the code's
    adjacency, and no `n_codons x n_codons` matrix exists anywhere in this function.
    """
    from .wiring import compose_seq

    y0 = jnp.asarray(y0, dtype=float)
    n_codons, K = y0.shape
    boxes = tensor_all([codon_box(K, dt, y0[i], f"codon{i}") for i in range(n_codons)], "codons")

    costs: List[Arrangement] = []
    gather: List[int] = []
    if slip:
        for (i, j, w) in pairs:
            costs.append(slip_box(w, D, f"slip{i}-{j}"))
            gather += list(range(i * K, (i + 1) * K)) + list(range(j * K, (j + 1) * K))
    if definiteness and beta:
        for i in range(n_codons):
            costs.append(definiteness_box(beta, K, f"def{i}"))
            gather += list(range(i * K, (i + 1) * K))
    if repertoire and gamma:
        costs.append(repertoire_box(gamma, n_codons, K))
        gather += list(range(n_codons * K))
    if not costs:                       # a code with no cost at all: the boxes sit still
        costs, gather = [_cost_box(0, lambda e: jnp.sum(e) * 0.0, "no_cost")], []

    return compose_seq(compose_seq(boxes, fanout(n_codons * K, gather)),
                       tensor_all(costs, "costs"))


_IN_POS = (jnp.zeros(0), trivial_omega(0))
_IN_DIR = (jnp.zeros(0), jnp.zeros(0))


def phi_step(a: Arrangement, shape: Tuple[int, int], normalize: bool = True) -> Array:
    """One `Phi` step of `code_arrangement`, returned as `(n_codons, K)` ODDS.

    With `normalize=True` each row is divided by its sum. That is not a correction to the
    dynamics: the log chart's sharp subtracts the same `dt <x, g>` from every coordinate of a
    codon, so the row is right up to one scalar, and normalizing chooses the representative on
    the simplex (module docstring). With `normalize=False` the raw `exp(y')` is returned, which
    is what prediction 3 measures.
    """
    n, K = shape
    O = Phi(a)
    _, _, s = O.with_state(O.state).run_one(_IN_POS, lambda _o: _IN_DIR)
    x = jnp.exp(a.P.position(s)[: n * K]).reshape(n, K)
    return x / x.sum(1, keepdims=True) if normalize else x


def wiring_force(a: Arrangement, shape: Tuple[int, int]) -> Array:
    """The covector the wiring delivers to the codon boxes, in the LOG chart, as `(n, K)`.

    This is `omega`'s `Q` component at the initial state: nothing has moved yet, so it is purely
    what the cost boxes sent back through the fan-out's transpose.
    """
    n, K = shape
    y = a.P.position(a.P.init)
    xi_q, _, _ = a.omega(y, jnp.zeros(a.out_dim_M), jnp.zeros(a.in_dim_N))
    return xi_q[: n * K].reshape(n, K)
