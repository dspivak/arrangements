"""The genetic code as an arrangement: P32 stage (d) piece 2.

Predictions are `notes/P32-D-PREDICTIONS-2026-09-29.md`, fixed before the module was written;
each test quotes the one it checks and says what kind of evidence it is (STATUS.md section 2).
The object under test is `arrangements/genetic_code.py`; the thing it must reproduce is
`scripts/genetic_code_stability.py`, whose inputs were sourced in
`notes/P32-D-SOURCES-2026-09-29.md`.

The reference gradient in these tests assembles the 61x61 matrix `W` BY HAND, exactly as the
script does. That is the point of the comparison: the arrangement never forms it.

Claude Fable 5.1, session ec5ece87, 2026-09-29.
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

from arrangements import genetic_code as gc
from arrangements.wiring import tensor_arrangements


# ---------------------------------------------------------------------------
# Helpers: the script's rule, written out here as the thing to be reproduced.
# ---------------------------------------------------------------------------


def script_gradient(P0, pairs, D, beta, gamma):
    """`g` exactly as `scripts/genetic_code_stability.py` forms it, `W` assembled by hand."""
    n = P0.shape[0]
    W = np.zeros((n, n))
    for i, j, w in pairs:
        W[i, j] = w
        W[j, i] = w
    g = 2 * W @ P0 @ np.asarray(D)
    if gamma:
        g = g - gamma / (n * P0.mean(0))[None, :]
    if beta:
        g = g - beta * (np.log(P0) + 1)
    return g


def script_step(P0, pairs, D, beta, gamma, eta):
    """One rowwise entropic mirror-descent step, log-shift and renormalization included."""
    th = np.log(P0) - eta * script_gradient(P0, pairs, D, beta, gamma)
    th -= th.max(1, keepdims=True)
    Pn = np.exp(th)
    return Pn / Pn.sum(1, keepdims=True)


def small_case(seed=1):
    """Four codons over three amino acids: a mini code with a mini polar-requirement scale."""
    n, K = 4, 3
    D = jnp.array([[0.0, 1.0, 4.0], [1.0, 0.0, 1.0], [4.0, 1.0, 0.0]])
    pairs = gc.scaled_pairs([(0, 1, 1.0), (1, 2, 0.5), (2, 3, 0.1), (0, 3, 1.0)], n)
    rng = np.random.default_rng(seed)
    P0 = rng.random((n, K))
    P0 /= P0.sum(1, keepdims=True)
    return n, K, D, pairs, P0


def standard_case(eps=0.02):
    """The standard code softened: 61 sense codons, 20 amino acids, own amino acid `1 - eps`."""
    sense, std = gc.standard_code()
    n, K = len(sense), 20
    D = gc.aa_distance()
    pairs = gc.scaled_pairs(gc.neighbour_pairs(sense), n)
    P0 = np.full((n, K), eps / (K - 1))
    for i, c in enumerate(sense):
        P0[i, gc.AAS.index(std[c])] = 1 - eps
    return n, K, D, pairs, P0


# ---------------------------------------------------------------------------


def test_the_code_has_61_sense_codons_and_263_wired_pairs():
    """The adjacency the wiring is built from, against the literature's count.

    EXTERNAL CHECK. Haig and Hurst's graph on the sense codons has 263 edges (quoted in
    Buhrman et al., arXiv:0909.1442v2, p. 4: "yielding 263 edges"), and `neighbour_pairs`
    produces one slip box per edge. A wrong adjacency would pass every numerical test below,
    since they compare against a `W` built from the same list.
    """
    sense, std = gc.standard_code()
    assert len(sense) == 61
    assert len(gc.neighbour_pairs(sense)) == 263
    assert std["ATG"] == "M" and std["TGG"] == "W" and std["TAA"] == "*"
    assert gc.POLAR_REQUIREMENT["D"] == 13.0 and gc.POLAR_REQUIREMENT["C"] == 4.8
    assert (gc.fh_weight(0, "T", "C"), gc.fh_weight(0, "T", "A")) == (1.0, 0.5)
    assert (gc.fh_weight(1, "T", "C"), gc.fh_weight(1, "T", "A")) == (0.5, 0.1)
    assert (gc.fh_weight(2, "T", "C"), gc.fh_weight(2, "T", "A")) == (1.0, 1.0)


def test_one_phi_step_is_one_mirror_descent_step_small():
    """PREDICTION 1: "one `Phi` step of the arrangement, with each codon's row renormalized,
    equals one step of the script's mirror descent from the same start, to 1e-12 or better."

    EXACT REPRODUCTION of an independently written rule. The script's step is recomputed here
    in numpy from a hand-assembled `W`; the arrangement computes it from 4 codon boxes, 4 slip
    boxes, 4 definiteness boxes and 1 repertoire box wired by a gather.
    """
    n, K, D, pairs, P0 = small_case()
    dt, beta, gamma = 0.02, 0.7, 1.3
    a = gc.code_arrangement(jnp.log(jnp.asarray(P0)), pairs, D, beta, gamma, dt)
    got = np.asarray(gc.phi_step(a, (n, K)))
    assert np.abs(got - script_step(P0, pairs, D, beta, gamma, dt)).max() < 1e-12


@pytest.mark.parametrize("beta,gamma", [(2.0, 3.0)])
def test_one_phi_step_is_one_mirror_descent_step_full_code(beta, gamma):
    """PREDICTION 2: "the same on the full 61 sense codons over 20 amino acids."

    EXACT REPRODUCTION at full size: 61 codon boxes, 263 slip boxes, 61 definiteness boxes and
    1 repertoire box, 386 in all. Slow (tens of seconds): the composite's forcing traces every
    box once.
    """
    n, K, D, pairs, P0 = standard_case()
    dt = 0.02
    a = gc.code_arrangement(jnp.log(jnp.asarray(P0)), pairs, D, beta, gamma, dt)
    got = np.asarray(gc.phi_step(a, (n, K)))
    assert np.abs(got - script_step(P0, pairs, D, beta, gamma, dt)).max() < 1e-12


def test_the_unnormalized_mismatch_is_one_scalar_per_codon():
    """PREDICTION 3: "unnormalized, the arrangement's row is the script's times
    `exp(dt <x, g>)`, one scalar per codon, because the log-chart Shahshahani sharp's `-1 1^T`
    term subtracts the same number from every amino-acid coordinate."

    CLOSED FORM. The predicted scalar is computed from `<x, g>` and compared; and the ratio is
    checked to be constant along each row, which is the claim that makes renormalizing legitimate
    rather than a fudge. Had the correction varied with the amino acid, the two tests above
    would fail and no renormalization would save them.
    """
    n, K, D, pairs, P0 = small_case(seed=7)
    dt, beta, gamma = 0.05, 0.4, 2.0
    a = gc.code_arrangement(jnp.log(jnp.asarray(P0)), pairs, D, beta, gamma, dt)
    raw = np.asarray(gc.phi_step(a, (n, K), normalize=False))
    g = script_gradient(P0, pairs, D, beta, gamma)
    ratio = raw / (P0 * np.exp(-dt * g))
    assert np.abs(ratio - ratio[:, :1]).max() < 1e-12          # constant along each row
    assert np.abs(ratio[:, 0] - np.exp(dt * (P0 * g).sum(1))).max() < 1e-12
    assert np.abs(ratio - 1.0).max() > 1e-4                    # and it is not trivially 1


def test_the_slip_force_emerges_from_the_wiring():
    """PREDICTION 4: "the force the wiring delivers to codon `c` equals the script's
    `-(2 (W P D)_c + repertoire + definiteness)`, while the arrangement code contains no 61x61
    matrix and no sum over neighbours."

    THE RESULT OF THIS STAGE, and the one that could have failed three ways named in advance:
    the factor 2 on an unordered pair, the adjoint of the gather, and double counting. Each slip
    box sees two codons' odds and the 20x20 `D`; the sum over a codon's nine neighbours exists
    only as the transpose of the fan-out. Checked at full size, where a codon really has nine.
    """
    n, K, D, pairs, P0 = standard_case()
    beta, gamma = 1.0, 3.0
    a = gc.code_arrangement(jnp.log(jnp.asarray(P0)), pairs, D, beta, gamma, 0.02)
    xi = np.asarray(gc.wiring_force(a, (n, K)))                # the covector, in the log chart
    want = -P0 * script_gradient(P0, pairs, D, beta, gamma)    # -x * g, the log-chart pullback
    assert np.abs(xi - want).max() < 1e-10
    # and the module holds no square matrix over codons: the boxes only ever see 20 x 20.
    assert all(b.out_dim_M == 2 * K for b in
               [gc.slip_box(w, D) for (_, _, w) in pairs[:5]])


def test_dropping_one_pair_moves_exactly_two_codons():
    """PREDICTION 4, the locality half, and PREDICTION 5's "a wrong adjoint leaks force between
    ports, and this is how it would show."

    NEGATIVE CONTROL. Delete one slip box and the force must change on exactly the two codons it
    joined, by exactly that box's own contribution, and on no other codon at all.
    """
    n, K, D, pairs, P0 = small_case(seed=3)
    beta, gamma = 0.0, 0.0
    full = gc.wiring_force(gc.code_arrangement(
        jnp.log(jnp.asarray(P0)), pairs, D, beta, gamma, 0.02), (n, K))
    i, j, w = pairs[1]
    less = gc.wiring_force(gc.code_arrangement(
        jnp.log(jnp.asarray(P0)), [p for k, p in enumerate(pairs) if k != 1], D,
        beta, gamma, 0.02), (n, K))
    d = np.asarray(full - less)
    untouched = [r for r in range(n) if r not in (i, j)]
    assert np.abs(d[untouched]).max() < 1e-14
    assert np.abs(d[i] - (-P0[i] * 2 * w * (np.asarray(D) @ P0[j]))).max() < 1e-12
    assert np.abs(d[j] - (-P0[j] * 2 * w * (np.asarray(D) @ P0[i]))).max() < 1e-12


def test_each_cost_can_be_switched_off_independently():
    """PREDICTION 5: "dropping the slip boxes leaves exactly repertoire plus definiteness;
    dropping the repertoire box leaves exactly slip plus definiteness."

    NEGATIVE CONTROL on the wiring's bookkeeping. The three costs are separate boxes, so each
    subset must give the gradient of exactly that subset -- no residue of a box that is not there.
    """
    n, K, D, pairs, P0 = small_case(seed=11)
    beta, gamma, dt = 0.9, 1.7, 0.02
    y0 = jnp.log(jnp.asarray(P0))

    def force(**kw):
        return np.asarray(gc.wiring_force(
            gc.code_arrangement(y0, pairs, D, beta, gamma, dt, **kw), (n, K)))

    assert np.abs(force(slip=False)
                  - (-P0 * script_gradient(P0, [], D, beta, gamma))).max() < 1e-12
    assert np.abs(force(repertoire=False)
                  - (-P0 * script_gradient(P0, pairs, D, beta, 0.0))).max() < 1e-12
    assert np.abs(force(definiteness=False)
                  - (-P0 * script_gradient(P0, pairs, D, 0.0, gamma))).max() < 1e-12


def test_a_codon_wired_to_no_neighbour_feels_no_slip():
    """PREDICTION 5: "a codon wired to no slip box receives zero slip force."

    NEGATIVE CONTROL. Codon 3 is left out of every pair; with the other costs off, its covector
    must be exactly zero, while the wired codons' is not.
    """
    n, K, D, _, P0 = small_case(seed=5)
    pairs = gc.scaled_pairs([(0, 1, 1.0), (1, 2, 0.5)], n)     # codon 3 isolated
    xi = np.asarray(gc.wiring_force(
        gc.code_arrangement(jnp.log(jnp.asarray(P0)), pairs, D, 0.0, 0.0, 0.02), (n, K)))
    assert np.abs(xi[3]).max() == 0.0
    assert np.abs(xi[:3]).max() > 1e-6


def test_flat_tensor_agrees_with_the_librarys_binary_fold():
    """PREDICTION 6: "the flat n-ary monoidal product agrees with the library's binary fold."

    TWO IMPLEMENTATIONS AGREEING (STATUS.md section 2). `tensor_all` exists only because folding
    `parallel_arrangements` 386 times nests the state 386 deep; it must compute the same direct
    sum, so it is checked against the fold on a case small enough for the fold.
    """
    n, K, D, pairs, P0 = small_case(seed=2)
    dt = 0.02
    y0 = jnp.log(jnp.asarray(P0))
    boxes = [gc.codon_box(K, dt, y0[i], f"codon{i}") for i in range(n)]
    flat, fold = gc.tensor_all(boxes), tensor_arrangements(boxes)
    assert (flat.P.Q_dim, flat.out_dim_N) == (fold.P.Q_dim, fold.out_dim_N)
    q = jnp.asarray(np.random.default_rng(0).normal(size=flat.P.Q_dim))
    z0, z1 = jnp.zeros(0), jnp.zeros(0)
    assert np.abs(np.asarray(flat.out_f(q, z0) - fold.out_f(q, z0))).max() < 1e-14
    for u, v in zip(flat.omega(q, z0, z1), fold.omega(q, z0, z1)):
        assert u.shape == v.shape
        if u.size:                      # the port components are zero-size for a closed box
            assert np.abs(np.asarray(u - v)).max() < 1e-14
    assert np.abs(np.asarray(flat.P.position(flat.P.init)
                             - jnp.concatenate([b.P.init for b in boxes]))).max() < 1e-14
