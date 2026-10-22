"""A unital map from the paper's attention operad into the addressed operad (book plan P64 (a);
results in the book repository, ``notes/runs/p64-0ca3a7b8/RESULTS.md``).

``from_paper`` (``test_addressed.py``) shows each paper group as one speaker at its center, place 0. It
preserves substitution but sends the paper's unit to the slot that resets address and query to 0, not to
the addressed unit, which forwards them. Here the group's speaker instead shows the sum of the addresses
and queries its members sent, each in its own frame: ``sum_i (Z_i - l_i(q))``. (Any weights that multiply
along composites also work, e.g. the paper's center weights gamma_{0i}; David chose none, 2026-10-21.)
CHECKS A PROOF (RESULTS.md §2): the unit goes to the unit, and substitution is preserved.
"""

import jax
import jax.numpy as jnp
import numpy as np

from arrangements.addressed import (
    AddressedArrangement, in_dim, out_dim, slot_dim, speaker, split_slot, substitute, unit, Shown,
)
from arrangements.attention import substitute as paper_substitute, unit_arrangement
from arrangements.wiring import compose_seq, tensor_arrangements

from tests.test_addressed import DH, DP, assert_same_op, from_paper, paper_part

jax.config.update("jax_enable_x64", True)


def from_paper_unital(att):
    """As ``from_paper``, but the speaker shows the sum of the addresses and queries its members sent."""
    sd, n = slot_dim(att.dim_H, att.dim_P), att.n

    def cost(q, m_out, n_in):
        msgs = jnp.concatenate([split_slot(m_out[t * sd:(t + 1) * sd], att.dim_H, att.dim_P)[2]
                                for t in range(n)]) if n else jnp.zeros(0)
        return att.cost(q, msgs, n_in)

    def place(q, Z):
        ell = att.placement(q)
        k = sum((Z[(i, 0)][0] - ell[i] for i in range(n)), jnp.zeros(att.dim_P))
        p = sum((Z[(i, 0)][1] - ell[i] for i in range(n)), jnp.zeros(att.dim_P))
        return k, p

    sp = Shown(place=place,
               up=lambda q, y, Z: att.transfers(q)[0][y[0] + 1],
               down=lambda q, x, Z: att.transfers(q)[x[0] + 1][0], label="sum")
    return AddressedArrangement(
        slots=(1,) * n, exposure=(sp,), dim_H=att.dim_H, dim_P=att.dim_P, P=att.P,
        placement=att.placement, hearing=lambda q, x, y, Z: att.transfers(q)[x[0] + 1][y[0] + 1],
        cost=cost, label=f"paper*:{att.label}",
    )


def same_arrangement(x, y, seed, draws=4):
    """Same sending, receiving and forcing at random points (both with the same configuration)."""
    r = np.random.default_rng(seed)
    gaps = {"out": 0.0, "in": 0.0, "omega": 0.0}
    for _ in range(draws):
        q = jnp.asarray(r.standard_normal(x.P.Q_dim) * 0.5) if x.P.Q_dim else jnp.zeros(0)
        m = jnp.asarray(r.standard_normal(x.out_dim_M))
        e = jnp.asarray(r.standard_normal(x.in_dim_N))
        gaps["out"] = max(gaps["out"], float(jnp.max(jnp.abs(x.out_f(q, m) - y.out_f(q, m)))))
        gaps["in"] = max(gaps["in"], float(jnp.max(jnp.abs(x.in_f(q, m, e) - y.in_f(q, m, e)))))
        for a, b in zip(x.omega(q, m, e), y.omega(q, m, e)):
            if a.size:
                gaps["omega"] = max(gaps["omega"], float(jnp.max(jnp.abs(a - b))))
    return gaps


def test_center_speaker_misses_the_unit_and_summing_speaker_hits_it():
    """The paper's unit: ``from_paper`` resets the places (the obstruction); ``from_paper_unital`` is the unit."""
    u_paper, u_addr = unit_arrangement(DH, DP), unit(1, DH, DP)
    old = same_arrangement(from_paper(u_paper).arr(), u_addr.arr(), seed=1)
    new = same_arrangement(from_paper_unital(u_paper).arr(), u_addr.arr(), seed=1)
    print("unit, center speaker:", old)
    print("unit, summing speaker:", new)
    assert old["out"] > 0.1
    assert max(new.values()) < 1e-13
    # and as operations, every datum (place, up, down, hearing, placement, sending, receiving, cost)
    q0 = jnp.zeros(0)
    assert_same_op(from_paper_unital(u_paper), u_addr, q0, q0, seed=2, atol=0.0)


def test_weighted_speaker_preserves_substitution():
    """Nested twice, uneven, with a nullary part: the composite of the images is the image of the composite,
    as operations (every place, transfer, hearing, placement, sending, receiving, cost) and as arrangements."""
    a, a1, a2 = paper_part(2, 61), paper_part(2, 62), paper_part(1, 63)
    g = [paper_part(1, 64), paper_part(2, 65), paper_part(0, 66)]
    j = from_paper_unital
    cases = [
        (substitute(j(a), [j(a1), j(a2)]), j(paper_substitute(a, [a1, a2]))),
        (substitute(substitute(j(a), [j(a1), j(a2)]), [j(c) for c in g]),
         j(paper_substitute(paper_substitute(a, [a1, a2]), g))),
    ]
    for here, there in cases:
        q = jnp.asarray(np.random.default_rng(67).standard_normal(here.P.Q_dim) * 0.5)
        assert_same_op(here, there, q, q, seed=68, atol=1e-11)
    b = substitute(j(a), [j(a1), j(a2)])
    gaps = same_arrangement(b.arr(), compose_seq(tensor_arrangements([j(a1).arr(), j(a2).arr()]), j(a).arr()), seed=70)
    print("explicit composite vs composite in arr:", gaps)
    assert max(gaps.values()) < 1e-10


def test_unit_laws_through_the_map():
    """``j(a)`` composed with ``j(unit)`` in each slot, and ``j(unit)`` composed with ``j(a)``, is ``j(a)``."""
    a = paper_part(2, 81)
    u = from_paper_unital(unit_arrangement(DH, DP))
    for here in [substitute(from_paper_unital(a), [u, u]), substitute(u, [from_paper_unital(a)])]:
        gaps = same_arrangement(here.arr(), from_paper_unital(a).arr(), seed=82)
        print("unit law:", gaps)
        assert max(gaps.values()) < 1e-11
