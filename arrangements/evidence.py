"""Gaussian evidence trees: the family the precision sent forward lives in (book P53 (h)).

The mathematics, with the predictions fixed before this file existed, is the book's
``notes/runs/p53h-82601336/RESULTS.md`` §2. A message ``m = (h, w)``, ``w > 0``, is Gaussian evidence about a point
``x`` of ``H = R^r`` in information form: the likelihood ``exp(<h, x> - w |x|^2 / 2)``, whose mean is the attention
paper's mean readout ``h / w`` and whose log-normalizer is ``A(h, w) = |h|^2 / (2w) - (r/2) log w``. A sum of messages
is a product of likelihoods.

A *link* ``(kappa, U, s)`` is the linear-Gaussian edge ``x = kappa U z + N(0, s I)`` from a parent's latent ``z`` to a
child's latent ``x``; it passes evidence up and predictions down,

    up(h, w)   = (kappa U^T h, kappa^2 w) / (1 + s w),
    down(h, w) = (kappa U h, w) / (kappa^2 + s w).

An *evidence tree* is an operation of the operad: a rooted tree whose leaves are numbered slots or nodes; a node holds
an integrator, own evidence ``c_k(q)`` (a datum ``y`` seen with precision ``pi`` is ``(pi y, pi)``) and one link per
child. At a node, with children sending ``m_i`` and the input ``e`` from above,

    sigma     = sum_k c_k + sum_i up_i(m_i)
    rho_i     = down_i(e + sum_k c_k + sum_{j != i} up_j(m_j))
    cost      = sum_iota A(iota) - A(sum_iota iota) + r sum_i log kappa_i,   iota in {c_k} u {up_i(m_i)},

the last being the *disagreement* of the evidence arriving at the node. ``EvidenceTree.arr`` evaluates a whole tree
node by node, without ``wiring.py``; that it equals the lens composite of its nodes is a test, not a definition.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact
from .coalgebra import CotCoalg, configuration
from .rvect import euclidean


class Link(NamedTuple):
    """The edge ``x = kappa U z + N(0, s I)``: ``kappa > 0``, ``U`` orthogonal, ``s >= 0``."""

    kappa: Array
    U: Array
    s: Array


def log_normalizer(m: Array) -> Array:
    """``A(h, w) = |h|^2 / (2w) - (r/2) log w``, the isotropic Gaussian's log-partition (without ``(r/2) log 2 pi``)."""
    h, w = m[:-1], m[-1]
    return h @ h / (2.0 * w) - 0.5 * h.shape[0] * jnp.log(w)


def up(link: Link, m: Array) -> Array:
    h, w = m[:-1], m[-1]
    d = 1.0 + link.s * w
    return jnp.concatenate([link.kappa * (link.U.T @ h) / d, jnp.reshape(link.kappa ** 2 * w / d, (1,))])


def down(link: Link, m: Array) -> Array:
    h, w = m[:-1], m[-1]
    d = link.kappa ** 2 + link.s * w
    return jnp.concatenate([link.kappa * (link.U @ h) / d, jnp.reshape(w / d, (1,))])


def disagreement(inputs: Sequence[Array]) -> Array:
    """``sum A(iota) - A(sum iota)``: minus the log probability that the readings report one common value."""
    total = inputs[0]
    for x in inputs[1:]:
        total = total + x
    return sum(log_normalizer(x) for x in inputs) - log_normalizer(total)


def datum(y, precision) -> Array:
    """The own evidence of a datum ``y`` seen with precision ``precision``: ``(precision y, precision)``."""
    y = jnp.ravel(jnp.asarray(y, float))
    return jnp.concatenate([precision * y, jnp.reshape(precision, (1,))])


@dataclass(frozen=True)
class Node:
    """One node: its integrator ``P``, its own evidence ``own(q) -> (p, r+1)`` and its links ``links(q) -> [Link]``,
    one per child, all functions of the node's configuration."""

    P: CotCoalg
    own: Callable[[Array], Array]
    links: Callable[[Array], List[Link]]
    label: str = ""


SLOT = None  # a numbered slot leaf: where another tree is grafted


@dataclass(frozen=True)
class EvidenceTree:
    """A node and its children, each a tree or ``SLOT``. ``r = dim H``."""

    r: int
    node: Node
    children: Tuple[Optional["EvidenceTree"], ...]
    label: str = ""

    # ---- shape ----

    @property
    def n(self) -> int:
        """The number of slots."""
        return sum(1 if c is SLOT else c.n for c in self.children)

    @property
    def Q_dim(self) -> int:
        return sum(c.Q_dim for c in self.children if c is not SLOT) + self.node.P.Q_dim

    def parameter(self) -> CotCoalg:
        """Children's integrators in order, then the node's: the order ``compose_seq`` produces."""
        P = None
        for c in self.children:
            if c is not SLOT:
                P = c.parameter() if P is None else P.tensor(c.parameter())
        return self.node.P if P is None else P.tensor(self.node.P)

    def _split(self, q: Array, m: Array):
        """Per child: its configuration slice (or ``None``) and its message slice; then the node's configuration."""
        k = self.r + 1
        qs, ms, qc, mc = [], [], 0, 0
        for c in self.children:
            if c is SLOT:
                qs.append(None)
                ms.append(m[mc:mc + k])
                mc += k
            else:
                qs.append(q[qc:qc + c.Q_dim])
                ms.append(m[mc:mc + c.n * k])
                qc += c.Q_dim
                mc += c.n * k
        return qs, ms, q[qc:qc + self.node.P.Q_dim]

    # ---- sending, receiving, cost: node by node ----

    def sigma(self, q: Array, m: Array) -> Array:
        qs, ms, qn = self._split(q, m)
        sent = [ms[i] if c is SLOT else c.sigma(qs[i], ms[i]) for i, c in enumerate(self.children)]
        links = self.node.links(qn)
        own = self.node.own(qn)
        total = jnp.sum(own, axis=0) if own.shape[0] else jnp.zeros(self.r + 1)
        for L, x in zip(links, sent):
            total = total + up(L, x)
        return total

    def rho(self, q: Array, m: Array, e: Array) -> Array:
        qs, ms, qn = self._split(q, m)
        sent = [ms[i] if c is SLOT else c.sigma(qs[i], ms[i]) for i, c in enumerate(self.children)]
        links = self.node.links(qn)
        own = self.node.own(qn)
        ups = [up(L, x) for L, x in zip(links, sent)]
        base = e + (jnp.sum(own, axis=0) if own.shape[0] else 0.0)
        out = []
        for i, c in enumerate(self.children):
            cavity = base + sum((u for j, u in enumerate(ups) if j != i), jnp.zeros(self.r + 1))
            e_i = down(links[i], cavity)
            out.append(e_i if c is SLOT else c.rho(qs[i], ms[i], e_i))
        return jnp.concatenate(out) if out else jnp.zeros(0)

    def cost(self, q: Array, m: Array) -> Array:
        qs, ms, qn = self._split(q, m)
        sent = [ms[i] if c is SLOT else c.sigma(qs[i], ms[i]) for i, c in enumerate(self.children)]
        links = self.node.links(qn)
        own = self.node.own(qn)
        inputs = [own[k] for k in range(own.shape[0])] + [up(L, x) for L, x in zip(links, sent)]
        total = disagreement(inputs) + self.r * sum((jnp.log(L.kappa) for L in links), 0.0)
        for i, c in enumerate(self.children):
            if c is not SLOT:
                total = total + c.cost(qs[i], ms[i])
        return total

    def arr(self) -> Arrangement:
        """The tree as a morphism ``B^(x)n -> B`` of ``arr``, ``B = <M_>0 | M_>0>``: evaluated node by node."""
        k = self.r + 1
        return Arrangement(
            P=self.parameter(),
            in_dim_M=self.n * k, out_dim_M=self.n * k, in_dim_N=k, out_dim_N=k,
            out_f=self.sigma,
            in_f=self.rho,
            omega=exact(lambda q, m, e: -self.cost(q, m) + 0.0 * jnp.sum(e)),
            label=self.label or self.node.label or f"tree({self.n})",
        )


def corolla(r: int, node: Node, n: int) -> EvidenceTree:
    """The single-node operation with ``n`` slots."""
    return EvidenceTree(r, node, tuple(SLOT for _ in range(n)), label=node.label)


def graft(t: EvidenceTree, parts: Sequence[Optional[EvidenceTree]], label: str = "") -> EvidenceTree:
    """Operad composition ``t(parts)``: the ``i``-th part grafted at the ``i``-th slot (``SLOT``, the identity,
    leaves it open)."""
    parts = list(parts)
    if len(parts) != t.n:
        raise ValueError(f"graft: {t.n} slots, {len(parts)} parts")

    def go(tree: EvidenceTree, it) -> EvidenceTree:
        kids = []
        for c in tree.children:
            kids.append(next(it) if c is SLOT else go(c, it))
        return EvidenceTree(tree.r, tree.node, tuple(kids), tree.label)

    out = go(t, iter(parts))
    return EvidenceTree(out.r, out.node, out.children, label or t.label)


def identity(r: int) -> Arrangement:
    """The bare slot as a morphism ``B -> B``: the identity lens, no matter, cost ``0``."""
    k = r + 1
    return Arrangement(
        P=configuration(euclidean(0)),
        in_dim_M=k, out_dim_M=k, in_dim_N=k, out_dim_N=k,
        out_f=lambda q, m: m,
        in_f=lambda q, m, e: e,
        omega=exact(lambda q, m, e: 0.0 * (jnp.sum(m) + jnp.sum(e))),
        label="slot",
    )


def prior_cap(prior: Array) -> Arrangement:
    """A box ``B -> I`` closing a tree: it sends the prior ``prior`` down and costs the disagreement of the prior
    with the root's evidence, ``A(pi) + A(m) - A(pi + m)``."""
    k = prior.shape[0]
    return Arrangement(
        P=configuration(euclidean(0)),
        in_dim_M=k, out_dim_M=k, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: jnp.zeros(0),
        in_f=lambda q, m, e: prior,
        omega=exact(lambda q, m, e: -disagreement([prior, m])),
        label="prior",
    )
