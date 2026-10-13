"""The operad of adaptive attention, realized in ``arr`` (def:arr).

The mathematics is ``background/affine_quadratic_operad.tex`` in the book repository
(*An operad of adaptive attention*), cited below by its own labels
(``def:arrangement``, ``eq:transfer``, ``eq:maps``, ``eq:composite-transfers``,
``prop:operad``, ...). An ``n``-ary attention arrangement ``a = (I, l, T, k)`` is an
operation of the symmetric operad ``A_B``; ``prop:operad`` says that
``(I_a, sigma_a, rho_a, omega_a)`` is an operad morphism into ``End(B)``, so each
such ``a`` is a morphism ``B^(x)n -> B`` of ``arr``. The interface is
``B = (E, M)`` with message space ``M = H x W``, ``W = R``, and input space
``E = M x P``; in the notation of def:arr, ``B^+ = M`` and ``B^- = E``, so the
paper's interaction space ``D_n = M^n x E`` *is* an ``Arrangement``'s pair ``(M^+, N^-)`` for an
arrangement ``B^(x)n -> B``, and the port dimensions are

    out_dim_M = n * (dim_H + 1),    in_dim_M = n * (dim_H + 1 + dim_P),
    out_dim_N = dim_H + 1,          in_dim_N = dim_H + 1 + dim_P.

``sigma_a`` (eq:maps) is ``out_f``, ``rho_a`` is ``in_f``, and the forcing
``omega_a = -dk_a`` (eq:forcing) -- the *full* differential on ``Q x D_n``, not
just the differential in ``q`` -- is ``exact(lambda q, m, n: -k(q, m, n))``. The
paper already uses the book's convention ``U = -k``, ``omega = dU``, so unlike
every CDLM port there is no sign translation here (STATUS.md §1).

**What is not implemented.** The paper's integrator is ``I = (Q, C, iota)`` with
``C`` an arbitrary polynomial comonad in ``Cat^#``; the response category can
encode more state than the configuration ``q in Q``. The book's ``arr`` uses the
concrete ``Store(S)`` comonad only (STATUS.md §3.3), so here
``CotCoalg`` -- a ``cot(Q)``-coalgebra ``(Q, S, u)``, i.e. ``Store(S) => cot(Q)``
-- *is* the whole of the integrator. The general ``C`` is not implemented, and
``CotCoalg`` is not to be presented as the paper's ``I``; nothing in this module
or in the tests needs the extra generality.

Two constraints that ``R^d`` does not enforce are guarded rather than pretended
away (notes/ATTENTION-STATUS.md §1):

* the admissible messages are ``{0_M} u (H x R_{>0})``; a transfer preserves them
  (``gamma > 0``), but nothing stops a caller from handing in a negative weight,
  so ``is_admissible`` is provided and asserted where the paper assumes it;
* the mean readout ``mean(h, w) = h / w`` (eq:mean) is defined only for ``w > 0``.
  It is a separate function and is never folded into a cost: the paper notes that
  applying it inside a cost would generally leave the quadratic family.

Both checks read concrete values, so they are for eager use (as in the tests),
not inside ``jit``.

Coordinates. A message ``m = (h, w) in M`` is the array ``[h..., w]`` of length
``dim_H + 1``; an input ``e = (m_0, p_0) in E`` is ``[h..., w, p...]`` of length
``dim_H + 1 + dim_P``; the ``n`` child messages and the ``n`` child inputs are the
concatenations in index order. A transfer (eq:transfer) is the pair
``(gamma, L)`` of ``Transfer``, never a dense ``End(M)`` matrix, and transfers are
combined by the algebra of eq:transfer-algebra.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, List, NamedTuple, Optional, Sequence, Tuple

import jax.numpy as jnp
from jax import Array

from .arrangement import Arrangement, exact
from .coalgebra import CotCoalg, configuration
from .rvect import euclidean, trivial


# ---------------------------------------------------------------------------
# Signals, positions, and weighted messages.
# ---------------------------------------------------------------------------


def msg_dim(dim_H: int) -> int:
    """``dim M = dim(H x W) = dim_H + 1``."""
    return dim_H + 1


def input_dim(dim_H: int, dim_P: int) -> int:
    """``dim E = dim(M x P) = dim_H + 1 + dim_P``."""
    return dim_H + 1 + dim_P


def message(h, w) -> Array:
    """The message ``m = (h, w) in M = H x W``."""
    return jnp.concatenate(
        [jnp.ravel(jnp.asarray(h, float)), jnp.reshape(jnp.asarray(w, float), (1,))]
    )


def zero_message(dim_H: int) -> Array:
    """``0_M = (0_H, 0)``; also the value of an empty sum of messages."""
    return jnp.zeros(dim_H + 1)


def signal(m: Array) -> Array:
    """The signal ``h in H`` of a message ``m = (h, w)`` (a *weighted total*)."""
    return m[:-1]


def weight(m: Array) -> Array:
    """The weight ``w in W = R`` of a message ``m = (h, w)``."""
    return m[-1]


def make_input(m: Array, p) -> Array:
    """The input ``e = (m, p) in E = M x P``."""
    return jnp.concatenate([m, jnp.ravel(jnp.asarray(p, float))])


def split_input(e: Array, dim_H: int) -> Tuple[Array, Array]:
    """An input ``e in E`` as its message and reference position ``(m_0, p_0)``."""
    return e[: dim_H + 1], e[dim_H + 1:]


def split_messages(m_out: Array, n: int, dim_H: int) -> Array:
    """The ``n`` child messages of ``M^n``, as an ``(n, dim_H + 1)`` array."""
    return jnp.reshape(m_out, (n, dim_H + 1))


def split_inputs(m_in: Array, n: int, dim_H: int, dim_P: int) -> Array:
    """The ``n`` child inputs of ``E^n``, as an ``(n, dim_H + 1 + dim_P)`` array."""
    return jnp.reshape(m_in, (n, input_dim(dim_H, dim_P)))


def _concrete(x) -> Optional[float]:
    """``float(x)`` if ``x`` carries a concrete value, else ``None`` (a tracer)."""
    try:
        return float(x)
    except Exception:  # a jax tracer, or not a scalar
        return None


def is_admissible(m: Array) -> bool:
    """Is ``m`` in ``M_adm = {0_M} u M_{>0}``, ``M_{>0} = H x R_{>0}``?

    Reads concrete values (not for use inside ``jit``). Nothing in ``R^{dim_H+1}``
    enforces this; the paper assumes it wherever a mean readout is taken.
    """
    m = jnp.asarray(m)
    w = _concrete(weight(m))
    if w is None:
        raise TypeError("is_admissible needs a concrete message, not a tracer")
    if w > 0.0:
        return True
    return w == 0.0 and bool(jnp.all(signal(m) == 0.0))


def assert_admissible(m: Array, where: str = "") -> None:
    """Raise unless ``m in M_adm``; used where the paper assumes admissibility."""
    if not is_admissible(m):
        raise ValueError(f"inadmissible message {jnp.asarray(m)}{' in ' + where if where else ''}")


def mean(m: Array) -> Array:
    """The mean readout ``mean(h, w) = h / w`` on ``M_{>0}`` (eq:mean).

    Kept a separate operation, defined only for positive total weight:
    communication retains both coordinates, and the paper notes that applying the
    mean readout *inside* a cost would generally leave the quadratic family.
    """
    w = weight(m)
    wc = _concrete(w)
    if wc is not None and not wc > 0.0:
        raise ValueError(f"mean is defined on M_{{>0}}; got weight {wc}")
    return signal(m) / w


# ---------------------------------------------------------------------------
# Message transfers (eq:transfer, eq:transfer-algebra).
# ---------------------------------------------------------------------------


class Transfer(NamedTuple):
    """A message transfer ``T_{gamma,L} : M -> M``, ``(h, w) |-> (L h, gamma w)`` (eq:transfer).

    An element of ``T_H``: either ``gamma > 0`` with ``L in End(H)``, or the zero
    map ``T_{0,0}``. A ``NamedTuple``, hence a jax pytree, so a family
    ``Q -> T_H`` is differentiated in the configuration like any other smooth map.
    The pair ``(gamma, L)`` is the representation throughout -- transfers are
    applied and combined through it, never as a dense ``End(M)`` matrix.
    """

    gamma: Array  # scalar weight multiplier
    L: Array      # (dim_H, dim_H) signal map


def transfer(gamma, L) -> Transfer:
    return Transfer(gamma=jnp.asarray(gamma, float), L=jnp.asarray(L, float))


def zero_transfer(dim_H: int) -> Transfer:
    """The zero map on ``M``, regarded as ``T_{0,0}``."""
    return Transfer(gamma=jnp.asarray(0.0), L=jnp.zeros((dim_H, dim_H)))


def identity_transfer(dim_H: int) -> Transfer:
    """The identity transfer ``M``: weight multiplier ``1``, signal map ``H``."""
    return Transfer(gamma=jnp.asarray(1.0), L=jnp.eye(dim_H))


def add_transfers(T: Transfer, T2: Transfer) -> Transfer:
    """``T_{gamma,L} + T_{gamma',L'} = T_{gamma+gamma', L+L'}`` (eq:transfer-algebra)."""
    return Transfer(gamma=T.gamma + T2.gamma, L=T.L + T2.L)


def compose_transfers(T: Transfer, T2: Transfer) -> Transfer:
    """``T_{gamma,L} . T_{gamma',L'} = T_{gamma gamma', L L'}`` (eq:transfer-algebra).

    Ordered composition ``T . T2``; signal maps need not commute.
    """
    return Transfer(gamma=T.gamma * T2.gamma, L=T.L @ T2.L)


def apply_transfer(T: Transfer, m: Array) -> Array:
    """``T_{gamma,L}(h, w) = (L h, gamma w)`` (eq:transfer), on coordinates."""
    return jnp.concatenate([T.L @ signal(m), jnp.reshape(T.gamma * weight(m), (1,))])


def is_transfer(T: Transfer) -> bool:
    """Is ``T`` an element of ``T_H`` -- the zero map, or ``gamma > 0``?

    Reads concrete values. ``gamma <= 0`` with ``L != 0`` is a linear map on ``M``
    that is not a message transfer: it does not preserve admissibility.
    """
    g = _concrete(T.gamma)
    if g is None:
        raise TypeError("is_transfer needs a concrete transfer, not a tracer")
    if g > 0.0:
        return True
    return g == 0.0 and bool(jnp.all(T.L == 0.0))


def is_zero_transfer(T: Transfer) -> bool:
    return bool(jnp.all(T.gamma == 0.0)) and bool(jnp.all(T.L == 0.0))


# A transfer family: ``Q -> T_H^{(n+1) x (n+1)}``, indexed ``[i][j]`` for
# ``i, j in {0, ..., n}``; index ``0`` is the exterior interface.
TransferMatrix = Tuple[Tuple[Transfer, ...], ...]
Transfers = Callable[[Array], TransferMatrix]
Placement = Callable[[Array], Array]          # q |-> (n, dim_P), the l_i(q)
Cost = Callable[[Array, Array, Array], Array]  # (q, m_out, n_in) |-> R_{>=0}


def zero_cost(q: Array, m_out: Array, n_in: Array) -> Array:
    """The zero cost ``k = 0`` (that of a pure placement arrangement)."""
    return jnp.asarray(0.0)


def identity_transfers(n: int, dim_H: int) -> Transfers:
    """``T_{ij} = M`` for ``i != j``, ``T_{ii} = 0``: the transfers of ``P_B``."""
    Z, I = zero_transfer(dim_H), identity_transfer(dim_H)

    def T(q: Array) -> TransferMatrix:
        return tuple(tuple(Z if i == j else I for j in range(n + 1)) for i in range(n + 1))

    return T


# ---------------------------------------------------------------------------
# Attention arrangements (def:arrangement) and their image in arr.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AttentionArrangement:
    """An ``n``-ary attention arrangement ``a = (I, l, T, k)`` (def:arrangement).

    * ``P``         -- the integrator ``I``: here a ``cot(Q)``-coalgebra
                       ``(Q, S, u)``, i.e. ``C = Store(S)`` only (module docstring).
    * ``placement`` -- ``l : Q -> P^n``, as ``q |-> (n, dim_P)``; ``l_0 = 0_P``.
    * ``transfers`` -- ``T : Q -> T_H^{(n+1)x(n+1)}``, ``T[i][j]`` the transfer from
                       sender ``j`` to receiver ``i``, with ``T[i][i] = 0``.
    * ``cost``      -- ``k : Q x D_n -> R_{>=0}``, ``k(q, -)`` of degree at most two.

    Index ``0`` is the exterior interface. ``arr()`` is the image under the operad
    morphism of prop:operad: the morphism ``B^(x)n -> B`` of ``arr`` with
    ``out_f = sigma_a``, ``in_f = rho_a`` and forcing ``omega_a = -dk_a``.
    """

    n: int
    dim_H: int
    dim_P: int
    P: CotCoalg
    placement: Placement
    transfers: Transfers
    cost: Cost = zero_cost
    label: str = ""

    # ---- derived dimensions ----

    @property
    def msg_dim(self) -> int:
        return msg_dim(self.dim_H)

    @property
    def input_dim(self) -> int:
        return input_dim(self.dim_H, self.dim_P)

    # ---- the sending and receiving maps (eq:maps) ----

    def sigma(self, q: Array, m_out: Array) -> Array:
        """``sigma_a(q, m) = sum_{j=1}^n T_{0j}(q) m_j`` (eq:maps); empty sum ``0_M``."""
        T = self.transfers(q)
        ms = split_messages(m_out, self.n, self.dim_H)
        out = zero_message(self.dim_H)
        for j in range(1, self.n + 1):
            out = out + apply_transfer(T[0][j], ms[j - 1])
        return out

    def rho(self, q: Array, m_out: Array, n_in: Array) -> Array:
        """``rho_a(q, m, (m_0, p_0))_i = (T_{i0} m_0 + sum_j T_{ij} m_j, p_0 + l_i(q))`` (eq:maps)."""
        T = self.transfers(q)
        ms = split_messages(m_out, self.n, self.dim_H)
        m_0, p_0 = split_input(n_in, self.dim_H)
        ell = self.placement(q)
        parts = []
        for i in range(1, self.n + 1):
            msg = apply_transfer(T[i][0], m_0)
            for j in range(1, self.n + 1):
                msg = msg + apply_transfer(T[i][j], ms[j - 1])
            parts.append(make_input(msg, p_0 + ell[i - 1]))
        if not parts:
            return jnp.zeros(0)
        return jnp.concatenate(parts)

    def received(self, q: Array, m_out: Array, n_in: Array, i: int) -> Array:
        """``rho_a(q, m, e)_i in E``, the input delivered to child ``i`` (``1 <= i <= n``)."""
        d = self.input_dim
        return self.rho(q, m_out, n_in)[(i - 1) * d: i * d]

    # ---- the image in arr (notes/DESIGN.md §5) ----

    def arr(self) -> Arrangement:
        """``a`` as a morphism ``B^(x)n -> B`` of ``arr`` (prop:operad).

        The forcing is the *full* differential ``omega_a = -dk_a`` on ``Q x D_n``
        (eq:forcing), including the message and position coordinates of the input.
        """
        return Arrangement(
            P=self.P,
            in_dim_M=self.n * self.input_dim,
            out_dim_M=self.n * self.msg_dim,
            in_dim_N=self.input_dim,
            out_dim_N=self.msg_dim,
            out_f=self.sigma,
            in_f=self.rho,
            omega=exact(lambda q, m_out, n_in: -self.cost(q, m_out, n_in)),
            label=self.label or f"attention({self.n})",
        )


def validate(a: AttentionArrangement, q: Optional[Array] = None) -> AttentionArrangement:
    """Check def:arrangement's standing conditions at one configuration.

    Namely: ``T_{ii}(q) = 0`` (a child does not receive its own message through
    this arrangement) and every ``T_{ij}(q) in T_H`` (eq:transfer: the zero map,
    or ``gamma > 0``, which is what preserves admissibility). ``q`` defaults to the
    position the parameter's initial state presents. This is a check at *one*
    configuration, not a proof for all ``q``.
    """
    if a.n < 0:
        raise ValueError("arity must be >= 0")
    if q is None:
        q = a.P.position(a.P.init)
    T = a.transfers(q)
    if len(T) != a.n + 1 or any(len(row) != a.n + 1 for row in T):
        raise ValueError(f"transfers must be ({a.n + 1})x({a.n + 1})")
    for i in range(a.n + 1):
        for j in range(a.n + 1):
            if i == j:
                if not is_zero_transfer(T[i][i]):
                    raise ValueError(f"T_{i}{i} must be zero (def:arrangement)")
            elif not is_transfer(T[i][j]):
                raise ValueError(f"T_{i}{j} is not a message transfer (eq:transfer)")
    ell = a.placement(q)
    if jnp.shape(ell) != (a.n, a.dim_P):
        raise ValueError(f"placement must return ({a.n}, {a.dim_P}), got {jnp.shape(ell)}")
    return a


def attention_arrangement(
    n: int,
    dim_H: int,
    dim_P: int,
    placement: Placement,
    transfers: Transfers,
    cost: Cost = zero_cost,
    parameter: Optional[CotCoalg] = None,
    label: str = "",
    check: bool = True,
) -> AttentionArrangement:
    """Build an ``n``-ary attention arrangement (def:arrangement) and check it.

    ``parameter`` is the integrator ``I``, a ``cot(Q)``-coalgebra; the default is
    the trivial parameter ``R^0`` (a static arrangement). ``check`` runs
    ``validate`` at the parameter's initial position.
    """
    if parameter is None:
        parameter = configuration(trivial())
    a = AttentionArrangement(
        n=n, dim_H=dim_H, dim_P=dim_P, P=parameter,
        placement=placement, transfers=transfers, cost=cost, label=label,
    )
    return validate(a) if check else a


def unit_arrangement(dim_H: int, dim_P: int, label: str = "1") -> AttentionArrangement:
    """The operad unit (prop:operad): arity one, ``I_u`` (one-point configuration),
    placement zero, cost zero, ``T_{01} = T_{10} = M``. Its sending and receiving
    maps are identities."""
    Z, I = zero_transfer(dim_H), identity_transfer(dim_H)
    return AttentionArrangement(
        n=1, dim_H=dim_H, dim_P=dim_P,
        P=configuration(trivial()),
        placement=lambda q: jnp.zeros((1, dim_P)),
        transfers=lambda q: ((Z, I), (I, Z)),
        cost=zero_cost,
        label=label,
    )


# ---------------------------------------------------------------------------
# Substitution (eq:composite-state, eq:composite-transfers, eq:composite-cost).
# ---------------------------------------------------------------------------


def substitute(
    a: AttentionArrangement, parts: Sequence[AttentionArrangement], label: str = "",
) -> AttentionArrangement:
    """The operad substitution ``b = a(a_1, ..., a_n)``, by the paper's explicit formulas.

    This is eq:composite-state, eq:composite-transfers and eq:composite-cost
    computed directly: no lens composition, no reference to ``wiring.py``. The
    composite has ``N = sum_i n_i`` children, indexed ``(i, j)`` in lexicographic
    order, and (prop:operad) it agrees with the composite in ``arr``,
    ``compose_seq(tensor_arrangements([a_1.arr(), ...]), a.arr())``.

    Configuration factors. The paper writes ``Q_b = Q_a x prod_i Q_{a_i}``
    (eq:composite-state); a product's factors are ordered by convention, and this
    module orders them ``Q_{a_1} x ... x Q_{a_n} x Q_a``, which is the order
    ``compose_seq`` produces, so that the two composites have literally the same
    parameter and their configurations can be compared coordinatewise.
    """
    parts = list(parts)
    n, dim_H, dim_P = a.n, a.dim_H, a.dim_P
    if len(parts) != n:
        raise ValueError(f"substitute: {a.label or 'a'} has arity {n}, got {len(parts)} parts")
    for i, p in enumerate(parts):
        if (p.dim_H, p.dim_P) != (dim_H, dim_P):
            raise ValueError(f"part {i}: signal/position spaces differ from the parent's")

    arities = [p.n for p in parts]
    N = sum(arities)
    starts = [sum(arities[:i]) for i in range(n + 1)]       # flat child-index offsets
    qd = [p.P.Q_dim for p in parts]
    qcut = [sum(qd[:i]) for i in range(n + 1)]              # configuration-block cuts
    Qa = a.P.Q_dim
    edim = input_dim(dim_H, dim_P)
    mdim = msg_dim(dim_H)

    # The integrator: the coherent product (eq:integrators), in the order above.
    P = None
    for p in parts:
        P = p.P if P is None else P.tensor(p.P)
    P = a.P if P is None else P.tensor(a.P)

    def split_q(q: Array) -> Tuple[List[Array], Array]:
        qs = [q[qcut[i]:qcut[i + 1]] for i in range(n)]
        return qs, q[qcut[n]: qcut[n] + Qa]

    def placement(q: Array) -> Array:
        """``l_{b,(i,j)} = l_{a,i}(q_a) + l_{a_i,j}(q_i)`` (eq:composite-state)."""
        qs, qa = split_q(q)
        la = a.placement(qa)
        rows = []
        for i in range(n):
            li = parts[i].placement(qs[i])
            for j in range(arities[i]):
                rows.append(la[i] + li[j])
        return jnp.stack(rows) if rows else jnp.zeros((0, dim_P))

    def transfers(q: Array) -> TransferMatrix:
        """eq:composite-transfers, each constituent evaluated at its own configuration."""
        qs, qa = split_q(q)
        Ta = a.transfers(qa)
        Ti = [parts[i].transfers(qs[i]) for i in range(n)]
        Z = zero_transfer(dim_H)
        T = [[Z for _ in range(N + 1)] for _ in range(N + 1)]
        for i in range(n):
            for j in range(1, arities[i] + 1):
                x = 1 + starts[i] + (j - 1)
                T[0][x] = compose_transfers(Ta[0][i + 1], Ti[i][0][j])
                T[x][0] = compose_transfers(Ti[i][j][0], Ta[i + 1][0])
                for r in range(n):
                    for s in range(1, arities[r] + 1):
                        y = 1 + starts[r] + (s - 1)
                        if i == r:
                            # inside a group: the group's own transfer (zero if j = s)
                            T[x][y] = Ti[i][j][s]
                        else:
                            # out through the sender's group, across a, in through the
                            # receiver's group; a's zero diagonal kills the i = r path.
                            T[x][y] = compose_transfers(
                                compose_transfers(Ti[i][j][0], Ta[i + 1][r + 1]), Ti[r][0][s]
                            )
        return tuple(tuple(row) for row in T)

    def cost(q: Array, m_out: Array, n_in: Array) -> Array:
        """``k_b = k_a(q_a, m~, e) + sum_i k_{a_i}(q_i, m_i, e_i)`` (eq:composite-cost),
        with the local interactions of eq:local-interactions."""
        qs, qa = split_q(q)
        ms = split_messages(m_out, N, dim_H)
        locals_m = [jnp.ravel(ms[starts[i]:starts[i + 1]]) for i in range(n)]
        tilde = [parts[i].sigma(qs[i], locals_m[i]) for i in range(n)]
        m_tilde = jnp.concatenate(tilde) if tilde else jnp.zeros(0)
        e_all = a.rho(qa, m_tilde, n_in)
        total = a.cost(qa, m_tilde, n_in)
        for i in range(n):
            total = total + parts[i].cost(qs[i], locals_m[i], e_all[i * edim:(i + 1) * edim])
        return total

    child_labels = ", ".join(p.label or "?" for p in parts)
    return AttentionArrangement(
        n=N, dim_H=dim_H, dim_P=dim_P, P=P,
        placement=placement, transfers=transfers, cost=cost,
        label=label or f"{a.label or 'a'}({child_labels})",
    )


def symmetric_action(a: AttentionArrangement, tau: Sequence[int]) -> AttentionArrangement:
    """The action of ``tau in S_n`` on ``A_B(n)`` (prop:operad).

    ``tau`` is given in one-line form as a permutation of ``range(n)``: slot ``i``
    of ``a . tau`` is slot ``tau[i]`` of ``a``, i.e. ``tau[i] = tau(i+1) - 1`` in
    the paper's 1-based indexing. Then

        l_{a.tau, i} = l_{a, tau(i)},   T^{a.tau}_{ij} = T^a_{tau(i) tau(j)},
        k_{a.tau}(q, m, e) = k_a(q, (m_{tau^{-1}(1)}, ..., m_{tau^{-1}(n)}), e),

    extending ``tau(0) = 0``: permutations relabel child indices and their message
    slots, fixing the exterior index.
    """
    n, dim_H = a.n, a.dim_H
    tau = tuple(int(t) for t in tau)
    if sorted(tau) != list(range(n)):
        raise ValueError(f"tau must be a permutation of range({n}), got {tau}")
    t = lambda i: 0 if i == 0 else tau[i - 1] + 1

    def placement(q: Array) -> Array:
        ell = a.placement(q)
        return jnp.stack([ell[tau[i]] for i in range(n)]) if n else ell

    def transfers(q: Array) -> TransferMatrix:
        T = a.transfers(q)
        return tuple(tuple(T[t(i)][t(j)] for j in range(n + 1)) for i in range(n + 1))

    def cost(q: Array, m_out: Array, n_in: Array) -> Array:
        if n == 0:
            return a.cost(q, m_out, n_in)
        ms = split_messages(m_out, n, dim_H)
        relabeled = [None] * n
        for i in range(n):
            relabeled[tau[i]] = ms[i]
        return a.cost(q, jnp.concatenate(relabeled), n_in)

    return replace(
        a, placement=placement, transfers=transfers, cost=cost,
        label=f"{a.label or 'a'}.{tau}",
    )


# ---------------------------------------------------------------------------
# The pure placement suboperad P_B <= A_B.
# ---------------------------------------------------------------------------


def placement_arrangement(
    n: int, dim_H: int, dim_P: int, placement: Placement,
    parameter: Optional[CotCoalg] = None, label: str = "placement",
) -> AttentionArrangement:
    """An operation of ``P_B``: ``T_{ij} = M`` for ``i != j`` and ``k = 0``.

    Its remaining data are the integrator and the placement map, composed by
    eq:composite-state. Even with zero cost it can receive position covectors from
    the components placed inside it.
    """
    return attention_arrangement(
        n, dim_H, dim_P, placement, identity_transfers(n, dim_H),
        cost=zero_cost, parameter=parameter, label=label,
    )


def forget_to_placement(a: AttentionArrangement) -> AttentionArrangement:
    """``A_B -> P_B``: forget the transfers and the cost, then supply the identity
    transfers and zero cost. Composed with ``P_B -> A_B`` this is the retraction of
    the paper's display, and it is the identity on ``P_B``."""
    return replace(
        a, transfers=identity_transfers(a.n, a.dim_H), cost=zero_cost,
        label=f"placement({a.label})" if a.label else "placement",
    )


def is_pure_placement(
    a: AttentionArrangement, q: Array, interactions: Sequence[Tuple[Array, Array]] = (),
) -> bool:
    """Is ``a`` in ``P_B`` -- identity transfers off the diagonal, zero cost?

    The transfers are checked at ``q``, the cost at the supplied interactions
    ``(m, e)``: a check at sample points, not a proof.
    """
    T = a.transfers(q)
    I = identity_transfer(a.dim_H)
    for i in range(a.n + 1):
        for j in range(a.n + 1):
            if i == j:
                if not is_zero_transfer(T[i][j]):
                    return False
            elif not (bool(jnp.all(T[i][j].gamma == I.gamma)) and bool(jnp.all(T[i][j].L == I.L))):
                return False
    return all(float(a.cost(q, m, e)) == 0.0 for m, e in interactions)


# ---------------------------------------------------------------------------
# Spatial attention (eq:spatial-state, eq:spatial-transfers, eq:spatial-mean).
# ---------------------------------------------------------------------------


def spatial_Q_dim(n: int, dim_H: int, dim_P: int) -> int:
    """``dim(P^n x End(H)^n)`` (eq:spatial-state)."""
    return n * dim_P + n * dim_H * dim_H


def spatial_config(positions, values) -> Array:
    """A configuration ``q = (p_1, ..., p_n, V_1, ..., V_n)`` of ``Q_n``, flattened."""
    return jnp.concatenate(
        [jnp.ravel(jnp.asarray(positions, float)), jnp.ravel(jnp.asarray(values, float))]
    )


def spatial_split(q: Array, n: int, dim_H: int, dim_P: int) -> Tuple[Array, Array]:
    """``q |-> (positions (n, dim_P), value maps (n, dim_H, dim_H))``."""
    k = n * dim_P
    return jnp.reshape(q[:k], (n, dim_P)), jnp.reshape(q[k:], (n, dim_H, dim_H))


def spatial_placement(n: int, dim_H: int, dim_P: int) -> Placement:
    """``l_i(q) = p_i`` (eq:spatial-state)."""
    return lambda q: spatial_split(q, n, dim_H, dim_P)[0]


PEAK_EPS = 1e-3


def spatial_kappa(ell_i: Array, ell_j: Array, tau: float, kernel: str = "bell") -> Array:
    """``kappa_ij = exp(-||l_i - l_j||^2 / 2 tau)`` (eq:spatial-transfers), the bell.

    ``kernel="peak"`` (book plan P43 (b), the listener page, de71554a): ``exp(-(sqrt(d^2 + eps^2) - eps) / tau)``,
    ``d = ||l_i - l_j||``, ``eps = PEAK_EPS``: the peak ``exp(-d / tau)`` made smooth at the speaker. Its
    exponent differs from ``-d/tau`` by ``(eps + O(eps^2 / d))/tau`` away from zero.
    The subtraction normalizes the weight to one at coincidence; its positional slope there is still zero."""
    d2 = jnp.sum((ell_i - ell_j) ** 2)
    if kernel == "peak":
        return jnp.exp(-(jnp.sqrt(d2 + PEAK_EPS ** 2) - PEAK_EPS) / tau)
    return jnp.exp(-d2 / (2.0 * tau))


def _spatial_frame(q: Array, n: int, dim_H: int, dim_P: int) -> Tuple[Array, Array]:
    """Positions and value maps indexed ``0..n``, with ``l_0 = 0_P`` and ``V_0 = H``."""
    pos, V = spatial_split(q, n, dim_H, dim_P)
    ell = jnp.concatenate([jnp.zeros((1, dim_P)), pos])
    Vs = jnp.concatenate([jnp.eye(dim_H)[None, :, :], V])
    return ell, Vs


def spatial_transfers(n: int, dim_H: int, dim_P: int, tau: float, kernel: str = "bell") -> Transfers:
    """``T_{ij}(q)(h, w) = (kappa_ij V_j h, kappa_ij w)`` (eq:spatial-transfers).

    A message transfer with ``gamma = kappa_ij > 0`` and ``L = kappa_ij V_j``.
    Nearby components exert greater influence on one another's input; the kernel
    uses relative positions, with the conventions ``l_0 = 0_P`` and ``V_0 = H``.
    """

    def T(q: Array) -> TransferMatrix:
        ell, Vs = _spatial_frame(q, n, dim_H, dim_P)
        rows = []
        for i in range(n + 1):
            row = []
            for j in range(n + 1):
                if i == j:
                    row.append(zero_transfer(dim_H))
                else:
                    k = spatial_kappa(ell[i], ell[j], tau, kernel)
                    row.append(Transfer(gamma=k, L=k * Vs[j]))
            rows.append(tuple(row))
        return tuple(rows)

    return T


def spatial_arrangement(
    n: int, dim_H: int, dim_P: int, tau: float, cost: Cost = zero_cost,
    parameter: Optional[CotCoalg] = None, init: Optional[Array] = None,
    mobility: float = 1.0, label: str = "spatial", kernel: str = "bell",
) -> AttentionArrangement:
    """An elementary spatial-attention arrangement over ``Q_n = P^n x End(H)^n``.

    With no ``parameter`` given, the integrator is the configuration coalgebra with
    Euclidean response ``mobility * I`` on ``Q_n``.
    """
    if parameter is None:
        parameter = configuration(euclidean(spatial_Q_dim(n, dim_H, dim_P), mobility), init=init)
    return attention_arrangement(
        n, dim_H, dim_P,
        spatial_placement(n, dim_H, dim_P), spatial_transfers(n, dim_H, dim_P, tau, kernel),
        cost=cost, parameter=parameter, label=label,
    )


def spatial_mean(
    n: int, dim_H: int, dim_P: int, tau: float, q: Array, messages: Array, i: int,
) -> Array:
    """The mean readout of receiver ``i``'s input in the spatial family (eq:spatial-mean):

        hbar_i = sum_{j in J_i} kappa_ij w_j V_j h_j / sum_{j in J_i} kappa_ij w_j,

    weighted spatial softmax with scores ``log w_j - ||l_i - l_j||^2 / 2 tau``. Here
    ``messages`` is the ``(n+1, dim_H+1)`` array of available messages, row ``0``
    being the exterior message ``m_0 = (w_0 h_0, w_0)``, and ``J_0 = {1..n}``,
    ``J_i = {0..n} \\ {i}`` for ``i > 0``. Admissibility is asserted: the paper's
    hypothesis is ``w_j > 0``, and the readout needs positive total weight.
    """
    ell, Vs = _spatial_frame(q, n, dim_H, dim_P)
    J = [j for j in range(n + 1) if j != i]   # J_0 = {1..n}; J_i = {0..n} \ {i}
    num = jnp.zeros(dim_H)
    den = jnp.asarray(0.0)
    for j in J:
        assert_admissible(messages[j], f"spatial_mean slot {j}")
        k = spatial_kappa(ell[i], ell[j], tau)
        num = num + k * (Vs[j] @ signal(messages[j]))   # kappa_ij w_j V_j h_j
        den = den + k * weight(messages[j])
    return mean(message(num, den))


# ---------------------------------------------------------------------------
# Private prediction (eq:prediction).
# ---------------------------------------------------------------------------


Predictor = Callable[[Array, Array], Array]  # (q, m_out) |-> e_hat in E


def prediction_cost(predictor: Predictor) -> Cost:
    """``k(q, m, e) = (1/2) ||e - pi(q, m)||^2`` (eq:prediction).

    The norm is the product Euclidean norm on ``E = M x P``. The predictor must be
    smooth and affine in ``m`` at each ``q``, and introduces no additional
    configuration factor; its dependence on ``q`` is differentiated. A prediction
    concerns the *weighted* input itself -- applying ``mean`` inside the cost would
    generally leave the quadratic family.
    """

    def k(q: Array, m_out: Array, n_in: Array) -> Array:
        d = n_in - predictor(q, m_out)
        return 0.5 * jnp.sum(d * d)

    return k


def fixed_prediction(e_hat: Array) -> Predictor:
    """The predictor ``pi(q, m) = ehat``, a prediction held fixed."""
    e_hat = jnp.asarray(e_hat, float)
    return lambda q, m_out: e_hat
