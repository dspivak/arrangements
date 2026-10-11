"""Contracted output and conserved payments (book project P24).

A contract allocates a divisible output once. Benefit is value per output unit
times received output; payment is a specified fraction of that benefit. Benefit
is not a cash receipt. The fund experiment separately sells allocated output.

The experiment's worker/supplier matching rule is an explicit model choice:
output claims are spatial accessibility times the offered unit payment, with a
fixed unsold claim for the worker's retained output. This is a continuously renewed piecework contract, not a universal
theory of contracts. The fund's offer falls with its available balance; the
other supplier's offer is maintained externally. Workers prefer wages and pay
a quadratic effort cost. A floor worker needs total wages above its need.

Every state, contract settlement, response and cash transfer is a box; the
whole is composed by wiring.py and run by functors.closed_step (checked by Phi).
Cash transfers force only their two account endpoints. A worker's desire does
not act on a supplier's cash balance. Overhead may put the fund into debt; a
bankruptcy boundary and finite resources of the external market are not modeled.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, zero_form
from .cascade import gate
from .coalgebra import configuration
from .rvect import trivial
from .stock_flow import stock_box, flow_box, _wire
from .wiring import compose_seq, tensor_arrangements
from .functors import closed_step

EMPTY = jnp.zeros(0)


def _static(out_dim_M, out_dim_N, fn, label):
    return Arrangement(configuration(trivial()), 0, out_dim_M, 0, out_dim_N,
                       lambda q, m: fn(m), lambda q, m, n: EMPTY,
                       zero_form, label)


def _terms(values, fractions):
    values, fractions = np.asarray(values, float), np.asarray(fractions, float)
    if (values.ndim != 1 or values.size == 0 or values.shape != fractions.shape
            or not np.all(np.isfinite(values)) or not np.all(np.isfinite(fractions))
            or np.any(values < 0) or np.any((fractions < 0) | (fractions > 1))):
        raise ValueError('values must be nonnegative and payment fractions in [0, 1]')
    return jnp.asarray(values), jnp.asarray(fractions)


def settlement(output, shares, values, fractions):
    """Flat array: allocations, gross benefits, payments, net benefits.

    This smooth map's domain has output >= 0, shares >= 0, sum(shares) <= 1.
    The public fixed-contract constructor checks its constant terms.
    """
    quantities = output * shares
    benefits = values * quantities
    payments = fractions * benefits
    return jnp.concatenate([quantities, benefits, payments, benefits - payments])


def contract(shares, values, payment_fractions, label='contract'):
    """A static box receiving one output amount and reporting its settlement."""
    values, fractions = _terms(values, payment_fractions)
    shares = np.asarray(shares, float)
    if (shares.shape != values.shape or not np.all(np.isfinite(shares))
            or np.any(shares < 0) or shares.sum() > 1 + 1e-12):
        raise ValueError('output shares must be nonnegative and total at most one')
    shares = jnp.asarray(shares)
    return _static(1, 4 * len(values),
                   lambda m: settlement(m[0], shares, values, fractions), label)


def fixed_exchange(output, shares, values, payment_fractions, *, balances=None, dt=1.):
    """One worker paid by several suppliers: no sales and no creation of cash.

    Output is a constant offered each round; states are supplier balances then
    the worker's balance. The contract and each supplier's payment are boxes.
    """
    c = contract(shares, values, payment_fractions)
    if not np.isfinite(output) or output < 0 or not np.isfinite(dt) or dt < 0:
        raise ValueError('output and dt must be finite and nonnegative')
    count = len(values)
    balances = np.zeros(count + 1) if balances is None else np.asarray(balances, float)
    if balances.shape != (count + 1,) or not np.all(np.isfinite(balances)):
        raise ValueError('one balance per supplier and one for the worker required')
    accounts = tensor_arrangements([
        replace(stock_box(x, dt), label=f'cash:{i}') for i, x in enumerate(balances)])
    output_box = _static(0, 1, lambda m: jnp.array([output]), 'worker:output')
    exchange = compose_seq(output_box, c)
    both = tensor_arrangements([accounts, exchange])
    dim = count + 1 + 4 * count
    flows = [replace(flow_box(lambda m, i=i: m[count + 1 + 2 * count + i],
                             dim, i, count), label=f'payment:{i}->worker')
             for i in range(count)]
    return compose_seq(compose_seq(both, _wire(list(range(dim)) * count, dim)),
                       tensor_arrangements(flows))


@dataclass(frozen=True)
class ContractRun:
    arr: Arrangement
    n: int
    settlement_box: Arrangement
    stock_dim: int

    def read(self, state):
        q = self.arr.P.position(state)
        m = self.settlement_box.out_f(EMPTY, q)
        n = self.n
        return dict(amplitudes=q[:n], positions=q[n:2*n], fund=q[2*n],
                    worker_cash=q[2*n+1:3*n+1], outside_cash=q[3*n+1],
                    market_cash=q[3*n+2], overhead_cash=q[3*n+3],
                    contracts=m[self.stock_dim:].reshape(n, 8))


def piecework(n, *, source=True, wmax=1., tau=.8, source_position=2.5,
              source_rate=1., value=2., outside_value=2., f0=20.,
              overhead=2., leak=.1, alpha=1., beta=1., needs=None, lam=.3,
              a0=.8, p0=.06, mu_a=.1, mu_p=.1, mu_f=.05,
              legacy_receipts=False, unsold_claim=.1):
    """Workers, two suppliers, contracts, a market and cash accounts.

    The historical receipt control credits the fund with the old spatially
    weighted product, while leaving contracts, payments and responses the same.
    It deliberately duplicates claims and is only a diagnostic comparison.
    Payment rates are gate(f, wmax, 1) and source_rate. Their ratios to the two
    valuations are the contract's payment fractions; rates never exceed values.
    The fund's valuation is its sale price in this experiment. The external
    supplier consumes its output and pays from a recorded external balance.
    """
    if (not isinstance(n, int) or n < 1 or tau <= 0 or value <= 0
            or outside_value <= 0 or unsold_claim <= 0 or not 0 <= wmax <= value
            or not 0 <= source_rate <= outside_value):
        raise ValueError('positive dimensions/scales; payment caps cannot exceed values')
    if any(not np.isfinite(x) for x in [tau, value, outside_value, wmax,
            source_rate, source_position, f0, overhead, leak, alpha, beta, lam,
            a0, p0, mu_a, mu_p, mu_f, unsold_claim]):
        raise ValueError('parameters must be finite')
    if min(overhead, leak, alpha, beta, lam, a0, mu_a, mu_p, mu_f) < 0:
        raise ValueError('costs, initial production and mobilities must be nonnegative')
    if needs is not None and (len(needs) != n or not np.all(np.isfinite(needs))
                              or np.any(np.asarray(needs) < 0)):
        raise ValueError('one finite nonnegative need per worker required')
    stock_dim = 3*n + 4
    fund_idx, outside_idx, market_idx, overhead_idx = 2*n, 3*n+1, 3*n+2, 3*n+3

    def offered(f):
        return gate(f, wmax, 1.)

    def shares_rates(p, f):
        rates = jnp.array([offered(f), source_rate if source else 0.])
        reaches = jnp.exp(-jnp.array([p*p, (p-source_position)**2]) / (2*tau))
        claims = reaches * rates
        shares = claims / (jnp.sum(claims) + unsold_claim)
        return shares, rates

    def local_cost(a, p, f, need):
        shares, rates = shares_rates(p, f)
        rate = jnp.dot(shares, rates)
        income = a * rate
        cost = .5*a*a - alpha*income
        if needs is None:
            return cost - beta*rate
        # log(softplus) evaluated stably far below the need.
        x = (income - need) / .05
        log_supply = jnp.log(.05) + jnp.where(
            x < -30., x, jnp.log(jax.nn.softplus(jnp.maximum(x, -30.))))
        return cost - lam*log_supply

    response = jax.grad(local_cost, argnums=(0, 1))
    values = jnp.array([value, outside_value])

    def settled(q):
        rows = []
        for j in range(n):
            shares, rates = shares_rates(q[n+j], q[fund_idx])
            rows.append(settlement(q[j], shares, values, rates / values))
        return jnp.concatenate([q, *rows])

    settlement_box = _static(stock_dim, stock_dim+8*n, settled, 'piecework:contracts')
    levels = [a0]*n + [p0]*n + [f0] + [0.]*n + [1e6, 1e6, 0.]
    mobilities = [mu_a]*n + [mu_p]*n + [mu_f]*(n+4)
    names = ([f'worker:{j}:effort' for j in range(n)]
             + [f'worker:{j}:placement' for j in range(n)] + ['fund:cash']
             + [f'worker:{j}:cash' for j in range(n)]
             + ['outside:cash', 'market:cash', 'overhead:cash'])
    stocks = tensor_arrangements([replace(stock_box(x, dt), label=label)
        for x, dt, label in zip(levels, mobilities, names)])
    arranged = compose_seq(stocks, settlement_box)
    dim = stock_dim + 8*n
    boxes = []
    for j in range(n):
        need = 0. if needs is None else float(needs[j])
        def worker_force(q, m, e, j=j, need=need):
            da, dp = response(m[j], m[n+j], m[fund_idx], need)
            force = jnp.zeros(dim).at[j].set(-da).at[n+j].set(-dp)
            return EMPTY, force, EMPTY
        boxes.append(Arrangement(configuration(trivial()), 0, dim, 0, 0,
            lambda q, m: EMPTY, lambda q, m, e: EMPTY, worker_force,
            f'worker:{j}:wages-and-effort'))
        off = stock_dim+8*j
        for supplier, payer in enumerate((fund_idx, outside_idx)):
            boxes.append(replace(flow_box(lambda m, k=off+4+supplier: m[k],
                dim, payer, 2*n+1+j), label=f'payment:{supplier}->worker:{j}'))
        def revenue(m, off=off, j=j):
            output = (jnp.exp(-m[n+j]**2/(2*tau))*m[j]
                      if legacy_receipts else m[off])
            return value * output
        boxes.append(replace(flow_box(revenue, dim, market_idx, fund_idx),
                             label=f'market:fund-output:{j}'))
    # The expense is nonnegative even when the fund is indebted; there is no
    # automatic income from the historical -leak*f term on a negative balance.
    boxes.append(replace(flow_box(
        lambda m: overhead + leak*.05*jax.nn.softplus(m[fund_idx]/.05),
        dim, fund_idx, overhead_idx), label='fund:overhead-and-leakage'))
    routed = compose_seq(arranged, _wire(list(range(dim))*len(boxes), dim))
    arr = compose_seq(routed, tensor_arrangements(boxes))
    return ContractRun(arr, n, settlement_box, stock_dim)


def runner(model, *, pre=4000, post=6000, initial=None):
    """State after pre steps, and continuation from an externally drained fund.

    The drain is an intervention between two runs, not an event the model makes.
    Only the fund state is changed; all subsequent operations run in the composite.
    """
    step = closed_step(model.arr, check=1)
    advance = jax.jit(lambda s: jax.lax.fori_loop(0, pre, lambda i, t: step(t), s))
    warmed = advance(model.arr.P.init if initial is None else initial)
    # The nonempty state leaves are exactly the named scalar stocks, in order.
    leaves, tree = jax.tree_util.tree_flatten(warmed)
    nonempty = [i for i, x in enumerate(leaves) if x.size]
    fund_leaf = nonempty[2*model.n]
    @jax.jit
    def after(drain):
        start = list(leaves)
        start[fund_leaf] = start[fund_leaf] - drain
        state = jax.tree_util.tree_unflatten(tree, start)
        return jax.lax.fori_loop(0, post, lambda i, t: step(t), state)
    return warmed, after
