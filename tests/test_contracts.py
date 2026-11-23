"""P24, contracts before the experiments (Codex, 193def97, 2026-10-10).

Predictions fixed before implementing contracts.py:
1. A ten-unit output allocated six/four, valued at two/three and paid at
   fractions one-half/one-third, gives benefits twelve/twelve, wages six/four,
   and net benefits six/eight. The worker receives ten; supplier debits total ten.
2. An allocation of zero to A gives A no output, benefit, payment or sale receipt,
   irrespective of B's valuation and payment. Doubling B's valuation changes B's
   exchange only. Fractions cannot allocate the same divisible output twice.
3. The fund's step is sale receipts minus wages, overhead and leakage, even when
   the workers' need grows; their want cannot create cash through a derivative.
4. At the old experiments' parameter settings, removing the original fund's
   claim on outside-supported product will lower its recovery margin. The old
   floor-worker result, recovery after every drain through 80, will fail.
   This is a prediction, not an assertion that the revised contracts preserve
   every feature of the earlier production model. In particular the need floor
   now applies to total wages; supplier payments buy attributed output.
5. With worker counts two/four/eight and total wage-rate cap four, the old rise
   of balance with worker count will be reduced by allocating output. It need
   not vanish: capacity, production and overhead still matter.

1--3 are implementation checks, built into the model. 4--5 can fail. Report both
the historical model and a matched control differing only in the original
fund's receipt rule. Drains are new starting balances chosen by the experimenter.
All updates must run as a composition of boxes through Phi, with the compiled
closed step checked against Phi. No evidence about arbitrary societies/cells.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from arrangements.contracts import contract, fixed_exchange, piecework, runner
from arrangements.functors import closed_step


def cash(arr, s):
    return np.asarray(arr.P.position(s))


def total_cash(model, s):
    """Every cash stock of a piecework model, summed (read by name: the configuration is worker by worker)."""
    r = model.read(s)
    return float(r['fund'] + np.sum(r['worker_cash']) + r['outside_cash'] + r['market_cash'] + r['overhead_cash'])


def labels(a, out=None):
    """Every label in ``a``'s construction (its boxes and parts, at every depth)."""
    out = set() if out is None else out
    out.add(a.label)
    for x in (a.construction or ())[1:]:
        labels(x, out)
    return out


def test_agreed_output_benefits_and_conserved_payments():
    c = contract([.6, .4], [2., 3.], [.5, 1/3])
    np.testing.assert_allclose(c.out_f(jnp.zeros(0), jnp.array([10.])),
                               [6, 4, 12, 12, 6, 4, 6, 8], atol=1e-13)
    arr = fixed_exchange(10., [.6, .4], [2., 3.], [.5, 1/3],
                         balances=[100., 100., 0.])
    step = closed_step(arr, check=2)
    s = step(arr.P.init)
    np.testing.assert_allclose(cash(arr, s), [94., 96., 10.], atol=1e-13)
    np.testing.assert_allclose(cash(arr, s).sum(), 200., atol=1e-13)
    assert 'contract' in arr.label and 'payment:0->worker' in arr.label


def test_outside_contract_cannot_pay_original_supplier():
    for outside_value in (3., 6.):
        arr = fixed_exchange(10., [0., 1.], [2., outside_value], [.5, 1/3],
                             balances=[100., 100., 0.])
        s = closed_step(arr, check=1)(arr.P.init)
        np.testing.assert_allclose(cash(arr, s),
                                   [100., 100.-10*outside_value/3, 10*outside_value/3])


@pytest.mark.parametrize('shares,values,fractions', [
    ([.6, .6], [2, 3], [.5, .5]), ([-.1, 1.1], [2, 3], [.5, .5]),
    ([.6, .4], [-2, 3], [.5, .5]), ([.6, .4], [2, 3], [1.1, .5]),
    ([.6, .4], [2, 3], [float('nan'), .5]),
])
def test_invalid_contract_cannot_duplicate_output(shares, values, fractions):
    with pytest.raises(ValueError):
        contract(shares, values, fractions)


def test_piecework_cash_is_transfers_not_the_workers_want():
    rows = []
    for need in (.3, 5.):
        model = piecework(1, needs=[need], alpha=.5, value=4., source_position=1.5)
        arr = model.arr
        before = model.read(arr.P.init)
        step = closed_step(arr, check=1)
        after = model.read(step(arr.P.init))
        settled = np.asarray(before['contracts'])[0]
        expected = .05*(4*settled[0] - settled[4] - 2. - .1*20.)
        np.testing.assert_allclose(after['fund'] - before['fund'], expected, atol=1e-12)
        np.testing.assert_allclose(total_cash(model, step(arr.P.init)), total_cash(model, arr.P.init), atol=1e-9)
        rows.append(float(after['fund']))
        assert {'piecework:contracts', 'payment:1->worker:0'} <= labels(arr)
    np.testing.assert_allclose(rows[0], rows[1], atol=1e-13)


def test_piecework_short_run_keeps_allocations_and_cash_consistent():
    model = piecework(2, source_position=1.5, value=4., alpha=.5,
                     needs=[.3, 1.2], a0=.4, overhead=1.)
    step = closed_step(model.arr, check=1)
    s = jax.jit(lambda s: jax.lax.fori_loop(0, 100, lambda i, x: step(x), s))(model.arr.P.init)
    r = model.read(s)
    output, rows = np.asarray(r['amplitudes']), np.asarray(r['contracts'])
    assert np.isfinite(cash(model.arr, s)).all()
    assert (output >= 0).all()
    assert np.all(rows[:, :2].sum(axis=1) <= output + 1e-12)
    np.testing.assert_allclose(total_cash(model, s), total_cash(model, model.arr.P.init), atol=1e-8)


def test_unreachable_or_nonpaying_supplier_cannot_claim_output():
    near = piecework(1, source=False, p0=0., a0=1.)
    far = piecework(1, source=False, p0=10., a0=1.)
    near_output = float(near.read(near.arr.P.init)['contracts'][0, 0])
    far_output = float(far.read(far.arr.P.init)['contracts'][0, 0])
    assert near_output > .8 and far_output < 1e-20
    unpaid = piecework(1, wmax=0., a0=1.)
    row = np.asarray(unpaid.read(unpaid.arr.P.init)['contracts'])[0]
    np.testing.assert_allclose(row[[0, 2, 4, 6]], 0., atol=0.)
    assert row[1] > 0 and row[5] > 0


@pytest.mark.parametrize('n', [1, 2, 8])
def test_piecework_well_factored_nested_equals_flat(n):
    """Book task well-factored-2ac0e890 (David, 2026-11-23: "good factorizations should have 2-7 boxes
    inside"): a box per worker (effort, placement, cash) and a box of its flows, the workers seven to a
    group, every box holding at most seven; the nested composite's own maps (``fast=False``) equal the
    built composite's (the flat maps) at points along a run and over closed steps."""
    from arrangements.nesting import box_counts, carry_state
    kw = dict(needs=[.3 + .1 * j for j in range(n)], value=4., alpha=.5, source_position=1.5)
    built, nested = piecework(n, **kw), piecework(n, fast=False, **kw)
    counts = box_counts(built.arr)
    assert counts == box_counts(nested.arr) and all(k <= 7 for _, k in counts), counts
    assert ('piecework:contracts', 5) in counts and counts[0] == ('(top level)', 3)
    step_b, step_n = closed_step(built.arr, check=1), closed_step(nested.arr, check=1)
    s_b, s_n = built.arr.P.init, carry_state(built.arr.P.init, nested.arr.P.init)
    e = jnp.zeros(0)
    for _ in range(10):
        q = built.arr.P.position(s_b)
        np.testing.assert_allclose(np.asarray(built.arr.omega(q, e, e)[0]),
                                   np.asarray(nested.arr.omega(q, e, e)[0]), atol=1e-12)
        s_b, s_n = step_b(s_b), step_n(s_n)
    np.testing.assert_allclose(cash(built.arr, s_b), cash(nested.arr, s_n), atol=1e-12)
