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
        old_cash = cash(arr, arr.P.init)[2:]
        new_cash = cash(arr, step(arr.P.init))[2:]
        np.testing.assert_allclose(new_cash.sum(), old_cash.sum(), atol=1e-9)
        rows.append(float(after['fund']))
        assert 'piecework:contracts' in arr.label and 'payment:1->worker:0' in arr.label
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
    np.testing.assert_allclose(cash(model.arr, s)[4:].sum(),
                               cash(model.arr, model.arr.P.init)[4:].sum(), atol=1e-8)


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
