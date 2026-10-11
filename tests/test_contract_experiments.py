"""P24 comparisons; predictions are fixed in test_contracts.py's docstring.

Each node is a separately capped experiment. P24_RESULTS selects the directory
where its machine-readable record is written before the next node is started.
The receipt control changes only the original fund's entitlement to product;
it is not the historical trajectory (that is rerun separately).
"""
import json
import os
from pathlib import Path

import numpy as np
import pytest

from arrangements.contracts import piecework, runner


def record(name, data):
    text = json.dumps(data, indent=2, allow_nan=False)
    if dest := os.environ.get('P24_RESULTS'):
        Path(dest, name + '.json').write_text(text + '\n')
    if 'warmup' in data:
        print(name, 'warmup fund', data['warmup']['fund'], 'drain / final fund',
              [(r['drain'], round(r['fund'], 5)) for r in data['rows']], flush=True)
    else:
        print(name, text, flush=True)


def summary(model, s):
    r = model.read(s)
    rows = np.asarray(r['contracts'])
    q = np.asarray(model.arr.P.position(s))
    assert np.isfinite(q).all()
    assert np.min(r['amplitudes']) >= -1e-10
    assert np.all(rows[:, :2].sum(axis=1) <= np.asarray(r['amplitudes']) + 1e-9)
    return dict(fund=float(r['fund']), positions=np.asarray(r['positions']).tolist(),
                amplitudes=np.asarray(r['amplitudes']).tolist(),
                fund_output=float(rows[:, 0].sum()), outside_output=float(rows[:, 1].sum()),
                fund_payments=float(rows[:, 4].sum()), outside_payments=float(rows[:, 5].sum()))


@pytest.mark.parametrize('n', [2, 4, 8])
@pytest.mark.parametrize('source', [True, False])
def test_cascade_worker_count(n, source):
    rows = []
    for legacy in (False, True):
        model = piecework(n, source=source, wmax=4/n, legacy_receipts=legacy)
        eq, after = runner(model)
        r = summary(model, eq)
        if r['fund'] > 0:
            lo, hi = 0., max(80., 3*r['fund'])
            # Check both endpoints before assuming a crossing exists.
            f_lo, f_hi = summary(model, after(lo))['fund'], summary(model, after(hi))['fund']
            if f_lo > 0 and f_hi < 0:
                for _ in range(16):
                    mid = (lo+hi)/2
                    if float(model.read(after(mid))['fund']) > 0:
                        lo = mid
                    else:
                        hi = mid
                threshold = (lo+hi)/2
            else:
                threshold = None
        else:
            threshold = None
        r.update(receipt_control=legacy, drain_threshold=threshold,
                 recovery_margin=None if threshold is None else threshold-r['fund'])
        rows.append(r)
    record(f'cascade-n{n}-source{int(source)}', dict(n=n, source=source, rows=rows))


@pytest.mark.parametrize('legacy', [False, True])
def test_account_drain(legacy):
    model = piecework(4, source_position=1.5, value=4., alpha=.5,
                     needs=[.3, .6, .9, 1.2], a0=.4, overhead=1.,
                     legacy_receipts=legacy)
    eq, after = runner(model)
    warmup = summary(model, eq)
    rows = []
    for drain in range(0, 81, 5):
        row = summary(model, after(float(drain)))
        row['drain'] = drain
        rows.append(row)
    record(f'account-control{int(legacy)}', dict(warmup=warmup, rows=rows))


def test_account_same_start():
    """Follow-up prediction fixed after v2's warmup comparison, before this run:
    the historical receipt rule rescues the fund even when both runs start from
    the contracted fund's full state and suffer the same drain of 35. This can
    fail; the earlier comparison alone cannot separate accumulated reserves
    from income during the crisis.
    """
    args = dict(source_position=1.5, value=4., alpha=.5, needs=[.3, .6, .9, 1.2],
                a0=.4, overhead=1.)
    contracted = piecework(4, **args)
    warm, after = runner(contracted)
    control = piecework(4, legacy_receipts=True, **args)
    _, other_after = runner(control, pre=0, initial=warm)
    rows = [summary(contracted, after(35.)), summary(control, other_after(35.))]
    record('account-same-start', dict(start=summary(contracted, warm), drain=35.,
                                    contracted=rows[0], receipt_control=rows[1]))
    # This assertion tests the new prediction, not merely whether the code ran.
    assert rows[0]['fund'] < 0 < rows[1]['fund']
