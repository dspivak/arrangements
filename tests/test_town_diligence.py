"""The insurer's own check (book Chapter 1, ``sec.cost_later``: "She knows they'll do their own due diligence from
there, because it's their money on the line"; linked from the footnote in ``sec.check_claims``).

For a reader of the book. ``eve_town(4, diligence=True)`` gives the insurer its books, a record of the claims it has
paid and the exposure it has written, and its own fit of how much smoke drives coughing, by least squares over every
claim paid so far. Its rate moves from her estimate toward its own fit by the share of the exposure's evidence whose
claims have arrived. So at first it prices by her account, and as its own money comes back to it, by its own evidence.
These books are not in ``eqn.town_account``: there the insurer's state is one number; here its check is added beside it.

Run it, run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    from arrangements.town import eve_town, run, summary
    town = eve_town(4, diligence=True)
    r = summary(town, run(town))
    print(round(r['first_claim_time'], 2), round(r['final_own_fit'], 3), round(r['final_premium'], 3), round(r['final_estimate'], 3))
    # 6.04 2.0 2.0 1.674

The first claim is paid at ``t = 6.04``, the insurer's own fit is the true ``beta = 2`` from then on, and its rate ends
there. Her estimate in this fuller town, read through a memory box, stalls at 1.674: once the plant has cleaned there is
little smoke to learn from. So in this town the check corrects her.

What the tests check. Until the first claim the town is stage 4's, step for step. The fit on paid claims is ``beta``
exactly, on noiseless records, and the rate ends within 2% of it. The court is not repaired by this, and the money lost
early is not recouped. With her report disconnected the harm is no worse than without her account. And on the estimator
as drawn (``fit_coughing=True``), which needs no rescue, the rate still ends within 2% of ``beta``.

Artificial: noiseless records, known recovery, one coefficient, unlimited credit.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``; results in ``notes/P29-F-RESULTS-2026-09-29.md``.
"""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from arrangements.town import Settings, build, eve_town, run, summary
from tests.test_town_clean import check_clean_money

STAGE3_HARM = 6.133659076472214   # eve_town(3), measured before the edit
WINDOW = 300                       # six units, one claims delay, at dt = .02


@pytest.fixture(scope='module')
def towns():
    plain, diligent = eve_town(4), eve_town(4, diligence=True)
    return plain, run(plain), diligent, run(diligent)


def first_paid(town, trajectory):
    return int(np.flatnonzero(np.diff(town.value(trajectory, 'money')[:, 2]) > 0)[0] + 1)


def test_until_the_first_claim_the_town_is_stage_four(towns):
    """Predictions 1 and 8: identical to stage 4 through the step the first claim is paid (a
    discrete identity: the target is her estimate exactly while no claim is on the books);
    harm within 5% of stage 4's and no higher."""
    plain, a, diligent, b = towns
    n = first_paid(plain, a)
    assert n == first_paid(diligent, b)
    for name in plain.slots:
        np.testing.assert_allclose(plain.value(a[:n+1], name), diligent.value(b[:n+1], name),
                                   atol=1e-12, rtol=0)
    before, after = summary(plain, a), summary(diligent, b)
    print('first claim t =', n*plain.settings.dt, 'harm', before['harm'], after['harm'])
    assert after['harm'] == pytest.approx(before['harm'], rel=.05)
    assert after['harm'] <= before['harm']
    check_clean_money(diligent, b)


def test_claims_fit_is_exact_and_the_rate_ends_at_beta(towns):
    """Predictions 5 and 2: least squares over every paid claim equals beta from the first claim
    (exact on noiseless records; the one-step fit stalls); the rate ends within 2% of beta; the
    fund's change over the last claims delay is under 5% of stage 4's fall there."""
    plain, a, diligent, b = towns
    beta = diligent.settings.beta
    books = diligent.value(b, 'books')
    n = first_paid(diligent, b)
    assert np.all(books[:n, 1] == 0)
    np.testing.assert_allclose(books[n:, 0]/books[n:, 1], beta, rtol=1e-9)
    after = summary(diligent, b)
    assert after['final_premium'] == pytest.approx(beta, rel=.02)
    fall4 = plain.value(a, 'money')[-1, 1] - plain.value(a, 'money')[-1-WINDOW, 1]
    change = diligent.value(b, 'money')[-1, 1] - diligent.value(b, 'money')[-1-WINDOW, 1]
    print('final premium', after['final_premium'], 'fund change last window', change,
          'stage 4', fall4)
    assert fall4 < 0
    assert abs(change) < .05*abs(fall4)


def test_the_court_is_not_repaired_and_the_early_loss_is_not_recouped(towns):
    """Predictions 6 and 7: the court's one-step fit still stalls where stage 4's does (within
    2%); the insurer's final fund is within .05 of stage 4's, since it prices at beta going
    forward and never above it."""
    plain, a, diligent, b = towns
    before, after = summary(plain, a), summary(diligent, b)
    print('court', before['final_assessed'], after['final_assessed'],
          'fund', before['final_insurer_fund'], after['final_insurer_fund'])
    assert after['final_assessed'] == pytest.approx(before['final_assessed'], rel=.02)
    assert after['final_assessed'] < .9*diligent.settings.beta
    assert after['final_insurer_fund'] == pytest.approx(before['final_insurer_fund'], abs=.05)


def test_with_her_report_disconnected_harm_is_no_worse_than_stage_three():
    """Prediction 3: the insurer blends the court's fit with its own claims fit, both delayed;
    harm no worse than stage 3's 6.13. Stage 3 with diligence is the same town on every shared
    coordinate (her estimate box is then connected to nothing)."""
    town = eve_town(4, diligence=True, connect_report=False)
    trajectory = run(town)
    result = summary(town, trajectory)
    print('disconnected harm', result['harm'], 'final premium', result['final_premium'])
    assert result['harm'] <= STAGE3_HARM
    three = eve_town(3, diligence=True)
    other = run(three)
    for name in three.slots:
        np.testing.assert_allclose(three.value(other, name), town.value(trajectory, name),
                                   atol=1e-12, rtol=0)
    check_clean_money(town, trajectory)


def test_diligence_on_the_estimator_as_drawn():
    """The same books on the book's own estimator (``fit_coughing``, ``account_town``'s fit), whose estimate
    needs no rescue: the rate still ends within 2% of beta, and the insurer's own fit is beta exactly."""
    town = eve_town(4, diligence=True, fit_coughing=True)
    trajectory = run(town)
    result = summary(town, trajectory)
    print('drawn estimator: final premium', result['final_premium'], 'own fit', result['final_own_fit'])
    assert result['final_premium'] == pytest.approx(town.settings.beta, rel=.02)
    assert result['final_own_fit'] == pytest.approx(town.settings.beta, rel=1e-9)


def test_diligence_needs_an_insurer():
    """Construction check: no insurer before stage 3."""
    for stage in (1, 2):
        with pytest.raises(ValueError):
            build(stage, Settings(), diligence=True)
