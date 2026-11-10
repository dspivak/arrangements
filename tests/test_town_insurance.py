"""The same town without her account (book Chapter 1, ``sec.check_claims``; ``sec.timely_costs`` tells the town with it).

For a reader of the book. ``eve_town(3)`` has the insurer and no estimator. The insurer reads only what a court has
fitted from paid claims, and a claim is paid some time after the smoke that caused it (``delay_steps``), so a quote
cannot lead evidence it has not received. Run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    from arrangements.town import eve_town, run, summary
    town = eve_town(3)
    r = summary(town, run(town))
    print(round(r['assessed_reaches_1'], 2), round(r['premium_reaches_1'], 2), round(r['harm'], 2))
    # 7.06 7.48 6.13

The court's fit reaches 1 at ``t = 7.06`` and the premium at ``t = 7.48``, after the first claim at ``t = 6.04``. With
her account the premium reaches 1 at ``t = 2.24`` (``tests/test_town_account.py``), and the harm over the run is about a
sixth of this.

What the tests check. ``test_book_the_same_town_without_her_account``: the premium is zero until the first claim, and
reaches 1 after it where hers reaches 1 before; the harm is more than five times the harm with her account, and both
towns end up clean. The others, on the plant of one dial: the quote alone moves as its declared response says, it lags
its evidence, and the money balances.

Artificial: quadratic policy pricing, no reserves, no competition, exclusions or default. A negative fund means borrowed
money, not risk borne well.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``; results in ``notes/P29-TOWN-RESULTS.md``.
"""
import json
import jax.numpy as jnp
import numpy as np
from arrangements.town import Settings, account_town, build, eve_town, run, summary
from tests.town_checks import check_money


def test_book_the_same_town_without_her_account():
    """``sec.check_claims``: "the same town without her account". The two-dial plant and the townspeople of
    ``account_town``, the insurer reading only paid claims through a court's fit: its premium stays at zero
    through the claims delay and climbs after the first claim, and the harm is several times the harm with
    her account, though both towns end up clean."""
    town, drawn = eve_town(3), account_town()
    a, b = run(town), run(drawn)
    premium = town.value(a, 'premium')[:, 0]
    claim = np.flatnonzero(np.diff(town.value(a, 'money')[:, 2]) > 1e-12)[0]+1
    assert np.max(np.abs(premium[:claim])) < 1e-12
    without, with_account = summary(town, a), summary(drawn, b)
    assert without['premium_reaches_1'] > claim*town.settings.dt > with_account['premium_reaches_1']
    assert without['harm'] > 5*with_account['harm']
    assert without['final_cleanliness'] > .9 and with_account['final_cleanliness'] > .9


def test_isolated_quote_has_the_declared_geometric_response():
    town = build(3, Settings(delay_steps=2))
    q = town.initial().at[town.slots['assessed']].set(2.)
    price = town.slots['premium']
    # All other published values held fixed: the tested box reads a fixed target.
    for n in range(20):
        np.testing.assert_allclose(q[price][0], 2*(1-.96**n), atol=1e-12)
        force = town.field(q)
        q = q.at[price].add(town.settings.dt*town.settings.quote_rate*force[price])


def test_historical_quote_lags_its_evidence_and_fund_balances():
    town = build(3)
    trajectory = run(town)
    result = summary(town, trajectory)
    print(json.dumps(result, sort_keys=True))
    assert result['premium_reaches_1'] > result['assessed_reaches_1']
    premium = town.value(trajectory, 'premium')[:, 0]
    assert np.max(np.abs(premium[:town.settings.delay_steps+2])) < 1e-12
    check_money(town, trajectory)
