"""P29 stage 3: an insurer reads only past claims through their assessor.

Prediction 3 in the pre-code design. Reasonable: a quote cannot lead evidence it
has not received. Artificial: quadratic policy pricing, no reserves requirement,
no competition, exclusions or default. A negative fund means borrowed money,
not successful risk bearing. No quarter-turn claim from the older insurer test.

Result (2026-09-28): both checks passed. Quote reaches 1 at t=7.48 after
assessor t=7.06; harm 8.681451. Fund ends -5.244528 under unlimited credit. Prediction 3 held.
Full results: notes/P29-TOWN-RESULTS.md.
"""
import json
import jax.numpy as jnp
import numpy as np
from arrangements.town import Settings, build, run, summary
from tests.town_checks import check_money


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
