"""P29 stage 2: delayed evidence, informed profit maximizer, conserved transfers.

Prediction 2 in the pre-code design. Reasonable: information and cash arrive at
different times. Artificial: all harm becomes a paid claim, credit is unlimited,
and the plant fits one coefficient from noise-free health records. This tests
transient excess output, not a general instability theorem for delayed lawsuits.
At other scales this is an observation delay; legal institutions have no mapping.

Result (2026-09-28, corrected initialization): both checks passed. Harm delayed
8.345398 versus prompt 3.855612 and informed 3.290836. Informed estimate stays exactly 2;
its trajectory is analytic. Prediction 2 held as transient overproduction, not oscillation.
Full results: notes/P29-TOWN-RESULTS.md.
"""
from dataclasses import replace
import json
import numpy as np
from arrangements.town import Settings, build, run, summary
from tests.town_checks import check_money


def test_court_receives_the_exact_dated_observation_and_nothing_early():
    c = Settings(delay_steps=7)
    town = build(2, c)
    trajectory = run(town, 1)
    delay = town.value(trajectory, 'delay')
    health = town.value(trajectory, 'coughing')[:, 0]
    memory = town.value(trajectory, 'memory')
    observed = np.column_stack([memory[:, 0]**2/2,
                               (health-memory[:, 1])/c.dt+c.recovery*memory[:, 1]])*memory[:, 2:3]
    np.testing.assert_allclose(delay[:c.delay_steps, -2:], 0, atol=0)
    np.testing.assert_allclose(delay[c.delay_steps:, -2:], observed[:-c.delay_steps], atol=1e-12)
    money = town.value(trajectory, 'money')
    assert np.max(np.abs(money[:c.delay_steps+2, 2])) < 1e-12
    check_money(town, trajectory)


def test_delay_increases_harm_and_peak_output_but_informed_plant_acts_early():
    c = Settings()
    delayed, prompt, informed = build(2, c), build(2, replace(c, delay_steps=0)), build(2, c, informed=True)
    result = {}
    for name, town in [('delayed', delayed), ('prompt', prompt), ('informed', informed)]:
        trajectory = run(town)
        result[name] = summary(town, trajectory)
        check_money(town, trajectory)
        if name == 'informed':
            np.testing.assert_allclose(town.value(trajectory, 'assessed'), c.beta, atol=1e-12)
            optimum = c.benefit/(1+c.beta)
            expected = optimum+(c.output0-optimum)*(1-c.dt*(1+c.beta))**np.arange(len(trajectory))
            np.testing.assert_allclose(town.value(trajectory, 'output')[:, 0], expected, atol=1e-12)
    print(json.dumps(result, sort_keys=True))
    assert result['delayed']['harm'] > result['prompt']['harm']
    assert result['delayed']['peak_output'] > result['prompt']['peak_output']
    assert result['informed']['harm'] < result['delayed']['harm']
    assert result['informed']['peak_output'] < .35
