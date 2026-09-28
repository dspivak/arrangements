"""P29 stage 4: learned harm sent forward to an insurer, with ablations.

Prediction 4 in the pre-code design. Reasonable: earlier data can change a priced
choice before settlement. Artificial: known recovery, noiseless observations,
correct one-parameter model, and automatic adoption of her estimate. No modeling
of understanding a book, constructing a wire, litigation or ethical care. A body
or cell may share the feedback mechanism; no money/court interpretation is made.

Result (2026-09-28): all 4 checks passed. Harm falls 55.2813% over 30 units
and 55.1688% at half dt. Her quote reaches 1 at t=1.56 before first claim t=6.04.
Ablation, nonzero-observation secrecy check and retroactivity comparison pass. Prediction 4 held.
Positive final cash .143891 is below unpaid claims .672598: no solvency result.
Full results: notes/P29-TOWN-RESULTS.md.
"""
from dataclasses import replace
import json
import jax
import jax.numpy as jnp
import numpy as np
from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from arrangements.town import EMPTY, Settings, build, own_cost, run, summary
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter
from tests.town_checks import check_money


def test_her_account_precedes_claims_and_reduces_harm_at_two_step_sizes():
    reports = []
    for dt in (.02, .01):
        c = Settings(dt=dt, delay_steps=round(6/dt))
        historical, account = build(3, c), build(4, c)
        hist_run, account_run = run(historical), run(account)
        before, after = summary(historical, hist_run), summary(account, account_run)
        check_money(account, account_run)
        assert after['estimate_reaches_1'] < 6
        assert after['premium_reaches_1'] < 6
        assert after['harm'] < .9*before['harm']
        reports.append({'dt': dt, 'historical': before, 'account': after,
                        'harm_reduction': 1-after['harm']/before['harm']})
    print(json.dumps(reports, sort_keys=True))


def test_disconnected_account_leaves_every_existing_coordinate_unchanged():
    historical = build(3, Settings(delay_steps=15))
    disconnected = build(4, Settings(delay_steps=15), connect_report=False)
    a, b = run(historical, 3), run(disconnected, 3)
    for name in historical.slots:
        np.testing.assert_allclose(historical.value(a, name), disconnected.value(b, name), atol=1e-12)


def test_learner_receives_no_secret_coefficient_and_no_exposure_freezes_wrong_fit():
    town = build(4, Settings(delay_steps=3))
    other = build(4, replace(town.settings, beta=19))
    q = town.initial().at[town.slots['estimate']].set(.5)
    q = q.at[town.slots['memory']].set(jnp.array([.8, .2, 1.]))
    q = q.at[town.slots['health']].set(.2+town.settings.dt*(2*.8**2/2-.2))
    assert float(town.field(q)[town.slots['estimate']][0]) > .1
    # Same observed state, different hidden beta: learning sees no difference.
    np.testing.assert_allclose(town.field(q)[town.slots['estimate']],
                               other.field(q)[other.slots['estimate']], atol=0)
    q = q.at[town.slots['output']].set(0).at[town.slots['health']].set(0)
    q = q.at[town.slots['memory']].set(jnp.array([0., 0., 1.]))
    np.testing.assert_allclose(town.field(q)[town.slots['estimate']], 0, atol=0)


def test_exact_downstream_reading_changes_a_fit_while_one_way_reading_does_not():
    # Same learner primitive, now substituted into a nontrivial reader. This
    # isolates the extra input covector that the town intentionally does not use.
    learner = own_cost('learner', [.4], 2, lambda q, e: .5*(q[0]*e[0]-e[1])**2, .1)
    def reader(exact_read):
        potential = lambda q, m, e: -.5*(m[0]-2)**2
        return Arrangement(P=trivial_parameter(), in_dim_M=2, out_dim_M=1,
                           in_dim_N=0, out_dim_N=0,
                           out_f=lambda q,m: EMPTY,
                           in_f=lambda q,m,e: jnp.array([1., .8]),
                           omega=exact(potential) if exact_read else
                           lambda q,m,e: (EMPTY, jnp.zeros(1), EMPTY))
    one_way = compose_seq(learner, reader(False))
    mutual = compose_seq(learner, reader(True))
    q = jnp.array([.4])
    np.testing.assert_allclose(one_way.omega(q, EMPTY, EMPTY)[0], [.4], atol=1e-12)
    np.testing.assert_allclose(mutual.omega(q, EMPTY, EMPTY)[0], [2.0], atol=1e-12)
