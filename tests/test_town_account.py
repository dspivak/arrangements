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
import pytest
from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from arrangements.functors import Phi
from arrangements.town import (
    EMPTY, Settings, account_boxes, account_town, build, eve_town, own_cost, run, smoke, summary)
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter
from tests.chain import IN_POS
from tests.town_checks import check_money


def test_book_the_town_is_the_four_boxes_she_draws():
    """``eqn.town_account``: the composite is those four boxes and their wires and nothing else, each box has
    the ports the book draws, the estimator's and the insurer's states are one number each, and only the
    power wire pushes back (``sec.better``, ``sec.cost_later``: the estimator "doesn't push anything back",
    the insurer "wouldn't push anything back")."""
    boxes = {b.label: b for b in account_boxes()}
    assert {name: (b.in_dim_N, b.out_dim_N, b.P.Q_dim) for name, b in boxes.items()} == {
        'plant': (1, 2, 2), 'townspeople': (2, 1, 1), 'estimator': (2, 1, 1), 'insurer': (1, 1, 1)}
    assert account_town().arr.label == '(town wiring . (((plant || townspeople) || estimator) || insurer))'
    q = {'plant': jnp.array([.7, .3]), 'townspeople': jnp.array([.2]),
         'estimator': jnp.array([.5]), 'insurer': jnp.array([.5])}
    pushes = {name: b.omega(q[name], EMPTY, jnp.full(b.in_dim_N, .6))[2] for name, b in boxes.items()}
    np.testing.assert_array_equal(pushes['townspeople'], [1.0, 0.0])    # power pushes back; smoke does not
    for name in ('plant', 'estimator', 'insurer'):
        np.testing.assert_array_equal(pushes[name], 0.0)


def test_book_the_run_is_the_dynamics_functors():
    """``run`` takes the composite's effort field and integrator; here it is checked equal to ``Phi``."""
    town = account_town()
    O, state, by_phi = Phi(town.arr), town.arr.P.init, []
    for _ in range(50):
        _, _, state = O.with_state(state).run_one(IN_POS, lambda _o: (EMPTY, EMPTY))
        by_phi.append(np.asarray(town.arr.P.position(state)))
    np.testing.assert_allclose(np.array(by_phi), run(town, 50*town.settings.dt)[1:], atol=1e-13)


def test_book_estimate_climbs_premium_follows_plant_feels_it_at_once():
    """``sec.cost_later``: "her estimate climbs to match the coughing, the premium climbs behind it, and the
    plant feels the premium on its smoke almost immediately"; then the plant cleans, "the townspeople keep
    nearly all their power and cough far less"."""
    no_return, town = eve_town(1), account_town()
    base, t = run(no_return), run(town)
    result, unpriced = summary(town, t), summary(no_return, base)
    beta = town.settings.beta
    assert result['estimate_reaches_1'] < result['premium_reaches_1']     # the premium climbs behind it
    assert result['final_estimate'] == pytest.approx(beta, rel=.05)       # climbs to match
    # the plant's effort field reads the premium with no box between: the same step
    q = town.initial().at[town.slots['output']].set(.7).at[town.slots['premium']].set(1.)
    plant_row = jax.jacfwd(town.field)(q)[town.slots['output'].start, town.slots['premium'].start]
    assert float(plant_row) < 0
    assert result['final_cleanliness'] > .9                               # it cleans up
    assert result['power'] > .95*unpriced['power']                        # keep nearly all their power
    assert result['final_coughing'] < .01*no_return.value(base, 'coughing')[-1, 0]   # cough far less


def test_book_the_court_and_the_ledger_change_nothing_on_the_four_boxes_and_the_premium_comes_first():
    """The fuller town, with memory, claims delay, court and money beside the four boxes, read one way: on
    the four boxes it is the drawn town, step for step, and its first paid claim comes well after the
    premium has climbed and the plant has cleaned. The later cost counts now (``sec.cost_later``)."""
    town, full = account_town(), eve_town(4, fit_coughing=True)
    a, b = run(town), run(full)
    for name in town.slots:
        np.testing.assert_allclose(town.value(a, name), full.value(b, name), atol=1e-12)
    claim = np.flatnonzero(np.diff(full.value(b, 'money')[:, 2]) > 1e-12)[0]+1
    cleanliness = town.value(a, 'cleanliness')[:, 0]
    half = np.flatnonzero(cleanliness >= .5*cleanliness[-1])[0]
    premium = np.flatnonzero(town.value(a, 'premium')[:, 0] >= 1)[0]
    assert premium < half < claim


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
    q = q.at[town.slots['coughing']].set(.2+town.settings.dt*(2*.8**2/2-.2))
    assert float(town.field(q)[town.slots['estimate']][0]) > .1
    # Same observed state, different hidden beta: learning sees no difference.
    np.testing.assert_allclose(town.field(q)[town.slots['estimate']],
                               other.field(q)[other.slots['estimate']], atol=0)
    q = q.at[town.slots['output']].set(0).at[town.slots['coughing']].set(0)
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
