"""Her account: a cost that would arrive later counts now (book Chapter 1, ``sec.cost_later``; linked from the
footnotes there and in ``sec.check_claims``).

For a reader of the book. ``account_town()`` (``arrangements/town.py``) is the town of ``eqn.town_account``, four boxes
as drawn, with the smoke copied at its dot:

* ``plant``: premium in, power and smoke out. It pays the premium ``p s^2/2`` on its smoke ``s``, where ``p`` is the
  number arriving on the wire, and reads it one way: it takes the price as given.
* ``townspeople``: power and smoke in, coughing out, as in ``eqn.town_system``.
* ``estimator``: coughing and smoke in, the estimate out. Its state is one number, her estimate of how much the smoke
  drives new coughing, and its cost is the gap between what that number predicts, ``estimate s^2/2`` new coughs a
  day, and what she measures, ``recovery coughing``, squared, so the number moves as the evidence comes in. It
  pushes nothing back against the coughing.
* ``insurer``: the estimate in, the premium out. Its state is one number, the rate ``p``, and it moves that number
  toward the estimate, which makes the premium ``p s^2/2`` exactly cover the claims she estimates the smoke will bring.
  It pushes nothing back on her account.

Run it, run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    from arrangements.town import account_town, run, summary
    town = account_town()
    r = summary(town, run(town))
    print({k: round(r[k], 3) for k in ('estimate_reaches_1', 'premium_reaches_1', 'final_estimate',
                                       'final_output', 'final_cleanliness', 'final_coughing')})
    # {'estimate_reaches_1': 1.76, 'premium_reaches_1': 2.24, 'final_estimate': 2.033, 'final_output': 0.995, 'final_cleanliness': 0.953, 'final_coughing': 0.002}

Her estimate climbs to match the coughing (the true coefficient is ``beta = 2``), the premium climbs behind it, the
plant cleans up, and the townspeople keep nearly all their power and cough far less: from a coughing of 1 with no
return route to about .002.

What the tests check. The ``test_book_*`` tests: the composite is those four boxes with those ports, and only the
power wire pushes back; ``run`` is the dynamics functor's run; the sentences above; and beside the four boxes, a court,
a claims delay and a ledger read one way and change nothing on them, step for step, while the first paid claim comes
long after the premium has climbed. The other tests run the earlier, fuller town with a plant of one dial (``build``),
whose estimate reads new coughing through a memory box: at two step sizes the premium precedes the first claim and the
harm falls; a report wired to nothing changes nothing; the fit sees no secret coefficient, only records; and an exact
reader of the fit bends it, where a one-way reader does not.

Artificial: known recovery, noiseless records, a correct one-coefficient model, and an insurer who adopts her estimate
automatically. Nothing here models litigation or anyone's reading of a book. Once the plant has cleaned there is little
smoke to learn from, so the estimate stops moving wherever it is then: near ``beta`` at the book's settings, far from it
at ``beta = 8`` (4.48).

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``; results in ``notes/P29-TOWN-RESULTS.md``.
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
