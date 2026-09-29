"""P29 stage (d): smoke per watt. The plant gets a second dial, cleanliness.

Predictions fixed before code: notes/P29-D-PREDICTIONS-2026-09-28.md. Results and the
weirdness check: notes/P29-D-RESULTS.md. Claude Opus 5.5, subagent of 8f53ae48, 2026-09-28.
Reasonable: a priced smoke liability makes a plant clean when cleaning is cheap relative to
output, and shrink when it is dear. Artificial: power consumed by nobody, quadratic cleaning
cost (the first bit of cleaning is free), instant smoke-to-illness, noiseless records.

Result (2026-09-28): all four predictions held. At p = 0 the plant never cleans (c = 0
exactly) at beta .5, 2, 8. At p = 2, k = .1: c = .952, x = .995, cleaning's share of the
harm reduction .998 (log) / .995 (Shapley). The threshold is k = 1 exactly, where the cost is
symmetric in x and 1-c. Her account: c reaches half its final value by t = 2.58 (k = .1),
first claim t = 6.04. Power over 30 units falls 3.4% (k = .1) against 63.5% one-dial.
Found: cleaning starves her fit (final estimate 1.674, not 2); at beta = 8 the informed
plant is trapped in a dearer minimum that the town's own path avoids.
"""
import json
import jax
import jax.numpy as jnp
import numpy as np
from arrangements.town import Settings, build, run, summary, smoke

SMALL, MODERATE, LARGE = .1, 1.0, 10.0


def plant_cost(x, c, p, k, benefit=1.0):
    return .5*(1+p*(1-c)**2)*x**2 - benefit*x + .5*k*c**2


def first_claim_step(town, trajectory):
    paid = np.flatnonzero(np.diff(town.value(trajectory, 'money')[:, 2]) > 1e-12)
    return int(paid[0]+1)


def check_clean_money(town, trajectory):
    """check_money for the two-dial plant: cleaning is an operating cost, premium is on smoke."""
    c = town.settings
    x = town.value(trajectory, 'output')[:, 0]
    cl = town.value(trajectory, 'cleanliness')[:, 0]
    s = smoke(town, trajectory)
    balances = town.value(trajectory, 'money')
    operating = c.benefit*x[:-1]-x[:-1]**2/2-c.clean_cost*cl[:-1]**2/2
    np.testing.assert_allclose(np.diff(balances.sum(axis=1)), c.dt*operating,
                               atol=2e-12, rtol=2e-10)
    if town.stage >= 3:
        premium = town.value(trajectory, 'premium')[:-1, 0]*s[:-1]**2/2
        np.testing.assert_allclose(np.diff(balances[:, 1]),
                                   c.dt*premium-np.diff(balances[:, 2]), atol=2e-12, rtol=2e-10)
    assert np.all(np.isfinite(trajectory))
    assert np.min(x) > 0 and np.min(cl) >= 0 and np.max(cl) < 1


def test_unpriced_plant_never_cleans_whatever_beta():
    """Prediction 1: at p = 0 (no return route) c stays 0 and x goes to 1, at every beta."""
    one_dial = build(1, Settings())
    one_x = one_dial.value(run(one_dial), 'output')[:, 0]
    table = {}
    for beta in (.5, 2., 8.):
        town = build(1, Settings(beta=beta, clean_cost=SMALL), clean=True)
        trajectory = run(town)
        cl = town.value(trajectory, 'cleanliness')[:, 0]
        np.testing.assert_array_equal(cl, 0.0)
        np.testing.assert_allclose(town.value(trajectory, 'output')[:, 0], one_x, atol=1e-13)
        x, s, h = one_x[-1], smoke(town, trajectory)[-1], town.value(trajectory, 'coughing')[-1, 0]
        np.testing.assert_allclose([x, s, h], [1, 1, beta/2], atol=1e-10)
        table[beta] = {'x': float(x), 'c': float(cl[-1]), 's': float(s), 'h': float(h)}
    # The plant's forcing reads nothing from the residents: no push comes back up the smoke.
    town = build(1, Settings(beta=8., clean_cost=SMALL), clean=True)
    jac = np.asarray(jax.jacfwd(town.field)(jnp.array([.7, .3, .2])))
    np.testing.assert_array_equal(jac[:2, 2], 0.0)
    assert jac[2, 0] != 0 and jac[2, 1] != 0
    print(json.dumps(table, sort_keys=True))


def test_priced_plant_cleans_at_a_true_minimum_and_threshold_is_k_one():
    """Prediction 2 at p = beta = 2 (informed plant): c > 0, positive definite Hessian;
    for small k cleaning carries most of the harm reduction; the split is 50/50 at k = 1."""
    report = {}
    for k in (SMALL, MODERATE, LARGE):
        town = build(2, Settings(clean_cost=k), clean=True, informed=True)
        trajectory = run(town, 60)
        np.testing.assert_allclose(town.value(trajectory, 'assessed'), 2., atol=1e-12)
        x = town.value(trajectory, 'output')[-1, 0]
        cl = town.value(trajectory, 'cleanliness')[-1, 0]
        point = jnp.array([x, cl])
        grad = jax.grad(lambda q: plant_cost(q[0], q[1], 2., k))(point)
        hess = np.asarray(jax.hessian(lambda q: plant_cost(q[0], q[1], 2., k))(point))
        np.testing.assert_allclose(grad, 0, atol=1e-7)  # settled: slow eigenvalue .30 at k = 1
        assert cl > 0 and np.linalg.eigvalsh(hess)[0] > 0
        u, s = 1-cl, (1-cl)*x
        log_share, shapley = np.log(u)/np.log(s), .5*(1-u*u)*(1+x*x)/(1-s*s)
        report[k] = {'x': float(x), 'c': float(cl), 's': float(s), 'log_share_c': float(log_share),
                     'shapley_share_c': float(shapley), 'min_eig': float(np.linalg.eigvalsh(hess)[0])}
        if k == SMALL:
            assert log_share > .5 and shapley > .5
        if k == MODERATE:  # the cost is symmetric in x and 1-c at k = benefit**2 = 1
            np.testing.assert_allclose([x, log_share, shapley], [1-cl, .5, .5], atol=1e-6)
            assert s > 1/3  # expectation D: settled smoke above the one-dial 1/3
        if k == LARGE:
            assert log_share < .5 and shapley < .5
    print(json.dumps(report, sort_keys=True))


def test_her_account_makes_the_plant_clean_before_the_first_claim():
    """Prediction 3, with stage 3 as control; also the cumulative split in the town."""
    no_return = build(1, Settings())
    x0 = no_return.value(run(no_return), 'output')[:, 0]
    report = {}
    for k in (SMALL, MODERATE, LARGE):
        historical = build(3, Settings(clean_cost=k), clean=True)
        account = build(4, Settings(clean_cost=k), clean=True)
        hist_run, acc_run = run(historical), run(account)
        check_clean_money(historical, hist_run)
        check_clean_money(account, acc_run)
        claim = first_claim_step(account, acc_run)
        assert claim == first_claim_step(historical, hist_run)
        np.testing.assert_array_equal(historical.value(hist_run, 'cleanliness')[:claim, 0], 0.0)
        cl = account.value(acc_run, 'cleanliness')[:, 0]
        half = int(np.flatnonzero(cl >= .5*cl[-1])[0])
        assert half < claim
        x, u = account.value(acc_run, 'output')[:, 0], 1-cl
        c_part = np.sum((.5*(1-u**2)*(x0**2+x**2))[:-1])
        x_part = np.sum((.5*(x0**2-x**2)*(1+u**2))[:-1])
        summ = summary(account, acc_run)
        report[k] = {'c_half_at': half*account.settings.dt, 'first_claim_at': claim*account.settings.dt,
                     'cumulative_share_c': float(c_part/(c_part+x_part)), 'harm': summ['harm'],
                     'final_estimate': summ['final_estimate'],
                     'historical_final_premium': summary(historical, hist_run)['final_premium']}
        if k == SMALL:
            assert report[k]['cumulative_share_c'] > .5
            # Found, not predicted: cleaning cuts the exposure her fit reads, so it stalls.
            assert summ['final_estimate'] < 1.8
            np.testing.assert_allclose(report[k]['historical_final_premium'], 2., atol=1e-6)
    print(json.dumps(report, sort_keys=True))


def test_power_to_other_towns_falls_less_than_with_one_dial():
    """Prediction 4: at each institution and k, power falls less than in the one-dial town;
    at small k the settled fall is under 5%."""
    no_return = build(1, Settings())
    base = summary(no_return, run(no_return))
    base_power = no_return.settings.dt*np.sum(no_return.value(run(no_return), 'output')[:-1, 0])
    report = {}
    for stage in (2, 3, 4):
        one = build(stage, Settings())
        one_run = run(one)
        one_x = one.value(one_run, 'output')[:, 0]
        one_fall = 1-one.settings.dt*np.sum(one_x[:-1])/base_power
        report[str(stage)] = {'one_dial_power_fall': float(one_fall),
                         'one_dial_harm': summary(one, one_run)['harm']}
        for k in (SMALL, MODERATE, LARGE):
            town = build(stage, Settings(clean_cost=k), clean=True)
            summ = summary(town, run(town))
            fall = 1-summ['power']/base_power
            assert fall < one_fall and summ['final_output'] > one_x[-1]
            report[str(stage)][str(k)] = {'power_fall': float(fall), 'settled_power': summ['final_output'],
                                'harm': summ['harm']}
            if k == SMALL:
                assert 1-summ['final_output'] < .05
    report['no_return_harm'] = base['harm']
    print(json.dumps(report, sort_keys=True))


def test_high_beta_liability_can_trap_the_informed_plant_but_not_the_town():
    """Able to fail: the x^2(1-c)^2 term is non-convex. At p = 8, k = .8 there are two minima.
    The informed plant (p = 8 at t = 0, starting small and dirty) settles in the dearer one;
    the town, whose p rises from 0 through claims, settles in the cheaper, cleaner one."""
    k, beta = .8, 8.
    informed = build(2, Settings(beta=beta, clean_cost=k), clean=True, informed=True)
    town = build(3, Settings(beta=beta, clean_cost=k), clean=True)
    points = {}
    for name, t in (('informed', informed), ('town', town)):
        trajectory = run(t, 100)  # the dirty minimum's slow eigenvalue needs the longer run
        p = t.value(trajectory, 'assessed' if name == 'informed' else 'premium')[-1, 0]
        np.testing.assert_allclose(p, beta, atol=1e-6)
        q = jnp.array([t.value(trajectory, 'output')[-1, 0], t.value(trajectory, 'cleanliness')[-1, 0]])
        f = lambda q: plant_cost(q[0], q[1], beta, k)
        np.testing.assert_allclose(jax.grad(f)(q), 0, atol=1e-7)
        assert np.linalg.eigvalsh(np.asarray(jax.hessian(f)(q)))[0] > 0
        points[name] = {'x': float(q[0]), 'c': float(q[1]), 's': float((1-q[1])*q[0]), 'cost': float(f(q))}
    assert points['informed']['x'] < .3 and points['town']['x'] > .8
    assert points['informed']['cost'] > points['town']['cost'] + .05
    print(json.dumps(points, sort_keys=True))
