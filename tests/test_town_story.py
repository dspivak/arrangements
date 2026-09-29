"""The sentences of the book's §1.6.1 that Eve's model stands behind, one test each.

Articulating Forms that Matter, §1.6.1, "This book is a calculus for working together": Eve
models a plant (liability in; power and smoke out) and townspeople (power and smoke in;
coughing out), puts them into a town, and runs it. `eve_town()` is that model.
David's re-edit of 2026-09-29 cut the settling sentences (the balance, dying away, climbing) in
favour of "coughs also heal at some rate … left as an open parameter in her model"; the three
tests that quoted them now check the model behind that parameter.
"""
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import pytest

from arrangements.town import Settings, eve_town, run, summary


def settled(settings, duration=30.0, stage=1):
    town = eve_town(stage, settings)
    trajectory = run(town, duration)
    return town, trajectory, summary(town, trajectory)


def test_output_settles_where_the_next_watt_costs_what_it_brings_in():
    # "invites more output up to the point where the next unit costs as much as it brings in,
    # and there it settles": marginal cost x equals the price per watt, benefit
    s = Settings()
    _, _, result = settled(s)
    assert result['final_output'] == pytest.approx(s.benefit, abs=1e-9)


def test_filtering_earns_nothing_so_the_plant_never_filters():
    # "cleaning costs money and brings in nothing of value to the plant, so its effort
    # landscape never invites this activity": from no filtering it never starts, and a
    # plant that starts filtered stops
    _, _, result = settled(Settings())
    assert result['final_cleanliness'] == 0.0
    # (slowly: the pull toward no filtering is the filter's own small running cost)
    town, trajectory, result = settled(Settings(clean0=.5), duration=100.0)
    cleanliness = town.value(trajectory, 'cleanliness')[:, 0]
    assert np.all(np.diff(cleanliness) < 0)
    assert result['final_cleanliness'] < 1e-4


def test_a_big_enough_liability_on_smoke_makes_it_filter():
    # "A big enough liability coming in on its smoke would change that"
    _, _, result = settled(Settings(liability=2.0))
    assert result['final_cleanliness'] > .9
    assert result['final_output'] > .99   # and it keeps its power


def test_coughing_settles_where_starts_equal_stops():
    # "Their coughing settles where the two drives balance each other: the number of people
    # who stop coughing equals the number who start"
    s = Settings()
    town, trajectory, result = settled(s)
    smoke = result['final_smoke']
    starts, stops = s.beta*smoke**2/2, s.recovery*result['final_coughing']
    assert stops == pytest.approx(starts, rel=1e-9)


def test_without_the_smoke_it_dies_away():
    # "without the smoke it would die away to nothing"
    _, _, result = settled(Settings(beta=0.0, coughing0=1.0))
    assert result['final_coughing'] == pytest.approx(0.0, abs=1e-6)


def test_if_coughs_stopped_passing_it_would_climb_without_end():
    # "if coughs stopped passing, it would climb without end"
    town, trajectory, _ = settled(Settings(recovery=0.0))
    coughing = town.value(trajectory, 'coughing')[:, 0]
    assert np.all(np.diff(coughing[100:]) > 0)
    assert coughing[-1] > 1.9*coughing[len(coughing)//2]


def test_her_run_harsher_smoke_more_coughing_same_power_never_filters():
    # "with milder smoke the people cough less, and with harsher smoke they cough more. But
    # either way the plant keeps making the same power and never filters the smoke, even
    # long-term"
    coughs, plants = [], []
    for beta in (.5, 2.0, 8.0):
        town, trajectory, result = settled(Settings(beta=beta))
        coughs.append(result['final_coughing'])
        plants.append(np.concatenate([town.value(trajectory, 'output'),
                                      town.value(trajectory, 'cleanliness')], axis=1))
        assert result['final_output'] == pytest.approx(1.0, abs=1e-9)
        assert result['final_cleanliness'] == 0.0
    assert coughs[0] < coughs[1] < coughs[2]
    # "their coughing does not yet enter the plant's decision": the plant's whole path is
    # the same whatever the smoke does to the townspeople
    assert np.array_equal(plants[0], plants[1]) and np.array_equal(plants[1], plants[2])
