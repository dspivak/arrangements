"""Independent monetary identities and admissibility checks for the P29 model."""
import numpy as np


def check_money(town, trajectory):
    c = town.settings
    x = town.value(trajectory, 'output')[:, 0]
    balances = town.value(trajectory, 'money')
    operating = c.benefit*x[:-1]-x[:-1]**2/2
    np.testing.assert_allclose(np.diff(balances.sum(axis=1)), c.dt*operating,
                               atol=2e-12, rtol=2e-10)
    if town.stage >= 3:
        premium = town.value(trajectory, 'premium')[:-1, 0]*x[:-1]**2/2
        claims_paid = np.diff(balances[:, 2])
        np.testing.assert_allclose(np.diff(balances[:, 1]), c.dt*premium-claims_paid,
                                   atol=2e-12, rtol=2e-10)
    assert np.all(np.isfinite(trajectory))
    assert np.min(x) > 0
    assert np.min(town.value(trajectory, 'coughing')) >= -1e-12
