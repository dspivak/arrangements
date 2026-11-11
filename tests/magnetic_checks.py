"""Distance assertions on actual field reads in the magnetic regression runs.

These inspect the existing smooth-interface composites, including intermediate
integrator stages, rather than inferring separation from sampled endpoints.
"""
import functools
import jax
import jax.numpy as jnp
import pytest
from jax.experimental import checkify
from arrangements import magnets as M, magnetic_circuits as C


def checked_jit(fun):
    checked = jax.jit(checkify.checkify(fun))
    @functools.wraps(fun)
    def run(*args):
        error, result = checked(*args)
        error.throw()
        return result
    return run


@pytest.fixture
def outside_smoothing_discs(monkeypatch):
    """Every evaluated source/reader separation in these regression cases exceeds r0."""
    for name in ('g', 'contact'):
        original = getattr(M, name)
        def guarded(d, *args, original=original):
            squared = jnp.sum(d*d, axis=-1)
            checkify.check(jnp.all(squared > M.r0**2),
                           'Regression field reading entered smoothing disc: squared distance {s}', s=squared)
            return original(d, *args)
        monkeypatch.setattr(M, name, guarded)
        if name == 'g':
            monkeypatch.setattr(C, 'g', guarded)  # winding module imports this field directly
