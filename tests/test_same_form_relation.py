"""Is "does not have to move" an equivalence relation? (book P41 (b); predictions first in
notes/runs/p41b-equivalence-3b93bdf7/PREDICTIONS.md in the book repo; Claude Opus 5.5, 3b93bdf7, 2026-10-15.)

A reader run under one source holds an attractor; fed a second source through the same port (the schedule
the environment's), it does not have to move when what it holds is an attractor of the second too. Write
``s -> s'``. For a reader with one attractor per source (the line-learner of test_same_form_learner.py) this
is the kernel of "source to its attractor", an equivalence. Here a reader with two wells under one source:

    one parameter q, model F(q, x) = q x_1 + q^2 x_2, squared error, gradient descent;
    A: x = (1, 0), label  1, loss (q - 1)^2 / 2, one well at 1;
    B: x = (0, 1), label  1, loss (q^2 - 1)^2 / 2, wells at +1 and -1;
    C: x = (1, 0), label -1, loss (q + 1)^2 / 2, one well at -1.

Every run is ``learning.train``: ``Phi`` of the training arrangement, the datum, the label and the loss as
boxes (plan P43 (b) job 5). Each source is one datum, so one step is one pass and there is no per-datum bias.
"""

import jax.numpy as jnp
import numpy as np

from arrangements.coalgebra import configuration
from arrangements.learning import parameterized_map, train
from arrangements.rvect import euclidean

ETA = 0.05
STEPS = 2000
SOURCES = {
    "A": (jnp.array([1.0, 0.0]), jnp.array([1.0])),
    "B": (jnp.array([0.0, 1.0]), jnp.array([1.0])),
    "C": (jnp.array([1.0, 0.0]), jnp.array([-1.0])),
}


def _F(q, x):
    return jnp.array([q[0] * x[0] + q[0] ** 2 * x[1]])


def _reader():
    return parameterized_map(_F, configuration(euclidean(1, ETA)), in_dim=2, out_dim=1)


def _hold(source, q0):
    q, _ = train(_reader(), jnp.array([q0]), [SOURCES[source]], steps=STEPS)
    return float(q[0])


def _moved(first, second, q0):
    """The reader holds what ``first`` left from ``q0``; how far does ``second`` move it?"""
    held = _hold(first, q0)
    q, _ = train(_reader(), jnp.array([held]), [SOURCES[second]], steps=STEPS)
    return held, abs(float(q[0]) - held)


def test_holds():
    """Each source's attractor from the start -0.5, and B's from +0.5."""
    for s, want in (("A", 1.0), ("B", -1.0), ("C", -1.0)):
        got = _hold(s, -0.5)
        print(s, "from -0.5 holds", got)
        assert abs(got - want) < 1e-9
    assert abs(_hold("B", 0.5) - 1.0) < 1e-9


def test_not_symmetric():
    """Predictions 1 and 2: A -> B, but not B -> A, from -0.5."""
    held, d = _moved("A", "B", -0.5)
    print("A then B: held", held, "moved", d)
    assert d < 1e-9
    held, d = _moved("B", "A", -0.5)
    print("B then A: held", held, "moved", d)
    assert abs(d - 2.0) < 1e-9


def test_not_transitive():
    """Prediction 3: A -> B and B -> C, but not A -> C."""
    assert _moved("A", "B", -0.5)[1] < 1e-9
    assert _moved("B", "C", -0.5)[1] < 1e-9
    held, d = _moved("A", "C", -0.5)
    print("A then C: held", held, "moved", d)
    assert abs(d - 2.0) < 1e-9


def test_depends_on_start():
    """Prediction 4: from +0.5, B holds +1 and B -> A."""
    held, d = _moved("B", "A", 0.5)
    print("from +0.5, B then A: held", held, "moved", d)
    assert d < 1e-9
