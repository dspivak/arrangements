"""arrangements: the category of arrangements ``arr``, executable.

A Python implementation of the principal constructions of *Articulating Forms
that Matter*, Chapter 3 (def:arr): ``cot(Q)``-coalgebras as parameters
(integrators are part of the arrangement), arrangements carrying a forcing
one-form (exact or not), pullback-and-sum composition, the polynomial
interpretation, and the one dynamics functor ``Phi : arr -> pc``
(sec.dynamics_functor). It replaces the executable companion of CDLM, which is
left unmodified at ``dap``; see README.md for what changed and STATUS.md for what
the tests do and do not establish.
"""

from .rvect import ReactiveVectorSpace
from .coalgebra import CotCoalg, CotCoalg2, CotCoalgK, configuration, phase, damped, gyro, phase_on_cotangent, iterate
from .arrangement import Arrangement, exact, zero_form
from .polynomial import Yon, Cot, DirichletProduct, PolyMap, nest_left, unnest_left
from .pc import PCMorphism
from .interpretation import interpretation, trivial_omega
from .functors import Phi, cot_object, cot_map, phi_laxator
from .pc2 import PCMorphism2, Phi2
from .pcK import PCMorphismK, PhiK
from .leapfrog import leapfrog
from .rk4 import rk4, rk4_gyro
from . import functors
from . import wiring
from . import learning

from . import attention

# The worked examples (extensions beyond the paper are marked as such in each module).
from . import amplifier
from . import control
from . import dc_motor
from . import gyroscope
from . import gyroscope_faithful
from . import hopfield
from . import kuramoto
from . import lc_circuit
from . import logic
from . import moleculardyn
from . import pinn
from . import reaction_network
from . import replicator
from . import system_id

__all__ = [
    "ReactiveVectorSpace",
    "CotCoalg",
    "CotCoalg2",
    "CotCoalgK",
    "configuration",
    "phase",
    "damped",
    "gyro",
    "phase_on_cotangent",
    "iterate",
    "PCMorphism2",
    "Phi2",
    "PCMorphismK",
    "PhiK",
    "leapfrog",
    "rk4",
    "rk4_gyro",
    "Arrangement",
    "exact",
    "zero_form",
    "Yon",
    "Cot",
    "DirichletProduct",
    "PolyMap",
    "nest_left",
    "unnest_left",
    "PCMorphism",
    "interpretation",
    "trivial_omega",
    "Phi",
    "cot_object",
    "cot_map",
    "phi_laxator",
    "functors",
    "wiring",
    "learning",
    "attention",
    # the worked examples:
    "amplifier",
    "control",
    "dc_motor",
    "gyroscope",
    "gyroscope_faithful",
    "hopfield",
    "kuramoto",
    "lc_circuit",
    "logic",
    "moleculardyn",
    "pinn",
    "reaction_network",
    "replicator",
    "system_id",
]
