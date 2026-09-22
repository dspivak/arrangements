"""dap: the category of arrangements ``arr``, executable.

A Python implementation of the principal constructions of *Articulating Forms
that Matter*, Chapter 3 (def:arr): ``cot(Q)``-coalgebras as parameters
(integrators are part of the arrangement), arrangements carrying a forcing
one-form (exact or not), pullback-and-sum composition, the polynomial
interpretation, and the one dynamics functor ``Phi : arr -> pc``
(sec.dynamics_functor). Ported from CDLM's ``dap-core``; see README.md and
DESIGN.md for what changed.
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

# The worked examples (extensions beyond the paper are marked as such in each module).
from . import system_id
from . import pinn
from . import gyroscope
from . import gyroscope_faithful
from . import logic

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
    # extensions (beyond the paper):
    "system_id",
    "pinn",
    "gyroscope",
    "gyroscope_faithful",
    "logic",
]
