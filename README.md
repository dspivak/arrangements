# arrangements

Build a dynamical system by wiring together components that each know only their own
physics, and get the composite — loading effects, back-reaction, and all — generated for
you rather than derived by hand.

Each component publishes what it reports forward and what it demands backward. Wiring
them determines the dynamics of the whole; nothing about the interconnection has to be
modeled separately. Kirchhoff's current law comes out as the transpose of copying a
value, action–reaction as the symmetry of second derivatives, and "insulation" as an
exact geometric condition rather than an idealization.

```python
import jax; jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.rvect import ReactiveVectorSpace
from arrangements.wiring import compose_seq, tensor_arrangements, trivial_parameter

C = 9.81  # weight density

def tank(A, v0):
    """State is the volume; the tank reports the pressure at its base."""
    Q = ReactiveVectorSpace(dim=1, sharp_fn=lambda v: jnp.array([[A / C]]))
    return Arrangement(
        P=configuration(Q, init=jnp.array([v0])),
        in_dim_M=0, out_dim_M=0, in_dim_N=0, out_dim_N=1,
        out_f=lambda v, m_out: C * jnp.array([v[0] / A]),
        in_f=lambda v, m_out, n_in: jnp.zeros(0),
        omega=zero_form, label="tank")

def pipe(kappa):
    """Stateless: hears both pressures, returns a covector to each."""
    V = lambda e: 0.5 * kappa * (e[0] - e[1]) ** 2
    return Arrangement(
        P=trivial_parameter(),
        in_dim_M=0, out_dim_M=2, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, e: jnp.zeros(0),
        in_f=lambda q, e, n: jnp.zeros(0),
        omega=exact(lambda q, e, n: -V(e)), label="pipe")

system = compose_seq(tensor_arrangements([tank(1.0, 3.0), tank(1.0, 1.0)]), pipe(0.05))
O = Phi(system)

state, IN_POS = O.state, (jnp.zeros(0), trivial_omega(0))
for _ in range(60):
    _, _, state = O.with_state(state).run_one(
        IN_POS, lambda _o: (jnp.zeros(0), jnp.zeros(0)))

(v1, v2), _ = state
print(v1[0], v2[0], v1[0] + v2[0])   # 2.0  2.0  4.0, to the last ulp
```

Nobody wrote down that tank 1's outflow depends on tank 2's level. It follows from the
pipe's potential and the wiring, and total water is conserved, because the two
covectors the pipe generates are equal and opposite. Change the tank's shape and only its
own `sharp` changes; change the coupler and only the coupler changes.

Conservation here is not a numerical accident and not a property of the step size: the
pipe's two covectors are equal and opposite whatever `kappa` and whatever the tanks'
shapes, so the sum is preserved by the update rule itself, up to floating point.

The same six primitives cover gradient descent with backpropagation, the heat and wave
equations on a graph, LC circuits, transistors wired into NAND gates and latches,
DC motors, molecular dynamics, Kuramoto oscillators, Hopfield networks (whose energy
function *emerges* as the composite potential), replicator dynamics, mass-action reaction
networks, and an operad of attention arrangements.

This is working code for a book in progress, not a release: it is here to say what is
true while the mathematics is being written. Interfaces will move.

## Run it

```bash
cd arrangements
uv sync --extra test
scripts/run_tests.sh --list
scripts/run_tests.sh 120 tests/test_polynomial.py
```

Python ≥ 3.10, JAX and NumPy. Tests run under `jax_enable_x64`; the runner also needs
`timeout`. Select a small batch, estimate its duration, and give it a limit with ample margin.
The runner streams all output and saves full logs. Read and report each batch before starting
another; split a slow file by pytest node ID (`tests/test_file.py::test_name`) before its
projected completion approaches the limit. With no selections the runner prints usage and exits.
See `STATUS.md` §2 for what a passing test establishes and how tests share helpers.

## The two ideas

**Forcing, not potential.** A component's driving datum is a one-form `omega`, which need
not be exact. This matters because one-way influence — a waterfall, a sensor, a
controller — *cannot* come from a potential: for an energy-derived coupling, port `i`'s
load depends on port `j` exactly when `j`'s depends on `i`, so a one-way coupler forces
`d omega != 0`. Couplers are not required to be power-conserving, and interconnection
never produces algebraic constraints.

**The integrator is part of the arrangement.** A parameter is a `cot(Q)`-coalgebra
`(Q, S, u)`: a state space with a rule for turning an incoming covector into a new state.
So there is one dynamics functor `Phi`, taking no integrator argument, and "the same
system run two ways" is not expressible. The heat equation and the wave equation are two
*different* arrangements sharing one wiring and one forcing, differing only in `u`.

Composition of forcings is pullback-and-sum, which agrees with "add the potentials" when
both are exact and is the only correct rule when they are not.

## Honest limits

[`STATUS.md`](STATUS.md) is the account this library keeps of itself: the conventions,
what the tests are evidence *of*, and the open questions the code depends on. Three
things from it worth knowing before you trust a number:

- **Expressibility is not a result.** Any smooth vector field can be written as a forcing
  by taking `omega = V^flat`, making one step `x + dt V(x)`. What that buys is nothing.
  Five examples carry such a direct realization beside a structured one, and the
  difference is the point: the structured version certifies a Lyapunov function and
  dissipativity inherited under interconnection, and the direct one certifies nothing.
- **The regression tests prove faithfulness, not correctness.** Ported examples reproduce
  trajectories from the predecessor implementation to 1.4e-12 or better, which shows the
  port preserved its behavior. Agreement between two implementations is consistency, not
  truth; for truth, look to the closed-form tests.
- **Functoriality is tested on instances, not proved.** A passing suite is a warning
  system, not a certificate.

## What's here

| module | role |
|---|---|
| `coalgebra.py` | `cot(Q)`-coalgebras (parameters, integrators): `CotCoalg`, `configuration`, `phase`, `tensor` |
| `cascade.py` | a fund that pays its parts by a gate, nested to any depth; workers, sources, a market |
| `account.py` | the cascade's one level with a report channel: workers who say what they want, a fund that reads it, three wires |
| `pain.py` | the cascade's one level with learning and pain: a fund that reads its own balance and sends it down, parts that turn learning to product when it is high |
| `collective.py` | many parts on one spatial-attention parent, with kernel weights and the onward message as instruments |
| `rand.py` | randomness: finite and Gaussian random variables, stochastic integrators, samplers, `seeded` |
| `rvect.py` | reactive vector spaces (the response `sharp`), as constructor inputs |
| `arrangement.py` | an arrangement, plus `exact` and `zero_form` |
| `polynomial.py`, `pc.py` | polynomials and `[p,q]`-coalgebras (Moore form) |
| `interpretation.py` | the state-free polynomial interpretation at a parameter position |
| `functors.py` | the dynamics functor `Phi`, `cot`, `phi_laxator` |
| `wiring.py` | composition (pullback-and-sum), chains, graphs, tensor |
| `learning.py` | gradient descent with backpropagation |
| `attention.py` | the attention operad: placements, transfers, substitution, spatial kernels |
| `leapfrog.py`, `rk4.py`, `pc2.py`, `pcK.py` | multi-stage integrators (functoriality open) |
| examples | `amplifier`, `dc_motor`, `lc_circuit`, `kuramoto`, `hopfield`, `replicator`, `moleculardyn`, `reaction_network`, `logic`, `control`, `gyroscope`, `system_id`, `pinn` |

`arrangements` and `arrangements-demo` are installed as scripts: the first builds and
runs an arrangement interactively, the second runs the worked examples.

Non-gradient examples that the predecessor realized by a trick — a detailed-balance
potential, oppositely signed sharps, forcing terms smuggled into the integrator — keep
that realization *and* carry a second one as a genuine non-exact forcing, with a test
that the two agree step for step. The direct realization extends reach: mass action then
runs Wegscheider-violating and irreversible networks, and the replicator runs
rock–paper–scissors, which the structured versions reject.

## The mathematics

This implements the category of arrangements developed in *Articulating Forms that
Matter* (D. I. Spivak, in preparation), Chapter 3, with the attention operad of a
companion paper. It replaces the executable companion to *Compositional dynamics in
learning and mechanics*, which lives unmodified at
[dspivak/dap](https://github.com/dspivak/dap) and was copied from, not changed. Docstrings cite the source statement they
implement, so a reader with either draft can find it. `notes/` holds the design notes
the code was built against, kept as the record of why it is shaped this way.

## License

See [LICENSE](LICENSE).
