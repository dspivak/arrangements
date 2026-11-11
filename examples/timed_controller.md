# Finite imagined control with observation and preparation

Research example in IMC on Euclidean spaces, extracted from the book's timed-controller
experiments a330db6f and 0676663f (2026-11-11). It is not a general synthesis theorem.

Run from the library root:

```sh
.venv/bin/python -m pytest tests/test_timed_controller.py -q
```

The module is `arrangements.timed_controller`. All physical steps and learning updates
execute composed arrangements through `Phi`; ordinary numerical calculations in the
tests are independent verification oracles and diagnostics. The example does not import
files from the book repository or read saved results. Its starting four controller
coefficients are the learned mass-one fixture from a330db6f, documented in the module.

## The connected composite

`online(mass, budget, split_command=True)` nests observation wiring around a motor-command
arrangement around the tensor of four parts:

| Part | Persistent state | Inputs | Outputs/configuration |
|---|---:|---:|---:|
| Participant and previous transition | 6 | 0 | 6 / 6 |
| Physical/computation clock and remaining duration stream | 22 | 0 | 22 / 22 |
| Model computer | 1 | 6 | 1 / 7 |
| Controller computer | 4 | 24 | 4 / 28 |

The computer configurations include zero auxiliary coordinates on which their inputs
induce inclinations; those 30 coordinates are not stored state. The composite has
33 stored coordinates, configuration dimension 63, no outer inputs and 13 diagnostic
outputs. `observations(arr, state)` reads those outputs as position, velocity, previous
position, previous velocity, previous command, previous duration, time, completed
controller update count, inverse mass and four controller coefficients.

The command's transfers are identities. Its field places the affine command on the
participant's position coordinate. Observation wiring has zero field. It sends the previous
transition to model learning and sends inverse mass, current position/velocity/time and
20 remaining durations to controller learning. The model's snapshot inputs have zero
pushback: fitting observations does not move the observed participant to fit a prediction.

Model learning composes a parameter holder, a normalized transition predictor and one-way
squared error. Controller learning composes a coefficient holder with frozen context,
an imagined execution and a terminal score. The imagined execution itself nests a
modeled particle, a clock, frozen coefficients and a motor command. Inner iterations
are supplied computer integrator updates, not a derivation of a material processor.

At the tested physical duration 0.1, the clock advances one interval and shifts the
remaining-duration stream. Each interval performs one model update and `budget`
controller updates; all parts read the old joint configuration. New coefficients affect
the following interval. `correct=False`, `learn=False`, `cold=True` provide the recorded
removal conditions. `split_command=False` returns the original bundled command/wiring
arrangement; its fields and all tested traces equal the further factorization.

## Minimal execution

```python
import jax
jax.config.update("jax_enable_x64", True)
from arrangements.timed_controller import DT, Z, online, run_step, observations

arr = online(mass=0.5, budget=10, split_command=True)
step = jax.jit(run_step(arr))
state = arr.P.init
for _ in range(20):
    state = step(state, Z, DT)
print(observations(arr, state))
```

## Separate timing experiment

`timing_learner()` composes a parameter holder, a smooth two-choice probability map,
a fixed outcome forecast, expected reward and its score. One hundred rehearsals
increase modeled expected reward from 2 to about 2.942. In the opposite actual world,
the same change lowers reward to about 1.058. It supplies no new observation and does
not yet govern the connected composite's decision to wait, rehearse or act.

## Scope and checks

The tests cover analytic model updates including zero excitation, frozen observations
and context, score removal, finite-difference gradients, units and tensor composition,
causal latency, an independent 20-interval replay, all 21 remaining-duration suffixes,
three complete command/wiring factorization traces and the conditional timing result.
They require float64 (enabled by the test suite's conftest).

The maps are smooth for each fixed finite loop count and positive actual mass. The
general smooth free/cofree construction remains conditional and is not proved here.
The physical schedule is 20 intervals of 0.1; the clock's smooth extension to other
durations is not claimed to implement an arbitrary processor schedule. Zero-duration
padded imagined steps hold the physical state fixed. Computation and actuation have no
energy budget. Estimated inverse mass is a point estimate, not a probability model.

Judgment: this is a reasonable finite control/rehearsal experiment. At bodily, cellular
or social scales, exact snapshots, an externally supplied computation budget and an
unpriced actuator are idealizations. The code establishes no biological account of
dreaming, no learned wait/go scheduler and no reliable deadline-control guarantee.
