"""P29's town, composed from systems with one-way observation and payment records.

Model choices and predictions: notes/P29-TOWN-PREDICTIONS-2026-09-28.md.
A profit maximizer uses a quadratic expected-liability/premium schedule. A court
learns its coefficient from delayed health observations; an insurer reads that
estimate or the mathematician's earlier estimate. Forecasts never create money.
This is a new composite using the same Arrangement/CotCoalg primitives as
system_id, insurance, and account; it does not reuse their example-specific boxes.
"""
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .arrangement import Arrangement, zero_form
from .coalgebra import configuration
from .rvect import euclidean
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

EMPTY = jnp.zeros(0)


@dataclass(frozen=True)
class Settings:
    benefit: float = 1.0
    beta: float = 2.0
    recovery: float = 1.0
    dt: float = .02
    output0: float = .1
    health0: float = 0.0
    learn_rate: float = 40.0
    quote_rate: float = 2.0
    delay_steps: int = 300


def own_cost(label, initial, inputs, cost, mobility):
    """Conditional cost differentiated only on the box's configuration.

    Zero input covector is the declared one-way boundary. The smooth input map
    retains its genuine derivative: no stop_gradient or custom derivative.
    """
    gradient = jax.grad(cost, argnums=0)
    return node(label, initial, inputs, lambda q, e: -gradient(q, e), mobility)


def node(label, initial, inputs, force, mobility):
    initial = jnp.atleast_1d(jnp.asarray(initial, dtype=float))
    return Arrangement(
        P=configuration(euclidean(len(initial), mobility), init=initial),
        in_dim_M=0, out_dim_M=0, in_dim_N=inputs, out_dim_N=len(initial),
        out_f=lambda q, m: q,
        in_f=lambda q, m, e: EMPTY,
        omega=lambda q, m, e: (force(q, e), EMPTY, jnp.zeros_like(e)),
        label=label)


@dataclass
class Town:
    arr: Arrangement
    slots: dict
    settings: Settings
    stage: int

    def field(self, q):
        return self.arr.omega(q, EMPTY, EMPTY)[0]

    def initial(self):
        return self.arr.P.position(self.arr.P.init)

    def value(self, trajectory, name):
        return np.asarray(trajectory)[..., self.slots[name]]


def build(stage, settings=Settings(), *, connect_report=True, informed=False):
    """Stages 1=no return, 2=law, 3=historical insurance, 4=her account.

    Informed control starts the liability estimator at the true coefficient.
    It uses the same delayed payouts. It is an explicit information advantage.
    """
    if stage not in (1, 2, 3, 4) or settings.delay_steps < 0:
        raise ValueError('stage must be 1..4 and delay_steps nonnegative')
    c = settings
    boxes, slots, sources = [], {}, []
    offset = 0

    def add(box, source):
        nonlocal offset
        slots[box.label] = slice(offset, offset + box.P.Q_dim)
        offset += box.P.Q_dim
        boxes.append(box)
        sources.append(source)

    def at(q, name):
        return q[slots[name]]

    def price(q):
        if stage == 1:
            return jnp.array([0.0])
        return at(q, 'premium' if stage >= 3 else 'assessed')

    add(own_cost('output', [c.output0], 1,
                 lambda q, e: .5 * (1 + e[0]) * q[0]**2 - c.benefit * q[0], c.dt), price)
    add(own_cost('health', [c.health0], 1,
                 lambda q, e: .5*c.recovery*q[0]**2 - q[0]*c.beta*e[0]**2/2, c.dt),
        lambda q: at(q, 'output'))

    if stage >= 2:
        add(node('memory', [c.output0, c.health0, 0.0], 3, lambda q, e: e-q, 1.0),
            lambda q: jnp.concatenate([at(q, 'output'), at(q, 'health'), jnp.ones(1)]))

        def observed(q):
            old = at(q, 'memory')
            incident = (at(q, 'health')[0]-old[1])/c.dt + c.recovery*old[1]
            return old[2]*jnp.array([old[0]**2/2, incident])

        if c.delay_steps:
            def shift(q, e):
                return jnp.concatenate([e, q[:-2]]) - q
            add(node('delay', np.zeros(2*c.delay_steps), 2, shift, 1.0), observed)
            settled = lambda q: at(q, 'delay')[-2:]
        else:
            settled = observed

        fit_cost = lambda q, e: .5 * (q[0]*e[0]-e[1])**2
        add(own_cost('assessed', [c.beta if informed else 0.0], 2,
                     fit_cost, c.dt*c.learn_rate), settled)

        if stage >= 3:
            add(own_cost('premium', [0.0], 1,
                         lambda q, e: .5*(q[0]-e[0])**2, c.dt*c.quote_rate),
                lambda q: at(q, 'estimate' if stage == 4 and connect_report else 'assessed'))
        if stage == 4:
            add(own_cost('estimate', [0.0], 2, fit_cost, c.dt*c.learn_rate), observed)

        def payments(q):
            output = at(q, 'output')[0]
            operating = c.benefit*output-output**2/2
            claims = settled(q)[1]
            if stage >= 3:
                premium = at(q, 'premium')[0]*output**2/2
                return jnp.array([operating-premium, premium-claims, claims])
            return jnp.array([operating-claims, 0.0, claims])

        add(node('money', [0., 0., 0.], 3, lambda q, e: e, c.dt), payments)

    inner = tensor_arrangements(boxes)
    outer = Arrangement(
        P=trivial_parameter(), in_dim_M=sum(b.in_dim_N for b in boxes),
        out_dim_M=offset, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: EMPTY,
        in_f=lambda q, m, e: jnp.concatenate([source(m) for source in sources]),
        omega=zero_form, label='town wiring')
    return Town(compose_seq(inner, outer), slots, c, stage)


def run(town, duration=30.0):
    """Run the actual composed forcing and product integrator, including records."""
    arr = town.arr
    n = round(duration/town.settings.dt)

    @jax.jit
    def scan(initial):
        def step(state, _):
            q = arr.P.position(state)
            force = arr.omega(q, EMPTY, EMPTY)[0]
            after = arr.P.step(state, force)
            return after, arr.P.position(after)
        _, rest = jax.lax.scan(step, initial, None, length=n)
        return jnp.concatenate([arr.P.position(initial)[None, :], rest], axis=0)
    return np.asarray(scan(arr.P.init))


def summary(town, trajectory):
    c = town.settings
    output = town.value(trajectory, 'output')[:, 0]
    result = {'harm': float(c.dt*np.sum(c.beta*output[:-1]**2/2)),
              'peak_output': float(np.max(output)), 'final_output': float(output[-1])}
    for name in ('assessed', 'premium', 'estimate'):
        if name in town.slots:
            values = town.value(trajectory, name)[:, 0]
            crossing = np.flatnonzero(values >= 1.0)
            result[name+'_reaches_1'] = float(crossing[0]*c.dt) if len(crossing) else None
            result['final_'+name] = float(values[-1])
    if 'money' in town.slots:
        money = town.value(trajectory, 'money')
        result['minimum_insurer_fund'] = float(money[:, 1].min())
        result['final_insurer_fund'] = float(money[-1, 1])
    return result
