"""The town of *Articulating Forms that Matter*, §1.6 ("Can we articulate forms that matter?").

For a reader of the book. `eve_town()` builds Eve's model from §1.6.1: a plant (power and smoke
out; from stage 2 on, also liability in, the wire she adds in §1.6.2) and the townspeople (power and smoke in; coughing out), wired into a town.
The plant runs for profit: each watt is worth `benefit` to the townspeople who draw it, making
more watts costs more (x^2/2), and filtering the smoke costs money (clean_cost*c^2/2) and earns
nothing. Smoke s = (1-c)x starts people coughing (beta*s^2/2 a day); coughs pass at the rate
`recovery`, so the more people cough, the more stop each day, and the coughing settles where
starts equal stops. The knobs are the fields of `Settings`; `beta` is how harsh the smoke is. Her run:

    from arrangements.town import eve_town, run, summary, Settings
    for beta in (.5, 2, 8):
        town = eve_town(settings=Settings(beta=beta))
        print(beta, summary(town, run(town)))

Whatever beta is, the plant keeps making the same power and never filters its smoke.
`tests/test_town_story.py` checks each sentence of §1.6.1 that this code stands behind.

For agents (P29): composed from systems with one-way observation and payment records.

Model choices and predictions: notes/P29-TOWN-PREDICTIONS-2026-09-28.md.
A profit maximizer uses a quadratic expected-liability/premium schedule. A court
learns its coefficient from delayed health observations; an insurer reads that
estimate or the mathematician's earlier estimate. Forecasts never create money.
This is a new composite using the same Arrangement/CotCoalg primitives as
system_id, insurance, and account; it does not reuse their example-specific boxes.

Stage (d), `clean=True` (notes/P29-D-PREDICTIONS-2026-09-28.md): the plant has two dials,
output x and cleanliness c, descended together in its own coordinates. Power x is published
(in (d) no box consumes it; (e) sends it to the residents); smoke s = (1-c)x reaches the residents and is what the records observe. The plant
publishes power and smoke itself, so its interface is power and smoke out, with liability in
from stage 2 on, as the book draws it (§1.6.1 without the liability wire, §1.6.2 with it); its
state (x, c) stays inside. Its cost adds the priced liability
on smoke and a cleaning cost clean_cost*c^2/2. With clean=False every stage is exactly the
one-dial town above.

Stage (e), `power_to_town=True` with `clean=True` (notes/P29-E-PREDICTIONS-2026-09-29.md):
the residents take power as well as smoke. They draw each watt and it is worth `benefit` to
them, so their one-form on the power input is the constant `benefit`: that wire pushes back
(retroactivity), while their one-form on smoke stays zero (an insulator). The plant's cost
loses its revenue term, which now reaches it as that push, pulled back through its power
output. Every trajectory is (d)'s.

Stage (f), `diligence=True` at stage 3 or 4 (notes/P29-F-PREDICTIONS-2026-09-29.md): the insurer
does its own due diligence. A record node 'books' keeps three running sums: S_uy = sum u*y and
C = sum u^2 over the claims it has paid (the delayed records, u = s^2/2, y = incidence), and
E = sum u^2 over the exposure it has written (known at once). Its own fit is least squares over
every claim so far, own = S_uy/C, and its rate descends (rate - target)^2/2 with
target = est + (C/E)*(own - est): its own fit carries the share of the exposure's evidence whose
claims have arrived, her estimate (or the court's fit, report disconnected) the rest. With
diligence=False every stage is unchanged.
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
    coughing0: float = 0.0
    learn_rate: float = 40.0
    quote_rate: float = 2.0
    delay_steps: int = 300
    clean_cost: float = 0.1
    clean0: float = 0.0


def own_cost(label, initial, inputs, cost, mobility, publish=None, push=None):
    """Conditional cost differentiated only on the box's configuration.

    Zero input covector is the declared one-way boundary. The smooth input map
    retains its genuine derivative: no stop_gradient or custom derivative.
    """
    gradient = jax.grad(cost, argnums=0)
    return node(label, initial, inputs, lambda q, e: -gradient(q, e), mobility, publish, push)


def node(label, initial, inputs, force, mobility, publish=None, push=None):
    """A system: `inputs` values in, `publish(q)` out (its configuration by default).

    `push(q, e)` is its one-form on its inputs, the push it returns up each input wire;
    zero by default, the one-way boundary.
    """
    initial = jnp.atleast_1d(jnp.asarray(initial, dtype=float))
    publish = publish or (lambda q: q)
    push = push or (lambda q, e: jnp.zeros_like(e))
    return Arrangement(
        P=configuration(euclidean(len(initial), mobility), init=initial),
        in_dim_M=0, out_dim_M=0, in_dim_N=inputs, out_dim_N=len(publish(initial)),
        out_f=lambda q, m: publish(q),
        in_f=lambda q, m, e: EMPTY,
        omega=lambda q, m, e: (force(q, e), EMPTY, push(q, e)),
        label=label)


@dataclass
class Town:
    arr: Arrangement
    slots: dict
    settings: Settings
    stage: int
    clean: bool = False

    def field(self, q):
        return self.arr.omega(q, EMPTY, EMPTY)[0]

    def initial(self):
        return self.arr.P.position(self.arr.P.init)

    def value(self, trajectory, name):
        return np.asarray(trajectory)[..., self.slots[name]]


def build(stage, settings=Settings(), *, connect_report=True, informed=False, clean=False,
          power_to_town=False, diligence=False, extra=(), exact_plant=False):
    """Stages 1=no return, 2=law, 3=historical insurance, 4=her account.

    Informed control starts the liability estimator at the true coefficient.
    It uses the same delayed payouts. It is an explicit information advantage.
    clean=True gives the plant its second dial (stage (d)); slots 'output' and
    'cleanliness' then index the two coordinates of the one box 'plant'.
    power_to_town=True (stage (e), needs clean=True) sends the power to the residents,
    who push back on it at `benefit` per watt; the plant's revenue arrives that way.
    diligence=True (stage (f), stage 3 or 4) gives the insurer its books and its own claims fit.
    extra=[(box, source), ...] wires further boxes into the town before it is closed (book P26 (b),
    `tests/test_composition_audit.py`): `box` a system built like `node`'s, `source` the name of a
    published value ('smoke', 'estimate', 'premium', ...) or a function of the published vector.
    exact_plant=True gives the plant its cost's covector on the price it reads, so it pushes back on the
    premium (a lobbying plant); the default plant is a price-taker, the declared one-way boundary.
    """
    if stage not in (1, 2, 3, 4) or settings.delay_steps < 0:
        raise ValueError('stage must be 1..4 and delay_steps nonnegative')
    if power_to_town and not clean:
        raise ValueError('power_to_town needs the two-dial plant (clean=True)')
    if diligence and stage < 3:
        raise ValueError('diligence needs an insurer (stage 3 or 4)')
    revenue = 0.0 if power_to_town else settings.benefit
    c = settings
    boxes, slots, outs, sources = [], {}, {}, []
    offset = sent = 0

    # slots index the configurations (what Town.value reads); outs index what each box
    # publishes (what the wiring reads). They differ only for the two-dial plant.
    def add(box, source):
        nonlocal offset, sent
        slots[box.label] = slice(offset, offset + box.P.Q_dim)
        outs[box.label] = slice(sent, sent + box.out_dim_N)
        offset += box.P.Q_dim
        sent += box.out_dim_N
        boxes.append(box)
        sources.append(source)

    def at(m, name):
        return m[outs[name]]

    # Eve's first plant (stage 1) has no liability port, as the book draws it (§1.6.1); the
    # port, and the price that arrives on it, come from stage 2 on.
    priced = stage >= 2

    def price(q):
        if not priced:
            return EMPTY
        return at(q, 'premium' if stage >= 3 else 'assessed')

    def rate(e):
        return e[0] if priced else 0.0

    def pushback(cost):
        # exact_plant: the covector -d_e cost on the price read, the push an exact reader returns
        if not exact_plant:
            return None
        return lambda q, e: -jax.grad(cost, argnums=1)(q, e)

    if clean:
        plant_cost = (lambda q, e: .5*(1 + rate(e)*(1-q[1])**2)*q[0]**2 - revenue*q[0]
                      + .5*c.clean_cost*q[1]**2)
        add(own_cost('plant', [c.output0, c.clean0], int(priced), plant_cost, c.dt,
                     publish=lambda q: jnp.array([q[0], (1-q[1])*q[0]]),
                     push=pushback(plant_cost)), price)
        slots['output'], slots['cleanliness'] = slice(0, 1), slice(1, 2)
        outs['output'], outs['smoke'] = slice(0, 1), slice(1, 2)
        outs['power'] = outs['output']

        def smoke(m):
            return at(m, 'smoke')

        def cleanliness(m):
            # the ledger reads smoke per watt, s/x = 1-c; output stays positive in every run
            x = at(m, 'output')
            return 1 - at(m, 'smoke')/jnp.where(x > 0, x, 1.0)
    else:
        plant_cost = lambda q, e: .5 * (1 + rate(e)) * q[0]**2 - c.benefit * q[0]
        add(own_cost('output', [c.output0], int(priced), plant_cost, c.dt,
                     push=pushback(plant_cost)), price)

        def smoke(m):
            return at(m, 'output')
    if power_to_town:
        # inputs (power, smoke): each watt drawn is worth `benefit`, pushed back up the power
        # wire; smoke returns nothing
        add(own_cost('coughing', [c.coughing0], 2,
                     lambda q, e: .5*c.recovery*q[0]**2 - q[0]*c.beta*e[1]**2/2, c.dt,
                     push=lambda q, e: jnp.array([c.benefit, 0.0])),
            lambda m: jnp.concatenate([at(m, 'power'), smoke(m)]))
    else:
        add(own_cost('coughing', [c.coughing0], 1,
                     lambda q, e: .5*c.recovery*q[0]**2 - q[0]*c.beta*e[0]**2/2, c.dt),
            smoke)

    if stage >= 2:
        add(node('memory', [c.output0, c.coughing0, 0.0], 3, lambda q, e: e-q, 1.0),
            lambda q: jnp.concatenate([smoke(q), at(q, 'coughing'), jnp.ones(1)]))

        def observed(q):
            old = at(q, 'memory')
            incident = (at(q, 'coughing')[0]-old[1])/c.dt + c.recovery*old[1]
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
            prior = 'estimate' if stage == 4 and connect_report else 'assessed'
            if diligence:
                def target(e):
                    # e = (prior, S_uy, C, E): own least-squares fit on paid claims, weighted by
                    # the share C/E of the exposure's evidence whose claims have arrived
                    guess, s_uy, paid, written = e
                    own = s_uy/jnp.where(paid > 0, paid, 1.0)
                    share = paid/jnp.where(written > 0, written, 1.0)
                    return guess + share*(own - guess)
                add(own_cost('premium', [0.0], 4,
                             lambda q, e: .5*(q[0]-target(e))**2, c.dt*c.quote_rate),
                    lambda q: jnp.concatenate([at(q, prior), at(q, 'books')]))
            else:
                add(own_cost('premium', [0.0], 1,
                             lambda q, e: .5*(q[0]-e[0])**2, c.dt*c.quote_rate),
                    lambda q: at(q, prior))
        if stage == 4:
            add(own_cost('estimate', [0.0], 2, fit_cost, c.dt*c.learn_rate), observed)
        if diligence:
            # inputs (u, y) of the claim settled now and u of the exposure written, a step back
            # so that its claim settles exactly one delay later
            add(node('books', [0., 0., 0.], 3,
                     lambda q, e: jnp.array([e[0]*e[1], e[0]**2, e[2]**2]), c.dt),
                lambda q: jnp.concatenate([settled(q), observed(q)[:1]]))

        def payments(q):
            output = at(q, 'output')[0]
            operating = c.benefit*output-output**2/2
            if clean:
                operating = operating - c.clean_cost*cleanliness(q)[0]**2/2
            claims = settled(q)[1]
            if stage >= 3:
                premium = at(q, 'premium')[0]*smoke(q)[0]**2/2
                return jnp.array([operating-premium, premium-claims, claims])
            return jnp.array([operating-claims, 0.0, claims])

        add(node('money', [0., 0., 0.], 3, lambda q, e: e, c.dt), payments)

    for box, source in extra:
        add(box, (lambda m, name=source: at(m, name)) if isinstance(source, str) else source)

    inner = tensor_arrangements(boxes)
    outer = Arrangement(
        P=trivial_parameter(), in_dim_M=sum(b.in_dim_N for b in boxes),
        out_dim_M=sent, in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: EMPTY,
        in_f=lambda q, m, e: jnp.concatenate([source(m) for source in sources]),
        omega=zero_form, label='town wiring')
    return Town(compose_seq(inner, outer), slots, c, stage, clean)


def eve_town(stage=1, settings=Settings(), **options):
    """Eve's town from the book's §1.6.1: the two-dial plant and townspeople who draw its power.

    Stage 1 is her first model, whose plant has no liability port; stages 2-4 add the court,
    the insurer and her account, as the book's later subsections do.
    """
    return build(stage, settings, clean=True, power_to_town=True, **options)


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


def smoke(town, trajectory):
    """Smoke reaching the residents: output, or (1-c)*output with the second dial."""
    output = town.value(trajectory, 'output')[..., 0]
    if town.clean:
        return (1-town.value(trajectory, 'cleanliness')[..., 0])*output
    return output


def summary(town, trajectory):
    c = town.settings
    output = town.value(trajectory, 'output')[:, 0]
    s = smoke(town, trajectory)
    result = {'harm': float(c.dt*np.sum(c.beta*s[:-1]**2/2)),
              'peak_output': float(np.max(output)), 'final_output': float(output[-1])}
    if town.clean:
        result['final_cleanliness'] = float(town.value(trajectory, 'cleanliness')[-1, 0])
        result['final_smoke'] = float(s[-1])
        result['final_coughing'] = float(town.value(trajectory, 'coughing')[-1, 0])
        result['power'] = float(c.dt*np.sum(output[:-1]))
    if 'books' in town.slots:
        books = town.value(trajectory, 'books')
        paid = np.flatnonzero(books[:, 1] > 0)
        result['first_claim_time'] = float(paid[0]*c.dt) if len(paid) else None
        result['final_own_fit'] = float(books[-1, 0]/books[-1, 1]) if len(paid) else None
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
