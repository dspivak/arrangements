"""The town of *Articulating Forms that Matter*, Chapter 1 (``sec.calculus_working``, ``sec.better``,
``sec.cost_later``; linked from the footnotes in ``sec.calculus_working`` and ``sec.check_claims``).

For a reader of the book. Eve draws her town as an arrangement of two systems (``eqn.town_system``), and
``eve_town()`` builds it, a composite of two boxes and their wires:

* The *plant*, with power and smoke out. It is run for profit. Its state is its output ``x`` and its cleanliness
  ``c``, and its smoke is ``s = (1-c) x``. Each watt sells for ``benefit``, more watts cost more to make
  (``x^2/2``), and filtering costs money (``clean_cost c^2/2``) and brings in nothing, so its effort field invites
  output up to where the next watt costs what it brings in, and never invites filtering.
* The *townspeople*, with power and smoke in and coughing out. Their state is how much they cough (the slot
  ``coughing``). Smoke starts coughs, ``beta s^2/2`` a day, and coughs heal at the rate ``recovery``, so the
  coughing settles where starts equal stops.
* The power wire pushes back: the townspeople draw each watt and pay ``benefit`` for it, and that is how the plant's
  revenue reaches it. The smoke wire does not push back; it is the waterfall of ``sec.waterfall``. In ``own_cost``
  and ``node``, ``push`` is what a box returns up its input wires, and its default, zero, is that one-way wire.

The knobs are the fields of ``Settings``; ``beta`` is how harsh the smoke is. Her run, run from the library's folder with it on the path (``PYTHONPATH=. .venv/bin/python example.py``):

    import jax; jax.config.update("jax_enable_x64", True)
    from arrangements.town import eve_town, run, summary, Settings
    for beta in (.5, 2, 8):
        town = eve_town(settings=Settings(beta=beta))
        r = summary(town, run(town))
        print(beta, round(r['final_output'], 3), r['final_cleanliness'], round(r['final_coughing'], 3))
    # 0.5 1.0 0.0 0.25
    # 2 1.0 0.0 1.0
    # 8 1.0 0.0 4.0

With milder smoke they cough less, with harsher smoke more, and either way the plant keeps making the same power
and never filters. ``tests/test_town_story.py`` checks each sentence of that story.

Her account is ``account_town()``, the town of ``eqn.town_account`` exactly as drawn: plant, townspeople, estimator
and insurer, and nothing else in the composite (``tests/test_town_account.py`` says how to run it).

``eve_town(stage)`` is the same town with what the book talks about and does not draw: stage 2 a court that fits
claims arriving after a delay, its assessment on the plant's liability wire (``eqn.town_liability`` draws that wire
arriving from outside the town and does not run it); stage 3 an insurer who reads the court; stage 4 her estimate as
well, read through a box that remembers yesterday, so its state is not the one number of ``account_town``'s
estimator. Beside these are records (``memory``, ``delay``, ``money``, and with ``diligence=True`` the insurer's
``books``). ``build`` is the general constructor, with a plant of one dial for the earlier runs. Every town is a
closed composite; ``run`` takes its effort field and integrator step by step, checked equal to the dynamics functor
``Phi`` in ``tests/test_town_smoke.py`` and ``tests/test_town_account.py``.

Artificial, on purpose: one coefficient, known recovery, noiseless records, instant smoke-to-coughing. The model
does not say what a real court or insurer would do.

For agents: the file's earlier head, with the predictions fixed before the code, what happened and the
weirdness checks, is in ``notes/P31-B-CH1-HISTORY-2026-11-03.md``; the model's choices in ``notes/P29-TOWN-PREDICTIONS-2026-09-28.md``.
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
          power_to_town=False, diligence=False, extra=(), exact_plant=False, fit_coughing=False):
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
    fit_coughing=True (stage 4) has the estimate read the smoke and the coughing directly, as `account_town`'s
    estimator does, in place of the delayed-by-one-step new coughing it reads through the memory box.
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
    def add(box, source, name=None):
        nonlocal offset, sent
        name = name or box.label
        slots[name] = slice(offset, offset + box.P.Q_dim)
        outs[name] = slice(sent, sent + box.out_dim_N)
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
        # wire; smoke returns nothing. The box is the book's townspeople; its state is their coughing.
        add(own_cost('townspeople', [c.coughing0], 2,
                     lambda q, e: .5*c.recovery*q[0]**2 - q[0]*c.beta*e[1]**2/2, c.dt,
                     push=lambda q, e: jnp.array([c.benefit, 0.0])),
            lambda m: jnp.concatenate([at(m, 'power'), smoke(m)]), name='coughing')
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
            reads = observed
            if fit_coughing:
                reads = lambda q: jnp.concatenate([smoke(q)**2/2, c.recovery*at(q, 'coughing')])
            add(own_cost('estimate', [0.0], 2, fit_cost, c.dt*c.learn_rate), reads)
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


def account_boxes(settings=Settings()):
    """The four systems of the book's ``eqn.account_systems`` and ``eqn.town_account``, each a box with the
    ports the book draws: plant (premium in; power, smoke out), townspeople (power, smoke in; coughing out),
    estimator (coughing, smoke in; estimate out), insurer (estimate in; premium out)."""
    c = settings
    plant = own_cost('plant', [c.output0, c.clean0], 1,
                     lambda q, e: .5*(1 + e[0]*(1-q[1])**2)*q[0]**2 + .5*c.clean_cost*q[1]**2, c.dt,
                     publish=lambda q: jnp.array([q[0], (1-q[1])*q[0]]))
    townspeople = own_cost('townspeople', [c.coughing0], 2,
                           lambda q, e: .5*c.recovery*q[0]**2 - q[0]*c.beta*e[1]**2/2, c.dt,
                           push=lambda q, e: jnp.array([c.benefit, 0.0]))
    estimator = own_cost('estimator', [0.0], 2,
                         lambda q, e: .5*(q[0]*e[1]**2/2 - c.recovery*e[0])**2, c.dt*c.learn_rate)
    insurer = own_cost('insurer', [0.0], 1, lambda q, e: .5*(q[0]-e[0])**2, c.dt*c.quote_rate)
    return [plant, townspeople, estimator, insurer]


def account_town(settings=Settings()):
    """The town of the book's ``eqn.town_account``, exactly as drawn: the four boxes of `account_boxes` and
    the wires between them, the smoke copied at its dot, nothing else in the composite. A `Town`, so `run`,
    `summary` and `smoke` take it; slots 'output', 'cleanliness', 'coughing', 'estimate', 'premium'."""
    boxes = account_boxes(settings)
    power, smoke_, coughing, estimate, premium = range(5)     # what the boxes publish, in order
    feeds = jnp.array([premium, power, smoke_, coughing, smoke_, estimate])
    wiring = Arrangement(
        P=trivial_parameter(), in_dim_M=sum(b.in_dim_N for b in boxes), out_dim_M=5,
        in_dim_N=0, out_dim_N=0,
        out_f=lambda q, m: EMPTY,
        in_f=lambda q, m, e: m[feeds],
        omega=zero_form, label='town wiring')
    slots = {'output': slice(0, 1), 'cleanliness': slice(1, 2), 'coughing': slice(2, 3),
             'estimate': slice(3, 4), 'premium': slice(4, 5)}
    return Town(compose_seq(tensor_arrangements(boxes), wiring), slots, settings, 4, True)


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
