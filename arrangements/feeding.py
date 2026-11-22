"""Feeding as a composite in IMC's Cartesian tangent subcategory.

The spatial coupling takes a fixed receiver beside its casing sources (feeding-imc.tex, eqn.feed). Objects
are interfaces, morphisms are Arrangement values, composition and tensor are
the library constructors. No new categorical construction is claimed.
Their integrators read configurations, forgetting mass momenta. The existing
Phi / PhiK interpretation computes the inclination; the parameter responds.
The browser's factored field is checked against this composite, including its
six source-output loads. A zero eye capacity is a browser limiting convention,
not a rod-segment integrator, so the library requires positive capacities.

``feeding`` is built from three constructors, ``seeker`` (the note's receiver; ``receiver``
is the same function), ``source`` and ``spatial_coupling``, each composed by
``compose_seq`` and ``tensor_arrangements`` from the note's general parts, in
``feeding_composed.py`` and ``feeding_parts.py`` (book P109 (e), 2026-11-19, the casings and
the body; P109 (h), 2026-11-21, the mouth's organs and the spatial coupling). The one field
still typed out as a whole is inside a circle: the memory's rates, ``_food_memory``.
``source_by_hand``, ``_body_by_hand``, ``receiver_by_hand`` and ``_encounter_by_hand`` are
the earlier single boxes, kept as checks; regrouping changed no equation.
"""
from __future__ import annotations
import functools
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, CotCoalgK
from .wiring import compose_seq, parallel_arrangements, tensor_arrangements, trivial_parameter

DEFAULTS = dict(m=1., g=2., beta=.2, kappa=40., x0=0., R=.5, kappa0=100., b0=1., f0=.1,
                lgrip=2., lpry=2., leat=2., a1=.2, a2=.6, alpha=.8, cgrip=.02, ctear=.3,
                xg=-.4, kg=400., eps=.05, Rh=.5, clift=.02, llift=2., M=.25, betac=.4,
                kappac=40., ktau=.16, A=6., omega=.6, gam=.05, lam=3., w=.5, csee=.002, fr=.1,
                kd=4000., bd=40., kr=10., epsd=.01, B=8., nup=.1, gamp=.005, cpat=.002, kexp=.1,
                latchRate=40., latchCost=.0001, memoryDrive=.2, memoryCost=.001, searchSeed=37)

def theta(t):
    # Safe branches prevent inactive 1/0 branches poisoning derivatives.
    # exp(-1/x) is below 1e-304 for x < 1/700, so it is taken as 0 there. That changes no
    # value above 1e-304 and keeps every derivative finite: with the cut at 0, a derivative
    # at a tiny positive x is 0 * inf in floating point (the first below 1e-154, the second
    # below 1e-77), and a composite pulls back through one more derivative of each circle
    # than a field typed out by hand takes.
    def rho(x):
        return jnp.where(x > 1/700, jnp.exp(-1 / jnp.where(x > 1/700, x, 1.)), 0.)
    a, b = rho(t), rho(1-t)
    return a/(a+b)

def near(u, radius):
    return theta((radius*radius-u*u)/(.75*radius*radius))

def _parameter(init, indices, response, stages, h, label):
    init = jnp.asarray(init, dtype=float)
    read = lambda s: s[jnp.asarray(indices, dtype=int)]
    if stages == 1:
        advance = lambda s, xi, tau: s+tau*response(s, xi)
        return CotCoalg(len(indices), init, read, lambda s, xi: advance(s, xi, h), label,
                        timed_step=lambda s, xi, tau: advance(s, xi, h*tau))
    if stages != 4:
        raise ValueError('Feeding supports one Euler round or four RK4 rounds')
    def mid(s, index, scale):
        x, ks=s
        return x+h*scale*ks[index]
    reads=(read, lambda s: read(mid(s, 0, .5)), lambda s: read(mid(s, 1, .5)), lambda s: read(mid(s, 2, 1.)))
    advances=(lambda s, xi: (s, (response(s,xi),)),
              lambda s, xi: (s[0],s[1]+(response(mid(s,0,.5),xi),)),
              lambda s, xi: (s[0],s[1]+(response(mid(s,1,.5),xi),)),
              lambda s, xi: s[0]+h/6*(s[1][0]+2*s[1][1]+2*s[1][2]+response(mid(s,2,1.),xi)))
    return CotCoalgK(len(indices), init, reads, advances, label=label,
                    durational=lambda tau: _parameter(init,indices,response,4,h*tau,label))

def source(place, food=10., *, parameters=None, stages=1, h=1.):
    """Nullary source with six outputs: food, opening, access, height, red, place.

    Composed by the library from the note's parts (``feeding_composed.source``): three
    integrands, gravity, the air's dragger and the ground's spring and damper.
    """
    from .feeding_composed import source as composed
    return composed(place, food, parameters=parameters, stages=stages, h=h)


def source_by_hand(place, food=10., *, parameters=None, stages=1, h=1.):
    """The same source as one box, its summed field typed out. Kept only as a check on
    ``source`` (``tests/test_feeding_composed.py``); the model does not use it."""
    p={**DEFAULTS, **(parameters or {})}
    def response(s, xi):
        return jnp.array([xi[0],xi[1],s[3]/p['m'],xi[2],xi[3]/p['beta'],xi[4]/p['bd']])
    P=_parameter([food,0.,p['xg'],0.,p['xg'],0.],[0,1,2,4,5],response,stages,h,'casing integrator')
    def cost(q,m,n):
        f,a,x,z,zg=q;v=x-p['xg'];dg=v-zg
        return (p['m']*p['g']*x+.5*p['kg']*theta(-v/p['eps'])*v*v+.5*p['kappa']*(x-z)**2
                +.5*p['kd']*theta(-v/p['epsd'])*dg*dg+.5*p['kr']*zg*zg)
    def out(q,m):
        f,a,x,z,zg=q
        return jnp.array([f,a,theta((a-p['a1'])/(p['a2']-p['a1'])),x,theta((f-p['fr'])/p['fr']),place])
    return Arrangement(P,0,0,0,6,out,lambda q,m,n:jnp.zeros(0),exact(lambda q,m,n:-cost(q,m,n)),label=f'casing({place:g})')

CENTRES = tuple(range(-9, 10))


def _search_map(weights, freshness):
    centres = jnp.asarray(CENTRES)
    basis = jnp.exp(-(centres[:, None] - centres[None, :])**2)
    usual = basis @ weights
    richness = usual * theta(usual / .05)
    return richness, .00001 + freshness**2 * (.012 + richness**2)


def _variate(seed, index):
    # Counter-based pseudorandom tape; all operations before conversion are uint32.
    x = jnp.asarray(seed, jnp.uint32) + (index.astype(jnp.uint32)+1)*jnp.uint32(0x9e3779b9)
    x = (x ^ (x >> 16))*jnp.uint32(0x21f0aaad)
    x = (x ^ (x >> 15))*jnp.uint32(0x735a2d97)
    return ((x ^ (x >> 15)).astype(float) + .5)/4294967296.


def _destination(masses, u):
    centres = jnp.asarray(CENTRES)
    left, right = CENTRES[0], CENTRES[-1]
    scale = jnp.where((centres == left) | (centres == right), 2., 1.)
    threshold = masses.sum()*u
    def bisect(_, bounds):
        lo, hi = bounds
        x = (lo+hi)/2
        cdf = scale*(theta((x-centres+1)/2) - .5*(centres == left))
        below = jnp.dot(masses, cdf) < threshold
        return jnp.where(below,x,lo), jnp.where(below,hi,x)
    lo, hi = jax.lax.fori_loop(0, 32, bisect, (float(left),float(right)))
    return (lo+hi)/2


def _return_rate(weight):
    return .1 * jax.nn.sigmoid(weight-jnp.log(7.))


def _food_memory(q, grip, seen, eat, p):
    """Learn typical yield at release and local return speed on a later look.

    The rates the memory's motors are told. Each motor reads the battery and drives its store
    at ``live(b)`` times its rate, so no rate here carries ``live``; ``grip`` is the rate the
    gripper tells its motor."""
    y, gaze, meal = q[3], q[3] + q[5], q[7]
    centres = jnp.asarray(CENTRES)
    size = len(CENTRES)
    usual, fresh = q[8:8+size], q[8+size:8+2*size]
    target, angle = q[8+2*size:10+2*size]
    weights, contact, empty, age = q[10+2*size:].reshape(4, size)
    here = jnp.exp(-(y-centres)**2)
    there = jnp.exp(-(gaze-centres)**2)
    release = jax.grad(theta)(q[1]) * (-grip) * theta(-grip/.05)
    meal_rate = eat - 4 * theta(-q[1]) * meal
    usual_rate = release * (-jnp.expm1(-meal/4) - jnp.dot(usual, here)) * here
    _, masses = _search_map(usual, fresh)
    speed = _return_rate(weights)
    fresh_rate = (speed*(1-fresh) - 4*(here+there)*fresh)
    looking = near(gaze-centres, .55)
    contact_level, last_empty = 10*contact, 10*empty
    dc = 12*(looking-contact_level)
    arrival = 2.5*jax.vmap(jax.grad(theta))(2.5*(contact_level-.1))*dc*theta(dc/.1)
    write = 12*looking*theta((contact_level-.55)/.25)
    cleared = theta((last_empty-.5)/.4)
    red = theta(seen/.5)
    def revisit_loss(w, elapsed):
        prediction = -jnp.expm1(-_return_rate(w)*elapsed)
        return .5*(prediction-red)**2
    weight_rate = -4*arrival*cleared*jax.vmap(jax.grad(revisit_loss))(weights, 80*age)
    empty_rate = .1*write*(1-red-last_empty)
    age_rate = cleared*(1-looking)/80 - write*age
    choosing = theta((jnp.cos(angle)-.4)/.35)
    finish = theta((jnp.cos(angle)-.1)/.25)
    near_goal = theta((2.25-(target-y)**2)/1.25)
    searching = theta(1-2*seen)
    clock_rate = searching*(near_goal + (1-near_goal)*finish)
    def select(_):
        n = jnp.floor(angle/(2*jnp.pi)+.5)
        return 4*searching*choosing*(_destination(masses, _variate(p['searchSeed'], n))-target)
    goal_rate = jax.lax.cond(choosing>0, select, lambda _: jnp.array(0.), None)
    rates = jnp.concatenate([jnp.array([meal_rate]), usual_rate, fresh_rate, jnp.array([goal_rate,clock_rate]),
                             weight_rate, .1*dc, empty_rate, age_rate])
    return rates, p['memoryDrive']*jnp.tanh((target-y)/1.5)


def _body(*, parameters, stages, h, learning, memory, cart):
    """Body stores and motors. Its seven inputs summarize the current encounter.

    Composed by the library from the note's parts (``feeding_composed.body``): gripper,
    lever, wheels, the eye's sweep and one motor per memory store, with their stores.
    """
    from .feeding_composed import body as composed
    return composed(parameters=parameters, stages=stages, h=h, learning=learning, memory=memory, cart=cart)


def _body_by_hand(*, parameters, stages, h, learning, memory, cart):
    """The same body as one box, its summed field typed out. Kept only as a check on
    ``_body`` (``tests/test_feeding_composed.py``); the model does not use it."""
    p={**DEFAULTS, **(parameters or {})}
    if learning:
        zeros = jnp.zeros(len(CENTRES))
        saved = memory if isinstance(memory, dict) else {'usual': zeros if memory is None else memory}
        usual = jnp.asarray(saved.get('usual', zeros), dtype=float)
        weights = jnp.asarray(saved.get('returnWeights', zeros), dtype=float)
        if usual.shape != zeros.shape or weights.shape != zeros.shape:
            raise ValueError("expected one yield and one return parameter per center")
        fresh = jnp.ones_like(usual)
        _, masses = _search_map(usual, fresh)
        target = _destination(masses, _variate(p['searchSeed'],jnp.array(0)))
        extra = [0., *usual, *fresh, target, 1.7, *weights, *zeros, *zeros, *zeros]
    else:
        extra = [0.,p['B']]
    strokes = 7 + len(extra)
    def response(s,xi):
        main=jnp.array([xi[0],xi[1]/p['cgrip'],xi[2]/p['clift'],s[4]/p['M'],xi[3],xi[4]/p['betac'],xi[5]/p['csee'],xi[6]/p['csee']])
        capacity = p['memoryCost'] if learning else p['cpat']
        return jnp.concatenate([main,xi[7:]/capacity])
    init=[6. if learning else 3.,-1.,p['xg'],cart,0.,cart,0.,p['A']]+extra
    P=_parameter(init,[0,1,2,3]+list(range(5,len(init))),response,stages,h,'receiver integrator')
    def omega(q,m,n):
        b,l,h,y,zw,r,sc=q[:7]; live=theta(b/p['b0']);hold=theta(l)
        seen,climb,keep,pl,ph,_,eaten=n
        qq=jnp.zeros_like(q)
        grip=p['lgrip']*live*(keep*(2-l)-(1-keep)*(l+1))
        lift=p['llift']*live*(hold*(p['x0']-h)+(1-hold)*(p['xg']-h))
        grip_torque=p['cgrip']*grip-pl;lift_torque=p['clift']*lift-ph
        qq=qq.at[1].add(grip_torque).at[2].add(lift_torque).at[0].add(-grip_torque*grip-lift_torque*lift)
        free=theta(1-2*seen);ring=p['A']**2-r*r-sc*sc
        look=live*(free*(p['omega']*sc+p['gam']*ring*r)+p['lam']*climb)
        scan=live*free*(-p['omega']*r+p['gam']*ring*sc)
        qq=qq.at[5].add(p['csee']*look).at[6].add(p['csee']*scan).at[0].add(-p['csee']*(look*look+scan*scan))
        if learning:
            # Each memory motor reads the battery: its rule is written with live = 1 and the
            # motor drives at live(b) times the rate it is told.
            rates, search = _food_memory(q[:strokes], p['lgrip']*(keep*(2-l)-(1-keep)*(l+1)), seen, eaten, p)
            rates = live * rates
            capacity = p['memoryCost']
        else:
            eta,zeta=q[7:9]
            ringp=p['B']**2-eta*eta-zeta*zeta
            rates=live*free*jnp.array([p['nup']*zeta+p['gamp']*ringp*eta,
                                      -p['nup']*eta+p['gamp']*ringp*zeta])
            search=p['kexp']*(eta-y)
            capacity=p['cpat']
        qq=qq.at[7:strokes].add(capacity*rates).at[0].add(-capacity*jnp.dot(rates,rates))
        spring=p['kappac']*(y-zw);torque=live*(p['ktau']*seen*r+free*search)
        qq=qq.at[3].add(-spring).at[4].add(spring+torque).at[0].add(-torque*(spring+torque)/p['betac'])
        return qq,jnp.zeros_like(m),jnp.zeros_like(n)
    return Arrangement(P,0,0,7,5,lambda q,m:q[jnp.array([0,1,2,3,5])],
                       lambda q,m,n:jnp.zeros(0),omega,label='body and motors')


def _mouth_organ(name, width, capacity, battery_per_unit, flow_index, battery_scale):
    """Expose this organ's available effort and settle its realized flow at the battery."""
    def output(q, body):
        effort = capacity * theta(body[0]/battery_scale) * theta(body[1])
        return jnp.append(body, effort)

    def exchange(q, body, surroundings):
        load = jnp.zeros_like(body).at[0].set(battery_per_unit * surroundings[flow_index])
        return jnp.zeros_like(q), load, jnp.zeros_like(surroundings)

    return Arrangement(trivial_parameter(), 7, width, 7, width+1, output,
                       lambda q,m,n:n, exchange, label=name)


def seeker(*, parameters=None, stages=1, h=1., learning=False, memory=None, cart=0.):
    """One seeker: the receiver ``R`` of the note. Its interface, organs and stores are
    independent of its partners.

    Composed by the library (``feeding_composed.seeker``): the body and its motors beside
    the opener and the digestor, each organ a ``push`` that settles its realized rate at the
    battery.
    """
    from .feeding_composed import seeker as composed
    return composed(parameters=parameters, stages=stages, h=h, learning=learning, memory=memory, cart=cart)


receiver = seeker


def receiver_by_hand(*, parameters=None, stages=1, h=1., learning=False, memory=None, cart=0.):
    """The same receiver with each mouth organ one box, its field typed out. Kept only as a
    check on ``seeker`` (``tests/test_feeding_composed.py``); the model does not use it."""
    p = {**DEFAULTS, **(parameters or {})}
    body = _body(parameters=p, stages=stages, h=h, learning=learning, memory=memory, cart=cart)
    opener = _mouth_organ('opener', 5, p['lpry'], -p['ctear'], 5, p['b0'])
    digestor = _mouth_organ('digestor', 6, p['leat'], p['alpha'], 6, p['b0'])
    return compose_seq(compose_seq(body, opener), digestor)


def spatial_coupling(count, *, parameters=None, stages=1, h=1.):
    """The spatial coupling ``W_n`` of one seeker and ``count`` sources (note ``eqn.feed``).

    Composed by the library (``feeding_composed.spatial_coupling``): per casing a contact
    and a latch, two pushes for the casing side of opening and eating, and one static map.
    """
    from .feeding_composed import spatial_coupling as composed
    return composed(count, parameters=parameters, stages=stages, h=h)


def _encounter_by_hand(count, p, stages, h):
    """The same spatial coupling as one box, its field typed out. Kept only as a check on
    ``spatial_coupling`` (``tests/test_feeding_composed.py``); the model does not use it.
    Each contact owns its stroke and latch."""
    def respond(s,xi):
        return jnp.concatenate([xi[:count]/p['bd'],xi[count:]/p['latchCost']])
    strokes = _parameter([0.]*(2*count), list(range(2*count)), respond,
                         stages, h, 'contact strokes and latches')

    def contacts(m):
        return m[7:].reshape((count, 6)).T

    def engagement(z):
        return theta((z[count:]-.3)/.2)

    def spring_energy(z, m):
        _, grip, tip, cart, _, _, _ = m[:7]
        _, _, _, height, _, place = contacts(m)
        stretch = height - tip
        gate = engagement(z)*theta(grip)*near(stretch,p['R'])*near(cart-place,p['Rh'])
        stroke = z[:count]
        return .5*jnp.sum(gate*(p['kappa0']*stretch**2+p['kd']*(stretch-stroke)**2)+p['kr']*stroke**2)

    spring_grad = jax.grad(spring_energy, argnums=(0,1))

    def available(m):
        food, _, _, height, _, place = contacts(m)
        return near(height-m[2],p['R'])*near(m[3]-place,p['Rh'])*theta((food-p['f0'])/p['f0'])

    def latch_rates(z,m):
        _, _, _, height, _, place = contacts(m)
        offered = available(m)
        score = offered/(1+((height-m[2])/p['R'])**2+((m[3]-place)/p['Rh'])**2)
        others = ~jnp.eye(count,dtype=bool)
        first = jnp.prod(jnp.where(others,theta((score[:,None]-score[None,:])/.05),1.),axis=1)
        latch = z[count:]
        free = jnp.prod(jnp.where(others,1-theta(latch/.3),1.),axis=1)
        held = theta((m[1]+.8)/.4)
        selected = engagement(z)
        acquire = theta((offered-.15)/.15)*first*(1-held)
        retain = 1-(1-offered)*(1-held)
        target = free*((1-selected)*acquire+selected*retain)
        return p['latchRate']*theta(m[0]/p['b0'])*(target-latch)

    def flows(z,m):
        food, opening, access, height, _, place = contacts(m)
        reach = near(m[3]-place,p['Rh'])
        contact = engagement(z)*reach*near(height-p['x0'],p['R'])
        return m[5]*contact*(1-opening), m[6]*contact*access*food

    def inputs(z,m,n):
        _, _, _, cart, look, _, _ = m[:7]
        _, _, _, _, red, place = contacts(m)
        offset = (cart+look-place)/p['w']
        signal = red*jnp.exp(-offset**2)
        keep = jnp.sum(engagement(z)*available(m))
        loads = -spring_grad(z,m)[1]
        opened,eaten = flows(z,m)
        return jnp.array([signal.sum(),jnp.sum(-2*offset*signal/p['w']),keep,
                          loads[1],loads[2],opened.sum(),eaten.sum()])

    def exchange(z,m,n):
        dz,dm = spring_grad(z,m)
        opened,eaten = flows(z,m)
        loads = (-dm).at[7::6].add(-eaten).at[8::6].add(opened)
        rate = latch_rates(z,m)
        torque = p['latchCost']*rate+dz[count:]
        loads = loads.at[0].add(-jnp.dot(torque,rate))
        return (-dz).at[count:].add(torque),loads,jnp.zeros_like(n)

    return Arrangement(strokes,7,7+6*count,0,2,lambda z,m:m[:2],inputs,exchange,
                       label='spatial coupling')


def couple(body, casings, *, parameters=None, stages=1, h=1., by_hand=False):
    """Put this receiver instance in an environment containing the given casing instances:
    ``W_n . (body (x) casing_1 (x) ... (x) casing_n)``. ``by_hand`` takes the typed-out
    coupling in place of the composed one, as a check."""
    p = {**DEFAULTS, **(parameters or {})}
    casings = list(casings)
    sources = tensor_arrangements(casings) if casings else Arrangement(
        trivial_parameter(),0,0,0,0,lambda q,m:m,lambda q,m,n:n,zero_form,label='no casings')
    coupling = (_encounter_by_hand(len(casings),p,stages,h) if by_hand
                else spatial_coupling(len(casings),parameters=p,stages=stages,h=h))
    return compose_seq(parallel_arrangements(body,sources), coupling)


def _contacts(strokes, latches):
    """The spatial coupling's state, which nests by its parts, from the contacts' strokes
    and latches."""
    as_coupling = _unravel_coupling(len(strokes))
    return as_coupling(jnp.concatenate([jnp.asarray(strokes, dtype=float), jnp.asarray(latches, dtype=float)]))


def add_casing(body, casings, state, new_casing, *, parameters=None, stages=1, h=1.):
    """Between complete steps, insert one new source and a free contact into the live world."""
    casings = tuple(casings)
    count = len(casings)
    (body_state, sources), contacts = state
    contacts = ravel_pytree(contacts)[0]
    if contacts.shape != (2*count,):
        raise ValueError("expected a stroke and latch for each existing casing")
    expanded = couple(body, (*casings,new_casing), parameters=parameters, stages=stages, h=h)
    sources = (sources,new_casing.P.init) if count else new_casing.P.init
    contacts = _contacts(jnp.append(contacts[:count],0.), jnp.append(contacts[count:],0.))
    return expanded, ((body_state,sources),contacts)

def feeding(places=(4.,-8.,8.), foods=(.05,10.,10.), *, parameters=None, stages=1, h=1., learning=False, memory=None, cart=0.):
    """Put a fixed receiver and the casing sources side by side in a spatial coupling:
    ``spatial_coupling(n) . (seeker (x) source_1 (x) ... (x) source_n)``, the three
    constructors each composed by the library from the note's parts."""
    if len(places)!=len(foods):raise ValueError('One food per casing')
    p={**DEFAULTS,**(parameters or {})}
    for key in ['m','M','beta','betac','cgrip','clift','csee','cpat','bd','b0','f0','fr','R','Rh','w','eps','epsd','latchCost','latchRate']:
        if p[key]<=0:raise ValueError(f'{key} must be positive')
    if p['a2']<=p['a1']:raise ValueError('a2 must exceed a1')
    if learning:
        if p['memoryCost'] <= 0: raise ValueError('memoryCost must be positive')
    parts=[source(x,f,parameters=p,stages=stages,h=h) for x,f in zip(places,foods)]
    body=receiver(parameters=p,stages=stages,h=h,learning=learning,memory=memory,cart=cart)
    return couple(body,parts,parameters=p,stages=stages,h=h)

_PATROL_STATE = 10   # the receiver's state without learning: eight body numbers and the two patrol stores


@functools.lru_cache(maxsize=None)
def _unravel(learning):
    """How a flat list of numbers nests into the receiver's state and into one casing's:
    read off the composites' own starting states, whatever parts they are composed from."""
    return (ravel_pytree(receiver(learning=learning).P.init)[1], ravel_pytree(source(0.).P.init)[1])


@functools.lru_cache(maxsize=None)
def _unravel_coupling(count):
    """How ``count`` strokes then ``count`` latches nest into the spatial coupling's state:
    read off the composite's own starting state."""
    return ravel_pytree(spatial_coupling(count).P.init)[1]


def pack_state(flat, count):
    """The browser state in the library tensor's actual nesting.

    The browser's order: the receiver's numbers (battery, grip, tip, cart, its momentum,
    wheel, look, scan, memory), then per casing its six (food, opening, height, momentum,
    air memory, ground stroke), its contact stroke and its latch.
    """
    flat=jnp.asarray(flat, dtype=float)
    nr=len(flat)-8*count
    as_body,as_source=_unravel(nr!=_PATROL_STATE)
    sources=jnp.zeros(0) if not count else as_source(flat[nr:nr+6])
    for k in range(1,count):sources=(sources,as_source(flat[nr+8*k:nr+6+8*k]))
    return (as_body(flat[:nr]),sources),_contacts(flat[nr+6::8],flat[nr+7::8])

def unpack_state(state, count):
    (body,sources),strokes=state
    strokes=ravel_pytree(strokes)[0]
    receiver_state=ravel_pytree(body)[0]
    parts=ravel_pytree(sources)[0].reshape((count,6)) if count else []
    full=[jnp.concatenate([part,strokes[k:k+1],strokes[count+k:count+k+1]]) for k,part in enumerate(parts)]
    return jnp.concatenate([receiver_state]+full)
