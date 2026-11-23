"""Feeding as a composite in IMC's Cartesian tangent subcategory.

The spatial coupling takes a fixed seeker beside its casing sources (feeding-imc.tex, eqn.feed). Objects
are interfaces, morphisms are Arrangement values, composition and tensor are
the library constructors. No new categorical construction is claimed.
Their integrators read configurations, forgetting mass momenta. The existing
Phi / PhiK interpretation computes the inclination; the parameter responds.
The library requires positive capacities.

``feeding`` is ``feeding_composed.feeding_with_slot`` (book P109 (i), act (5), 2026-11-23):
the seeker with a separator mouth (its chamber), a husk collector and a slot filled by the
gut wall, casings without a hinge, and the pile on the ground, each composed by
``compose_seq`` and ``tensor_arrangements`` from the note's general parts, in
``feeding_composed.py``, ``feeding_recursive.py`` and ``feeding_parts.py``. The memory's
rates are nine circles, one per store, each in its learner's box with the motor it tells
(``_memory_rules``, ``feeding_composed.learner``; P109 (f), act (3), 2026-11-22).
``source_by_hand`` and ``_body_by_hand`` are single boxes with their fields typed out, kept
as checks on ``source`` and ``feeding_composed.body``. The hinged model (a casing with an
opening, the opener and digestor, their spatial coupling, and the page's flat state layout)
left the library on 2026-11-23; it is in the history before that date.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from .arrangement import Arrangement, exact
from .coalgebra import CotCoalg, CotCoalgK

DEFAULTS = dict(m=1., g=2., beta=.2, kappa=40., x0=0., R=.5, kappa0=100., b0=1., f0=.1,
                # (The hinged model's lpry, a1, a2, alpha and ctear left with it on 2026-11-23.)
                lgrip=2., leat=2., cgrip=.02,
                xg=-.4, kg=400., eps=.05, Rh=.5, clift=.02, llift=2., M=.25, betac=.4,
                # ktau: the most torque the drive applies toward what the eye holds, reached
                # beyond rDrive of it; bDrive: the battery level at which the drive is fully awake
                # (book P109 (c), David, 2026-11-18: "it should keep approaching the place").
                # bDrive is b0, as for every other motor: with bDrive = 0.1 a cart on battery 0.3
                # drives at full torque, empties the battery to 0.03 before it arrives, and the
                # gripper, waking at b0, can no longer close (.scratch/p109-c5f9ecc9/drive_check.py).
                # ktau = 0.3: a speed near ktau / betac = 0.75 and 0.3 of battery per unit of
                # distance; at 0.6 the page's second trip of seed 127 arrives on an empty battery.
                kappac=40., ktau=.3, rDrive=.25, bDrive=1., A=6., omega=.6, gam=.05, lam=3., w=.5, csee=.002, fr=.1,
                kd=4000., bd=40., kr=10., epsd=.01, B=8., nup=.1, gamp=.005, cpat=.002, kexp=.1,
                latchRate=40., latchCost=.0001, memoryDrive=.2, memoryCost=.001, searchSeed=37,
                # The separator mouth and its slot (book P109 (i)): pass fraction, battery cost per
                # gram, chamber and collector capacities, the collector's clearance, and the
                # absorber's energy per gram and uptake rate.
                mouthPass=.8, mouthCost=.03, chamberCap=8., collectorCap=4., clearance=1.,
                # uptake: the gut wall's rate, slow, so digestion goes on while the cart moves (book
                # P109 (k); David, 2026-11-23: "it can take a good amount of time"); 2 before (k).
                absorbYield=1., uptake=.15,
                # A robot-style seeker inside another's slot (book P109 (j)): the rate at which it
                # lays out the mass it is handed among its casings, and its battery's cord to the reserve.
                layRate=1., cord=2.,
                # Book P109 (k), David, 2026-11-23, b679b569: "local batteries per part: i can use it
                # however i want and it gets recharged at a rate set by the larger arrangement". Each
                # organ has a battery of capacity cLocal, corded to the robot's by a spring of
                # conductance cordOrgan; its motors are awake from wLocal of their own battery up.
                # The robot's battery stops charging at battCap (the gut wall slows over the last
                # capWidth); hunger is theta((battCap - b - absorbYield z) / hungerWidth), z the
                # chamber. Each casing has a lid of capacity cLid, driven by the hands' lid motor at
                # lidRate, open for eating from lidOpen up; an open lid spills at spillRate times
                # the casing's speed. The numbers are b679b569's, proposed: cLocal small, since an
                # organ's battery sits at the robot's level and so holds cLocal times it; battCap 12.5,
                # since from battery 1 two casings bring the battery plus the chamber to about 11.8
                # (c5f9ecc9's 16 left out what the motors spend while the gut works,
                # .scratch/p109k-b679b569/small.py).
                # Satiety (b679b569, proposed, from the small example: without a memory, hunger
                # chatters, letting go and taking back the one casing): a store in the mouth, driven
                # at satRate toward 1 once the room cap - b - absorbYield z is below sateRoom and
                # toward 0 once it is above hungryRoom, each over hungerWidth; hunger is 1 - satiety.
                cLocal=.02, wLocal=.1, cordOrgan=1., battCap=12.5, capWidth=1., hungerWidth=.5,
                sateRoom=1., hungryRoom=3.5, satRate=4.,
                cLid=.02, lidRate=4., lidOpen=.5, spillRate=1., vSpill=.5,
                # The basal cost, a constant drain on the robot's battery while it holds some (David,
                # 2026-11-23, 222a18e7: "yes to constant drain"), so a sated robot grows hungry at rest.
                # 0.02 is 222a18e7's choice: at 0.05 the patrol cart of the note's episode and one seed of the
                # page's two-site trip ran dry between meals; 0.02 is the largest of the two tried that did not.
                basal=.02)

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
    """Nullary source with four outputs: food, height, red, place. No hinge: the mouth tears
    the food out (the hinged casing, with an opening and an access readout, left 2026-11-23).

    Composed by the library from the note's parts (``feeding_composed.source``): two
    integrands, gravity, the air's dragger and the ground's spring and damper.
    """
    from .feeding_composed import source as composed
    return composed(place, food, parameters=parameters, stages=stages, h=h)


def source_by_hand(place, food=10., *, parameters=None, stages=1, h=1.):
    """The same source as one box, its summed field typed out. Kept only as a check on
    ``source`` (``tests/test_feeding_composed.py``); the model does not use it."""
    p={**DEFAULTS, **(parameters or {})}
    def response(s, xi):
        return jnp.array([xi[0],s[2]/p['m'],xi[1],xi[2]/p['beta'],xi[3]/p['bd']])
    P=_parameter([food,p['xg'],0.,p['xg'],0.],[0,1,3,4],response,stages,h,'casing integrator')
    def cost(q,m,n):
        f,x,z,zg=q;v=x-p['xg'];dg=v-zg
        return (p['m']*p['g']*x+.5*p['kg']*theta(-v/p['eps'])*v*v+.5*p['kappa']*(x-z)**2
                +.5*p['kd']*theta(-v/p['epsd'])*dg*dg+.5*p['kr']*zg*zg)
    def out(q,m):
        f,x,z,zg=q
        return jnp.array([f,x,theta((f-p['fr'])/p['fr']),place])
    return Arrangement(P,0,0,0,4,out,lambda q,m,n:jnp.zeros(0),exact(lambda q,m,n:-cost(q,m,n)),label=f'casing({place:g})')

CENTRES = tuple(range(-9, 10))


def usual_at(weights, x):
    """The usual meal yield at the places ``x``: one bump per centre, weighted by its store."""
    return jnp.exp(-(jnp.asarray(x)[..., None] - jnp.asarray(CENTRES))**2) @ weights


def _search_map(weights, freshness):
    usual = usual_at(weights, jnp.asarray(CENTRES, dtype=float))
    richness = usual * theta(usual / .05)
    return richness, .00001 + freshness**2 * (.012 + richness**2)


def density_at(masses, x):
    """The search density at the places ``x``: the slope of the cumulative curve over its
    total, so that its area over the line is one."""
    slope = jax.vmap(jax.grad(_cumulative, argnums=1), in_axes=(None, 0))(masses, jnp.atleast_1d(x))
    return slope / masses.sum()


def _variate(seed, index):
    # Counter-based pseudorandom tape; all operations before conversion are uint32.
    x = jnp.asarray(seed, jnp.uint32) + (index.astype(jnp.uint32)+1)*jnp.uint32(0x9e3779b9)
    x = (x ^ (x >> 16))*jnp.uint32(0x21f0aaad)
    x = (x ^ (x >> 15))*jnp.uint32(0x735a2d97)
    return ((x ^ (x >> 15)).astype(float) + .5)/4294967296.


def _cumulative(masses, x):
    """The search map's cumulative curve at ``x``: the masses' smoothed steps summed."""
    centres = jnp.asarray(CENTRES)
    left, right = CENTRES[0], CENTRES[-1]
    scale = jnp.where((centres == left) | (centres == right), 2., 1.)
    return jnp.dot(masses, scale*(theta((x-centres+1)/2) - .5*(centres == left)))


@jax.custom_jvp
def _destination(masses, u):
    """The inverse of the cumulative curve at the share ``u`` of its total, computed to
    rounding: 52 halvings of ``[-9, 9]`` leave an interval of ``18/2^52``, about ``4e-15``.
    Its derivative is the inverse's own (below), not the halving's, which is zero."""
    threshold = masses.sum()*u
    def bisect(_, bounds):
        lo, hi = bounds
        x = (lo+hi)/2
        below = _cumulative(masses, x) < threshold
        return jnp.where(below,x,lo), jnp.where(below,hi,x)
    lo, hi = jax.lax.fori_loop(0, 52, bisect, (float(CENTRES[0]),float(CENTRES[-1])))
    return (lo+hi)/2


@_destination.defjvp
def _destination_jvp(primals, tangents):
    # The implicit function theorem on C(m, x) = u sum(m): dx = (du sum(m) + u sum(dm) - dC/dm . dm) / (dC/dx).
    masses, u = primals
    dm, du = tangents
    x = _destination(masses, u)
    slope = jax.grad(_cumulative, argnums=1)(masses, x)
    moved = jax.jvp(lambda m: _cumulative(m, x), (masses,), (dm,))[1]
    return x, (du*masses.sum() + u*dm.sum() - moved)/slope


def _return_rate(weight):
    return .1 * jax.nn.sigmoid(weight-jnp.log(7.))


def _looking(gaze):
    """How squarely the eye rests on each centre."""
    return near(gaze-jnp.asarray(CENTRES), .55)


def _write(contact, gaze):
    """Sustained sight of a centre: the eye has rested on it long enough to write."""
    return 12*_looking(gaze)*theta((10*contact-.55)/.25)


def _cleared(empty):
    return theta((10*empty-.5)/.4)


def _memory_rules(p):
    """The memory's learners, one rule each: ``(store, what the rule reads, rate)``.

    Each rule is the circle that tells its store's motor a rate; the motor reads the battery
    and drives its store at ``live(b)`` times that rate, so no rate here carries ``live``.
    What a rule may read: the stores by name (``meal``, ``usual``, ``fresh``, ``target``,
    ``angle``, ``weights``, ``contact``, ``empty``, ``age``), and ``grip``, ``grip_rate``
    (the rate the gripper tells its motor), ``y`` (the cart), ``gaze`` (cart plus look),
    ``seen`` and ``eaten``. Book P109 (f), act (3): one circle per learner.
    """
    centres = jnp.asarray(CENTRES)
    here = lambda y: jnp.exp(-(y-centres)**2)

    def meal(v):
        return jnp.atleast_1d(v['eaten'] - 4*theta(-v['grip'])*v['meal'])

    def usual(v):
        # The squared-error learner of typical yield: at release, the usual yield near the
        # cart moves toward what the meal gave, -expm1(-meal/4).
        g = v['grip_rate']
        release = jax.grad(theta)(v['grip']) * (-g) * theta(-g/.05)
        return release * (-jnp.expm1(-v['meal']/4) - jnp.dot(v['usual'], here(v['y']))) * here(v['y'])

    def fresh(v):
        speed = _return_rate(v['weights'])
        return speed*(1-v['fresh']) - 4*(here(v['y'])+here(v['gaze']))*v['fresh']

    def target(v):
        _, masses = _search_map(v['usual'], v['fresh'])
        angle, searching = v['angle'], theta(1-2*v['seen'])
        choosing = theta((jnp.cos(angle)-.4)/.35)
        def select(_):
            n = jnp.floor(angle/(2*jnp.pi)+.5)
            return 4*searching*choosing*(_destination(masses, _variate(p['searchSeed'], n))-v['target'])
        return jnp.atleast_1d(jax.lax.cond(choosing>0, select, lambda _: jnp.array(0.), None))

    def angle(v):
        finish = theta((jnp.cos(v['angle'])-.1)/.25)
        near_goal = theta((2.25-(v['target']-v['y'])**2)/1.25)
        return jnp.atleast_1d(theta(1-2*v['seen'])*(near_goal + (1-near_goal)*finish))

    def weights(v):
        # The squared-error learner of the return rate: on arrival at a cleared centre, each
        # weight descends the gradient of the revisit loss, half the squared gap between the
        # predicted chance of finding food again and how red the place now looks.
        contact_level = 10*v['contact']
        dc = 12*(_looking(v['gaze'])-contact_level)
        arrival = 2.5*jax.vmap(jax.grad(theta))(2.5*(contact_level-.1))*dc*theta(dc/.1)
        red = theta(v['seen']/.5)
        def revisit_loss(w, elapsed):
            prediction = -jnp.expm1(-_return_rate(w)*elapsed)
            return .5*(prediction-red)**2
        return -4*arrival*_cleared(v['empty'])*jax.vmap(jax.grad(revisit_loss))(v['weights'], 80*v['age'])

    def contact(v):
        return .1*12*(_looking(v['gaze'])-10*v['contact'])

    def empty(v):
        return .1*_write(v['contact'], v['gaze'])*(1-theta(v['seen']/.5)-10*v['empty'])

    def age(v):
        return (_cleared(v['empty'])*(1-_looking(v['gaze']))/80
                - _write(v['contact'], v['gaze'])*v['age'])

    return [('meal', ('meal', 'grip', 'eaten'), meal),
            ('usual', ('usual', 'meal', 'y', 'grip', 'grip_rate'), usual),
            ('fresh', ('fresh', 'weights', 'y', 'gaze'), fresh),
            ('target', ('target', 'angle', 'usual', 'fresh', 'seen'), target),
            ('angle', ('angle', 'target', 'y', 'seen'), angle),
            ('weights', ('weights', 'contact', 'empty', 'age', 'gaze', 'seen'), weights),
            ('contact', ('contact', 'gaze'), contact),
            ('empty', ('empty', 'contact', 'gaze', 'seen'), empty),
            ('age', ('age', 'empty', 'contact', 'gaze'), age)]


def _memory_values(q, grip_rate, seen, eat):
    """The values the memory's rules read, from the receiver's configuration ``q``."""
    size = len(CENTRES)
    usual, fresh = q[8:8+size], q[8+size:8+2*size]
    weights, contact, empty, age = q[10+2*size:].reshape(4, size)
    return dict(y=q[3], gaze=q[3]+q[5], grip=q[1], grip_rate=grip_rate, seen=seen, eaten=eat,
                meal=q[7], usual=usual, fresh=fresh, target=q[8+2*size], angle=q[9+2*size],
                weights=weights, contact=contact, empty=empty, age=age)


def _food_memory(q, grip, seen, eat, p):
    """Learn typical yield at release and local return speed on a later look.

    The rates the memory's motors are told, each from its own rule (``_memory_rules``),
    concatenated in the stores' order; and the pull the search sends the wheels. ``grip`` is
    the rate the gripper tells its motor."""
    v = _memory_values(q, grip, seen, eat)
    rates = jnp.concatenate([rule(v) for _, _, rule in _memory_rules(p)])
    return rates, p['memoryDrive']*jnp.tanh((v['target']-v['y'])/1.5)


def _body_by_hand(*, parameters, stages, h, learning, memory, cart):
    """``feeding_composed.body`` as one box, its summed field typed out. Kept only as a check
    on it (``tests/test_feeding_composed.py``); the model does not use it."""
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
        spring=p['kappac']*(y-zw)
        torque=theta(b/p['bDrive'])*(p['ktau']*seen*jnp.tanh(r/p['rDrive'])+free*search)
        qq=qq.at[3].add(-spring).at[4].add(spring+torque).at[0].add(-torque*(spring+torque)/p['betac'])
        return qq,jnp.zeros_like(m),jnp.zeros_like(n)
    return Arrangement(P,0,0,7,5,lambda q,m:q[jnp.array([0,1,2,3,5])],
                       lambda q,m,n:jnp.zeros(0),omega,label='body and motors')


def feeding(places=(4.,-8.,8.), foods=(.05,10.,10.), *, parameters=None, stages=1, h=1., learning=False,
            memory=None, cart=0., filler=None, pile=0., battery=None):
    """Feeding with a separator mouth: ``feeding_composed.feeding_with_slot``, the seeker with
    its chamber, husk collector and slot (filled by ``filler``, the gut wall by default), the
    casings without a hinge, and the pile on the ground, under the spatial coupling."""
    from .feeding_composed import feeding_with_slot
    if len(places)!=len(foods):raise ValueError('One food per casing')
    p={**DEFAULTS,**(parameters or {})}
    for key in ['m','M','beta','betac','cgrip','clift','csee','cpat','bd','b0','f0','fr','R','Rh','w','eps','epsd',
                'latchCost','latchRate','chamberCap','collectorCap']:
        if p[key]<=0:raise ValueError(f'{key} must be positive')
    if learning and p['memoryCost'] <= 0: raise ValueError('memoryCost must be positive')
    return feeding_with_slot(places, foods, filler=filler, parameters=p, stages=stages, h=h, learning=learning,
                             memory=memory, cart=cart, pile=pile, battery=battery)
