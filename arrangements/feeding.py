"""Feeding as a composite in IMC's Cartesian tangent subcategory.

The source and receiver are the factors of feeding-imc.tex, eqn.feed. Objects
are interfaces, morphisms are Arrangement values, composition and tensor are
the library constructors. No new categorical construction is claimed.
Their integrators read configurations, forgetting mass momenta. The existing
Phi / PhiK interpretation computes the inclination; the parameter responds.
The browser's factored field is checked against this composite, including its
six source-output loads. A zero eye capacity is a browser limiting convention,
not a rod-segment integrator, so the library requires positive capacities.
"""
from __future__ import annotations
import jax
import jax.numpy as jnp
from .arrangement import Arrangement, exact
from .coalgebra import CotCoalg, CotCoalgK
from .wiring import compose_seq, tensor_arrangements, trivial_parameter

DEFAULTS = dict(m=1., g=2., beta=.2, kappa=40., x0=0., R=.5, kappa0=100., b0=1., f0=.1,
                lgrip=2., lpry=2., leat=2., a1=.2, a2=.6, alpha=.8, cgrip=.02, ctear=.3,
                xg=-.4, kg=400., eps=.05, Rh=.5, clift=.02, llift=2., M=.25, betac=.4,
                kappac=40., ktau=.16, A=6., omega=.6, gam=.05, lam=3., w=.5, csee=.002, fr=.1,
                kd=4000., bd=40., kr=10., epsd=.01, B=8., nup=.1, gamp=.005, cpat=.002, kexp=.1, memoryDrive=.2, memoryCost=.001, searchSeed=37)

def theta(t):
    # Safe branches prevent inactive 1/0 branches poisoning derivatives.
    def rho(x):
        return jnp.where(x > 0, jnp.exp(-1 / jnp.where(x > 0, x, 1.)), 0.)
    a, b = rho(t), rho(1-t)
    return a/(a+b)

def near(u, radius):
    return theta((radius*radius-u*u)/(.75*radius*radius))

def _parameter(init, indices, response, stages, h, label):
    init = jnp.asarray(init, dtype=float)
    read = lambda s: s[jnp.asarray(indices)]
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
    """Nullary source with six outputs: food, opening, access, height, red, place."""
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


def _food_memory(q, grip, live, seen, eat, p):
    """Learn typical yield at release and local return speed on a later look."""
    y, gaze, meal = q[3], q[3] + q[5], q[7]
    centres = jnp.asarray(CENTRES)
    size = len(CENTRES)
    usual, fresh = q[8:8+size], q[8+size:8+2*size]
    target, angle = q[8+2*size:10+2*size]
    weights, contact, empty, age = q[10+2*size:].reshape(4, size)
    here = jnp.exp(-(y-centres)**2)
    there = jnp.exp(-(gaze-centres)**2)
    release = jax.grad(theta)(q[1]) * (-grip) * theta(-grip/.05)
    meal_rate = eat - 4 * live * theta(-q[1]) * meal
    usual_rate = release * (-jnp.expm1(-meal/4) - jnp.dot(usual, here)) * here
    _, masses = _search_map(usual, fresh)
    speed = _return_rate(weights)
    fresh_rate = live*(speed*(1-fresh) - 4*(here+there)*fresh)
    looking = near(gaze-centres, .55)
    contact_level, last_empty = 10*contact, 10*empty
    dc = 12*live*(looking-contact_level)
    arrival = 2.5*jax.vmap(jax.grad(theta))(2.5*(contact_level-.1))*dc*theta(dc/.1)
    write = 12*live*looking*theta((contact_level-.55)/.25)
    cleared = theta((last_empty-.5)/.4)
    red = theta(seen/.5)
    def revisit_loss(w, elapsed):
        prediction = -jnp.expm1(-_return_rate(w)*elapsed)
        return .5*(prediction-red)**2
    weight_rate = -4*arrival*cleared*jax.vmap(jax.grad(revisit_loss))(weights, 80*age)
    empty_rate = .1*write*(1-red-last_empty)
    age_rate = live*cleared*(1-looking)/80 - write*age
    choosing = theta((jnp.cos(angle)-.4)/.35)
    finish = theta((jnp.cos(angle)-.1)/.25)
    near_goal = theta((2.25-(target-y)**2)/1.25)
    searching = live*theta(1-2*seen)
    clock_rate = searching*(near_goal + (1-near_goal)*finish)
    def select(_):
        n = jnp.floor(angle/(2*jnp.pi)+.5)
        return 4*searching*choosing*(_destination(masses, _variate(p['searchSeed'], n))-target)
    goal_rate = jax.lax.cond(choosing>0, select, lambda _: jnp.array(0.), None)
    rates = jnp.concatenate([jnp.array([meal_rate]), usual_rate, fresh_rate, jnp.array([goal_rate,clock_rate]),
                             weight_rate, .1*dc, empty_rate, age_rate])
    return rates, p['memoryDrive']*jnp.tanh((target-y)/1.5)


def receiver(count, *, parameters=None, stages=1, h=1., learning=False, memory=None, cart=0.):
    """Consumes six outputs of each source; reads out battery and grip."""
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
        return jnp.concatenate([main,xi[7:strokes]/capacity,xi[strokes:]/p['bd']])
    init=[6. if learning else 3.,-1.,p['xg'],cart,0.,cart,0.,p['A']]+extra+[0.]*count
    P=_parameter(init,[0,1,2,3]+list(range(5,len(init))),response,stages,h,'receiver integrator')
    def attachment(q, m):
        b,l,h,y,zw,r,sc=q[:7]
        cost=0.
        for k in range(count):
            f,a,e,x,red,place=m[6*k:6*k+6];u=x-h
            za=q[strokes+k];da=u-za
            cost+=.5*theta(l)*near(u,p['R'])*near(y-place,p['Rh'])*(p['kappa0']*u*u+p['kd']*da*da)+.5*p['kr']*za*za
        return cost
    attachment_grad=jax.grad(attachment,argnums=(0,1))
    def omega(q,m,n):
        b,l,h,y,zw,r,sc=q[:7]; live=theta(b/p['b0']);hold=theta(l)
        dq,dm=attachment_grad(q,m);qq=-dq;mm=-dm
        pl,ph=qq[1],qq[2]
        seen=0.;climb=0.;keep=0.;eaten=0.
        for k in range(count):
            f,a,e,x,red,place=m[6*k:6*k+6]
            delta=(y+r-place)/p['w'];bell=jnp.exp(-delta*delta)
            seen+=red*bell;climb+=red*(-2*delta/p['w'])*bell
            keep+=near(x-h,p['R'])*near(y-place,p['Rh'])*theta((f-p['f0'])/p['f0'])
            mouth=live*near(x-p['x0'],p['R'])*near(y-place,p['Rh'])*hold
            pry=p['lpry']*mouth*(1-a);eat=p['leat']*mouth*e*f
            eaten+=eat
            mm=mm.at[6*k].add(-eat).at[6*k+1].add(pry)
            qq=qq.at[0].add(p['alpha']*eat-p['ctear']*pry)
        grip=p['lgrip']*live*(keep*(2-l)-(1-keep)*(l+1))
        lift=p['llift']*live*(hold*(p['x0']-h)+(1-hold)*(p['xg']-h))
        grip_torque=p['cgrip']*grip-pl;lift_torque=p['clift']*lift-ph
        qq=qq.at[1].add(grip_torque).at[2].add(lift_torque).at[0].add(-grip_torque*grip-lift_torque*lift)
        free=theta(1-2*seen);ring=p['A']**2-r*r-sc*sc
        look=live*(free*(p['omega']*sc+p['gam']*ring*r)+p['lam']*climb)
        scan=live*free*(-p['omega']*r+p['gam']*ring*sc)
        qq=qq.at[5].add(p['csee']*look).at[6].add(p['csee']*scan).at[0].add(-p['csee']*(look*look+scan*scan))
        if learning:
            rates, search = _food_memory(q[:strokes], grip, live, seen, eaten, p)
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
        return qq,mm,jnp.zeros(0)
    return Arrangement(P,0,6*count,0,2,lambda q,m:q[:2],lambda q,m,n:jnp.zeros(0),omega,label='food-learning receiver' if learning else 'receiver')

def feeding(places=(4.,-8.,8.), foods=(.05,10.,10.), *, parameters=None, stages=1, h=1., learning=False, memory=None, cart=0.):
    """Tensor sources, then fill the receiver's internal interfaces."""
    if len(places)!=len(foods):raise ValueError('One food per casing')
    p={**DEFAULTS,**(parameters or {})}
    for key in ['m','M','beta','betac','cgrip','clift','csee','cpat','bd','b0','f0','fr','R','Rh','w','eps','epsd']:
        if p[key]<=0:raise ValueError(f'{key} must be positive')
    if p['a2']<=p['a1']:raise ValueError('a2 must exceed a1')
    if learning:
        if p['memoryCost'] <= 0: raise ValueError('memoryCost must be positive')
    parts=[source(x,f,parameters=p,stages=stages,h=h) for x,f in zip(places,foods)]
    if parts:
        sources=tensor_arrangements(parts)
    else:
        from .arrangement import zero_form
        sources=Arrangement(trivial_parameter(),0,0,0,0,lambda q,m:m,lambda q,m,n:n,zero_form,label='no casings')
    return compose_seq(sources,receiver(len(places),parameters=p,stages=stages,h=h,learning=learning,memory=memory,cart=cart))

def pack_state(flat, count):
    """The browser state in the library tensor's actual nesting."""
    flat=jnp.asarray(flat)
    nr=len(flat)-7*count
    sources=jnp.zeros(0) if not count else flat[nr:nr+6]
    for k in range(1,count):sources=(sources,flat[nr+7*k:nr+6+7*k])
    receiver_state=jnp.concatenate([flat[:nr],jnp.array([flat[nr+6+7*k] for k in range(count)])])
    return sources,receiver_state

def unpack_state(state, count):
    sources,receiver_state=state
    parts=[]
    for _ in range(count-1):sources,right=sources;parts.insert(0,right)
    if count:parts.insert(0,sources)
    nr=len(receiver_state)-count
    full=[jnp.concatenate([part,receiver_state[nr+k:nr+k+1]]) for k,part in enumerate(parts)]
    return jnp.concatenate([receiver_state[:nr]]+full)
