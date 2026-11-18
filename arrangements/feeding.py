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
                kd=4000., bd=40., kr=10., epsd=.01, B=8., nup=.1, gamp=.005, cpat=.002, kexp=.1)

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

def receiver(count, *, parameters=None, stages=1, h=1.):
    """Consumes six outputs of each source; reads out battery and grip."""
    p={**DEFAULTS, **(parameters or {})}
    def response(s,xi):
        main=jnp.array([xi[0],xi[1]/p['cgrip'],xi[2]/p['clift'],s[4]/p['M'],xi[3],xi[4]/p['betac'],xi[5]/p['csee'],xi[6]/p['csee'],xi[7]/p['cpat'],xi[8]/p['cpat']])
        return jnp.concatenate([main,xi[9:]/p['bd']])
    P=_parameter([3.,-1.,p['xg'],0.,0.,0.,0.,p['A'],0.,p['B']]+[0.]*count,[0,1,2,3,5,6,7,8,9]+list(range(10,10+count)),response,stages,h,'receiver integrator')
    def attachment(q, m):
        b,l,h,y,zw,r,sc,eta,zeta=q[:9]
        cost=0.
        for k in range(count):
            f,a,e,x,red,place=m[6*k:6*k+6];u=x-h
            za=q[9+k];da=u-za
            cost+=.5*theta(l)*near(u,p['R'])*near(y-place,p['Rh'])*(p['kappa0']*u*u+p['kd']*da*da)+.5*p['kr']*za*za
        return cost
    attachment_grad=jax.grad(attachment,argnums=(0,1))
    def omega(q,m,n):
        b,l,h,y,zw,r,sc,eta,zeta=q[:9]; live=theta(b/p['b0']);hold=theta(l)
        dq,dm=attachment_grad(q,m);qq=-dq;mm=-dm
        pl,ph=qq[1],qq[2]
        seen=0.;climb=0.;keep=0.
        for k in range(count):
            f,a,e,x,red,place=m[6*k:6*k+6]
            delta=(y+r-place)/p['w'];bell=jnp.exp(-delta*delta)
            seen+=red*bell;climb+=red*(-2*delta/p['w'])*bell
            keep+=near(x-h,p['R'])*near(y-place,p['Rh'])*theta((f-p['f0'])/p['f0'])
            mouth=live*near(x-p['x0'],p['R'])*near(y-place,p['Rh'])*hold
            pry=p['lpry']*mouth*(1-a);eat=p['leat']*mouth*e*f
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
        ringp=p['B']**2-eta*eta-zeta*zeta
        pat_e=live*free*(p['nup']*zeta+p['gamp']*ringp*eta)
        pat_z=live*free*(-p['nup']*eta+p['gamp']*ringp*zeta)
        qq=qq.at[7].add(p['cpat']*pat_e).at[8].add(p['cpat']*pat_z).at[0].add(-p['cpat']*(pat_e*pat_e+pat_z*pat_z))
        spring=p['kappac']*(y-zw);torque=live*(p['ktau']*seen*r+p['kexp']*free*(eta-y))
        qq=qq.at[3].add(-spring).at[4].add(spring+torque).at[0].add(-torque*(spring+torque)/p['betac'])
        return qq,mm,jnp.zeros(0)
    return Arrangement(P,0,6*count,0,2,lambda q,m:q[:2],lambda q,m,n:jnp.zeros(0),omega,label='receiver')

def feeding(places=(4.,-8.,8.), foods=(.05,10.,10.), *, parameters=None, stages=1, h=1.):
    """Tensor sources, then fill the receiver's internal interfaces."""
    if len(places)!=len(foods):raise ValueError('One food per casing')
    p={**DEFAULTS,**(parameters or {})}
    for key in ['m','M','beta','betac','cgrip','clift','csee','cpat','bd','b0','f0','fr','R','Rh','w','eps','epsd']:
        if p[key]<=0:raise ValueError(f'{key} must be positive')
    if p['a2']<=p['a1']:raise ValueError('a2 must exceed a1')
    parts=[source(x,f,parameters=p,stages=stages,h=h) for x,f in zip(places,foods)]
    if parts:
        sources=tensor_arrangements(parts)
    else:
        from .arrangement import zero_form
        sources=Arrangement(trivial_parameter(),0,0,0,0,lambda q,m:m,lambda q,m,n:n,zero_form,label='no casings')
    return compose_seq(sources,receiver(len(places),parameters=p,stages=stages,h=h))

def pack_state(flat, count):
    """The browser state in the library tensor's actual nesting."""
    flat=jnp.asarray(flat)
    sources=jnp.zeros(0) if not count else flat[10:16]
    for k in range(1,count):sources=(sources,flat[10+7*k:16+7*k])
    receiver_state=jnp.concatenate([flat[:10],jnp.array([flat[16+7*k] for k in range(count)])])
    return sources,receiver_state

def unpack_state(state, count):
    sources,receiver_state=state
    parts=[]
    for _ in range(count-1):sources,right=sources;parts.insert(0,right)
    if count:parts.insert(0,sources)
    full=[jnp.concatenate([part,receiver_state[10+k:11+k]]) for k,part in enumerate(parts)]
    return jnp.concatenate([receiver_state[:10]]+full)
