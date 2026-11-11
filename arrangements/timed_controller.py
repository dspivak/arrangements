"""Finite imagined control and observed-model learning in IMC on Euclidean spaces.

Research example, not a general free/cofree synthesis or deadline guarantee.
See examples/timed_controller.md for factorization, interfaces and limitations.
Extracted from the book's a330db6f and 0676663f experiments (2026-11-11).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from .arrangement import Arrangement, exact, zero_form
from .coalgebra import CotCoalg, configuration
from .functors import Phi
from .interpretation import trivial_omega
from .rvect import euclidean
from .system_id import one_way_squared_error
from .wiring import compose_seq, parallel_arrangements, tensor_arrangements, trivial_parameter

Z = jnp.zeros(0)
IN = (Z, trivial_omega(0))
DT = 0.1
# Learned by the earlier mass-one, 3000-update experiment; a fixture, not synthesis here.
PRETRAINED = jnp.array([1.3555143971688006, -0.5768814956932359,
                       0.039326516587624566, -1.160580009612816])


def holder(parameter, label):
    return Arrangement(parameter, 0, 0, 0, parameter.Q_dim,
                       lambda q, m: q, lambda q, m, n: Z, zero_form, label)

def participant(mass=1.0):
    def update(s, covector, dt):
        x, v = s
        acceleration = covector[0] / mass
        return jnp.array([x + dt*v + 0.5*dt*dt*acceleration, v + dt*acceleration])
    parameter = CotCoalg(2, jnp.zeros(2), lambda s: s,
                         lambda s, xi: update(s, xi, 1.0),
                         f"mass={mass:g}/held-force", timed_step=update)
    return holder(parameter, "particle(x,v)" if mass == 1.0 else f"particle(x,v;mass={mass:g})")

def clock_part():
    parameter = CotCoalg(1, jnp.zeros(1), lambda s: s,
                         lambda s, xi: s+1, "clock",
                         timed_step=lambda s, xi, dt: s+dt)
    return holder(parameter, "clock(t)")

def command(theta, x, v, t):
    return jnp.dot(theta, jnp.array([1.0, x, v, t]))

def build_execution(mass=1.0):
    fixed = holder(configuration(euclidean(4, 0.0)), "frozen-controller-coefficients")
    parts = parallel_arrangements(parallel_arrangements(participant(mass), clock_part()), fixed)

    def force(q, observed, n):
        x, v, t = observed[:3]
        u = command(observed[3:], x, v, t)
        # The command source supplies a load to the position coordinate.
        # No terminal-score force acts on the physical participant here.
        return Z, jnp.concatenate([jnp.array([u]), jnp.zeros(6)]), Z

    controller = Arrangement(trivial_parameter(), 0, 7, 0, 3,
                             lambda q, m: m[:3], lambda q, m, n: Z,
                             force, "affine-command(x,v,t;theta)")
    return compose_seq(parts, controller)

def execution_state(theta, initial=jnp.zeros(2)):
    return (((initial, jnp.zeros(1)), theta), Z)

def unpack(state):
    (((xv, clock), theta), _) = state
    return xv, clock[0], theta

def scorer(velocity_weight=1.0):
    def potential(q, final, n):
        return -0.5*((final[0]-1.0)**2 + velocity_weight*final[1]**2)
    return Arrangement(trivial_parameter(), 0, 2, 0, 0,
                       lambda q, m: Z, lambda q, m, n: Z,
                       exact(potential), f"terminal-score(position=1,velocity-weight={velocity_weight})")

def static(fn, din, dout, label):
    return Arrangement(trivial_parameter(), 0, din, 0, dout,
                       lambda q,m: fn(m), lambda q,m,n: Z, zero_form, label)


def run_step(arr):
    machine = Phi(arr)
    def step(state, context=Z, duration=1.0):
        return machine.with_state(state).run_one(
            IN, lambda o: (jnp.zeros(arr.out_dim_N), context), duration=duration)[2]
    return step


def imaginary_execution():
    # Inverse mass is an independent frozen coordinate, not a trainable force source.
    def move(s, xi, dt):
        x,v,a = s
        return jnp.array([x+dt*v+0.5*dt*dt*a*xi[0], v+dt*a*xi[0], a])
    particle = holder(CotCoalg(3,jnp.array([0.,0.,1.]),lambda s:s,
                       lambda s,xi:move(s,xi,1.),'modeled-particle',timed_step=move),
                       'fiction-particle(x,v,inverse-mass)')
    fixed = holder(configuration(euclidean(4,0.)), 'frozen-controller')
    parts = tensor_arrangements([particle,clock_part(),fixed])
    def force(q,m,n):
        u = command(m[4:],m[0],m[1],m[3])
        return Z,jnp.concatenate([jnp.array([u]),jnp.zeros(7)]),Z
    router = Arrangement(trivial_parameter(),0,8,0,2,lambda q,m:m[:2],
                         lambda q,m,n:Z,force,'affine-command-in-fiction')
    return compose_seq(parts,router)


IMAGINED = imaginary_execution()
IMAGINE_STEP = run_step(IMAGINED)


def imagine(theta, context):
    a,x,v,t = context[:4]
    state = (((jnp.array([x,v,a]),jnp.array([t])),theta),Z)
    final = jax.lax.fori_loop(0,20,lambda i,s:IMAGINE_STEP(s,Z,context[4+i]),state)
    return final[0][0][0][:2]


def model_learner():
    # Context: previous x,v, command, duration, observed next x,v.
    weights = holder(configuration(euclidean(1,0.5)), 'inverse-mass-holder')
    def prediction_and_target(m):
        a,x,v,u,dt = m[:5]
        b = jnp.array([0.5*dt*dt*u,dt*u])
        scale = jnp.sqrt(1e-6+jnp.dot(b,b))
        pred = (jnp.array([x+dt*v,v])+a*b)/scale
        return jnp.concatenate([pred,m[5:]/scale])
    predictor = static(prediction_and_target,7,4,'normalized-transition-and-observed-target')
    error = compose_seq(predictor,one_way_squared_error(2))
    def omega(q,m,n):
        load = error.omega(error.P.position(error.P.init),jnp.concatenate([m,n]),Z)[1]
        # A supplied snapshot is heard without pushing on its input ports.
        return Z,load[:1],jnp.zeros(6)
    route = Arrangement(trivial_parameter(),0,1,6,1,lambda q,m:m,
                        lambda q,m,n:Z,omega,'snapshot-input['+error.label+']')
    return compose_seq(weights,route)


MODEL = model_learner()
_MODEL_STEP = run_step(MODEL)
def MODEL_STEP(a, context, duration=1.0):
    return _MODEL_STEP((a,Z),context,duration)[0]


def controller_learner(cut=False):
    # A context snapshot is carried by frozen parameter coordinates, so composing
    # holder, rollout and scorer differentiates only the four controller coefficients.
    def update(s,xi,dt):
        theta,context = s[:4],s[4:]
        rate = 0.03/(1.+context[0]**2)
        return jnp.concatenate([theta+dt*rate*xi[:4],context])
    parameter = CotCoalg(28,jnp.zeros(28),lambda s:s,lambda s,xi:update(s,xi,1.),
                         'theta-learns-context-frozen',timed_step=update)
    weights = holder(parameter,'controller-weights-and-frozen-context')
    rollout = static(lambda m:imagine(m[:4],m[4:]),28,2,
                     'remaining-time-depth-20['+IMAGINED.label+']')
    score = scorer()
    if cut:
        score = Arrangement(score.P,0,2,0,0,score.out_f,score.in_f,zero_form,'cut-score')
    return compose_seq(compose_seq(weights,rollout),score)


CONTROL = controller_learner()
CONTROL_STEP = run_step(CONTROL)


def improve(theta,context,count,duration=1.0):
    state = ((jnp.concatenate([theta,context]),Z),Z)
    state = jax.lax.fori_loop(0,count,lambda i,s:CONTROL_STEP(s,Z,duration),state)
    return state[0][0][:4]


def computer(kind,count,initial):
    n = len(initial)
    c = 6 if kind == 'model' else 24
    def update(s,xi,dt):
        context = xi[n:]
        if kind == 'model':
            return jax.lax.fori_loop(0,count,lambda i,a:MODEL_STEP(a,context,dt/DT),s)
        return improve(s,context,count,dt/DT)
    P = CotCoalg(n+c,initial,lambda s:jnp.concatenate([s,jnp.zeros(c)]),
                 lambda s,xi:update(s,xi,1.),kind+'-computer',timed_step=update)
    def omega(q,m,n_in):
        return jnp.concatenate([jnp.zeros(n),n_in]),Z,jnp.zeros(c)
    inner = MODEL if kind == 'model' else CONTROL
    return Arrangement(P,0,0,c,n,lambda q,m:q[:n],lambda q,m,n:Z,omega,
                       f'{kind}-computer[{count} updates; {inner.label}]')


def physical(mass):
    def move(s,xi,dt):
        x,v = s[:2]
        u = xi[0]
        return jnp.array([x+dt*v+0.5*dt*dt*u/mass,v+dt*u/mass,x,v,u,dt])
    P = CotCoalg(6,jnp.zeros(6),lambda s:s,lambda s,xi:move(s,xi,1.),
                 f'actual-mass={mass}',timed_step=move)
    return holder(P,f'actual-particle-and-transition(mass={mass})')


def online(mass,budget,correct=True,learn=True,cold=False,split_command=False):
    k = budget if learn else 0
    def advance(s,xi,dt):
        shifted = jnp.concatenate([s[3:],jnp.zeros(1)])
        return jnp.concatenate([s[:2]+jnp.array([dt,k*dt/DT]),s[2:]+(dt/DT)*(shifted-s[2:])])
    clock = holder(CotCoalg(22,jnp.concatenate([jnp.zeros(2),jnp.full(20,DT)]),lambda s:s,lambda s,xi:advance(s,xi,1.),
                       'physical-and-computation-clock',timed_step=advance),'clock(time,completed-updates,remaining-duration-stream)')
    model = computer('model',int(correct),jnp.ones(1))
    control = computer('controller',k,jnp.zeros(4) if cold else PRETRAINED)
    parts = tensor_arrangements([physical(mass),clock,model,control])
    def inputs(m):
        return jnp.concatenate([m[2:6],m[:2],m[28:29],m[:2],m[6:7],m[8:28]])
    def force(q,m,n):
        u = command(m[29:],m[0],m[1],m[6])
        return Z,jnp.concatenate([jnp.array([u]),jnp.zeros(32)]),Z
    router = Arrangement(trivial_parameter(),30,33,0,13,lambda q,m:jnp.concatenate([m[:8],m[28:]]),
                         lambda q,m,n:inputs(m),force,'route-observed-transition-and-model-to-computers')
    if split_command:
        # Exact further factorization of the original stateless inclined router:
        # identity transfers + command field, then the original transfers + zero field.
        command_arr = Arrangement(trivial_parameter(),30,33,30,33,
                              lambda q,m:m,lambda q,m,n:n,
                              lambda q,m,n:(Z,force(Z,m,Z)[1],jnp.zeros(30)),
                              'motor-command-with-identity-transfers')
        routing = Arrangement(trivial_parameter(),30,33,0,13,
                              router.out_f,router.in_f,zero_form,
                              'observation-and-snapshot-wiring')
        return compose_seq(compose_seq(parts,command_arr),routing)
    return compose_seq(parts,router)


def observations(arr,state):
    return arr.out_f(arr.P.position(state),Z)


def evaluate(mass,theta):
    arr = build_execution(mass)
    step = run_step(arr)
    state = execution_state(theta)
    def body(s,i):
        xv,t,th = unpack(s)
        u = command(th,xv[0],xv[1],t)
        nxt = step(s,Z,DT)
        nv,_,_ = unpack(nxt)
        return nxt,jnp.concatenate([xv,jnp.array([u,DT]),nv])
    state,transitions = jax.lax.scan(body,state,jnp.arange(20))
    xv,_,_ = unpack(state)
    return xv,transitions



def timing_learner(cut=False):
    weights=holder(configuration(euclidean(1,.2)),'timing-parameter')
    def choice(m):
        w=jax.nn.sigmoid(m[0])
        return jnp.array([1.-w,w])
    policy=static(choice,1,2,'probabilities(now,next-opportunity)')
    forecast=static(lambda m:jnp.array([[.75,.25],[.25,.75]])@m,2,2,'fixed-outcome-forecast')
    value=static(lambda m:jnp.array([4.*m[1]]),2,1,'expected-reward')
    score=Arrangement(trivial_parameter(),0,1,0,0,lambda q,m:Z,lambda q,m,n:Z,
                      zero_form if cut else exact(lambda q,m,n:m[0]),'score-load' if not cut else 'cut-score-load')
    prediction=compose_seq(compose_seq(policy,forecast),value)
    return compose_seq(weights,compose_seq(prediction,score))
