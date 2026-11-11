"""Regression checks for the finite IMC research example.

Predictions: preserve analytic model updates, gradient transport, one-way observations,
all 21 duration suffixes, causal latency, unit/tensor checks and exact further factorization.
Rehearsal should help under the matching timing model and hurt under its opposite.
These are Euclidean finite tests, not guarantees for arbitrary smooth spaces or biology.
"""
import time
import jax
import jax.numpy as jnp
import numpy as np

from arrangements import timed_controller as c
from arrangements.timed_controller import (
    Z, DT, PRETRAINED, MODEL, MODEL_STEP, CONTROL_STEP, imagine, improve,
    controller_learner, online, observations, run_step,
    build_execution, execution_state, unpack, command,
)
from arrangements.arrangement import Arrangement, zero_form
from arrangements.system_id import one_way_squared_error
from arrangements.wiring import compose_seq, parallel_arrangements, trivial_parameter

def save(name, data):
    # The research runner writes JSON; tests only check its numerical certificates.
    assert name == 'checks'
    assert data['gradient_error'] < 2e-8
    assert data['schedule_replay_error'] < 1e-10
    assert data['all_21_suffixes_error'] < 1e-12

def test_model_controller_and_causal_schedule():
    started = time.perf_counter()
    mstep = jax.jit(MODEL_STEP)
    cases = []
    for u in [-2.,0.,1.]:
        for dt in [0.,0.1,0.25]:
            c = jnp.array([.2,-.3,u,dt,.2-.3*dt+.5*dt*dt*u*4.,-.3+dt*u*4.])
            actual = mstep(jnp.ones(1),c,1.)[0]
            b2 = (.5*dt*dt*u)**2+(dt*u)**2
            expected = 1.+.5*b2/(1e-6+b2)*3.
            np.testing.assert_allclose(actual,expected,atol=1e-12)
            np.testing.assert_array_equal(MODEL.omega(jnp.ones(1),Z,c)[2],jnp.zeros(6))
            cases.append([u,dt,float(actual)])
    loss = one_way_squared_error(2)
    np.testing.assert_array_equal(loss.omega(Z,jnp.array([1.,2.,3.,4.]),Z)[1][2:],jnp.zeros(2))
    theta=jnp.array([.2,-.1,.3,-.2]); context=jnp.concatenate([jnp.array([1.7,.4,-.3,.5]),jnp.full(15,DT),jnp.zeros(5)])
    cost = jax.jit(lambda th: .5*jnp.sum((imagine(th,context)-jnp.array([1.,0.]))**2))
    grad = jax.jit(jax.grad(cost))(theta)
    eps=1e-5
    finite=jnp.array([(cost(theta+eps*jnp.eye(4)[i])-cost(theta-eps*jnp.eye(4)[i]))/(2*eps) for i in range(4)])
    np.testing.assert_allclose(grad,finite,atol=2e-8)
    updated=jax.jit(improve,static_argnums=2)(theta,context,1)
    np.testing.assert_allclose(updated,theta-.03/(1+context[0]**2)*grad,atol=1e-12)
    state=((jnp.concatenate([theta,context]),Z),Z)
    cut=jax.jit(run_step(controller_learner(True)))(state)
    np.testing.assert_array_equal(cut[0][0],state[0][0])
    frozen=jax.jit(CONTROL_STEP)(state)[0][0][4:]
    np.testing.assert_array_equal(frozen,context)
    # Parallel composite: two models, different contexts, shared duration.
    pair=parallel_arrangements(MODEL,MODEL)
    c1=jnp.array([0.,0.,1.,.1,.02,.4]); c2=c1.at[2].set(-1.).at[4:].set(jnp.array([-.005,-.1]))
    together=jax.jit(run_step(pair))(((jnp.ones(1),Z),(jnp.ones(1),Z)),jnp.concatenate([c1,c2]),.7)
    for got,want in zip(together,(mstep(jnp.ones(1),c1,.7),mstep(jnp.ones(1),c2,.7))):
        np.testing.assert_allclose(got[0],want,atol=1e-12)
    # Static identity around the observable model preserves both state and loads.
    identity=Arrangement(trivial_parameter(),6,1,6,1,lambda q,m:m,lambda q,m,n:n,zero_form,'identity')
    ident=jax.jit(run_step(compose_seq(MODEL,identity)))(((jnp.ones(1),Z),Z),c1,.7)
    np.testing.assert_allclose(ident[0][0],mstep(jnp.ones(1),c1,.7),atol=1e-12)
    system=online(.5,10)
    step=jax.jit(run_step(system)); state=system.P.init
    t0=time.perf_counter()
    state=step(state,Z,DT); jax.block_until_ready(state)
    compile_seconds=time.perf_counter()-t0
    t0=time.perf_counter()
    for _ in range(10): state=step(state,Z,DT)
    jax.block_until_ready(state)
    warm=time.perf_counter()-t0
    # Replay the same causal schedule through separate Phi machines, not the wiring.
    real=build_execution(.5); realstep=jax.jit(run_step(real))
    actual_state=execution_state(PRETRAINED); a=jnp.ones(1); th=PRETRAINED
    previous=jnp.zeros(6); composed=system.P.init; maxerr=0.
    for tick in range(20):
        xv,t,_=unpack(actual_state)
        nxt=realstep((((xv,jnp.array([t])),th),Z),Z,DT)
        nextxv,nextt,_=unpack(nxt)
        nexta=mstep(a,previous,1.)
        nextth=jax.jit(improve,static_argnums=2)(th,jnp.concatenate([a,xv,jnp.array([t]),jnp.full(20-tick,DT),jnp.zeros(tick)]),10)
        previous=jnp.concatenate([xv,jnp.array([command(th,xv[0],xv[1],t),DT]),nextxv])
        composed=step(composed,Z,DT)
        got=observations(system,composed)
        want=jnp.concatenate([nextxv,previous[:4],jnp.array([nextt,(tick+1)*10.]),nexta,nextth])
        maxerr=max(maxerr,float(jnp.max(jnp.abs(got-want))))
        np.testing.assert_allclose(got,want,atol=1e-10)
        actual_state,a,th=nxt,nexta,nextth
    # At every suffix, the imagined duration stream must equal the physical one.
    # This catches resubdividing the remaining horizon into extra decision times.
    suffix_error=0.
    imagined_jit=jax.jit(imagine)
    for tick in range(21):
        ts=jnp.concatenate([jnp.full(20-tick,DT),jnp.zeros(tick)])
        cx=jnp.concatenate([jnp.array([2.,.4,-.3,tick*DT]),ts])
        predicted=imagined_jit(theta,cx)
        reference_state=(((jnp.array([.4,-.3]),jnp.array([tick*DT])),theta),Z)
        for _ in range(20-tick): reference_state=realstep(reference_state,Z,DT)
        actual=unpack(reference_state)[0]
        suffix_error=max(suffix_error,float(jnp.max(jnp.abs(predicted-actual))))
        np.testing.assert_allclose(predicted,actual,atol=1e-12)
    zero=online(.5,0); zstate=jax.jit(run_step(zero))(zero.P.init,Z,DT)
    first=step(system.P.init,Z,DT)
    np.testing.assert_allclose(observations(zero,zstate)[:6],observations(system,first)[:6],atol=1e-12)
    save('checks',{'model_cases':cases,'gradient_error':float(jnp.max(jnp.abs(grad-finite))),
                  'schedule_replay_error':maxerr,'all_21_suffixes_error':suffix_error,'ten_steps_seconds':warm,'compile_seconds':compile_seconds,
                  'total_seconds':time.perf_counter()-started,'label':system.label})


def test_exact_command_wiring_factorization():
    started=time.perf_counter(); results=[]; field_error=0.; label=''
    for mass in [.25,1.,4.]:
        old=c.online(mass,10); new=c.online(mass,10,split_command=True)
        assert old.P.Q_dim==new.P.Q_dim==63
        for q in [jnp.zeros(63),jnp.linspace(-.2,.3,63),jnp.linspace(.6,-.1,63)]:
            np.testing.assert_allclose(old.out_f(q,c.Z),new.out_f(q,c.Z),atol=1e-14)
            np.testing.assert_allclose(old.in_f(q,c.Z,c.Z),new.in_f(q,c.Z,c.Z),atol=1e-14)
            for a,b in zip(old.omega(q,c.Z,c.Z),new.omega(q,c.Z,c.Z)):
                np.testing.assert_allclose(a,b,atol=1e-14)
                if a.size:field_error=max(field_error,float(jnp.max(jnp.abs(a-b))))
        os,ns=old.P.init,new.P.init
        assert sum(x.size for x in jax.tree.leaves(ns))==33
        ostep,nstep=jax.jit(c.run_step(old)),jax.jit(c.run_step(new))
        error=0.
        for tick in range(20):
            os,ns=ostep(os,c.Z,.1),nstep(ns,c.Z,.1)
            a,b=c.observations(old,os),c.observations(new,ns)
            np.testing.assert_allclose(a,b,atol=1e-12)
            error=max(error,float(jnp.max(jnp.abs(a-b))))
        results.append({'mass':mass,'intervals':20,'max_trace_error':error})
        label=new.label

def theta(state):
    return state[0][0]

def test_imagined_timing_and_wrong_model():
    started=time.perf_counter()
    arr=c.timing_learner(); step=jax.jit(c.run_step(arr)); s=arr.P.init
    first=step(s); np.testing.assert_allclose(theta(first),.1,atol=1e-14)
    t0=time.perf_counter()
    for _ in range(10): s=step(s)
    jax.block_until_ready(s); elapsed=time.perf_counter()-t0
    history=[]; s=arr.P.init
    for i in range(101):
        w=float(jax.nn.sigmoid(theta(s)))
        history.append({'updates':i,'wait_probability':w,'modeled_and_matching_reward':1+2*w,'opposite_reward':3-2*w})
        if i<100:s=step(s)
    assert all(b['modeled_and_matching_reward']>a['modeled_and_matching_reward'] for a,b in zip(history,history[1:]))
    assert all(b['opposite_reward']<a['opposite_reward'] for a,b in zip(history,history[1:]))
    cut=c.timing_learner(True); stopped=jax.jit(c.run_step(cut))(cut.P.init)
    assert float(theta(stopped))==0.
    pair=parallel_arrangements(arr,arr)
    together=jax.jit(c.run_step(pair))((arr.P.init,arr.P.init))
    for got in together:
        for x,y in zip(jax.tree.leaves(got),jax.tree.leaves(first)):
            np.testing.assert_allclose(x,y,atol=1e-14)
    identity=c.static(lambda m:m,0,0,'identity')
    with_id=compose_seq(arr,identity)
    got=jax.jit(c.run_step(with_id))((arr.P.init,Z))[0]
    for x,y in zip(jax.tree.leaves(got),jax.tree.leaves(first)):
        np.testing.assert_allclose(x,y,atol=1e-14)
