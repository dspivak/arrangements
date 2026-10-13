"""P44 predictions: notes/runs/p44-bregman-codex-20261013/RESULTS.md.

Closed-form and one-step checks, not convergence or biological evidence.
Every modeled update is Phi of a composed arrangement. Private prediction
error is a cost; these tests do not identify it with integrity or need.
"""
import json
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from arrangements import attention as A
from arrangements.bregman_attention import bregman_arrangement, divergence
from arrangements.arrangement import Arrangement, exact
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean, trivial, ReactiveVectorSpace
from arrangements.wiring import compose_seq, tensor_arrangements
from arrangements.functors import Phi

ZERO = jnp.zeros(0)
EPS = A.PEAK_EPS


def flat(state):
    return np.concatenate([np.asarray(x).reshape(-1) for x in jax.tree_util.tree_leaves(state)])


def close(actual, expected):
    np.testing.assert_allclose(actual, expected, rtol=2e-11, atol=2e-12)


def phi_step(arr, m, e, incoming=None, xi=None):
    incoming = incoming or (lambda x: jnp.zeros_like(x))
    xi = jnp.zeros(arr.out_dim_N) if xi is None else xi
    return flat(Phi(arr).run_one((m, incoming), lambda _: (xi, e))[2])


def potential(kind):
    if kind == 'bell':
        return lambda x: 0.5*jnp.vdot(x, x)
    return lambda x: jnp.sum(x*jnp.log(x)-x)


def node(n, kind, offset, name):
    # Positive sites for entropy, smoothly parameterized by all-real log sites.
    q = jnp.log(jnp.linspace(0.7+offset, 1.7+offset, n+1))
    endpoints = lambda q, i, j: (jnp.exp(q[j:j+1]), jnp.exp(q[i:i+1]))
    kwargs = {'potential':potential(kind)}
    if kind == 'peak':
        endpoints = lambda q, i, j: (jnp.zeros(1), jnp.sqrt((jnp.exp(q[j:j+1])-jnp.exp(q[i:i+1]))**2+EPS**2)-EPS)
        kwargs = {'divergence_fn':lambda x,y:jnp.sum(y)}
    return bregman_arrangement(n, 2, 1,
        parameter=configuration(euclidean(n+1, .01), init=q),
        placement=lambda q: jnp.exp(q[1:, None]),
        endpoints=endpoints, **kwargs,
        # Noncommuting maps detect a reversed path in signal transfer.
        values=lambda q: jnp.stack([jnp.array([[1., .1*(j+1)], [.07*j, .8]]) for j in range(n+1)]),
        cost=lambda q, m, e: .025*jnp.sum((e-jnp.sum(m)*.1)**2), label=name)


@pytest.mark.parametrize('kind', ['bell', 'entropy', 'peak'])
def test_three_level_nesting(kind):
    root = node(2, kind, .2, 'root')
    mids = [node(1, kind, .1+i*.2, f'middle-{i}') for i in range(2)]
    leaves = [node(1, kind, .3+i*.2, f'leaf-{i}') for i in range(2)]
    groups = [A.substitute(m, [l]) for m, l in zip(mids, leaves)]
    direct = A.substitute(root, groups)
    wired = compose_seq(tensor_arrangements([
        compose_seq(l.arr(), m.arr()) for l, m in zip(leaves, mids)]), root.arr())
    q = direct.P.position(direct.P.init)
    # Expected path from exposed child 2 to child 1, independent scalar formula.
    path = [(leaves[0],1,0), (mids[0],1,0), (root,1,2),
            (mids[1],0,1), (leaves[1],0,1)]
    xs, ys = [], []
    signal_map = np.eye(2)
    for a,i,j in path:
        sites = np.exp(np.asarray(a.P.init))
        xs.append(sites[j]); ys.append(sites[i])
        signal_map = signal_map @ np.array([[1., .1*(j+1)], [.07*j, .8]])
    xs, ys = np.array(xs), np.array(ys)
    exponent = (.5*np.sum((xs-ys)**2) if kind=='bell'
                else np.sum(xs*np.log(xs/ys)-xs+ys))
    # Direct sum: ordinary interior divergence, or entropy's boundary restriction.
    if kind=='peak':
        exponent=np.sum(np.sqrt((xs-ys)**2+EPS**2)-EPS)
        boundary_distances=jnp.sqrt((jnp.array(xs)-jnp.array(ys))**2+EPS**2)-EPS
        close(jnp.sum(boundary_distances),exponent)
    else:
        close(divergence(potential(kind), jnp.array(xs), jnp.array(ys)), exponent)
    T = direct.transfers(q)[1][2]
    close(T.gamma, np.exp(-exponent)); close(T.L, np.exp(-exponent)*signal_map)
    if kind == 'entropy':
        assert abs(float(T.gamma)-float(direct.transfers(q)[2][1].gamma)) > 1e-5
    m = jnp.array([.2, -.3, 1., .7, .4, .8]); e = jnp.array([.1,.3,.5,.2])
    close(direct.sigma(q,m), wired.out_f(q,m))
    close(direct.rho(q,m,e), wired.in_f(q,m,e))
    for x,y in zip(direct.arr().omega(q,m,e), wired.omega(q,m,e)): close(x,y)
    incoming = lambda x: -.03*x
    a, b = Phi(direct.arr()), Phi(wired)
    for _ in range(3):
        sa = a.run_one((m,incoming), lambda _: (jnp.array([.2,-.1,.3]),e))[2]
        sb = b.run_one((m,incoming), lambda _: (jnp.array([.2,-.1,.3]),e))[2]
        close(flat(sa), flat(sb)); a=a.with_state(sa); b=b.with_state(sb)
    # Other bracketing, with the product-factor coherence permutation.
    other = A.substitute(A.substitute(root,mids), leaves)
    qo = jnp.concatenate([leaves[0].P.init,leaves[1].P.init,mids[0].P.init,mids[1].P.init,root.P.init])
    close(other.sigma(qo,m), direct.sigma(q,m)); close(other.rho(qo,m,e), direct.rho(q,m,e))
    close(other.cost(qo,m,e), direct.cost(q,m,e))
    print(json.dumps({'stage':'nesting','kind':kind,'path_divergence':float(exponent),
                      'weight':float(T.gamma),'label':wired.label}))


def component(message, cost, dim_P, name):
    d = len(message)+dim_P
    return Arrangement(P=configuration(trivial()), in_dim_M=0,out_dim_M=0,
        in_dim_N=d,out_dim_N=len(message),out_f=lambda q,m: message,
        in_f=lambda q,m,e: ZERO, omega=exact(lambda q,m,e: -cost(e)),label=name)


def receiver_system(kind, distance, mobility=.05, mu=0., target=(1.,0.)):
    # Only the receiver moves. Two distinct signal coordinates make each
    # sender's marginal contribution to the private prediction checkable.
    positions = lambda q: jnp.array([0.,q[0],distance,-distance])
    def endpoints(q,i,j):
        p=positions(q)
        if kind=='bell': return p[j:j+1],p[i:i+1]
        d=jnp.sqrt((p[j]-p[i])**2+EPS**2)-EPS
        return jnp.zeros(1),jnp.reshape(d,(1,))
    kwargs = ({'potential':potential('bell')} if kind=='bell' else
              {'divergence_fn':lambda x,y: jnp.sum(y)}) # continuous boundary restriction D(0,d)
    parent=bregman_arrangement(3,2,1, parameter=configuration(euclidean(1,mobility),init=jnp.array([mu])),
        placement=lambda q: positions(q)[1:,None],endpoints=endpoints,label=kind+'-parent',**kwargs)
    private=lambda e: .5*jnp.sum((e[:2]-jnp.array(target))**2)
    parts=[component(jnp.zeros(3),private,1,'private-prediction'),
           component(jnp.array([1.,0.,1.]),lambda e:0.,1,'source-positive'),
           component(jnp.array([0.,1.,1.]),lambda e:0.,1,'source-negative')]
    return compose_seq(tensor_arrangements(parts),parent.arr())


@pytest.mark.parametrize('kind',['bell','peak'])
@pytest.mark.parametrize('distance',[.02,1.,5.])
def test_receiver_near_and_far(kind,distance):
    arr=receiver_system(kind,distance,mobility=.005); e=jnp.zeros(4)
    g=np.exp(-distance**2/2) if kind=='bell' else np.exp(-(np.sqrt(distance**2+EPS**2)-EPS))
    slope=distance if kind=='bell' else distance/np.sqrt(distance**2+EPS**2)
    help_contribution=g*(1-g)*slope
    harm_contribution=g*g*slope
    expected=help_contribution+harm_contribution
    force=float(arr.omega(jnp.zeros(1),ZERO,e)[0][0])
    close(force,expected)
    q1=phi_step(arr,ZERO,e)[0]; close(q1,.005*expected)
    assert 0<q1<distance
    gain=lambda d: np.exp(-d*d/2) if kind=='bell' else np.exp(-(np.sqrt(d*d+EPS**2)-EPS))
    assert gain(distance-q1)>g and gain(distance+q1)<g
    # Cost signs, not sender names, cause the movement: reverse the prediction.
    reversed_arr=receiver_system(kind,distance,mobility=.005,target=(0.,1.))
    close(phi_step(reversed_arr,ZERO,e)[0],-q1)
    print(json.dumps({'stage':'receiver','kernel':kind,'distance':distance,'weight':float(g),
                      'help_push':float(help_contribution),'harm_push':float(harm_contribution),
                      'move':float(q1),'label':arr.label}))


@pytest.mark.parametrize('kind',['bell','peak'])
def test_coincidence_is_stationary(kind):
    arr=receiver_system(kind,0.,target=(0.,0.))
    close(arr.omega(jnp.zeros(1),ZERO,jnp.zeros(4))[0],jnp.zeros(1))
    close(phi_step(arr,ZERO,jnp.zeros(4)),[0.])
    # Cost wants both inputs reduced, yet the spatial derivative is zero.
    print(json.dumps({'stage':'coincidence','kernel':kind,'move':0.,'private_cost':1.}))


@pytest.mark.parametrize('kind',['bell'])
def test_inverse_hessian_mean_step(kind):
    # Quadratic potential is global; positive source sites chosen for comparison.
    # The log-weight cost uses the attention interface’s positive-weight domain.
    sources=jnp.array([[.7,1.5],[2.,.6]])
    mu=jnp.array([1.1,.9]); loud=jnp.array([.8,1.3]); phi=potential(kind)
    sharp=lambda q: jnp.linalg.inv(jax.hessian(phi)(q))
    parent=bregman_arrangement(2,1,2,
        parameter=configuration(ReactiveVectorSpace(2,sharp),init=mu),
        placement=lambda q:sources,
        endpoints=lambda q,i,j: (q if j==0 else sources[j-1],q if i==0 else sources[i-1]),
        potential=phi,label='Bregman-senders')
    parts=[component(jnp.array([0.,loud[j]]),lambda e:0.,2,f'source-{j}') for j in range(2)]
    send=compose_seq(tensor_arrangements(parts),parent.arr())
    # The log cost is its own consuming box; composed, not a feedback callback.
    loss=Arrangement(P=configuration(trivial()),in_dim_M=4,out_dim_M=2,
        in_dim_N=0,out_dim_N=0,out_f=lambda q,m:ZERO,in_f=lambda q,m,e:jnp.zeros(4),
        omega=exact(lambda q,m,e:jnp.log(m[-1])),label='negative-log-weight')
    arr=compose_seq(send,loss)
    gam=loud*jnp.exp(-jax.vmap(lambda x:divergence(phi,x,mu))(sources))
    center=np.asarray(jnp.sum(gam[:,None]*sources,axis=0)/jnp.sum(gam))
    close(phi_step(arr,ZERO,ZERO),center)
    print(json.dumps({'stage':'mean-step','kind':kind,'mean':center.tolist(),'label':arr.label}))


def test_near_peak_overshoots_at_original_step():
    """Retain the refuted .05-step prediction, not just the repaired .005 case."""
    distance=.02
    arr=receiver_system('peak',distance,mobility=.05)
    q1=phi_step(arr,ZERO,jnp.zeros(4))[0]
    assert q1>2*distance
    print(json.dumps({'stage':'overshoot','distance':distance,'move':float(q1),
                      'new_help_distance':float(abs(distance-q1))}))


def test_entropy_private_prediction_force_and_update():
    """General interior Bregman formula, independently checked in log coordinates.

    Log coordinates keep every site positive under every finite Euler step.
    This is a different finite-step integrator from Euler in the site itself;
    an inverse-Hessian response does not make those discretizations identical.
    """
    sources=jnp.array([[.7,1.5],[2.,.6]])
    mu=jnp.array([1.1,.9]); q=jnp.log(mu); phi=potential('entropy')
    sites=lambda q:jnp.vstack([jnp.ones(2),jnp.exp(q)[None,:],sources])
    parent=bregman_arrangement(3,2,2,
        parameter=configuration(ReactiveVectorSpace(2,lambda q:.01*jnp.diag(jnp.exp(-q))),init=q),
        placement=lambda q:sites(q)[1:],
        endpoints=lambda q,i,j:(sites(q)[j],sites(q)[i]),potential=phi,label='entropy-log-sites')
    parts=[component(jnp.zeros(3),lambda e:.5*jnp.sum((e[:2]-jnp.array([1.,0.]))**2),2,'private-prediction'),
           component(jnp.array([1.,0.,1.]),lambda e:0.,2,'source-positive'),
           component(jnp.array([0.,1.,1.]),lambda e:0.,2,'source-negative')]
    arr=compose_seq(tensor_arrangements(parts),parent.arr())
    gam=np.exp(-np.sum(np.asarray(sources)*np.log(np.asarray(sources)/np.asarray(mu))-np.asarray(sources)+np.asarray(mu),axis=1))
    help=gam*(np.array([1.,0.])-gam)
    site_force=np.diag(1/np.asarray(mu))@np.sum(help[:,None]*(np.asarray(sources)-np.asarray(mu)),axis=0)
    log_force=np.asarray(mu)*site_force
    close(arr.omega(q,ZERO,jnp.zeros(5))[0],log_force)
    actual=phi_step(arr,ZERO,jnp.zeros(5))
    close(actual,np.asarray(q)+.01*log_force/np.asarray(mu))
    assert np.all(np.exp(actual)>0)
    # The negative-log-weight special case's proposed site-Euler step, as an
    # ordinary algebraic check beside this arrangement, not its Phi update.
    weights=gam/gam.sum()
    center=np.sum(weights[:,None]*np.asarray(sources),axis=0)
    logweight_force=np.diag(1/np.asarray(mu))@(center-np.asarray(mu))
    close(np.asarray(mu)+np.diag(np.asarray(mu))@logweight_force,center)
    print(json.dumps({'stage':'entropy-private','log_force':log_force.tolist(),
                      'new_site':np.exp(actual).tolist(),'help_signs':help.tolist(),
                      'mean_identity':'algebraic check only; site-Euler update is partial'}))
