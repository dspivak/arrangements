"""P76 (j) predictions, checked on composed smooth arrangements through PhiK.

Fixed circuits: induction, reciprocal fluxes, midpoint step, energy and time
convergence. Pair/matrix formulas below are independent oracles, not library
couplings. The quasistatic per-length model is reasonable for separated long
coils; instantaneous action and arbitrary self-inductances are its limitations.
"""
import time
import jax
import numpy as np
import pytest
from magnetic_checks import checked_jit, outside_smoothing_discs

pytestmark = pytest.mark.usefixtures("outside_smoothing_discs")
from arrangements import magnetic_circuits as C, smooth as sm


def matrix(places, inductances, widths, turns):
    """Independent line-current flux sum, in units mu0=1, per length."""
    n = len(places)
    result = np.diag(np.array(inductances, float))
    ends = []
    for (x, y, theta), h in zip(places, widths):
        axis = h * np.array([np.cos(theta), np.sin(theta)])
        ends.append([np.array([x, y]) + axis, np.array([x, y]) - axis])
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            result[i, j] = sum(-a * b * turns[i] * turns[j] * np.log(np.linalg.norm(u - v)) / (2*np.pi)
                               for a, u in zip([1,-1], ends[i]) for b, v in zip([1,-1], ends[j]))
    return result


def run(places, L, widths, turns, Q0, tau=0.05, K=16, steps=10):
    arr = C.fixed_circuits([C.circuit(l, h=h, charge0=q, tau=tau, K=K, turns=n)
                           for l,h,n,q in zip(L,widths,turns,Q0)], places)
    O = sm.PhiK(arr)
    step = checked_jit(lambda s: O.with_state(s).run_one(sm.CLOSED_IN, sm.closed_tell)[1])
    flat = lambda s: np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(s)])
    state = O.state
    output = [flat(state)]
    start = time.monotonic()
    for _ in range(steps):
        state = step(state)
        output.append(flat(state))
    return arr, np.array(output).reshape(steps+1, len(Q0), 3), time.monotonic()-start


PLACES = [[-1,0,0],[1,0,0]]


def test_two_circuits_recover_midpoint_and_energy():
    arr, states, elapsed = run(PLACES,[1,1],[.5,.5],[1,1],[1,0],steps=800)
    L = matrix(PLACES,[1,1],[.5,.5],[1,1])
    assert np.allclose(L,L.T)
    A = np.block([[np.zeros((2,2)), -np.linalg.inv(L)], [np.eye(2),np.zeros((2,2))]])
    step = np.linalg.solve(np.eye(4)-.025*A,np.eye(4)+.025*A)
    oracle = np.array([1.,0.,0.,0.])
    gap = 0
    for s in states[1:]:
        oracle = step@oracle
        gap = max(gap,np.max(np.abs(np.r_[s[:,0],s[:,1]]-oracle)))
    Q,phi,I=states[:,:,0],states[:,:,1],states[:,:,2]
    energy = (np.sum(Q**2,axis=1)+np.einsum('bi,ij,bj->b',I,L,I))/2
    residual = np.max(np.abs(I@L.T-phi))
    print('\nlabel',arr.label,'seconds',elapsed,'M',L[0,1], 'max capacitor 2 energy', np.max(Q[:,1]**2/2),
          'midpoint gap',gap,'energy range',np.ptp(energy),'flux residual',residual)
    assert gap < 1e-10 and residual < 1e-10 and np.ptp(energy)<1e-8
    assert np.max(np.abs(Q[:,1]))>.5


def test_step_halving_tends_to_exact_flow():
    L = matrix(PLACES,[1,1],[.5,.5],[1,1])
    eigen, basis = np.linalg.eigh(L)
    errors=[]
    for tau in [.2,.1,.05]:
        _,s,_=run(PLACES,[1,1],[.5,.5],[1,1],[1,0],tau=tau,steps=round(40/tau))
        coords=basis.T@np.array([1.,0.])
        Q=basis@(np.cos(40/np.sqrt(eigen))*coords)
        phi=basis@(np.sqrt(eigen)*np.sin(40/np.sqrt(eigen))*coords)
        errors.append(np.max(np.abs(s[-1,:,:2]-np.stack([Q,phi],axis=1))))
    print('\ntrajectory errors', errors, 'ratios',np.array(errors[:-1])/errors[1:])
    assert all(3.5<a/b<4.5 for a,b in zip(errors,errors[1:]))


def test_reverse_winding_and_disconnect():
    _,a,_=run(PLACES,[1,1],[.5,.5],[1,1],[1,0],steps=100)
    _,b,_=run(PLACES,[1,1],[.5,.5],[1,-1],[1,0],steps=100)
    _,z,_=run(PLACES,[1,1],[.5,.5],[1,0],[1,0],steps=100)
    assert np.allclose(a[:,0],b[:,0],atol=1e-12)
    assert np.allclose(a[:,1],-b[:,1],atol=1e-12)
    assert np.max(np.abs(z[:,1]))==0


def test_three_distinct_circuits_and_geometry():
    places=[[-1,0,.2],[1,.3,1.1],[0,2,-.4]]
    ls,hs,ns=[1,1.2,.8],[.4,.3,.5],[1,-.8,.7]
    _,s,_=run(places,ls,hs,ns,[1,.2,0],steps=100)
    L=matrix(places,ls,hs,ns)
    assert np.allclose(L,L.T,atol=1e-14)
    assert np.max(np.abs(s[:,:,2]@L.T-s[:,:,1]))<1e-10
    E=(np.sum(s[:,:,0]**2,axis=1)+np.einsum('bi,ij,bj->b',s[:,:,2],L,s[:,:,2]))/2
    assert np.ptp(E)<1e-10


def test_rounds_reduce_residual_and_noncontraction_is_visible():
    gaps=[]
    L=matrix(PLACES,[1,1],[.5,.5],[1,1])
    for K in [2,4,8,16]:
        _,s,_=run(PLACES,[1,1],[.5,.5],[1,1],[1,0],K=K,steps=1)
        gaps.append(np.max(np.abs(s[-1,:,2]@L.T-s[-1,:,1])))
    print('\nround residuals',gaps)
    assert gaps[-1]<1e-12 and gaps[0]>1e-5
    # Positive inductance matrix, but a step far outside this iteration's
    # contraction range: the code must not be mistaken for an exact solver.
    _,bad,_=run(PLACES,[1,1],[.5,.5],[1,1],[1,0],tau=4,K=16,steps=2)
    assert np.max(np.abs(bad))>100


def test_body_type_and_plot_naturality():
    import jax.numpy as jnp
    from arrangements.magnets import FIELDS
    a=C.circuit()
    assert a.N_in==a.N_out==FIELDS
    winding=C.winding(1,.5)
    def plot(u):
        return (jnp.array([u[0],u[1],0.0]),(),(lambda r:u[2]*r[0],lambda r:0.,jnp.array([u[3]])))
    reparam=lambda v:jnp.array([v[0]**2,v[1],v[0]*v[1],v[2]])
    v=jnp.array([.3,.7,-.4])
    _,pull=jax.vjp(reparam,v)
    assert np.allclose(winding.omega(lambda v:plot(reparam(v)),v),pull(winding.omega(plot,reparam(v)))[0])
