"""P76 (j): external position pushback and a held/free cart composite.

All dynamics run through smooth.PhiK. Same physical limits as magnets.py;
externally imposed positions supply/remove mechanical work.
"""
import pytest
from magnetic_checks import checked_jit, outside_smoothing_discs

pytestmark = pytest.mark.usefixtures("outside_smoothing_discs")
import jax
import jax.numpy as jnp
import numpy as np
from arrangements import magnets as M, open_magnets as OM, smooth as sm


def pushback(bodies,positions,cart_of=None):
    arr=OM.open_system(bodies,cart_of)
    O=sm.PhiK(arr)
    def tell(op):
        return jax.tree_util.tree_map(jnp.zeros_like,op[0][1]),positions
    outputs,_=O.run_one(sm.CLOSED_IN,tell)
    omega=outputs[0][1]
    return arr,omega(lambda x:x,positions),omega


def test_open_pushback_equals_plane_on_magnets_loop_and_bumpers():
    bodies=[M.magnet(1,.25),M.dead_loop(1,.4),M.bumper(1,.25),M.bumper(.7,0)]
    cart_of=[0,1,0,1]
    positions=(jnp.array([-1.,.1,.4]),jnp.array([1.,.2,-.3]))
    arr,loads,omega=pushback(bodies,positions,cart_of)
    closed=M.system(bodies,positions,np.zeros((2,3)),np.ones((2,3)),cart_of=cart_of)
    q=jnp.concatenate(positions)
    want=closed.omega(lambda x:(x,(),()),q)
    assert np.allclose(np.concatenate(loads),want,atol=1e-12)
    assert np.linalg.norm(want)>1e-4
    print('\nlabel',arr.label,'loads',loads,'gap',np.max(np.abs(np.concatenate(loads)-want)))


def test_units_and_nonlinear_plot_naturality():
    parts=[M.magnet(1,.25),M.magnet(-.7,.4)]
    positions=(jnp.array([-1.,.1,.4]),jnp.array([1.,.2,-.3]))
    a,loads,omega=pushback(parts,positions)
    for other in [sm.compose_seq(sm.identity((),()),a),
                  sm.compose_seq(a,sm.identity(a.N_in,a.N_out))]:
        got=other.omega(lambda x:(jnp.zeros(other.P.Q_dim),(),x),positions)
        assert np.allclose(np.concatenate(got),np.concatenate(loads))
    def plot(u):
        return (positions[0]+jnp.array([u[0]**2,u[1],0.]),
                positions[1]+jnp.array([u[0],0.,u[1]*u[0]]))
    u=jnp.array([.2,-.3])
    _,pull=jax.vjp(plot,u)
    assert np.allclose(omega(plot,u),pull(omega(lambda x:x,plot(u)))[0])


def test_no_held_carts_is_existing_field():
    a,b=OM.open_field(2,held=[]),M.field(2)
    parts=sm.tensor([M.magnet(1,.25),M.magnet(1,.4),
                     M.cart([1,1,1],[-1,0,.3],[0,0,0]),
                     M.cart([1,1,1],[1,0,-.4],[0,0,0])])
    X,Y=sm.PhiK(sm.compose_seq(parts,a)),sm.PhiK(sm.compose_seq(parts,b))
    x,y=X.state,Y.state
    sx=checked_jit(lambda s:X.with_state(s).run_one(sm.CLOSED_IN,sm.closed_tell)[1])
    sy=checked_jit(lambda s:Y.with_state(s).run_one(sm.CLOSED_IN,sm.closed_tell)[1])
    for _ in range(10): x,y=sx(x),sy(y)
    for p,q in zip(jax.tree_util.tree_leaves(x),jax.tree_util.tree_leaves(y)):
        assert np.allclose(p,q,atol=1e-12)


def held_run(steps=100,tau=.01):
    bodies=[M.magnet(1,.25),M.magnet(.7,.4)]
    held=(jnp.array([-1.,.1,.4]),)
    free=M.cart([1,1,.2],[1,.2,-.3],[0,0,0],tau=tau)
    arr=OM.open_system(bodies,held=[0],carts=[free])
    O=sm.PhiK(arr)
    def tell(op):return jax.tree_util.tree_map(jnp.zeros_like,op[0][1]),held
    step=checked_jit(lambda s:O.with_state(s).run_one(sm.CLOSED_IN,tell)[1])
    state=O.state
    flat=lambda s:np.concatenate([np.asarray(x).ravel() for x in jax.tree_util.tree_leaves(s)])
    out=[flat(state)]
    for _ in range(steps):
        state=step(state)
        out.append(flat(state))
    return arr,np.array(out)


def test_one_held_one_free():
    arr,s=held_run()
    assert np.isfinite(s).all()
    assert np.linalg.norm(s[-1]-s[0])>1e-4
    print('\nheld/free label',arr.label,'final free cart',s[-1])
