"""Inspection must agree with the library interpretation without stepping state.

Predictions: nonlinear composite and two-input tensor inclinations agree with
their factors; identity transfers preserve observations; exported frames retain
state rather than inventing an integrator box. These are software/model checks,
not claims about biological or social feeding.
"""
import json
import subprocess
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
from arrangements.arrangement import Arrangement, exact, zero_form
from arrangements.coalgebra import configuration
from arrangements.rvect import euclidean
from arrangements.wiring import compose_seq, parallel_arrangements, trivial_parameter
from arrangements.explore import inspect, scene, write_html
from arrangements.functors import Phi
from arrangements.interpretation import trivial_omega
from arrangements.feeding import feeding, pack_state, unpack_state
from arrangements.feeding import add_casing, couple, receiver, source

jax.config.update('jax_enable_x64', True)
EMPTY=jnp.zeros(0)

def primitive(value):
    return Arrangement(configuration(euclidean(1),jnp.array([value])),0,0,1,1,
        lambda q,m:q*q,lambda q,m,n:EMPTY,exact(lambda q,m,n:-.5*(q[0]-n[0])**2),label='quadratic')

def test_tensor_and_nonlinear_composite_trace():
    a,b=primitive(2.),primitive(3.)
    pair=parallel_arrangements(a,b)
    outer=Arrangement(trivial_parameter(),2,2,1,1,lambda q,m:jnp.array([m[0]*m[1]]),
        lambda q,m,n:jnp.array([n[0],2*n[0]]),exact(lambda q,m,n:-.1*m[0]*m[1]),label='reader')
    arr=compose_seq(pair,outer)
    trace=inspect(arr,inputs=[.7],loads=[1.3])
    left,right=trace['children']
    np.testing.assert_allclose(trace['inclination'],left['inclination'])
    np.testing.assert_allclose(left['inclination'],jnp.concatenate([r['inclination'] for r in left['children']]))
    np.testing.assert_allclose(left['n_out'],right['m_out'])
    np.testing.assert_allclose(left['n_load'],right['m_load'])
    before=jax.tree.map(lambda x:np.array(x),arr.P.init)
    for _ in range(2):inspect(arr,inputs=[.7],loads=[1.3])
    for x,y in zip(jax.tree.leaves(before),jax.tree.leaves(arr.P.init)):np.testing.assert_array_equal(x,y)

def test_identity_and_opaque_primitive_export(tmp_path):
    identity=Arrangement(trivial_parameter(),1,1,1,1,lambda q,m:m,lambda q,m,n:n,zero_form,label='id')
    r=inspect(identity,values=[2.],inputs=[3.],loads=[4.],incoming=lambda n:5*n)
    np.testing.assert_array_equal(r['n_out'],[2.]);np.testing.assert_array_equal(r['m_load'],[4.]);np.testing.assert_array_equal(r['n_in_load'],[15.])
    sc=scene(inspect(primitive(2.),inputs=[1.],loads=[3.]))
    assert any(n['kind']=='state' for n in sc['nodes'])
    assert not any(n['kind']=='box' and n['label']=='Internal state' for n in sc['nodes'])
    path=write_html(sc,tmp_path/'explorer.html')
    assert 'ArrangementExplorer.View' in path.read_text()
    assert 'E.layout=layout' in path.read_text()
    viewer=Path(__file__).resolve().parents[1]/'viewer'
    result=subprocess.run(['node','-e',"const E=require('./explorer-core.js');E.validate(JSON.parse(process.argv[1]));",json.dumps(sc)],cwd=viewer,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_tensor_scene_keeps_product_coordinates_and_boundary_directions():
    arr=parallel_arrangements(primitive(2.),primitive(3.))
    sc=scene(inspect(arr,inputs=[1.,2.],loads=[3.,4.]))
    assert all(n['side'] in ('in','out') for n in sc['nodes'] if n['kind']=='boundary')
    # Each coordinate goes directly to its own factor and carries its own load.
    at = lambda source, target: next(w for w in sc['wires'] if (w['source'],w['target'])==(source,target))
    assert at('root/ni/0','root/a/ni/0')['value'] == 1.
    assert at('root/ni/1','root/b/ni/0')['value'] == 2.
    assert at('root/a/no/0','root/no/0')['load'] == 3.
    assert at('root/b/no/0','root/no/1')['load'] == 4.
    viewer=Path(__file__).resolve().parents[1]/'viewer'
    script="""const E=require('./explorer-core.js');require('./explorer-layout.js');
    const sc=JSON.parse(process.argv[1]),m=new E.Model(sc);
    for(const n of sc.nodes)if(m.children(n.id).length){m.enter(n.id);const p=m.project();E.layout(p);
      if(p.nodes.some(n=>n.id.startsWith('boundary:')))throw Error('Explicit interface duplicated as a synthetic boundary');} """
    result=subprocess.run(['node','-e',script,json.dumps(sc)],cwd=viewer,capture_output=True,text=True)
    assert result.returncode==0,result.stderr

def test_feeding_browser_field_and_port_loads_match_library_phi():
    viewer=Path(__file__).resolve().parents[1]/'viewer'
    source="""const F=require('./feeding-model.js'),A=require('./feeding-arrangement.js');
    const samples=[];let s=F.START.slice();
    for(let i=0;i<=30000;i++){if([0,6000,10000,23000,30000].includes(i)){
      const a=A.evaluate(s,F.PLACES,F.DEFAULTS);samples.push({s,rate:a.rate,loads:F.PLACES.map((_,k)=>{
        const targets=['Food ','Opening ','Access ','Height ','Red ','Place '].map(x=>x+(k+1));
        const map=new Map(a.nodes.map(n=>[n.id,n]));
        const sourceNodes=new Set(a.nodes.filter(n=>{while(n){if(n.id==='casing-'+k)return true;n=map.get(n.parent);}return false;}).map(n=>n.id));
        return targets.map(label=>a.wires.filter(w=>w.label===label&&sourceNodes.has(w.source)&&!sourceNodes.has(w.target)).reduce((s,w)=>s+w.load,0));
      })});}s=F.rk4(s,.002,F.PLACES,F.DEFAULTS);}console.log(JSON.stringify(samples));"""
    samples=json.loads(subprocess.check_output(['node','-e',source],cwd=viewer,text=True))
    parameters=json.loads(subprocess.check_output(['node','-e',"console.log(JSON.stringify(require('./feeding-model.js').DEFAULTS))"],cwd=viewer,text=True))
    arr=feeding(parameters=parameters);phi=Phi(arr);tau=1e-5
    # JIT one real Phi step; no hand integration stands in for it.
    def step(state):
        _,fiber=phi.step(state,tau)
        return fiber((EMPTY,trivial_omega(0)))[1]((jnp.zeros(2),EMPTY))[1]
    step=jax.jit(step)
    for sample in samples:
        state=pack_state(sample['s'],3)
        new=unpack_state(step(state),3)
        np.testing.assert_allclose(new,jnp.array(sample['s'])+tau*jnp.array(sample['rate']),atol=1e-12)
        trace=inspect(arr,state)
        np.testing.assert_allclose(trace['children'][1]['m_load'][7:],np.array(sample['loads']).reshape(-1),atol=1e-10)

def test_feeding_four_rounds_agree_with_browser_rk4():
    from arrangements.pcK import PhiK
    viewer=Path(__file__).resolve().parents[1]/'viewer'
    parameters=json.loads(subprocess.check_output(['node','-e',"console.log(JSON.stringify(require('./feeding-model.js').DEFAULTS))"],cwd=viewer,text=True))
    arr=feeding(stages=4,h=.002,parameters=parameters)
    phi=PhiK(arr)
    # pcK rounds interface is exercised by its public run_one below.
    out,new=phi.run_one((EMPTY,trivial_omega(0)),lambda out:(jnp.zeros(2),EMPTY))
    viewer=Path(__file__).resolve().parents[1]/'viewer'
    expected=json.loads(subprocess.check_output(['node','-e',"const F=require('./feeding-model.js');console.log(JSON.stringify(F.rk4(F.START,.002,F.PLACES,F.DEFAULTS)));"],cwd=viewer,text=True))
    np.testing.assert_allclose(unpack_state(new,3),expected,atol=1e-12)


def test_multiround_inspection_reads_without_advancing():
    arr=feeding(stages=4,h=.002)
    observed=inspect(arr)
    np.testing.assert_allclose(observed['configuration'],arr.P.reads[0](arr.P.init))
    assert observed['children'][1]['label']=='spatial coupling'


def test_two_round_inspection_uses_actual_carrier():
    from dataclasses import replace
    from arrangements.coalgebra import as_two_stage
    a=primitive(2.)
    a=replace(a,P=as_two_stage(a.P))
    first=inspect(a)
    mid=a.P.advance(a.P.init,jnp.array([.2]))
    second=inspect(a,mid,round_index=1)
    np.testing.assert_allclose(first['configuration'],a.P.read1(a.P.init))
    np.testing.assert_allclose(second['configuration'],a.P.read2(mid))


def test_smooth_plot_observation_and_export():
    from arrangements import smooth
    arr=smooth.include(primitive(2.))
    r=inspect(arr,inputs=(jnp.array([1.]),))
    np.testing.assert_allclose(r['n_out'][0],[4.])
    np.testing.assert_allclose(r['inclination'],[-1.])
    assert r['n_in_load'] is None  # an unspecified plot is not a zero load
    sc=scene(r)
    json.dumps(sc,allow_nan=False)
    assert any(w['load'] is None for w in sc['wires'])


def test_feeding_zero_and_three_casings_use_same_composition_api():
    for places,foods in [((),()),((0.,1.,2.),(3.,4.,5.))]:
        arr=feeding(places,foods)
        flat=jnp.array([3.,-1.,-.4,0.,0.,0.,0.,6.,0.,8.]+[x for f in foods for x in [f,0.,-.4,0.,-.4,0.,0.,0.]])
        np.testing.assert_allclose(unpack_state(pack_state(flat,len(foods)),len(foods)),flat)
        r=inspect(arr,pack_state(flat,len(foods)))
        assert len(r['children'][1]['m_load'])==7+6*len(foods)


def test_a_single_receiver_releases_its_first_casing_before_feeding_from_the_next():
    params = {'kexp': 0., 'ktau': 0.}
    step = dict(parameters=params, stages=4, h=.002)
    body = receiver(**step)
    apple = source(0., 10., **step)
    for count in (0, 1, 2, 4):
        world = couple(body, [apple]*count, **step)
        assert world.construction[1].construction[1] is body
        assert body.in_dim_N == body.out_dim_N == 7

    from arrangements.pcK import PhiK
    def advance(world, state, steps):
        dynamics = PhiK(world)
        def tick(_, state):
            return dynamics.with_state(state).run_one((EMPTY,trivial_omega(0)),
                                                     lambda _: (jnp.zeros(2),EMPTY))[1]
        return jax.lax.fori_loop(0,steps,tick,state)

    empty_world = couple(body,[],**step)
    first, initial = add_casing(body,[],empty_world.P.init,apple,**step)
    assert initial[0][0] is empty_world.P.init[0][0]
    held = advance(first,initial,500)
    assert held[1][1] > .99
    # Carry the actual body and the first contact into the enlarged world.
    (same_body,old_apple),_ = held
    both,inserted = add_casing(body,[apple],held,apple,**step)
    assert inserted[0][0] is same_body
    during = advance(both,inserted,500)
    flat = unpack_state(during,2)
    # The casing's state nests by its parts now; its food is the first number of it.
    from jax.flatten_util import ravel_pytree
    assert flat[10] < ravel_pytree(old_apple)[0][0] and flat[18] == 10. and flat[19] == 0. and flat[25] == 0.
    after = unpack_state(advance(both,during,3000),2)
    assert after[10] < .1 and after[18] < .1


def test_separate_coordinate_maps_keep_inactive_inputs_and_individual_loads():
    a = Arrangement(configuration(euclidean(1), jnp.array([2.])), 1, 2, 1, 3,
                    lambda q,m: jnp.array([m[0]*m[1], q[0], 7.]),
                    lambda q,m,n: n - m[1],
                    lambda q,m,n: (n, jnp.array([q[0], 0.]), jnp.zeros(1)))
    trace = inspect(a, values=[0.,3.], inputs=[5.], loads=[2.,3.,4.], incoming=lambda z: z)
    sc = scene(trace)
    def reads(id):
        return {w['label'] for w in sc['wires'] if w['target']=='root/map/'+id and w['argument']}
    assert reads('no/0') == {'Internal output 1','Internal output 2'}
    assert reads('no/1') == {'Configuration'}
    assert reads('no/2') == set()
    assert reads('mi/0') == {'External input','Internal output 2'}
    assert reads('mo/0') == {'Configuration'}
    assert reads('q/0') == {'External input'}
    assert all(n.get('possibleInputs') is not True for n in sc['nodes'])
    for node,load in [('root/state',trace['inclination'][0]),('root/mo/0',trace['m_load'][0]),
                      ('root/mo/1',trace['m_load'][1]),('root/ni/0',trace['n_in_load'][0])]:
        assert sum(w['load'] for w in sc['wires'] if w['source']==node) == load


def test_unresolved_coordinate_rules_are_labeled_and_smooth_transfers_keep_plot_loads():
    a = Arrangement(trivial_parameter(), 0, 2, 0, 1,
                    lambda q,m: jax.lax.cond(m[0]>0, lambda x:x[:1], lambda x:x[1:], m),
                    lambda q,m,n: EMPTY, zero_form)
    sc = scene(inspect(a, values=[0.,2.]))
    result = next(n for n in sc['nodes'] if n['id']=='root/map/no/0')
    assert result['possibleInputs'] and result['result']==2.
    from arrangements import smooth
    arr = smooth.include(primitive(2.))
    sc = scene(inspect(arr, inputs=(jnp.array([4.]),)))
    assert [(w['label'],w['value']) for w in sc['wires'] if w['target']=='root/map/no/0'] == [('Configuration',2.)]
    plot = next(n for n in sc['nodes'] if n['id']=='root/loads')
    assert dict((x['label'],x['value']) for x in plot['observations'])['Configuration inclination']==[2.]
