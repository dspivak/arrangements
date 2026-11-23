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
from arrangements.feeding import feeding

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

# The two checks of the hand-typed page model (feeding-model.js, feeding-arrangement.js) against the library
# left with that model on 2026-11-23; the page now steps with code generated from the composite, checked by
# tests/test_feeding_page.py and tests/test_export_step_js.py.


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


def test_feeding_one_and_three_casings_use_same_composition_api():
    for places,foods in [((0.,),(3.,)),((0.,1.,2.),(3.,4.,5.))]:
        arr=feeding(places,foods)
        _,world,coupling=arr.construction
        # the coupling is sent the seeker's six outputs, each casing's four and the pile's two
        assert coupling.label=='spatial coupling'
        assert world.out_dim_N==coupling.out_dim_M==6+4*len(foods)+2
        assert world.in_dim_N==coupling.in_dim_M==8


def _stores_by_key(arr):
    """Each store of a feeding composite by a key that does not depend on the count of casings
    (as the page's ``feedingStoreKeys``): its casing and label, or its path and label, and an
    occurrence count for repeats."""
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "export_step_js", Path(__file__).resolve().parents[1] / "scripts" / "export_step_js.py")
    exporter = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exporter)
    seen, keys, casing = {}, {}, 0
    for s in exporter.stores(arr):
        if s["path"] and s["path"][0].startswith("casing("):
            casing += s["label"] == "food"
            base = ("casing", casing - 1, s["label"])
        else:
            base = (tuple(s["path"]), s["label"])
        seen[base] = seen.get(base, 0) + 1
        keys[(*base, seen[base] - 1)] = s
    return keys


def test_a_single_seeker_releases_its_first_casing_before_feeding_from_the_next():
    """One casing held and eaten from; a second set down at the same place while the latch holds
    the first, its state carried over store by store as the page carries it: the second is not
    touched until the first is released, then both are eaten. (Until 2026-11-23 this ran on the
    hinged model with ``feeding.add_casing``.)"""
    from jax.flatten_util import ravel_pytree
    from arrangements.pcK import PhiK
    params = {'kexp': 0., 'ktau': 0.}
    step = dict(parameters=params, stages=4, h=.002, learning=False)
    one = feeding((0.,), (10.,), **step)
    two = feeding((0., 0.), (10., 10.), **step)

    def advance(world, flat, steps):
        dynamics = PhiK(world)
        unravel = ravel_pytree(world.P.init)[1]
        def tick(_, state):
            return dynamics.with_state(state).run_one((EMPTY,trivial_omega(0)), lambda _: (jnp.zeros(2),EMPTY))[1]
        return np.asarray(ravel_pytree(jax.lax.fori_loop(0,steps,tick,unravel(jnp.asarray(flat))))[0])

    keys_one, keys_two = _stores_by_key(one), _stores_by_key(two)
    at = lambda keys, key: keys[key]["start"]
    latch = lambda k: next(key for key in keys_two if key[-2] == f'contact latch {k} store')
    food = lambda k: ('casing', k, 'food', 0)
    held = advance(one, ravel_pytree(one.P.init)[0], 500)
    assert held[at(keys_one, latch(1))] > .99
    # Carry every store into the two-casing world; the new casing and its contact keep their start.
    carried = np.asarray(ravel_pytree(two.P.init)[0]).copy()
    for key, s in keys_two.items():
        if key in keys_one:
            carried[s["start"]:s["start"] + s["size"]] = held[keys_one[key]["start"]:keys_one[key]["start"] + s["size"]]
    during = advance(two, carried, 500)
    assert during[at(keys_two, food(0))] < held[at(keys_one, food(0))]
    assert during[at(keys_two, food(1))] == 10. and during[at(keys_two, latch(2))] == 0.
    after = advance(two, during, 3000)
    assert after[at(keys_two, food(0))] < .1 and after[at(keys_two, food(1))] < .1


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
