const test=require('node:test'),assert=require('node:assert/strict');
// The explorer's model on the feeding scene read from the library composite (feeding-explorer-data.js, written by
// scripts/export_feeding_explorer.py) through its adapter. Until 2026-11-23 these tests ran on the hand-typed
// graph feeding-arrangement.js and its models feeding-model.js and food-learning.js, retired then (subagent of
// c5f9ecc9, Claude Opus 5.5). Cases that only pinned the hand-typed graph against those models were dropped: the
// apple added beside the held apple, the field against the independent model, the 60-unit factored episode, the
// learning observer, the digestion-effort readouts, the fingers' dependencies, and the per-motor list of heard
// wires; feeding-explorer-composite.test.js checks every drawn formula against the library's loads and every motor's
// two pushes and walls, tests/test_explore_composite.py the loads at every store against the composite's own field,
// and feeding-slot.test.js a casing added mid-run.
const E=require('./explorer-core.js'),C=require('./feeding-explorer-composite.js'),data=require('./feeding-explorer-data.js');
const scene=()=>C.compositeAdapter(data).snapshot();
test('fork load adds signed branch contributions independently of display scale',()=>{
  const branches=[{id:'a',load:[2,-1],loadScale:3},{id:'b',load:[-2,4],loadScale:8}];
  const trunk=E.forkWire(branches);
  assert.deepEqual(trunk.load,[0,3]);assert.equal(trunk.loadScale,8);
  assert.deepEqual(trunk.branches,branches);
  assert.equal(E.forkWire([{load:1},{load:-1}]).load,0);
  assert.equal(E.forkWire([{load:1},{load:null}]).load,null);
  assert.equal(E.forkWire([{load:[1,2]},{load:[2]}]).load,null);
  assert.notEqual(E.format(1e-12),E.format(0));
});
const close=(a,b,tol=1e-10)=>{assert.equal(a.length,b.length);a.forEach((v,i)=>assert.ok(Math.abs(v-b[i])<tol,`${i}: ${v} vs ${b[i]}`));};
test('navigation exposes the seeker, then its wheels, while outside parts become boundary ports',()=>{
  const m=new E.Model(scene());assert(m.enter('seeker-with-a-slot'));
  const p=m.project();assert(!p.nodes.some(n=>n.id==='casing-4'));assert(p.nodes.some(n=>n.kind==='boundary'));
  assert(m.enter('wheels'));assert(!m.project().nodes.some(n=>n.id==='seeker-with-a-slot'));
  assert(m.project().nodes.some(n=>n.id==='cart'&&n.kind==='state'));
});
test('group, nested group, ungroup, undo and saved layouts preserve every crossing observation',()=>{
  const sc=scene(),original=JSON.stringify(sc),m=new E.Model(sc);m.enter('body-and-motors');
  const id=m.group(['seeing','wheels'],'Guidance');assert(m.project().nodes.some(n=>n.id===id));
  assert.equal(JSON.stringify(sc),original);m.enter(id);assert.equal(m.children().length,2);
  for(const w of m.project().wires){const real=sc.wires.find(x=>x.id===w.id);if(!real)continue;assert.deepEqual(w.value,real.value);assert.deepEqual(w.load,real.load);}
  const saved=m.save(),other=new E.Model(sc,saved);assert.equal(other.scope,id);assert.equal(other.children().length,2);
  m.enter('body-and-motors');const outer=m.group([id,'hand'],'Movement');m.enter(outer);assert(m.ungroup(id));
  assert.equal(m.children().length,3);assert(m.undo());assert(m.nodes.has(id));
  m.resetGroups();assert.equal(m.groups.length,0);assert(m.undo());assert.equal(m.groups.length,2);
});
test('invalid hierarchy and saved layouts are rejected, unknown loads remain unknown',()=>{
  const sc=scene(),m=new E.Model(sc);assert.throws(()=>m.group(['eye','casing-4']));
  assert.equal(m.restore({version:1,root:sc.root,groups:[{id:'bad',parent:sc.root,members:['bad','seeker-with-a-slot']}]}),false);
  const bad=JSON.parse(JSON.stringify(sc));bad.nodes.find(n=>n.id==='seeker-with-a-slot').parent='mouth';assert.throws(()=>E.validate(bad));
  assert.equal(E.intensity(null),null);assert.equal(E.packetSpeed(null,1,'load'),null);
});
test('a load sets its packets\' speed at one spacing; a value\'s packets are steady',()=>{
  // packets per unit of model time passing a point: speed over the one spacing
  const P=E.packets,rate=(x,s)=>{const v=E.packetSpeed(x,s,'load');return v?v/P.spacing:0;};
  assert.equal(rate(0,1),0);
  // feeding's nonzero loads span ten decades at scale 1: each decade a distinct rate, and the smallest still flows
  const loads=[1e-3,1e-2,.5,4.5,20,90],rates=loads.map(x=>rate(x,1));
  for(let i=1;i<rates.length;i++)assert.ok(rates[i]>1.05*rates[i-1]);
  // each tenfold step about doubles the rate; the meal's load on the battery (about 12) beats the wheels' drain (0.7)
  assert.ok(Math.abs(rate(1,1)/rate(.1,1)-2)<.05);assert.ok(rate(12,1)>2*rate(.7,1));
  assert.ok(Math.abs(rate(P.top,1)-P.fastest/P.spacing)<1e-9);assert.equal(rate(1e6,1),rate(P.top,1));
  assert.ok(Math.abs(rate(1e-9,1)-P.slowest/P.spacing)<1e-12);
  // at the page's top playback, 8 units a second, a packet moves under half the spacing in a 60 Hz frame
  assert.ok(P.fastest*8/60<P.spacing/2);
  assert.equal(rate(-.5,1),rate(.5,1));assert.equal(rate([3,4],10),rate(.5,1));
  assert.ok(Math.abs(rate(1,2)-rate(.5,1))<1e-12);
  assert.equal(E.packetSpeed(0,1,'value'),P.steady);assert.equal(E.packetSpeed(7,1,'value'),P.steady);
  assert.equal(E.packetSpeed(null,1,'value'),null);
});
test('a load\'s sign is its packets\' direction of travel; a value\'s go forward',()=>{
  const V=(x,k='load')=>E.packetVelocity(x,1,k);
  // positive from source to target: the meal's charge travels back to the battery, the wheels' drain away from it
  assert.ok(V(12)<0);assert.ok(V(-.7)>0);
  for(const x of [1e-3,.7,12,500])assert.equal(V(-x),-V(x));
  for(const x of [1e-3,.7,12])assert.equal(Math.abs(V(x)),E.packetSpeed(x,1,'load'));
  assert.equal(V(0),null);assert.equal(V(null),null);
  // a hose's load has no one sign: back, at the speed of its size
  assert.equal(V([3,-4]),-E.packetSpeed([3,-4],1,'load'));
  assert.ok(V(-5,'value')>0);assert.ok(V(5,'value')>0);
});
test('a field that reads a wire and never pushes it sends no load on it, at any state',()=>{
  // the memory's battery payment, −c_mem u² db: it reads the meal rate u and pushes the battery b
  const pay={form:{letters:{'Meal rate':'u','Battery':'b'},terms:[{c:'-c_mem*u**2',h:'b'}]}};
  assert.equal(E.pushes(pay,'Meal rate'),false);assert.equal(E.pushes(pay,'Battery'),true);
  // through an abbreviation an h uses
  const live={form:{letters:{'Battery':'b','Look':'r'},where:[['live','θ(b/b_0)']],terms:[{c:'r',h:'live'}]}};
  assert.equal(E.pushes(live,'Battery'),true);assert.equal(E.pushes(live,'Look'),false);
  // no written form, or no letter: may push
  assert.equal(E.pushes({},'Battery'),true);assert.equal(E.pushes(pay,'Grip'),true);
  // and on the feeding run read from the composite, every wire the marks are found not to push carries zero load
  const adapter=C.compositeAdapter(data,{maxSteps:1e9});const read=new Set();
  for(let n=0;n<1800;n+=150){
    const m=new E.Model(adapter.snapshot());
    for(const id of m.nodes.keys())if(m.children(id).length){m.enter(id);
      for(const w of m.project().wires){const node=m.nodes.get(w.target);
        if(node?.form&&!E.pushes(node,w.label)){read.add(w.label+' into '+w.target);assert.equal(E.magnitude(w.load)??0,0,w.label+' into '+w.target);}}}
    adapter.step(150*data.h);
  }
  assert.ok(read.size>0);
  console.log('wires read and never pushed:',read.size);
});
test('a wire into a box carries a load back exactly when some mark it enters inside can push it',()=>{
  // A box of two marks: `pay` reads the rate u and pushes the battery b; `drive` reads u and pushes the store q.
  const mark=(id,pushed,letter)=>({id,parent:'m',kind:'field',form:{letters:{'Rate':'u',[pushed]:letter},terms:[{c:'u',h:letter}]}});
  const sc={version:1,root:'r',nodes:[{id:'r',kind:'box'},{id:'m',parent:'r',kind:'box',glyph:'motor'},
    mark('m/drive','Store','q'),mark('m/pay','Battery','b'),{id:'plain',parent:'m',kind:'field'},
    ...['rate','store','battery','other'].map(id=>({id,parent:'r',kind:'boundary',side:'in'}))],
    wires:[['rate','m/drive','Rate'],['rate','m/pay','Rate'],['store','m/drive','Store'],['battery','m/pay','Battery'],
      ['other','plain','Other']].map(([source,target,label])=>({id:source+'>'+target,source,target,label,value:1,load:0}))};
  const m=new E.Model(sc),wires=m.project().wires,by=label=>wires.filter(w=>w.label===label);
  for(const w of wires){assert.equal(w.target,'m');assert.equal(w.deepTarget,sc.wires.find(x=>x.id===w.id).target);}
  // one wire at a time, then the two branches of the rate joined: read by both marks, pushed by neither
  for(const w of by('Rate'))assert.equal(E.carriesLoad(m.nodes,w),false);
  assert.equal(E.carriesLoad(m.nodes,E.aggregate(by('Rate'))[0]),false);
  assert.equal(E.carriesLoad(m.nodes,by('Store')[0]),true);assert.equal(E.carriesLoad(m.nodes,by('Battery')[0]),true);
  // a mark with no written form may push; and a wire joined with one that is pushed carries a load
  assert.equal(E.carriesLoad(m.nodes,by('Other')[0]),true);
  assert.equal(E.carriesLoad(m.nodes,E.forkWire([...by('Rate'),...by('Store')])),true);
  // inside the box the same wires are judged by the same marks
  m.enter('m');for(const w of m.project().wires)assert.equal(E.carriesLoad(m.nodes,w),w.label!=='Rate',w.id);
});
test('a value\'s shade is its place in its wire\'s range, light low and dark high',()=>{
  const S=E.packets.shade,shade=(v,lo,hi)=>E.valueShade((v-lo)/(hi-lo));
  assert.equal(E.valueShade(0),S.light);assert.equal(E.valueShade(1),S.dark);
  assert.equal(E.valueShade(null),E.valueShade(.5));assert.equal(E.valueShade(7),S.dark);
  // the same in mV or V, and with zero moved: v -> a v + b, a > 0
  for(const v of [-65,-20,30])assert.equal(shade(v,-70,40),shade(v/1000+.07,-.07+.07,.04+.07));
  // darker for higher: the sum of the channels falls
  const sum=c=>[1,3,5].reduce((s,i)=>s+parseInt(c.slice(i,i+2),16),0);
  assert.ok(sum(E.valueShade(.2))>sum(E.valueShade(.8)));
});
test('reading and regrouping never changes the simulation trajectory',()=>{
  const a=C.compositeAdapter(data),b=C.compositeAdapter(data);const m=new E.Model(a.snapshot());
  for(let i=0;i<30;i++){a.step(.02);b.step(.02);m.update(a.snapshot());m.enter('body-and-motors');if(i===0)m.group(['seeing','wheels'],'Test');m.project();}
  close(Array.from(a.state),Array.from(b.state),1e-13);
});
test('each feeding map publishes one distinct readout',()=>{
  const sc=scene();
  const outputKeys=id=>new Set(sc.wires.filter(w=>w.source===id).map(E.signalKey));
  for(const node of sc.nodes.filter(n=>n.kind==='operation'))assert.ok(outputKeys(node.id).size<=1,node.id);
});
test('a shared coordinate shows a forward dependency if any branch reads it',()=>{
  const load={source:'s',target:'f',label:'Height',value:2,load:-3,argument:false};
  const read={...load,load:0,argument:true};
  for(const branches of [[load],[load,read],[read,load]]){
    assert.equal(E.forkWire(branches).argument,branches.includes(read));
    assert.equal(E.aggregate(branches)[0].argument,branches.includes(read));
  }
});
