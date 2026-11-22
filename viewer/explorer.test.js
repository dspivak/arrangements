const test=require('node:test'),assert=require('node:assert/strict');
const E=require('./explorer-core.js'),A=require('./feeding-arrangement.js'),F=require('./feeding-model.js');
const scene=()=>A.evaluate(F.START,F.PLACES,F.DEFAULTS);
test('adding an apple beside the held apple changes only the surroundings',()=>{
  const adapter=A.adapter({places:[0],foods:[10],parameters:{kexp:0,ktau:0}});
  adapter.step(1);
  const before=adapter.snapshot();
  assert.ok(before.casings[0].eat>0);
  adapter.addCasing(0);
  const after=adapter.snapshot();
  const internals=scene=>{
    const descendants=new Set(['receiver']);
    for(const node of scene.nodes)if(descendants.has(node.parent))descendants.add(node.id);
    return {
      nodes:scene.nodes.filter(n=>descendants.has(n.id)).map(n=>[n.id,n.parent,n.kind,n.label]),
      wires:scene.wires.filter(w=>descendants.has(w.source)&&descendants.has(w.target)).map(w=>w.id)
    };
  };
  assert.deepEqual(internals(after),internals(before));
  assert.equal(after.time,before.time);
  assert.deepEqual(after.state.slice(0,before.state.length),before.state);
  let switched=false;
  for(let tick=0;tick<70;tick++){
    adapter.step(.1);
    const [first,second]=adapter.snapshot().casings;
    if(first.selected>0){assert.equal(second.selected,0);assert.equal(second.pry,0);assert.equal(second.eat,0);}
    if(second.eat>0){assert.equal(first.selected,0);switched=true;}
  }
  assert.ok(switched);
  const tie=A.adapter({places:[0,0],foods:[10,10],parameters:{ktau:0,kexp:0}});
  tie.step(.4);
  assert.deepEqual(tie.snapshot().casings.map(c=>c.selected),[0,0]);
});
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
test('feeding field agrees with independent existing model at all sampled states',()=>{
  let seed=125,rand=()=>{seed=(1664525*seed+1013904223)>>>0;return seed/2**32;};
  for(let n=0;n<80;n++){
    const places=n%3===0?[]:n%3===1?[0]:[4,-3,1];
    const s=F.start(places,places.map(()=>10*rand()),0,F.DEFAULTS);
    s[0]=rand()*5;s[1]=rand()*3-1;s[2]=rand()*.8-.5;s[3]=rand()*10-5;s[4]=rand()-.5;s[5]=s[3]+rand()*.2-.1;s[6]=rand()*12-6;s[7]=rand()*12-6;
    s[8]=rand()*16-8;s[9]=rand()*16-8;
    for(let k=0;k<places.length;k++){const i=F.NR+F.NC*k;s[i+1]=rand();s[i+2]=rand()*.7-.5;s[i+3]=rand()-.5;s[i+4]=s[i+2]+rand()*.1;s[i+5]=rand()*.03;s[i+6]=rand()*.03;}
    for(const csee of [.002,0]){const P={...F.DEFAULTS,csee};const a=A.evaluate(s,places,P),b=F.field(s,places,P);E.validate(a);close(a.rate,b.rate);close([a.energy,a.loss],[b.energy,b.loss]);}
  }
});
test('every state inclination is the sum of displayed contributions',()=>{
  const sc=scene();for(const n of sc.nodes.filter(n=>n.kind==='state')){
    const sum=sc.wires.filter(w=>w.source===n.id&&w.inclination).reduce((s,w)=>s+w.load,0);
    assert.ok(Math.abs(sum-n.inclination)<1e-12,n.id);
  }
  assert.ok(sc.wires.some(w=>w.load!==0));assert.ok(sc.wires.some(w=>w.load===0));
});
test('navigation exposes receiver, then wheels, while outside parts become boundary ports',()=>{
  const m=new E.Model(scene());assert(m.enter('receiver'));
  const p=m.project();assert(!p.nodes.some(n=>n.id==='casing-0'));assert(p.nodes.some(n=>n.kind==='boundary'));
  assert(m.enter('wheels'));assert(!m.project().nodes.some(n=>n.id==='receiver'));assert(m.project().nodes.some(n=>n.id==='cart'&&n.kind==='state'));
});
test('group, nested group, ungroup, undo and saved layouts preserve every crossing observation',()=>{
  const sc=scene(),original=JSON.stringify(sc),m=new E.Model(sc);m.enter('receiver');
  const id=m.group(['eye','wheels'],'Guidance');assert(m.project().nodes.some(n=>n.id===id));
  assert.equal(JSON.stringify(sc),original);m.enter(id);assert.equal(m.children().length,2);
  for(const w of m.project().wires){const real=sc.wires.find(x=>x.id===w.id);assert.equal(w.value,real.value);assert.equal(w.load,real.load);}
  const saved=m.save(),other=new E.Model(sc,saved);assert.equal(other.scope,id);assert.equal(other.children().length,2);
  m.enter('receiver');const outer=m.group([id,'lever'],'Movement');m.enter(outer);assert(m.ungroup(id));
  assert.equal(m.children().length,3);assert(m.undo());assert(m.nodes.has(id));
  m.resetGroups();assert.equal(m.groups.length,0);assert(m.undo());assert.equal(m.groups.length,2);
});
test('invalid hierarchy and saved layouts are rejected, unknown loads remain unknown',()=>{
  const sc=scene(),m=new E.Model(sc);assert.throws(()=>m.group(['eye','casing-0']));
  assert.equal(m.restore({version:1,root:'feeding',groups:[{id:'bad',parent:'feeding',members:['bad','receiver']}]}),false);
  const bad=JSON.parse(JSON.stringify(sc));bad.nodes[1].parent='receiver';assert.throws(()=>E.validate(bad));
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
  // and on the feeding run, every wire the marks are found not to push carries zero load
  const L=require('./food-learning.js');let s=L.start();const read=new Set();
  for(let n=0;n<1800;n++){
    if(n%150===0){const m=new E.Model(A.observeLearning(s,L.PLACES.slice(),L.DEFAULTS,n*.004));
      for(const id of m.nodes.keys())if(m.children(id).length){m.enter(id);
        for(const w of m.project().wires){const node=m.nodes.get(w.target);
          if(node?.form&&!E.pushes(node,w.label)){read.add(w.label+' into '+w.target);assert.equal(E.magnitude(w.load)??0,0,w.label+' into '+w.target);}}}}
    s=L.rk4(s,.004,L.PLACES.slice(),L.DEFAULTS);
  }
  assert.ok(read.has('Meal rate into pay-mem')||[...read].some(x=>x.startsWith('Meal rate')),[...read].join('; '));
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
test('every feeding motor has the one mark, and is load-free exactly at the wires it only hears',()=>{
  const L=require('./food-learning.js'),places=[0,5];
  const heard={gripper:['Chosen grip rate','Other grip load'],lift:['Chosen lift rate','Other tip load'],
    drive:['Drive torque','Other wheel load'],'look-motor':['Look rate'],'scan-motor':['Scan rate'],
    'patrol-motor':['Patrol point rate'],'patrol-scan-motor':['Patrol scan rate']};
  for(const {label,id} of L.stores)heard[id+'-motor']=[label+' rate'];
  places.forEach((_,k)=>{heard['attachment-'+k+'/latch-drive']=['Latch rate '+(k+1),'Other latch load'];});
  const scenes=[[A.observeLearning(L.start(places,[10,8]),places,L.DEFAULTS),5+L.stores.length+places.length],
    [A.adapter({places,foods:[10,8]}).snapshot(),7+places.length]];
  for(const [sc,count] of scenes){
    const m=new E.Model(sc),motors=sc.nodes.filter(n=>n.glyph==='motor');
    assert.equal(motors.length,count);
    // every part named a motor is one, and every one opens into its two pushes
    for(const n of sc.nodes)if(n.kind==='box'&&/ motor$/.test(n.label))assert.equal(n.glyph,'motor',n.id);
    for(const n of motors){
      assert.deepEqual(m.children(n.id).map(c=>c.kind),['field','field'],n.id);
      m.enter(n.parent);
      const entering=m.project().wires.filter(w=>w.target===n.id),labels=[...new Set(entering.map(w=>w.label))];
      const free=labels.filter(label=>!E.carriesLoad(m.nodes,E.forkWire(entering.filter(w=>w.label===label))));
      assert.deepEqual(free.sort(),heard[n.id].slice().sort(),n.id);
      // what is left is the store it pushes and the battery it charges
      assert.equal(labels.length-free.length,2,n.id);assert.ok(labels.includes('Battery'),n.id);
      // and the node names them: the wire that powers it and the wire it turns, the two it pushes
      assert.equal(n.powered,'Battery',n.id);assert.ok(labels.includes(n.turns)&&!free.includes(n.turns)&&n.turns!==n.powered,n.id);
    }
  }
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
  const a=A.adapter(),b=A.adapter();const m=new E.Model(a.snapshot());
  for(let i=0;i<30;i++){a.step(.02);b.step(.02);m.update(a.snapshot());m.enter('receiver');if(i===0)m.group(['eye','wheels'],'Test');m.project();}
  close(a.snapshot().state,b.snapshot().state,1e-13);
});
test('factored browser episode reaches the reference food and cart after 60 time units',()=>{
  let s=F.START.slice(),reference=s.slice();for(let n=0;n<30000;n++){s=A.rk4(s,.002,F.PLACES,F.DEFAULTS);reference=F.rk4(reference,.002,F.PLACES,F.DEFAULTS);}
  close(s,reference,1e-8);
});

test('the learning observer traces the live meal, destination, motors and casing states',()=>{
  const L=require('./food-learning.js'),p=L.DEFAULTS,places=L.PLACES.slice();
  let s=L.start(), observedMeal=false;
  for(let n=0;n<=2000;n++){
    if(n%250===0){
      const sc=A.observeLearning(s,places,p,n*.004), f=L.field(s,places,p);
      E.validate(sc);
      assert.deepEqual(sc.state,s);
      close(sc.rate,f.rate);
      close([sc.energy,sc.loss,sc.torque],[f.energy,f.loss,f.torque]);
      const nodes=new Map(sc.nodes.map(node=>[node.id,node]));
      assert.equal(nodes.has('patrol'),false);
      assert.equal(nodes.get('destination').value,f.target);
      for(const {id,start,end} of L.stores){
        const load=f.rate.slice(start,end).map(v=>p.memoryCost*v);
        const value=nodes.get(id).value;
        close(Array.isArray(value)?value:[value],s.slice(start,end));
        assert.deepEqual(nodes.get(id).inclination,load.length===1?load[0]:load);
      }
      const opening=sc.wires.find(w=>w.target==='opener/field'&&w.label==='Opening rate');
      const eating=sc.wires.find(w=>w.target==='digestor/field'&&w.label==='Food taken');
      close([opening.value,eating.value],[f.pryAll,f.eatAll]);
      for(let k=0;k<places.length;k++){
        const {food,opening,height,selected,pry,eat}=sc.casings[k];
        close([food,opening,height],s.slice(L.NR+k*L.NC,L.NR+k*L.NC+3));
        close([selected,pry,eat],[f.casings[k].selected,f.casings[k].pry,f.casings[k].eat]);
      }
      if(f.eatAll>0)observedMeal=true;
    }
    if(n===1750){ // Introduce a second casing beside the one being eaten.
      places.push(places[0]);s.push(...L.casing(10,p));
    }
    s=L.rk4(s,.004,places,p);
  }
  assert.ok(observedMeal);
  assert.equal(s.at(-L.NC),10);
});

test('each feeding map publishes one distinct readout with its own dependencies',()=>{
  const L=require('./food-learning.js'),p=L.DEFAULTS,places=[0,0,5];
  const state=L.start(places,[10,8,.05]),sc=A.observeLearning(state,places,p);
  const nodes=new Map(sc.nodes.map(n=>[n.id,n]));
  const inputs=id=>sc.wires.filter(w=>w.target===id);
  const outputKeys=id=>new Set(sc.wires.filter(w=>w.source===id).map(E.signalKey));
  for(const node of sc.nodes.filter(n=>n.kind==='operation'))assert.ok(outputKeys(node.id).size<=1,node.id);
  const args=id=>inputs(id).map(w=>w.source).sort();
  assert.deepEqual(args('sum-eaten'),places.map((_,i)=>'attachment-'+i+'/eating'));
  assert.deepEqual(args('attachment-0/opening'),['attachment-0/at-mouth','casing-0/opening','in-opener']);
  assert.deepEqual(args('attachment-0/eating'),['attachment-0/at-mouth','casing-0/access','casing-0/food','in-digestor']);
  const leaves=id=>{
    if(id.startsWith('in-') || nodes.get(id).kind==='state' || !inputs(id).length)return [id];
    return [...new Set(args(id).flatMap(leaves))].sort();
  };
  const perCasing=keys=>places.flatMap((_,i)=>keys.map(key=>'casing-'+i+'/'+key));
  assert.deepEqual(leaves('attachment-0/latch-rate'),[...perCasing(['height','food','latch','place']),
    'in-grip','in-tip','in-cart'].sort());
  assert.deepEqual(leaves('sum-keep'),[...perCasing(['food','height','latch','place']),'in-tip','in-cart'].sort());
  assert.deepEqual(leaves('sum-opened'),[...perCasing(['opening','height','latch','place']),'in-opener','in-cart'].sort());
  assert.deepEqual(leaves('sum-eaten'),[...perCasing(['food','opening','height','latch','place']),'in-digestor','in-cart'].sort());
  // No rate map reads the battery: every motor reads it itself (David, 2026-11-22, 779795d2).
  assert.deepEqual(args('meal-rate'),['grip','meal','sum-eaten']);
  assert.deepEqual(args('eye-contact-rate'),['eye-contact','memory-gaze']);
  assert.deepEqual(args('clock-rate'),['cart','clock','destination','sum-red']);
  // A motor's box has no outputs (David, 2026-11-22, 779795d2, "B"): the yield's rule computes the grip's rate
  // from what that rate reads, and nothing outside the grip's box reads the grip rate.
  assert.deepEqual(args('yield-rate'),['cart','grip','meal','sum-keep','yield']);
  assert.deepEqual(args('grip-rate'),['grip','sum-keep']);
  const within=(id,box)=>{for(let n=nodes.get(id);n;n=nodes.get(n.parent))if(n.id===box)return true;return false;};
  for(const n of sc.nodes.filter(n=>n.glyph==='motor')){
    const box=n.parent;assert.equal(nodes.get(box).kind,'box',n.id);
    assert.deepEqual(sc.wires.filter(w=>within(w.source,box)&&!within(w.target,box)),[],box+' has no outputs');
  }
});

test('varying the digestion effort during contact changes eating alone among the six readouts',()=>{
  const L=require('./food-learning.js'),places=[0,0,4],p=L.DEFAULTS,s=L.start(places,[10,7,3]);
  s[1]=2;s[2]=p.x0;s[L.NR+1]=.7;s[L.NR+2]=p.x0;s[L.NR+7]=1;
  const six=params=>{
    const sc=A.observeLearning(s,places,params),value=id=>sc.wires.find(w=>w.source===id).value;
    return ['attachment-0/latch-rate','attachment-1/latch-rate','attachment-2/latch-rate',
      'sum-keep','sum-opened','sum-eaten'].map(value);
  };
  const before=six(p),after=six({...p,leat:p.leat/2});
  assert.ok(before[4]>0&&before[5]>0);
  close(before.slice(0,5),after.slice(0,5));assert.equal(after[5],before[5]/2);
});

test('the fingers, battery and spring each show the dependencies of their own load',()=>{
  const sc=A.adapter().snapshot();
  const inputs=id=>sc.wires.filter(w=>w.target===id&&w.argument!==false).map(w=>w.label).sort();
  assert.deepEqual(inputs('gripper/field'),['Battery','Chosen grip rate','Other grip load']);
  assert.deepEqual(inputs('gripper/field-1'),['Battery','Chosen grip rate','Grip torque']);
  assert.deepEqual(inputs('casing-0/gravity/field'),[]);
  assert.deepEqual(inputs('casing-0/spring'),['Air memory 1','Height 1']);
  for(const n of sc.nodes.filter(n=>n.kind==='field')){
    assert.equal(n.dependencyView,true,n.id);
    assert.ok(n.result!==undefined,n.id);
    assert.ok(new Set(sc.wires.filter(w=>w.source===n.id).map(E.signalKey)).size<=1,n.id);
  }
});

test('a shared coordinate shows a forward dependency if any branch reads it',()=>{
  const load={source:'s',target:'f',label:'Height',value:2,load:-3,argument:false};
  const read={...load,load:0,argument:true};
  for(const branches of [[load],[load,read],[read,load]]){
    assert.equal(E.forkWire(branches).argument,branches.includes(read));
    assert.equal(E.aggregate(branches)[0].argument,branches.includes(read));
  }
});
