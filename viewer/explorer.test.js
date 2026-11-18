const test=require('node:test'),assert=require('node:assert/strict');
const E=require('./explorer-core.js'),A=require('./feeding-arrangement.js'),F=require('./feeding-model.js');
const scene=()=>A.evaluate(F.START,F.PLACES,F.DEFAULTS);
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
  assert.equal(E.intensity(null),null);assert.notEqual(E.color(null),E.color(0));
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
