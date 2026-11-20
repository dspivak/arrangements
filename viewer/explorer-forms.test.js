// node --test viewer/explorer-forms.test.js — a field mark's 1-form: read, written, and checked against its wires.
const test=require('node:test'),assert=require('node:assert/strict');
const Forms=require('./explorer-forms.js'),A=require('./feeding-arrangement.js'),L=require('./food-learning.js');
// A formula on one line, its thin spaces (between the factors of a product) made ordinary ones.
const line=form=>Forms.plain(Forms.written(form)).replace(/[   ]/g,' ');
const close=(a,b)=>assert.ok(Math.abs(a-b)<=1e-6*Math.max(1,Math.abs(b)),`${a} is not ${b}`);

test('formulas are read with Python\'s precedences',()=>{
  close(Forms.value(Forms.parse('-x**2'),{x:3}),-9);
  close(Forms.value(Forms.parse('2**-1'),{}),.5);
  close(Forms.value(Forms.parse('a - b - c'),{a:1,b:2,c:3}),-4);
  close(Forms.value(Forms.parse('a/b/c'),{a:8,b:2,c:2}),2);
  close(Forms.value(Forms.parse('κ*max(x, y_1)'),{κ:2,x:3,y_1:5}),10);
  assert.throws(()=>Forms.parse('x +'));assert.throws(()=>Forms.parse('x ^ 2'));assert.throws(()=>Forms.parse('(x'));
});

test('the zero form, a push, a cost and a Petri transition are written and give their loads',()=>{
  // The zero form: no terms, written 0, no load anywhere.
  const zero={letters:{In:'x'},terms:[]};
  assert.equal(line(zero),'0');assert.deepEqual(Forms.loads(zero,{x:2}),{x:0});
  // Push F dx: F comes back through x and nothing through F.
  const push={letters:{Force:'F',Position:'x'},terms:[{c:'F',h:'x'}]};
  assert.equal(line(push),'F dx');
  const p=Forms.loads(push,{F:3,x:7});close(p.x,3);close(p.F,0);
  // The spring, a cost of the difference: equal and opposite loads on its two ends, and none at stiffness 0.
  const spring={letters:{Left:'x',Right:'y'},constants:{κ:2},terms:[{c:'-1',h:'κ/2*(x - y)**2'}]};
  assert.equal(line(spring),'−d(κ/2 (x − y)²)');
  const s=Forms.loads(spring,{x:3,y:1});close(s.x,-4);close(s.y,4);
  const slack=Forms.loads({...spring,constants:{κ:0}},{x:3,y:1});close(slack.x,0);close(slack.y,0);
  // The spring with a third terminal: the stiffness wire is pushed back by minus half the squared stretch.
  const third=Forms.loads({terms:spring.terms},{κ:2,x:3,y:1});close(third.κ,-2);
  // A Petri transition A + B -> C at the rate k a b: minus the rate at A and at B, plus the rate at C.
  const petri={letters:{A:'a',B:'b',C:'c'},constants:{k:.5},where:[['rate','k*a*b']],terms:[{c:'rate',h:'c - a - b'}]};
  assert.equal(line(petri),'rate d(c − a − b)');
  const t=Forms.loads(petri,{a:2,b:3,c:1});close(t.a,-3);close(t.b,-3);close(t.c,3);
  // A motor at a rate u: two terms, each reading only u.
  assert.equal(line({terms:[{c:'C*u',h:'q'},{c:'-C*u**2',h:'b'}]}),'C u dq − C u² db');
});

test('a mistyped formula is caught against the loads on the wires',()=>{
  const wires=[{label:'Left',value:3,load:-4},{label:'Right',value:1,load:4}];
  const form=h=>({letters:{Left:'x',Right:'y'},constants:{κ:2},terms:[{c:'-1',h}]});
  assert.equal(Forms.check(form('κ/2*(x - y)**2'),wires).ok,true);
  assert.equal(Forms.check(form('κ*(x - y)**2'),wires).ok,false);           // a dropped half
  assert.equal(Forms.check(form('κ/2*(x + y)**2'),wires).ok,false);          // a wrong sign
  // A load on a wire the form has no letter for is a mismatch, not ignored.
  assert.match(Forms.check({letters:{Left:'x'},terms:[]},wires).error,/no letter/);
  // A small load is held to the same relative standard as a large one.
  assert.equal(Forms.check({letters:{In:'x'},terms:[{c:'0.002',h:'x'}]},[{label:'In',value:1,load:.0021}]).ok,false);
});

test('every formula on a feeding field mark gives the loads the model returns, through a meal',()=>{
  const p=L.DEFAULTS,places=L.PLACES.slice();
  // A form is an identity between two functions of the state, so it is checked off the outing too: at each
  // sampled moment, the state itself and four states near it (a fixed pseudo-random sequence). Along the
  // outing alone many loads are 0 (a latch at rest, a motor at rate 0), and a wrong formula passes at 0.
  let s=L.start(),seed=12345;const written=new Map(),missing=new Set(),laws=new Map();
  const random=()=>(seed=(seed*1664525+1013904223)%4294967296)/4294967296;
  const near=state=>state.map(v=>v+.3*(random()-.5)*(.2+Math.abs(v)));
  const finite=x=>Array.isArray(x)?x.every(finite):x===null||typeof x!=='number'||Number.isFinite(x);
  function examine(state,time,where){
    let sc;try{sc=A.observeLearning(state,places,p,time);}catch(error){return;}   // a state the model itself refuses
    if(!sc.wires.every(w=>finite(w.value)&&finite(w.load)))return;
    for(const node of sc.nodes)if(node.kind==='field'){
      const kind=node.id.replace(/\d+/g,'#');
      if(!node.form){missing.add(kind);continue;}
      const verdict=Forms.check(node.form,sc.wires.filter(w=>w.target===node.id).map(w=>({label:w.label,value:w.value,load:w.load})));
      assert.ok(verdict.ok,`${node.id} (${line(node.form)}) ${where}: ${verdict.error||JSON.stringify(verdict.rows.filter(r=>!r.ok))}`);
      written.set(kind,(written.get(kind)||0)+(verdict.rows.some(r=>[].concat(r.want).some(x=>Math.abs(x)>1e-7))?1:0));
    }
    // A static map's law gives the value on the wire leaving it.
    for(const node of sc.nodes)if(node.law){
      const leaving=sc.wires.find(w=>w.source===node.id);
      const verdict=Forms.lawCheck(node.law,sc.wires.filter(w=>w.target===node.id).map(w=>({label:w.label,value:w.value})),leaving.value);
      assert.ok(verdict.ok,`${node.id} ${where}: ${verdict.error||`its law gives ${verdict.got}, the model ${verdict.want}`}`);
      laws.set(node.id,(laws.get(node.id)||0)+(Math.abs(leaving.value)>1e-7?1:0));
    }
  }
  // The destination moves only while the search clock is choosing (its angle near a whole turn), which this
  // outing does not reach between samples; one more state at each moment puts the clock there.
  const clock=L.stores.find(store=>store.id==='clock').start;
  for(let n=0;n<=2000;n++){
    if(n%50===0){
      examine(s,n*.004,'at step '+n);
      for(let k=0;k<4;k++)examine(near(s),n*.004,`near step ${n}, state ${k}`);
      for(let k=0;k<2;k++){
        const choosing=near(s);choosing[clock]=2*Math.PI*Math.round(choosing[clock]/(2*Math.PI))+.5*(random()-.5);
        examine(choosing,n*.004,`near step ${n}, the clock choosing`);
      }
      // The return weights move only as the gaze arrives at a centre it has been away from and last saw empty.
      const range=id=>L.stores.find(store=>store.id===id);
      for(let k=0;k<3;k++){
        const arriving=near(s);
        for(const [id,v] of [['eye-contact',.02],['last-empty',.09],['time-away',.05]])for(let i=range(id).start;i<range(id).end;i++)arriving[i]=v;
        arriving[6]=12*(random()-.5);
        examine(arriving,n*.004,`near step ${n}, the gaze arriving`);
      }
      // The grip's own load from a contact is the slope of the hold gate, which is 0 unless the grip is part closed.
      const closing=near(s);closing[1]=.1+.8*random();
      examine(closing,n*.004,`near step ${n}, the grip part closed`);
    }
    s=L.rk4(s,.004,places,p);
  }
  for(const id of ['look-motor/field','look-motor/field-#','scan-motor/field','scan-motor/field-#'])assert.ok(written.has(id),id+' has a formula');
  for(const id of ['look-rate','scan-rate','drive-torque'])assert.ok(laws.get(id)>=10,id+' has a law, checked at ten values that are not 0 or more');
  console.log(`field marks with a checked formula: ${written.size}; without one yet: ${missing.size}`);
  for(const id of [...missing].sort())console.log('  no formula yet: '+id);
  // Each written form has to have met a load that is not 0 at ten states or more, or it is unproven.
  const thin=[...written].filter(([,count])=>count<10).map(([kind,count])=>`${kind} (${count})`);
  assert.deepEqual(thin,[],'forms checked at fewer than ten nonzero loads');
});

test('a motor at a rate hears its rate alone: each mark is wired to the rate and to the one wire it pushes',()=>{
  // The eye in the learning receiver, and the patrol receiver's four motors (which no page shows).
  const scenes=[A.observeLearning(L.start(),L.PLACES.slice(),L.DEFAULTS)];
  const patrol=A.adapter();patrol.step(.5);scenes.push(patrol.snapshot());
  let seen=0;
  for(const sc of scenes)for(const node of sc.nodes)if(node.kind==='field'&&/^(look|scan|patrol|patrol-scan)-motor\//.test(node.id)){
    const part=node.id.split('/')[0],rate=part.replace('-motor','-rate'),entering=sc.wires.filter(w=>w.target===node.id);
    const pushed=part==='look-motor'?'look':part==='scan-motor'?'scan':part==='patrol-motor'?'patrol-point':'patrol-scan';
    assert.deepEqual(entering.map(w=>w.source).sort(),[rate,node.label==='Battery payment'?'battery':pushed].sort(),node.id);
    const verdict=Forms.check(node.form,entering.map(w=>({label:w.label,value:w.value,load:w.load})));
    assert.ok(verdict.ok,`${node.id}: ${verdict.error||JSON.stringify(verdict.rows)}`);
    seen++;
  }
  assert.equal(seen,12);   // two marks for each of look and scan in both receivers, and for the two patrol motors
  // The drive hears its torque and, for the payment, the axle's load on the wheel; what the torque law reads
  // enters the torque's static map.
  for(const sc of scenes){
    const sources=id=>sc.wires.filter(w=>w.target===id).map(w=>w.source).sort();
    assert.deepEqual(sources('drive/field'),['drive-torque','wheel']);
    assert.deepEqual(sources('drive/field-1'),['battery','drive-torque','wheel-load']);
    // A spring is one mark with no box: its two wires, equal and opposite loads, and its cost as its form.
    const axle=sc.nodes.find(n=>n.id==='axle'),ends=sc.wires.filter(w=>w.target==='axle');
    assert.equal(axle.glyph,'spring');assert.deepEqual(ends.map(w=>w.source).sort(),['cart','wheel']);
    close(ends[0].load+ends[1].load,0);
    assert.ok(Forms.check(axle.form,ends.map(w=>({label:w.label,value:w.value,load:w.load}))).ok);
    assert.equal(line(axle.form),'−d(κ_w/2 (y − z_w)²)');
    for(const id of ['drive/field','drive/field-1']){
      const node=sc.nodes.find(n=>n.id===id);
      assert.ok(Forms.check(node.form,sc.wires.filter(w=>w.target===id).map(w=>({label:w.label,value:w.value,load:w.load}))).ok,id);
    }
  }
  // The patrol receiver's laws, like the learning one's, give the values on their wires.
  let laws=0;
  for(const node of scenes[1].nodes)if(node.law){
    const sc=scenes[1],leaving=sc.wires.find(w=>w.source===node.id);
    const verdict=Forms.lawCheck(node.law,sc.wires.filter(w=>w.target===node.id).map(w=>({label:w.label,value:w.value})),leaving.value);
    assert.ok(verdict.ok,`${node.id}: ${verdict.error||`its law gives ${verdict.got}, the model ${verdict.want}`}`);
    laws++;
  }
  assert.equal(laws,6);   // look, scan, patrol point, patrol scan, drive torque, the axle's load on the wheel
  // A law that is mistyped is caught: the look rate with its climb term dropped.
  const sc=scenes[0],look=sc.nodes.find(n=>n.id==='look-rate'),entering=sc.wires.filter(w=>w.target==='look-rate').map(w=>({label:w.label,value:w.value}));
  const wrong={...look.law,where:[...look.law.where.slice(0,-1),['look_r','live*free*(ν*ψ + γ*(A**2 - r**2 - ψ**2)*r)']]};
  const climbing=entering.map(r=>r.label==='Change with gaze'?{...r,value:.5}:r);
  const at=readings=>Forms.environment(look.law,Object.fromEntries(readings.map(r=>[look.law.letters[r.label],r.value]))).look_r;
  assert.equal(Forms.lawCheck(look.law,climbing,at(climbing)).ok,true);
  assert.equal(Forms.lawCheck(wrong,climbing,at(climbing)).ok,false);
});
