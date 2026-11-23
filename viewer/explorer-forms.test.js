// node --test viewer/explorer-forms.test.js — a field mark's 1-form: read, written, and checked against its wires.
// The checks of the feeding marks themselves ran on the hand-typed graph feeding-arrangement.js until 2026-11-23,
// when it was retired with its models (subagent of c5f9ecc9, Claude Opus 5.5): every formula through a meal, the
// motors at a rate, and the contact's springs. On the composite they are feeding-explorer-composite.test.js (every
// formula drawn gives the library's loads, before and after steps; every motor's two pushes and its walls) and
// tests/test_explore_composite.py (springs with a third terminal, loads at the stores against the composite's field).
const test=require('node:test'),assert=require('node:assert/strict');
const Forms=require('./explorer-forms.js');
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

test('a static map sends its wire\'s load back through its law\'s derivative',()=>{
  // The product of two wires, with the load 3 on the wire leaving it: 3 b comes back along a and 3 a along b.
  const law={letters:{A:'a',B:'b'},where:[['c','a*b']]},wires=(la,lb)=>[{label:'A',value:2,load:la},{label:'B',value:5,load:lb}];
  assert.equal(Forms.pullbackCheck(law,wires(15,6),3).ok,true);
  assert.equal(Forms.pullbackCheck(law,wires(15,5),3).ok,false);
  // The unit: no load on the wire leaving it, none sent back; and a load sent back from nothing is caught.
  assert.equal(Forms.pullbackCheck(law,wires(0,0),0).ok,true);
  assert.equal(Forms.pullbackCheck(law,wires(1,0),0).ok,false);
  // Two maps in a row send back what their composite does: c = a b, then e = c², at a = 2, b = 5, load 1 on e.
  const square={letters:{C:'c'},where:[['e','c**2']]},whole={letters:{A:'a',B:'b'},where:[['e','(a*b)**2']]};
  assert.equal(Forms.pullbackCheck(square,[{label:'C',value:10,load:20}],1).ok,true);
  assert.equal(Forms.pullbackCheck(law,wires(100,40),20).ok,true);
  assert.equal(Forms.pullbackCheck(whole,wires(100,40),1).ok,true);
  // A load on a wire the law has no letter for is a mismatch, not ignored.
  assert.match(Forms.pullbackCheck({letters:{A:'a'},where:[['c','a']]},wires(3,1),3).error,/no letter/);
});
