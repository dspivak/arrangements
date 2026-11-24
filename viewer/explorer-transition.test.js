// A Petri transition's mark (EXPLORER.md, *A Petri transition is drawn as a machine*): its arcs and their weights
// are the ones read off its written form, the layout gives each wire the row of its role, and a half has its open
// side. The first tests are on a scene typed here, so that they do not move with the feeding model; the last is on
// the feeding scene read from the library composite. (Book P109 (n), run c34e1b43, Claude Fable 5.2, 2026-11-23.)
const test=require('node:test'),assert=require('node:assert/strict');
const E=require('./explorer-core.js');require('./explorer-layout.js');
const Forms=require('./explorer-forms.js');
globalThis.ArrangementExplorer=E;globalThis.ArrangementForms=Forms;
require('./explorer.js');
const store=(id,label)=>({id,parent:'root',label,kind:'state',value:1});
const wire=(id,source,label)=>({id,source,target:'t',label,value:1,load:0,valueScale:1,loadScale:1});
// rate J, d(0.8 chamber + 0.2 husks - 0.03 battery - food): it takes from the battery and the food, gives to the
// chamber and the husks, and only hears its rate.
const WHOLE={letters:{Battery:'e',Chamber:'z',Husks:'m',Food:'f',Rate:'J'},constants:{p:.8,c:.03},words:{},terms:[{c:'J',h:'p*z + (1 - p)*m - c*e - f'}]};
// (the mark is wired to the stores its form has letters for)
const scene=(extra={},form=WHOLE)=>({version:1,title:'A transition',root:'root',time:0,
  nodes:[{id:'root',parent:null,label:'A transition',kind:'box'},store('battery','Battery'),store('chamber','Chamber'),store('husks','Husks'),
    store('food','Food'),store('rate','Rate'),
    {id:'t',parent:'root',label:'Machine',kind:'field',glyph:'transition',rubric:'Petri transition',...extra,form}],
  wires:[wire('w1','chamber','Chamber'),wire('w2','rate','Rate'),wire('w3','battery','Battery'),wire('w4','husks','Husks'),wire('w5','food','Food')]
    .filter(w=>w.label in form.letters)});
const laidOut=s=>{
  const project=new E.Model(s).project();
  for(const n of project.nodes)if(n.glyph==='transition')n.arcs=E.transitionArcs(n.form);
  return E.layout(project).positions.get('t');
};
const near=(a,b)=>assert.ok(Math.abs(a-b)<1e-9,`${a} vs ${b}`);
test('a transition\'s arcs and weights are those read off its form',()=>{
  const arcs=E.transitionArcs(scene().nodes.at(-1).form);
  assert.deepEqual(Object.keys(arcs).sort(),['Battery','Chamber','Food','Husks']);   // the rate's wire has no arc
  near(arcs.Battery,-.03);near(arcs.Food,-1);near(arcs.Chamber,.8);near(arcs.Husks,.2);
  const p=laidOut(scene());
  assert.equal(p.glyph,'transition');
  const rows=role=>p.inputs.filter(port=>port.arc===role).map(port=>port.label).sort();
  assert.deepEqual(rows('take'),['Battery','Food']);
  assert.deepEqual(rows('heard'),['Rate']);
  assert.deepEqual(rows('give'),['Chamber','Husks']);
  for(const port of p.inputs)near(port.weight??0,arcs[port.label]??0);
  // the wires stand in the picture's order: takes above the machine, the heard wire at its side, gives below
  const S=p.transition,y=port=>port.y-p.y;
  for(const port of p.inputs){
    if(port.arc==='take')assert.ok(y(port)<S.top);
    if(port.arc==='heard')assert.ok(y(port)>S.top&&y(port)<S.bottom);
    if(port.arc==='give')assert.ok(y(port)>S.bottom);
  }
  assert.deepEqual(p.inputs.map(y),p.inputs.map(y).slice().sort((a,b)=>a-b));
  assert.equal(S.above,2);assert.equal(S.below,2);assert.equal(S.breakY,undefined);
});
test('a half has one more slot on its open side, a break there, and the words that say where its other half is',()=>{
  // the half that settles: its takes are in the other half
  const settles=laidOut(scene({open:'takes',openWords:'from the casing, in the spatial coupling'},
    {letters:{Battery:'e',Chamber:'z',Husks:'m',Rate:'J'},constants:{p:.8,c:.03},words:{},terms:[{c:'J',h:'p*z + (1 - p)*m - c*e'}]}));
  assert.equal(settles.transition.above,2);          // the battery's arc and the cut one
  assert.equal(settles.transition.below,2);
  assert.ok(settles.transition.breakY<settles.transition.top);
  assert.deepEqual(settles.openLines,['from the casing,','in the spatial coupling']);
  // the half that reaches: F d(-x), it takes from x, and its gives are in the other half
  const reaches=laidOut(scene({open:'gives',openWords:'to the mouth'},{letters:{Rate:'F',Food:'x'},constants:{},words:{},terms:[{c:'F',h:'-x'}]}));
  assert.deepEqual(reaches.inputs.filter(port=>port.arc).map(port=>[port.label,port.arc]).sort(),[['Food','take'],['Rate','heard']]);
  assert.equal(reaches.transition.below,1);          // the cut arc alone
  assert.ok(reaches.transition.breakY>reaches.transition.bottom);
  assert.deepEqual(reaches.openLines,['to the mouth']);
  // a whole transition has no break
  assert.equal(laidOut(scene()).openLines,undefined);
});
test('a mark whose form is not a rate times d(a fixed signed sum) is drawn as a plain field',()=>{
  assert.equal(E.transitionArcs({letters:{Food:'x',Rate:'F'},constants:{},terms:[{c:'F',h:'x**2'}]}),null);
  assert.equal(E.transitionArcs(undefined),null);
  const p=laidOut(scene({},{letters:{Food:'x',Rate:'F'},constants:{},words:{},terms:[{c:'F',h:'x**2'}]}));
  assert.equal(p.glyph,undefined);assert.equal(p.transition,undefined);
});
test('the feeding scene: every Petri transition is a machine, spilling and absorbing whole, taking in two halves',()=>{
  const data=require('./feeding-explorer-data.js'),nodes=data.scene.nodes,fields=nodes.filter(n=>n.kind==='field');
  for(const n of fields.filter(n=>n.rubric==='Petri transition'))assert.equal(n.glyph,'transition',n.id);
  const marks=fields.filter(n=>n.glyph==='transition'),named=word=>marks.filter(n=>n.label.toLowerCase().includes(word));
  const weights=n=>Object.values(E.transitionArcs(n.form)).sort((a,b)=>a-b);
  assert.equal(named('spilling').length,3);
  for(const n of named('spilling')){assert.equal(n.open,undefined);assert.deepEqual(weights(n),[-1,1]);}
  assert.equal(named('taking').length,3);
  // each half is a plain machine with its own arrows; its rate's wire is what joins it to the other half
  for(const n of named('taking')){assert.equal(n.open,undefined);assert.deepEqual(weights(n),[-1]);}
  const [settle]=named('settle');
  assert.equal(settle.open,undefined);
  assert.equal(weights(settle).length,3);
  const [gut]=named('gut wall'),[out]=named('husk collector: export');
  assert.equal(weights(gut).length,2);assert.ok(weights(gut)[0]<0&&weights(gut)[1]>0);
  assert.deepEqual(weights(out),[-1,1]);
});
