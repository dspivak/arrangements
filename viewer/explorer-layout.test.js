const test=require('node:test'),assert=require('node:assert/strict');
const E=require('./explorer-core.js'),G=require('./explorer-layout.js'),A=require('./feeding-arrangement.js');
const close=(a,b)=>assert.ok(Math.hypot(a.x-b.x,a.y-b.y)<1e-6,JSON.stringify([a,b]));
function check(project,options){
  const g=G.layout(project,options);
  for(const w of g.edges){
    const source=g.positions.get(w.source),target=g.positions.get(w.target);
    const out=source.outputs.find(p=>p.key===w.outKey),input=target.inputs.find(p=>p.key===w.inKey);
    close(w.segments[0].a,out.fork||out);close(w.segments.at(-1).b,input);
    assert.ok(out.x>=source.x&&input.x<=target.x,'outputs right, inputs left');
    const first=w.segments[0],last=w.segments.at(-1);
    assert.equal((first.c||first.b).y,first.a.y);assert.ok((first.c||first.b).x>first.a.x,'leaves rightward');
    assert.equal((last.d||last.a).y,last.b.y);assert.ok((last.d||last.a).x<last.b.x,'enters rightward');
    for(const s of w.segments)for(let i=0;i<=60;i++){
      const p=G.sample(s,i/60);
      assert.ok(p.x>=0&&p.x<=g.width&&p.y>=0&&p.y<=g.height,'path stays in viewport');
      for(const n of project.nodes.filter(n=>n.kind!=='boundary'&&n.id!==w.source&&n.id!==w.target)){
        const box=g.positions.get(n.id),withTitle={...box,y:box.y-12,h:box.h+24};
        assert.ok(!G.inside(p,withTitle,-1),`${w.source} -> ${w.target} crosses ${n.id} or its title`);
      }
    }
  }
  for(const p of g.positions.values())for(const ports of [p.inputs,p.outputs]){
    assert.equal(new Set(ports.map(p=>p.key)).size,ports.length,'distinct port identities');
    for(let i=1;i<ports.length;i++)assert.ok(Math.abs(ports[i].y-ports[i-1].y)>=19,'ports do not overlap');
  }
  for(const f of g.forks){close(f.start,f.port);assert.ok(f.end.x>f.start.x);assert.equal(f.end.y,f.start.y);}
  return g;
}
test('each casing has six output ports; receiver has eighteen inputs and two outputs',()=>{
  const m=new E.Model(A.adapter().snapshot()),g=check(m.project());
  assert.equal(g.positions.get('receiver').inputs.length,18);assert.equal(g.positions.get('receiver').outputs.length,2);
  for(let i=0;i<3;i++)assert.equal(g.positions.get('casing-'+i).outputs.length,6);
  assert.equal(g.edges.length,20);
});
test('every feeding level has connected horizontal endpoints and clears unrelated parts',()=>{
  const m=new E.Model(A.adapter().snapshot());let levels=0;
  for(const n of m.scene.nodes)if(m.children(n.id).length){m.enter(n.id);check(m.project());levels++;}
  assert.ok(levels>40);
});
test('nested grouping preserves distinct crossing ports and routes them',()=>{
  const m=new E.Model(A.adapter().snapshot());m.enter('receiver');const group=m.group(['eye','wheels'],'Guidance');
  check(m.project());m.enter(group);check(m.project());m.enter('receiver');const outer=m.group([group,'lever'],'Movement');
  check(m.project());m.enter(outer);check(m.project());m.ungroup();check(m.project());m.undo();check(m.project());
});
test('equal values with equal labels from different readouts stay distinct after projection',()=>{
  const scene={version:1,root:'r',nodes:[{id:'r',kind:'box'},
    {id:'a',parent:'r',kind:'box'},{id:'b',parent:'r',kind:'box'},
    {id:'x',parent:'a',kind:'state'},{id:'y',parent:'a',kind:'state'},{id:'z',parent:'b',kind:'field'}],
    wires:['x','y'].map((source,i)=>({id:source,source,target:'z',label:'Position',value:0,load:i+1}))};
  const m=new E.Model(scene),p=m.project(),g=check(p);assert.equal(g.edges.length,2);
  assert.equal(g.positions.get('a').outputs.length,2);assert.equal(g.positions.get('b').inputs.length,2);
  const keys=g.positions.get('a').outputs.map(p=>p.key);scene.wires[0].value=4;m.update(scene);
  assert.deepEqual(G.layout(m.project()).positions.get('a').outputs.map(p=>p.key),keys);
});
test('a shared output has one fork and a separate returning load on each branch',()=>{
  const m=new E.Model(A.adapter().snapshot());m.enter('wheels');const g=check(m.project()),port=g.positions.get('cart').outputs[0];
  assert.ok(port.wires.length>1);assert.ok(port.fork);
  for(const w of port.wires)close(w.start,port.fork);
  assert.ok(port.wires.some(w=>w.load===0));assert.ok(port.wires.every(w=>Number.isFinite(w.load)));
});
test('inconsistent values on one output and distinct readouts joined at one input are rejected',()=>{
  const scene=A.adapter().snapshot(),wire=scene.wires[0],same=scene.wires.find(w=>w!==wire&&E.signalKey(w)===E.signalKey(wire));
  assert.ok(same);same.value=wire.value+1;assert.throws(()=>E.validate(scene),/Conflicting values/);
  const sc=A.adapter().snapshot(),target=sc.wires[0].target,ws=sc.wires.filter(w=>w.target===target);
  ws[0].targetPort='input';ws[1].targetPort='input';assert.throws(()=>E.validate(sc),/Distinct readouts/);
});
test('moving a part reroutes every crossing wire, preserves port order, and leaves observations unchanged',()=>{
  const m=new E.Model(A.adapter().snapshot()),p=m.project(),before=JSON.stringify(p),initial=G.layout(p);
  const original=initial.positions.get('casing-0').outputs.map(p=>p.key);
  for(const placements of [{'casing-0':{x:335,y:187}},{'casing-0':{x:1090,y:1000},receiver:{x:800,y:421}}]){
    const moved=check(p,{placements});
    assert.deepEqual(moved.positions.get('casing-0').outputs.map(p=>p.key),original);
    assert.notEqual(moved.edges[0].path,initial.edges[0].path);
    assert.equal(JSON.stringify(p),before);
  }
  assert.throws(()=>G.layout(p,{placements:{receiver:{x:290,y:421}}}),/Leave space/);
});
test('tiny pinch events produce gentle continuous zoom independent of event grouping',()=>{
  let small=1;for(let i=0;i<10;i++)small=G.wheelZoom(small,-2);
  assert.ok(small>1.05&&small<1.07,'a short pinch changes size by about six percent at the requested doubled sensitivity');
  assert.ok(Math.abs(small-G.wheelZoom(1,-20))<1e-12,'same gesture distance, same size');
  for(let i=0;i<10;i++)small=G.wheelZoom(small,2);
  assert.ok(Math.abs(small-1)<1e-12,'opposite pinch restores size away from limits');
  assert.equal(G.wheelZoom(1,1e6),.25);assert.equal(G.wheelZoom(1,-1e6),4);
  assert.equal(G.wheelZoom(1,1,1),G.wheelZoom(1,16,0));
});
