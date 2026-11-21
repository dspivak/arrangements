const test=require('node:test'),assert=require('node:assert/strict');
const E=require('./explorer-core.js'),G=require('./explorer-layout.js'),A=require('./feeding-arrangement.js');
const close=(a,b)=>assert.ok(Math.hypot(a.x-b.x,a.y-b.y)<1e-6,JSON.stringify([[a.x,a.y],[b.x,b.y]]));
test('the two directions separate along the route and meet at each port and fork',()=>{
  const model=new E.Model(A.adapter().snapshot());
  model.enter('receiver');
  const {edges,forks}=G.layout(model.project());
  assert.ok(edges.length&&forks.length);
  for(const wire of [...edges,...forks]){
    assert.notEqual(wire.tracks.value,wire.tracks.load);
    for(const track of Object.values(wire.tracks)){
      const points=track.match(/-?[\d.]+/g).map(Number);
      for(const [actual,expected] of [[points.slice(0,2),wire.start],[points.slice(-2),wire.end]]){
        assert.ok(Math.hypot(actual[0]-expected.x,actual[1]-expected.y)<.001);
      }
    }
  }
});
test('two wires meet only where they cross, at a point, never along a stretch',()=>{
  // David, 2026-11-21: the Mouth's incoming wires were "blocked by other wires … crossings should raise codimension".
  // A wire runs on top of another where it lies within 10 px of it, nearly parallel, away from its two ends; before
  // the separating pass that was 61% of the receiver's wire length and 29% of the mouth's.
  const L=require('./food-learning.js');
  const m=new E.Model(A.observeLearning(L.start(),L.PLACES.slice(),L.DEFAULTS,0));
  for(const scope of ['receiver','mouth','eye','lever']){
    m.enter(scope);const {edges}=G.layout(m.project());
    const samples=edges.map(w=>w.segments.flatMap(s=>Array.from({length:40},(_,i)=>{
      const a=G.sample(s,i/40),b=G.sample(s,(i+.5)/40),d=Math.hypot(b.x-a.x,b.y-a.y)||1;
      return {x:a.x,y:a.y,tx:(b.x-a.x)/d,ty:(b.y-a.y)/d,len:2*d};})));
    let total=0,over=0;
    samples.forEach((pts,i)=>{const w=edges[i];for(const q of pts){total+=q.len;
      if(Math.hypot(q.x-w.start.x,q.y-w.start.y)<=20||Math.hypot(q.x-w.end.x,q.y-w.end.y)<=20)continue;
      if(samples.some((o,j)=>j!==i&&o.some(r=>Math.hypot(r.x-q.x,r.y-q.y)<10&&Math.abs(r.tx*q.tx+r.ty*q.ty)>.9)))over+=q.len;}});
    assert.ok(over/total<.04,scope+': '+(100*over/total).toFixed(1)+'% of wire length runs on another wire');
    // and a wire moved off another never doubles back on itself: a forward wire never runs leftward
    for(const w of edges)if(w.end.x>w.start.x){const pts=w.segments.flatMap(s=>Array.from({length:12},(_,i)=>G.sample(s,i/12)));
      let back=0;for(let i=0;i+1<pts.length;i++)back+=Math.max(0,pts[i].x-pts[i+1].x-.5);
      assert.ok(back<4,scope+': '+w.label+' runs back '+back.toFixed(0)+' px');}
  }
});
function check(project,options){
  const g=G.layout(project,options);
  for(const w of g.edges){
    const source=g.positions.get(w.source),target=g.positions.get(w.target);
    // A wire ends at a port, or, where a hose is unwrapped, at the hose's mouth (G.endAt).
    const out=G.endAt(source,'outputs',w.outKey),input=G.endAt(target,'inputs',w.inKey);
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
test('the receiver keeps its seven named inputs and outputs as casings enter the world',()=>{
  let reference;
  for(const places of [[],[0],[0,1],[0,1,3,5]]){
    const model=new E.Model(A.adapter({places,foods:places.map(()=>10)}).snapshot());
    const layout=check(model.project()),body=layout.positions.get('receiver');
    const ports=[body.inputs,body.outputs].map(side=>side.map(port=>port.key));
    assert.deepEqual(ports.map(side=>side.length),[7,7]);
    if(reference)assert.deepEqual(ports,reference);
    reference=ports;
    for(let i=0;i<places.length;i++)assert.equal(layout.positions.get('casing-'+i).outputs.length,6);
  }
});
test('the food-learning receiver has the same ports with zero, one and four casings',()=>{
  const L=require('./food-learning.js');
  let reference;
  for(const places of [[],[0],[0,1,3,5]]){
    const sc=A.observeLearning(L.start(places,places.map(()=>10)),places,L.DEFAULTS);
    const model=new E.Model(sc),body=check(model.project()).positions.get('receiver');
    const ports=[body.inputs,body.outputs].map(side=>side.map(port=>port.key));
    assert.deepEqual(ports.map(side=>side.length),[7,7]);
    if(reference)assert.deepEqual(ports,reference);
    reference=ports;
    model.enter('receiver');check(model.project());
    model.enter('memory');check(model.project());
  }
});
test('every feeding level has connected horizontal endpoints and clears unrelated parts',()=>{
  const m=new E.Model(A.adapter().snapshot());let levels=0;
  for(const n of m.scene.nodes)if(m.children(n.id).length){m.enter(n.id);check(m.project());levels++;}
  assert.ok(levels>=38);   // 38 since a spring is one mark and a dragger is drawn as its store and its spring, with no box
});
test('a motor is one compact mark: the wire it turns ends at its pulley, the battery\'s at its plug, the rest at its side',()=>{
  const L=require('./food-learning.js'),places=[0,5];
  const m=new E.Model(A.observeLearning(L.start(places,[10,8]),places,L.DEFAULTS));
  const parents=new Set(m.scene.nodes.filter(n=>n.glyph==='motor').map(n=>n.parent));let seen=0;
  assert.deepEqual([...parents].sort(),['attachment-0','attachment-1','eye','lever','memory','wheels']);
  for(const parent of parents){
    m.enter(parent);const project=m.project(),g=check(project);
    for(const n of project.nodes.filter(n=>n.glyph==='motor')){
      const p=g.positions.get(n.id);seen++;
      assert.equal(p.radius,19);assert.equal(p.outputs.length,0,'a motor has input ports only');
      assert.ok(p.inputs.length>=3,n.id);
      // top to bottom: what it turns (one wire, or the strands of one hose), what it only hears, what powers it
      const M=E.geometry.motor,roles=p.inputs.map(port=>port.motor),name=port=>port.group?port.group.label:port.label;
      assert.deepEqual(roles,[...roles].sort((a,b)=>['turns','heard','powered'].indexOf(a)-['turns','heard','powered'].indexOf(b)),n.id);
      assert.ok(roles.includes('turns')&&roles.includes('heard'),n.id);assert.equal(roles.filter(r=>r==='powered').length,1,n.id);
      for(const port of p.inputs){
        assert.equal(port.motor,name(port)===n.turns?'turns':name(port)===n.powered?'powered':'heard',n.id);
        const at=[port.glyphX-p.glyphX,port.glyphY];
        if(port.motor==='turns')assert.deepEqual(at,[-M.hub,M.hubY]);
        else if(port.motor==='powered')assert.deepEqual(at,[M.plugX,M.cordY]);
        else{assert.equal(at[0],-M.half);assert.ok(at[1]>M.top&&at[1]<M.bottom,'a heard wire ends on the body\'s side');}
        assert.ok(port.x<p.x+port.glyphX-10,'the terminal stands left of the picture');
      }
      // the pulley above the body, the plug below it, and under them the count of its parts, inside the part's rectangle
      assert.ok(M.hubY<M.top&&M.cordY>M.bottom&&p.h/2>=M.cordY+27);
    }
  }
  assert.equal(seen,5+L.stores.length+places.length);
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
  const scene=A.adapter().snapshot(),wire=scene.wires.find(w=>w.source==='battery' && w.label==='Battery');
  const same=scene.wires.find(w=>w!==wire&&E.signalKey(w)===E.signalKey(wire));
  assert.ok(same);same.value=wire.value+1;assert.throws(()=>E.validate(scene),/Conflicting values/);
  const sc=A.adapter().snapshot(),ws=sc.wires.filter(w=>w.target==='grip-rate');
  ws[0].targetPort='input';ws[1].targetPort='input';assert.throws(()=>E.validate(sc),/Distinct readouts/);
});
test('moving a part reroutes every crossing wire, preserves port order, and leaves observations unchanged',()=>{
  const m=new E.Model(A.adapter().snapshot()),p=m.project(),before=JSON.stringify(p),initial=G.layout(p);
  const original=initial.positions.get('casing-0').outputs.map(p=>p.key);
  const start=initial.positions.get('casing-0'),body=initial.positions.get('receiver');
  for(const placements of [
    {'casing-0':{x:start.x+32,y:start.y}},
    {'casing-0':{x:body.x,y:initial.height+120},receiver:{x:body.x+140,y:body.y}}
  ]){
    const moved=check(p,{placements});
    assert.deepEqual(moved.positions.get('casing-0').outputs.map(p=>p.key),original);
    const edge=g=>g.edges.find(e=>e.source==='casing-0');
    assert.notEqual(edge(moved).path,edge(initial).path);
    assert.equal(JSON.stringify(p),before);
  }
  assert.throws(()=>G.layout(p,{placements:{receiver:{x:start.x,y:start.y}}}),/Leave space/);
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

test('an operation with many true inputs uses compact marks and individually spaced terminals',()=>{
  const sizes=[];
  for(const count of [1,6,26]){
    const sc={version:1,root:'r',nodes:[{id:'r',kind:'box'},{id:'sum',parent:'r',kind:'operation',label:'Sum'}],wires:[]};
    for(let i=0;i<count;i++){
      sc.nodes.push({id:'i'+i,parent:'r',kind:'boundary',side:'in',label:'Input '+i});
      sc.wires.push({id:'w'+i,source:'i'+i,target:'sum',label:'Input '+i,value:i,load:0});
    }
    const sum=check(new E.Model(sc).project()).positions.get('sum');
    for(const port of sum.inputs){
      assert.ok(Math.abs(Math.hypot(port.glyphX-sum.glyphX,port.glyphY)-sum.radius)<1e-8);
      assert.ok(port.x<sum.x+port.glyphX);
    }
    sizes.push(sum);
  }
  assert.equal(sizes[0].radius,sizes[2].radius);
  assert.ok(sizes[2].w < 150 && sizes[2].h > sizes[0].h);
});

// Hoses: a wire that carries a list of n numbers is n wires bundled, and wrapping is the product's
// isomorphism between n wires and one hose, on values and on returning loads.
// A store of n numbers read by a field, which loads it, and by a static map, which does not.
function hoseScene(n,coordinates){
  const value=Array.from({length:n},(_,i)=>i+1),load=value.map(v=>-v/10);
  return {version:1,root:'r',nodes:[{id:'r',kind:'box',label:'R'},{id:'s',parent:'r',kind:'state',label:'Store',value},
    {id:'f',parent:'r',kind:'field',label:'Field'},{id:'g',parent:'r',kind:'operation',label:'Map'}],
    wires:[{id:'s>f',source:'s',target:'f',label:'Position',value,load,...(coordinates?{coordinates}:{})},
      {id:'s>g',source:'s',target:'g',label:'Position',value,load:value.map(v=>v/100),...(coordinates?{coordinates}:{})}]};
}
test('a wire has one strand per number, and its strands give back its value and its load',()=>{
  assert.equal(E.dimension({value:3,load:1}),1);
  assert.equal(E.dimension({value:[1,2,3],load:null}),3);
  assert.equal(E.dimension({value:null,load:[0,0]}),2);
  for(const value of [[5],'open',null,[[1,2],[3,4]],{a:1}])assert.equal(E.dimension({value,load:0}),1);
  const scene=hoseScene(3,['x','y','z']),[loaded,read]=scene.wires;
  // Unwrapping then wrapping is the identity: the list of the strands' values is the hose's value, and so for loads.
  for(const w of scene.wires){
    const strands=w.value.map((_,i)=>E.strand(w,i));
    assert.deepEqual(strands.map(s=>s.value),w.value);assert.deepEqual(strands.map(s=>s.load),w.load);
    assert.deepEqual(strands.map(s=>s.label),['Position x','Position y','Position z']);
    assert.ok(strands.every(s=>E.dimension(s)===1));
  }
  assert.equal(E.strand({label:'Yield',value:[4,5],load:[0,0]},1).label,'Yield 2');
  // On a readout shared by two branches the loads add strand by strand: the trunk's strand is the sum of the
  // branches' strands, the same number either way round.
  const trunk=E.forkWire([loaded,read]);
  for(let i=0;i<3;i++)assert.ok(Math.abs(E.strand(trunk,i).load-(E.strand(loaded,i).load+E.strand(read,i).load))<1e-12);
});
test('a long list is one hose port at each end until an end is unwrapped into one port per number',()=>{
  const project=new E.Model(hoseScene(6)).project(),wrapped=check(project);
  const store=wrapped.positions.get('s'),field=wrapped.positions.get('f');
  assert.equal(store.outputs.length,1);assert.equal(store.outputs[0].hose,6);assert.equal(store.groups.length,0);
  assert.equal(field.inputs.length,1);assert.equal(field.inputs[0].hose,6);
  assert.equal(store.outputs[0].end,G.hoseEnd('s','outputs',store.outputs[0].key));
  // Unwrap the store's end: six ports, their strands meeting at the hose's mouth, where the readout forks.
  const open=check(project,{toggled:[store.outputs[0].end]}),s=open.positions.get('s'),[group]=s.groups;
  assert.equal(s.outputs.length,6);assert.ok(s.outputs.every((port,i)=>port.group===group&&port.index===i&&!port.hose));
  assert.deepEqual(s.outputs.map(port=>port.label),[1,2,3,4,5,6].map(i=>'Position '+i));
  assert.equal(group.dimension,6);assert.ok(group.x>s.outputs[0].x+30,'the mouth stands off the part');
  assert.ok(group.y>s.outputs[0].y&&group.y<s.outputs[5].y);
  for(const port of s.outputs){close(port.strand.a,port);close(port.strand.b,group);assert.notEqual(port.tracks.value,port.tracks.load);}
  assert.ok(group.fork&&group.wires.length===2);
  for(const w of open.edges)close(w.start,group.fork);
  // The other end of the same hose is still wrapped: wires and hose meet at the mouth, on the way.
  assert.equal(open.positions.get('f').inputs.length,1);assert.equal(open.positions.get('f').inputs[0].hose,6);
  assert.ok(open.width>wrapped.width,'the column makes room for the fan');
  // Unwrap the field's end as well: the hose now runs from mouth to mouth.
  const both=check(project,{toggled:[store.outputs[0].end,field.inputs[0].end]}),f=both.positions.get('f'),[entering]=f.groups;
  assert.equal(f.inputs.length,6);assert.ok(entering.x<f.inputs[0].x-30);
  for(const port of f.inputs){close(port.strand.a,entering);close(port.strand.b,port);}
  close(both.edges.find(w=>w.target==='f').end,entering);
  assert.equal(both.positions.get('g').inputs.length,1,'the map, not toggled, keeps its hose port');
});
test('a short list is one port per number until the viewer wraps it; the limit is an option',()=>{
  const project=new E.Model(hoseScene(3,['x','y','z'])).project(),open=check(project),s=open.positions.get('s');
  assert.deepEqual(s.outputs.map(port=>port.label),['Position x','Position y','Position z']);
  assert.equal(open.positions.get('f').inputs.length,3);
  const wrapped=check(project,{toggled:[s.groups[0].end]});
  assert.equal(wrapped.positions.get('s').outputs.length,1);assert.equal(wrapped.positions.get('s').outputs[0].hose,3);
  assert.equal(check(project,{hoseAbove:2}).positions.get('s').outputs.length,1);
  assert.equal(check(new E.Model(hoseScene(6)).project(),{hoseAbove:6}).positions.get('s').outputs.length,6);
});
test('a hose stays wrapped where it crosses a boundary, and every hose end of the food memory unwraps and routes',()=>{
  const L=require('./food-learning.js'),places=[0,1];
  const model=new E.Model(A.observeLearning(L.start(places,places.map(()=>10)),places,L.DEFAULTS));
  model.enter('yield-motor');
  const inside=check(model.project()),crossing=[...inside.positions.values()].filter(p=>p.kind==='boundary').flatMap(p=>[...p.inputs,...p.outputs]).filter(port=>port.hose);
  assert.ok(crossing.length>=2);assert.ok(crossing.every(port=>port.hose===19&&port.end===undefined));
  model.enter('memory');
  const project=model.project(),wrapped=check(project);let ends=0;
  for(const id of ['yield','yield-rate','yield-motor','return','return-rate']){
    const p=wrapped.positions.get(id);
    for(const port of [...p.inputs,...p.outputs].filter(port=>port.end)){
      const open=check(project,{toggled:[port.end]}),group=open.positions.get(id).groups.find(g=>g.end===port.end);
      assert.equal(group.ports.length,19);assert.equal(port.wires[0].standsFor,'ℝ^[−9, 9]','the hose says what its list stands in for');assert.equal(group.ports[0].label,port.label+' at −9');assert.equal(group.ports[18].label,port.label+' at 9');
      ends++;
    }
  }
  assert.ok(ends>=10);
});
