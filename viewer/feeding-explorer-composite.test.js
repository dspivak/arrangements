/* The feeding explorer read from the library composite (book P109 (f), act (4)).
 * `node --test viewer/feeding-explorer-composite.test.js` from the library's folder. The data modules are
 * written by scripts/export_feeding_explorer.py; their readings agree with the library to 1e-12
 * (tests/test_explore_composite.py checks the diagram against the composite's own field). */
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('fs'),path=require('path');
const E=require('./explorer-core.js'),G=require('./explorer-layout.js'),Forms=require('./explorer-forms.js');
global.ArrangementForms=Forms;
const C=require('./feeding-explorer-composite.js');
Forms.functions.θ=C.theta;Forms.functionWords.θ=C.thetaWords;
const modules=['feeding-explorer-data.js','feeding-explorer-robot-data.js'].filter(f=>fs.existsSync(path.join(__dirname,f)));

// Every drawn formula, evaluated at the values on its wires, gives the loads the library returned there.
function checkForms(scene){
  let checked=0;
  for(const node of scene.nodes.filter(n=>n.form)){
    const entering=scene.wires.filter(w=>w.target===node.id).map(w=>({label:w.label,value:w.value,load:w.load}));
    const verdict=Forms.check(node.form,entering);
    assert.ok(verdict.ok,`${node.id}: ${verdict.error||JSON.stringify(verdict.rows)}`);
    checked++;
  }
  return checked;
}

for(const file of modules){
  const data=require('./'+file);
  test(file+': the scene is valid and every formula drawn holds, before and after steps',()=>{
    const adapter=C.compositeAdapter(data);
    let scene=adapter.snapshot();
    E.validate(scene);
    const before=checkForms(scene);
    assert.ok(before>40,`${before} formulas`);
    adapter.step(data.h*25);
    scene=adapter.snapshot();
    E.validate(scene);
    assert.equal(checkForms(scene),before);
    assert.ok(adapter.time>0);
  });
  test(file+': nothing on the page is typed by hand; every mark has its rubric kind and every motor its two pushes',()=>{
    const scene=C.compositeAdapter(data).snapshot();
    for(const n of scene.nodes.filter(n=>n.kind==='field'))assert.ok(n.rubric,n.id);
    for(const m of scene.nodes.filter(n=>n.glyph==='motor')){
      const pushes=scene.nodes.filter(n=>n.parent===m.id);
      assert.equal(pushes.length,2,m.id);
      assert.ok(m.turns&&m.powered,m.id);
    }
    // A motor's black walls: the battery and its store carry loads back; what it is told carries none.
    const model=new E.Model(scene);
    for(const m of scene.nodes.filter(n=>n.glyph==='motor')){
      model.scope=m.parent;const project=model.project();
      const entering=project.wires.filter(w=>w.target===m.id),labels=new Set(entering.map(w=>w.label));
      for(const label of labels){
        const loaded=E.carriesLoad(model.nodes,{branches:entering.filter(w=>w.label===label)});
        assert.equal(loaded,label===m.turns||label===m.powered,`${m.id}: ${label}`);
      }
    }
  });
  test(file+': everything in a drawn formula is explained, by an entering wire or in words',()=>{
    for(const n of data.scene.nodes){
      const form=n.form||n.law;if(!form)continue;
      assert.deepEqual(Forms.unexplained(form),[],n.id);
      // The letters are on the terminals: each is on the one kind of wire that enters the mark.
      const entering=new Set(data.scene.wires.filter(w=>w.target===n.id).map(w=>w.label));
      for(const label of Object.keys(form.letters))assert.ok(entering.has(label),`${n.id}: no wire ${label}`);
      if(n.form){
        const line=Forms.plain(Forms.written(form));
        assert.ok(line.length<=44,`${n.id}: its formula is cut on the canvas: ${line}`);
        // A list is as long as the words under the formula say: its whole hose enters the mark.
        for(const [letter,length] of Object.entries(form.lists||{})){
          const label=Object.keys(form.letters).find(l=>form.letters[l]===letter);
          assert.deepEqual(data.scene.wires.map((w,i)=>[w,data.branches[i].width]).filter(([w])=>w.target===n.id&&w.label===label).map(([,k])=>k),[length],`${n.id}: ${letter}`);
        }
        // A sum sign stands only where there is a list to add over, and then its index is explained.
        assert.equal(line.includes('Σ'),!!form.lists,`${n.id}: ${line}`);
      }
    }
  });
  test(file+': every level lays out',()=>{
    const model=new E.Model(C.compositeAdapter(data).snapshot());
    let levels=0;
    for(const n of model.scene.nodes)if(model.children(n.id).length){
      model.enter(n.id);const g=G.layout(model.project());levels++;
      for(const p of g.positions.values())for(const ports of [p.inputs,p.outputs])
        assert.equal(new Set(ports.map(p=>p.key)).size,ports.length,'distinct port identities at '+n.id);
    }
    assert.ok(levels>20,`${levels} levels`);
  });
}
test('the data modules exist',()=>assert.ok(modules.length>=1));
