/* Versioned, renderer-independent observations. No simulation or grouping math here. */
(function(root) {
  'use strict';
  const copy = value => JSON.parse(JSON.stringify(value));
  const numbers = x => Array.isArray(x) ? x.flatMap(numbers) : typeof x === 'number' ? [x] : [];
  const magnitude = x => x == null || !numbers(x).length ? null : Math.hypot(...numbers(x));
  // Signal identity survives projection and changes of value. Two equal numbers
  // from different readouts must never accidentally become the same port.
  const signalKey = w => w.signal ?? JSON.stringify([w.source,w.sourcePort ?? w.label]);
  function validate(scene) {
    if (scene.version !== 1 || !Array.isArray(scene.nodes) || !Array.isArray(scene.wires)) throw Error('Expected an explorer version-1 scene');
    const nodes = new Map();
    for (const n of scene.nodes) {
      if (!n.id || nodes.has(n.id)) throw Error('Missing or duplicate node id: '+n.id);
      if (!['box','state','operation','field','boundary'].includes(n.kind)) throw Error('Unknown node kind: '+n.kind);
      nodes.set(n.id,n);
    }
    if (!nodes.has(scene.root) || nodes.get(scene.root).parent != null) throw Error('Invalid root');
    for (const n of scene.nodes) {
      let p=n; const seen=new Set();
      while (p.parent != null) {
        if (seen.has(p.id)) throw Error('Parent cycle'); seen.add(p.id);
        p=nodes.get(p.parent); if (!p) throw Error('Missing parent');
      }
      if (p.id!==scene.root) throw Error('Disconnected hierarchy');
    }
    const ids=new Set(),signals=new Map(),inputs=new Map();
    for (const w of scene.wires) {
      if (!w.id || ids.has(w.id) || !nodes.has(w.source) || !nodes.has(w.target)) throw Error('Invalid wire: '+w.id);
      ids.add(w.id);
      for (const key of ['value','load']) {
        if (w[key]!==null && w[key]!==undefined && numbers(w[key]).some(v=>!Number.isFinite(v))) throw Error('Nonfinite '+key+' at '+w.id);
      }
      for (const key of ['valueScale','loadScale']) if (w[key]!=null && !(w[key]>0 && Number.isFinite(w[key]))) throw Error('Invalid scale');
      const signal=signalKey(w),value=JSON.stringify(w.value??null);
      if(signals.has(signal)&&signals.get(signal)!==value)throw Error('Conflicting values for one output port: '+w.label);
      signals.set(signal,value);
      if(w.targetPort!=null){const key=JSON.stringify([w.target,w.targetPort]);
        if(inputs.has(key)&&inputs.get(key)!==signal)throw Error('Distinct readouts require distinct input ports: '+w.targetPort);
        inputs.set(key,signal);}
    }
    return scene;
  }
  class Model {
    constructor(scene, saved) {
      this.scope=scene.root; this.groups=[]; this.history=[]; this.next=1; this.update(scene);
      if (saved) this.restore(saved);
    }
    update(scene) {
      validate(scene); this.scene=scene;
      const available=new Set(scene.nodes.map(n=>n.id));
      // Topology edits remove invalid groups, including dependent nested groups.
      this.groups=this.groups.filter(g=>{
        if (!available.has(g.parent) || !g.members.every(id=>available.has(id))) return false;
        available.add(g.id); return true;
      });
      this.rebuild(); if(!this.nodes.has(this.scope)) this.scope=scene.root;
    }
    rebuild() {
      this.nodes=new Map(this.scene.nodes.map(n=>[n.id,{...n}]));
      for(const g of this.groups) {
        this.nodes.set(g.id,{id:g.id,parent:g.parent,label:g.label,kind:'box',userGroup:true});
        for(const id of g.members) this.nodes.get(id).parent=g.id;
      }
    }
    children(id=this.scope) {return [...this.nodes.values()].filter(n=>n.parent===id);}
    ancestors(id) {const out=[]; while(id!=null) {const n=this.nodes.get(id); if(!n)break; out.unshift(n);id=n.parent;}return out;}
    enter(id) {if(!this.children(id).length)return false;this.scope=id;return true;}
    group(ids,label='Group') {
      ids=[...new Set(ids)];
      if(ids.length<2 || ids.some(id=>!this.nodes.has(id)||this.nodes.get(id).parent!==this.scope||this.nodes.get(id).kind==='boundary')) throw Error('Select at least two parts at this level');
      this.history.push(this.save());
      let id; do {id='user-group-'+this.next++;}while(this.nodes.has(id));
      this.groups.push({id,parent:this.scope,label:label.trim()||'Group',members:ids});this.rebuild();return id;
    }
    ungroup(id=this.scope) {
      const g=this.groups.find(g=>g.id===id);if(!g)return false;
      this.history.push(this.save());
      // Move direct members and nested user groups out without flattening them.
      for(const other of this.groups) {
        if(other.parent===id)other.parent=g.parent;
        other.members=other.members.flatMap(x=>x===id?g.members:[x]);
      }
      this.groups=this.groups.filter(x=>x.id!==id);
      if(this.scope===id)this.scope=g.parent;
      this.rebuild();return true;
    }
    save(){return copy({version:1,root:this.scene.root,scope:this.scope,groups:this.groups,next:this.next});}
    restore(saved) {
      if(saved.version!==1||saved.root!==this.scene.root)return false;
      const previous=this.save();
      try {
        this.groups=[]; this.rebuild();
        for(const g of saved.groups||[]) {
          if(this.nodes.has(g.id)||!this.nodes.has(g.parent)||g.members.length<2||new Set(g.members).size!==g.members.length||g.members.some(id=>!this.nodes.has(id)||this.nodes.get(id).parent!==g.parent)) throw Error('Invalid saved group');
          this.groups.push(copy(g));this.rebuild();
        }
        this.scope=this.nodes.has(saved.scope)?saved.scope:this.scene.root;this.next=saved.next||1;return true;
      } catch {this.groups=previous.groups;this.scope=previous.scope;this.next=previous.next;this.rebuild();return false;}
    }
    undo(){const old=this.history.pop();return old?this.restore(old):false;}
    resetGroups(){this.history.push(this.save());this.groups=[];this.scope=this.scene.root;this.rebuild();}
    project() {
      const nodes=this.children(), wires=[], boundary=new Map();
      const representative=id=>{
        let n=this.nodes.get(id);
        while(n && n.parent!==this.scope) {if(n.id===this.scope)return null;n=this.nodes.get(n.parent);}
        return n?.id||null;
      };
      const port=(w,side)=>{
        const id='boundary:'+side+':'+signalKey(w);
        if(!boundary.has(id))boundary.set(id,{id,parent:this.scope,kind:'boundary',label:w.label,side,wire:w.id});
        return id;
      };
      for(const w of this.scene.wires) {
        let source=representative(w.source),target=representative(w.target);
        if((!source&&!target)||(source&&source===target))continue;
        // An explicitly supplied boundary already ends this view. The segment
        // on its other side belongs to the enclosing view, not a second port.
        if(source&&!target&&source===w.source&&this.nodes.get(source).kind==='boundary')continue;
        if(!source&&target&&target===w.target&&this.nodes.get(target).kind==='boundary')continue;
        if(!source)source=port(w,'in');if(!target)target=port(w,'out');
        wires.push({...w,signal:signalKey(w),source,target});
      }
      // Display bundles retain every individual wire and its own scale and load.
      const bundles=new Map();
      for(const w of wires) {
        const key=JSON.stringify([w.source,w.target]);
        if(!bundles.has(key))bundles.set(key,{id:key,source:w.source,target:w.target,wires:[]});
        bundles.get(key).wires.push(w);
      }
      return {nodes:nodes.concat([...boundary.values()]),wires,bundles:[...bundles.values()]};
    }
  }
  function sumLoads(branches) {
    const first=branches[0].load;
    if(branches.length===1)return first;
    function add(a,b){
      if(typeof a==='number'&&typeof b==='number')return a+b;
      if(!Array.isArray(a)||!Array.isArray(b)||a.length!==b.length)return null;
      const sum=a.map((v,i)=>add(v,b[i]));
      return sum.includes(null)?null:sum;
    }
    return branches.slice(1).reduce((total,branch)=>add(total,branch.load),first);
  }
  function forkWire(branches){
    return {...branches[0],branches,argument:branches.some(w=>w.argument!==false),load:sumLoads(branches),
      loadScale:Math.max(...branches.map(w=>w.loadScale??1))};
  }
  function aggregate(wires) {
    const groups=new Map();
    for(const w of wires){
      const key=JSON.stringify([w.source,w.target,signalKey(w),w.targetPort,w.valueScale,w.loadScale]);
      if(!groups.has(key))groups.set(key,{...w,branches:[]});
      groups.get(key).branches.push(w);
    }
    return [...groups.values()].map(w=>({...w,argument:w.branches.some(b=>b.argument!==false),load:sumLoads(w.branches)}));
  }
  function intensity(value,scale=1) {const v=magnitude(value);return v==null?null:Math.min(1,v/scale);}
  function color(value,scale=1,kind='value') {
    const u=intensity(value,scale);if(u===null)return '#89929f';
    const a=kind==='value'?[76,145,183]:[201,127,47], b=kind==='value'?[0,80,132]:[163,46,41];
    return `rgb(${a.map((x,i)=>Math.round(x+(b[i]-x)*u)).join(',')})`;
  }
  function format(value) {
    if(value==null)return 'unavailable';
    if(Array.isArray(value))return '['+value.map(format).join(', ')+']';
    if(typeof value==='object')return '{'+Object.entries(value).map(([k,v])=>k+': '+format(v)).join(', ')+'}';
    if(typeof value==='number')return value===0?'0.000':Math.abs(value)>=1e4||Math.abs(value)<.001?value.toExponential(2):value.toFixed(3);
    return String(value);
  }
  const api={Model,validate,magnitude,intensity,color,format,aggregate,forkWire,signalKey};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ArrangementExplorer=api;
})(typeof globalThis!=='undefined'?globalThis:this);
