/* Versioned, renderer-independent observations. No simulation or grouping math here. */
(function(root) {
  'use strict';
  const copy = value => JSON.parse(JSON.stringify(value));
  const numbers = x => Array.isArray(x) ? x.flatMap(numbers) : typeof x === 'number' ? [x] : [];
  const magnitude = x => x == null || !numbers(x).length ? null : Math.hypot(...numbers(x));
  // Signal identity survives projection and changes of value. Two equal numbers
  // from different readouts must never accidentally become the same port.
  const signalKey = w => w.signal ?? JSON.stringify([w.source,w.sourcePort ?? w.label]);
  // One port per number. A wire that carries a list of n numbers is n wires bundled, a hose: `dimension` is
  // its n (1 for a number, and for a value that is not a list of numbers), `coordinate` the name of its i-th
  // wire (the wire's own `coordinates`, else its number in the list), and `strand` that wire alone, with its
  // own value and its own returning load. Wrapping and unwrapping change the drawing only: the hose's value
  // is the list of its strands' values and its load the list of their loads, the product R^n of n copies of
  // R and its projections, with nothing added or dropped.
  const list=x=>Array.isArray(x)&&x.length>1&&x.every(v=>typeof v==='number');
  const dimension = w => list(w.value)?w.value.length:w.value==null&&list(w.load)?w.load.length:1;
  const coordinate = (w,i) => w.coordinates?.[i] ?? String(i+1);
  const strand = (w,i) => ({...w,label:w.label+' '+coordinate(w,i),strandOf:w,index:i,
    value:Array.isArray(w.value)?w.value[i]:w.value,load:Array.isArray(w.load)?w.load[i]:w.load});
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
        // `deepTarget` keeps the wire's actual endpoint, the mark it enters inside the part drawn (`carriesLoad`).
        wires.push({...w,signal:signalKey(w),source,target,deepTarget:w.target});
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
  // A track's packets. Each track has one color, blue for values and amber for loads. Packets are always `spacing`
  // pixels apart; a load's size x, against the wire's scale s, sets their speed, so the number passing a point per
  // unit of time is speed/spacing (David, 2026-11-20, 66ce7547: "keep spacing the same and increase speed rather than
  // vice versa, at least on loads"). Over a feeding run the nonzero loads span ten decades (median 0.01 s, 99% under
  // 3.1 s; a contact spring held off its rest, as in a test state, near 700 s), so each tenfold step in x multiplies the speed by one factor: `slowest` at x ≤ `bottom`·s,
  // `fastest` at x ≥ `top`·s, in pixels per unit of model time. The factor is about 2 (David, 2026-11-20, 66ce7547,
  // yes to doubling, "i'd like the meal's battery laod ot stand out"): the meal pushes the battery with a load near
  // 12 while the wheels drain it with about 0.7, and at 1.5 per decade the two moved 66 and 40. `fastest` is kept low enough that a packet moves well
  // under half the spacing in a frame at the page's top playback speed, so motion never reads backward. Speed is seen
  // only in motion: paused, a load's size is read on selection. A load lives in a cotangent space, whose zero is
  // canonical, so no packets means no load. A value's size depends on where a chart puts zero, so by default its
  // packets move at one speed and show only direction; `valuePackets: 'size'` codes values like loads.
  const packets={spacing:60,dash:4,slowest:7,fastest:220,bottom:1e-3,top:1e2,steady:20,valuePackets:'steady',
    color:{value:'#4c91b7',load:'#c97f2f'},ink:{value:'#084675',load:'#963816'},shade:{light:'#bfe1f6',dark:'#062b5c'}};
  function packetSpeed(value,scale=1,kind='load') {
    if(kind==='value'&&packets.valuePackets==='steady')return magnitude(value)==null?null:packets.steady;
    const x=magnitude(value);if(!x)return null;
    const u=Math.min(1,Math.max(0,Math.log(x/(scale*packets.bottom))/Math.log(packets.top/packets.bottom)));
    return packets.slowest*Math.pow(packets.fastest/packets.slowest,u);
  }
  // The packets' velocity along the wire, positive from source to target. A value's packets go forward. A load on a
  // wire carrying one number shows its sign by the direction of travel (David, 2026-11-21, 66ce7547: "it should be
  // about direction of travel, i.e. velocity"): a positive load, one that pushes the source's value up, travels back
  // to the source, as the meal's load feeds the battery; a negative one travels away from it, as the wheels' drain
  // does. So the map from a load to its velocity is odd, v(-x) = -v(x), where the speed alone saw only |x|. A hose's
  // load, a list of numbers, has no one sign, so its packets travel back, the load track's direction, and show only
  // the size of the list; each strand of an unwrapped hose is one number and shows its own sign.
  function packetVelocity(value,scale=1,kind='load') {
    const speed=packetSpeed(value,scale,kind);if(!speed)return null;
    if(kind==='value')return speed;
    return typeof value==='number'&&value<0?speed:-speed;
  }
  // Whether the field mark `node` can send a load back on the wire `label`. Its form is Σ c dh (explorer-forms.js),
  // so the load on a wire is the form's component along that wire's coordinate, and that is zero at every state when
  // the wire's letter occurs in no h, nor in an abbreviation an h uses: the field reads the wire (in some c) and
  // never pushes it. Such a wire's amber track is not drawn, as a wire whose value means nothing has no blue one
  // (David, 2026-11-21, 66ce7547: "we completely remove blue wires when they're meaningless; maybe the same for
  // amber?"). A node without a written form, or a wire without a letter, may push it.
  function pushes(node,label) {
    const form=node?.form,letter=form?.letters?.[label];if(!form||!letter||!Array.isArray(form.terms))return true;
    const names=text=>new Set(String(text).match(/[\p{L}_][\p{L}\p{N}_]*/gu)||[]);
    const where=new Map((form.where||[]).map(([n,e])=>[n,e]));
    const seen=new Set(),todo=form.terms.flatMap(t=>[...names(t.h)]);
    while(todo.length){const n=todo.pop();if(seen.has(n))continue;seen.add(n);if(where.has(n))todo.push(...names(where.get(n)));}
    return seen.has(letter);
  }
  // Whether a wire as drawn at some level can carry a load back: whether any of the marks it actually enters, however
  // deep inside the part drawn, can push it. `w` is a wire of `project()`, or several of them joined by `aggregate`
  // or `forkWire` (its `branches`); `nodes` is the model's map of nodes. Where it cannot, the part is load-free at
  // that input port, and the view draws no amber track there and the book's mark for it (Ch 3: "We draw a black
  // wall at each input port where an arrangement is load-free"; Ch 4 def.walled). So a motor is load-free at the
  // rate it is sent, since each of its two pushes only reads it.
  function carriesLoad(nodes,w) {
    return (w.branches||[w]).some(b=>pushes(nodes.get(b.deepTarget??b.target),b.label));
  }
  // A value's beads, all one shade at a time, light where the value is lowest of what its wire has carried and
  // dark where it is highest (David, 2026-11-21, 66ce7547: "different colored beads, light blue vs dark vlue for
  // different values", and "yes all beads share one shade"). The place u = (v − low)/(high − low) in the wire's
  // range is unchanged by a change of units or of zero, v ↦ av + b with a > 0, which is what a value, unlike a
  // load, allows: its size depends on where a chart puts zero, its order and its proportions of difference do not.
  // A wire that has carried one value so far, or a value that is not a number (a hose's list), takes the middle.
  function valueShade(u) {
    const t=u==null||!Number.isFinite(u)?.5:Math.min(1,Math.max(0,u)),{light,dark}=packets.shade;
    const hex=c=>[1,3,5].map(i=>parseInt(c.slice(i,i+2),16));
    const a=hex(light),b=hex(dark);
    return '#'+a.map((x,i)=>Math.round(x+(b[i]-x)*t).toString(16).padStart(2,'0')).join('');
  }
  function format(value) {
    if(value==null)return 'unavailable';
    if(Array.isArray(value))return '['+value.map(format).join(', ')+']';
    if(typeof value==='object')return '{'+Object.entries(value).map(([k,v])=>k+': '+format(v)).join(', ')+'}';
    if(typeof value==='number')return value===0?'0.000':Math.abs(value)>=1e4||Math.abs(value)<.001?value.toExponential(2):value.toFixed(3);
    return String(value);
  }
  const api={Model,validate,magnitude,intensity,packets,packetSpeed,packetVelocity,valueShade,pushes,carriesLoad,format,aggregate,forkWire,signalKey,dimension,coordinate,strand};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ArrangementExplorer=api;
})(typeof globalThis!=='undefined'?globalThis:this);
