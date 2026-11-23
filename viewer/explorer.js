(function(root) {
  'use strict';
  const E=root.ArrangementExplorer, NS='http://www.w3.org/2000/svg';
  function html(tag,attrs={},parent,text) {
    const e=document.createElement(tag);
    for(const [k,v] of Object.entries(attrs)) e.setAttribute(k,v);
    if(text!=null)e.textContent=text;if(parent)parent.appendChild(e);return e;
  }
  function svg(tag,attrs={},parent,text) {
    const e=document.createElementNS(NS,tag);
    for(const [k,v] of Object.entries(attrs))e.setAttribute(k,v);
    if(text!=null)e.textContent=text;if(parent)parent.appendChild(e);return e;
  }
  // A field mark's 1-form (explorer-forms.js). A page that does not load that script draws as before.
  const Forms=root.ArrangementForms;
  // Formula runs as SVG tspans or as HTML: a one-letter name slanted, a subscript or superscript small and shifted.
  function writeSvg(parent,runs){
    let shift=0;
    for(const r of runs){
      const s=r.shift||0,span=svg('tspan',{},parent,r.text);
      if(s!==shift)span.setAttribute('dy',(shift-s)*4);
      if(s)span.setAttribute('font-size','72%');
      if(r.italic)span.setAttribute('font-style','italic');
      shift=s;
    }
    if(shift)svg('tspan',{dy:shift*4},parent,'​');
  }
  function writeHtml(parent,runs){
    for(const r of runs)html(r.shift===1?'sup':r.shift===-1?'sub':r.italic?'i':'span',{},parent,r.text);
  }
  class View {
    constructor(container,adapter,options={}) {
      this.adapter=adapter;this.options=options;this.renderers=options.renderers||{};
      this.running=false;this.style='illustrated';this.selection=new Set();this.selectMode=false;this.focusWire=null;this.focusNode=null;
      this.last=0;this.accumulator=0;this.speed=1;this.destroyed=false;this.cleanups=[];
      this.placements={};this.zoom='fit';this.drag=null;
      // The hose ends the viewer has wrapped or unwrapped, at each level (explorer-layout.js, `hoseEnd`).
      this.hoses={};
      const scene=adapter.snapshot();this.model=new E.Model(scene);
      this.storageKey=options.storageKey;
      if(this.storageKey)try{const s=JSON.parse(localStorage.getItem(this.storageKey));if(s){this.model.restore(s);if(s.placements&&typeof s.placements==='object')this.placements=s.placements;if(s.zoom==='fit'||Number(s.zoom)>=.25&&Number(s.zoom)<=4)this.zoom=s.zoom;
        if(s.hoses&&typeof s.hoses==='object')for(const [scope,ends] of Object.entries(s.hoses))if(Array.isArray(ends))this.hoses[scope]=ends.filter(end=>typeof end==='string');}}catch{}
      this.el=html('section',{class:'explorer','data-playing':'false'},container);
      const header=html('header',{class:'ex-header'},this.el),title=html('div',{},header);
      html('div',{class:'ex-kicker'},title,'Arrangement explorer');
      html('h1',{},title,scene.title);
      html('p',{class:'ex-subtitle'},title,'Follow values out and loads back. Open a box to see what is inside.');
      if(options.sceneURL)html('a',{href:options.sceneURL,class:'ex-link'},header,options.sceneLabel||'Original animation');
      const controls=html('div',{class:'ex-controls'},this.el);
      if (options.playback) {
        controls.append(options.playback.controls);
      } else {
        this.play=this.button(controls,'Play',()=>this.setRunning(!this.running),'ex-play');
        this.play.disabled=!adapter.step;
        this.button(controls,'One step',()=>{this.setRunning(false);this.advance(.02);this.refresh();}).disabled=!adapter.step;
        this.button(controls,'Start over',()=>{this.setRunning(false);adapter.reset?.();this.refresh();}).disabled=!adapter.reset;
        const speedLabel=html('label',{},controls,'Speed '),speed=html('select',{'aria-label':'Playback speed'},speedLabel);
        for(const x of [.5,1,2,4])html('option',{value:x},speed,x+'×');speed.value='1';speed.onchange=()=>{this.speed=+speed.value;};
      }
      this.time=html('output',{class:'ex-time'},controls);
      const styleLabel=html('label',{},controls,'View '),style=html('select',{'aria-label':'Drawing style'},styleLabel);
      html('option',{value:'illustrated'},style,'Illustrated');html('option',{value:'wiring'},style,'Wiring');
      style.onchange=()=>{this.style=style.value;this.render();};
      const flowLabel=html('label',{},controls,'Wires '),flow=html('select',{'aria-label':'Wire display'},flowLabel);
      for(const [value,label] of [['both','Values and loads'],['load','Loads only'],['value','Values only']])html('option',{value},flow,label);
      this.el.dataset.flow=flow.value;
      flow.onchange=()=>{this.el.dataset.flow=flow.value;};
      const zoomLabel=html('label',{},controls,'Zoom '),zoom=html('select',{'aria-label':'Diagram size'},zoomLabel);
      this.zoomSelect=zoom;this.zoomLevels=[.25,.5,.75,1,1.25,1.5,2,3,4];
      html('option',{value:'fit'},zoom,'Fit');for(const value of this.zoomLevels)html('option',{value},zoom,value*100+'%');
      this.syncZoomControl();zoom.onchange=()=>this.setZoom(zoom.value);
      this.button(controls,'−',()=>this.zoomBy(-1),'ex-zoom-button').setAttribute('aria-label','Zoom out');
      this.button(controls,'+',()=>this.zoomBy(1),'ex-zoom-button').setAttribute('aria-label','Zoom in');
      this.crumbs=html('nav',{class:'ex-breadcrumbs','aria-label':'Inside arrangement'},this.el);
      this.scopeAnimation=html('section',{class:'ex-scope-animation','aria-label':'Current arrangement animation'},this.el);
      this.scopeFigure=html('figure',{},this.scopeAnimation);
      this.scopeCaption=html('figcaption',{},this.scopeFigure);
      this.scopePicture=svg('svg',{viewBox:'-80 -55 160 110',role:'img'},this.scopeFigure);
      if(options.rootAnimation)this.scopeAnimation.append(options.rootAnimation);
      const toolbar=html('div',{class:'ex-toolbar'},this.el);
      this.selectButton=this.button(toolbar,'Select parts',()=>{this.selectMode=!this.selectMode;this.selectButton.setAttribute('aria-pressed',String(this.selectMode));this.render();});
      this.selectButton.setAttribute('aria-pressed','false');
      this.selectedLabel=html('span',{class:'ex-help'},toolbar);
      const groupTools=html('div',{class:'ex-group-tools'},toolbar);
      this.groupName=html('input',{placeholder:'Group name','aria-label':'Group name',maxlength:'50'},groupTools);
      this.groupButton=this.button(groupTools,'Put a box around',()=>this.act(()=>{this.model.group([...this.selection],this.groupName.value);this.selection.clear();this.groupName.value='';this.say('Grouped. Open the new box to see its contents.');}));
      this.ungroupButton=this.button(groupTools,'Remove this box',()=>this.act(()=>this.model.ungroup()));
      this.undoButton=this.button(groupTools,'Undo grouping',()=>this.act(()=>this.model.undo()));
      this.button(groupTools,'Original grouping',()=>this.act(()=>this.model.resetGroups()));
      this.button(groupTools,'Auto arrange',()=>{delete this.placements[this.model.scope];this.persist();this.render();this.say('Automatic positions restored at this level.');});
      const workspace=html('div',{class:'ex-workspace'},this.el),canvasWrap=html('div',{class:'ex-canvas-wrap'},workspace);
      this.canvas=svg('svg',{class:'ex-canvas',role:'group','aria-label':'Arrangement diagram',preserveAspectRatio:'xMinYMin meet'},canvasWrap);
      this.canvasWrap=canvasWrap;
      this.onDragMove=e=>this.moveDrag(e);this.onDragEnd=e=>this.endDrag(e);this.onDragCancel=e=>this.endDrag(e,true);
      window.addEventListener('pointermove',this.onDragMove,{passive:false});window.addEventListener('pointerup',this.onDragEnd);window.addEventListener('pointercancel',this.onDragCancel);
      this.canvas.addEventListener('wheel',e=>{if(e.ctrlKey||e.metaKey){e.preventDefault();this.pinchZoom(e);}},{passive:false});
      this.sidebar=html('aside',{class:'ex-sidebar','aria-label':'Selected part or wire'},workspace);
      const legend=html('div',{class:'ex-legend'},this.el);
      const valueSpan=html('span',{},legend,'Values → ');html('i',{class:'ex-packets value'},valueSpan);
      valueSpan.append(' bead shade: light = lowest so far, dark = highest');
      if(E.packets.valuePackets!=='steady'){valueSpan.append(' small ');html('i',{class:'ex-packets value fast'},valueSpan);valueSpan.append(' large');}
      const loadSpan=html('span',{},legend,'Loads · small ');html('i',{class:'ex-packets load slow'},loadSpan);
      loadSpan.append(' large ');html('i',{class:'ex-packets load fast'},loadSpan);
      html('span',{},legend,'Faster = larger load · toward the source = positive, raising it · away = negative, lowering it');
      html('p',{class:'ex-help'},this.el,'Click a box to enter; drag to move it. Shift-click parts to group them. Click a port to follow its readout. A hose is several wires bundled, one per number: click its mouth to unwrap that end into its wires, and again to wrap them. Zoom with + / − or Ctrl/⌘ + scroll. Auto arrange restores positions at this level. Alt + arrow keys move a focused part.');
      this.status=html('div',{class:'ex-status',role:'status','aria-live':'polite'},this.el);
      this.refresh();
      this.tick=now=>{
        if(this.destroyed)return;
        const elapsed=this.last?Math.min(.1,(now-this.last)/1000):0;this.last=now;
        if(this.running){
          try{this.advance(elapsed*this.speed);if(!this.lastRender||now-this.lastRender>=100){this.refresh();this.lastRender=now;}}
          catch(error){this.setRunning(false);this.say(error.message,true);}
        }
        this.raf=requestAnimationFrame(this.tick);
      };
      if(adapter.step && !options.playback)this.raf=requestAnimationFrame(this.tick);
    }
    button(parent,label,action,cls=''){const b=html('button',{type:'button',class:cls},parent,label);b.onclick=action;return b;}
    say(text,error=false){this.status.textContent=text;this.status.classList.toggle('ex-error',error);}
    setRunning(on){
      if(this.options.playback)this.options.playback.setPlaying(on);
      if(this.play)this.play.textContent=on?'Pause':'Play';
      this.running=on;this.el.dataset.playing=String(on);this.last=0;
    }
    advance(dt){this.adapter.step?.(dt);if(this.adapter.ended?.())this.setRunning(false);}
    act(fn){try{fn();this.persist();this.selection.clear();this.focusWire=null;this.focusSignal=null;this.render();}catch(error){this.say(error.message,true);}}
    persist(){if(this.storageKey)try{localStorage.setItem(this.storageKey,JSON.stringify({...this.model.save(),placements:this.placements,zoom:this.zoom,hoses:this.hoses}));}catch{}}
    // Wrap or unwrap one end of a hose at this level. Like moving a part, it changes the drawing only.
    toggleHose(end){
      const scope=this.model.scope,ends=new Set(this.hoses[scope]||[]),before=this.hoses[scope];
      if(ends.has(end))ends.delete(end);else ends.add(end);
      this.hoses[scope]=[...ends];
      try{this.layout(this.model.project(),true);}catch(error){this.hoses[scope]=before;if(!before)delete this.hoses[scope];this.say(error.message);return false;}
      this.persist();this.render();return true;
    }
    syncZoomControl(){
      const select=this.zoomSelect;let custom=select.querySelector('[data-custom-zoom]');
      if(![...select.options].some(o=>!o.hasAttribute('data-custom-zoom')&&o.value===String(this.zoom))){
        if(!custom)custom=html('option',{'data-custom-zoom':'true'},select);
        custom.value=String(this.zoom);custom.textContent=Math.round(Number(this.zoom)*100)+'%';
      }else custom?.remove();
      select.value=String(this.zoom);
    }
    pinchZoom(event){
      const current=this.pendingZoom?.scale??this.canvas.getScreenCTM().a;
      this.pendingZoom={scale:E.wheelZoom(current,event.deltaY,event.deltaMode),clientX:event.clientX,clientY:event.clientY};
      if(this.zoomFrame)return;
      this.zoomFrame=requestAnimationFrame(()=>{
        const next=this.pendingZoom;this.pendingZoom=null;this.zoomFrame=null;
        this.setZoom(next.scale,next,false);
        clearTimeout(this.zoomSaveTimer);this.zoomSaveTimer=setTimeout(()=>this.persist(),200);
      });
    }
    setZoom(value,event,save=true){
      if(this.zoomFrame){cancelAnimationFrame(this.zoomFrame);this.zoomFrame=null;this.pendingZoom=null;}
      const rect=this.canvas.getBoundingClientRect(),wrap=this.canvasWrap.getBoundingClientRect();
      const cursor={x:event?.clientX??(wrap.left+this.canvasWrap.clientWidth/2),y:event?.clientY??((Math.max(rect.top,0)+Math.min(rect.bottom,innerHeight))/2)};
      const world=new DOMPoint(cursor.x,cursor.y).matrixTransform(this.canvas.getScreenCTM().inverse());
      this.zoom=String(value);this.syncZoomControl();this.render();
      const after=world.matrixTransform(this.canvas.getScreenCTM());
      this.canvasWrap.scrollLeft+=after.x-cursor.x;window.scrollBy(0,after.y-cursor.y);if(save)this.persist();
    }
    zoomBy(direction,event){
      const current=this.canvas.getScreenCTM().a;
      const value=direction>0?this.zoomLevels.find(v=>v>current*1.1)??4:this.zoomLevels.slice().reverse().find(v=>v<current/1.1)??.25;
      this.setZoom(value,event);
    }
    beginDrag(node,event){
      if(node.kind==='boundary'||event.button!==0||event.shiftKey||this.selectMode||event.target.closest('[data-port],[data-hose-end]'))return;
      const p=this.positions.get(node.id);
      this.drag={id:node.id,scope:this.model.scope,pointer:event.pointerId,x:p.x,y:p.y,clientX:event.clientX,clientY:event.clientY,
        scrollX:this.canvasWrap.scrollLeft,scrollY:window.scrollY+this.canvasWrap.scrollTop,scale:this.canvas.getScreenCTM().a,
        before:this.placements[this.model.scope]?.[node.id],active:false};
    }
    movePart(id,x,y){
      const scope=this.model.scope,previous=this.placements[scope]||{},next={...previous,[id]:{x,y}};
      this.placements[scope]=next;
      try{this.layout(this.model.project(),true);this.render();return true;}
      catch(error){this.placements[scope]=previous;this.say(error.message);return false;}
    }
    moveDrag(event){
      const d=this.drag;if(!d||event.pointerId!==d.pointer||d.scope!==this.model.scope)return;
      const dx=event.clientX-d.clientX,dy=event.clientY-d.clientY;
      if(!d.active&&Math.hypot(dx,dy)<5)return;
      if(!d.active){d.active=true;this.canvas.setPointerCapture(event.pointerId);this.el.dataset.dragging='true';}
      event.preventDefault();
      const moved=this.movePart(d.id,d.x+(dx+this.canvasWrap.scrollLeft-d.scrollX)/d.scale,
        d.y+(dy+window.scrollY+this.canvasWrap.scrollTop-d.scrollY)/d.scale);d.didMove=moved||d.didMove;
    }
    endDrag(event,cancel=false){
      const d=this.drag;if(!d||event.pointerId!==d.pointer)return;
      if(d.active&&!cancel)this.moveDrag(event);
      this.drag=null;this.el.dataset.dragging='false';
      if(this.canvas.hasPointerCapture(event.pointerId))this.canvas.releasePointerCapture(event.pointerId);
      if(!d.active)return;
      if(cancel){if(d.before)this.placements[d.scope][d.id]=d.before;else delete this.placements[d.scope][d.id];}
      this.suppressClick=d.id;setTimeout(()=>{this.suppressClick=null;},0);
      this.persist();this.render();this.say(cancel?'Move cancelled.':d.didMove?'Position saved. Click to open; drag to move.':'Leave space between parts; position unchanged.');
    }
    refresh(){
      if(this.options.playback){
        this.running=this.options.playback.playing;
        this.el.dataset.playing=String(this.running);
      }
      this.model.update(this.adapter.snapshot());for(const id of this.selection)if(!this.model.nodes.has(id))this.selection.delete(id);this.render();}
    enter(id){if(this.model.enter(id)){this.selection.clear();this.focusWire=null;this.focusSignal=null;this.focusNode=null;this.persist();this.render();}}
    choose(node,event){
      if(this.suppressClick===node.id)return;
      if((this.selectMode||event.shiftKey)&&node.kind!=='boundary'){
        if(this.selection.has(node.id))this.selection.delete(node.id);else this.selection.add(node.id);this.render();return;
      }
      if(this.model.children(node.id).length){this.enter(node.id);return;}
      this.focusNode=node.id;this.focusWire=node.wire||null;this.focusSignal=null;this.render();
    }
    layout(project,strict=false){
      const toggled=this.hoses[this.model.scope],hoseAbove=this.options.hoseAbove;
      const key=JSON.stringify([this.model.scope,project.nodes.map(n=>[n.id,n.kind,n.column,n.row,n.userGroup]),
        project.wires.map(w=>[w.id,w.source,w.target,w.signal,w.targetPort,w.label,E.dimension(w)]),this.placements[this.model.scope],toggled]);
      if(key!==this.layoutKey){
        try{this.geometry=E.layout(project,{placements:this.placements[this.model.scope],toggled,hoseAbove});}
        catch(error){if(strict)throw error;delete this.placements[this.model.scope];this.geometry=E.layout(project,{toggled,hoseAbove});this.say('The parts changed; automatic positions restored at this level.');}
        this.layoutKey=key;
      }
      else E.aggregate(project.wires).forEach((w,i)=>Object.assign(this.geometry.edges[i],w));
      return this.geometry;
    }
    drawWire(parent,w,d,tracks,start,end,trunk=false){
      const branches=w.branches||[w],selected=this.focusSignal?E.signalKey(w)===this.focusSignal:branches.some(b=>b.id===this.focusWire);
      const g=svg('g',{role:'button',tabindex:0,'data-wire':w.id,'data-trunk':String(trunk),
        'aria-label':w.label+'; inspect value and returning load',opacity:this.focusWire&&!selected?.16:1},parent);
      const reading=w.label+(w.argument===false?' · receives load ':' · value '+E.format(w.value)+' · returning load ')+E.format(w.load);
      svg('title',{},g,reading);g.setAttribute('aria-label',reading);
      const hose=E.dimension(w)>1;
      svg('path',{d,fill:'none',stroke:'white','stroke-width':hose?23:12,'stroke-linejoin':'round'},g);
      if(hose)this.drawHose(g,d);
      for(const kind of this.kinds(w)){
        // The arrowhead points the way the packets travel: a value forward, a load back to its source when it is
        // positive (or zero, or a hose's), forward when it is negative (explorer-core.js, `packetVelocity`).
        const magnitude=E.magnitude(w[kind]),lane=kind==='value'?1:-1;
        const velocity=E.packetVelocity(w[kind],w[kind+'Scale'],kind),sign=velocity?Math.sign(velocity):lane,forward=sign>0;
        const g2=svg('g',{'data-flow-track':kind,'data-level':magnitude==null?'unknown':magnitude===0?'zero':'nonzero'},g);
        this.drawTrack(g2,w,kind,tracks[kind],selected);
        const x=forward?end.x-10:start.x+10,y=(forward?end.y:start.y)-lane*E.geometry.trackOffset;
        svg('path',{d:`M ${x-sign*4} ${y-3.5} L ${x+sign*2} ${y} L ${x-sign*4} ${y+3.5}`,
          fill:'none',stroke:magnitude==null?'#89929f':E.packets.ink[kind],'stroke-width':2.6,'stroke-linejoin':'round','data-direction':forward?'forward':'backward'},g2);
      }
      svg('path',{d,fill:'none',stroke:'transparent','stroke-width':12},g);
      const select=()=>{this.focusWire=branches[0].id;this.focusSignal=trunk?E.signalKey(w):null;this.focusNode=null;this.render();};
      g.onclick=select;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
    }
    // The stretch of a wire inside a compact mark, from its labeled terminal to the mark itself:
    // the same value and load tracks as drawWire, so both are seen to reach the map or the field.
    // A strand of an unwrapped hose (explorer-core.js, `strand`) is drawn the same way, and says on hover which
    // of the hose's wires it is, with its value and load.
    drawLead(parent,w,tracks,d){
      const selected=this.focusSignal?E.signalKey(w)===this.focusSignal:(w.branches||[w]).some(b=>b.id===this.focusWire);
      const g=svg('g',{'data-lead':w.id,opacity:this.focusWire&&!selected?.16:1},parent);
      // `d` is the stretch's middle line: a hose stays a hose all the way into the mark, so that it is never
      // seen to turn into one wire, which would read as a projection.
      if(d&&E.dimension(w)>1)this.drawHose(g,d);
      if(w.strandOf){g.setAttribute('data-strand',w.index);
        svg('title',{},g,w.label+(w.argument===false?' · receives load ':' · value '+E.format(w.value)+' · returning load ')+E.format(w.load));}
      for(const kind of this.kinds(w))this.drawTrack(g,w,kind,tracks[kind],selected);
    }
    // The tracks a wire draws: no blue one where its value means nothing, and no amber one where the field it
    // enters reads it and never pushes it, so that its load is zero at every state (explorer-core.js, `pushes`).
    // That holds at every level: a wire into a box is judged by the marks it enters inside the box (`carriesLoad`).
    kinds(w){
      const kinds=w.argument===false?['load']:['value','load'];
      return kinds.filter(k=>k!=='load'||E.carriesLoad(this.model.nodes,w));
    }
    // One track: a line in the track's one color, and packets on it, evenly spaced, whose speed codes the size and
    // whose direction of travel shows a load's sign (explorer-core.js, `packets`, `packetVelocity`). A zero load has no packets, and a
    // missing observation dotted gray. Each track keeps how far its packets have travelled, signed, advanced by their
    // velocity over the model time since its last drawing, so a packet speeds up, slows down or turns round in place
    // when the load changes, and never jumps; a track not drawn for over a unit of time (another level was open)
    // resumes where it was. Start over starts the distance again. With reduced motion the packets hold still.
    drawTrack(parent,w,kind,path,selected){
      const magnitude=E.magnitude(w[kind]);
      svg('path',{d:path,fill:'none',stroke:magnitude==null?'#89929f':E.packets.color[kind],'stroke-width':selected?3.5:2.5,
        'data-kind':kind,'stroke-linejoin':'round','stroke-linecap':'round',
        'stroke-dasharray':magnitude==null?'2 5':'none'},parent);
      const velocity=E.packetVelocity(w[kind],w[kind+'Scale'],kind);if(!velocity)return;
      const still=typeof matchMedia==='function'&&matchMedia('(prefers-reduced-motion: reduce)').matches;
      const time=this.model.scene.time||0,key=(w.id||w.label)+'|'+kind;
      this.travel??=new Map();
      const last=this.travel.get(key),dt=last?time-last.time:0;
      // A packet never moves as far as half the spacing between two drawings: further, and the eye pairs it with
      // the packet behind and sees it run backward, which now reads as the opposite sign. The HH+ATP page's sodium
      // and potassium loads, 270 times their scale at a spike, reach 37 pixels a frame at its 4× playback.
      const reach=.45*E.packets.spacing,step=Math.max(-reach,Math.min(reach,velocity*dt));
      const distance=!last||dt<0?0:last.distance+(dt<=1?step:0);
      this.travel.set(key,{time,distance});
      const spacing=E.packets.spacing,dashes=E.packets.dash+' '+(spacing-E.packets.dash),offset=still?0:-(distance%spacing);
      // A value's beads take its shade in the range its wire has carried since the start (explorer-core.js,
      // `valueShade`), and a white rim, so that a bead of the track's own blue still shows on it.
      let ink=E.packets.ink[kind];
      if(kind==='value'){
        this.ranges??=new Map();const v=w.value,seen=this.ranges.get(key);
        const range=typeof v!=='number'?null:!seen||dt<0?{low:v,high:v}:{low:Math.min(seen.low,v),high:Math.max(seen.high,v)};
        if(range)this.ranges.set(key,range);
        const u=range&&range.high>range.low?(v-range.low)/(range.high-range.low):null;
        ink=E.valueShade(u);
        svg('path',{d:path,fill:'none',stroke:'white','stroke-width':selected?8:7,'stroke-linecap':'round',
          'stroke-dasharray':dashes,'stroke-dashoffset':offset},parent);
      }
      svg('path',{d:path,fill:'none',stroke:ink,'stroke-width':selected?5.5:4.5,'stroke-linecap':'round','data-shade':ink,
        'stroke-dasharray':dashes,'data-packet-speed':Math.abs(velocity).toFixed(1),
        'data-packet-velocity':velocity.toFixed(1),class:'ex-pulse'+(velocity>0?'':' ex-back'),
        'stroke-dashoffset':offset},parent);
    }
    // A hose, the shape around wires that are bundled: a tube around the middle line `d` of a wire that carries
    // a list of numbers, ribbed so that it reads as a hose and not as one thick wire. The value track and the
    // load track run inside it, colored by the size of the whole list.
    drawHose(parent,d){
      svg('path',{d,fill:'none',stroke:'#7d91a4','stroke-width':17,'stroke-linejoin':'round','data-hose':'tube'},parent);
      svg('path',{d,fill:'none',stroke:'#eef2f6','stroke-width':14,'stroke-linejoin':'round'},parent);
      svg('path',{d,fill:'none',stroke:'#c3cfda','stroke-width':14,'stroke-dasharray':'1.4 8.6'},parent);
    }
    // The mouth of a hose, the icon that wraps and unwraps it: an oval, the hose's open end seen at a slant,
    // with the number of wires in the hose beside it. On a part it is a wrapped end, the part's one hose port
    // for that many numbers; away from the part, where the strands of an unwrapped end meet the hose, it is
    // the unwrapped one. Clicking it changes that end to the other. `end` is the end's key (none where the
    // end cannot be unwrapped: a boundary port, a spring's terminal); `outward` is 1 to write the count to
    // the right of the mouth and -1 to its left, the side away from the part or from the port's name.
    // `standsFor` is what the model says the list stands in for (a wire's `standsFor`, such as 'ℝ^[−9, 9]' for
    // a function on a line held as n coefficients): written after the count, the part after ^ raised.
    // It is written where a hose leaves a part and where it crosses a boundary (there above the port's name,
    // `rise` higher), not at each terminal it enters, where the names stand close.
    drawMouth(parent,x,y,count,end,wrapped,label,outward,standsFor,rise=12){
      const g=svg('g',{class:'ex-hose-end','data-wrapped':String(wrapped),role:end?'button':'img'},parent);
      const say=label+' · '+count+' wires in a hose'+(standsFor?', standing in for '+standsFor:'')+(end?wrapped?' · click to unwrap them':' · click to wrap them':'');
      svg('title',{},g,say);g.setAttribute('aria-label',say);
      svg('circle',{cx:x,cy:y,r:12,fill:'transparent'},g);
      svg('ellipse',{cx:x,cy:y,rx:4.8,ry:9.6,fill:'white',stroke:'#4f6578','stroke-width':1.8},g);
      svg('ellipse',{cx:x,cy:y,rx:1.7,ry:5.4,fill:'#4f6578'},g);
      svg('text',{x:x+outward*8,y:y-rise,'text-anchor':outward>0?'start':'end',class:'ex-port-label ex-hose-count',stroke:'white','stroke-width':3,'paint-order':'stroke'},g,String(count));
      if(standsFor){
        const [base,power]=standsFor.split('^'),tag=g.lastChild;
        svg('tspan',{},tag,' for '+base);
        if(power){svg('tspan',{dy:-4,'font-size':'82%'},tag,power);svg('tspan',{dy:4},tag,'​');}
      }
      if(!end)return;
      g.setAttribute('data-hose-end',end);g.setAttribute('tabindex',0);
      const toggle=e=>{e.stopPropagation();this.toggleHose(end);};
      g.onclick=toggle;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();toggle(e);}};
    }
    // A spring, the field minus d of half a stiffness times a squared difference: no mark, only the bend that
    // joins its two wires, the bend's upright stretch drawn as a resistor in the book's load red (loadcolor,
    // red!80!black) with the stiffness written over the top. A resistor has this same field.
    // Two more of the kind. A spring to an anchor has one wire: the resistor runs from it down to a hatched bar,
    // the fixed end, held at 0. A spring with a third terminal has, between its two ends, the wire that sets its
    // stiffness: it meets the resistor from the side at an arrowhead, as the wiper of a variable resistor does,
    // and the spring pushes back on it too.
    drawSpring(g,node,p,selected){
      const ends=p.inputs.filter(port=>!port.third),ys=ends.map(port=>port.glyphY).sort((a,b)=>a-b),x=p.glyphX;
      const top=ys[0],bottom=node.anchor?top+E.geometry.springRows-2*E.geometry.springTurn:ys.at(-1);
      const teeth=6,step=(bottom-top-8)/teeth;
      let d=`M ${x} ${top} L ${x} ${top+4}`;
      for(let k=0;k<teeth;k++)d+=` L ${x+(k%2?-7:7)} ${top+4+(k+.5)*step}`;
      d+=` L ${x} ${bottom-4} L ${x} ${bottom}`;
      svg('rect',{x:x-13,y:top-2,width:26,height:bottom-top+4,rx:6,fill:selected?'#e0effa':'transparent',
        stroke:selected?'#1474ab':'none','stroke-width':selected?3:1.5,class:'ex-focus-ring'},g);
      svg('path',{d,fill:'none',stroke:'#cc0000','stroke-width':2.4,'stroke-linejoin':'round','stroke-linecap':'round','data-glyph':'spring'},g);
      if(node.anchor){
        let bar=`M ${x-11} ${bottom} L ${x+11} ${bottom}`;
        for(let k=-2;k<=2;k++)bar+=` M ${x+k*5+2} ${bottom} l -4 6`;
        svg('path',{d:bar,fill:'none',stroke:'#5b6772','stroke-width':1.8,'stroke-linecap':'round','data-glyph':'anchor'},g);
      }
      const third=p.inputs.find(port=>port.third);
      if(third){
        const y=third.glyphY,tip=x-2;
        svg('path',{d:`M ${tip-7} ${y-4.5} L ${tip} ${y} L ${tip-7} ${y+4.5} Z`,fill:'#cc0000',stroke:'#cc0000','stroke-width':1,'stroke-linejoin':'round','data-glyph':'third-terminal'},g);
      }
      if(Forms&&node.stiffness){
        const rows=p.inputs.map(port=>port.y-p.y),tag=svg('text',{x:x+3,y:Math.min(...rows)-9,'text-anchor':'middle',class:'ex-form ex-stiffness'},g);
        writeSvg(tag,Forms.runs(Forms.parse(node.stiffness)));
      }
    }
    // A motor, a part that pushes a load on a store and charges the battery the work of that push (two pushes that
    // share a rate): one mark, a picture of a motor (David, 2026-11-21, of a circle with the letter M: "looks good. i
    // picture of a motor might be better."), standing upright so that what runs it and what it runs are told apart
    // (of the picture lying on its side, every wire entering one end: "it's unlcear here which is meant to be
    // running it (battery) and which is being run by it (grip). the motor could be written at a 90° angle so that
    // it was clearer what's the cord and what's the torque"). Its finned body is on the fill of a field's mark,
    // outlined in the load red of the spring's resistor. Its shaft goes up to a pulley, where the wire of the store
    // it turns ends; its cord leaves its bottom for a plug, where the battery's wire ends; the wires it only hears
    // (the rate or the torque it is sent, the other load on its store) end at the body's side, behind the black
    // wall of a load-free port (`drawWall`). The places are explorer-layout.js's `MOTOR`. Clicked, it opens into
    // its pushes, whose count is under it.
    drawMotor(g,node,p,selected,inside){
      const x=p.glyphX,M=E.geometry.motor,red=selected?'#1474ab':'#cc0000',cord='#3d4650';
      // the cord and its plug, then the shaft and its pulley, then the body over their ends
      svg('path',{d:`M ${x} ${M.bottom} L ${x} ${M.cordY-4} Q ${x} ${M.cordY} ${x-4} ${M.cordY} L ${x+M.plugX+5} ${M.cordY}`,fill:'none',stroke:cord,'stroke-width':1.8,'stroke-linecap':'round','data-glyph':'cord'},g);
      svg('rect',{x:x+M.plugX,y:M.cordY-3.4,width:5.5,height:6.8,rx:1.4,fill:cord},g);
      svg('rect',{x:x-2.6,y:M.hubY,width:5.2,height:M.top-M.hubY+2,fill:red},g);
      svg('circle',{cx:x,cy:M.hubY,r:M.hub,fill:'#f8e8dc',stroke:red,'stroke-width':1.8,'data-glyph':'pulley'},g);
      svg('rect',{x:x-M.half,y:M.top,width:2*M.half,height:M.bottom-M.top,rx:5,fill:selected?'#e0effa':'#f8e8dc',stroke:red,'stroke-width':selected?3:2.2,class:'ex-focus-ring','data-glyph':'motor'},g);
      // the body's cooling fins, and the band of its end bell at the shaft's end
      for(const dx of [-5,0,5])svg('path',{d:`M ${x+dx} ${M.top+8} L ${x+dx} ${M.bottom-4.5}`,fill:'none',stroke:red,'stroke-width':1.5,'stroke-linecap':'round'},g);
      svg('path',{d:`M ${x-M.half+1.5} ${M.top+4} L ${x+M.half-1.5} ${M.top+4}`,fill:'none',stroke:red,'stroke-width':1.5},g);
      // under the battery's wire, which may come up to the plug from the row below it
      if(inside)svg('text',{x,y:M.cordY+24,'text-anchor':'middle',class:'ex-secondary'},g,inside+' inside');
    }
    // The book's mark for a load-free input port (Ch 3: "We draw a black wall at each input port where an arrangement
    // is load-free"): a short thick black bar across the wire at the port, in place of the port's ring. At a hose
    // port it stands just before the hose's mouth and spans the tube.
    drawWall(parent,x,y,hose){
      const at=hose?x-9:x,half=hose?12:7.5;
      svg('path',{d:`M ${at} ${y-half} L ${at} ${y+half}`,fill:'none',stroke:'#111','stroke-width':3.4,'data-wall':'load-free'},parent);
    }
    rendererFor(node){return this.renderers[node.renderer]||this.renderers[node.kind];}
    drawAnimation(group,node,width,height){
      try{
        const cleanup=this.rendererFor(node)({group,node,scene:this.model.scene,svg,width,height});
        if(typeof cleanup==='function')this.cleanups.push(cleanup);
      }catch(error){this.setRunning(false);this.say('Animation: '+error.message,true);}
    }
    renderScopeAnimation(){
      const node=this.model.nodes.get(this.model.scope),illustrated=this.style==='illustrated';
      const rootAnimation=this.options.rootAnimation;
      const showRoot=illustrated&&node.id===this.model.scene.root&&!!rootAnimation;
      const showFigure=illustrated&&!showRoot&&!!this.rendererFor(node);
      this.scopeAnimation.hidden=!showRoot&&!showFigure;
      this.scopeFigure.hidden=!showFigure;
      if(rootAnimation)rootAnimation.hidden=!showRoot;
      this.scopePicture.replaceChildren();
      if(showFigure){
        this.scopeCaption.textContent=node.label;
        this.scopePicture.setAttribute('aria-label',node.label+' animation');
        this.drawAnimation(svg('g',{},this.scopePicture),node,160,110);
      }
    }
    render(){
      for(const cleanup of this.cleanups)cleanup();this.cleanups=[];
      const focused=document.activeElement?.getAttribute('data-node');
      const project=this.model.project(),{positions,width,height,edges,forks}=this.layout(project);
      this.project=project;this.positions=positions;
      this.time.textContent='Time '+E.format(this.model.scene.time??0);
      this.selectedLabel.textContent=this.selection.size?this.selection.size+' selected':'';
      this.groupButton.disabled=this.selection.size<2;this.ungroupButton.disabled=!this.model.nodes.get(this.model.scope)?.userGroup;this.undoButton.disabled=!this.model.history.length;
      this.crumbs.replaceChildren();
      this.model.ancestors(this.model.scope).forEach((n,i)=>{if(i)html('span',{},this.crumbs,'/');this.button(this.crumbs,n.label,()=>this.enter(n.id));});
      this.renderScopeAnimation();
      this.canvas.replaceChildren();this.canvas.setAttribute('viewBox',`0 0 ${width} ${height}`);
      this.canvas.style.width=this.drag?.active?width*this.drag.scale+'px':this.zoom==='fit'?`min(100%, ${width}px)`:width*Number(this.zoom)+'px';
      this.canvas.style.minWidth=this.drag?.active||this.zoom!=='fit'?'0':Math.min(600,width)+'px';
      this.canvas.style.minHeight=this.drag?.active||this.zoom!=='fit'?'0':'';
      const wireLayer=svg('g',{},this.canvas);
      // The current scope is an opened arrangement. Its ports lie on its boundary.
      svg('rect',{x:50,y:25,width:width-100,height:height-50,rx:6,fill:'none',stroke:'#d9e1e8','stroke-dasharray':'5 5'},wireLayer);
      for(const w of edges)this.drawWire(wireLayer,w,w.path,w.tracks,w.start,w.end);
      for(const fork of forks){
        const w=E.forkWire(fork.wires.flatMap(w=>w.branches)),d=`M ${fork.start.x} ${fork.start.y} L ${fork.end.x} ${fork.end.y}`;
        this.drawWire(wireLayer,w,d,fork.tracks,fork.start,fork.end,true);
        svg('circle',{cx:fork.end.x,cy:fork.end.y,r:E.dimension(w)>1?6:3.8,fill:'#425f73',class:'ex-fork','data-fork':fork.port.key},wireLayer);
      }
      // An unwrapped end of a hose: its strands, one wire per number, between the part's ports and the mouth.
      for(const p of positions.values())for(const group of p.groups){
        const out=group.side==='outputs';
        const whole=out&&group.wires.length>1?E.forkWire(group.wires.flatMap(w=>w.branches||[w])):group.wires[0];
        for(const port of group.ports)this.drawLead(wireLayer,E.strand(whole,port.index),port.tracks);
        this.drawMouth(wireLayer,group.x,group.y,group.dimension,group.end,false,group.label,out?1:-1,out?whole.standsFor:undefined);
      }
      let lawCard=null;
      for(const node of project.nodes){
        const p=positions.get(node.id),selected=this.selection.has(node.id),hasChildren=this.model.children(node.id).length;
        const g=svg('g',{transform:`translate(${p.x},${p.y})`,class:'ex-node',tabindex:0,role:'button','data-node':node.id,
          'aria-label':node.label+(hasChildren?' — open':' — inspect')+(selected?' — selected':'')},this.canvas);
        if(node.kind==='boundary'){
          svg('circle',{r:4.5,fill:'#fff',stroke:'#76889b','stroke-width':1.5},g);
          const left=p.x<width/2;
          const label=node.label.length>24?node.label.slice(0,22)+'…':node.label;
          // A hose stays wrapped where it crosses the boundary of the opened box; its name stands clear of the tube.
          const crossing=[...p.inputs,...p.outputs].find(port=>port.hose);
          svg('text',{x:left?8:-8,y:crossing?-13:-7,'text-anchor':left?'start':'end',class:'ex-port-label',stroke:'white','stroke-width':3,'paint-order':'stroke'},g,label);
          if(crossing)this.drawMouth(g,0,0,crossing.hose,null,true,node.label,(left?1:-1)*(crossing.wires[0].standsFor?1:-1),crossing.wires[0].standsFor,crossing.wires[0].standsFor?27:12);
        }else{
          if(node.kind==='box'&&!p.radius)svg('rect',{x:-p.w/2,y:-p.h/2,width:p.w,height:p.h,rx:this.style==='wiring'?0:12,fill:selected?'#e0effa':'#f5f9fc',stroke:selected?'#1474ab':'#a9becf','stroke-width':selected?3:1.5,class:'ex-focus-ring'},g);
          else if(p.radius){
            for(const port of [...p.inputs,...p.outputs]){
              const left=port.side==='in',px=port.x-p.x,py=port.y-p.y;
              const bend=p.glyphX+(left?-1:1)*(p.radius+10);
              // The value and its load run on from the terminal into the mark, on the wire's own two tracks.
              const T={x:px,y:py},B={x:bend,y:py},K={x:bend,y:port.glyphY},G={x:port.glyphX,y:port.glyphY};
              // A spring's wires are joined by a bend, the cap of string diagrams: each runs level to the
              // bend's corner and turns there into the upright stretch.
              const corner={x:port.glyphX,y:py};
              // A spring's third terminal, the wire that sets its stiffness, runs level to the resistor's side.
              const segments=port.third?[{a:T,b:G}]:node.glyph==='spring'?[{a:T,b:B},{a:B,c:corner,d:corner,b:G}]
                :left?[{a:T,b:B},{a:B,c:B,d:K,b:G}]:[{a:G,c:K,d:B,b:B},{a:B,b:T}];
              const whole=port.wires.length>1?E.forkWire(port.wires.flatMap(w=>w.branches||[w])):port.wires[0];
              this.drawLead(g,port.group?E.strand(whole,port.index):whole,E.geometry.wireTracks(segments),E.geometry.path(segments));
            }
            if(node.glyph==='spring')this.drawSpring(g,node,p,selected);
            else if(node.glyph==='motor')this.drawMotor(g,node,p,selected,hasChildren);
            else svg('circle',{cx:p.glyphX,r:p.radius,fill:selected?'#e0effa':node.kind==='field'?'#f8e8dc':'#eef1f4',
              stroke:selected?'#1474ab':'#91a1af','stroke-width':selected?3:1.5,class:'ex-focus-ring'},g);
          }
          // A store is one blue box, the book's fill for a part that only stores a state
          // (articulating.tex, matterfillcolor = valuecolor!10); it has no fictitious input port.
          else if(node.kind==='state')svg('rect',{x:-p.w/2,y:-p.h/2,width:p.w,height:p.h,rx:this.style==='wiring'?0:12,fill:selected?'#d5d5f5':'#e6e6f9',stroke:selected?'#1474ab':'#4d4dd2','stroke-width':selected?3:1.5,class:'ex-focus-ring'},g);
          else if(selected)svg('ellipse',{rx:85,ry:54,fill:'#e0effa',stroke:'#1474ab','stroke-dasharray':'4 4',class:'ex-focus-ring'},g);
          if(this.style==='illustrated'&&this.rendererFor(node)){
            const animation=svg('g',{transform:node.kind==='state'?`translate(0,${-p.h/2+40}) scale(.7)`:p.radius?`translate(${p.glyphX},0) scale(.43)`:'scale(.62)'},g);
            this.drawAnimation(animation,node,p.w,p.h);
          }else if(node.kind==='state'){
            // The stored value keeps its place under the store's top, above the readouts, however many there are.
            const v=E.magnitude(node.value)||0,top=svg('g',{transform:`translate(0,${-p.h/2+66})`},g);
            svg('line',{x1:-58,x2:58,y1:0,y2:0,stroke:'#c1d5e4','stroke-width':4},top);
            svg('circle',{cx:48*Math.tanh(v/5),cy:0,r:9,fill:'#237bab'},top);
            svg('text',{y:25,'text-anchor':'middle',class:'ex-secondary'},top,E.format(node.value).slice(0,28));
          }else if(hasChildren&&node.glyph!=='motor'){
            svg('text',{y:4,'text-anchor':'middle',class:'ex-secondary'},g,'Open');
          }else if(node.symbol){
            svg('text',{x:p.glyphX??0,y:4,'text-anchor':'middle',class:'ex-secondary'},g,node.symbol);
          }
          for(const port of [...p.inputs,...p.outputs]){
            const px=port.x-p.x,py=port.y-p.y,left=port.side==='in';
            if(node.kind==='state')svg('line',{x1:-64,x2:p.w/2,y1:py,y2:py,stroke:'#b9cedd','stroke-width':2},g);
            const mark=svg('g',{'data-port':port.key,'data-side':port.side,role:'button',tabindex:0,'aria-label':port.label+' '+port.side+' port; follow readout'},g);
            svg('circle',{cx:px,cy:py,r:10,fill:'transparent'},mark);
            // One port carries one number. A port that carries a list is a hose port, drawn below as the
            // hose's mouth; a strand's port is one number of an unwrapped hose.
            // An input port no mark inside can push is load-free: the book's black wall stands there, in place of the ring.
            const walled=left&&node.kind!=='state'&&!E.carriesLoad(this.model.nodes,{branches:port.wires.flatMap(w=>w.branches||[w])});
            if(walled)this.drawWall(mark,px,py,port.hose);
            else if(!port.hose)svg('circle',{cx:px,cy:py,r:4,fill:'white',stroke:'#386b88','stroke-width':1.6},mark);
            svg('title',{},mark,port.label+' · '+(left?'input':'output')+(walled?' · load-free: read here, never pushed':''));
            const strand=port.group&&E.strand(port.wires[0],port.index),place=strand&&E.coordinate(port.wires[0],port.index);
            const text=node.kind==='state'?(port.hose?port.hose+' numbers':strand?place+':  '+E.format(strand.value):E.format(port.wires[0].value))
              :strand&&!p.radius&&port.label.length>16?'… '+place:port.label;
            // A store's readout is written under its line; above it would sit on the stored value (y=25).
            const tag=svg('text',{x:node.kind==='state'?0:px+(left?9:-9),y:node.kind==='state'?py+15:py-6,'text-anchor':node.kind==='state'?'middle':left?'start':'end',class:'ex-port-label'},mark,!p.radius && text.length>16?text.slice(0,14)+'…':text);
            // The letter the field's formula uses for this terminal, after its name; a strand's is its
            // hose's letter with the strand's number under it.
            // A hose terminal's name lies on the hose's tube: a white edge keeps it readable.
            if(port.hose&&p.radius)for(const [k,v] of [['stroke','white'],['stroke-width',3],['paint-order','stroke']])tag.setAttribute(k,v);
            const hoseLetter=Forms&&(node.form||node.law)?.letters?.[port.group?port.group.label:port.label];
            const letter=hoseLetter&&port.group&&!hoseLetter.includes('_')?hoseLetter+'_'+(port.index+1):hoseLetter;
            if(letter){svg('tspan',{},tag,'  ');const span=svg('tspan',{class:'ex-form'},tag);writeSvg(span,Forms.runs(Forms.parse(letter)));}
            if(port.hose)this.drawMouth(g,px,py,port.hose,port.end,true,port.label,left?-1:1,left?undefined:port.wires[0].standsFor);
            const selectPort=e=>{e.stopPropagation();this.focusSignal=port.wires[0].signal;this.focusWire=port.wires[0].branches[0].id;this.focusNode=null;this.render();};
            mark.onclick=selectPort;mark.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectPort(e);}};
          }
          svg('text',{y:-p.h/2-12,'text-anchor':'middle',class:'ex-part-label'},g,node.label);
          // A field's 1-form under its mark, red when it disagrees with the loads on the mark's wires. A form
          // too long for the canvas is cut here and whole in the sidebar: a long one is a field not yet factored.
          // Clicked, it or its mark opens the side panel, where each thing in it is explained (`writeGlossary`).
          if(Forms&&node.form){
            const line=Forms.written(node.form),long=Forms.plain(line).length>44;
            const tag=svg('text',{y:p.h/2+15,'text-anchor':'middle',class:'ex-form ex-form-open'+(this.formCheck(node).ok?'':' ex-form-wrong'),stroke:'white','stroke-width':3,'paint-order':'stroke'},g);
            writeSvg(tag,long?[{text:Forms.plain(line).slice(0,42)+'…'}]:line);
            svg('title',{},tag,'Click: what each letter is');
          }
          // A static map's law is written under its circle while the circle is the one chosen: each
          // abbreviation on its own line, whole, red when it disagrees with the value on the wire leaving it.
          // It is drawn after every part, on a white card, so that no wire or title crosses it.
          if(Forms&&node.law&&this.focusNode===node.id)
            lawCard={x:p.x+(p.glyphX??0),y:p.y+p.h/2+6,lines:Forms.definitions(node.law),wrong:!this.lawCheck(node).ok};
          if(hasChildren&&node.glyph!=='motor')svg('text',{x:p.w/2-9,y:p.h/2-8,'text-anchor':'end',class:'ex-secondary'},g,this.model.children(node.id).length+' inside');
          if(this.selectMode)svg('circle',{cx:-p.w/2+10,cy:-p.h/2+12,r:6,fill:selected?'#13669b':'white',stroke:'#13669b'},g);
        }
        g.onpointerdown=e=>this.beginDrag(node,e);g.onclick=e=>this.choose(node,e);
        g.onkeydown=e=>{
          if(e.altKey&&node.kind!=='boundary'&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){
            e.preventDefault();const dx=e.key==='ArrowLeft'?-20:e.key==='ArrowRight'?20:0,dy=e.key==='ArrowUp'?-20:e.key==='ArrowDown'?20:0;
            if(this.movePart(node.id,p.x+dx,p.y+dy))this.persist();return;
          }
          if(e.key==='Enter'||e.key===' '){e.preventDefault();this.choose(node,e);}
        };
      }
      if(lawCard){
        const w=Math.max(...lawCard.lines.map(line=>Forms.plain(line).length))*6.9+24,h=lawCard.lines.length*19+10;
        const x=Math.min(Math.max(lawCard.x,58+w/2),width-58-w/2),card=svg('g',{class:'ex-law-card','pointer-events':'none'},this.canvas);
        svg('rect',{x:x-w/2,y:lawCard.y,width:w,height:h,rx:7,fill:'white',stroke:lawCard.wrong?'#b3261e':'#c9a98f','stroke-width':1.2},card);
        lawCard.lines.forEach((line,k)=>writeSvg(svg('text',{x,y:lawCard.y+19+k*19,'text-anchor':'middle',class:'ex-form'+(lawCard.wrong?' ex-form-wrong':'')},card),line));
      }
      this.renderDetails();
      if(focused){const el=[...this.canvas.querySelectorAll('[data-node]')].find(el=>el.getAttribute('data-node')===focused);el?.focus({preventScroll:true});}
    }
    // A field's form against the values and loads now on the wires that enter its mark.
    formCheck(node){
      return Forms.check(node.form,this.model.scene.wires.filter(w=>w.target===node.id).map(w=>({label:w.label,value:w.value,load:w.load})));
    }
    // A static map's law against the values now on its entering wires and the value on the wire leaving it.
    // And the loads on its entering wires against the load on the wires leaving it, sent back through the law's
    // derivative (`carries` says whether there is such a load now).
    lawCheck(node){
      const all=this.model.scene.wires,leaving=all.filter(edge=>edge.source===node.id),entering=all.filter(edge=>edge.target===node.id);
      const value=Forms.lawCheck(node.law,entering.map(edge=>({label:edge.label,value:edge.value})),leaving[0]?.value);
      if(!value.ok||!leaving.length)return value;
      const load=E.forkWire(leaving).load,back=Forms.pullbackCheck(node.law,entering.map(edge=>({label:edge.label,value:edge.value,load:edge.load})),load);
      return {...value,ok:back.ok,error:back.error,carries:!!E.magnitude(load)};
    }
    // A part's description, the subscripts in its formulas lowered.
    writeDescription(text){
      const p=html('p',{},this.sidebar,Forms?undefined:text);
      if(Forms)writeHtml(p,Forms.wordRuns(text,true));
    }
    // Under a form in the sidebar: the wires entering the mark by letter, then each thing that is not a wire.
    writeGlossary(form){
      const wires=Object.entries(form.letters||{});
      if(wires.length){
        const p=html('p',{class:'ex-form'},this.sidebar);
        wires.forEach(([label,letter],k)=>writeHtml(p,[...(k?[{text:',  '}]:[]),...Forms.runs(Forms.parse(letter)),{text:': '+label}]));
      }
      for(const {head,words,tail} of Forms.glossary(form))writeHtml(html('p',{class:'ex-form'},this.sidebar),[...head,...(words?[{text:': '},...Forms.wordRuns(words)]:[]),...(tail||[])]);
    }
    // The constants of a law, on one line of the sidebar: name = number.
    writeConstants(written){
      const constants=Object.entries(written.constants||{});
      if(!constants.length)return;
      const p=html('p',{class:'ex-form'},this.sidebar);
      constants.forEach(([name,v],k)=>writeHtml(p,[...(k?[{text:',  '}]:[]),...Forms.runs(Forms.parse(name)),{text:' = '+E.format(v)}]));
    }
    renderDetails(){
      this.sidebar.replaceChildren();
      const n=this.model.nodes.get(this.focusNode),w=this.model.scene.wires.find(w=>w.id===this.focusWire);
      html('h2',{},this.sidebar,n?.label||w?.label||this.model.nodes.get(this.model.scope).label);
      const all=this.model.scene.wires;
      const sourceOf=node=>{
        const seen=new Set();
        while(node?.kind==='boundary'&&!seen.has(node.id)){
          seen.add(node.id);
          const upstream=all.filter(edge=>edge.target===node.id);
          if(upstream.length!==1)break;
          node=this.model.nodes.get(upstream[0].source);
        }
        return node;
      };
      const producer=sourceOf(n || ((this.focusSignal||this.focusNode) && w && this.model.nodes.get(w.source)));
      if(producer?.kind==='operation'||producer?.kind==='field'){
        const inputs=new Map(all.filter(edge=>edge.target===producer.id&&edge.argument!==false).map(edge=>[E.signalKey(edge),edge]));
        const outputs=new Map(all.filter(edge=>edge.source===producer.id).map(edge=>[E.signalKey(edge),edge]));
        if(outputs.size>1&&!this.focusSignal){
          html('h3',{},this.sidebar,'Readouts');
          for(const [signal,edge] of outputs)this.button(this.sidebar,edge.label,()=>{
            this.focusNode=null;this.focusSignal=signal;this.focusWire=edge.id;this.render();
          });
        }else if(producer.dependencyView||producer.kind==='operation'||outputs.size){
          const possible=producer.possibleInputs||outputs.size>1||(producer.kind==='field'&&!producer.dependencyView);
          const label=outputs.size>1?w.label:producer.label;
          html('h3',{},this.sidebar,(possible?'Possible inputs to ':'Inputs to ')+label);
          if(!inputs.size)html('p',{},this.sidebar,possible?'No input readouts recorded':'Constant for this arrangement');
          for(const edge of inputs.values()){
            const source=sourceOf(this.model.nodes.get(edge.source));
            const button=this.button(this.sidebar,edge.label,()=>{
              this.enter(source.parent);
              this.focusNode=source.id;this.focusSignal=null;this.focusWire=null;this.render();
            });
            button.title='Inspect '+source.label;
          }
          if(producer.result!==undefined){
            html('h3',{},this.sidebar,producer.resultLabel||'Result');
            html('output',{},this.sidebar,E.format(producer.result)+(producer.resultUnit?' '+producer.resultUnit:''));
          }
        }
      }
      if(n){
        for(const reading of n.observations||[]){
          html('h3',{},this.sidebar,reading.label);html('output',{},this.sidebar,E.format(reading.value));
        }
        if(Forms&&n.form){
          html('h3',{},this.sidebar,'Inclination field');
          writeHtml(html('output',{class:'ex-form'},this.sidebar),Forms.written(n.form));
          this.writeGlossary(n.form);
          const verdict=this.formCheck(n);
          html('p',{class:verdict.ok?'ex-form-agrees':'ex-form-wrong'},this.sidebar,verdict.ok?'This formula gives the loads now on its wires.'
            :verdict.error||('This formula disagrees with the model: '+verdict.rows.filter(r=>!r.ok).map(r=>`at ${r.letter} it gives ${E.format(r.got)}, the model ${E.format(r.want)}`).join('; ')+'.'));
        }
        // A static map's law, in the letters at its terminals, checked against the value on the wire leaving it.
        // The heading is the book's term for a part that is stateless and has no inclination field.
        if(Forms&&n.law){
          html('h3',{},this.sidebar,'Static arrangement');
          for(const line of Forms.definitions(n.law))writeHtml(html('p',{class:'ex-form'},this.sidebar),line);
          this.writeConstants(n.law);
          const verdict=this.lawCheck(n);
          html('p',{class:verdict.ok?'ex-form-agrees':'ex-form-wrong'},this.sidebar,verdict.ok?'This formula gives the value now on its wire'
            +(verdict.carries?', and the load on that wire goes back along each entering wire times the formula’s derivative there.':'.')
            :(verdict.error||`This formula gives ${E.format(verdict.got)}, the model ${E.format(verdict.want)}`)+'.');
        }
        if(n.description)this.writeDescription(n.description);
        for(const [key,label] of [['value','State'],['configuration','Readout'],['inclination','Total inclination']])if(n[key]!==undefined){html('h3',{},this.sidebar,label);html('output',{},this.sidebar,E.format(n[key]));}
      }
      if(n||w)this.button(this.sidebar,'Show all wires',()=>{this.focusWire=null;this.focusSignal=null;this.focusNode=null;this.render();});
      let wires=[];
      if(w){const edge=this.geometry.edges.find(e=>e.branches.some(b=>b.id===w.id));wires=this.focusSignal?this.project.wires.filter(w=>E.signalKey(w)===this.focusSignal):edge?.branches||[w];}
      else if(n)wires=this.project.wires.filter(w=>w.source===n.id||w.target===n.id);
      if(!n&&!w){
        const description=this.model.nodes.get(this.model.scope)?.description;if(description)this.writeDescription(description);
        html('p',{},this.sidebar,'Open a box to replace this view with its contents. Use the path above to come back out.');
        html('h3',{},this.sidebar,'At this level');
        for(const node of this.model.children())this.button(this.sidebar,node.label,()=>this.choose(node,{shiftKey:false}));
        html('p',{},this.sidebar,'Blue arrows carry values forward. Amber arrows carry loads back; the packets are evenly spaced, and the larger the load, the faster they move, so the more pass per second. A positive load, one that raises its source, travels back toward the source; a negative one travels away from it, and its arrowhead turns round. Paused, read a load by selecting its wire. A zero load has no packets; a wire whose field reads it and never pushes it, so that its load is always zero, has no amber track. A dotted gray track has no numeric observation.');
        if(this.options.description)html('p',{},this.sidebar,this.options.description);
      }
      if(this.focusSignal&&wires.length>1){
        const total=E.forkWire(wires);
        html('h3',{},this.sidebar,'Total returning load');
        html('output',{},this.sidebar,E.format(total.load)+(total.loadUnit?' '+total.loadUnit:''));
      }
      for(const wire of E.aggregate(wires)){
        const row=html('div',{class:'ex-wire-detail'},this.sidebar);html('strong',{},row,wire.label);
        if(this.focusSignal)html('p',{},row,'To '+(this.model.nodes.get(wire.target)?.label||'the boundary'));
        for(const [key,label] of [['value',wire.argument===false?'Coordinate value':'Value'],['load',wire.inclination?'Inclination':'Returning load']]){
          const line=html('label',{},row,label+' ');html('output',{},line,E.format(wire[key]));
          if(key==='load'||E.packets.valuePackets!=='steady'){const s=wire[key+'Scale']??1,unit=wire[key+'Unit']?' '+wire[key+'Unit']:'';
            html('div',{class:'ex-scale'},row,'Packets slowest at |'+key+'| ≤ '+E.format(s*E.packets.bottom)+unit+', fastest at ≥ '+E.format(s*E.packets.top)+unit+'; each tenfold step '+Math.pow(E.packets.fastest/E.packets.slowest,1/Math.log10(E.packets.top/E.packets.bottom)).toFixed(2)+'× faster');}
        }
        if(E.dimension(wire)>1)html('p',{},row,'A hose: '+E.dimension(wire)+' wires bundled, one per number'+(wire.standsFor?', standing in for '+wire.standsFor.replace('^','^('  )+(wire.standsFor.includes('^')?')':''):'')+'. Click a mouth of the hose to unwrap that end into its wires.');
        if(wire.description)html('p',{},row,wire.description);
        if(wire.branches.length>1)html('p',{},row,'The '+wire.branches.length+' internal contributions add to this returning load.');
      }
    }
    destroy(){this.options.playback?.setPlaying(false);this.destroyed=true;cancelAnimationFrame(this.raf);if(this.zoomFrame)cancelAnimationFrame(this.zoomFrame);clearTimeout(this.zoomSaveTimer);this.persist();window.removeEventListener('pointermove',this.onDragMove);window.removeEventListener('pointerup',this.onDragEnd);window.removeEventListener('pointercancel',this.onDragCancel);for(const cleanup of this.cleanups)cleanup();this.cleanups=[];this.adapter.dispose?.();this.el.remove();}
  }
  E.View=View;
})(globalThis);
