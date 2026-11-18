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
  class View {
    constructor(container,adapter,options={}) {
      this.adapter=adapter;this.options=options;this.renderers=options.renderers||{};
      this.running=false;this.style='illustrated';this.selection=new Set();this.selectMode=false;this.focusWire=null;this.focusNode=null;
      this.last=0;this.accumulator=0;this.speed=1;this.destroyed=false;this.cleanups=[];
      this.placements={};this.zoom='fit';this.drag=null;
      const scene=adapter.snapshot();this.model=new E.Model(scene);
      this.storageKey=options.storageKey;
      if(this.storageKey)try{const s=JSON.parse(localStorage.getItem(this.storageKey));if(s){this.model.restore(s);if(s.placements&&typeof s.placements==='object')this.placements=s.placements;if(s.zoom==='fit'||Number(s.zoom)>=.25&&Number(s.zoom)<=4)this.zoom=s.zoom;}}catch{}
      this.el=html('section',{class:'explorer','data-playing':'false'},container);
      const header=html('header',{class:'ex-header'},this.el),title=html('div',{},header);
      html('div',{class:'ex-kicker'},title,'Arrangement explorer');
      html('h1',{},title,scene.title);
      html('p',{class:'ex-subtitle'},title,'Follow values out and loads back. Open a box to see what is inside.');
      if(options.sceneURL)html('a',{href:options.sceneURL,class:'ex-link'},header,options.sceneLabel||'Original animation');
      const controls=html('div',{class:'ex-controls'},this.el);
      this.play=this.button(controls,'Play',()=>this.setRunning(!this.running),'ex-play');
      this.play.disabled=!adapter.step;
      this.button(controls,'One step',()=>{this.setRunning(false);this.advance(.02);this.refresh();}).disabled=!adapter.step;
      this.button(controls,'Start over',()=>{this.setRunning(false);adapter.reset?.();this.refresh();}).disabled=!adapter.reset;
      this.time=html('output',{class:'ex-time'},controls);
      const speedLabel=html('label',{},controls,'Speed '),speed=html('select',{'aria-label':'Playback speed'},speedLabel);
      for(const x of [.5,1,2,4])html('option',{value:x},speed,x+'×');speed.value='1';speed.onchange=()=>{this.speed=+speed.value;};
      const styleLabel=html('label',{},controls,'View '),style=html('select',{'aria-label':'Drawing style'},styleLabel);
      html('option',{value:'illustrated'},style,'Illustrated');html('option',{value:'wiring'},style,'Wiring');
      style.onchange=()=>{this.style=style.value;this.render();};
      const zoomLabel=html('label',{},controls,'Zoom '),zoom=html('select',{'aria-label':'Diagram size'},zoomLabel);
      this.zoomSelect=zoom;this.zoomLevels=[.25,.5,.75,1,1.25,1.5,2,3,4];
      html('option',{value:'fit'},zoom,'Fit');for(const value of this.zoomLevels)html('option',{value},zoom,value*100+'%');
      this.syncZoomControl();zoom.onchange=()=>this.setZoom(zoom.value);
      this.button(controls,'−',()=>this.zoomBy(-1),'ex-zoom-button').setAttribute('aria-label','Zoom out');
      this.button(controls,'+',()=>this.zoomBy(1),'ex-zoom-button').setAttribute('aria-label','Zoom in');
      this.crumbs=html('nav',{class:'ex-breadcrumbs','aria-label':'Inside arrangement'},this.el);
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
      for(const [label,kind] of [['Values forward','value'],['Loads back','load']]) {
        const span=html('span',{},legend,label+' · low');html('i',{class:'ex-ramp '+kind},span);span.append('high');
      }
      html('span',{},legend,'Color = magnitude · signed numbers on selection');
      html('p',{class:'ex-help'},this.el,'Click a box to enter; drag to move it. Shift-click parts to group them. Click a port to follow its readout. Zoom with + / − or Ctrl/⌘ + scroll. Auto arrange restores positions at this level. Alt + arrow keys move a focused part.');
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
      this.raf=requestAnimationFrame(this.tick);
    }
    button(parent,label,action,cls=''){const b=html('button',{type:'button',class:cls},parent,label);b.onclick=action;return b;}
    say(text,error=false){this.status.textContent=text;this.status.classList.toggle('ex-error',error);}
    setRunning(on){this.running=on;this.play.textContent=on?'Pause':'Play';this.el.dataset.playing=String(on);this.last=0;}
    advance(dt){this.adapter.step?.(dt);if(this.adapter.ended?.())this.setRunning(false);}
    act(fn){try{fn();this.persist();this.selection.clear();this.focusWire=null;this.focusSignal=null;this.render();}catch(error){this.say(error.message,true);}}
    persist(){if(this.storageKey)try{localStorage.setItem(this.storageKey,JSON.stringify({...this.model.save(),placements:this.placements,zoom:this.zoom}));}catch{}}
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
      if(node.kind==='boundary'||event.button!==0||event.shiftKey||this.selectMode||event.target.closest('[data-port]'))return;
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
    refresh(){this.model.update(this.adapter.snapshot());for(const id of this.selection)if(!this.model.nodes.has(id))this.selection.delete(id);this.render();}
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
      const key=JSON.stringify([this.model.scope,project.nodes.map(n=>[n.id,n.kind,n.column,n.row,n.userGroup]),
        project.wires.map(w=>[w.id,w.source,w.target,w.signal,w.targetPort,w.label]),this.placements[this.model.scope]]);
      if(key!==this.layoutKey){
        try{this.geometry=E.layout(project,{placements:this.placements[this.model.scope]});}
        catch(error){if(strict)throw error;delete this.placements[this.model.scope];this.geometry=E.layout(project);this.say('The parts changed; automatic positions restored at this level.');}
        this.layoutKey=key;
      }
      else E.aggregate(project.wires).forEach((w,i)=>Object.assign(this.geometry.edges[i],w));
      return this.geometry;
    }
    drawWire(parent,w,d,start,end,trunk=false){
      const branches=w.branches||[w],selected=this.focusSignal?E.signalKey(w)===this.focusSignal:branches.some(b=>b.id===this.focusWire);
      const g=svg('g',{role:'button',tabindex:0,'data-wire':w.id,'data-trunk':String(trunk),
        'aria-label':w.label+'; inspect value and returning load',opacity:this.focusWire&&!selected?.16:1},parent);
      svg('title',{},g,w.label+' · value '+E.format(w.value)+' · load '+E.format(w.load));
      // Both colors follow the same complete path and meet the same port.
      // A thin return stroke leaves the blue value stroke visible on either side.
      svg('path',{d,fill:'none',stroke:'white','stroke-width':9,'stroke-linejoin':'round'},g);
      for(const [kind,weight] of [['value',selected?7:5],['load',selected?3:2]]){
        const color=E.color(w[kind],w[kind+'Scale'],kind);
        svg('path',{d,fill:'none',stroke:color,'stroke-width':weight,'data-kind':kind,'stroke-linecap':'round'},g);
        if(E.magnitude(w[kind])>1e-10)svg('path',{d,fill:'none',stroke:color,'stroke-width':kind==='value'?7:4,
          'stroke-linecap':'round','stroke-dasharray':'1 59',class:'ex-pulse'+(kind==='load'?' ex-back':''),
          'stroke-dashoffset':-(this.model.scene.time||0)*20*(kind==='load'?-1:1),
          style:`animation-duration:${3/this.speed}s;animation-delay:-${((this.model.scene.time||0)%3)/this.speed}s`},g);
        if(!trunk){const sign=kind==='value'?1:-1,x=kind==='value'?end.x-9:start.x+9,y=kind==='value'?end.y:start.y;
          svg('path',{d:`M ${x-sign*4} ${y-4} L ${x+sign*3} ${y} L ${x-sign*4} ${y+4}`,fill:'none',stroke:color,'stroke-width':2},g);}
      }
      svg('path',{d,fill:'none',stroke:'transparent','stroke-width':12},g);
      const select=()=>{this.focusWire=branches[0].id;this.focusSignal=null;this.focusNode=null;this.render();};
      g.onclick=select;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
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
      this.canvas.replaceChildren();this.canvas.setAttribute('viewBox',`0 0 ${width} ${height}`);
      this.canvas.style.width=this.drag?.active?width*this.drag.scale+'px':this.zoom==='fit'?'100%':width*Number(this.zoom)+'px';
      this.canvas.style.minWidth=this.drag?.active||this.zoom!=='fit'?'0':'';
      this.canvas.style.minHeight=this.drag?.active||this.zoom!=='fit'?'0':'';
      const wireLayer=svg('g',{},this.canvas);
      // The current scope is an opened arrangement. Its ports lie on its boundary.
      svg('rect',{x:50,y:25,width:width-100,height:height-50,rx:6,fill:'none',stroke:'#d9e1e8','stroke-dasharray':'5 5'},wireLayer);
      for(const w of edges)this.drawWire(wireLayer,w,w.path,w.start,w.end);
      for(const fork of forks){
        const wires=fork.wires.flatMap(w=>w.branches).map(w=>({...w,target:'shared-readout',targetPort:undefined}));
        const w=E.aggregate(wires)[0],d=`M ${fork.start.x} ${fork.start.y} L ${fork.end.x} ${fork.end.y}`;
        this.drawWire(wireLayer,w,d,fork.start,fork.end,true);
        svg('circle',{cx:fork.end.x,cy:fork.end.y,r:3.8,fill:'#425f73',class:'ex-fork','data-fork':fork.port.key},wireLayer);
      }
      for(const node of project.nodes){
        const p=positions.get(node.id),selected=this.selection.has(node.id),hasChildren=this.model.children(node.id).length;
        const g=svg('g',{transform:`translate(${p.x},${p.y})`,class:'ex-node',tabindex:0,role:'button','data-node':node.id,
          'aria-label':node.label+(hasChildren?' — open':' — inspect')+(selected?' — selected':'')},this.canvas);
        if(node.kind==='boundary'){
          svg('circle',{r:4.5,fill:'#fff',stroke:'#76889b','stroke-width':1.5},g);
          const left=p.x<width/2;
          const label=node.label.length>24?node.label.slice(0,22)+'…':node.label;
          svg('text',{x:left?8:-8,y:-7,'text-anchor':left?'start':'end',class:'ex-port-label',stroke:'white','stroke-width':3,'paint-order':'stroke'},g,label);
        }else{
          if(node.kind==='box')svg('rect',{x:-p.w/2,y:-p.h/2,width:p.w,height:p.h,rx:this.style==='wiring'?0:12,fill:selected?'#e0effa':'#f5f9fc',stroke:selected?'#1474ab':'#a9becf','stroke-width':selected?3:1.5,class:'ex-focus-ring'},g);
          else if(node.kind==='operation'||node.kind==='field')svg('circle',{r:p.radius,fill:selected?'#e0effa':node.kind==='field'?'#f8e8dc':'#eef1f4',stroke:selected?'#1474ab':'#c4cdd5','stroke-width':selected?3:1,class:'ex-focus-ring'},g);
          else if(selected)svg('ellipse',{rx:85,ry:54,fill:'#e0effa',stroke:'#1474ab','stroke-dasharray':'4 4',class:'ex-focus-ring'},g);
          // State has no rectangular frame and no fictitious input port.
          const renderer=this.style==='illustrated'&&(this.renderers[node.renderer]||this.renderers[node.kind]);
          if(renderer){
            try{const animation=svg('g',{transform:node.kind==='state'?`translate(0,${-p.h/2+40}) scale(.7)`:'scale(.62)'},g);const cleanup=renderer({group:animation,node,scene:this.model.scene,svg,width:p.w,height:p.h});if(typeof cleanup==='function')this.cleanups.push(cleanup);}
            catch(error){this.setRunning(false);this.say('Animation: '+error.message,true);}
          }else if(node.kind==='state'){
            const v=E.magnitude(node.value)||0;
            svg('line',{x1:-58,x2:58,y1:0,y2:0,stroke:'#c1d5e4','stroke-width':4},g);
            svg('circle',{cx:48*Math.tanh(v/5),cy:0,r:9,fill:'#237bab'},g);
            svg('text',{y:25,'text-anchor':'middle',class:'ex-secondary'},g,E.format(node.value).slice(0,28));
          }else{
            svg('text',{y:4,'text-anchor':'middle',class:'ex-secondary'},g,hasChildren?'Open':node.kind==='operation'?'Map':node.kind==='field'?'Field':'');
          }
          for(const port of [...p.inputs,...p.outputs]){
            const px=port.x-p.x,py=port.y-p.y,left=port.side==='in';
            if(node.kind==='state')svg('line',{x1:-64,x2:64,y1:py,y2:py,stroke:'#b9cedd','stroke-width':2},g);
            const mark=svg('g',{'data-port':port.key,'data-side':port.side,role:'button',tabindex:0,'aria-label':port.label+' '+port.side+' port; follow readout'},g);
            svg('circle',{cx:px,cy:py,r:10,fill:'transparent'},mark);
            svg('circle',{cx:px,cy:py,r:4,fill:'white',stroke:'#386b88','stroke-width':1.6},mark);
            svg('title',{},mark,port.label+' · '+(left?'input':'output'));
            const text=node.kind==='state'?E.format(port.wires[0].value):port.label;
            svg('text',{x:node.kind==='state'?0:px+(left?9:-9),y:py-6,'text-anchor':node.kind==='state'?'middle':left?'start':'end',class:'ex-port-label'},mark,text.length>16?text.slice(0,14)+'…':text);
            const selectPort=e=>{e.stopPropagation();this.focusSignal=port.wires[0].signal;this.focusWire=port.wires[0].branches[0].id;this.focusNode=null;this.render();};
            mark.onclick=selectPort;mark.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();selectPort(e);}};
          }
          svg('text',{y:-p.h/2-12,'text-anchor':'middle',class:'ex-part-label'},g,node.label);
          if(hasChildren)svg('text',{x:p.w/2-9,y:p.h/2-8,'text-anchor':'end',class:'ex-secondary'},g,this.model.children(node.id).length+' inside');
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
      this.renderDetails();
      if(focused){const el=[...this.canvas.querySelectorAll('[data-node]')].find(el=>el.getAttribute('data-node')===focused);el?.focus({preventScroll:true});}
    }
    renderDetails(){
      this.sidebar.replaceChildren();
      const n=this.model.nodes.get(this.focusNode),w=this.model.scene.wires.find(w=>w.id===this.focusWire);
      html('h2',{},this.sidebar,n?.label||w?.label||this.model.nodes.get(this.model.scope).label);
      if(n){
        if(n.description)html('p',{},this.sidebar,n.description);
        for(const [key,label] of [['value','State'],['configuration','Readout'],['inclination','Total inclination']])if(n[key]!==undefined){html('h3',{},this.sidebar,label);html('output',{},this.sidebar,E.format(n[key]));}
      }
      if(n||w)this.button(this.sidebar,'Show all wires',()=>{this.focusWire=null;this.focusSignal=null;this.focusNode=null;this.render();});
      let wires=[];
      if(w){const edge=this.geometry.edges.find(e=>e.branches.some(b=>b.id===w.id));wires=this.focusSignal?this.project.wires.filter(w=>E.signalKey(w)===this.focusSignal):edge?.branches||[w];}
      else if(n)wires=this.project.wires.filter(w=>w.source===n.id||w.target===n.id);
      if(!n&&!w){
        const description=this.model.nodes.get(this.model.scope)?.description;if(description)html('p',{},this.sidebar,description);
        html('p',{},this.sidebar,'Open a box to replace this view with its contents. Use the path above to come back out.');
        html('h3',{},this.sidebar,'At this level');
        for(const node of this.model.children())this.button(this.sidebar,node.label,()=>this.choose(node,{shiftKey:false}));
        html('p',{},this.sidebar,'Blue travels with the value; amber/red travels back with the load. A zero load has no moving marks. Each wire has its own fixed scale.');
        if(this.options.description)html('p',{},this.sidebar,this.options.description);
      }
      for(const wire of E.aggregate(wires)){
        const row=html('div',{class:'ex-wire-detail'},this.sidebar);html('strong',{},row,wire.label);
        if(this.focusSignal)html('p',{},row,'To '+(this.model.nodes.get(wire.target)?.label||'the boundary'));
        for(const [key,label] of [['value','Value'],['load',wire.inclination?'Inclination':'Returning load']]){
          const line=html('label',{},row,label+' ');html('output',{},line,E.format(wire[key]));
          html('div',{class:'ex-scale'},row,'High color at |'+key+'| ≥ '+(wire[key+'Scale']??1)+(wire[key+'Unit']?' '+wire[key+'Unit']:''));
        }
        if(wire.description)html('p',{},row,wire.description);
        if(wire.branches.length>1)html('p',{},row,'The '+wire.branches.length+' internal contributions add to this returning load.');
      }
    }
    destroy(){this.destroyed=true;cancelAnimationFrame(this.raf);if(this.zoomFrame)cancelAnimationFrame(this.zoomFrame);clearTimeout(this.zoomSaveTimer);this.persist();window.removeEventListener('pointermove',this.onDragMove);window.removeEventListener('pointerup',this.onDragEnd);window.removeEventListener('pointercancel',this.onDragCancel);for(const cleanup of this.cleanups)cleanup();this.cleanups=[];this.adapter.dispose?.();this.el.remove();}
  }
  E.View=View;
})(globalThis);
