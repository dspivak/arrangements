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
      const scene=adapter.snapshot();this.model=new E.Model(scene);
      this.storageKey=options.storageKey;
      if(this.storageKey)try{const s=JSON.parse(localStorage.getItem(this.storageKey));if(s)this.model.restore(s);}catch{}
      this.el=html('section',{class:'explorer','data-playing':'false'},container);
      const header=html('header',{class:'ex-header'},this.el),title=html('div',{},header);
      html('div',{class:'ex-kicker'},title,'Arrangement explorer');
      html('h1',{},title,scene.title);
      html('p',{class:'ex-subtitle'},title,'Follow values out and loads back. Open a box to see what is inside.');
      if(options.sceneURL)html('a',{href:options.sceneURL,class:'ex-link'},header,'Original animation');
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
      const workspace=html('div',{class:'ex-workspace'},this.el),canvasWrap=html('div',{class:'ex-canvas-wrap'},workspace);
      this.canvas=svg('svg',{class:'ex-canvas',role:'group','aria-label':'Arrangement diagram'},canvasWrap);
      this.sidebar=html('aside',{class:'ex-sidebar','aria-label':'Selected part or wire'},workspace);
      const legend=html('div',{class:'ex-legend'},this.el);
      for(const [label,kind] of [['Values forward','value'],['Loads back','load']]) {
        const span=html('span',{},legend,label+' · low');html('i',{class:'ex-ramp '+kind},span);span.append('high');
      }
      html('span',{},legend,'Color = magnitude · signed numbers on selection');
      html('p',{class:'ex-help'},this.el,'Click a box to enter. Shift-click parts, or turn on Select parts, to group them. Click a wire for its values, loads and scales. Unboxed moving marks are internal state.');
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
    act(fn){try{fn();this.persist();this.selection.clear();this.focusWire=null;this.render();}catch(error){this.say(error.message,true);}}
    persist(){if(this.storageKey)try{localStorage.setItem(this.storageKey,JSON.stringify(this.model.save()));}catch{}}
    refresh(){this.model.update(this.adapter.snapshot());for(const id of this.selection)if(!this.model.nodes.has(id))this.selection.delete(id);this.render();}
    enter(id){if(this.model.enter(id)){this.selection.clear();this.focusWire=null;this.focusNode=null;this.persist();this.render();}}
    choose(node,event){
      if((this.selectMode||event.shiftKey)&&node.kind!=='boundary'){
        if(this.selection.has(node.id))this.selection.delete(node.id);else this.selection.add(node.id);this.render();return;
      }
      if(this.model.children(node.id).length){this.enter(node.id);return;}
      this.focusNode=node.id;this.focusWire=node.wire||null;this.render();
    }
    layout(project){
      const inner=project.nodes.filter(n=>n.kind!=='boundary'),bounds=project.nodes.filter(n=>n.kind==='boundary');
      const authored=!this.model.nodes.get(this.model.scope)?.userGroup&&!inner.some(n=>n.userGroup);
      const columns=inner.length<=3?inner.length:Math.min(3,Math.ceil(Math.sqrt(inner.length)));
      const rows=Math.max(Math.ceil(inner.length/Math.max(1,columns)),...inner.map(n=>(authored?n.row??0:0)+1));
      const height=Math.max(530,rows*180+100,Math.ceil(bounds.length/2)*58+100),width=1100,positions=new Map();
      inner.forEach((n,i)=>{
        const column=authored&&n.column!=null?n.column:i%Math.max(1,columns),row=authored&&n.row!=null?n.row:Math.floor(i/Math.max(1,columns));
        const customColumns=authored&&inner.some(n=>n.column!=null)?3:columns;
        positions.set(n.id,{x:230+column*(640/Math.max(1,customColumns-1)),y:110+row*180,w:178,h:112});
      });
      const ins=bounds.filter(n=>n.side==='in'||(!n.side&&n.label.includes('inputs'))),outs=bounds.filter(n=>!ins.includes(n));
      for(const [list,x] of [[ins,45],[outs,1055]])list.forEach((n,i)=>positions.set(n.id,{x,y:60+(i+1)*(height-110)/(list.length+1),w:12,h:12}));
      return {positions,width,height};
    }
    render(){
      for(const cleanup of this.cleanups)cleanup();this.cleanups=[];
      const focused=document.activeElement?.getAttribute('data-node');
      const project=this.model.project(),{positions,width,height}=this.layout(project);
      this.project=project;this.positions=positions;
      this.time.textContent='Time '+E.format(this.model.scene.time??0);
      this.selectedLabel.textContent=this.selection.size?this.selection.size+' selected':'';
      this.groupButton.disabled=this.selection.size<2;this.ungroupButton.disabled=!this.model.nodes.get(this.model.scope)?.userGroup;this.undoButton.disabled=!this.model.history.length;
      this.crumbs.replaceChildren();
      this.model.ancestors(this.model.scope).forEach((n,i)=>{if(i)html('span',{},this.crumbs,'/');this.button(this.crumbs,n.label,()=>this.enter(n.id));});
      this.canvas.replaceChildren();this.canvas.setAttribute('viewBox',`0 0 ${width} ${height}`);
      const wireLayer=svg('g',{},this.canvas);
      const drawnBundles=project.bundles.flatMap(b=>this.style==='wiring'?E.aggregate(b.wires).map((w,i,all)=>({...b,wires:w.branches,lane:(i-(all.length-1)/2)*17,label:w.label})): [b]);
      for(const bundle of drawnBundles){
        const a=positions.get(bundle.source),b=positions.get(bundle.target);if(!a||!b)continue;
        const direction=b.x>=a.x?1:-1, ax=a.x+direction*a.w/2,bx=b.x-direction*b.w/2;
        const same=Math.abs(a.x-b.x)<30;
        const lane=bundle.lane||0;
        const path=offset=>same?`M ${ax} ${a.y+offset+lane} C ${ax+125} ${a.y+offset+lane}, ${bx+125} ${b.y+offset+lane}, ${bx} ${b.y+offset+lane}`:
          `M ${ax} ${a.y+offset+lane} C ${ax+(bx-ax)*.48} ${a.y+offset+lane}, ${ax+(bx-ax)*.52} ${b.y+offset+lane}, ${bx} ${b.y+offset+lane}`;
        const selected=bundle.wires.some(w=>w.id===this.focusWire),observations=E.aggregate(bundle.wires);
        const g=svg('g',{role:'button',tabindex:0,'aria-label':bundle.wires.map(w=>w.label).join(', ')+'; inspect values and loads',opacity:(this.focusWire&&!selected)?0.18:1},wireLayer);
        for(const [kind,offset] of [['value',-4],['load',4]]){
          const sorted=observations.slice().sort((u,v)=>(E.intensity(v[kind],v[kind+'Scale'])??-1)-(E.intensity(u[kind],u[kind+'Scale'])??-1));
          const w=sorted[0],amount=E.magnitude(w[kind]);
          svg('path',{d:path(offset),fill:'none',stroke:E.color(w[kind],w[kind+'Scale'],kind),'stroke-width':selected?4:2.8,opacity:selected?1:.72},g);
          if(amount>1e-10)svg('path',{d:path(offset),fill:'none',stroke:E.color(w[kind],w[kind+'Scale'],kind),'stroke-width':5,'stroke-linecap':'round','stroke-dasharray':'1 59',class:'ex-pulse'+(kind==='load'?' ex-back':''),'stroke-dashoffset':-(this.model.scene.time||0)*20*(kind==='load'?-1:1)},g);
          const x=same?ax+90:(ax+bx)/2,y=(a.y+b.y)/2+offset+lane;
          const sign=kind==='value'?direction:-direction;
          svg('path',{d:`M ${x-sign*5} ${y-4} L ${x+sign*3} ${y} L ${x-sign*5} ${y+4}`,fill:'none',stroke:E.color(w[kind],w[kind+'Scale'],kind),'stroke-width':2},g);
        }
        svg('path',{d:path(0),fill:'none',stroke:'transparent','stroke-width':this.style==='wiring'?12:22},g);
        if(bundle.label)svg('text',{x:(ax+bx)/2,y:(a.y+b.y)/2+lane-6,'text-anchor':'middle',class:'ex-secondary',stroke:'white','stroke-width':3,'paint-order':'stroke'},g,bundle.label);
        if(observations.length>1)svg('text',{x:(ax+bx)/2,y:(a.y+b.y)/2-14,'text-anchor':'middle',class:'ex-secondary'},g,observations.length+' values');
        const select=()=>{this.focusWire=bundle.wires[0].id;this.focusNode=null;this.render();};g.onclick=select;g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();select();}};
      }
      for(const node of project.nodes){
        const p=positions.get(node.id),selected=this.selection.has(node.id),hasChildren=this.model.children(node.id).length;
        const g=svg('g',{transform:`translate(${p.x},${p.y})`,class:'ex-node',tabindex:0,role:'button','data-node':node.id,
          'aria-label':node.label+(hasChildren?' — open':' — inspect')+(selected?' — selected':'')},this.canvas);
        if(node.kind==='boundary'){
          svg('circle',{r:6,fill:'#fff',stroke:'#76889b','stroke-width':2},g);
          const left=p.x<width/2;
          const label=node.label.length>24?node.label.slice(0,22)+'…':node.label;
          svg('text',{x:left?12:-12,y:-13,'text-anchor':left?'start':'end',class:'ex-secondary'},g,label);
        }else{
          if(node.kind==='box')svg('rect',{x:-p.w/2,y:-p.h/2,width:p.w,height:p.h,rx:this.style==='wiring'?0:12,fill:selected?'#e0effa':'#f5f9fc',stroke:selected?'#1474ab':'#a9becf','stroke-width':selected?3:1.5,class:'ex-focus-ring'},g);
          else if(node.kind==='operation')svg('ellipse',{cx:0,cy:-4,rx:67,ry:40,fill:selected?'#e0effa':'#eef1f4',stroke:selected?'#1474ab':'#c4cdd5','stroke-width':selected?3:1,class:'ex-focus-ring'},g);
          else if(node.kind==='field')svg('path',{d:'M -52 24 L 0 -28 L 52 24 Z',fill:'#f8e8dc',stroke:'#b77e55','stroke-width':selected?3:1,class:'ex-focus-ring'},g);
          else if(selected)svg('ellipse',{rx:85,ry:54,fill:'#e0effa',stroke:'#1474ab','stroke-dasharray':'4 4',class:'ex-focus-ring'},g);
          // State has no rectangular frame and no fictitious input port.
          const renderer=this.style==='illustrated'&&(this.renderers[node.renderer]||this.renderers[node.kind]);
          if(renderer){
            try{const cleanup=renderer({group:g,node,scene:this.model.scene,svg,width:p.w,height:p.h});if(typeof cleanup==='function')this.cleanups.push(cleanup);}
            catch(error){this.setRunning(false);this.say('Animation: '+error.message,true);}
          }else if(node.kind==='state'){
            const v=E.magnitude(node.value)||0;
            svg('line',{x1:-58,x2:58,y1:0,y2:0,stroke:'#c1d5e4','stroke-width':4},g);
            svg('circle',{cx:48*Math.tanh(v/5),cy:0,r:9,fill:'#237bab'},g);
            svg('text',{y:25,'text-anchor':'middle',class:'ex-secondary'},g,E.format(node.value).slice(0,28));
          }else{
            svg('text',{y:4,'text-anchor':'middle',class:'ex-secondary'},g,hasChildren?'Click to look inside':node.kind==='operation'?'Static operation':'');
          }
          svg('text',{y:-p.h/2-12,'text-anchor':'middle',class:'ex-part-label'},g,node.label);
          if(hasChildren)svg('text',{x:p.w/2-9,y:p.h/2-8,'text-anchor':'end',class:'ex-secondary'},g,this.model.children(node.id).length+' inside');
          if(this.selectMode)svg('circle',{cx:-p.w/2+10,cy:-p.h/2+12,r:6,fill:selected?'#13669b':'white',stroke:'#13669b'},g);
        }
        g.onclick=e=>this.choose(node,e);g.onkeydown=e=>{if(e.key==='Enter'||e.key===' '){e.preventDefault();this.choose(node,e);}};
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
      if(n||w)this.button(this.sidebar,'Show all wires',()=>{this.focusWire=null;this.focusNode=null;this.render();});
      let wires=[];
      if(w){const bundle=this.project.bundles.find(b=>b.wires.some(x=>x.id===w.id));wires=bundle?.wires||[w];}
      else if(n)wires=this.project.wires.filter(w=>w.source===n.id||w.target===n.id);
      if(!n&&!w){
        html('p',{},this.sidebar,'Open a box to replace this view with its contents. Use the path above to come back out.');
        html('h3',{},this.sidebar,'At this level');
        for(const node of this.model.children())this.button(this.sidebar,node.label,()=>this.choose(node,{shiftKey:false}));
        html('p',{},this.sidebar,'Blue travels with the value; amber/red travels back with the load. A zero load has no moving marks. Each wire has its own fixed scale.');
        if(this.options.description)html('p',{},this.sidebar,this.options.description);
      }
      for(const wire of E.aggregate(wires)){
        const row=html('div',{class:'ex-wire-detail'},this.sidebar);html('strong',{},row,wire.label);
        for(const [key,label] of [['value','Value'],['load',wire.inclination?'Inclination':'Returning load']]){
          const line=html('label',{},row,label+' ');html('output',{},line,E.format(wire[key]));
          html('div',{class:'ex-scale'},row,'High color at |'+key+'| ≥ '+(wire[key+'Scale']??1)+(wire[key+'Unit']?' '+wire[key+'Unit']:''));
        }
        if(wire.description)html('p',{},row,wire.description);
        if(wire.branches.length>1)html('p',{},row,'The '+wire.branches.length+' internal contributions add to this returning load.');
      }
    }
    destroy(){this.destroyed=true;cancelAnimationFrame(this.raf);for(const cleanup of this.cleanups)cleanup();this.cleanups=[];this.adapter.dispose?.();this.el.remove();}
  }
  E.View=View;
})(globalThis);
