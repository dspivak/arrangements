/* Geometry only: distinct left inputs/right outputs, shared readout forks,
 * horizontal port tangents, and obstacle-aware routes. No model state changes. */
(function(root){
  'use strict';
  const E=typeof module!=='undefined'&&module.exports?require('./explorer-core.js'):root.ArrangementExplorer;
  const point=(x,y)=>({x,y}),same=(a,b)=>Math.abs(a.x-b.x)+Math.abs(a.y-b.y)<1e-6;
  function wheelZoom(current,delta,mode=0){
    if(!Number.isFinite(delta))return current;
    const pixels=delta*(mode===1?16:mode===2?800:1);
    return Math.max(.25,Math.min(4,current*Math.exp(-pixels*.003)));
  }
  const line=(a,b)=>({a,b}),curve=(a,c,d,b)=>({a,c,d,b});
  const round=x=>Math.round(x*1000)/1000;
  const path=segments=>'M '+round(segments[0].a.x)+' '+round(segments[0].a.y)+segments.map(s=>s.c?
    ` C ${round(s.c.x)} ${round(s.c.y)} ${round(s.d.x)} ${round(s.d.y)} ${round(s.b.x)} ${round(s.b.y)}`:
    ` L ${round(s.b.x)} ${round(s.b.y)}`).join('');
  function sample(s,t){
    if(!s.c)return point(s.a.x+(s.b.x-s.a.x)*t,s.a.y+(s.b.y-s.a.y)*t);
    const u=1-t;return point(u*u*u*s.a.x+3*u*u*t*s.c.x+3*u*t*t*s.d.x+t*t*t*s.b.x,
      u*u*u*s.a.y+3*u*u*t*s.c.y+3*u*t*t*s.d.y+t*t*t*s.b.y);
  }
  // The two tracks meet at the port or fork. Between the ends they sit on
  // opposite sides of the route, including on vertical and returning bends.
  const trackOffset=4;
  function wireTracks(segments){
    const points=[segments[0].a];
    const distance=(a,b)=>Math.hypot(b.x-a.x,b.y-a.y);
    for(const segment of segments){
      const {a,b,c,d}=segment;
      if(c){
        const count=Math.max(1,Math.ceil((distance(a,c)+distance(c,d)+distance(d,b))/8));
        for(let i=1;i<count;i++)points.push(sample(segment,i/count));
      }else if(distance(a,b)>12){
        for(const t of [6/distance(a,b),1-6/distance(a,b)])points.push(sample(segment,t));
      }
      if(!same(points.at(-1),b))points.push(b);
    }
    const value=[],load=[];
    for(let i=0;i<points.length;i++){
      const p=points[i],prev=points[Math.max(0,i-1)],next=points[Math.min(points.length-1,i+1)];
      const length=distance(prev,next);
      const offset=i>0&&i<points.length-1&&length ? trackOffset/length : 0;
      const dx=(next.y-prev.y)*offset,dy=-(next.x-prev.x)*offset;
      const move=i?'L':'M';
      value.push(move+round(p.x+dx)+' '+round(p.y+dy));
      load.push(move+round(p.x-dx)+' '+round(p.y-dy));
    }
    return {value:value.join(' '),load:load.join(' ')};
  }
  const inside=(p,r,pad=0)=>p.x>r.x-r.w/2-pad+.01&&p.x<r.x+r.w/2+pad-.01&&p.y>r.y-r.h/2-pad+.01&&p.y<r.y+r.h/2+pad-.01;
  function clear(segments,rects,width,height){
    return segments.every(s=>Array.from({length:41},(_,i)=>sample(s,i/40)).every(p=>
      p.x>=18&&p.x<=width-18&&p.y>=22&&p.y<=height-22&&!rects.some(r=>inside(p,r,12))));
  }
  function soften(points){
    // Remove collinear grid points before making generous smooth turns.
    const pts=[];
    for(const p of points){if(pts.length&&same(pts.at(-1),p))continue;
      while(pts.length>1){const a=pts.at(-2),b=pts.at(-1);
        if(Math.abs((b.x-a.x)*(p.y-b.y)-(b.y-a.y)*(p.x-b.x))>1e-6)break;
        pts.pop();}pts.push(p);}
    const out=[];let start=pts[0];
    for(let i=1;i<pts.length-1;i++){
      const a=pts[i-1],b=pts[i],c=pts[i+1],u=Math.hypot(b.x-a.x,b.y-a.y),v=Math.hypot(c.x-b.x,c.y-b.y);
      const r=Math.min(36,u/2,v/2),before=point(b.x+(a.x-b.x)*r/u,b.y+(a.y-b.y)*r/u),after=point(b.x+(c.x-b.x)*r/v,b.y+(c.y-b.y)*r/v);
      if(!same(start,before))out.push(line(start,before));
      out.push(curve(before,point(before.x+(b.x-before.x)*.5523,before.y+(b.y-before.y)*.5523),
        point(after.x+(b.x-after.x)*.5523,after.y+(b.y-after.y)*.5523),after));start=after;
    }
    if(!same(start,pts.at(-1)))out.push(line(start,pts.at(-1)));
    return out;
  }
  function gridRoute(a,b,rects,width,height,lane){
    const sorted=xs=>[...new Set(xs)].sort((a,b)=>a-b);
    const clearance=34+lane;
    const xs=sorted([a.x,b.x,24,width-24,...rects.flatMap(r=>[r.x-r.w/2-clearance,r.x+r.w/2+clearance])].filter(x=>x>=18&&x<=width-18));
    const ys=sorted([a.y,b.y,38,height-38,...rects.flatMap(r=>[r.y-r.h/2-clearance,r.y+r.h/2+clearance])].filter(y=>y>=22&&y<=height-22));
    const id=(x,y)=>y*xs.length+x,coords=i=>point(xs[i%xs.length],ys[Math.floor(i/xs.length)]);
    const start=id(xs.indexOf(a.x),ys.indexOf(a.y)),end=id(xs.indexOf(b.x),ys.indexOf(b.y));
    const blocked=new Set();for(let j=0;j<ys.length;j++)for(let i=0;i<xs.length;i++)if(rects.some(r=>inside(point(xs[i],ys[j]),r,18)))blocked.add(id(i,j));
    const visible=(p,q)=>!rects.some(r=>p.y===q.y?
      p.y>r.y-r.h/2-18&&p.y<r.y+r.h/2+18&&Math.max(p.x,q.x)>r.x-r.w/2-18&&Math.min(p.x,q.x)<r.x+r.w/2+18:
      p.x>r.x-r.w/2-18&&p.x<r.x+r.w/2+18&&Math.max(p.y,q.y)>r.y-r.h/2-18&&Math.min(p.y,q.y)<r.y+r.h/2+18);
    const key=(i,d)=>i+':'+d,initial=key(start,0),best=new Map([[initial,0]]),previous=new Map();
    const queue=[{i:start,dir:0,cost:0,score:0,key:initial}];let finish;
    while(queue.length){queue.sort((a,b)=>b.score-a.score);const cur=queue.pop();if(cur.cost!==best.get(cur.key))continue;
      if(cur.i===end){finish=cur.key;break;}const p=coords(cur.i),x=cur.i%xs.length,y=Math.floor(cur.i/xs.length);
      for(const [nx,ny,dir] of [[x-1,y,1],[x+1,y,1],[x,y-1,2],[x,y+1,2]]){
        if(nx<0||nx>=xs.length||ny<0||ny>=ys.length)continue;const i=id(nx,ny),q=coords(i),k=key(i,dir);
        if(blocked.has(i)||!visible(p,q))continue;
        const cost=cur.cost+Math.abs(p.x-q.x)+Math.abs(p.y-q.y)+(cur.dir&&dir!==cur.dir?30:0);
        if(cost>=(best.get(k)??Infinity))continue;best.set(k,cost);previous.set(k,cur.key);
        queue.push({i,dir,cost,key:k,score:cost+Math.abs(q.x-b.x)+Math.abs(q.y-b.y)});
      }
    }
    if(!finish)throw Error('Could not route a wire between its ports');
    const pts=[];for(let k=finish;k;k=previous.get(k))pts.push(coords(+k.split(':')[0]));return pts.reverse();
  }
  function route(a,b,source,target,rects,width,height,lane){
    const start=point(Math.max(a.x+32,source.x+source.w/2+34),a.y);
    const end=point(Math.min(b.x-32,target.x-target.w/2-34),b.y);
    let middle;
    if(end.x>=start.x){const arm=(end.x-start.x)/2;
      const candidate=[curve(start,point(start.x+arm,start.y),point(end.x-arm,end.y),end)];
      if(clear(candidate,rects,width,height))middle=candidate;
    }
    // A feedback wire first tries the book's symmetric, rounded U.
    if(!middle&&end.x<start.x){
      const crossed=rects.filter(r=>r.x+r.w/2>=end.x&&r.x-r.w/2<=start.x);
      const lanes=[Math.min(...crossed.map(r=>r.y-r.h/2))-42-lane,Math.max(...crossed.map(r=>r.y+r.h/2))+42+lane];
      lanes.sort((u,v)=>Math.abs(u-start.y)+Math.abs(u-end.y)-Math.abs(v-start.y)-Math.abs(v-end.y));
      for(const y of lanes){const ar=Math.max(24,Math.abs(start.y-y)/3),br=Math.max(24,Math.abs(end.y-y)/3);
        const topA=point(start.x,y),topB=point(end.x,y),candidate=[
          curve(start,point(start.x+ar,start.y),point(start.x+ar,y),topA),line(topA,topB),
          curve(topB,point(end.x-br,y),point(end.x-br,end.y),end)];
        if(clear(candidate,rects,width,height)){middle=candidate;break;}
      }
    }
    if(!middle){
      const pts=gridRoute(start,end,rects,width,height,lane);
      return soften([a,...pts,b]);
      // Short paths can consist of one point. The port leads still form a wire.
    }
    return [line(a,start),...middle,line(end,b)];
  }
  function layout(project,options={}){
    const edges=E.aggregate(project.wires).map(w=>({...w,signal:E.signalKey(w)}));
    const positions=new Map(),inner=project.nodes.filter(n=>n.kind!=='boundary'),boundaries=project.nodes.filter(n=>n.kind==='boundary');
    for(const n of project.nodes)positions.set(n.id,{id:n.id,kind:n.kind,inputs:[],outputs:[],x:0,y:0,w:0,h:0});
    for(const w of edges){
      const a=positions.get(w.source),b=positions.get(w.target);w.outKey=w.signal;w.inKey=w.targetPort??w.signal;
      for(const [p,side,key] of [[a,'outputs',w.outKey],[b,'inputs',w.inKey]]){
        let port=p[side].find(v=>v.key===key);if(!port){port={key,label:w.label,wires:[]};p[side].push(port);}port.wires.push(w);
      }
    }
    const authored=inner.some(n=>n.column!=null)&&!inner.some(n=>n.userGroup);
    const column=new Map();
    if(authored){const cols=[...new Set(inner.map(n=>n.column??0))].sort((a,b)=>a-b);inner.forEach(n=>column.set(n.id,cols.indexOf(n.column??0)));}
    else{
      // Stable dependency ordering. Break a cycle at the vertex with most outgoing
      // dependencies; backward dependencies are routed as feedback, never reversed.
      const todo=inner.map(n=>n.id),placed=[];
      while(todo.length){let candidates=todo.filter(id=>!edges.some(e=>e.target===id&&todo.includes(e.source)&&e.source!==id));
        if(!candidates.length)candidates=[todo.slice().sort((a,b)=>edges.filter(e=>e.source===b).length-edges.filter(e=>e.source===a).length)[0]];
        for(const id of candidates){const before=edges.filter(e=>e.target===id&&placed.includes(e.source));
          column.set(id,before.length?Math.max(...before.map(e=>column.get(e.source)))+1:0);placed.push(id);todo.splice(todo.indexOf(id),1);}
      }
    }
    for(const n of inner){const p=positions.get(n.id),count=Math.max(p.inputs.length,p.outputs.length);
      p.column=column.get(n.id);p.w=n.kind==='state'?156:260;p.h=Math.max(132,(count+1)*22);
      if(n.kind==='box'){
        const labelWidth=ports=>ports.reduce((w,port)=>Math.max(w,Math.min(16,port.label.length)*6.7+16),0);
        p.w=Math.max((n.label??n.id).length*8+16,labelWidth(p.inputs)+labelWidth(p.outputs)+(n.renderer?90:48),90);
        p.h=Math.max(n.renderer?102:54,count*22+26);
      }
      if(n.kind==='operation'||n.kind==='field'){
        // The mark has a fixed size. Its labeled terminals occupy their own
        // rows. Adding an argument extends only that list of terminals.
        p.radius=n.renderer?34:15;
        const labelWidth=ports=>ports.reduce((width,port)=>
          Math.max(width,port.label.length*6.7+18),0);
        const left=labelWidth(p.inputs),right=labelWidth(p.outputs);
        p.w=Math.max(left+2*p.radius+right,(n.label??n.id).length*7.8+8);
        p.glyphX=(left-right)/2;
        p.h=Math.max(2*p.radius+10,count*20+12);
      }
      if(n.kind==='state')p.h=Math.max(132,100+count*24);
    }
    const columns=[...new Set(inner.map(n=>column.get(n.id)))].sort((a,b)=>a-b),groups=columns.map(c=>inner.filter(n=>column.get(n.id)===c).sort((a,b)=>(a.row??0)-(b.row??0)));
    const heights=groups.map(ns=>ns.reduce((s,n)=>s+positions.get(n.id).h+68,0)-68);
    const incoming=boundaries.filter(n=>n.side==='in'||(!n.side&&positions.get(n.id).outputs.length>0)),outgoing=boundaries.filter(n=>!incoming.includes(n));
    let height=Math.max(270,Math.max(0,...heights)+152,Math.max(incoming.length,outgoing.length)*26+140);
    let x=160;
    for(const ns of groups){const width=Math.max(...ns.map(n=>positions.get(n.id).w));let y=(height-heights[groups.indexOf(ns)])/2;
      for(const n of ns){const p=positions.get(n.id);p.x=x+width/2;p.y=y+p.h/2;y+=p.h+68;}x+=width+148;
    }
    let width=Math.max(420,x-148+160);
    for(const n of inner){const p=positions.get(n.id);p.orderY=p.y;
      const moved=options.placements?.[n.id];if(!moved)continue;
      if(!Number.isFinite(moved.x)||!Number.isFinite(moved.y))throw Error('Invalid saved position');
      p.x=Math.max(150+p.w/2,moved.x);p.y=Math.max(76+p.h/2,moved.y);
      width=Math.max(width,p.x+p.w/2+160);height=Math.max(height,p.y+p.h/2+110);
    }
    for(let i=0;i<inner.length;i++)for(let j=i+1;j<inner.length;j++){
      const a=positions.get(inner[i].id),b=positions.get(inner[j].id);
      if(Math.abs(a.x-b.x)<(a.w+b.w)/2+84&&Math.abs(a.y-b.y)<(a.h+b.h)/2+64)throw Error('Leave space between parts; the last clear position is kept.');
    }
    for(const [ns,bx] of [[incoming,50],[outgoing,width-50]])ns.forEach((n,i)=>Object.assign(positions.get(n.id),{x:bx,y:80+(i+1)*(height-160)/(ns.length+1)}));
    function anchors(p){
      for(const [ports,side] of [[p.inputs,'in'],[p.outputs,'out']]){
        ports.sort((a,b)=>{
          const ordinate=port=>port.wires.reduce((s,w)=>{const p=positions.get(side==='in'?w.source:w.target);return s+(p.orderY??p.y);},0)/port.wires.length;
          return ordinate(a)-ordinate(b)||a.label.localeCompare(b.label);
        });
        ports.forEach((port,i)=>{
          let dy=(i-(ports.length-1)/2)*22,dx=p.w/2;
          if(p.kind==='boundary'){dy=0;dx=0;}
          else if(p.kind==='state'){dy=-p.h/2+100+i*24;dx=p.w/2;}  // the port sits on the store's blue box
          else if(p.radius){
            dy=(i-(ports.length-1)/2)*20;
            const angle=ports.length===1?0:((i/(ports.length-1))-.5)*1.9;
            port.glyphX=p.glyphX+(side==='in'?-1:1)*p.radius*Math.cos(angle);
            port.glyphY=p.radius*Math.sin(angle);
          }
          port.side=side;port.x=p.x+(side==='in'?-dx:dx);port.y=p.y+dy;
        });
      }
    }
    for(const p of positions.values())anchors(p);
    // External ports align with the internal port they serve whenever possible.
    for(const ns of [incoming,outgoing]){
      const desired=n=>{const p=positions.get(n.id),ps=[...p.inputs,...p.outputs];
        const isInput=incoming.includes(n);
        const ys=ps.flatMap(port=>port.wires.map(w=>{const other=positions.get(isInput?w.target:w.source);return (isInput?other.inputs.find(v=>v.key===w.inKey):other.outputs.find(v=>v.key===w.outKey)).y;}));
        return ys.length?ys.reduce((a,b)=>a+b)/ys.length:p.y;};
      const ordered=ns.map(n=>({n,y:desired(n)})).sort((a,b)=>a.y-b.y);let last=56;
      for(const o of ordered){const p=positions.get(o.n.id);p.y=Math.max(o.y,last+26);last=p.y;}
      last=height-56;for(const o of ordered.reverse()){const p=positions.get(o.n.id);p.y=Math.min(p.y,last-26);last=p.y;anchors(p);}
    }
    // Include the title above each part in the obstacle, so a routed wire cannot
    // obscure its name. Vary corridor lanes to separate unrelated feedback wires.
    const rects=inner.map(n=>{const p=positions.get(n.id);return {...p,y:p.y-12,h:p.h+24};}),forks=[];
    for(const p of positions.values())for(const port of p.outputs)if(port.wires.length>1){
      port.fork=point(port.x+19,port.y);
      const start=point(port.x,port.y),end=port.fork;
      forks.push({port,start,end,wires:port.wires,tracks:wireTracks([line(start,end)])});
    }
    for(const w of edges){const a=positions.get(w.source),b=positions.get(w.target),out=a.outputs.find(p=>p.key===w.outKey),input=b.inputs.find(p=>p.key===w.inKey);
      w.start=out.fork||point(out.x,out.y);w.end=point(input.x,input.y);
      w.segments=route(w.start,w.end,a,b,rects,width,height,(edges.indexOf(w)%4)*7);w.path=path(w.segments);
      w.tracks=wireTracks(w.segments);
    }
    return {positions,width,height,edges,forks};
  }
  E.layout=layout;E.wheelZoom=wheelZoom;E.geometry={sample,path,inside,trackOffset,wireTracks};
  if(typeof module!=='undefined'&&module.exports)module.exports={layout,sample,path,inside,wheelZoom};
})(globalThis);
