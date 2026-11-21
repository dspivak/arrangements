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
  // A polyline from port to port is clear when its stretch between the two port leads is, as an S's is.
  const clearBetween=(pts,rects,width,height)=>clear(pts.slice(2,-1).map((q,i)=>line(pts[i+1],q)),rects,width,height);
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
      // A forward wire is a rounded step, whose stretches `separate` can move: every S between two columns would
      // be steepest at the same place, on top of each other, and an S cannot be moved off another wire's level
      // stretch. The S is kept for a wire whose step would not clear the parts.
      {const mx=start.x+arm,pts=[a,start,point(mx,start.y),point(mx,end.y),end,b];
        if(clearBetween(pts,rects,width,height))return {points:pts};}
      const candidate=[curve(start,point(start.x+arm,start.y),point(end.x-arm,end.y),end)];
      if(clear(candidate,rects,width,height))middle=candidate;
    }
    // A feedback wire first tries the book's symmetric, rounded U.
    if(!middle&&end.x<start.x){
      const crossed=rects.filter(r=>r.x+r.w/2>=end.x&&r.x-r.w/2<=start.x);
      const lanes=[Math.min(...crossed.map(r=>r.y-r.h/2))-42-lane,Math.max(...crossed.map(r=>r.y+r.h/2))+42+lane];
      lanes.sort((u,v)=>Math.abs(u-start.y)+Math.abs(u-end.y)-Math.abs(v-start.y)-Math.abs(v-end.y));
      // The U is a polyline, rounded by `soften` after `separate` has given it a lane of its own.
      for(const y of lanes){
        const pts=[a,start,point(start.x,y),point(end.x,y),end,b];
        if(clearBetween(pts,rects,width,height))return {points:pts};
      }
    }
    // Short paths can consist of one point. The port leads still form a wire.
    if(!middle)return {points:[a,...gridRoute(start,end,rects,width,height,lane),b]};
    return {segments:[line(a,start),...middle,line(end,b)]};
  }
  // Wires that share a channel are given lanes of their own, so that two wires meet only where they cross, at a
  // point, and never run on top of each other along a stretch (David, 2026-11-21, 66ce7547: "crossings should raise
  // codimension"): before this pass, 58% of the feeding explorer's wire length lay within 10 px of another wire
  // running the same way. Each routed wire is a polyline whose first and last stretches, at its ports, stay put.
  // A stretch between is moved across its own direction: upright stretches less than `gap` apart whose spans
  // overlap form a channel, and are spread `gap` apart within the free room around them (no part's rectangle,
  // and not past the wire's own ports). Their order across the channel is the one that crosses least: a stretch
  // is placed by which way the wire turns at its two ends. One turning back the same way at both ends (a U
  // opening left) goes nearest that side, the shorter one nearer; one turning left at its top and right at its
  // bottom goes further left the lower it lies, so that wires dropping to a column of ports nest without
  // crossing. Level stretches are then done the same way, turned a quarter.
  function separate(polylines,rects,width,height,gap=12){
    const simplify=pts=>{const out=[];
      for(const p of pts){if(out.length&&same(out.at(-1),p))continue;
        while(out.length>1){const a=out.at(-2),b=out.at(-1);if(Math.abs((b.x-a.x)*(p.y-b.y)-(b.y-a.y)*(p.x-b.x))>1e-6)break;out.pop();}
        out.push(point(p.x,p.y));}
      return out;};
    // A long first or last stretch keeps only 16 px at its port, so that branches leaving one fork, or wires
    // entering ports in a row, can part soon after: the rest of the stretch may move, joined to the stub by a jog.
    const stub=pts=>{if(pts.length<2)return pts;
      const cut=(p,q)=>{const d=Math.hypot(q.x-p.x,q.y-p.y);return d>48?point(p.x+(q.x-p.x)*16/d,p.y+(q.y-p.y)*16/d):null;};
      const head=cut(pts[0],pts[1]),out=[pts[0],...(head?[head,point(head.x,head.y)]:[]),...pts.slice(1,-1)];
      const tail=cut(pts.at(-1),pts.at(-2));
      return [...out,...(tail?[point(tail.x,tail.y),tail]:[]),pts.at(-1)];};
    const lines=polylines.map(simplify).map(stub),short={x:0,y:0};
    for(const axis of ['x','y','x','y']){
      // axis 'x': upright stretches, moved in x; axis 'y': level stretches, moved in y.
      const along=axis==='x'?'y':'x',items=[];
      lines.forEach((pts,w)=>{for(let k=1;k+2<pts.length;k++){const p=pts[k],q=pts[k+1];
        if(Math.abs(p[axis]-q[axis])>1e-6||Math.abs(p[along]-q[along])<1e-6)continue;
        const lo=Math.min(p[along],q[along]),hi=Math.max(p[along],q[along]);
        const [first,last]=p[along]<q[along]?[pts[k-1],pts[k+2]]:[pts[k+2],pts[k-1]];
        // which way the wire turns at the stretch's low end and at its high end, −1 or +1 across the channel
        // (an end joined to a stub has no turn yet; it takes the other end's, so that the stretch nests as a U)
        let dl=Math.sign(first[axis]-p[axis]),dh=Math.sign(last[axis]-p[axis]);
        if(!dl)dl=dh;if(!dh)dh=dl;
        // An upright stretch stays between the wire's points on either side of it, so that the level stretches
        // before and after it keep their directions and the wire never doubles back on itself. (A level stretch
        // moved past its neighbour only turns an upright the other way, a zigzag, not a loop.)
        // The bound is read from the wire as it stands when its channel is placed, since another of its stretches
        // may have moved since.
        const s0=Math.sign(p[axis]-pts[k-1][axis]),s1=Math.sign(q[axis]-pts[k+2][axis]);
        items.push({w,k,at:p[axis],lo,hi,dl,dh,get bound(){const b=[-Infinity,Infinity],now=lines[w];if(axis!=='x')return b;
          for(const [s,far] of [[s0,now[k-1]],[s1,now[k+2]]]){if(s>0)b[0]=Math.max(b[0],far[axis]+8);else if(s<0)b[1]=Math.min(b[1],far[axis]-8);}
          return b;}});}});
      const meets=(u,v)=>u.lo<v.hi-1&&v.lo<u.hi-1;
      const group=it=>it.dl<0&&it.dh<0?0:it.dl<0&&it.dh>0?1:it.dl>0&&it.dh<0?2:3;
      const order=(u,v)=>group(u)-group(v)||(group(u)===0?(u.hi-u.lo)-(v.hi-v.lo):group(u)===3?(v.hi-v.lo)-(u.hi-u.lo):
        group(u)===1?(v.lo+v.hi)-(u.lo+u.hi):(u.lo+u.hi)-(v.lo+v.hi))||u.w-v.w||u.k-v.k;
      // A channel's place: its wires in order, `gap` apart (less if the room is short), centred where they were,
      // inside the free room: nothing of a part's rectangle across the channel's span, the page's margins, and
      // each stretch kept short of its own wire's ports.
      function place(cluster){
        cluster.sort(order);
        const lo=Math.min(...cluster.map(c=>c.lo)),hi=Math.max(...cluster.map(c=>c.hi)),center=cluster.reduce((s,c)=>s+c.at,0)/cluster.length;
        const room=[axis==='x'?18:22,(axis==='x'?width-18:height-22)];
        for(const r of rects){const [rl,rh,al,ah]=axis==='x'?[r.x-r.w/2-14,r.x+r.w/2+14,r.y-r.h/2,r.y+r.h/2]:[r.y-r.h/2-14,r.y+r.h/2+14,r.x-r.w/2,r.x+r.w/2];
          if(ah<=lo||al>=hi)continue;if(rh<=center)room[0]=Math.max(room[0],rh);else if(rl>=center)room[1]=Math.min(room[1],rl);}
        const free=Math.max(0,room[1]-room[0]),step=cluster.length<2?gap:Math.min(gap,free/(cluster.length+1)),span=step*(cluster.length-1);
        const begin=Math.min(room[1]-step/2-span,Math.max(room[0]+step/2,center-span/2));
        // Each stretch is then held between its own wire's neighbouring points: pushed on past its neighbours
        // in the channel where it must be, but never so that its wire doubles back.
        const to=cluster.map((c,i)=>free?begin+i*step:c.at);
        for(let i=1;i<to.length;i++)to[i]=Math.max(to[i],Math.min(to[i-1]+step,cluster[i].bound[1]));
        for(let i=to.length-2;i>=0;i--)to[i]=Math.min(to[i],Math.max(to[i+1]-step,cluster[i].bound[0]));
        cluster.forEach((c,i)=>{c.to=Math.min(c.bound[1],Math.max(c.bound[0],to[i]));});        return {cluster,step:free?step:gap,lack:cluster.length<2?0:(cluster.length+1)*gap-free};
      }
      // Channels first by where the stretches lie; then any two channels whose placed wires come closer than a
      // gap, across spans that meet, are one channel, placed again, until no two are.
      items.sort((u,v)=>u.at-v.at);
      let channels=[];const done=new Set();
      for(const seed of items){if(done.has(seed))continue;const cluster=[seed];done.add(seed);
        for(let grew=true;grew;){grew=false;
          for(const it of items)if(!done.has(it)&&!cluster.some(c=>c.w===it.w)&&cluster.some(c=>Math.abs(c.at-it.at)<gap&&meets(c,it))){cluster.push(it);done.add(it);grew=true;}}
        channels.push(place(cluster));}
      for(let merged=true;merged;){merged=false;
        outer:for(let i=0;i<channels.length;i++)for(let j=i+1;j<channels.length;j++){
          const A=channels[i],B=channels[j],near=Math.min(A.step,B.step,gap)-.5;
          // (two stretches of one wire never share a channel: each is held by the other's place)
          if(!A.cluster.some(u=>B.cluster.some(v=>u.w===v.w))&&A.cluster.some(u=>B.cluster.some(v=>meets(u,v)&&Math.abs(u.to-v.to)<near))){
            channels.splice(j,1);channels[i]=place([...A.cluster,...B.cluster]);merged=true;break outer;}}}
      for(const channel of channels){
        // placed again as the wires now stand, then written, one channel at a time
        const {cluster,lack}=place(channel.cluster);
        // what the channel lacks of `gap` between each two of its wires and half a gap at each side
        short[axis]=Math.max(short[axis],lack);
        for(const c of cluster){const pts=lines[c.w];pts[c.k][axis]=c.to;pts[c.k+1][axis]=c.to;}      }
    }
    return {lines,short};
  }
  // The rows of a spring's terminals, and how far down its bend each wire has turned where it meets the resistor.
  const SPRING_ROWS=46,SPRING_TURN=9;
  // A spring with a third terminal (`third`, the label of the wire that sets its stiffness) has that terminal
  // on a row of its own between its two ends, where its wire meets the resistor from the side. A spring to an
  // anchor (`anchor`) has one end; the row below it is the anchor's, which has no port.
  const springRows=p=>p.inputs.length+(p.anchor?1:0),springPitch=p=>p.third?32:SPRING_ROWS;
  // Hoses. A wire of dimension n > 1 (explorer-core.js, `dimension`) is n wires bundled. At each of its two
  // ends it is either wrapped, one hose port on the part, or unwrapped, n ports on the part, one per number,
  // whose strands run to the hose's mouth a little way off. A list of at most `options.hoseAbove` numbers
  // (4 by default) is unwrapped until the viewer wraps it, a longer one wrapped until the viewer unwraps
  // it; `options.toggled` lists the ends the viewer has changed, each by its `hoseEnd` key. A boundary port
  // and a spring's terminal stay wrapped. Unwrapped, the part has a `group` for the end: its n ports, and
  // the mouth's place (x, y), where the hose begins or ends.
  const hoseEnd=(node,side,key)=>JSON.stringify([node,side,key]);
  const endAt=(p,side,key)=>p.groups.find(g=>g.side===side&&g.key===key)||p[side].find(v=>v.key===key);
  function layout(project,options={}){
    const edges=E.aggregate(project.wires).map(w=>({...w,signal:E.signalKey(w)}));
    const toggled=new Set(options.toggled||[]),hoseAbove=options.hoseAbove??4;
    const positions=new Map(),inner=project.nodes.filter(n=>n.kind!=='boundary'),boundaries=project.nodes.filter(n=>n.kind==='boundary');
    for(const n of project.nodes)positions.set(n.id,{id:n.id,kind:n.kind,glyph:n.glyph,third:n.third,anchor:n.anchor,inputs:[],outputs:[],groups:[],x:0,y:0,w:0,h:0});
    for(const w of edges){
      const a=positions.get(w.source),b=positions.get(w.target);w.outKey=w.signal;w.inKey=w.targetPort??w.signal;
      w.dimension=E.dimension(w);
      for(const [p,side,key] of [[a,'outputs',w.outKey],[b,'inputs',w.inKey]]){
        const end=hoseEnd(p.id,side,key),fixed=p.kind==='boundary'||p.glyph==='spring';
        if(w.dimension>1&&!fixed&&(w.dimension>hoseAbove)===toggled.has(end)){
          let group=p.groups.find(g=>g.side===side&&g.key===key);
          if(!group){
            group={key,side,end,label:w.label,dimension:w.dimension,wires:[],ports:[]};p.groups.push(group);
            for(let i=0;i<w.dimension;i++){
              const port={key:key+'#'+i,label:w.label+' '+E.coordinate(w,i),wires:group.wires,group,index:i};
              group.ports.push(port);p[side].push(port);
            }
          }
          group.wires.push(w);continue;
        }
        let port=p[side].find(v=>v.key===key);if(!port){port={key,label:w.label,wires:[]};p[side].push(port);}port.wires.push(w);
        if(w.dimension>1){port.hose=w.dimension;if(!fixed)port.end=end;}
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
        // A terminal of a field with a written 1-form carries, after its name, the letter the formula uses.
        // (A strand's letter is its hose's, with the strand's number under it.)
        const lettered=port=>{const letter=(n.form||n.law)?.letters?.[port.group?port.group.label:port.label];
          return port.label.length+(letter?letter.replace('_','').length+(port.group?4:2):0);};
        const labelWidth=ports=>ports.reduce((width,port)=>
          Math.max(width,lettered(port)*6.7+18),0);
        const left=labelWidth(p.inputs),right=labelWidth(p.outputs);
        p.w=Math.max(left+2*p.radius+right,(n.label??n.id).length*7.8+8);
        p.glyphX=(left-right)/2;
        p.h=Math.max(2*p.radius+10,count*20+12);
        // A spring is a bend joining its wires, with a resistor on the bend's upright stretch: its terminals
        // stand further apart, so that the resistor between them can be seen.
        if(n.glyph==='spring')p.h=(springRows(p)-1)*springPitch(p)+40;
      }
      if(n.kind==='state')p.h=Math.max(132,100+count*24);
      // How far from the part an unwrapped hose's mouth stands: further for a taller fan of strands.
      const pitch=n.kind==='state'?24:p.radius?20:22;
      for(const group of p.groups)group.reach=Math.min(90,40+(group.dimension-1)*pitch/6);
    }
    const reach=(n,side)=>Math.max(0,...positions.get(n.id).groups.filter(g=>g.side===side).map(g=>g.reach));
    const columns=[...new Set(inner.map(n=>column.get(n.id)))].sort((a,b)=>a-b),groups=columns.map(c=>inner.filter(n=>column.get(n.id)===c).sort((a,b)=>(a.row??0)-(b.row??0)));
    // The parts are placed with `rowGap` between two parts of a column and `colGap` between columns, at first 68 and
    // 148; when a channel between them has too little room for its wires to lie `separate`'s gap apart, the gaps
    // grow by what the most crowded channel lacked, and the parts are placed again, at most three more times.
    // The gaps are found with every part where the layout puts it; the viewer's own placements, saved against
    // those gaps, are then applied once.
    let rowGap=68,colGap=148,placed;
    const moved=options.placements&&Object.keys(options.placements).length>0;
    for(let round=0;round<4;round++){
      placed=place(rowGap,colGap,false);
      if(placed.short.x<=1&&placed.short.y<=1)break;
      if(round===3)break;
      rowGap=Math.min(68+600,rowGap+Math.ceil(placed.short.y));colGap=Math.min(148+600,colGap+Math.ceil(placed.short.x));
    }
    if(moved)placed=place(rowGap,colGap,true);
    placed.result.gaps={row:rowGap,column:colGap,short:placed.short};
    return placed.result;
    function place(rowGap,colGap,saved){
    const heights=groups.map(ns=>ns.reduce((s,n)=>s+positions.get(n.id).h+rowGap,0)-rowGap);
    const incoming=boundaries.filter(n=>n.side==='in'||(!n.side&&positions.get(n.id).outputs.length>0)),outgoing=boundaries.filter(n=>!incoming.includes(n));
    let height=Math.max(270,Math.max(0,...heights)+152,Math.max(incoming.length,outgoing.length)*26+140);
    // The margins, between the boundary ports and the first and last columns, grow with the gap between columns.
    let x=colGap+12;
    for(const ns of groups){const width=Math.max(...ns.map(n=>positions.get(n.id).w));let y=(height-heights[groups.indexOf(ns)])/2;
      // A column leaves room on each side for the fans of its parts' unwrapped hoses.
      x+=Math.max(...ns.map(n=>reach(n,'inputs')));
      for(const n of ns){const p=positions.get(n.id);p.x=x+width/2;p.y=y+p.h/2;y+=p.h+rowGap;}x+=width+colGap+Math.max(...ns.map(n=>reach(n,'outputs')));
    }
    let width=Math.max(420,x+12);
    for(const n of inner){const p=positions.get(n.id);p.orderY=p.y;
      const moved=saved&&options.placements?.[n.id];if(!moved)continue;
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
          // The strands of one unwrapped hose share their wires, so they stand together, in the list's order.
          const name=port=>port.group?port.group.label:port.label;
          return ordinate(a)-ordinate(b)||name(a).localeCompare(name(b))||(a.index??-1)-(b.index??-1);
        });
        // A spring's third terminal stands between its two ends, wherever its wire comes from.
        if(p.glyph==='spring'&&p.third&&side==='in'){
          const k=ports.findIndex(port=>port.label===p.third);
          if(k>=0)ports.splice(1,0,...ports.splice(k,1));
        }
        ports.forEach((port,i)=>{
          let dy=(i-(ports.length-1)/2)*22,dx=p.w/2;
          if(p.kind==='boundary'){dy=0;dx=0;}
          else if(p.kind==='state'){dy=-p.h/2+100+i*24;dx=p.w/2;}  // the port sits on the store's blue box
          else if(p.radius){
            dy=(i-(ports.length-1)/2)*20;
            const angle=ports.length===1?0:((i/(ports.length-1))-.5)*1.9;
            port.glyphX=p.glyphX+(side==='in'?-1:1)*p.radius*Math.cos(angle);
            port.glyphY=p.radius*Math.sin(angle);
            // A spring's wire runs level to the bend and turns into its upright stretch, where it ends.
            if(p.glyph==='spring'){
              const middle=(springRows(p)-1)/2;
              dy=(i-middle)*springPitch(p);
              port.glyphX=p.glyphX;port.glyphY=dy+Math.sign(middle-i)*SPRING_TURN;
              // The third terminal's wire runs level to the resistor and ends at its side.
              if(port.label===p.third){port.third=true;port.glyphX=p.glyphX-10;port.glyphY=dy;}
            }
          }
          port.side=side;port.x=p.x+(side==='in'?-dx:dx);port.y=p.y+dy;
        });
      }
      // An unwrapped hose's mouth stands level with the middle of its ports; each strand leaves its port
      // level and enters the mouth level.
      for(const group of p.groups){
        const out=group.side==='outputs',d=out?group.reach:-group.reach;
        group.x=group.ports[0].x+d;group.y=group.ports.reduce((sum,port)=>sum+port.y,0)/group.ports.length;
        for(const port of group.ports){
          const at=point(port.x,port.y),mouth=point(group.x,group.y);
          const bend=curve(at,point(at.x+d/2,at.y),point(mouth.x-d/2,mouth.y),mouth);
          port.strand=out?bend:curve(bend.b,bend.d,bend.c,bend.a);port.tracks=wireTracks([port.strand]);
        }
      }
    }
    for(const p of positions.values())anchors(p);
    // External ports align with the internal port they serve whenever possible.
    for(const ns of [incoming,outgoing]){
      const desired=n=>{const p=positions.get(n.id),ps=[...p.inputs,...p.outputs];
        const isInput=incoming.includes(n);
        const ys=ps.flatMap(port=>port.wires.map(w=>{const other=positions.get(isInput?w.target:w.source);return (isInput?endAt(other,'inputs',w.inKey):endAt(other,'outputs',w.outKey)).y;}));
        return ys.length?ys.reduce((a,b)=>a+b)/ys.length:p.y;};
      const ordered=ns.map(n=>({n,y:desired(n)})).sort((a,b)=>a.y-b.y);let last=56;
      for(const o of ordered){const p=positions.get(o.n.id);p.y=Math.max(o.y,last+26);last=p.y;}
      last=height-56;for(const o of ordered.reverse()){const p=positions.get(o.n.id);p.y=Math.min(p.y,last-26);last=p.y;anchors(p);}
    }
    // Include the title above each part in the obstacle, so a routed wire cannot
    // obscure its name. Vary corridor lanes to separate unrelated feedback wires.
    const rects=inner.map(n=>{const p=positions.get(n.id);return {...p,y:p.y-12,h:p.h+24};}),forks=[];
    // A shared readout forks just past its port; an unwrapped hose forks just past its mouth.
    for(const p of positions.values())for(const port of [...p.outputs.filter(port=>!port.group),...p.groups.filter(g=>g.side==='outputs')])if(port.wires.length>1){
      port.fork=point(port.x+19,port.y);
      const start=point(port.x,port.y),end=port.fork;
      forks.push({port,start,end,wires:port.wires,tracks:wireTracks([line(start,end)])});
    }
    const routes=edges.map((w,i)=>{const a=positions.get(w.source),b=positions.get(w.target),out=endAt(a,'outputs',w.outKey),input=endAt(b,'inputs',w.inKey);
      w.start=out.fork||point(out.x,out.y);w.end=point(input.x,input.y);
      return route(w.start,w.end,a,b,rects,width,height,(i%4)*7);});
    const polylines=routes.map(r=>r.points).filter(Boolean),{lines:lanes,short}=separate(polylines,rects,width,height);
    let next=0;
    edges.forEach((w,i)=>{w.segments=routes[i].segments||soften(lanes[next++]);w.path=path(w.segments);w.tracks=wireTracks(w.segments);});
    return {result:{positions,width,height,edges,forks},short};
    }
  }
  E.layout=layout;E.wheelZoom=wheelZoom;E.hoseEnd=hoseEnd;E.geometry={sample,path,inside,trackOffset,wireTracks,springRows:SPRING_ROWS,springTurn:SPRING_TURN};
  if(typeof module!=='undefined'&&module.exports)module.exports={layout,sample,path,inside,wheelZoom,hoseEnd,endAt};
})(globalThis);
