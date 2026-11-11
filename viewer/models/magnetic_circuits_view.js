// UI for the fixed-circuit experiment in magnets.html; model in magnetic_circuits.js.
(function() {
  "use strict";
  const C=window.magneticCircuitsModel, get=id=>document.getElementById("cp-"+id);
  const cv=get("canvas"),ctx=cv.getContext("2d"),hist=get("history"),hc=hist.getContext("2d");
  const colors=["#2a5db0","#bd4b32","#5e8a39"];
  const css=name=>getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  let bodies=[],time=0,playing=false,flip=1,last=0,owed=0,lastText=0,records=[];
  function number(id,lo,hi,fallback) {
    const v=Number(get(id).value);
    return Number.isFinite(v)?Math.min(hi,Math.max(lo,v)):fallback;
  }
  function reset() {
    const gap=number("gap",1.5,4,2);get("gap").value=gap;
    const places=[[-gap/2,0,0],[gap/2,0,0],[0,2.5,Math.PI/2]];
    bodies=places.slice(0,Number(get("count").value)).map((p,i)=>({x:p[0],y:p[1],th:p[2],
      L:1,C:1,h:.5,turns:i===1?flip:1,Q:i===0?1:0,phi:0,I:0}));
    time=0;owed=0;playing=false;get("play").textContent="Release capacitor 1";
    records=[];record();draw();text();arrangementPicture();
  }
  // The selected composite in the category of smooth-interface arrangements.
  // The round node is the static placed(n) field routing, not a dynamical part.
  function arrangementPicture(){
    const n=bodies.length,cy=n*240+140,height=n===1?350:cy+145;
    const label=(x,y,t,small=false)=>`<text x="${x}" y="${y}" text-anchor="middle"${small?' class="small"':''}>${t}</text>`;
    const wire=(d,arrow=false,attrs="")=>`<path class="wire" d="${d}"${arrow?' marker-end="url(#cp-forward)"':''} ${attrs}/>`;
    let svg=`<svg viewBox="0 0 1160 ${height}" role="img" aria-labelledby="cp-wd-title cp-wd-desc">
      <title id="cp-wd-title">${n} capacitor–coil circuit${n===1?'':'s'}</title>
      <desc id="cp-wd-desc">Each capacitor sends voltage to its own coil and receives current back.
      ${n===1?'The isolated coil receives zero external field.':'A static routing map expresses and sums the other coils’ fields for each receiving coil.'}</desc>
      <defs>
        <marker id="cp-forward" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="var(--value)"/></marker>
        <marker id="cp-back" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto"><path d="M0 0 L10 5 L0 10Z" fill="var(--load)"/></marker>
      </defs>
      <rect x="20" y="20" width="1120" height="${height-40}" rx="18" fill="var(--ground)" stroke="var(--rule)"/>
      <text x="48" y="55">${n} circuit${n===1?'':'s'} · positions held fixed</text>`;
    // Nested return paths remain disjoint. Every wire enters a box horizontally.
    for(let i=0;i<n;i++){
      const dy=i*240,field=180+dy,out=210+dy,voltage=240+dy;
      const others=bodies.map((_,j)=>j+1).filter(j=>j!==i+1);
      if(n>1){
        const portY=cy+35*(1-2*i/(n-1));
        const dx=Math.sqrt(70*70-(portY-cy)**2),left=80+40*i,right=1120-40*i;
        svg+=wire(`M900 ${out} H${right-20} Q${right} ${out} ${right} ${out+20} V${portY-20} Q${right} ${portY} ${right-20} ${portY} H${640+dx}`,true,`data-emitter="${i+1}"`);
        svg+=wire(`M${640-dx} ${portY} H${left+20} Q${left} ${portY} ${left} ${portY-20} V${field+20} Q${left} ${field} ${left+20} ${field} H700`,true,`data-receiver="${i+1}" data-sources="${others.join(',')}"`);
        svg+=label(475,field-18,`field from coil${others.length>1?'s':''} ${others.join(' + ')}`);
      }else{
        svg+=`<circle cx="560" cy="180" r="18" fill="var(--rule)" stroke="var(--muted)"/>`;
        svg+=label(560,186,'0')+label(530,145,'no external field',true)+wire('M578 180 H700',true,'data-receiver="1" data-sources=""');
        svg+=wire('M900 210 H1022',true)+`<circle cx="1040" cy="210" r="18" fill="var(--rule)" stroke="var(--muted)"/>`;
        svg+=label(1000,260,'field unused here',true);
      }
      svg+=`<g data-circuit="${i+1}">
        <rect class="box" x="260" y="${190+dy}" width="220" height="100" rx="10"/>
        ${label(370,222+dy,'Capacitor '+(i+1))}${label(370,251+dy,'stores charge',true)}
        ${wire(`M480 ${voltage} H700`,true)}
        ${label(580,voltage-16,'voltage')}
        <path d="M615 ${voltage} H545" stroke="var(--load)" stroke-width="3" marker-end="url(#cp-back)"/>
        <text x="580" y="${voltage+28}" text-anchor="middle" style="fill:var(--load)">−current</text>
        <rect class="box" x="700" y="${145+dy}" width="200" height="130" rx="10"/>
        ${label(800,182+dy,'Coil '+(i+1))}${label(800,214+dy,'stores total flux',true)}${label(800,242+dy,'reads out its field',true)}
        </g>`;
      if(n>1)svg+=label(965,out-14,'field '+(i+1),true);
    }
    if(n>1){
      svg+=`<circle cx="640" cy="${cy}" r="70" fill="var(--rule)" stroke="var(--muted)"/>`;
      svg+=label(640,cy-8,'place &amp; sum')+label(640,cy+18,'other fields');
      svg+=label(640,cy+105,'Express the other fields in each coil’s frame; add them. Store nothing.',true);
    }
    get('wiring').innerHTML=svg+'</svg>';
    get('wiring-summary').textContent=n===1?
      'One capacitor sends its voltage to its coil; current comes back through the same port and changes the stored charge. There is no other coil supplying an external field.':
      `${n===2?'Two':'Three'} capacitor–coil circuits form one closed arrangement. Each capacitor sends its voltage to its own coil and receives current back. Each coil receives ${n===2?'the other coil’s field':'the sum of the other two coils’ fields'}.`;
  }

  function record(){records.push({time,charge:bodies.map(b=>b.Q),current:bodies.map(b=>b.I),
    flux:bodies.map(b=>b.phi),energy:C.energy(bodies)});}
  function text(){
    get("time").textContent="time "+time.toFixed(2).padStart(7," ")+" model units";
    const capacitors=bodies.reduce((sum,b)=>sum+b.Q*b.Q/(2*b.C),0);
    get("energy").textContent="Energy: capacitors "+(100*capacitors/.5).toFixed(1)+"% · shared magnetic field "+(100*(C.energy(bodies)-capacitors)/.5).toFixed(1)+"% · total "+C.energy(bodies).toFixed(9);
    get("flip").disabled=bodies.length===1;
    if(bodies.length===1){
      const b=bodies[0],phase=((Math.atan2(b.I,b.Q)%(2*Math.PI))+2*Math.PI)%(2*Math.PI);
      const stage=Math.round(phase/(Math.PI/2))%4;
      const descriptions=["Charged capacitor; current is near zero. Its voltage starts the current.",
        "Nearly empty capacitor; current is near its maximum. Magnetic energy keeps it flowing, charging the opposite polarity.",
        "Oppositely charged capacitor; current is near zero and about to reverse.",
        "Nearly empty again; current flows the other way and restores the starting charges."];
      get("story").textContent=descriptions[stage]+" (Nearest quarter-cycle stage.)";
      return;
    }
    get("story").textContent=time===0?"Start: capacitor 1 holds all the energy. Capacitor 2 is empty. Release capacitor 1 to begin.":
      "The second capacitor now holds "+(100*bodies[1].Q*bodies[1].Q/bodies[1].C).toFixed(1)+"% of the starting energy, with no connecting wire.";
  }
  function fieldPicture(){
    const M=window.magnetsModel, sources=bodies.flatMap(b=>M.wiresOf({...b,kind:"magnet",I:b.turns*b.I}));
    const scale=bodies.length===3?80:100,cx=600,cy=bodies.length===3?300:220;
    const sx=x=>cx+x*scale,sy=y=>cy-y*scale;
    const x0=50,y0=65,nx=110,ny=31,dx=10,dy=10;
    const values=new Float64Array((nx+1)*(ny+1));
    for(let j=0;j<=ny;j++)for(let i=0;i<=nx;i++){
      const x=(x0+i*dx-cx)/scale,y=(cy-y0-j*dy)/scale;
      values[j*(nx+1)+i]=sources.reduce((sum,w)=>sum+w.I*M.g(x-w.x,y-w.y),0);
    }
    ctx.save();ctx.strokeStyle=css("--line");ctx.lineWidth=1.4;ctx.beginPath();
    for(let j=0;j<ny;j++)for(let i=0;i<nx;i++){
      const a=values[j*(nx+1)+i],b=values[j*(nx+1)+i+1],c=values[(j+1)*(nx+1)+i+1],d=values[(j+1)*(nx+1)+i];
      const lo=Math.min(a,b,c,d),hi=Math.max(a,b,c,d);if(!Number.isFinite(hi-lo)||hi-lo>.6)continue;
      for(let k=Math.ceil(lo/.018);k*.018<=hi;k++){
        const lev=k*.018+1e-10,pts=[],x=x0+i*dx,y=y0+j*dy;
        if((a<lev)!==(b<lev))pts.push([x+dx*(lev-a)/(b-a),y]);
        if((b<lev)!==(c<lev))pts.push([x+dx,y+dy*(lev-b)/(c-b)]);
        if((d<lev)!==(c<lev))pts.push([x+dx*(lev-d)/(c-d),y+dy]);
        if((a<lev)!==(d<lev))pts.push([x,y+dy*(lev-a)/(d-a)]);
        for(let k=0;k+1<pts.length;k+=2){ctx.moveTo(...pts[k]);ctx.lineTo(...pts[k+1]);}
      }
    }
    ctx.stroke();
    bodies.forEach((b,i)=>{
      const e=M.ends(b);ctx.strokeStyle=colors[i];ctx.lineWidth=5;ctx.beginPath();ctx.moveTo(sx(e[0].x),sy(e[0].y));ctx.lineTo(sx(e[1].x),sy(e[1].y));ctx.stroke();
      e.forEach((p,k)=>{
        const x=sx(p.x),y=sy(p.y),I=(k===0?1:-1)*b.turns*b.I;
        ctx.fillStyle=css("--panel");ctx.beginPath();ctx.arc(x,y,14,0,2*Math.PI);ctx.fill();ctx.stroke();
        if(Math.abs(I)>.001){
          if(I>0){ctx.fillStyle=colors[i];ctx.beginPath();ctx.arc(x,y,5,0,2*Math.PI);ctx.fill();}
          else {ctx.beginPath();ctx.moveTo(x-6,y-6);ctx.lineTo(x+6,y+6);ctx.moveTo(x-6,y+6);ctx.lineTo(x+6,y-6);ctx.stroke();}
        }
      });
      ctx.font="bold 20px system-ui";ctx.textAlign="center";ctx.fillStyle=colors[i];
      ctx.fillText("Coil "+(i+1),sx(b.x),sy(b.y)+42);
    });
    ctx.font="20px system-ui";ctx.textAlign="left";ctx.fillStyle=css("--ink");
    ctx.fillText(bodies.length===1?"1. One coil and its magnetic field":"1. The coils share a magnetic field",30,34);
    ctx.font="16px system-ui";ctx.fillStyle=css("--muted");ctx.fillText(bodies.length===1?"End-on view · watch one circuit oscillate on its own":"End-on view · no wire connects one circuit to another",30,58);
    if(time===0){ctx.textAlign="center";ctx.fillText("No current yet, so no magnetic field yet",600,365);}
    ctx.restore();
  }
  function along(points,distance){
    const lengths=points.slice(1).map((p,i)=>Math.hypot(p[0]-points[i][0],p[1]-points[i][1]));
    const total=lengths.reduce((a,b)=>a+b,0);let d=((distance%total)+total)%total;
    for(let i=0;i<lengths.length;i++){if(d<=lengths[i])return [points[i][0]+(points[i+1][0]-points[i][0])*d/lengths[i],points[i][1]+(points[i+1][1]-points[i][1])*d/lengths[i]];d-=lengths[i];}
    return points[0];
  }
  function wiringPicture(){
    ctx.textAlign="left";ctx.font="20px system-ui";ctx.fillStyle=css("--ink");
    ctx.fillText("2. Each coil is wired to its own capacitor",30,415);
    const width=cv.width/bodies.length;
    bodies.forEach((b,i)=>{
      const x=(i+.5)*width,top=520,bottom=675,left=x-120,right=x+120;
      const points=[[x-18,top],[left,top],[left,bottom],[x-80,bottom]];
      for(let k=0;k<=240;k++){const t=k/240;points.push([x-80+160*t,bottom+27*Math.sin(t*12*Math.PI)]);}
      points.push([right,bottom],[right,top],[x+18,top]);
      ctx.strokeStyle=colors[i];ctx.lineWidth=4;ctx.beginPath();points.forEach((p,k)=>k?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();
      // Large copper windings, with the endpoints still on the same closed wire.
      ctx.strokeStyle="#d49a60";ctx.lineWidth=5;ctx.beginPath();
      points.slice(3,245).forEach((p,k)=>k?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke();
      const passed=(i===0?1:0)-b.Q;
      const length=points.slice(1).reduce((sum,p,k)=>sum+Math.hypot(p[0]-points[k][0],p[1]-points[k][1]),0);
      ctx.fillStyle=colors[i];
      for(let k=0;k<16;k++){const p=along(points,k*length/16+passed*160);ctx.beginPath();ctx.arc(...p,4,0,2*Math.PI);ctx.fill();}
      // Capacitor charge is on its two plates, not drawn in the insulating gap.
      ctx.fillStyle=css("--panel");ctx.fillRect(x-30,top-40,60,80);
      ctx.strokeStyle=colors[i];ctx.lineWidth=6;ctx.beginPath();ctx.moveTo(x-18,top-32);ctx.lineTo(x-18,top+32);ctx.moveTo(x+18,top-32);ctx.lineTo(x+18,top+32);ctx.stroke();
      ctx.font="bold 27px system-ui";ctx.textAlign="center";ctx.fillStyle=colors[i];
      const sign=b.Q>=0?1:-1;
      if(Math.abs(b.Q)>.003){ctx.fillText(sign>0?"+":"−",x-47,top-12);ctx.fillText(sign>0?"−":"+",x+47,top-12);}
      ctx.font="bold 21px system-ui";ctx.fillText("Capacitor "+(i+1),x,472);
      ctx.font="19px monospace";ctx.fillText("charge "+signed(b.Q),x,578);
      ctx.font="20px system-ui";ctx.fillText("Coil "+(i+1)+(b.turns<0?" · reversed":""),x,735);
      ctx.font="19px monospace";ctx.fillText("current "+signed(b.I),x,765);
      ctx.fillStyle=css("--rule");ctx.fillRect(x-110,805,220,16);
      ctx.fillStyle=colors[i];ctx.fillRect(x-110,805,220*Math.min(1,b.Q*b.Q/b.C),16);
      ctx.font="16px system-ui";ctx.fillStyle=css("--muted");
      ctx.fillText("energy in capacitor "+(i+1)+": "+(100*b.Q*b.Q/b.C).toFixed(1)+"% of the start",x,847);
    });
  }
  function signed(v){return (v<0?"−":"+")+Math.abs(v).toFixed(4);}
  function draw(){
    ctx.fillStyle=css("--panel");ctx.fillRect(0,0,cv.width,cv.height);
    fieldPicture();
    ctx.strokeStyle=css("--rule");ctx.beginPath();ctx.moveTo(25,385);ctx.lineTo(cv.width-25,385);ctx.stroke();
    wiringPicture();
    hc.fillStyle=css("--panel");hc.fillRect(0,0,hist.width,hist.height);
    const span=Math.max(40,time),left=65,right=hist.width-25,y=125;
    hc.strokeStyle=css("--rule");hc.beginPath();hc.moveTo(left,y);hc.lineTo(right,y);hc.stroke();
    hc.font="18px system-ui";hc.fillStyle=css("--muted");hc.textAlign="left";
    hc.fillText("Charge on each capacitor (same colors as the coils)",left,24);hc.fillText("0 model units",left,235);
    hc.textAlign="right";hc.fillText(span.toFixed(1)+" model units",right,235);
    bodies.forEach((b,i)=>{hc.strokeStyle=colors[i];hc.lineWidth=2;hc.beginPath();
      records.forEach((r,k)=>{const x=left+(right-left)*r.time/span,yy=y-85*r.charge[i];
        if(k===0)hc.moveTo(x,yy);else hc.lineTo(x,yy);});hc.stroke();});
  }
  get("play").onclick=()=>{playing=!playing;owed=0;get("play").textContent=playing?"Pause circuits":"Release capacitor 1";text();draw();};
  get("reset").onclick=reset;
  get("quarter").onclick=()=>{
    if(bodies.length!==1){get("count").value="1";reset();}
    playing=false;owed=0;get("play").textContent="Release capacitor 1";
    const quarter=Math.PI/2,target=(Math.floor((time+1e-8)/quarter)+1)*quarter;
    const tau=number("tau",.01,.1,.05);
    while(time<target-1e-12){const dt=Math.min(tau,target-time);bodies=C.step(bodies,dt);time+=dt;record();}
    draw();text();
  };
  get("flip").onclick=()=>{flip=-flip;reset();};
  get("count").onchange=reset;get("gap").onchange=reset;
  get("tau").onchange=()=>{get("tau").value=number("tau",.01,.1,.05);owed=0;};
  get("download").onclick=()=>{
    const blob=new Blob([JSON.stringify({bodies,rounds:16,records},null,2)],{type:"application/json"});
    const url=URL.createObjectURL(blob),a=document.createElement("a");a.href=url;a.download="coil-circuit-steps.json";a.click();
    setTimeout(()=>URL.revokeObjectURL(url),1000);
  };
  function tick(now){
    const dt=last?Math.min((now-last)/1000,.1):0;last=now;
    if(playing){
      owed+=dt;const tau=number("tau",.01,.1,.05);
      while(owed>=tau){
        bodies=C.step(bodies,tau);time+=tau;owed-=tau;record();
        if(!Number.isFinite(C.energy(bodies)) || C.residual(bodies)>1e-6){
          playing=false;get("play").textContent="Release capacitor 1";
          get("energy").textContent="The field exchange did not settle. Recharge to restart.";break;
        }
      }
    }
    if(now-lastText>100){draw();if(playing)text();lastText=now;}
    requestAnimationFrame(tick);
  }
  reset();requestAnimationFrame(tick);
})();
