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
    time=0;owed=0;playing=false;get("play").textContent="Play circuits";
    records=[];record();draw();text();
  }
  function record(){records.push({time,charge:bodies.map(b=>b.Q),current:bodies.map(b=>b.I),
    flux:bodies.map(b=>b.phi),energy:C.energy(bodies)});}
  function text(){
    get("time").textContent="time "+time.toFixed(2).padStart(7," ")+" s";
    get("energy").textContent="Total energy "+C.energy(bodies).toFixed(9)+" · started at 0.500000000";
  }
  function draw(){
    ctx.fillStyle=css("--panel");ctx.fillRect(0,0,cv.width,cv.height);
    const width=cv.width/bodies.length;
    bodies.forEach((b,i)=>{
      const x=(i+.5)*width,ink=css("--ink");
      ctx.strokeStyle=ink;ctx.lineWidth=3;ctx.beginPath();
      ctx.moveTo(x-90,105);ctx.lineTo(x-90,290);ctx.lineTo(x+90,290);ctx.lineTo(x+90,105);ctx.lineTo(x+20,105);
      ctx.moveTo(x-20,105);ctx.lineTo(x-90,105);ctx.stroke();
      // Capacitor plates and a coil in its own circuit: no wires between circuits.
      ctx.beginPath();ctx.moveTo(x-20,75);ctx.lineTo(x-20,135);ctx.moveTo(x+20,75);ctx.lineTo(x+20,135);ctx.stroke();
      ctx.fillStyle=css("--panel");ctx.fillRect(x-65,275,130,30);
      ctx.beginPath();for(let k=0;k<5;k++)ctx.arc(x-50+25*k,290,12.5,Math.PI,0);ctx.stroke();
      ctx.fillStyle=colors[i];ctx.fillRect(x-20,150,40,-65*b.Q);
      ctx.font="22px system-ui";ctx.textAlign="center";ctx.fillStyle=ink;
      ctx.fillText("Circuit "+(i+1)+(b.turns<0?" · reversed":""),x,40);
      ctx.font="20px monospace";ctx.fillStyle=colors[i];
      ctx.fillText("charge "+b.Q.toFixed(4),x,195);
      ctx.fillText("current "+b.I.toFixed(4),x,350);
      ctx.font="17px system-ui";ctx.fillStyle=css("--muted");
      ctx.fillText("capacitor",x,225);ctx.fillText("coil",x,385);
      ctx.fillText("position ("+b.x.toFixed(2)+", "+b.y.toFixed(2)+")",x,425);
    });
    hc.fillStyle=css("--panel");hc.fillRect(0,0,hist.width,hist.height);
    const span=Math.max(40,time),left=65,right=hist.width-25,y=125;
    hc.strokeStyle=css("--rule");hc.beginPath();hc.moveTo(left,y);hc.lineTo(right,y);hc.stroke();
    hc.font="18px system-ui";hc.fillStyle=css("--muted");hc.textAlign="left";
    hc.fillText("Capacitor charge",left,24);hc.fillText("0 s",left,235);
    hc.textAlign="right";hc.fillText(span.toFixed(1)+" s",right,235);
    bodies.forEach((b,i)=>{hc.strokeStyle=colors[i];hc.lineWidth=2;hc.beginPath();
      records.forEach((r,k)=>{const x=left+(right-left)*r.time/span,yy=y-85*r.charge[i];
        if(k===0)hc.moveTo(x,yy);else hc.lineTo(x,yy);});hc.stroke();});
  }
  get("play").onclick=()=>{playing=!playing;owed=0;get("play").textContent=playing?"Pause circuits":"Play circuits";text();draw();};
  get("reset").onclick=reset;
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
          playing=false;get("play").textContent="Play circuits";
          get("energy").textContent="The field exchange did not settle. Recharge to restart.";break;
        }
      }
    }
    if(now-lastText>100){draw();if(playing)text();lastText=now;}
    requestAnimationFrame(tick);
  }
  reset();requestAnimationFrame(tick);
})();
