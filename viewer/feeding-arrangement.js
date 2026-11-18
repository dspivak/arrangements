/* A factored tangent arrangement in IMC, from feeding-imc.tex.
 * State readouts, stateless one-form terms and their summed inclinations are
 * evaluated together. The browser steps THIS field; feeding-model.js is an
 * independent reference. Internal state is never represented as a box.
 * Scalar readout wires have identity pullback; red/access maps have zero
 * downstream load here. Every other derivative is in the spring terms below.
 */
(function(root){
  'use strict';
  const F=typeof module!=='undefined'&&module.exports?require('./feeding-model.js'):root.Feeding;
  const NR=F.NR,NC=F.NC,theta=F.theta;
  if(NR!==10||NC!==7)throw Error('Feeding explorer expects the patrol/contact model (10 receiver and 7 casing states).');
  function derivative(t){if(t<=0||t>=1)return 0;const a=Math.exp(-1/t),b=Math.exp(-1/(1-t));return a*b*(1/t**2+1/(1-t)**2)/(a+b)**2;}
  function near(u,R){const t=(R*R-u*u)/(.75*R*R);return [theta(t),derivative(t)*(-2*u)/(.75*R*R)];}
  function evaluate(s,places,P=F.DEFAULTS,time=0,trace=true){
    const nodes=[],wires=[],qload=Array(s.length).fill(0),rate=Array(s.length).fill(0);
    const efforts=new Set();
    const node=(id,parent,label,kind='box',extra={})=>{
      if(kind==='effort'){
        efforts.add(id);
        if(trace){nodes.push({id,parent,label,kind:'box',...extra});nodes.push({id:id+'/field',parent:id,label:'Inclination field',kind:'field',description:'The stateless part’s own inclination field; no integrator.'});}
      }else if(trace)nodes.push({id,parent,label,kind,...extra});return id;
    };
    node('feeding',null,'Feeding');node('receiver','feeding','Receiver','box',{renderer:'receiver',column:2,row:Math.max(0,(places.length-1)/2)});
    node('battery','receiver','Battery','state',{value:s[0],renderer:'battery',column:0,row:1});
    node('wheels','receiver','Wheels','box',{renderer:'wheels',column:1,row:2});
    node('lever','receiver','Lever','box',{renderer:'lever',column:2,row:0});
    node('eye','receiver','Eye','box',{renderer:'eye',column:1,row:0});
    node('mouth','receiver','Mouth','box',{renderer:'mouth',column:2,row:2,description:'Opening and digestion are factored once per casing. Each opener acts on its named casing and charges the shared battery. It acts only while that casing is held at the mouth and the battery has energy.'});
    node('patrol','receiver','Patrol','box',{column:1,row:1});
    node('patrol-point','patrol','Patrol point','state',{value:s[8]});node('patrol-scan','patrol','Patrol scan','state',{value:s[9]});
    node('patrol-motor','patrol','Patrol motor','effort');node('patrol-scan-motor','patrol','Patrol scan motor','effort');
    const stateIds=['battery','grip','tip','cart','cart','wheel','look','scan','patrol-point','patrol-scan'];
    const labels=['Battery','Grip','Tip height','Cart position','Cart momentum','Wheel position','Look distance','Scan','Patrol point','Patrol scan'];
    node('grip','lever','Grip','state',{value:s[1]});node('tip','lever','Tip height','state',{value:s[2],renderer:'tip'});
    node('cart','wheels','Cart','state',{value:[s[3],s[4]],configuration:s[3],renderer:'cart'});
    node('dragger','wheels','Dragger','box',{renderer:'spring',stretch:s[3]-s[5]});
    node('wheel','dragger','Wheel','state',{value:s[5],renderer:'wheel'});
    node('axle','dragger','Axle spring','effort',{renderer:'spring',stretch:s[3]-s[5]});
    node('drive','wheels','Drive motor','effort');
    node('look','eye','Look distance','state',{value:s[6],renderer:'look'});node('scan','eye','Scan','state',{value:s[7]});
    node('vision','eye','Red at gaze','operation',{renderer:'vision'});
    node('look-motor','eye','Look motor','effort');node('scan-motor','eye','Scan motor','effort');
    node('gripper','lever','Grip motor','effort');node('lift','lever','Lift motor','effort');
    node('grip-rate','lever','Grip rate','operation');node('lift-rate','lever','Lift rate','operation');
    const scales=[10,2,.4,8,1,8,6,6,8,8];
    function wire(source,target,label,value,load,extra={}){
      if(efforts.has(source))source+='/field';if(efforts.has(target))target+='/field';
      if(trace)wires.push({id:source+'>'+target+':'+label,source,target,label,value,load,valueScale:1,loadScale:1,...extra});
    }
    function term(target,i,load=0){
      qload[i]+=load;
      wire(stateIds[i],target,labels[i],s[i],load,{valueScale:scales[i]||1,loadScale:i===0?1:i===1?.1:1,inclination:true});
    }
    const [b,l,h,y,my,zw,r,scan,eta,zeta]=s;
    const live=theta(b/P.b0),hold=theta(l),dhold=derivative(l);
    let seen=0,climb=0,K=0,Pl=0,Ph=0,Py=0,energy=b+my*my/(2*P.M)+.5*P.kappac*(y-zw)**2,loss=0;
    const data=[];
    for(let k=0;k<places.length;k++){
      const start=NR+NC*k,[f,a,x,mu,z,zg,za]=s.slice(start,start+NC),place=places[k],id='casing-'+k;
      node(id,'feeding','Casing '+(k+1),'box',{renderer:'casing',index:k,column:0,row:k});
      node(id+'/food',id,'Food','state',{value:f,renderer:'food'});node(id+'/opening',id,'Opening','state',{value:a});
      node(id+'/body',id,'Casing body','box',{renderer:'casing',index:k});
      node(id+'/height',id+'/body','Height','state',{value:[x,mu],configuration:x,renderer:'tip'});
      node(id+'/air',id+'/body','Air dragger','box',{renderer:'spring',stretch:x-z});
      node(id+'/memory',id+'/air','Air memory','state',{value:z});node(id+'/spring',id+'/air','Spring','effort',{renderer:'spring',stretch:x-z});
      node(id+'/ground',id+'/body','Ground','effort');node(id+'/gravity',id+'/body','Gravity','effort');
      node(id+'/ground-memory',id+'/body','Ground memory','state',{value:zg});
      node(id+'/red',id,'Red','operation',{renderer:'red',index:k});node(id+'/access',id,'Access','operation');node(id+'/place',id,'Place','operation',{value:place});
      stateIds.push(id+'/food',id+'/opening',id+'/height',id+'/height',id+'/memory',id+'/ground-memory',id+'/attachment-memory');
      labels.push('Food '+(k+1),'Opening '+(k+1),'Height '+(k+1),'Momentum '+(k+1),'Air memory '+(k+1),'Ground memory '+(k+1),'Attachment memory '+(k+1));scales.push(10,1,.4,1,.4,.4,.4);
      const red=theta((f-P.fr)/P.fr),access=theta((a-P.a1)/(P.a2-P.a1)),u=(y+r-place)/P.w, bell=Math.exp(-u*u);
      seen+=red*bell;climb+=red*(-2*u/P.w)*bell;
      const [nv,dnv]=near(x-h,P.R),[nh,dnh]=near(y-place,P.Rh),keep=theta((f-P.f0)/P.f0);
      K+=nv*nh*keep;
      const stretch=x-h,da=stretch-za,G=hold*nv*nh,W=P.kappa0*stretch*stretch+P.kd*da*da;
      const px=-G*(P.kappa0*stretch+P.kd*da)-.5*hold*dnv*nh*W;
      const pl=-.5*dhold*nv*nh*W,py=-.5*hold*nv*dnh*W;
      Pl+=pl;Ph-=px;Py+=py;
      const attBox='attachment-'+k,att=attBox+'/springs';node(attBox,'lever','Attachment '+(k+1),'box',{renderer:'spring',stretch});
      node(att,attBox,'Contact springs','effort',{renderer:'spring',stretch:da});node(id+'/attachment-memory',attBox,'Attachment memory','state',{value:za});
      term(att,1,pl);term(att,2,-px);term(att,3,py);term(att,start+2,px);wire(id+'/place',att,'Place '+(k+1),place,-py,{valueScale:6});
      term(att,start+6,G*P.kd*da-P.kr*za);
      term('grip-rate',start);term('grip-rate',start+2);wire(id+'/place','grip-rate','Place '+(k+1),place,0,{valueScale:6});
      term(id+'/red',start);term(id+'/access',start+1);
      wire(id+'/red','vision','Red '+(k+1),red,0);wire(id+'/place','vision','Place '+(k+1),place,0,{valueScale:6});
      const atMouth=live*near(x-P.x0,P.R)[0]*nh*hold,pry=P.lpry*atMouth*(1-a),eat=P.leat*atMouth*access*f;
      const opener='opener-'+k,eater='eater-'+k;
      node(opener,'mouth','Casing '+(k+1)+' opener','effort',{description:'Opening interaction with casing '+(k+1)+'. Its returning loads increase that casing’s opening and spend battery energy.'});
      node(eater,'mouth','Casing '+(k+1)+' digestion','effort');
      term(opener,start+1,pry);term(opener,0,-P.ctear*pry);term(opener,1);term(opener,3);term(opener,start+2);
      wire(id+'/place',opener,'Place '+(k+1),place,0,{valueScale:6});
      term(eater,start,-eat);term(eater,0,P.alpha*eat);term(eater,1);term(eater,3);term(eater,start+2);
      wire(id+'/place',eater,'Place '+(k+1),place,0,{valueScale:6});wire(id+'/access',eater,'Access '+(k+1),access,0);
      term(id+'/spring',start+2,-P.kappa*(x-z));term(id+'/spring',start+4,P.kappa*(x-z));term(id+'/gravity',start+2,-P.m*P.g);
      const v=x-P.xg,dg=v-zg,tg=theta(-v/P.eps),dtg=-derivative(-v/P.eps)/P.eps,td=theta(-v/P.epsd),dtd=-derivative(-v/P.epsd)/P.epsd;
      term(id+'/ground',start+2,-P.kg*(tg*v+.5*dtg*v*v)-P.kd*(td*dg+.5*dtd*dg*dg));
      term(id+'/ground',start+5,P.kd*td*dg-P.kr*zg);
      rate[start+2]=mu/P.m;
      energy+=f+mu*mu/(2*P.m)+P.m*P.g*x+.5*P.kg*tg*v*v+.5*G*W+.5*P.kappa*(x-z)**2+.5*P.kr*za*za+.5*P.kd*td*dg*dg+.5*P.kr*zg*zg;
      loss+=P.kappa**2/P.beta*(x-z)**2+P.ctear*pry+(1-P.alpha)*eat+(qload[start+5]**2+qload[start+6]**2)/P.bd;
      data.push({red,access,pry,eat,atMouth,px,pl,py});
    }
    const free=theta(1-2*seen),ring=P.A*P.A-r*r-scan*scan;
    const lookR=live*(free*(P.omega*scan+P.gam*ring*r)+P.lam*climb),lookS=live*free*(-P.omega*r+P.gam*ring*scan);
    const torque=live*(P.ktau*seen*r+P.kexp*free*(eta-y)),grip=P.lgrip*live*(K*(2-l)-(1-K)*(l+1)),lift=P.llift*live*(hold*(P.x0-h)+(1-hold)*(P.xg-h));
    const ringp=P.B*P.B-eta*eta-zeta*zeta,patE=live*free*(P.nup*zeta+P.gamp*ringp*eta),patZ=live*free*(-P.nup*eta+P.gamp*ringp*zeta);
    for(const [id,i,speed] of [['patrol-motor',8,patE],['patrol-scan-motor',9,patZ]]){
      term(id,8,i===8?P.cpat*speed:0);term(id,9,i===9?P.cpat*speed:0);term(id,0,-P.cpat*speed*speed);wire('vision',id,'Seen',seen,0);
    }
    term('vision',3);term('vision',6);
    for(const id of ['look-motor','scan-motor']){
      term(id,6,id==='look-motor'?P.csee*lookR:0);term(id,7,id==='scan-motor'?P.csee*lookS:0);
      term(id,0,-P.csee*(id==='look-motor'?lookR**2:lookS**2));
      wire('vision',id,'Seen',seen,0);wire('vision',id,'Climb',climb,0,{valueScale:3});
    }
    for(const i of [0,1,2,3])term('grip-rate',i);
    for(const i of [0,1,2])term('lift-rate',i);
    // Motors receive the actual other load, rather than differentiating their coefficients.
    const gripTorque=P.cgrip*grip-Pl,liftTorque=P.clift*lift-Ph;
    term('gripper',1,gripTorque);term('gripper',0,-gripTorque*grip);
    term('lift',2,liftTorque);term('lift',0,-liftTorque*lift);
    wire('grip-rate','gripper','Chosen grip rate',grip,0,{valueScale:4});wire('lift-rate','lift','Chosen lift rate',lift,0);
    // Expose computed other-load wires as values with zero returning load.
    // They are dependencies, not a force returning through a coefficient wire.
    for(let k=0;k<data.length;k++){
      wire('attachment-'+k+'/springs','gripper','Other grip load '+(k+1),data[k].pl,0,{valueScale:1});
      wire('attachment-'+k+'/springs','lift','Other tip load '+(k+1),-data[k].px,0,{valueScale:1});
    }
    const spring=P.kappac*(y-zw),wheel=(spring+torque)/P.betac;
    term('axle',3,-spring);term('axle',5,spring);term('drive',5,torque);term('drive',0,-torque*wheel);term('drive',6);
    term('drive',8);term('drive',3);
    wire('vision','drive','Seen',seen,0);wire('axle','drive','Other wheel load',spring,0);
    rate[0]=qload[0];rate[1]=qload[1]/P.cgrip;rate[2]=qload[2]/P.clift;rate[3]=my/P.M;rate[4]=qload[3];rate[5]=qload[5]/P.betac;
    // The existing page permits zero eye cost: the continuous limit has the
    // chosen rates and zero inclination/payment; do not divide zero by zero.
    rate[6]=P.csee===0?lookR:qload[6]/P.csee;rate[7]=P.csee===0?lookS:qload[7]/P.csee;
    rate[8]=P.cpat===0?patE:qload[8]/P.cpat;rate[9]=P.cpat===0?patZ:qload[9]/P.cpat;
    for(let k=0;k<places.length;k++){const i=NR+NC*k;rate[i]=qload[i];rate[i+1]=qload[i+1];rate[i+3]=qload[i+2];rate[i+4]=qload[i+4]/P.beta;rate[i+5]=qload[i+5]/P.bd;rate[i+6]=qload[i+6]/P.bd;}
    loss+=P.cgrip*grip*grip+P.clift*lift*lift+P.betac*wheel*wheel+P.csee*(lookR*lookR+lookS*lookS)+P.cpat*(patE*patE+patZ*patZ);
    node('battery-output','feeding','Battery out','boundary',{side:'out'});node('grip-output','feeding','Grip out','boundary',{side:'out'});
    wire('battery','battery-output','Battery',b,0,{valueScale:10});wire('grip','grip-output','Grip',l,0,{valueScale:2});
    if(trace)for(const n of nodes)if(n.kind==='state'){const i=stateIds.indexOf(n.id);n.inclination=qload[i];}
    return {rate,qload,energy,loss,seen,torque,live,hold,casings:data,version:1,title:'Feeding',root:'feeding',time,nodes,wires,state:s.slice(),places:places.slice()};
  }
  function rk4(s,h,places,P){
    const add=(a,b,c)=>a.map((v,i)=>v+c*b[i]),field=s=>evaluate(s,places,P,0,false).rate;
    const k1=field(s),k2=field(add(s,k1,h/2)),k3=field(add(s,k2,h/2)),k4=field(add(s,k3,h));
    return s.map((v,i)=>v+h*(k1[i]+2*k2[i]+2*k3[i]+k4[i])/6);
  }
  function adapter(options={}){
    const P={...F.DEFAULTS,...options.parameters},places=(options.places||F.PLACES).slice(),foods=(options.foods||F.FOODS).slice();
    if(places.length!==foods.length)throw Error('One food value is needed per casing');
    let state=F.start(places,foods,options.cart||0,P),time=0;
    return {snapshot:()=>evaluate(state,places,P,time),
      step:dt=>{if(!(dt>=0&&dt<=1))throw Error('Step duration must be between zero and one');let remaining=dt;while(remaining>1e-12){const h=Math.min(.002,remaining);state=rk4(state,h,places,P);time+=h;remaining-=h;}},
      reset:()=>{state=F.start(places,foods,options.cart||0,P);time=0;},
      setState:(s,t=0)=>{if(s.length!==NR+NC*places.length||!s.every(Number.isFinite))throw Error('Invalid state');state=s.slice();time=t;},parameters:P};
  }
  const api={evaluate,rk4,adapter,near,derivative};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.FeedingArrangement=api;
})(globalThis);
