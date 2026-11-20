/* A factored tangent arrangement in IMC, from feeding-imc.tex.
 * State readouts, stateless one-form terms and their summed inclinations are
 * evaluated together. The patrol adapter steps this field. The learning observer reads
 * the current food-learning outing and its memory-motor rates. Internal state is never represented as a box.
 * Scalar readout wires have identity pullback; red/access maps have zero
 * downstream load here. Every other derivative is in the spring terms below.
 */
(function(root){
  'use strict';
  const F=typeof module!=='undefined'&&module.exports?require('./feeding-model.js'):root.Feeding;
  const Learning = typeof module !== 'undefined' && module.exports ? require('./food-learning.js') : root.FoodLearning;
  const Forms = typeof module !== 'undefined' && module.exports ? require('./explorer-forms.js') : root.ArrangementForms;
  const NR=F.NR,NC=F.NC,theta=F.theta;
  // The note's smooth step, for the formulas written on the field marks (explorer-forms.js).
  if(Forms)Forms.functions.θ=theta;
  if(Forms)Forms.functions.dθ=t=>derivative(t);
  if(NR!==10||NC!==8)throw Error('Feeding explorer expects the patrol model with external contact memory.');
  function derivative(t){if(t<=0||t>=1)return 0;const a=Math.exp(-1/t),b=Math.exp(-1/(1-t));return a*b*(1/t**2+1/(1-t)**2)/(a+b)**2;}
  function near(u,R){const t=(R*R-u*u)/(.75*R*R);return [theta(t),derivative(t)*(-2*u)/(.75*R*R)];}
  // A coefficient may act on a coordinate it never reads (gravity is the
  // simplest example). Preserve that returning load on the coordinate's wire.
  function separateFields(nodes,wires,places,learning,P){
    const incoming=new Map(), published=new Map(), byId=new Map(nodes.map(n=>[n.id,n]));
    for(const edge of wires){
      if(!incoming.has(edge.target))incoming.set(edge.target,new Map());
      incoming.get(edge.target).set(edge.source,edge);
    }
    const fields=new Set(), branches=[];
    const zero=value=>Array.isArray(value)?value.map(zero):0;
    function split(part,rows){
      const field=part+'/field', ports=incoming.get(field), results=new Map();
      fields.add(field);
      // A row's fifth entry is the mark's 1-form (explorer-forms.js): the explorer writes it on the mark and
      // checks it against the loads on the mark's wires, so it is the note's formula, not a second model.
      for(const [index,[on,name,args,output,form]] of rows.entries()){
        const target=ports.get(on), id=index?field+'-'+index:field;
        const result={id,parent:part,label:name,kind:'field',dependencyView:true,
          result:target.load,resultLabel:'Contribution to '+target.label.toLowerCase(),...(form?{form}:{})};
        if(index)nodes.push(result);else Object.assign(byId.get(field),result,{description:''});
        for(const key of new Set([...args,on])){
          const edge=ports.get(key)||results.get(key);
          branches.push({...edge,target:id,id:edge.source+'>'+id+':'+edge.label,
            load:key===on?target.load:zero(edge.value),argument:args.includes(key)});
        }
        results.set(name,{source:id,sourcePort:'coefficient',label:name,value:target.load,valueScale:target.loadScale,loadScale:1});
        if(output)published.set(field+'|'+output,id);
      }
    }
    function motor(part,coordinate,rate,other,name,forms=[]){
      split(part,[[coordinate,name,[rate,other],undefined,forms[0]],['battery','Battery payment',[name,rate],undefined,forms[1]]]);
    }
    // The lever's motors, the note's sec.gates: a torque τ = c u − P at the rate u pushes the coordinate and pays τ u.
    // The first mark's formula is written in the letters at its terminals (u and P), not in τ, which it gives.
    const lever=(cost,rateLabel,otherLabel,coordinate,q,load,c)=>{
      const form=(letters,terms)=>({letters,constants:{[c]:cost},terms});
      return [form({[rateLabel]:'u',[otherLabel]:load,[coordinate]:q},[{c:c+'*u - '+load,h:q}]),
        form({[rateLabel]:'u','Battery':'b'},[{c:'-τ*u',h:'b'}])];
    };
    const gripForms=lever(P.cgrip,'Chosen grip rate','Other grip load','Grip','ℓ','P_ℓ','c_grip');
    const liftForms=lever(P.clift,'Chosen lift rate','Other tip load','Tip height','h','P_h','c_lift');
    gripForms[1].letters={'Grip torque':'τ','Chosen grip rate':'u','Battery':'b'};
    liftForms[1].letters={'Lift torque':'τ','Chosen lift rate':'u','Battery':'b'};
    motor('gripper','grip','grip-rate','sum-grip-load','Grip torque',gripForms);
    motor('lift','tip','lift-rate','sum-tip-load','Lift torque',liftForms);
    // The mouth's two organs, each one push on the battery at the total rate its casings realize (pry, eat):
    // tearing costs c_tear per unit of opening, and digestion gives back the fraction α of the food taken.
    split('opener',[['battery','Opening cost',['sum-opened'],undefined,
      {letters:{'Opening rate':'pry','Battery':'b'},constants:{c_tear:P.ctear},terms:[{c:'-c_tear*pry',h:'b'}]}]]);
    split('digestor',[['battery','Energy from food',['sum-eaten'],undefined,
      {letters:{'Food taken':'eat','Battery':'b'},constants:{α:P.alpha},terms:[{c:'α*eat',h:'b'}]}]]);
    // The dragger's spring and each casing's air spring are not split into pushes: each is one spring mark
    // (evaluate, springMark), the cost of a difference, with equal and opposite loads on its two wires.
    // The drive, the note's motor (eqn.motor, eqn.cart) sent the torque τ_drive and the axle's load P_w on the
    // wheel: τ_drive dz_w − τ_drive (τ_drive + P_w)/β_w db, the wheel's rate being (τ_drive + P_w)/β_w. The torque
    // arrives on a wire from its static map (Drive torque), so each mark hears only what its formula names.
    split('drive',[['wheel','Motor inclination',['drive-torque'],undefined,
        {letters:{'Drive torque':'τ_drive','Wheel position':'z_w'},terms:[{c:'τ_drive',h:'z_w'}]}],
      ['battery','Battery payment',['drive-torque','wheel-load'],undefined,
        {letters:{'Drive torque':'τ_drive','Other wheel load':'P_w','Battery':'b'},constants:{β_w:P.betac},
          terms:[{c:'-τ_drive*(τ_drive + P_w)/β_w',h:'b'}]}]]);
    // A motor at a chosen rate u (the note's sec.general, after eqn.motor, with no other load on its store):
    // c u dq − c u² db. Each mark is wired to the rate, which it reads, and to the one wire it pushes; what the
    // rate itself reads enters the rate's static map, outside the motor.
    function atRate(part,store,storeLabel,q,rate,rateLabel,c,cost){
      const form=(letters,terms)=>({letters:{[rateLabel]:'u',...letters},constants:{[c]:cost},terms});
      split(part,[[store,'Motor inclination',[rate],undefined,form({[storeLabel]:q},[{c:c+'*u',h:q}])],
        ['battery','Battery payment',[rate],undefined,form({'Battery':'b'},[{c:'-'+c+'*u**2',h:'b'}])]]);
    }
    // The eye, the note's sec.gates: the sweep's two motors, each at its rate (Look rate, Scan rate).
    atRate('look-motor','look','Look distance','r','look-rate','Look rate','c_see',P.csee);
    atRate('scan-motor','scan','Scan','ψ','scan-rate','Scan rate','c_see',P.csee);
    if(learning){
      // A memory motor at its rate u costs the battery c_mem |u|²: c_mem u dq − c_mem u² db. A store of several
      // numbers has lists on its wires: its motor is that many side by side, and the battery pays their sum.
      for(const {id,label,start,end} of Learning.stores){
        const rateLabel=label+' rate',form=(letters,terms)=>({letters:{[rateLabel]:'u',...letters},
          constants:{c_mem:P.memoryCost},terms});
        split(id+'-motor',[[id,'Memory inclination',[id+'-rate'],undefined,form({[label]:'q'},[{c:'c_mem*u',h:'q'}])],
          ['battery','Battery payment',[id+'-rate'],undefined,form({'Battery':'b'},[{c:end===start+1?'-c_mem*u**2':'-c_mem*Σ(u**2)',h:'b'}])]]);
      }
    }else{
      atRate('patrol-motor','patrol-point','Patrol point','η','patrol-rate','Patrol point rate','c_pat',P.cpat);
      atRate('patrol-scan-motor','patrol-scan','Patrol scan','ζ','patrol-scan-rate','Patrol scan rate','c_pat',P.cpat);
    }
    for(let k=0;k<places.length;k++){
      const c='casing-'+k+'/', contact='attachment-'+k+'/', springs=contact+'springs';
      const height=c+'height', place=c+'place', latch=c+'latch', memory=c+'attachment-memory';
      const hx='Height '+(k+1),air='Air memory '+(k+1),gm='Ground memory '+(k+1);
      // Gravity pushes the height by -m g. The air spring, the cost κ (x - z)²/2, is one spring mark (evaluate).
      split(c+'gravity',[[height,'Weight',[],undefined,
        {letters:{[hx]:'x'},constants:{m:P.m,g:P.g},terms:[{c:'-m*g',h:'x'}]}]]);
      // The ground: a stiff floor under x_g (switched on by θ over the width ε) and a damper whose stroke is the ground
      // memory z_g; v is the height above the floor and δ the damper's stretch. Marks: the load on the height, then on z_g.
      const ground={constants:{k_g:P.kg,k_d:P.kd,k_r:P.kr,ε:P.eps,ε_d:P.epsd,x_g:P.xg},
        where:[['v','x - x_g'],['δ','v - z_g'],['tg','θ(-v/ε)'],['dtg','-dθ(-v/ε)/ε'],['td','θ(-v/ε_d)'],['dtd','-dθ(-v/ε_d)/ε_d']]};
      split(c+'ground',[[height,'Ground force',[height,c+'ground-memory'],undefined,
          {...ground,letters:{[hx]:'x',[gm]:'z_g'},terms:[{c:'-(k_g*(tg*v + dtg*v**2/2) + k_d*(td*δ + dtd*δ**2/2))',h:'x'}]}],
        [c+'ground-memory','Load on ground memory',[height,c+'ground-memory'],undefined,
          {...ground,letters:{[hx]:'x',[gm]:'z_g'},terms:[{c:'k_d*td*δ - k_r*z_g',h:'z_g'}]}]]);
      const positions=['in-tip',height,'in-cart',place], contactInputs=[...positions,'in-grip',latch,memory];
      // The contact spring is the cost U = ½ S(n) θ(ℓ) near(x - h) near_h(y - p) (κ_0 (x - h)² + k_d δ²) of the
      // latch n, the grip ℓ, the tip h, the casing height x, the cart y, the place p and the attachment memory
      // z_a, with δ = x - h - z_a. Each mark carries minus one partial of U (the memory's also its damper -k_r z_a).
      const lx={'Tip height':'h',[hx]:'x','Cart position':'y',['Place '+(k+1)]:'p','Grip':'ℓ',['Latch '+(k+1)]:'n',['Attachment memory '+(k+1)]:'z_a'};
      const cs={constants:{R:P.R,R_h:P.Rh,κ_0:P.kappa0,k_d:P.kd,k_r:P.kr},letters:lx,
        where:[['S','θ((n - 0.3)/0.2)'],['dS','dθ((n - 0.3)/0.2)/0.2'],['hold','θ(ℓ)'],['dhold','dθ(ℓ)'],
          ['u','x - h'],['tv','(R**2 - u**2)/(0.75*R**2)'],['nv','θ(tv)'],['dnv','dθ(tv)*(-2*u)/(0.75*R**2)'],
          ['uh','y - p'],['th','(R_h**2 - uh**2)/(0.75*R_h**2)'],['nh','θ(th)'],['dnh','dθ(th)*(-2*uh)/(0.75*R_h**2)'],
          ['δ','u - z_a'],['W','κ_0*u**2 + k_d*δ**2'],['G','S*hold*nv*nh']]};
      const spring=(c,h,letters=lx)=>({...cs,letters,terms:[{c,h}]});
      split(springs,[
        ['in-grip','Grip load',contactInputs,'Grip load '+(k+1),spring('-S*dhold*nv*nh*W/2','ℓ')],
        ['in-tip','Tip load',contactInputs,'Tip load '+(k+1),spring('G*(κ_0*u + k_d*δ) + S*hold*dnv*nh*W/2','h')],
        [height,'Load on casing',['Tip load'],undefined,{letters:{'Tip load':'F',[hx]:'x'},terms:[{c:'-F',h:'x'}]}],
        ['in-cart','Cart load',contactInputs,undefined,spring('-S*hold*nv*dnh*W/2','y')],
        [place,'Load on place',['Cart load'],undefined,{letters:{'Cart load':'F',['Place '+(k+1)]:'p'},terms:[{c:'-F',h:'p'}]}],
        [memory,'Load on attachment memory',contactInputs,undefined,spring('G*k_d*δ - k_r*z_a','z_a')],
        [latch,'Spring load on latch',contactInputs,'Other latch load',spring('-dS*hold*nv*nh*W/2','n')]
      ]);
      // The latch motor at the rate u_n applies the torque τ = C_c u_n - P_n (P_n the spring's load on the latch) and pays τ u_n.
      split(contact+'latch-drive',[[latch,'Latch torque',[contact+'latch-rate',springs+'/field'],undefined,
          {letters:{['Latch rate '+(k+1)]:'u_n','Other latch load':'P_n',['Latch '+(k+1)]:'n'},constants:{C_c:P.latchCost},
            terms:[{c:'C_c*u_n - P_n',h:'n'}]}],
        ['in-battery','Battery payment',['Latch torque',contact+'latch-rate'],undefined,
          {letters:{'Latch torque':'τ',['Latch rate '+(k+1)]:'u_n','Battery':'b'},terms:[{c:'-τ*u_n',h:'b'}]}]]);
    }
    const kept=wires.filter(edge=>!fields.has(edge.target));
    for(const edge of [...kept,...branches]){
      const source=published.get(edge.source+'|'+edge.label);
      if(source){edge.source=source;edge.sourcePort='coefficient';edge.id=source+'>'+edge.target+':'+edge.label;}
    }
    return kept.concat(branches);
  }
  function evaluate(s,places,P=F.DEFAULTS,time=0,trace=true,learning=null){
    const receiverSize = learning ? Learning.NR : NR;
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
    node('mouth','receiver','Mouth','box',{renderer:'mouth',column:2,row:2,description:'One opener and one digestor work on the casing held at the mouth.'});
    node('opener','mouth','Opener','effort');
    node('digestor','mouth','Digestor','effort');
    node('encounter','feeding','Spatial coupling','box',{column:1,row:0});
    // The receiver has these same seven readouts even in an empty world.
    for(const [id,label] of [['battery','Battery'],['grip','Grip'],['tip','Tip height'],['cart','Cart position'],
      ['look','Look distance'],['opener','Opening effort'],['digestor','Digestion effort']]){
      node('in-'+id,'encounter',label,'boundary',{side:'in'});
    }
    node('gaze','encounter','Gaze position','operation',{column:0});
    for(const [id,label] of [['red','Red at gaze'],['climb','Change with gaze'],['keep','Food within grip'],
      ['grip-load','Other grip load'],['tip-load','Other tip load'],['opened','Opening rate'],['eaten','Food taken']]){
      node('sum-'+id,'encounter',label,'operation',{column:2,symbol:places.length?'Σ':'0',description:'Sum of the contributions from the contacts.'});
    }
    node('opener/effort','opener','Opening effort','operation');
    node('digestor/effort','digestor','Digestion effort','operation');
    const stateIds=['battery','grip','tip','cart','cart','wheel','look','scan'];
    const labels=['Battery','Grip','Tip height','Cart position','Cart momentum','Wheel position','Look distance','Scan'];
    if (learning) {
      node('memory','receiver','Food memory and search','box',{column:1,row:1});
      node('memory-gaze','memory','Gaze position','operation');
      node('memory-drive','memory','Search pull','operation');
      for (const {id,label,start,end} of Learning.stores) {
        const value = end === start+1 ? s[start] : s.slice(start,end);
        node(id,'memory',label,'state',{value});
        node(id+'-rate','memory',label+' rate','operation');
        node(id+'-motor','memory',label+' motor','effort');
      }
      // A vector store has one vector readout. Its scalar coordinates occupy
      // their usual places in the integrator, before the casings.
      stateIds.length=labels.length=receiverSize;
    } else {
      node('patrol','receiver','Patrol','box',{column:1,row:1});
      node('patrol-point','patrol','Patrol point','state',{value:s[8]});
      node('patrol-scan','patrol','Patrol scan','state',{value:s[9]});
      node('patrol-motor','patrol','Patrol motor','effort');
      node('patrol-scan-motor','patrol','Patrol scan motor','effort');
      stateIds.push('patrol-point','patrol-scan');labels.push('Patrol point','Patrol scan');
    }
    node('grip','lever','Grip','state',{value:s[1]});node('tip','lever','Tip height','state',{value:s[2],renderer:'tip'});
    node('cart','wheels','Cart','state',{value:[s[3],s[4]],configuration:s[3],renderer:'cart'});
    // The note's dragger is drawn as what it is made of, with no box around it: the store it holds (the wheel)
    // and the spring between that store and the position it reads (David, 2026-11-20: "it shoudlbe explicit").
    node('wheel','wheels','Wheel','state',{value:s[5],renderer:'wheel'});
    // A spring (the note's sec.general): minus d of half a stiffness times the squared difference of two
    // readings. One mark with no box, drawn by the explorer as a bend joining its two wires with a red resistor
    // on it (glyph 'spring'); its loads on the two wires are equal and opposite. `pull` is its load on the second.
    const springMark=(id,parent,label,stiffness,k,[firstLabel,x],[secondLabel,y],pull)=>node(id,parent,label,'field',{glyph:'spring',stiffness,
      dependencyView:true,result:pull,resultLabel:'Load on '+secondLabel.toLowerCase()+'; its negative on '+firstLabel.toLowerCase(),
      form:{letters:{[firstLabel]:x,[secondLabel]:y},constants:{[stiffness]:k},terms:[{c:'-1',h:`${stiffness}/2*(${x} - ${y})**2`}]}});
    springMark('axle','wheels','Axle spring','κ_w',P.kappac,['Cart position','y'],['Wheel position','z_w'],P.kappac*(s[3]-s[5]));
    // The load the axle's spring puts on the wheel, which the drive motor is sent (the note's eqn.cart: "The
    // circle computes P_w"), a static map of the cart's and the wheel's positions.
    node('wheel-load','wheels','Other wheel load','operation',{law:{letters:{'Cart position':'y','Wheel position':'z_w'},
      constants:{κ_w:P.kappac},where:[['P_w','κ_w*(y - z_w)']]}});
    node('drive','wheels','Drive motor','effort');
    node('look','eye','Look distance','state',{value:s[6],renderer:'look'});node('scan','eye','Scan','state',{value:s[7]});

    node('look-motor','eye','Look motor','effort');node('scan-motor','eye','Scan motor','effort');
    // The eye is the note's sweep on (r, ψ) (sec.gates). Its two rates are static maps, each with its law
    // written once (explorer-forms.js, lawCheck); a motor reads its own rate and nothing else. A law here is
    // gated by live (the battery holds some) and free (nothing red is seen).
    const gates=[['live','θ(b/b_0)'],['free','θ(1 - 2*seen)']];
    const gatedLaw=(letters,constants,name,law)=>({letters:{'Battery':'b','Red at gaze':'seen',...letters},
      constants:{b_0:P.b0,...constants},where:[...gates,[name,law]]});
    // The drive's torque (the note's eqn.cart: "a circle reading b, y, r, the search destination d, and the
    // redness at the gaze"), a static map outside the motor, which is sent the torque it gives.
    node('drive-torque','wheels','Drive torque','operation',{law:learning
      ?gatedLaw({'Look distance':'r','Search pull':'pull'},{k_τ:P.ktau},'τ_drive','live*(k_τ*seen*r + free*pull)')
      :gatedLaw({'Look distance':'r','Patrol point':'η','Cart position':'y'},{k_τ:P.ktau,k_exp:P.kexp},
        'τ_drive','live*(k_τ*seen*r + free*k_exp*(η - y))')});
    const eyeLetters={'Look distance':'r','Scan':'ψ'},eyeConstants={A:P.A,ν:P.omega,γ:P.gam};
    node('look-rate','eye','Look rate','operation',{law:gatedLaw({...eyeLetters,'Change with gaze':'climb'},{...eyeConstants,λ_look:P.lam},
      'look_r','live*(free*(ν*ψ + γ*(A**2 - r**2 - ψ**2)*r) + λ_look*climb)')});
    node('scan-rate','eye','Scan rate','operation',{law:gatedLaw(eyeLetters,eyeConstants,
      'look_ψ','live*free*(-ν*r + γ*(A**2 - r**2 - ψ**2)*ψ)')});
    if(!learning){
      const letters={'Patrol point':'η','Patrol scan':'ζ'},constants={B:P.B,ν_p:P.nup,γ_p:P.gamp};
      node('patrol-rate','patrol','Patrol point rate','operation',{law:gatedLaw(letters,constants,
        'patrol_η','live*free*(ν_p*ζ + γ_p*(B**2 - η**2 - ζ**2)*η)')});
      node('patrol-scan-rate','patrol','Patrol scan rate','operation',{law:gatedLaw(letters,constants,
        'patrol_ζ','live*free*(-ν_p*η + γ_p*(B**2 - η**2 - ζ**2)*ζ)')});
    }
    node('gripper','lever','Grip motor','effort');node('lift','lever','Lift motor','effort');
    node('grip-rate','lever','Grip rate','operation');node('lift-rate','lever','Lift rate','operation');
    const scales=[10,2,.4,8,1,8,6,6,8,8];
    scales.length=receiverSize;
    // A spatial store holds one number for each centre on the line, so its wires carry a list: a hose in the
    // explorer, whose wires are named by their centres (explorer-core.js, `coordinate`).
    const atCentres=learning?Learning.centres.map(c=>'at '+String(c).replace('-','−')):[];
    function wire(source,target,label,value,load,extra={}){
      if(efforts.has(source))source+='/field';if(efforts.has(target))target+='/field';
      // Such a list stands in for a function on the displayed line [−9, 9] (the note's sec.food_expectation:
      // U(x) = Σ w_i φ_i(x) for the usual yield, one number per centre for the rest); the hose's mouth says so.
      const coordinates=Array.isArray(value)&&value.length===atCentres.length?{coordinates:atCentres,standsFor:'ℝ^[−9, 9]'}:{};
      if(trace)wires.push({id:source+'>'+target+':'+label,source,target,label,value,load,valueScale:1,loadScale:1,...coordinates,...extra});
    }
    function term(target,i,load=0){
      qload[i]+=load;
      wire(stateIds[i],target,labels[i],s[i],load,{valueScale:scales[i]||1,loadScale:i===0?1:i===1?.1:1,inclination:true});
    }
    const [b,l,h,y,my,zw,r,scan,eta,zeta]=s;
    const live=theta(b/P.b0),hold=theta(l),dhold=derivative(l);
    const contact=F.encounter(s,places,P,receiverSize), mouth=F.mouthEfforts(b,l,P);
    let pryAll=0,eatAll=0,latchPay=0;
    let seen=0,climb=0,K=0,Pl=0,Ph=0,Py=0,energy=b+my*my/(2*P.M)+.5*P.kappac*(y-zw)**2,loss=0;
    const data=[];
    for(let k=0;k<places.length;k++){
      const start=receiverSize+NC*k,[f,a,x,mu,z,zg,za]=s.slice(start,start+NC),place=places[k],id='casing-'+k;
      node(id,'feeding','Casing '+(k+1),'box',{renderer:'casing',index:k,column:0,row:k});
      node(id+'/food',id,'Food','state',{value:f,renderer:'food'});node(id+'/opening',id,'Opening','state',{value:a});
      node(id+'/body',id,'Casing body','box',{renderer:'casing',index:k});
      node(id+'/height',id+'/body','Height','state',{value:[x,mu],configuration:x,renderer:'tip'});
      node(id+'/memory',id+'/body','Air memory','state',{value:z});springMark(id+'/spring',id+'/body','Air spring','κ',P.kappa,['Height '+(k+1),'x'],['Air memory '+(k+1),'z'],P.kappa*(x-z));
      node(id+'/ground',id+'/body','Ground','effort');node(id+'/gravity',id+'/body','Gravity','effort');
      node(id+'/ground-memory',id+'/body','Ground memory','state',{value:zg});
      node(id+'/red',id,'Red','operation',{renderer:'red',index:k});node(id+'/access',id,'Access','operation');node(id+'/place',id,'Place','operation',{value:place});
      stateIds.push(id+'/food',id+'/opening',id+'/height',id+'/height',id+'/memory',id+'/ground-memory',id+'/attachment-memory',id+'/latch');
      labels.push('Food '+(k+1),'Opening '+(k+1),'Height '+(k+1),'Momentum '+(k+1),'Air memory '+(k+1),'Ground memory '+(k+1),'Attachment memory '+(k+1),'Latch '+(k+1));scales.push(10,1,.4,1,.4,.4,.4,1);
      const red=theta((f-P.fr)/P.fr),access=theta((a-P.a1)/(P.a2-P.a1)),u=(y+r-place)/P.w, bell=Math.exp(-u*u);
      seen+=red*bell;climb+=red*(-2*u/P.w)*bell;
      const [nv,dnv]=near(x-h,P.R),[nh,dnh]=near(y-place,P.Rh),keep=theta((f-P.f0)/P.f0);
      const selected=contact.engaged[k];
      const kept=selected*nv*nh*keep;
      K+=kept;
      const stretch=x-h,da=stretch-za,G=selected*hold*nv*nh,W=P.kappa0*stretch*stretch+P.kd*da*da;
      const px=-G*(P.kappa0*stretch+P.kd*da)-.5*selected*hold*dnv*nh*W;
      const pl=-.5*selected*dhold*nv*nh*W,py=-.5*selected*hold*nv*dnh*W;
      Pl+=pl;Ph-=px;Py+=py;
      const attBox='attachment-'+k,att=attBox+'/springs';node(attBox,'encounter','Contact '+(k+1),'box',{renderer:'spring',stretch,column:1,row:k});
      node(att,attBox,'Contact springs','effort',{renderer:'spring',stretch:da});node(id+'/attachment-memory',attBox,'Attachment memory','state',{value:za});
      wire('in-grip',att,'Grip',l,pl);wire('in-tip',att,'Tip height',h,-px);wire('in-cart',att,'Cart position',y,py);
      wire(att,'sum-grip-load','Grip load '+(k+1),pl,0);
      wire(att,'sum-tip-load','Tip load '+(k+1),-px,0);
      term(att,start+2,px);wire(id+'/place',att,'Place '+(k+1),place,-py,{valueScale:6});
      term(att,start+6,G*P.kd*da-P.kr*za);
      node(id+'/latch',attBox,'Contact latch','state',{value:s[start+7]});
      node(attBox+'/latch-drive',attBox,'Latch motor','effort');
      const latchDot=contact.latchRate[k], springLatch=-.5*contact.engageSlope[k]*hold*nv*nh*W;
      const latchTorque=P.latchCost*latchDot-springLatch;
      term(att,start+7,springLatch);term(attBox+'/latch-drive',start+7,latchTorque);
      latchPay+=latchTorque*latchDot;loss+=P.latchCost*latchDot*latchDot;
      wire('in-battery',attBox+'/latch-drive','Battery',b,-latchTorque*latchDot);
      wire(att,attBox+'/latch-drive','Other latch load',springLatch,0);
      const share=selected*contact.mouth[k],atMouth=live*hold*share;
      const pry=mouth.opening*share*(1-a),eat=mouth.digestion*share*access*f;
      pryAll+=pry;eatAll+=eat;

      // Each stateless operation below has one readout and receives only the
      // arguments used to compute it. Contact-local names survive projection.
      const op=(key,label,parent=attBox)=>node(attBox+'/'+key,parent,label,'operation');
      const offer=op('offer','Food within reach'), engaged=op('engaged','Engagement');
      const gripFood=op('held-food','Food within grip'), reachMouth=op('mouth-reach','Within reach of mouth');
      const at=op('at-mouth','Contact at mouth'), opening=op('opening','Opening contribution');
      const eating=op('eating','Food contribution'), latchRate=op('latch-rate','Latch rate');
      const placeTo=target=>wire(id+'/place',target,'Place '+(k+1),place,0,{valueScale:6});
      const cartTo=target=>wire('in-cart',target,'Cart position',y,0);
      const tipTo=target=>wire('in-tip',target,'Tip height',h,0);
      const offeredTo=target=>wire(offer,target,'Food within reach '+(k+1),contact.available[k],0);
      const engagedTo=target=>wire(engaged,target,'Engagement '+(k+1),selected,0);
      term(offer,start);term(offer,start+2);placeTo(offer);cartTo(offer);tipTo(offer);
      term(engaged,start+7);
      offeredTo(gripFood);engagedTo(gripFood);
      wire(gripFood,'sum-keep','Food within grip '+(k+1),kept,0);
      term(reachMouth,start+2);placeTo(reachMouth);cartTo(reachMouth);
      engagedTo(at);wire(reachMouth,at,'Mouth reach '+(k+1),contact.mouth[k],0);
      wire(at,opening,'Contact at mouth '+(k+1),share,0);
      wire(at,eating,'Contact at mouth '+(k+1),share,0);
      wire('in-opener',opening,'Opening effort',mouth.opening,0);
      wire('in-digestor',eating,'Digestion effort',mouth.digestion,0);
      term(opening,start+1,pry);term(eating,start,-eat);
      wire(id+'/access',eating,'Access '+(k+1),access,0);
      wire(opening,'sum-opened','Opening rate '+(k+1),pry,0);
      wire(eating,'sum-eaten','Food taken '+(k+1),eat,0);

      offeredTo(latchRate);engagedTo(latchRate);term(latchRate,start+7);
      wire('in-battery',latchRate,'Battery',b,0);wire('in-grip',latchRate,'Grip',l,0);
      if(places.length>1){
        const score=op('score','Contact score'), priority=op('priority','Wins score comparison');
        const free=op('unblocked','Other latches free');
        offeredTo(score);term(score,start+2);placeTo(score);cartTo(score);tipTo(score);
        for(let j=0;j<places.length;j++){
          wire('attachment-'+j+'/score',priority,'Score '+(j+1),contact.score[j],0);
          if(j!==k)wire('casing-'+j+'/latch',free,'Latch '+(j+1),s[receiverSize+NC*j+7],0,{inclination:true});
        }
        wire(priority,latchRate,'First choice '+(k+1),contact.firstChoice[k],0);
        wire(free,latchRate,'Other latches free '+(k+1),contact.unblocked[k],0);
      }
      wire(latchRate,attBox+'/latch-drive','Latch rate '+(k+1),latchDot,0);
      term(id+'/red',start);term(id+'/access',start+1);
      const offset=op('offset','Offset from gaze'), redness=op('red','Red at gaze');
      const gradient=op('gradient','Change with gaze');
      wire('gaze',offset,'Gaze',y+r,0);placeTo(offset);
      wire(id+'/red',redness,'Red '+(k+1),red,0);wire(offset,redness,'Offset '+(k+1),u,0);
      wire(redness,gradient,'Red at gaze '+(k+1),red*bell,0);
      wire(offset,gradient,'Offset '+(k+1),u,0);
      wire(redness,'sum-red','Red at gaze '+(k+1),red*bell,0);
      wire(gradient,'sum-climb','Change with gaze '+(k+1),-2*u/P.w*red*bell,0);
      term(id+'/spring',start+2,-P.kappa*(x-z));term(id+'/spring',start+4,P.kappa*(x-z));term(id+'/gravity',start+2,-P.m*P.g);
      const v=x-P.xg,dg=v-zg,tg=theta(-v/P.eps),dtg=-derivative(-v/P.eps)/P.eps,td=theta(-v/P.epsd),dtd=-derivative(-v/P.epsd)/P.epsd;
      term(id+'/ground',start+2,-P.kg*(tg*v+.5*dtg*v*v)-P.kd*(td*dg+.5*dtd*dg*dg));
      term(id+'/ground',start+5,P.kd*td*dg-P.kr*zg);
      rate[start+2]=mu/P.m;
      energy+=f+mu*mu/(2*P.m)+P.m*P.g*x+.5*P.kg*tg*v*v+.5*G*W+.5*P.kappa*(x-z)**2+.5*P.kr*za*za+.5*P.kd*td*dg*dg+.5*P.kr*zg*zg;
      loss+=P.kappa**2/P.beta*(x-z)**2+P.ctear*pry+(1-P.alpha)*eat+(qload[start+5]**2+qload[start+6]**2)/P.bd;
      data.push({food:f,opening:a,height:x,red,access,pry,eat,atMouth,px,pl,py,selected});
    }
    term('in-battery',0,-latchPay);term('in-grip',1,Pl);term('in-tip',2,Ph);term('in-cart',3,Py);term('in-look',6);
    wire('in-cart','gaze','Cart position',y,0);wire('in-look','gaze','Look distance',r,0);
    wire('sum-keep','grip-rate','Food within grip',K,0);
    for(const [organ,label,effort] of [['opener','Opening effort',mouth.opening],['digestor','Digestion effort',mouth.digestion]]){
      const map=organ+'/effort';term(map,0);term(map,1);wire(map,'in-'+organ,label,effort,0);
    }
    wire('sum-opened','opener','Opening rate',pryAll,0);wire('sum-eaten','digestor','Food taken',eatAll,0);
    term('opener',0,-P.ctear*pryAll);term('digestor',0,P.alpha*eatAll);
    const free=theta(1-2*seen),ring=P.A*P.A-r*r-scan*scan;
    const lookR=live*(free*(P.omega*scan+P.gam*ring*r)+P.lam*climb),lookS=live*free*(-P.omega*r+P.gam*ring*scan);
    const searchPull=learning ? learning.pull : P.kexp*(eta-y);
    const torque=live*(P.ktau*seen*r+free*searchPull),grip=P.lgrip*live*(K*(2-l)-(1-K)*(l+1)),lift=P.llift*live*(hold*(P.x0-h)+(1-hold)*(P.xg-h));
    if (learning) {
      // Dependencies of the nine rates in FoodLearning.field. Spatial rates
      // are vector readouts in centre order, as in the learning receiver.
      const dependencies={
        meal:['battery','grip','meal','eaten'],
        yield:['grip','grip-rate','meal','cart','yield'],
        unexplored:['battery','cart','gaze','unexplored','return'],
        destination:['battery','seen','clock','destination','yield','unexplored'],
        clock:['battery','seen','clock','cart','destination'],
        return:['battery','gaze','eye-contact','last-empty','return','time-away','seen'],
        'eye-contact':['battery','gaze','eye-contact'],
        'last-empty':['battery','gaze','eye-contact','last-empty','seen'],
        'time-away':['battery','gaze','eye-contact','last-empty','time-away']
      };
      term('memory-gaze',3);term('memory-gaze',6);
      const readings=new Map(Learning.stores.map(({id,label,start,end})=>
        [id,[id,label,end===start+1?s[start]:s.slice(start,end)]]));
      readings.set('gaze',['memory-gaze','Gaze',y+r]);
      readings.set('seen',['sum-red','Red at gaze',seen]);
      readings.set('grip-rate',['grip-rate','Chosen grip rate',grip]);
      readings.set('eaten',['sum-eaten','Food taken',eatAll]);
      term('memory-drive',3);
      wire('destination','memory-drive','Destination',learning.target,0);
      wire('memory-drive','drive-torque','Search pull',learning.pull,0);
      for (const {id,label,start,end} of Learning.stores) {
        const scalar=end===start+1, speed=learning.rate.slice(start,end);
        const load=speed.map(v=>P.memoryCost*v);
        const value=scalar?s[start]:s.slice(start,end), inclination=scalar?load[0]:load;
        const motor=id+'-motor', zero=scalar?0:speed.map(()=>0);
        const rateMap=id+'-rate';
        for(const source of dependencies[id]){
          if(source==='battery')term(rateMap,0);
          else if(source==='grip')term(rateMap,1);
          else if(source==='cart')term(rateMap,3);
          else {
            const [from,name,reading]=readings.get(source);
            wire(from,rateMap,name,reading,Array.isArray(reading)?reading.map(()=>0):0);
          }
        }
        wire(rateMap,motor,label+' rate',scalar?speed[0]:speed,zero);
        wire(id,motor,label,value,inclination,{inclination:true});
        term(motor,0,-P.memoryCost*speed.reduce((sum,v)=>sum+v*v,0));
        for(let j=start;j<end;j++){rate[j]=speed[j-start];qload[j]=load[j-start];}
        if(trace)nodes.find(n=>n.id===id).inclination=inclination;
      }
      loss+=learning.pay.memory;
    } else {
      const ringp=P.B*P.B-eta*eta-zeta*zeta;
      const patE=live*free*(P.nup*zeta+P.gamp*ringp*eta),patZ=live*free*(-P.nup*eta+P.gamp*ringp*zeta);
      for(const [x,i,speed] of [['patrol',8,patE],['patrol-scan',9,patZ]]){
        const map=x+'-rate',motor=x+'-motor';
        term(map,0);term(map,8);term(map,9);wire('sum-red',map,'Red at gaze',seen,0);
        wire(map,motor,labels[i]+' rate',speed,0);
        term(motor,i,P.cpat*speed);term(motor,0,-P.cpat*speed*speed);
        rate[i]=speed;
      }
      loss+=P.cpat*(patE*patE+patZ*patZ);
    }

    // The five readings (four for the scan) enter the rate's static map; the motor hears the rate alone,
    // pushes its own store and charges the battery.
    for(const [x,i,speed] of [['look',6,lookR],['scan',7,lookS]]){
      const map=x+'-rate',motor=x+'-motor';
      term(map,0);term(map,6);term(map,7);wire('sum-red',map,'Red at gaze',seen,0);
      if(x==='look')wire('sum-climb',map,'Change with gaze',climb,0,{valueScale:3});
      wire(map,motor,x==='look'?'Look rate':'Scan rate',speed,0);
      term(motor,i,P.csee*speed);term(motor,0,-P.csee*speed**2);
    }
    for(const i of [0,1])term('grip-rate',i);
    for(const i of [0,1,2])term('lift-rate',i);
    // Motors receive the actual other load, rather than differentiating their coefficients.
    const gripTorque=P.cgrip*grip-Pl,liftTorque=P.clift*lift-Ph;
    term('gripper',1,gripTorque);term('gripper',0,-gripTorque*grip);
    term('lift',2,liftTorque);term('lift',0,-liftTorque*lift);
    wire('grip-rate','gripper','Chosen grip rate',grip,0,{valueScale:4});wire('lift-rate','lift','Chosen lift rate',lift,0);
    // Expose computed other-load wires as values with zero returning load.
    // They are dependencies, not a force returning through a coefficient wire.
    wire('sum-grip-load','gripper','Other grip load',Pl,0);wire('sum-tip-load','lift','Other tip load',Ph,0);
    const spring=P.kappac*(y-zw),wheel=(spring+torque)/P.betac;
    term('axle',3,-spring);term('axle',5,spring);term('drive',5,torque);term('drive',0,-torque*wheel);
    // What the torque law reads enters its static map; the drive hears the torque and the axle's load on the wheel.
    term('drive-torque',0);term('drive-torque',6);
    if(!learning){term('drive-torque',8);term('drive-torque',3);}
    wire('sum-red','drive-torque','Red at gaze',seen,0);wire('drive-torque','drive','Drive torque',torque,0);
    term('wheel-load',3);term('wheel-load',5);wire('wheel-load','drive','Other wheel load',spring,0);
    rate[0]=qload[0];rate[1]=qload[1]/P.cgrip;rate[2]=qload[2]/P.clift;rate[3]=my/P.M;rate[4]=qload[3];rate[5]=qload[5]/P.betac;
    // The existing page permits zero eye cost: the continuous limit has the
    // chosen rates and zero inclination/payment; do not divide zero by zero.
    rate[6]=P.csee===0?lookR:qload[6]/P.csee;rate[7]=P.csee===0?lookS:qload[7]/P.csee;
    for(let k=0;k<places.length;k++){const i=receiverSize+NC*k;rate[i]=qload[i];rate[i+1]=qload[i+1];rate[i+3]=qload[i+2];rate[i+4]=qload[i+4]/P.beta;rate[i+5]=qload[i+5]/P.bd;rate[i+6]=qload[i+6]/P.bd;rate[i+7]=qload[i+7]/P.latchCost;}
    loss+=P.cgrip*grip*grip+P.clift*lift*lift+P.betac*wheel*wheel+P.csee*(lookR*lookR+lookS*lookS);
    node('battery-output','feeding','Battery out','boundary',{side:'out'});node('grip-output','feeding','Grip out','boundary',{side:'out'});
    wire('battery','battery-output','Battery',b,0,{valueScale:10});wire('grip','grip-output','Grip',l,0,{valueScale:2});
    if(trace)for(const n of nodes)if(n.kind==='state' && n.inclination===undefined){const i=stateIds.indexOf(n.id);n.inclination=qload[i];}
    const diagramWires=trace?separateFields(nodes,wires,places,learning,P):wires;
    return {rate,qload,energy,loss,seen,torque,live,hold,casings:data,version:1,title:'Feeding',root:'feeding',time,nodes,wires:diagramWires,state:s.slice(),places:places.slice()};
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
      addCasing:(place,food=10)=>{
        if(!Number.isFinite(place)||!Number.isFinite(food)||food<0)throw Error('Provide a finite place and a nonnegative amount of food.');
        places.push(place);foods.push(food);state.push(...F.casing(food,P));
      },
      step:dt=>{if(!(dt>=0&&dt<=1))throw Error('Step duration must be between zero and one');let remaining=dt;while(remaining>1e-12){const h=Math.min(.002,remaining);state=rk4(state,h,places,P);time+=h;remaining-=h;}},
      reset:()=>{state=F.start(places,foods,options.cart||0,P);time=0;},
      setState:(s,t=0)=>{if(s.length!==NR+NC*places.length||!s.every(Number.isFinite))throw Error('Invalid state');state=s.slice();time=t;},parameters:P};
  }
  function observeLearning(state, places, parameters, time=0) {
    const field=Learning.field(state,places,parameters);
    return evaluate(state,places,parameters,time,true,field);
  }
  const api={evaluate,rk4,adapter,observeLearning,near,derivative};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.FeedingArrangement=api;
})(globalThis);
