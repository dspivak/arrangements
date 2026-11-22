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
    // Every motor reads the battery (David, 2026-11-22, 779795d2: "its activity depends on live battery"): it is
    // only as active as live = θ(b/b_0), so each of its marks hears the battery as well as what it is told.
    function motor(part,coordinate,rate,other,name,forms=[]){
      split(part,[[coordinate,name,[rate,other,'battery'],undefined,forms[0]],['battery','Battery payment',[name,rate,'battery'],undefined,forms[1]]]);
    }
    // The lever's motors, the note's sec.gates: told the rate u, a motor drives its store at live·u by the torque
    // τ = c live u − P, and pays τ live u. The first mark's formula is written in the letters at its terminals.
    const lever=(cost,rateLabel,otherLabel,coordinate,q,load,c)=>{
      const form=(letters,terms)=>({letters,constants:{[c]:cost,b_0:P.b0},terms});
      return [form({[rateLabel]:'u',[otherLabel]:load,[coordinate]:q,'Battery':'b'},[{c:c+'*θ(b/b_0)*u - '+load,h:q}]),
        form({[rateLabel]:'u','Battery':'b'},[{c:'-τ*θ(b/b_0)*u',h:'b'}])];
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
    // The drive, the note's motor (eqn.motor, eqn.cart) told the torque τ_drive and the axle's load P_w on the
    // wheel. It applies g τ_drive, g = θ(b/b_0): g τ_drive dz_w − g τ_drive (g τ_drive + P_w)/β_w db, the wheel's rate
    // being (g τ_drive + P_w)/β_w. The torque arrives on a wire from its static map (Drive torque).
    split('drive',[['wheel','Motor inclination',['drive-torque','battery'],undefined,
        {letters:{'Drive torque':'τ_drive','Wheel position':'z_w','Battery':'b'},constants:{b_0:P.b0},terms:[{c:'θ(b/b_0)*τ_drive',h:'z_w'}]}],
      ['battery','Battery payment',['drive-torque','wheel-load','battery'],undefined,
        {letters:{'Drive torque':'τ_drive','Other wheel load':'P_w','Battery':'b'},constants:{β_w:P.betac,b_0:P.b0},
          terms:[{c:'-θ(b/b_0)*τ_drive*(θ(b/b_0)*τ_drive + P_w)/β_w',h:'b'}]}]]);
    // A motor told a rate u (the note's sec.general, after eqn.motor, with no other load on its store): it drives
    // its store at live·u, c θ(b/b_0) u dq − c (θ(b/b_0) u)² db. Each mark is wired to the rate and the battery,
    // which it reads, and to the one wire it pushes; what the rate itself reads enters the rate's static map.
    function atRate(part,store,storeLabel,q,rate,rateLabel,c,cost){
      const form=(letters,terms)=>({letters:{[rateLabel]:'u','Battery':'b',...letters},constants:{[c]:cost,b_0:P.b0},terms});
      split(part,[[store,'Motor inclination',[rate,'battery'],undefined,form({[storeLabel]:q},[{c:c+'*θ(b/b_0)*u',h:q}])],
        ['battery','Battery payment',[rate,'battery'],undefined,form({},[{c:'-'+c+'*(θ(b/b_0)*u)**2',h:'b'}])]]);
    }
    // The eye, the note's sec.gates: the sweep's two motors, each at its rate (Look rate, Scan rate).
    atRate('look-motor','look','Look distance','r','look-rate','Look rate','c_see',P.csee);
    atRate('scan-motor','scan','Scan','ψ','scan-rate','Scan rate','c_see',P.csee);
    if(learning){
      // A memory motor told the rate u drives its store at live·u and costs the battery c_mem |live u|²:
      // c_mem θ(b/b_0) u dq − c_mem (θ(b/b_0) u)² db. A store of several numbers has lists on its wires: its motor
      // is that many side by side, and the battery pays their sum.
      for(const {id,label,start,end} of Learning.stores){
        const rateLabel=label+' rate',form=(letters,terms)=>({letters:{[rateLabel]:'u','Battery':'b',...letters},
          constants:{c_mem:P.memoryCost,b_0:P.b0},terms});
        split(id+'-motor',[[id,'Memory inclination',[id+'-rate','battery'],undefined,form({[label]:'q'},[{c:'c_mem*θ(b/b_0)*u',h:'q'}])],
          ['battery','Battery payment',[id+'-rate','battery'],undefined,form({},[{c:end===start+1?'-c_mem*(θ(b/b_0)*u)**2':'-c_mem*Σ((θ(b/b_0)*u)**2)',h:'b'}])]]);
      }
    }else{
      atRate('patrol-motor','patrol-point','Patrol point','η','patrol-rate','Patrol point rate','c_pat',P.cpat);
      atRate('patrol-scan-motor','patrol-scan','Patrol scan','ζ','patrol-scan-rate','Patrol scan rate','c_pat',P.cpat);
    }
    for(let k=0;k<places.length;k++){
      const c='casing-'+k+'/', contact='attachment-'+k+'/', springs=contact+'springs';
      const height=c+'height', latch=c+'latch';
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
      // The contact's springs are not split into pushes: they are drawn as the note's lever has them (sec.gates),
      // a spring and a damper sent one gate, each a spring mark with a third terminal (evaluate, contactSprings).
      // The latch motor told the rate u_n drives the latch at live·u_n by the torque τ = C_c θ(b/b_0) u_n - P_n
      // (P_n the spring's load on the latch) and pays τ θ(b/b_0) u_n.
      split(contact+'latch-drive',[[latch,'Latch torque',[contact+'latch-rate',springs+'/latch-load','in-battery'],undefined,
          {letters:{['Latch rate '+(k+1)]:'u_n','Other latch load':'P_n',['Latch '+(k+1)]:'n','Battery':'b'},constants:{C_c:P.latchCost,b_0:P.b0},
            terms:[{c:'C_c*θ(b/b_0)*u_n - P_n',h:'n'}]}],
        ['in-battery','Battery payment',['Latch torque',contact+'latch-rate','in-battery'],undefined,
          {letters:{'Latch torque':'τ',['Latch rate '+(k+1)]:'u_n','Battery':'b'},constants:{b_0:P.b0},terms:[{c:'-τ*θ(b/b_0)*u_n',h:'b'}]}]]);
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
    // A motor (the note's sec.general: "A stateless part that pushes a load τ on a store q and pays the battery for
    // it"). Every motor is drawn with the one mark, the explorer's glyph 'motor', and opens into its two pushes.
    // `told` says what it is sent: a rate, a rate and the other load on its store, or a torque and that load.
    const sent={rate:'the rate at which to drive its store',
      'rate and load':'the rate at which to drive its store, and the load the rest of the arrangement puts on that store',
      'torque and load':'the torque to push with, and the load the rest of the arrangement puts on its store'};
    // `turns` is the label of its store's wire, which ends at the pulley on its shaft; the battery's ends at its plug.
    const motorPart=(id,parent,label,turns,told='rate',place={})=>node(id,parent,label,'effort',{...place,glyph:'motor',turns,powered:'Battery',
      description:'A motor: a stateless part that pushes a load on a store and pays the battery for it. It is sent '+sent[told]
        +'. It reads what it is sent and never pushes it: each such terminal is load-free, and carries a black wall.'
        +' It also reads the battery, and is only as active as the battery is live, θ(b/b_0): a rate it is told is'
        +' driven at live times that rate, a torque applied at live times that torque.'});
    // A motor's box (David, 2026-11-22, 779795d2: "you're right, B."): the static map that computes what the motor is
    // told, and the motor. It has no outputs: a part that needs the same rate reads the same inputs and computes it.
    const drivePart=(id,parent,label,told,place={})=>node(id,parent,label,'box',{...place,
      description:'A part that drives one store: the static map that computes '+told+', and the motor it tells, which'
        +' also reads the battery. It has no outputs; a part that needs the same value reads the same inputs and computes it.'});
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
        drivePart(id+'-part','memory',label+' drive','the rate of the '+label.toLowerCase());
        node(id+'-rate',id+'-part',label+' rate','operation');
        motorPart(id+'-motor',id+'-part',label+' motor',label);
      }
      // A vector store has one vector readout. Its scalar coordinates occupy
      // their usual places in the integrator, before the casings.
      stateIds.length=labels.length=receiverSize;
    } else {
      node('patrol','receiver','Patrol','box',{column:1,row:1});
      node('patrol-point','patrol','Patrol point','state',{value:s[8]});
      node('patrol-scan','patrol','Patrol scan','state',{value:s[9]});
      // The patrol is a sweep too, drawn as the eye is: each motor in its own part with the static map of its rate.
      drivePart('patrol-part','patrol','Patrol point drive','the patrol point’s rate');
      drivePart('patrol-scan-part','patrol','Patrol scan drive','the patrol scan’s rate');
      motorPart('patrol-motor','patrol-part','Patrol motor','Patrol point');
      motorPart('patrol-scan-motor','patrol-scan-part','Patrol scan motor','Patrol scan');
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
    // Two more of the same kind. A spring with a third terminal (`third`, the label and letter of a wire): its
    // stiffness is the constant times the value on that wire, and it pushes back there by minus half the constant
    // times the squared stretch (the note's eqn.spring, "the energy it takes to change its stiffness"). A spring
    // to an anchor (`second` null): its second end is held at 0, and `pull` is its load on its one wire.
    const springMark=(id,parent,label,stiffness,k,[firstLabel,x],second,pull,third,place={})=>{
      const K=third?stiffness+'*'+third[1]:stiffness,stretch=second?`(${x} - ${second[1]})`:x;
      return node(id,parent,label,'field',{glyph:'spring',stiffness:K,...(third?{third:third[0]}:{}),...(second?{}:{anchor:true}),...place,
        dependencyView:true,result:pull,resultLabel:second?'Load on '+second[0].toLowerCase()+'; its negative on '+firstLabel.toLowerCase():'Load on '+firstLabel.toLowerCase(),
        form:{letters:{[firstLabel]:x,...(second?{[second[0]]:second[1]}:{}),...(third?{[third[0]]:third[1]}:{})},constants:{[stiffness]:k},
          terms:[{c:'-1',h:`${K}/2*${stretch}**2`}]}});
    };
    springMark('axle','wheels','Axle spring','κ_w',P.kappac,['Cart position','y'],['Wheel position','z_w'],P.kappac*(s[3]-s[5]));
    // The load the axle's spring puts on the wheel, which the drive motor is sent (the note's eqn.cart: "The
    // circle computes P_w"), a static map of the cart's and the wheel's positions.
    node('wheel-load','wheels','Other wheel load','operation',{law:{letters:{'Cart position':'y','Wheel position':'z_w'},
      constants:{κ_w:P.kappac},where:[['P_w','κ_w*(y - z_w)']]}});
    drivePart('wheel-part','wheels','Wheel drive','the torque');
    motorPart('drive','wheel-part','Drive motor','Wheel position','torque and load');
    node('look','eye','Look distance','state',{value:s[6],renderer:'look',column:0,row:0});node('scan','eye','Scan','state',{value:s[7],column:0,row:1});

    // The eye's sweep (the note's sec.general) is two motors, each in its own part with the static map of its rate.
    drivePart('look-part','eye','Look drive','the look distance’s rate',{column:1,row:1});
    drivePart('scan-part','eye','Scan drive','the scan’s rate',{column:1,row:0});
    motorPart('look-motor','look-part','Look motor','Look distance');motorPart('scan-motor','scan-part','Scan motor','Scan');
    // The eye is the note's sweep on (r, ψ) (sec.gates). Its two rates are static maps, each with its law
    // written once (explorer-forms.js, lawCheck); a motor reads its rate and the battery. A law here is
    // gated by free (nothing red is seen); none reads the battery, since each motor does.
    const gates=[['free','θ(1 - 2*seen)']];
    const gatedLaw=(letters,constants,name,law)=>({letters:{'Red at gaze':'seen',...letters},
      constants,where:[...gates,[name,law]]});
    // The drive's torque (the note's eqn.cart: "a circle reading y, r, the search destination d, and the
    // redness at the gaze"), a static map outside the motor, which is told the torque it gives.
    node('drive-torque','wheel-part','Drive torque','operation',{law:learning
      ?gatedLaw({'Look distance':'r','Search pull':'pull'},{k_τ:P.ktau},'τ_drive','k_τ*seen*r + free*pull')
      :gatedLaw({'Look distance':'r','Patrol point':'η','Cart position':'y'},{k_τ:P.ktau,k_exp:P.kexp},
        'τ_drive','k_τ*seen*r + free*k_exp*(η - y)')});
    const eyeLetters={'Look distance':'r','Scan':'ψ'},eyeConstants={A:P.A,ν:P.omega,γ:P.gam};
    node('look-rate','look-part','Look rate','operation',{law:gatedLaw({...eyeLetters,'Change with gaze':'climb'},{...eyeConstants,λ_look:P.lam},
      'look_r','free*(ν*ψ + γ*(A**2 - r**2 - ψ**2)*r) + λ_look*climb')});
    node('scan-rate','scan-part','Scan rate','operation',{law:gatedLaw(eyeLetters,eyeConstants,
      'look_ψ','free*(-ν*r + γ*(A**2 - r**2 - ψ**2)*ψ)')});
    if(!learning){
      const letters={'Patrol point':'η','Patrol scan':'ζ'},constants={B:P.B,ν_p:P.nup,γ_p:P.gamp};
      node('patrol-rate','patrol-part','Patrol point rate','operation',{law:gatedLaw(letters,constants,
        'patrol_η','free*(ν_p*ζ + γ_p*(B**2 - η**2 - ζ**2)*η)')});
      node('patrol-scan-rate','patrol-scan-part','Patrol scan rate','operation',{law:gatedLaw(letters,constants,
        'patrol_ζ','free*(-ν_p*η + γ_p*(B**2 - η**2 - ζ**2)*ζ)')});
    }
    // The note's gripper (eqn.gripper) and the lift: each its rate's static map and its motor. The Lever around
    // them is a drawing group, no box of the note's: nothing leaves it but what its stores read out.
    drivePart('grip-part','lever','Grip drive','the grip’s rate');drivePart('lift-part','lever','Lift drive','the tip’s rate');
    motorPart('gripper','grip-part','Grip motor','Grip','rate and load');motorPart('lift','lift-part','Lift motor','Tip height','rate and load');
    node('grip-rate','grip-part','Grip rate','operation');node('lift-rate','lift-part','Lift rate','operation');
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
    // A store's wire to a part, with the load that part returns on it. `term` also adds that load to the
    // store's total; `read` draws the wire alone, for a load whose total the model has already counted.
    function read(target,i,load=0){
      wire(stateIds[i],target,labels[i],s[i],load,{valueScale:scales[i]||1,loadScale:i===0?1:i===1?.1:1,inclination:true});
    }
    function term(target,i,load=0){qload[i]+=load;read(target,i,load);}
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
      node(att,attBox,'Contact springs','box',{renderer:'spring',stretch:da});node(id+'/attachment-memory',attBox,'Attachment memory','state',{value:za});
      // The model's loads from this contact's springs, minus the partials of U = ½ G (κ_0 u² + k_d δ²) + ½ k_r z_a²
      // written out (the note's sec.gates, P_x, P_ℓ, P_y, P_z): these are what the stores are charged. The wires
      // drawn below carry the same loads factored through the springs and the gates' maps, and the tests compare.
      qload[start+2]+=px;qload[start+6]+=G*P.kd*da-P.kr*za;
      node(id+'/latch',attBox,'Contact latch','state',{value:s[start+7]});
      drivePart(attBox+'/latch-part',attBox,'Latch drive','the latch’s rate');
      motorPart(attBox+'/latch-drive',attBox+'/latch-part','Latch motor','Latch '+(k+1),'rate and load');
      const latchDot=contact.latchRate[k], springLatch=-.5*contact.engageSlope[k]*hold*nv*nh*W;
      const latchTorque=P.latchCost*latchDot-springLatch;
      qload[start+7]+=springLatch;term(attBox+'/latch-drive',start+7,latchTorque);
      latchPay+=latchTorque*latchDot;loss+=P.latchCost*latchDot*latchDot;
      wire('in-battery',attBox+'/latch-drive','Battery',b,-latchTorque*latchDot);
      const share=selected*contact.mouth[k],atMouth=live*hold*share;
      const pry=mouth.opening*share*(1-a),eat=mouth.digestion*share*access*f;
      pryAll+=pry;eatAll+=eat;

      // Each stateless operation below has one readout and receives only the
      // arguments used to compute it. Contact-local names survive projection.
      const op=(key,label,parent=attBox,extra={})=>node(attBox+'/'+key,parent,label,'operation',extra);
      const named=name=>name+' '+(k+1);
      const offer=op('offer','Food within reach');
      // The engagement s = θ((n − 0.3)/0.2) of the latch n (the note's s_k), one of the contact gate's four factors.
      const engaged=op('engaged','Engagement',attBox,{law:{letters:{[named('Latch')]:'n'},where:[['s','θ((n - 0.3)/0.2)']]}});
      if(trace){
        // The contact's springs, as the note's lever has them (sec.gates): "a spring and a damper sent the same
        // gate G", the spring the stiffness κ_0 G and the damper k_d G, with the damper's return spring. Static
        // maps compute the stretch u = x − h and the gate's four factors (the engagement s, hold = θ(ℓ), and the
        // two nearnesses N and H) and their product G; the tip spring joins the casing's height and the tip, the
        // damper's spring joins the stretch and the stroke z_a, and each has a third terminal, fed G, where it
        // pushes back by minus half its constant times its squared stretch. That load returns through the maps
        // to the grip, the latch, the heights and the places, each map sending it on times its law's derivative.
        const map=(key,label,law,column,row=0,parent=att)=>node(att+'/'+key,parent,label,'operation',{law,column,row});
        const Rk={R:P.R},stiff={κ_0:P.kappa0,k_d:P.kd};
        map('stretch','Stretch',{letters:{[named('Height')]:'x','Tip height':'h'},where:[['u','x - h']]},0,0);
        map('near-tip','Near the tip',{letters:{[named('Stretch')]:'u'},constants:Rk,where:[['N','θ((R**2 - u**2)/(0.75*R**2))']]},1,0);
        map('over','Cart at casing',{letters:{'Cart position':'y',[named('Place')]:'p'},constants:{R_h:P.Rh},
          where:[['H','θ((R_h**2 - (y - p)**2)/(0.75*R_h**2))']]},0,1);
        map('hold','Hold',{letters:{'Grip':'ℓ'},where:[['hold','θ(ℓ)']]},0,2);
        map('gate','Gate',{letters:{[named('Engagement')]:'s',[named('Hold')]:'hold',[named('Near the tip')]:'N',[named('Cart at casing')]:'H'},
          where:[['G','s*hold*N*H']]},2,0);
        const gate=[named('Gate'),'G'],tipSpring=att+'/tip-spring',damperSpring=att+'/damper-spring';
        springMark(tipSpring,att,'Tip spring','κ_0',P.kappa0,[named('Height'),'x'],['Tip height','h'],P.kappa0*G*stretch,gate,{column:3,row:0});
        springMark(damperSpring,att,'Damper spring','k_d',P.kd,[named('Stretch'),'u'],[named('Attachment memory'),'z_a'],P.kd*G*da,gate,{column:3,row:1});
        springMark(att+'/return-spring',att,'Return spring','k_r',P.kr,[named('Attachment memory'),'z_a'],null,-P.kr*za,undefined,{column:3,row:2});
        // The loads, from the springs back. On the gate's wire: −½ κ_0 u² and −½ k_d δ², in all −W/2.
        const onGate=-.5*W,onNear=onGate*selected*hold*nh,onOver=onGate*selected*hold*nv;
        const onStretch=-P.kd*G*da+onNear*dnv;
        read(tipSpring,start+2,-P.kappa0*G*stretch);wire('in-tip',tipSpring,'Tip height',h,P.kappa0*G*stretch);
        wire(att+'/gate',tipSpring,gate[0],G,-.5*P.kappa0*stretch*stretch);
        wire(att+'/stretch',damperSpring,named('Stretch'),stretch,-P.kd*G*da);read(damperSpring,start+6,P.kd*G*da);
        wire(att+'/gate',damperSpring,gate[0],G,-.5*P.kd*da*da);
        read(att+'/return-spring',start+6,-P.kr*za);
        read(att+'/stretch',start+2,onStretch);wire('in-tip',att+'/stretch','Tip height',h,-onStretch);
        wire(att+'/stretch',att+'/near-tip',named('Stretch'),stretch,onNear*dnv);
        wire('in-cart',att+'/over','Cart position',y,onOver*dnh);
        wire(id+'/place',att+'/over',named('Place'),place,-onOver*dnh,{valueScale:6});
        wire('in-grip',att+'/hold','Grip',l,onGate*selected*nv*nh*dhold);
        wire(att+'/hold',att+'/gate',named('Hold'),hold,onGate*selected*nv*nh);
        wire(att+'/near-tip',att+'/gate',named('Near the tip'),nv,onNear);
        wire(att+'/over',att+'/gate',named('Cart at casing'),nh,onOver);
        wire(engaged,att+'/gate',named('Engagement'),selected,onGate*hold*nv*nh);
        // What the motors are told (the note's eqn.motor: a motor "must be sent the true P"): the load these
        // springs put on the grip, on the tip and on the latch, each as a value, so each a static map. They
        // read the energy the two springs would hold at G = 1, E = W/2, which is minus the load on the gate's wire.
        // They stand in a box of their own, so that the springs and their gate are seen without them.
        const told=node(att+'/told',att,'Loads told to the motors','box',{column:4,row:0,
          description:'A motor must be sent the load the rest of the arrangement puts on its store. These static arrangements compute the load the contact’s springs put on the grip, on the tip and on the latch, each as a value.'});
        map('energy','Energy at full gate',{letters:{[named('Stretch')]:'u',[named('Attachment memory')]:'z_a'},constants:stiff,
          where:[['E','(κ_0*u**2 + k_d*(u - z_a)**2)/2']]},0,0,told);
        wire(att+'/stretch',att+'/energy',named('Stretch'),stretch,0);read(att+'/energy',start+6);
        const energy=to=>wire(att+'/energy',to,named('Energy at full gate'),.5*W,0);
        const nearTo=to=>wire(att+'/near-tip',to,named('Near the tip'),nv,0),overTo=to=>wire(att+'/over',to,named('Cart at casing'),nh,0);
        const holdTo=to=>wire(att+'/hold',to,named('Hold'),hold,0),sTo=to=>wire(engaged,to,named('Engagement'),selected,0);
        const gripLoad=map('grip-load','Grip load',{letters:{'Grip':'ℓ',[named('Engagement')]:'s',[named('Near the tip')]:'N',
          [named('Cart at casing')]:'H',[named('Energy at full gate')]:'E'},where:[['P_ℓ','-s*dθ(ℓ)*N*H*E']]},1,0,told);
        wire('in-grip',gripLoad,'Grip',l,0);sTo(gripLoad);nearTo(gripLoad);overTo(gripLoad);energy(gripLoad);
        wire(gripLoad,'sum-grip-load','Grip load '+(k+1),pl,0);
        const tipLoad=map('tip-load','Tip load',{letters:{[named('Engagement')]:'s',[named('Hold')]:'hold',[named('Cart at casing')]:'H',
          [named('Stretch')]:'u',[named('Attachment memory')]:'z_a'},constants:{...Rk,...stiff},
          where:[['t','(R**2 - u**2)/(0.75*R**2)'],['δ','u - z_a'],
            ['P_h','s*hold*H*(θ(t)*(κ_0*u + k_d*δ) + dθ(t)*(-2*u)/(0.75*R**2)*(κ_0*u**2 + k_d*δ**2)/2)']]},1,1,told);
        sTo(tipLoad);holdTo(tipLoad);overTo(tipLoad);wire(att+'/stretch',tipLoad,named('Stretch'),stretch,0);read(tipLoad,start+6);
        wire(tipLoad,'sum-tip-load','Tip load '+(k+1),-px,0);
        const latchLoad=map('latch-load','Latch load',{letters:{[named('Latch')]:'n',[named('Hold')]:'hold',[named('Near the tip')]:'N',
          [named('Cart at casing')]:'H',[named('Energy at full gate')]:'E'},where:[['P_n','-dθ((n - 0.3)/0.2)/0.2*hold*N*H*E']]},1,2,told);
        read(latchLoad,start+7);holdTo(latchLoad);nearTo(latchLoad);overTo(latchLoad);energy(latchLoad);
        wire(latchLoad,attBox+'/latch-drive','Other latch load',springLatch,0);
      }
      const gripFood=op('held-food','Food within grip'), reachMouth=op('mouth-reach','Within reach of mouth');
      const at=op('at-mouth','Contact at mouth'), opening=op('opening','Opening contribution');
      const eating=op('eating','Food contribution'), latchRate=op('latch-rate','Latch rate',attBox+'/latch-part');
      const placeTo=target=>wire(id+'/place',target,'Place '+(k+1),place,0,{valueScale:6});
      const cartTo=target=>wire('in-cart',target,'Cart position',y,0);
      const tipTo=target=>wire('in-tip',target,'Tip height',h,0);
      const offeredTo=target=>wire(offer,target,'Food within reach '+(k+1),contact.available[k],0);
      const engagedTo=target=>wire(engaged,target,'Engagement '+(k+1),selected,0);
      term(offer,start);term(offer,start+2);placeTo(offer);cartTo(offer);tipTo(offer);
      // The latch's wire into the engagement carries the load the gate sends back through it, the springs' whole
      // load on the latch (counted above).
      read(engaged,start+7,springLatch);
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
      wire('in-grip',latchRate,'Grip',l,0);
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
      wire(latchRate,attBox+'/latch-drive','Latch rate '+(k+1),contact.latchTold[k],0);
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
    // What each motor is told (…T) carries no live: the motor reads the battery and drives at live times it.
    const lookRT=free*(P.omega*scan+P.gam*ring*r)+P.lam*climb,lookST=free*(-P.omega*r+P.gam*ring*scan);
    const lookR=live*lookRT,lookS=live*lookST;
    const searchPull=learning ? learning.pull : P.kexp*(eta-y);
    const torqueT=P.ktau*seen*r+free*searchPull,gripT=P.lgrip*(K*(2-l)-(1-K)*(l+1)),liftT=P.llift*(hold*(P.x0-h)+(1-hold)*(P.xg-h));
    const torque=live*torqueT,grip=live*gripT,lift=live*liftT;
    if (learning) {
      // Dependencies of the nine rates in FoodLearning.field. Spatial rates
      // are vector readouts in centre order, as in the learning receiver.
      // None reads the battery: each memory motor reads it itself. The yield's rule needs the rate the gripper tells
      // its motor; it reads what that rate is computed from, the grip and the food within grip, and computes it.
      const dependencies={
        meal:['grip','meal','eaten'],
        yield:['grip','keep','meal','cart','yield'],
        unexplored:['cart','gaze','unexplored','return'],
        destination:['seen','clock','destination','yield','unexplored'],
        clock:['seen','clock','cart','destination'],
        return:['gaze','eye-contact','last-empty','return','time-away','seen'],
        'eye-contact':['gaze','eye-contact'],
        'last-empty':['gaze','eye-contact','last-empty','seen'],
        'time-away':['gaze','eye-contact','last-empty','time-away']
      };
      term('memory-gaze',3);term('memory-gaze',6);
      const readings=new Map(Learning.stores.map(({id,label,start,end})=>
        [id,[id,label,end===start+1?s[start]:s.slice(start,end)]]));
      readings.set('gaze',['memory-gaze','Gaze',y+r]);
      readings.set('seen',['sum-red','Red at gaze',seen]);
      readings.set('keep',['sum-keep','Food within grip',K]);
      readings.set('eaten',['sum-eaten','Food taken',eatAll]);
      term('memory-drive',3);
      wire('destination','memory-drive','Destination',learning.target,0);
      wire('memory-drive','drive-torque','Search pull',learning.pull,0);
      for (const {id,label,start,end} of Learning.stores) {
        const scalar=end===start+1, speed=learning.rate.slice(start,end), told=learning.told.memory.slice(start-8,end-8);
        const load=speed.map(v=>P.memoryCost*v);
        const value=scalar?s[start]:s.slice(start,end), inclination=scalar?load[0]:load;
        const motor=id+'-motor', zero=scalar?0:speed.map(()=>0);
        const rateMap=id+'-rate';
        for(const source of dependencies[id]){
          if(source==='grip')term(rateMap,1);
          else if(source==='cart')term(rateMap,3);
          else {
            const [from,name,reading]=readings.get(source);
            wire(from,rateMap,name,reading,Array.isArray(reading)?reading.map(()=>0):0);
          }
        }
        wire(rateMap,motor,label+' rate',scalar?told[0]:told,zero);
        wire(id,motor,label,value,inclination,{inclination:true});
        term(motor,0,-P.memoryCost*speed.reduce((sum,v)=>sum+v*v,0));
        for(let j=start;j<end;j++){rate[j]=speed[j-start];qload[j]=load[j-start];}
        if(trace)nodes.find(n=>n.id===id).inclination=inclination;
      }
      loss+=learning.pay.memory;
    } else {
      const ringp=P.B*P.B-eta*eta-zeta*zeta;
      const patET=free*(P.nup*zeta+P.gamp*ringp*eta),patZT=free*(-P.nup*eta+P.gamp*ringp*zeta);
      const patE=live*patET,patZ=live*patZT;
      for(const [x,i,speed,told] of [['patrol',8,patE,patET],['patrol-scan',9,patZ,patZT]]){
        const map=x+'-rate',motor=x+'-motor';
        term(map,8);term(map,9);wire('sum-red',map,'Red at gaze',seen,0);
        wire(map,motor,labels[i]+' rate',told,0);
        term(motor,i,P.cpat*speed);term(motor,0,-P.cpat*speed*speed);
        rate[i]=speed;
      }
      loss+=P.cpat*(patE*patE+patZ*patZ);
    }

    // The four readings (three for the scan) enter the rate's static map; the motor hears the rate and the
    // battery, pushes its own store and charges the battery.
    for(const [x,i,speed,told] of [['look',6,lookR,lookRT],['scan',7,lookS,lookST]]){
      const map=x+'-rate',motor=x+'-motor';
      term(map,6);term(map,7);wire('sum-red',map,'Red at gaze',seen,0);
      if(x==='look')wire('sum-climb',map,'Change with gaze',climb,0,{valueScale:3});
      wire(map,motor,x==='look'?'Look rate':'Scan rate',told,0);
      term(motor,i,P.csee*speed);term(motor,0,-P.csee*speed**2);
    }
    term('grip-rate',1);
    for(const i of [1,2])term('lift-rate',i);
    // Motors receive the actual other load, rather than differentiating their coefficients.
    const gripTorque=P.cgrip*grip-Pl,liftTorque=P.clift*lift-Ph;
    term('gripper',1,gripTorque);term('gripper',0,-gripTorque*grip);
    term('lift',2,liftTorque);term('lift',0,-liftTorque*lift);
    wire('grip-rate','gripper','Chosen grip rate',gripT,0,{valueScale:4});wire('lift-rate','lift','Chosen lift rate',liftT,0);
    // Expose computed other-load wires as values with zero returning load.
    // They are dependencies, not a force returning through a coefficient wire.
    wire('sum-grip-load','gripper','Other grip load',Pl,0);wire('sum-tip-load','lift','Other tip load',Ph,0);
    const spring=P.kappac*(y-zw),wheel=(spring+torque)/P.betac;
    term('axle',3,-spring);term('axle',5,spring);term('drive',5,torque);term('drive',0,-torque*wheel);
    // What the torque law reads enters its static map; the drive hears the torque, the axle's load on the wheel
    // and the battery.
    term('drive-torque',6);
    if(!learning){term('drive-torque',8);term('drive-torque',3);}
    wire('sum-red','drive-torque','Red at gaze',seen,0);wire('drive-torque','drive','Drive torque',torqueT,0);
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
