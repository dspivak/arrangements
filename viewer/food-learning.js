// The search is a tangent IMC receiver. Its spatial parameters, clock and
// destination are stores driven through the same battery-paid motors as the eye.
(function (root) {
  const base = typeof module !== 'undefined' ? require('./feeding-model.js') : root.Feeding;
  const centres = Array.from({length: 19}, (_, i) => i - 9);
  const usual = 9, fresh = usual + centres.length, goal = fresh + centres.length;
  const clock = goal + 1, weights = clock + 1, contact = weights + centres.length;
  const empty = contact + centres.length, age = empty + centres.length, NR = age + centres.length;
  const PLACES = [4, 8, -5], FOODS = [10, 10, 0.05];
  const DEFAULTS = {...base.DEFAULTS, memoryDrive: 0.2, memoryCost: 0.001, searchSeed: 37};
  const bump = distance => Math.exp(-distance * distance);
  const basis = centres.map(x => centres.map(i => bump(x - i)));
  const usualMemory = state => state.slice(usual, fresh);
  const learnedMemory = state => ({usual: usualMemory(state), returnWeights: state.slice(weights,contact)});
  // One parameter for the return rate at each center; zero starts it at 1/80.
  const returnRate = w => .05*(1+Math.tanh((w-Math.log(7))/2));
  const usualAt = (state, place) => centres.reduce((sum,c,i) => sum + state[usual+i]*bump(place-c), 0);
  const left = centres[0], right = centres[centres.length-1];
  const massScale = centres.map(c => c===left || c===right ? 2 : 1);
  // Clipped, normalized smooth bumps. Their integral on the displayed line is one.
  function componentCDF(x, c, i) {
    return massScale[i]*(base.theta((x-c+1)/2) - (c===left ? 0.5 : 0));
  }
  function searchMap(state) {
    const yieldAt = basis.map(row => row.reduce((sum,value,i) => sum + value*state[usual+i],0));
    const richness = yieldAt.map(x => x*base.theta(x/0.05));
    const masses = richness.map((x,i) => 0.00001 + state[fresh+i]**2*(0.012 + x*x));
    const total = masses.reduce((a,b)=>a+b,0);
    return {richness, masses, total};
  }
  function density(map,x) {
    if (x<left || x>right) return 0;
    return centres.reduce((sum,c,i) => sum + map.masses[i]*massScale[i]*0.5*base.dtheta((x-c+1)/2),0)/map.total;
  }
  // A seeded tape of reproducible uniform variates, indexed only while choosing.
  function variate(seed, n) {
    let x = (seed + Math.imul(n+1,0x9e3779b9)) >>> 0;
    x = Math.imul(x ^ (x >>> 16),0x21f0aaad);
    x = Math.imul(x ^ (x >>> 15),0x735a2d97);
    return (( (x ^ (x >>> 15)) >>> 0) + 0.5)/4294967296;
  }
  function quantile(map,u) {
    let lo=left, hi=right;
    for(let k=0;k<32;k++) {
      const x=(lo+hi)/2;
      const mass=centres.reduce((sum,c,i)=>sum+map.masses[i]*componentCDF(x,c,i),0);
      if(mass < u*map.total) lo=x; else hi=x;
    }
    return (lo+hi)/2;
  }
  function start(places=PLACES, foods=FOODS, cart=0, p=DEFAULTS, memory={}) {
    const body=base.start(places,foods,cart,p);
    body[0]=6;
    const zeros=centres.map(()=>0);
    const yields=Array.isArray(memory) || ArrayBuffer.isView(memory) ? memory : memory.usual ?? zeros;
    const speeds=memory.returnWeights ?? zeros;
    if(yields.length!==centres.length || speeds.length!==centres.length) throw Error('Expected one parameter per center');
    const state=[...body.slice(0,8),0,...yields,...centres.map(()=>1),cart,1.7,
      ...speeds,...zeros,...zeros,...zeros,...body.slice(base.NR)];
    state[goal]=quantile(searchMap(state),variate(p.searchSeed,0));
    return state;
  }
  function field(state, places, p) {
    const cart=state[3], gaze=cart+state[6], meal=state[8], target=state[goal];
    const pull=p.memoryDrive*Math.tanh((target-cart)/1.5);
    const body=[...state.slice(0,8),0,0,...state.slice(NR)];
    const f=base.field(body,places,p,{pull});
    const mealRate=f.eatAll-4*f.live*base.theta(-state[1])*meal;
    const release=base.dtheta(state[1])*(-f.grip)*base.theta(-f.grip/0.05);
    const mealError=-Math.expm1(-meal/4)-usualAt(state,cart);
    const map=searchMap(state);
    const usualRates=[], freshRates=[], weightRates=[], contactRates=[], emptyRates=[], ageRates=[];
    const red=base.theta(f.seen/.5);
    const angle=state[clock], cosine=Math.cos(angle);
    const choosing=base.theta((cosine-0.4)/0.35);
    const finishChoice=base.theta((cosine-0.1)/0.25);
    const atTarget=base.theta((2.25-(target-cart)**2)/1.25);
    const awakeAndSearching=f.live*f.free;
    const clockRate=awakeAndSearching*(atTarget+(1-atTarget)*finishChoice);
    let goalRate=0;
    if(choosing>0){
      const n=Math.floor(angle/(2*Math.PI)+0.5);
      goalRate=4*awakeAndSearching*choosing*(quantile(map,variate(p.searchSeed,n))-target);
    }
    let cost=mealRate**2+clockRate**2+goalRate**2;
    for(let i=0;i<centres.length;i++){
      const here=bump(cart-centres[i]), there=bump(gaze-centres[i]);
      const a=release*mealError*here;
      const availability=state[fresh+i];
      const speed=returnRate(state[weights+i]);
      const b=f.live*(speed*(1-availability)
        -(4*here+4*there)*availability);
      // The rising edge samples the previous visit before sustained sight writes the new one.
      const looking=base.theta((.55**2-(gaze-centres[i])**2)/(.75*.55**2));
      const c=10*state[contact+i], lastEmpty=10*state[empty+i];
      const dc=12*f.live*(looking-c);
      const arrival=2.5*base.dtheta(2.5*(c-.1))*dc*base.theta(dc/.1);
      const write=12*f.live*looking*base.theta((c-.55)/.25);
      const cleared=base.theta((lastEmpty-.5)/.4);
      const elapsed=80*state[age+i];
      const exposure=speed*elapsed;
      const prediction=-Math.expm1(-exposure);
      const slope=exposure*Math.exp(-exposure)*(1-speed/.1);
      const dw=4*arrival*cleared*(red-prediction)*slope;
      const de=.1*write*(1-red-lastEmpty);
      const dt=f.live*cleared*(1-looking)/80-write*state[age+i];
      usualRates.push(a); freshRates.push(b); weightRates.push(dw);
      contactRates.push(.1*dc); emptyRates.push(de); ageRates.push(dt);
      cost+=a*a+b*b+dw*dw+(.1*dc)**2+de*de+dt*dt;
    }
    f.pay.memory=p.memoryCost*cost;
    f.rate[0]-=f.pay.memory;
    f.rate=[...f.rate.slice(0,8),mealRate,...usualRates,...freshRates,goalRate,clockRate,
      ...weightRates,...contactRates,...emptyRates,...ageRates,...f.rate.slice(base.NR)];
    f.loss+=f.pay.memory;
    f.pull=pull; f.target=target;
    return f;
  }
  const rk4=(s,h,places,p)=>base.rk4(s,h,places,p,field);
  const maps=(state,place,map=searchMap(state))=>[usualAt(state,place),density(map,place)];
  const api={DEFAULTS,NR,NC:base.NC,PLACES,FOODS,start,field,rk4,maps,searchMap,usualMemory,learnedMemory,returnRate,casing:base.casing};
  if(typeof module!=='undefined' && module.exports)module.exports=api;
  else root.FoodLearning=api;
})(this);
