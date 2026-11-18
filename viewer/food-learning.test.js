const assert = require('node:assert/strict');
const F = require('./food-learning.js');
const h = .004;
function outing(s, places, time, p, sample) {
  const initial = F.field(s, places, p);
  let lost = 0, previous = initial.loss;
  for (let i=0; i<Math.round(time/h); i++) {
    if (sample && i%250===0) sample(i*h,s);
    s = F.rk4(s,h,places,p);
    const loss = F.field(s,places,p).loss;
    lost += (previous+loss)*h/2;
    previous = loss;
  }
  assert.ok(Math.abs(F.field(s,places,p).energy+lost-initial.energy)<.015);
  return s;
}
function mass(state, lo, hi) {
  const map=F.searchMap(state), n=200;
  let sum=0;
  for(let i=0;i<n;i++) sum+=F.maps(state,lo+(hi-lo)*(i+.5)/n,map)[1];
  return sum*(hi-lo)/n;
}
const p=F.DEFAULTS, records=[], samples=[];
const places=[-8,8], foods=[10,10];
let memory;
for(const x of places) {
  const s=outing(F.start([x],[10],x,p,memory),[x],14,p);
  assert.ok(s[F.NR]<.1);
  memory=F.usualMemory(s);
}
for(const seed of [37,127]) {
  const settings={...p,searchSeed:seed};
  const start=F.start(places,foods,0,settings,memory);
  const first=seed===37 ? 0 : 1, other=1-first;
  const target=F.field(start,places,settings).target;
  assert.ok(Math.abs(target-places[first])<2);
  assert.ok(Math.abs(mass(start,-9,9)-1)<1e-5);
  let inspected=false, committed=false;
  const end=outing(start,places,60,settings,(t,s)=>{
    if(t===2) committed=Math.abs(F.field(s,places,settings).target-target)<1e-8;
    if(!inspected && s[F.NR+7*first]<.1 && s[F.NR+7*other]>9){
      inspected=true;
      const here=places[first], elsewhere=places[other];
      assert.ok(mass(s,here-1,here+1)<.005);
      assert.ok(mass(s,elsewhere-1,elsewhere+1)>.5);
    }
  });
  assert.ok(committed && inspected && end[F.NR]<.1 && end[F.NR+7]<.1);
  records.push({places,foods,cart:0,seed,memory,end,time:60});
}
// One remembered site across the full test span; the old local slope stalled.
for(const x of [-8,8]) {
  const trained=outing(F.start([x],[10],x,p),[x],14,p);
  const memory=F.usualMemory(trained), start=F.start([x],[10],-x,p,memory);
  assert.ok(Math.abs(F.field(start,[x],p).target-x)<1);
  // Identical revisit histories have the same return rate, even after unequal meals.
  const cleared=F.start([],[],0,p,memory);
  const fresh=9+memory.length;
  cleared.fill(.02,fresh,fresh+memory.length);
  const rates=F.field(cleared,[],p).rate;
  assert.ok(Math.abs(rates[fresh+x+9]-rates[fresh-x+9])<1e-14);
  const end=outing(start,[x],40,p);
  assert.ok(end[F.NR]<.1);
  records.push({places:[x],foods:[10],cart:-x,seed:p.searchSeed,memory,end,time:40});
}
// Blank memory explores. With two nearby meals, a sighting overrides the search.
const blank=outing(F.start(places,foods),places,60,p);
assert.ok(blank[F.NR]<.1 && blank[F.NR+7]<.1);
const local=outing(F.start(),F.PLACES,25,p,(time,state)=>{
  const field = F.field(state, F.PLACES, p);
  if (time === 2) {
    assert.ok(state[3] < 1 && field.seen > .9 && field.torque > 0);
    assert.ok(F.maps(state, 4)[1] < .001);
  }
  samples.push({time,state,rate:field.rate});
});
assert.ok(local[F.NR]<.1 && local[F.NR+7]<.1);
// Hold the cart at zero; direct its eye to +4, away for ten units, and back.
// Both branches begin with the same real empty observation. Only the casing changes.
const fixed={...p,A:0,omega:0,gam:0,lam:0,ktau:0,memoryDrive:0};
const site=4, siteIndex=site+9;
const siteRate=s=>F.returnRate(F.learnedMemory(s).returnWeights[siteIndex]);
// Refill the first depleted casing in an ordinary outing while the cart looks away.
const departed=samples.find(({time})=>time===10).state;
assert.ok(departed[F.NR]<.1 && Math.abs(departed[3]+departed[6]-site)>1);
const replenished=departed.slice();
replenished[F.NR]=10;
const fedAgain=outing(replenished,F.PLACES,15,p);
assert.ok(siteRate(local)<siteRate(departed) && siteRate(fedAgain)>siteRate(departed));
assert.ok(outing(fedAgain,F.PLACES,5,p)[F.NR]<.1);
let observed=F.start([site],[.05],0,fixed);
observed[6]=site;
observed=outing(observed,[site],2,fixed);
const before=siteRate(observed);
assert.equal(before,siteRate(F.start()));
observed[6]=-site;
observed=outing(observed,[site],10,fixed);
const revisits=[];
for(const food of [.05,10]) {
  const start=observed.slice();
  start[F.NR]=food;
  assert.ok(F.field(start,[site],fixed).seen < 1e-8);
  assert.equal(siteRate(start),before);
  start[6]=site;
  const end=outing(start,[site],1,fixed);
  const after=siteRate(end);
  assert.ok(food<1 ? after<before : after>before);
  assert.deepEqual(F.usualMemory(end),F.usualMemory(observed));
  assert.equal(F.learnedMemory(end).returnWeights[1],0);
  const watching=outing(end,[site],2,fixed);
  assert.ok(Math.abs(siteRate(watching)-after)<1e-13);
  const memory=F.learnedMemory(end);
  assert.deepEqual(F.learnedMemory(F.start([],[],0,p,memory)),memory);
  // After a fresh outing each branch has the same yield, gaze and empty casing.
  // The learned difference must actually reach the distribution used to choose flags.
  let cleared=F.start([site],[.05],0,fixed,memory);
  cleared[6]=site;
  cleared=outing(cleared,[site],2,fixed);
  assert.equal(siteRate(cleared),after);
  cleared[6]=-site;
  cleared=outing(cleared,[site],10,fixed);
  revisits.push({start,end,after,probability:mass(cleared,3.5,4.5)});
}
assert.ok(revisits[0].probability<revisits[1].probability);
if(process.argv.includes('--json')) console.log(JSON.stringify({samples,records,revisits,fixed}));
else console.log('Search, distant recall, local depletion, energy and paired revisit learning passed.',
  'Chance of visible food after ten units (prior, empty, refilled):',
  [before,...revisits.map(r=>r.after)].map(rate=>(-Math.expm1(-10*rate)).toFixed(5)).join(', '));
