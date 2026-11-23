// Whole search trips on the page's generated model: feeding-slot-N.js, written from the library composite
// feeding_composed.feeding_with_slot by scripts/export_step_js.py. Nothing about feeding is typed here; the
// checks read the generated module's stores (by name, from its metadata), observe and curve.
// Ported from food-learning.test.js (the hand-typed hinge model, retired 2026-11-23) by a subagent of
// c5f9ecc9, Claude Opus 5.5. What changed: the scenarios are the same places, foods, seeds and durations;
// the paired revisit learning, which needs knobs the page does not export (A, omega, gam, lam, ktau and
// memoryDrive at zero), runs in Python on the library composite, tests/test_feeding_slot_learning.py; the
// check that identical revisit histories give equal return rates, which read the hand-typed field's rates
// with no casing, is not ported (a generated module has at least one casing). New: the husks are
// (1 - pass share) of every gram taken, and a casing added mid-run, empty and far away, changes nothing.
//   node feeding-slot.test.js
const assert = require('node:assert/strict');
const {feedingCarry} = require('./feeding-view.js');
const modules = {};
const load = n => (modules[n] ||= require(`./feeding-slot-${n}.js`));
const M1 = load(1), M2 = load(2), M3 = load(3);

function stores(M) {
  const byLabel = new Map(), casings = [];
  for (const s of M.stores) {
    if ('casing' in s) (casings[s.casing] ||= {})[s.label] = s;
    else if (!byLabel.has(s.label)) byLabel.set(s.label, s);
  }
  return {at: label => byLabel.get(label).start, casing: (k, label) => casings[k][label].start,
          slice: (state, label) => Array.from(state.slice(byLabel.get(label).start, byLabel.get(label).start + byLabel.get(label).size))};
}
const params = (M, changes = {}) => Float64Array.from(M.knobs.map((k, i) => changes[k] ?? M.defaults[i]));
const zeros = M => new Float64Array(M.centres.length);
function start(M, places, foods, {cart = 0, memory = null, seed} = {}) {
  const p = params(M, seed === undefined ? {} : {searchSeed: seed});
  const usual = memory ? memory : zeros(M);
  return {M, places, p, s: Float64Array.from(M.start(places, foods, p, usual, zeros(M), [cart], [6]))};
}
// Run for `time` units; `sample(t, run)` is called every unit of time.
function outing(run, time, sample) {
  const steps = Math.round(time / run.M.h), every = Math.round(1 / run.M.h);
  for (let i = 0; i < steps; i++) {
    if (sample && i % every === 0) sample(i * run.M.h, run);
    run.s = run.M.step(run.s, run.places, run.p);
  }
  return run;
}
const food = (run, k) => run.s[stores(run.M).casing(k, 'food')];
const usual = run => stores(run.M).slice(run.s, 'usual yield');
// The search density's mass over [lo, hi], by the trapezoid rule on the module's sample places.
function mass(run, lo, hi) {
  const {density} = run.M.curve(run.s, [0]);
  let sum = 0;
  for (let i = 1; i < run.M.samples.length; i++) {
    const a = run.M.samples[i - 1], b = run.M.samples[i];
    if (a >= lo - 1e-9 && b <= hi + 1e-9) sum += (b - a) * (density[i - 1] + density[i]) / 2;
  }
  return sum;
}
const destination = run => run.s[stores(run.M).at('destination')];

// Training: one full casing at each end, eaten in 14 units; the usual yield is remembered.
let memory = null;
for (const x of [-8, 8]) {
  const run = outing(start(M1, [x], [10], {cart: x, memory}), 14);
  assert.ok(food(run, 0) < .1, `the casing at ${x} is eaten in 14 units`);
  memory = usual(run);
}

// Two seeds send a cart that remembers both ends to opposite ends first; it feeds at both.
const places = [-8, 8], foods = [10, 10];
for (const seed of [37, 127]) {
  const run = start(M2, places, foods, {memory, seed});
  const first = seed === 37 ? 0 : 1, other = 1 - first;
  const target = destination(run);
  assert.ok(Math.abs(target - places[first]) < 2, `seed ${seed} first heads to ${places[first]}`);
  assert.ok(Math.abs(mass(run, -9, 9) - 1) < 1e-3, 'the search density has area one');
  let inspected = false, committed = false;
  outing(run, 60, (t, r) => {
    if (t === 2) committed = Math.abs(destination(r) - target) < 1e-8;
    if (!inspected && food(r, first) < .1 && food(r, other) > 9) {
      inspected = true;
      assert.ok(mass(r, places[first] - 1, places[first] + 1) < .005, 'an emptied place loses its search weight');
      assert.ok(mass(r, places[other] - 1, places[other] + 1) > .5, 'the other remembered place gains it');
    }
  });
  assert.ok(committed && inspected && food(run, 0) < .1 && food(run, 1) < .1, `seed ${seed} feeds at both ends`);
}

// Distant recall: trained at one end, started at the other, the cart heads back and eats.
for (const x of [-8, 8]) {
  const trained = outing(start(M1, [x], [10], {cart: x}), 14);
  const run = start(M1, [x], [10], {cart: -x, memory: usual(trained)});
  assert.ok(Math.abs(destination(run) - x) < 1, `recall of ${x} from ${-x}`);
  assert.ok(food(outing(run, 40), 0) < .1, `the cart reaches ${x} and eats`);
}

// Blank memory explores and finds both casings.
const blank = outing(start(M2, places, foods), 60);
assert.ok(food(blank, 0) < .1 && food(blank, 1) < .1, 'a blank memory finds both casings');

// The page's start: at 2 the eye holds +4 and the wheels push toward it; +4 has no search weight.
// The husks are (1 - pass share) of every gram taken, all along.
const local = start(M3, M3.places, M3.foods);
const before = M3.places.map((_, k) => food(local, k));
const husks = r => r.s[stores(M3).at('husk collector store')] + r.s[stores(M3).at('pile on the ground store')];
let husksChecked = 0;
outing(local, 35, (t, r) => {
  const o = M3.observe(r.s, r.places, r.p);
  if (t === 2) {
    assert.ok(r.s[stores(M3).at('cart')] < 1 && o.told[0] > .9 && o.told_motors[0] > 0, 'at 2 the eye holds +4');
    const {density} = M3.curve(r.s, [4]);
    assert.ok(density[M3.samples.length] < .001, 'no search weight at +4 while it is in view');
  }
  const taken = before.reduce((acc, f, k) => acc + f - food(r, k), 0);
  assert.ok(Math.abs(husks(r) - (1 - M3.parameters.mouthPass) * taken) < 1e-9, 'husks are the rest of each gram');
  husksChecked += taken > 1;
});
assert.ok(food(local, 0) < .1 && food(local, 1) < .1 && husksChecked > 10, 'the cart eats +4 and +8');

// A casing added mid-run, empty and far from the cart, changes nothing: the state carried into the
// three-casing module steps as the two-casing one does.
const two = outing(start(M2, [4, 8], [10, 10]), 5);
const three = {M: M3, places: [4, 8, -8.5], p: two.p,
               s: Float64Array.from(M3.start([4, 8, -8.5], [10, 10, .05], two.p, zeros(M3), zeros(M3), [0], [6]))};
feedingCarry(M2, two.s, M3, three.s);
outing(two, 2); outing(three, 2);
const keys = require('./feeding-view.js').feedingStoreKeys;
let largest = 0;
for (const [key, store] of keys(M2)) {
  const other = keys(M3).get(key);
  for (let j = 0; j < store.size; j++) largest = Math.max(largest, Math.abs(two.s[store.start + j] - three.s[other.start + j]));
}
assert.ok(largest < 1e-12, `an empty far casing changed the others by ${largest}`);

console.log('Search to remembered food, distant recall, local depletion, blank search, husk accounting and a casing',
  'added mid-run passed on the generated model. Largest change from the added casing:', largest);
