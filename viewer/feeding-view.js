/* The standalone scene and the explorer observe one food-learning outing per page.
   The model is the library's composite, feeding_composed.feeding_with_slot (the separator mouth, the husk
   collector, the pile on the ground, the gut wall in the slot, casings without a hinge), stepped by code
   GENERATED from it by scripts/export_step_js.py: one module per count of casings, feeding-slot-N.js, global
   FeedingSlotN, with step, observe, start and curve, and the metadata naming every store. No equation of the
   model is typed here; the numbers shown are read off observe's fields and the module's metadata.
   A casing added during a run loads the module for one more casing and carries the state over store by store,
   matched by name (keyOf below); the new casing's stores start as that module's start() makes them.
   The page object's observe(observer) calls observer({state, places, params, time, module, observed}).
   Claude Opus 5.5, subagent of c5f9ecc9, 2026-11-22 (P109 (f) act (1), second half). */
const FEEDING_BASE = (typeof document !== 'undefined' && document.currentScript)
  ? new URL('.', document.currentScript.src).href : '';
const FEEDING_MAX_CASINGS = 6;

function feedingModule(count) {
  return globalThis['FeedingSlot' + count] || null;
}

// Each store of a generated module named by a key that does not depend on the count of casings:
// "casing k|label" for a casing's stores, "path|label" otherwise, with an occurrence count for repeats.
function feedingStoreKeys(module) {
  const seen = new Map(), keys = new Map();
  for (const store of module.stores) {
    const base = 'casing' in store ? `casing ${store.casing}|${store.label}` : `${store.path.join('/')}|${store.label}`;
    const n = seen.get(base) || 0;
    seen.set(base, n + 1);
    keys.set(base + '|' + n, store);
  }
  return keys;
}

// Copy every store of `from` (a state of module `fromModule`) into `to` (a state of `toModule`) where
// both modules have it; the rest of `to` is left as it is.
function feedingCarry(fromModule, from, toModule, to) {
  const old = feedingStoreKeys(fromModule);
  for (const [key, store] of feedingStoreKeys(toModule)) {
    const source = old.get(key);
    if (source && source.size === store.size)
      for (let j = 0; j < store.size; j++) to[store.start + j] = from[source.start + j];
  }
  return to;
}

function loadFeedingModule(count) {
  const ready = feedingModule(count);
  if (ready) return Promise.resolve(ready);
  return new Promise((resolve, reject) => {
    const script = document.createElement('script');
    script.src = FEEDING_BASE + 'feeding-slot-' + count + '.js';
    script.onload = () => (feedingModule(count) ? resolve(feedingModule(count)) : reject(Error('no module ' + count)));
    script.onerror = () => reject(Error('could not load ' + script.src));
    document.head.appendChild(script);
  });
}

function createFeedingView(container, options = {}) {
  container.innerHTML = `<div class="wrap">
<h1 class="standalone">Feeding</h1>
  <p class="lede">The eye watches for red casings while the cart searches. When it sees one, the cart heads there to eat. The mouth tears food off the casing it holds: of each gram, a share goes into its chamber and the husks into a collector, which empties onto a pile on the ground. The gut wall in the robot's slot takes up what the chamber holds and charges the battery. A meal teaches the cart how much food a place yields; a later look teaches it how soon to come back.</p>

  <div class="ask">
    <span class="q">Will the cart find its way back to food it remembers?</span>
    <button class="primary" id="play">Play</button>
    <button id="one-step" hidden>One step</button>
    <button id="outing">New outing · keep memory</button>
    <button id="reset">Start over · forget</button>
    <span class="hint" style="margin:0">time runs at <span class="knob" id="k-speed" tabindex="0"></span> units a second</span>
  </div>

  <div class="card">
    <h2>The line</h2>
    <svg id="scene" viewBox="0 0 940 420" role="img" aria-label="The cart on the line with its chamber and husk collector, its eye's gaze, the casings on the ground, the pile of husks, and the search probability density beneath the destination flag"></svg>
    <p class="hint"><strong>Next flag · probability per unit distance.</strong> The shaded area under each stretch of ground is its chance of getting the next flag. The curve updates as the cart explores. The flag marks the destination already chosen.</p>
    <div class="tools">
      <button id="enter-receiver" hidden>Open Receiver</button>
      <button id="add-full">Add a full casing</button>
      <button id="add-empty">Add an empty casing</button>
      <span class="hint" style="margin:0" id="casing-hint">During an outing, click a depleted casing to refill it. You can also add a casing at any time, up to ${FEEDING_MAX_CASINGS}. Before Play, drag the cart or a casing, click a casing to toggle its food, or drag it off either end to remove it (one casing always stays).</span>
    </div>
    <p style="margin:var(--s3) 0 0">The battery starts at <span class="knob" id="k-batt" tabindex="0"></span>. The eye sweeps out to <span class="knob" id="k-A" tabindex="0"></span> either way, and holding its gaze costs <span class="knob" id="k-see" tabindex="0"></span>; the wheels push toward what the eye holds with a torque of <span class="knob" id="k-tau" tabindex="0"></span>, easing off only within a quarter of a unit of it. Between sightings, the cart heads toward one search destination. The flag marks that destination.</p>
  </div>

  <div class="card">
    <h2>Usual meal yield</h2>
    <svg id="maps" viewBox="0 0 940 140" role="img" aria-label="Usual meal yield along the line"></svg>
    <p class="hint">A new outing restores the starting food and keeps what the cart learned about meal size and return time.</p>
  </div>

  <div class="card">
    <h2>What moves it, now</h2>
    <div class="calc" id="calc"></div>
  </div>

  <div class="card">
    <h2>Over time</h2>
    <svg id="chart" viewBox="0 0 940 300" aria-label="Battery and food over time; the cart's place and where the eye looks"></svg>
    <div class="key">
      <span><i style="background:var(--batt)"></i>battery</span><span><i style="background:var(--food)"></i>food left in all casings</span>
      <span><i style="background:var(--husk)"></i>husks on the pile</span>
      <span><i style="background:var(--wall)"></i>the cart's place</span><span><i style="background:var(--eye)"></i>where the eye looks</span>
    </div>
  </div>

  <p class="standalone"><a href="feeding-explorer.html">Explore the feeding arrangement and its parts.</a></p>
  <div class="more">
    <button id="b-what" aria-expanded="false">What is this?</button>
    <button id="b-check" aria-expanded="false">What was checked</button>
  </div>
  <p class="explain" id="t-what" hidden>Each casing sends its redness to the scanning eye. The mouth has two halves: the spatial coupling reaches, realizing the rate at which food leaves the held casing, and the mouth settles it, keeping the pass share in its chamber and sending the husks to the collector. The robot's slot is filled by the gut wall, which takes up a share of the chamber per unit of time and pays the battery for each gram. Meal size trains the usual yield. The search curve gives less weight to neighborhoods the eye or body has just covered; its area is one. The robot pays for changing these memories, running the eye, the gripper, the lever and the wheels, and for each gram the mouth tears off, from the battery its gut wall charges.</p>
  <p class="explain" id="t-check" hidden>The page steps with JavaScript written from the library's composite by scripts/export_step_js.py, one file per count of casings; tests/test_export_step_js.py checks that its step and every number shown here equal the library's to 1e-12. viewer/feeding-slot.test.js runs whole outings on the generated code: the cart finds a remembered meal at either end of the line, two seeds send it to opposite ends first, a place it has emptied loses its search weight, and every gram the mouth takes ends in the chamber, the gut wall or the husks.</p>
</div>`;
  const elements = new Map(Array.from(container.querySelectorAll('[id]'), e => [e.id, e]));
  const element = id => elements.get(id);
  const controls = document.createElement('div');
  if (options.onChange) {
    controls.className = 'feeding-ui';
    controls.append(container.querySelector('.ask'));
    container.querySelectorAll('.standalone').forEach(e => { e.hidden = true; });
  }
  const NS = 'http://www.w3.org/2000/svg';
  const FIG = ' ';
  const fw = (v, dec, width, sign) => {          // fixed-width number (PRINCIPLES 12)
    let s = (sign && v >= 0 ? '+' : '') + v.toFixed(dec);
    if (sign && v < 0) s = '−' + (-v).toFixed(dec);
    return s.padStart(width, FIG);
  };
  const FULL = 10, EMPTY = 0.05;

  // ---------- the generated model: the module for the present count of casings ----------
  let M = feedingModule(3) || feedingModule(options.casings || 3);
  if (!M) throw Error('feeding-view.js: load feeding-slot-3.js first');
  const PARAM = M.parameters;
  const defaults = () => Float64Array.from(M.defaults);
  const knobIndex = name => M.knobs.indexOf(name);
  let places = M.places.slice(), foods = M.foods.map(f => (f > 1 ? FULL : EMPTY)), cart0 = 0, batt0 = 6;
  let params = defaults();
  let s, t, playing = false, speed = 2, hist, obs = null, switching = false;
  let carriedMemory = null;
  // Where each store sits in the present module's state.
  let layout;
  function index(module) {
    const byLabel = new Map(), casings = [];
    for (const store of module.stores) {
      if ('casing' in store) (casings[store.casing] ||= {})[store.label] = store.start;
      else if (!byLabel.has(store.label)) byLabel.set(store.label, store.start);
    }
    return {byLabel, casings};
  }
  layout = index(M);
  const at = label => {
    const i = layout.byLabel.get(label);
    if (i === undefined) throw Error('no store ' + label);
    return i;
  };
  const value = label => s[at(label)];
  const casingAt = (k, label) => layout.casings[k][label];
  const zeros = () => new Float64Array(M.centres.length);
  function startState() {
    const memory = carriedMemory || {usual: zeros(), returnWeights: zeros()};
    return Float64Array.from(M.start(places, foods, params, memory.usual, memory.returnWeights, [cart0], [batt0]));
  }
  function restart() { s = startState(); t = 0; hist = []; observeNow(); record(); }
  function observeNow() { obs = M.observe(s, places, params); }
  const totalFood = () => obs.casings.reduce((acc, v, i) => acc + (i % 4 === 0 ? v : 0), 0);
  function record() { hist.push([t, value('battery'), totalFood(), value('cart'), value('cart') + value('look'), obs.pile[0]]); }
  // Switch to the module for another count of casings, carrying every store over by its name.
  function switchTo(module, carry) {
    const old = s, oldModule = M;
    M = module; layout = index(M);
    s = startState();
    if (carry) feedingCarry(oldModule, old, M, s);
    observeNow();
  }
  function changeCount(count, carry, then) {
    if (count < 1 || count > FEEDING_MAX_CASINGS) return;
    switching = true;
    loadFeedingModule(count).then(module => { switchTo(module, carry); switching = false; then(); draw(); },
      error => { switching = false; console.error(error); });
  }
  restart();

  // ---------- knobs: a number shown is a number you can slide, where it is shown ----------
  function knob(id, get, set, lo, hi, per, fmt) {
    const el = element(id);
    const show = () => { el.textContent = fmt(get()); };
    let x0 = null, v0 = 0;
    el.addEventListener('pointerdown', e => { x0 = e.clientX; v0 = get(); el.setPointerCapture(e.pointerId); });
    el.addEventListener('pointermove', e => { if (x0 === null) return; set(Math.min(hi, Math.max(lo, v0 + (e.clientX - x0) * per))); show(); draw(); });
    el.addEventListener('pointerup', () => { x0 = null; });
    el.addEventListener('keydown', e => {
      const d = e.key === 'ArrowRight' || e.key === 'ArrowUp' ? 1 : e.key === 'ArrowLeft' || e.key === 'ArrowDown' ? -1 : 0;
      if (d) { set(Math.min(hi, Math.max(lo, get() + d * per * 10))); show(); draw(); e.preventDefault(); }
    });
    show();
    return show;
  }
  const param = name => params[knobIndex(name)];
  const setParam = (name, v) => { params[knobIndex(name)] = v; if (t === 0) restart(); else observeNow(); };
  knob('k-speed', () => speed, v => { speed = v; }, 0.2, 8, 0.02, v => fw(v, 1, 3));
  knob('k-batt', () => batt0, v => { batt0 = v; if (t === 0) restart(); }, 0.2, 10, 0.02, v => fw(v, 1, 4));
  knob('k-A', () => param('A'), v => setParam('A', v), 1, 10, 0.02, v => fw(v, 1, 4));
  knob('k-see', () => param('csee'), v => setParam('csee', v), 0.0005, 0.05, 0.0001, v => fw(v, 3, 5));
  knob('k-tau', () => param('ktau'), v => setParam('ktau', v), 0.05, 1.5, 0.01, v => fw(v, 2, 4));
  // ---------- the scene: the line seen from the side; heights drawn larger than places ----------
  const scene = element('scene');
  const X0 = 30, X1 = 910, LO = -9, HI = 9, GY = 230, HS = 150;     // world place -> screen x; height -> screen y
  const sx = p => X0 + (X1 - X0) * (p - LO) / (HI - LO);
  const px = X => LO + (X - X0) / (X1 - X0) * (HI - LO);
  const sy = x => GY - (x - PARAM.xg) * HS;
  function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || scene).appendChild(e); return e; }
  // A store's level as a bar inside the receiver: its mass, against the capacity the model gives it.
  function level(x, y, w, h, mass, capacity, color, name, parent) {
    el('rect', { x, y, width: w, height: h, rx: 3, fill: 'var(--panel)', stroke: color, 'stroke-width': 1.5 }, parent);
    const fh = (h - 4) * Math.max(0, Math.min(1, mass / capacity));
    el('rect', { x: x + 2, y: y + h - 2 - fh, width: w - 4, height: fh, rx: 2, fill: color, opacity: .55 }, parent);
    el('text', { x: x + w / 2, y: y - 4, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }, parent).textContent = name;
    el('text', { x: x + w / 2, y: y + h + 13, 'text-anchor': 'middle', 'font-size': 11, fill: color, 'font-family': 'var(--data)' }, parent).textContent = fw(mass, 2, 5);
  }
  function drawScene(samples) {
    scene.textContent = '';
    const y = value('cart'), r = value('look'), h = value('tip'), grip = value('engagement');
    const seen = obs.told[0];
    // ground and its scale
    el('rect', { x: X0, y: GY + 12, width: X1 - X0, height: 8, fill: 'var(--earth)', rx: 2 });
    for (let p = Math.ceil(LO); p <= HI; p += 1) {
      el('line', { x1: sx(p), x2: sx(p), y1: GY + 20, y2: GY + (p % 5 === 0 ? 30 : 25), stroke: 'var(--muted)' });
      if (p % 5 === 0) el('text', { x: sx(p), y: GY + 46, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' }).textContent = p;
    }
    // the pile of husks on the ground, under the line: the collector empties onto it
    const pile = obs.pile[0], pw = 30 + 50 * Math.min(1, Math.sqrt(pile / 10)), ph = 4 + 22 * Math.min(1, Math.sqrt(pile / 10));
    const pcx = X1 - 70, pby = GY + 82;
    el('path', { d: `M ${pcx - pw} ${pby} Q ${pcx} ${pby - 2 * ph} ${pcx + pw} ${pby} Z`, fill: 'var(--husk)', opacity: .8 });
    el('text', { x: pcx - pw - 8, y: pby - 2, 'text-anchor': 'end', 'font-size': 12, fill: 'var(--muted)' }).textContent = 'pile of husks on the ground';
    el('text', { x: pcx, y: pby + 15, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--husk)', 'font-family': 'var(--data)' }).textContent = fw(pile, 2, 5);
    // the eye's sweep: how far it can look either way
    const A = param('A');
    el('line', { x1: sx(y - A), x2: sx(y + A), y1: 26, y2: 26, stroke: 'var(--eye)', opacity: .25, 'stroke-width': 6, 'stroke-linecap': 'round' });
    const right = sx(y + A) + 160 < X1;
    el('text', { x: right ? sx(y + A) + 8 : Math.max(X0, sx(y - A)) - 8, y: 30, 'text-anchor': right ? 'start' : 'end',
      'font-size': 12, fill: 'var(--muted)' }).textContent = 'how far the eye sweeps';
    drawSearchDensity(samples, value('destination'));
    // the casings: closed shells, no hinge; the mouth tears the food out
    places.forEach((p, k) => {
      const f = s[casingAt(k, 'food')], x = s[casingAt(k, 'casing')];
      const red = obs.casings[4 * k + 2], cx = sx(p), cy = sy(x) + 12, w = 44, hh = 30;
      const g = el('g', { class: 'grab' });
      el('path', { d: `M ${cx - w / 2} ${cy - hh} Q ${cx - w / 2} ${cy} ${cx} ${cy} Q ${cx + w / 2} ${cy} ${cx + w / 2} ${cy - hh} Q ${cx} ${cy - hh - 8} ${cx - w / 2} ${cy - hh} Z`,
        fill: 'var(--panel)', stroke: red > 0.5 ? 'var(--food)' : 'var(--empty)', 'stroke-width': 3 }, g);
      const fh = (hh - 6) * Math.min(1, f / FULL);
      el('rect', { x: cx - w / 2 + 6, y: cy - 3 - fh, width: w - 12, height: fh, rx: 3, fill: 'var(--food)', opacity: .35 + .5 * red }, g);
      el('text', { x: cx, y: cy - hh - 12, 'text-anchor': 'middle', 'font-size': 12, fill: red > 0.5 ? 'var(--food)' : 'var(--muted)', 'font-family': 'var(--data)' }, g).textContent = fw(f, 1, 4);
      el('rect', { x: cx - w / 2 - 6, y: cy - hh - 12, width: w + 12, height: hh + 36, fill: 'transparent' }, g);
      g.addEventListener('pointerdown', e => dragStart(e, k));
    });
    // the cart: body with its chamber, husk collector and slot; wheels; the lever down to its tip at height h
    const cx = sx(y), body = sy(PARAM.x0) - 6, BW = 168, BH = 64;
    const cg = el('g', { class: 'grab' });
    el('rect', { x: cx - BW / 2, y: body - BH, width: BW, height: BH, rx: 6, fill: 'var(--panel)', stroke: 'var(--wall)', 'stroke-width': 2 }, cg);
    level(cx - BW / 2 + 12, body - BH + 18, 26, 26, value('mouth ready store'), PARAM.chamberCap, 'var(--food)', 'chamber', cg);
    level(cx - BW / 2 + 56, body - BH + 18, 26, 26, value('husk collector store'), PARAM.collectorCap, 'var(--husk)', 'husks', cg);
    el('rect', { x: cx + 6, y: body - BH + 14, width: BW / 2 - 16, height: 34, rx: 4, fill: 'var(--faint)', stroke: 'var(--open)', 'stroke-dasharray': '4 3' }, cg);
    el('text', { x: cx + 6 + (BW / 2 - 16) / 2, y: body - BH + 28, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }, cg).textContent = 'slot:';
    el('text', { x: cx + 6 + (BW / 2 - 16) / 2, y: body - BH + 42, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--open)' }, cg).textContent = 'gut wall';
    const turn = (value('cart') / 0.12) % (2 * Math.PI);
    for (const dx of [-50, 50]) {
      const wx = cx + dx;
      el('line', { x1: wx, x2: wx, y1: body, y2: GY + 12 - 14, stroke: 'var(--wall)', 'stroke-width': 2 }, cg);
      el('circle', { cx: wx, cy: GY + 12 - 14, r: 14, fill: 'var(--panel)', stroke: 'var(--wall)', 'stroke-width': 2.5 }, cg);
      el('line', { x1: wx, y1: GY - 2, x2: wx + 12 * Math.cos(turn), y2: GY - 2 + 12 * Math.sin(turn), stroke: 'var(--wall)', 'stroke-width': 2 }, cg);
    }
    el('rect', { x: cx - BW / 2 - 4, y: body - BH - 6, width: BW + 8, height: GY - body + BH + 6, fill: 'transparent' }, cg);
    cg.addEventListener('pointerdown', e => dragStart(e, 'cart'));
    if (options.openReceiver) {
      el('title', {}, cg).textContent = t > 0 ? 'Open Receiver' : 'Click to open Receiver or drag to place it';
      if (t > 0) {
        cg.style.cursor = 'pointer';
        cg.addEventListener('click', options.openReceiver);
      }
    }
    // lever: from the mouth down to its tip; the jaws close as the grip rises
    const tipy = sy(h) - 4, hold = Math.max(0, Math.min(1, (grip + 1) / 3));
    el('line', { x1: cx, x2: cx, y1: body, y2: tipy, stroke: 'var(--grip)', 'stroke-width': 4, 'stroke-linecap': 'round' });
    for (const side of [-1, 1])
      el('line', { x1: cx, y1: tipy, x2: cx + side * (26 - 12 * hold), y2: tipy + 6, stroke: 'var(--grip)', 'stroke-width': 4, 'stroke-linecap': 'round' });
    // the eye and its gaze: a line from the eye to the place it looks at
    const ex = cx + (BW / 2 - 10) * Math.sign(r || 1), eyY = body - BH - 10, lx = sx(y + r), seenUnit = Math.min(1, seen);
    el('line', { x1: ex, y1: eyY, x2: lx, y2: GY - 6, stroke: 'var(--eye)', 'stroke-width': 1.5 + 2.5 * seenUnit, 'stroke-dasharray': seen > 0.5 ? '' : '5 4', opacity: .9 });
    el('circle', { cx: ex, cy: eyY, r: 7, fill: 'var(--panel)', stroke: 'var(--eye)', 'stroke-width': 2.5 });
    el('circle', { cx: lx, cy: GY - 6, r: 5 + 6 * seenUnit, fill: seen > 0.05 ? 'var(--food)' : 'var(--eye)', opacity: .35 + .5 * seenUnit });
    // battery gauge, top left
    const GW = 160, frac = Math.min(1, value('battery') / 20);
    el('rect', { x: X0, y: 52, width: GW, height: 14, rx: 3, fill: 'none', stroke: 'var(--wall)', 'stroke-width': 1.5 });
    el('rect', { x: X0 + 2, y: 54, width: Math.max(0, (GW - 4) * frac), height: 10, rx: 2, fill: 'var(--batt)', opacity: .85 });
    el('text', { x: X0 + GW + 10, y: 64, 'font-size': 13, fill: 'var(--batt)', 'font-family': 'var(--data)' }).textContent = 'battery ' + fw(value('battery'), 2, 5);
    el('text', { x: X1, y: 64, 'text-anchor': 'end', 'font-size': 13, fill: 'var(--muted)', 'font-family': 'var(--data)' }).textContent = 'time ' + fw(t, 1, 5);
  }
  let drag = null, moved = false;
  function dragStart(e, what) {
    if (playing || t > 0) {
      if (what !== 'cart' && obs.casings[4 * what + 2] < 0.5) { s[casingAt(what, 'food')] = FULL; observeNow(); draw(); }
      return;
    }
    drag = what; moved = false; scene.setPointerCapture(e.pointerId);
  }
  function svgX(e) { const b = scene.getBoundingClientRect(); return (e.clientX - b.left) * 940 / b.width; }
  scene.addEventListener('pointermove', e => {
    if (drag === null) return;
    moved = true;
    const p = Math.round(px(svgX(e)) * 10) / 10;
    if (drag === 'cart') cart0 = Math.min(HI - 0.5, Math.max(LO + 0.5, p));
    else places[drag] = p;
    restart(); draw();
  });
  scene.addEventListener('pointerup', () => {
    if (drag === null) return;
    if (drag === 'cart' && !moved && options.openReceiver) {
      drag = null;
      options.openReceiver();
      return;
    }
    const k = drag;
    drag = null;
    if (k !== 'cart') {
      if (!moved) foods[k] = foods[k] > 1 ? EMPTY : FULL;
      else if ((places[k] < LO + 0.3 || places[k] > HI - 0.3) && places.length > 1) {
        places.splice(k, 1); foods.splice(k, 1);
        changeCount(places.length, false, restart);
        return;
      }
    }
    restart(); draw();
  });
  function freePlace() {
    // the free place farthest from everything already on the line; while running, the casing joins the run as it is
    const here = t > 0 ? value('cart') : cart0;
    let best = 0, bestd = -1;
    for (let p = LO + 1; p <= HI - 1; p += 0.5) {
      const d = Math.min(Math.abs(p - here), ...places.map(q => Math.abs(p - q)));
      if (d > bestd) { bestd = d; best = p; }
    }
    return best;
  }
  function addCasing(food, place = freePlace()) {
    if (!Number.isFinite(place) || !Number.isFinite(food) || food < 0) throw Error("Invalid casing");
    if (places.length >= FEEDING_MAX_CASINGS || switching) return Promise.resolve(false);
    places.push(place); foods.push(food);
    return new Promise(resolve => changeCount(places.length, t > 0, () => {
      if (t === 0) { hist = []; }
      else if (hist.at(-1)[0] === t) hist.pop();
      record(); resolve(true);
    }));
  }
  element('add-full').addEventListener('click', () => addCasing(FULL));
  element('add-empty').addEventListener('click', () => addCasing(EMPTY));

  // ---------- the calculation of the present moment, in the order the mathematics has it ----------
  function drawCalc(atCart) {
    const rate = label => obs.rate[at(label)];
    const [seen, , , , , taken] = obs.told;
    const [torque, gripRate, liftRate, free] = obs.told_motors;
    const [gripper, lever, wheels, eye, memory, latches, mouth, gut, total] = obs.pays;
    const [, intoChamber, husks, uptake, toPile] = obs.flows;
    let k = -1, most = 0;
    places.forEach((_, i) => { const d = -obs.rate[casingAt(i, 'food')]; if (d > most) { most = d; k = i; } });
    element('calc').innerHTML =
      `<p>The <span class="e">eye</span> looks <span class="out">${fw(value('look'), 2, 6, true)}</span> from the cart, at the place <span class="out">${fw(value('cart') + value('look'), 2, 6, true)}</span>; redness there: <span class="out">${fw(seen, 2, 4)}</span>. So its scan runs at <span class="out">${fw(free, 2, 4)}</span> of full speed, and its look distance changes at <span class="out">${fw(rate('look'), 2, 6, true)}</span>.</p>` +
      `<p>At the cart, the usual meal score is <span class="out">${fw(atCart.usual, 2, 5)}</span>. The search density here is <span class="out">${fw(atCart.density, 2, 5)}</span>. The search destination is <span class="out">${fw(value('destination'), 2, 5, true)}</span>. The wheels apply torque <span class="out">${fw(torque, 3, 6, true)}</span>.</p>` +
      `<p>The <span class="l">gripper</span> is told the rate <span class="out">${fw(gripRate, 2, 6, true)}</span> (grip <span class="out">${fw(value('engagement'), 2, 5, true)}</span>), the <span class="l">lever</span> the rate <span class="out">${fw(liftRate, 2, 6, true)}</span> (tip at height <span class="out">${fw(value('tip'), 2, 5, true)}</span>).` +
      (k >= 0 && taken > 1e-4 ? ` The mouth tears food off casing ${k + 1} at <span class="out">${fw(taken, 2, 5)}</span> grams per unit of time.</p>` : ` The mouth takes nothing.</p>`) +
      `<p>Of each gram taken, <span class="out">${PARAM.mouthPass.toFixed(2)}</span> goes into the <span class="f">chamber</span> (<span class="out">${fw(intoChamber, 2, 5)}</span>) and the rest, husks, into the <span class="a">collector</span> (<span class="out">${fw(husks, 2, 5)}</span>), which empties onto the pile at <span class="out">${fw(toPile, 2, 5)}</span>. The <span class="a">gut wall</span> takes up <span class="out">${fw(uptake, 2, 5)}</span> from the chamber, which holds <span class="out">${fw(value('mouth ready store'), 2, 5)}</span>.</p>` +
      `<p>The <span class="b">battery</span> moves by what the gut wall gives, <span class="out">${fw(gut, 3, 6, true)}</span> (<span class="out">${PARAM.absorbYield.toFixed(2)}</span> per gram taken up), what the mouth pays, <span class="out">${fw(mouth, 3, 6, true)}</span> (<span class="out">${PARAM.mouthCost.toFixed(2)}</span> per gram torn off), and what the motors pay: gripper <span class="out">${fw(gripper, 3, 6, true)}</span>, lever <span class="out">${fw(lever, 3, 6, true)}</span>, wheels <span class="out">${fw(wheels, 3, 6, true)}</span>, eye <span class="out">${fw(eye, 3, 6, true)}</span>, memory <span class="out">${fw(memory, 3, 6, true)}</span>, contact latches <span class="out">${fw(latches, 3, 6, true)}</span>; in all <span class="out">${fw(total, 3, 6, true)}</span>.</p>`;
  }

  // ---------- the record over time (the axis compresses; the page never grows) ----------
  const chart = element('chart');
  function drawChart() {
    chart.textContent = '';
    const T = Math.max(20, t), C0 = 60, C1 = 920;
    const xs = tt => C0 + (C1 - C0) * tt / T;
    const panel = (top, hgt, lo, hi, cols, label) => {
      const ys = v => top + hgt - hgt * (v - lo) / (hi - lo);
      el('line', { x1: C0, x2: C1, y1: top + hgt, y2: top + hgt, stroke: 'var(--rule)' }, chart);
      if (lo < 0) el('line', { x1: C0, x2: C1, y1: ys(0), y2: ys(0), stroke: 'var(--rule)', 'stroke-dasharray': '2 3' }, chart);
      el('text', { x: C0 - 8, y: ys(hi) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)' }, chart).textContent = hi;
      el('text', { x: C0 - 8, y: ys(lo) + 4, 'text-anchor': 'end', 'font-size': 11, fill: 'var(--muted)' }, chart).textContent = lo;
      el('text', { x: C0, y: top - 6, 'font-size': 12, fill: 'var(--muted)' }, chart).textContent = label;
      for (const [i, col, wdt] of cols) {
        const d = hist.map((hh, k) => `${k ? 'L' : 'M'} ${xs(hh[0]).toFixed(1)} ${ys(Math.min(hi, Math.max(lo, hh[i]))).toFixed(1)}`).join(' ');
        el('path', { d, fill: 'none', stroke: col, 'stroke-width': wdt || 2 }, chart);
      }
    };
    const top = Math.max(...hist.map(hh => Math.max(hh[1], hh[2], hh[5])));
    const fmax = Math.max(12, Math.ceil(top / 5) * 5);
    panel(20, 100, 0, fmax, [[1, 'var(--batt)'], [2, 'var(--food)'], [5, 'var(--husk)']], 'battery and grams');
    panel(150, 120, LO, HI, [[4, 'var(--eye)', 1.2], [3, 'var(--wall)', 2.5]], 'place on the line');
    el('line', { x1: xs(t), x2: xs(t), y1: 14, y2: 274, stroke: 'var(--accent)', 'stroke-width': 1 }, chart);
    const step = T > 60 ? 20 : T > 30 ? 10 : 5;
    for (let k = 0; k <= T; k += step)
      el('text', { x: xs(k), y: 292, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }, chart).textContent = k;
  }

  // The usual yield and the search density along the line, and at the cart (generated: curve).
  function mapSamples() {
    const c = M.curve(s, [value('cart')]);
    const samples = M.samples.map((place, i) => ({x: sx(place), yieldScore: c.usual[i], density: c.density[i]}));
    const last = M.samples.length;
    return {samples, atCart: {usual: c.usual[last], density: c.density[last]}};
  }

  function curvePath(samples, ordinate) {
    return samples.map((sample, i) =>
      (i ? 'L' : 'M') + sample.x.toFixed(2) + ' ' + ordinate(sample).toFixed(2)).join(' ');
  }

  function drawSearchDensity(samples, target) {
    const plot = el('g', {id: 'search-density'});
    const top = GY + 112, bottom = GY + 180;
    const peak = Math.max(...samples.map(sample => sample.density));
    const limit = Math.max(0.1, Math.ceil(peak * 10) / 10);
    const ys = density => bottom - (bottom - top) * density / limit;
    for (const v of [0, limit]) {
      const y = ys(v);
      el('line', {x1: X0, x2: X1, y1: y, y2: y, stroke: 'var(--rule)'}, plot);
      el('text', {x: X0 - 5, y: y + 4, 'text-anchor': 'end', 'font-size': 11}, plot).textContent = v;
    }
    const curve = curvePath(samples, sample => ys(sample.density));
    el('path', {d: curve + ` L ${X1} ${bottom} L ${X0} ${bottom} Z`, fill: 'var(--open)', opacity: 0.19}, plot);
    el('path', {d: curve, fill: 'none', stroke: 'var(--open)', 'stroke-width': 2}, plot);
    const x = sx(target);
    el('line', {x1: x, x2: x, y1: GY + 102, y2: bottom, stroke: 'var(--open)', 'stroke-dasharray': '3 3'}, plot);
    el('path', {d: `M ${x} ${GY+102} v -20 l 12 4 l -12 5`, fill: 'var(--open)', stroke: 'var(--open)'}, plot);
  }

  function drawYield(samples) {
    const plot = element('maps');
    plot.textContent = '';
    const top = Math.max(1, Math.ceil(Math.max(...samples.map(sample => sample.yieldScore))));
    const ys = v => 95 - 65 * v / top;
    el('line', {x1:X0, x2:X1, y1:ys(0), y2:ys(0), stroke:'var(--rule)'}, plot);
    for (const v of [0,top]) el('text', {x:X0-8, y:ys(v)+4, 'text-anchor':'end', fill:'var(--muted)'}, plot).textContent = v;
    for (const place of [-8,-4,0,4,8]) {
      el('text', {x:sx(place), y:135, 'text-anchor':'middle', fill:'var(--muted)'}, plot).textContent = place;
    }
    el('path', {d: curvePath(samples, sample => ys(sample.yieldScore)), fill:'none', stroke:'var(--open)', 'stroke-width':3}, plot);
  }

  let lastObservation = -Infinity;
  function draw(force = true) {
    const now = performance.now();
    if (!force && now - lastObservation < 100) return;     // numbers rewritten about ten times a second
    lastObservation = now;
    observeNow();
    const {samples, atCart} = mapSamples();
    drawScene(samples); drawYield(samples); drawChart(); drawCalc(atCart);
    element('add-full').disabled = element('add-empty').disabled = places.length >= FEEDING_MAX_CASINGS;
    options.onChange?.();
  }

  // ---------- running: time on the page is the model's time ----------
  const H = M.h;
  let last = null, animation, owed = 0;
  function step(dt) {
    // The generated step is four Runge-Kutta rounds of the composite's fixed length h; what is left
    // of dt is owed to the next call.
    owed += dt;
    while (owed > H / 2) {
      s = M.step(s, places, params); t += H; owed -= H;
      if (t - hist[hist.length - 1][0] > 0.05) { observeNow(); record(); }
    }
  }
  function frame(now) {
    if (!playing) { last = null; return; }
    if (last !== null && !switching) step(Math.min(0.1, (now - last) / 1000) * speed);
    last = now;
    draw(false);
    if (playing) animation = requestAnimationFrame(frame);
  }
  const playBtn = element('play');
  function setPlaying(on) {
    cancelAnimationFrame(animation);
    last = null;
    playing = on;
    playBtn.textContent = playing ? 'Pause' : 'Play';
    if (playing) animation = requestAnimationFrame(frame);
  }
  playBtn.addEventListener('click', () => { setPlaying(!playing); draw(); });
  const stepButton = element('one-step');
  if (options.onChange) {
    stepButton.hidden = false;
    stepButton.addEventListener('click', () => { setPlaying(false); step(.02); draw(); });
  }
  const enterButton = element('enter-receiver');
  if (options.openReceiver) {
    enterButton.hidden = false;
    enterButton.addEventListener('click', options.openReceiver);
  }
  const learned = () => ({usual: s.slice(at('usual yield'), at('usual yield') + M.centres.length),
    returnWeights: s.slice(at('return rate'), at('return rate') + M.centres.length)});
  element('outing').addEventListener('click', () => {
    setPlaying(false);
    params[knobIndex('searchSeed')] = crypto.getRandomValues(new Uint32Array(1))[0];
    carriedMemory = learned();
    restart(); draw();
  });
  element('reset').addEventListener('click', () => {
    setPlaying(false);
    params[knobIndex('searchSeed')] = M.defaults[knobIndex('searchSeed')];
    carriedMemory = null; restart(); draw();
  });
  for (const k of ['what', 'check']) {
    const b = element('b-' + k), p = element('t-' + k);
    b.addEventListener('click', () => { p.hidden = !p.hidden; b.setAttribute('aria-expanded', String(!p.hidden)); });
  }
  // #t=N opens the page already run to time N (for checking a page mid-run in headless Chrome; PRINCIPLES.md)
  const jump = /t=([\d.]+)/.exec(location.hash);
  if (jump && Number.isFinite(+jump[1])) step(+jump[1]);
  draw();
  return {
    controls,
    get playing() { return playing; },
    setPlaying, addCasing,
    observe(observer) { return observer({state: s, places, params, time: t, module: M, observed: obs}); }
  };
}

if (typeof module !== 'undefined' && module.exports) module.exports = {feedingStoreKeys, feedingCarry};
