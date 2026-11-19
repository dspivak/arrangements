/* The standalone scene and the explorer observe one food-learning outing per page. */
function createFeedingView(container, options = {}) {
  container.innerHTML = `<div class="wrap">
<h1 class="standalone">Feeding</h1>
  <p class="lede">The eye watches for red casings while the cart searches. When it sees one, the cart heads there to eat. A meal teaches the cart how much food a place yields. A later look teaches it how soon to come back. If the place is still empty, its chance of getting the next flag recovers more slowly.</p>

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
    <svg id="scene" viewBox="0 0 940 392" role="img" aria-label="The cart on the line, its eye's gaze, the casings on the ground, and the search probability density beneath the destination flag"></svg>
    <p class="hint"><strong>Next flag · probability per unit distance.</strong> The shaded area under each stretch of ground is its chance of getting the next flag. The curve updates as the cart explores. The flag marks the destination already chosen.</p>
    <div class="tools">
      <button id="enter-receiver" hidden>Open Receiver</button>
      <button id="add-full">Add a full casing</button>
      <button id="add-empty">Add an empty casing</button>
      <span class="hint" style="margin:0">During an outing, click a depleted casing to refill it. You can also add a casing at any time. Before Play, drag the cart or a casing, click a casing to toggle its food, or drag it off either end to remove it.</span>
    </div>
    <p style="margin:var(--s3) 0 0">The battery starts at <span class="knob" id="k-batt" tabindex="0"></span>. The eye sweeps out to <span class="knob" id="k-A" tabindex="0"></span> either way, and holding its gaze costs <span class="knob" id="k-see" tabindex="0"></span>; the wheels push with <span class="knob" id="k-tau" tabindex="0"></span> per unit of distance to what the eye holds. Between sightings, the cart heads toward one search destination. The flag marks that destination.</p>
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
      <span><i style="background:var(--wall)"></i>the cart's place</span><span><i style="background:var(--eye)"></i>where the eye looks</span>
    </div>
  </div>

  <p class="standalone"><a href="feeding-explorer.html">Explore the food-learning receiver and its parts.</a></p>
  <div class="more">
    <button id="b-what" aria-expanded="false">What is this?</button>
    <button id="b-check" aria-expanded="false">What was checked</button>
  </div>
  <p class="explain" id="t-what" hidden>Each casing sends its redness to the scanning eye. Meal size trains its usual yield. The search curve gives less weight to neighborhoods the eye or body has just covered. Its area is one. It pays for changing these memories, running the eye and driving the wheels from the same battery that its meals recharge.</p>
  <p class="explain" id="t-check" hidden>The learning receiver is run in tangent IMC. A test compares its library steps with the browser calculation. When the eye returns after ten time units, an empty casing lowers the predicted chance of finding food after that wait, and a refilled casing raises it. With meals remembered at −8 and +8, two runs go to opposite ends first. Each cart feeds at both ends.</p>
</div>`;
  const elements = new Map(Array.from(container.querySelectorAll('[id]'), e => [e.id, e]));
  const element = id => elements.get(id);
  const controls = document.createElement('div');
  if (options.onChange) {
    controls.className = 'feeding-ui';
    controls.append(container.querySelector('.ask'));
    container.querySelectorAll('.standalone').forEach(e => { e.hidden = true; });
  }
  const F = window.FoodLearning;
  const P = { ...F.DEFAULTS };
  const NS = 'http://www.w3.org/2000/svg';
  const FIG = ' ';
  const fw = (v, dec, width, sign) => {          // fixed-width number (PRINCIPLES 12)
    let s = (sign && v >= 0 ? '+' : '') + v.toFixed(dec);
    if (sign && v < 0) s = '−' + (-v).toFixed(dec);
    return s.padStart(width, FIG);
  };
  const FULL = 10, EMPTY = 0.05;
  let places = F.PLACES.slice(), foods = F.FOODS.map(f => (f > 1 ? FULL : EMPTY)), cart0 = 0, batt0 = 6;
  let s, t, playing = false, speed = 2, hist;
  let carriedMemory;
  function restart() { s = F.start(places, foods, cart0, P, carriedMemory); s[0] = batt0; t = 0; hist = []; record(); }
  const totalFood = () => places.reduce((acc, _, k) => acc + s[F.NR + F.NC * k], 0);
  function record() { hist.push([t, s[0], totalFood(), s[3], s[3] + s[6]]); }
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
  knob('k-speed', () => speed, v => { speed = v; }, 0.2, 8, 0.02, v => fw(v, 1, 3));
  knob('k-batt', () => batt0, v => { batt0 = v; if (t === 0) { s[0] = v; hist = []; record(); } }, 0.2, 10, 0.02, v => fw(v, 1, 4));
  knob('k-A', () => P.A, v => { P.A = v; if (t === 0) { s[7] = v; s[6] = 0; } }, 1, 10, 0.02, v => fw(v, 1, 4));
  knob('k-see', () => P.csee, v => { P.csee = v; }, 0, 0.05, 0.0001, v => fw(v, 3, 5));
  knob('k-tau', () => P.ktau, v => { P.ktau = v; }, 0.02, 1, 0.002, v => fw(v, 2, 4));
  // ---------- the scene: the line seen from the side; heights drawn larger than places ----------
  const scene = element('scene');
  const X0 = 30, X1 = 910, LO = -9, HI = 9, GY = 230, HS = 150;     // world place -> screen x; height -> screen y
  const sx = p => X0 + (X1 - X0) * (p - LO) / (HI - LO);
  const px = X => LO + (X - X0) / (X1 - X0) * (HI - LO);
  const sy = x => GY - (x - P.xg) * HS;
  function el(tag, attrs, parent) { const e = document.createElementNS(NS, tag); for (const k in attrs) e.setAttribute(k, attrs[k]); (parent || scene).appendChild(e); return e; }
  function drawScene(samples, fd) {
    scene.textContent = '';
    const y = s[3], r = s[6], h = s[2];
    // ground and its scale
    el('rect', { x: X0, y: GY + 12, width: X1 - X0, height: 8, fill: 'var(--earth)', rx: 2 });
    for (let p = Math.ceil(LO); p <= HI; p += 1) {
      el('line', { x1: sx(p), x2: sx(p), y1: GY + 20, y2: GY + (p % 5 === 0 ? 30 : 25), stroke: 'var(--muted)' });
      if (p % 5 === 0) el('text', { x: sx(p), y: GY + 46, 'text-anchor': 'middle', 'font-size': 12, fill: 'var(--muted)' }).textContent = p;
    }
    // the eye's sweep: how far it can look either way
    el('line', { x1: sx(y - P.A), x2: sx(y + P.A), y1: 26, y2: 26, stroke: 'var(--eye)', opacity: .25, 'stroke-width': 6, 'stroke-linecap': 'round' });
    el('text', { x: sx(y + P.A) + 8, y: 30, 'font-size': 12, fill: 'var(--muted)' }).textContent = 'how far the eye sweeps';
    drawSearchDensity(samples, fd.target);
    // the casings
    places.forEach((p, k) => {
      const [f, a, x] = s.slice(F.NR + F.NC * k, F.NR + F.NC * k + 3);
      const red = fd.casings[k].red, cx = sx(p), cy = sy(x) + 12, w = 44, hh = 30;
      const g = el('g', { class: 'grab' });
      const gapw = 32 * a;
      el('path', { d: `M ${cx - gapw / 2} ${cy - hh} L ${cx - w / 2} ${cy - hh} Q ${cx - w / 2} ${cy} ${cx} ${cy} Q ${cx + w / 2} ${cy} ${cx + w / 2} ${cy - hh} L ${cx + gapw / 2} ${cy - hh}`,
        fill: 'var(--panel)', stroke: red > 0.5 ? 'var(--food)' : 'var(--empty)', 'stroke-width': 3 }, g);
      const fh = (hh - 6) * Math.min(1, f / FULL);
      el('rect', { x: cx - w / 2 + 6, y: cy - 3 - fh, width: w - 12, height: fh, rx: 3, fill: 'var(--food)', opacity: .35 + .5 * red }, g);
      el('text', { x: cx, y: cy - hh - 8, 'text-anchor': 'middle', 'font-size': 12, fill: red > 0.5 ? 'var(--food)' : 'var(--muted)', 'font-family': 'var(--data)' }, g).textContent = fw(f, 1, 4);
      el('rect', { x: cx - w / 2 - 6, y: cy - hh - 8, width: w + 12, height: hh + 32, fill: 'transparent' }, g);
      g.addEventListener('pointerdown', e => dragStart(e, k));
    });
    // the cart: body, wheels, mouth at height x0, the lever down to its tip at height h
    const cx = sx(y), body = sy(P.x0) - 6;
    const cg = el('g', { class: 'grab' });
    el('rect', { x: cx - 46, y: body - 34, width: 92, height: 34, rx: 6, fill: 'var(--panel)', stroke: 'var(--wall)', 'stroke-width': 2 }, cg);
    el('text', { x: cx, y: body - 13, 'text-anchor': 'middle', 'font-size': 13 }, cg).textContent = 'receiver';
    const turn = (s[5] / 0.12) % (2 * Math.PI);
    for (const dx of [-30, 30]) {
      const wx = cx + dx, wy = body + 4;
      el('line', { x1: wx, x2: wx, y1: body, y2: GY + 12 - 14, stroke: 'var(--wall)', 'stroke-width': 2 }, cg);
      el('circle', { cx: wx, cy: GY + 12 - 14, r: 14, fill: 'var(--panel)', stroke: 'var(--wall)', 'stroke-width': 2.5 }, cg);
      el('line', { x1: wx, y1: GY - 2, x2: wx + 12 * Math.cos(turn), y2: GY - 2 + 12 * Math.sin(turn), stroke: 'var(--wall)', 'stroke-width': 2 }, cg);
    }
    el('rect', { x: cx - 50, y: body - 40, width: 100, height: GY - body + 40, fill: 'transparent' }, cg);
    cg.addEventListener('pointerdown', e => dragStart(e, 'cart'));
    if (options.openReceiver) {
      el('title', {}, cg).textContent = t > 0 ? 'Open Receiver' : 'Click to open Receiver or drag to place it';
      if (t > 0) {
        cg.style.cursor = 'pointer';
        cg.addEventListener('click', options.openReceiver);
      }
    }
    // lever: from the mouth down to its tip; jaws close as the grip holds
    const tipy = sy(h) - 4, hold = fd.hold;
    el('line', { x1: cx, x2: cx, y1: body, y2: tipy, stroke: 'var(--grip)', 'stroke-width': 4, 'stroke-linecap': 'round' });
    for (const side of [-1, 1])
      el('line', { x1: cx, y1: tipy, x2: cx + side * (26 - 12 * hold), y2: tipy + 6, stroke: 'var(--grip)', 'stroke-width': 4, 'stroke-linecap': 'round' });
    // the eye and its gaze: a line from the eye to the place it looks at
    const ex = cx + 34 * Math.sign(r || 1), eyY = body - 46, lx = sx(y + r);
    el('line', { x1: ex, y1: eyY, x2: lx, y2: GY - 6, stroke: 'var(--eye)', 'stroke-width': 1.5 + 2.5 * fd.seen, 'stroke-dasharray': fd.seen > 0.5 ? '' : '5 4', opacity: .9 });
    el('circle', { cx: ex, cy: eyY, r: 7, fill: 'var(--panel)', stroke: 'var(--eye)', 'stroke-width': 2.5 });
    el('circle', { cx: lx, cy: GY - 6, r: 5 + 6 * fd.seen, fill: fd.seen > 0.05 ? 'var(--food)' : 'var(--eye)', opacity: .35 + .5 * fd.seen });
    // battery gauge, top left
    const BW = 160, frac = Math.min(1, s[0] / 12);
    el('rect', { x: X0, y: 52, width: BW, height: 14, rx: 3, fill: 'none', stroke: 'var(--wall)', 'stroke-width': 1.5 });
    el('rect', { x: X0 + 2, y: 54, width: Math.max(0, (BW - 4) * frac), height: 10, rx: 2, fill: 'var(--batt)', opacity: .85 });
    el('text', { x: X0 + BW + 10, y: 64, 'font-size': 13, fill: 'var(--batt)', 'font-family': 'var(--data)' }).textContent = 'battery ' + fw(s[0], 2, 5);
    el('text', { x: X1, y: 64, 'text-anchor': 'end', 'font-size': 13, fill: 'var(--muted)', 'font-family': 'var(--data)' }).textContent = 'time ' + fw(t, 1, 5);
  }
  let drag = null, moved = false;
  function dragStart(e, what) {
    if (playing || t > 0) {
      if (what !== 'cart') {
        const food = F.NR + F.NC * what;
        if (s[food] <= P.fr) { s[food] = FULL; draw(); }
      }
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
    if (drag !== 'cart') {
      if (!moved) foods[drag] = foods[drag] > 1 ? EMPTY : FULL;
      else if (places[drag] < LO + 0.3 || places[drag] > HI - 0.3) { places.splice(drag, 1); foods.splice(drag, 1); }
    }
    drag = null; restart(); draw();
  });
  function freePlace() {
    // the free place farthest from everything already on the line; while running, the casing joins the run as it is
    const here = t > 0 ? s[3] : cart0;
    let best = 0, bestd = -1;
    for (let p = LO + 1; p <= HI - 1; p += 0.5) {
      const d = Math.min(Math.abs(p - here), ...places.map(q => Math.abs(p - q)));
      if (d > bestd) { bestd = d; best = p; }
    }
    return best;
  }
  function addCasing(food, place = freePlace()) {
    if (!Number.isFinite(place) || !Number.isFinite(food) || food < 0) throw Error("Invalid casing");
    places.push(place); foods.push(food);
    s.push(...F.casing(food, P));
    if (hist.at(-1)[0] === t) hist.pop();
    record(); draw();
  }
  element('add-full').addEventListener('click', () => addCasing(FULL));
  element('add-empty').addEventListener('click', () => addCasing(EMPTY));

  // ---------- the calculation of the present moment, in the order the mathematics has it ----------
  function drawCalc(fd) {
    const r = fd.rate;
    const k = fd.casings.findIndex(c => c.selected > 0);
    const c = k >= 0 ? fd.casings[k] : null;
    element('calc').innerHTML =
      `<p>The <span class="e">eye</span> looks <span class="out">${fw(s[6], 2, 6, true)}</span> from the cart, at the place <span class="out">${fw(s[3] + s[6], 2, 6, true)}</span>; redness there: <span class="out">${fw(fd.seen, 2, 4)}</span>. So its scan runs at <span class="out">${fw(fd.free, 2, 4)}</span> of full speed, and its look distance changes at <span class="out">${fw(r[6], 2, 6, true)}</span>.</p>` +
      `<p>At the cart, the usual meal score is <span class="out">${fw(F.maps(s, s[3])[0], 2, 5)}</span>. The search density here is <span class="out">${fw(F.maps(s, s[3])[1], 2, 5)}</span>. The search destination is <span class="out">${fw(fd.target, 2, 5, true)}</span>. The wheels apply torque <span class="out">${fw(fd.torque, 3, 6, true)}</span>.</p>` +
      `<p>The <span class="l">lever</span> grips at <span class="out">${fw(fd.grip, 2, 6, true)}</span> (grip <span class="out">${fw(s[1], 2, 5, true)}</span>, holding <span class="out">${fw(fd.hold, 2, 4)}</span>) and lifts its tip at <span class="out">${fw(fd.lift, 2, 6, true)}</span> (tip <span class="out">${fw(s[2] - P.xg, 2, 5)}</span> above the ground).` +
      (c ? ` Casing ${k+1} is in contact. At the mouth it pries at <span class="out">${fw(c.pry, 2, 5)}</span> and the receiver eats at <span class="out">${fw(c.eat, 2, 5)}</span>.</p>` : ` The grip is free.</p>`) +
      `<p>The <span class="b">battery</span> moves by ${P.alpha.toFixed(2)}×eat − ${P.ctear.toFixed(2)}×pry − what the motors cost: gripper <span class="out">${fw(fd.pay.grip, 3, 6, true)}</span>, lever <span class="out">${fw(fd.pay.lift, 3, 6, true)}</span>, wheels <span class="out">${fw(fd.pay.drive, 3, 6, true)}</span>, eye <span class="out">${fw(fd.pay.eye, 3, 6, true)}</span>, memory <span class="out">${fw(fd.pay.memory, 3, 6, true)}</span>, contact <span class="out">${fw(fd.pay.contact, 3, 6, true)}</span>; in all <span class="out">${fw(r[0], 3, 6, true)}</span>.</p>`;
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
    const fmax = Math.max(12, Math.ceil(hist[0][2] / 5) * 5);
    panel(20, 100, 0, fmax, [[1, 'var(--batt)'], [2, 'var(--food)']], 'energy');
    panel(150, 120, LO, HI, [[4, 'var(--eye)', 1.2], [3, 'var(--wall)', 2.5]], 'place on the line');
    el('line', { x1: xs(t), x2: xs(t), y1: 14, y2: 274, stroke: 'var(--accent)', 'stroke-width': 1 }, chart);
    const step = T > 60 ? 20 : T > 30 ? 10 : 5;
    for (let k = 0; k <= T; k += step)
      el('text', { x: xs(k), y: 292, 'text-anchor': 'middle', 'font-size': 11, fill: 'var(--muted)' }, chart).textContent = k;
  }

  function mapSamples() {
    const map = F.searchMap(s);
    return Array.from({length: 181}, (_, i) => {
      const place = LO + i * (HI - LO) / 180;
      const [yieldScore, density] = F.maps(s, place, map);
      return {x: sx(place), yieldScore, density};
    });
  }

  function curvePath(samples, ordinate) {
    return samples.map((sample, i) =>
      (i ? 'L' : 'M') + sample.x.toFixed(2) + ' ' + ordinate(sample).toFixed(2)).join(' ');
  }

  function drawSearchDensity(samples, target) {
    const plot = el('g', {id: 'search-density'});
    const top = GY + 82, bottom = GY + 150;
    const peak = Math.max(...samples.map(sample => sample.density));
    const limit = Math.max(0.1, Math.ceil(peak * 10) / 10);
    const ys = density => bottom - (bottom - top) * density / limit;
    for (const value of [0, limit]) {
      const y = ys(value);
      el('line', {x1: X0, x2: X1, y1: y, y2: y, stroke: 'var(--rule)'}, plot);
      el('text', {x: X0 - 5, y: y + 4, 'text-anchor': 'end', 'font-size': 11}, plot).textContent = value;
    }
    const curve = curvePath(samples, sample => ys(sample.density));
    el('path', {d: curve + ` L ${X1} ${bottom} L ${X0} ${bottom} Z`, fill: 'var(--open)', opacity: 0.19}, plot);
    el('path', {d: curve, fill: 'none', stroke: 'var(--open)', 'stroke-width': 2}, plot);
    const x = sx(target);
    el('line', {x1: x, x2: x, y1: GY + 72, y2: bottom, stroke: 'var(--open)', 'stroke-dasharray': '3 3'}, plot);
    el('path', {d: `M ${x} ${GY+72} v -20 l 12 4 l -12 5`, fill: 'var(--open)', stroke: 'var(--open)'}, plot);
  }

  function drawYield(samples) {
    const plot = element('maps');
    plot.textContent = '';
    const top = Math.max(1, Math.ceil(Math.max(...samples.map(sample => sample.yieldScore))));
    const ys = value => 95 - 65 * value / top;
    el('line', {x1:X0, x2:X1, y1:ys(0), y2:ys(0), stroke:'var(--rule)'}, plot);
    for (const value of [0,top]) el('text', {x:X0-8, y:ys(value)+4, 'text-anchor':'end', fill:'var(--muted)'}, plot).textContent = value;
    for (const place of [-8,-4,0,4,8]) {
      el('text', {x:sx(place), y:135, 'text-anchor':'middle', fill:'var(--muted)'}, plot).textContent = place;
    }
    el('path', {d: curvePath(samples, sample => ys(sample.yieldScore)), fill:'none', stroke:'var(--open)', 'stroke-width':3}, plot);
  }

  let lastObservation = -Infinity;
  function draw(force = true) {
    const now = performance.now();
    if (options.onChange && !force && now - lastObservation < 100) return;
    lastObservation = now;
    render(F.field(s, places, P));
    options.onChange?.();
  }
  function render(fd) {
    const samples = mapSamples();
    drawScene(samples, fd); drawYield(samples); drawChart(); drawCalc(fd);
  }

  // ---------- running: time on the page is the model's time ----------
  const H = 0.004;
  let last = null, animation;
  function step(dt) {
    let todo = dt;
    while (todo > 1e-12) { const h = Math.min(H, todo); s = F.rk4(s, h, places, P); t += h; todo -= h; }
    if (t - hist[hist.length - 1][0] > 0.05) record();
  }
  function frame(now) {
    if (!playing) { last = null; return; }
    if (last !== null) step(Math.min(0.1, (now - last) / 1000) * speed);
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
  element('outing').addEventListener('click', () => {
    setPlaying(false);
    P.searchSeed = crypto.getRandomValues(new Uint32Array(1))[0];
    carriedMemory = F.learnedMemory(s);
    restart(); draw();
  });
  element('reset').addEventListener('click', () => {
    setPlaying(false);
    P.searchSeed = F.DEFAULTS.searchSeed;
    carriedMemory = undefined; restart(); draw();
  });
  for (const k of ['what', 'check']) {
    const b = element('b-' + k), p = element('t-' + k);
    b.addEventListener('click', () => { p.hidden = !p.hidden; b.setAttribute('aria-expanded', String(!p.hidden)); });
  }
  // #t=N opens the page already run to time N (for checking a page mid-run in headless Chrome; PRINCIPLES.md)
  const jump = /t=([\d.]+)/.exec(location.hash);
  if (jump && Number.isFinite(+jump[1])) { const T = +jump[1]; while (t < T - 1e-9) step(Math.min(0.05, T - t)); }
  draw();
  return {
    controls,
    get playing() { return playing; },
    setPlaying, addCasing,
    observe(observer) { return observer(s, places, P, t); }
  };
}
