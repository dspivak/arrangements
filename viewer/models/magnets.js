// The page's model for magnets.html: a loop typed by hand that stands beside the library's arrangement
// arrangements/magnets.py (``plane . (bodies)``, book plan P76 (h)) and is checked equal to it by
// viewer/models/check_magnets.js at the settings of tests/test_magnets.py.
//
// A body is {kind, I, h, ell, bump, x, y, th, px, py, pth, m, J}: kind "wire", "magnet" or "loop"; I its current; h its half-width;
// ell a dead loop's inductance; bump the strength of the bumper at each of its points (0 or absent: none); (x, y, th) its cart's
// position and angle; (px, py, pth) the cart's momenta; m, J its mass and inertia.
// Two fields are entered: that of a long straight wire, g, and the contact field of one bumper, (sigma/r)^6. Claude Fable 5.2, session 2524b69e, 2026-11-10.
(function (root) {
  "use strict";
  const TWO_PI = 2 * Math.PI;

  function g(dx, dy) { return -Math.log(dx * dx + dy * dy) / (4 * Math.PI); }

  function ends(b) {            // the two points +h and -h on the body's own axis
    const c = Math.cos(b.th), s = Math.sin(b.th);
    return [{ x: b.x + b.h * c, y: b.y + b.h * s }, { x: b.x - b.h * c, y: b.y - b.h * s }];
  }

  function wiresOf(b) {         // the wires whose fields a body emits
    if (b.kind === "wire") return [{ x: b.x, y: b.y, I: b.I }];
    if (b.kind === "magnet") { const e = ends(b); return [{ x: e[0].x, y: e[0].y, I: b.I }, { x: e[1].x, y: e[1].y, I: -b.I }]; }
    return [];
  }

  function fieldAt(bodies, skip, x, y) {     // the sum of the other bodies' fields at a point
    let A = 0;
    for (let j = 0; j < bodies.length; j++) {
      if (j === skip) continue;
      for (const w of wiresOf(bodies[j])) A += w.I * g(x - w.x, y - w.y);
    }
    return A;
  }

  // What each body receives and what comes back on its cart's position: flux (or, for a wire, the field at it), a dead loop's
  // current, and the force and torque.
  function loads(bodies) {
    const out = bodies.map(() => ({ fx: 0, fy: 0, tq: 0, flux: 0, current: 0 }));
    const em = bodies.map(wiresOf);
    function push(i, p, fx, fy) {
      out[i].fx += fx; out[i].fy += fy;
      out[i].tq += (p.x - bodies[i].x) * fy - (p.y - bodies[i].y) * fx;
    }
    function on(a, b) {         // the force on the wire a from the wire b: I_a I_b grad g
      const dx = a.x - b.x, dy = a.y - b.y, k = -a.I * b.I / (TWO_PI * (dx * dx + dy * dy));
      return [k * dx, k * dy];
    }
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      if (b.kind === "wire") { out[i].flux = fieldAt(bodies, i, b.x, b.y); out[i].current = b.I; }
      else { const e = ends(b); out[i].flux = fieldAt(bodies, i, e[0].x, e[0].y) - fieldAt(bodies, i, e[1].x, e[1].y);
             out[i].current = b.kind === "loop" ? -out[i].flux / b.ell : b.I; }
    }
    for (let i = 0; i < bodies.length; i++) for (let j = i + 1; j < bodies.length; j++)
      for (const a of em[i]) for (const c of em[j]) { const F = on(a, c); push(i, a, F[0], F[1]); push(j, c, -F[0], -F[1]); }
    // the contact field: a bumper of strength b.bump at each point where the body makes or reads the magnetic field; two bumpers
    // on different bodies push apart with the force 6 b_a b_c sigma^6 d / |d|^8 (the library's bumper, sigma = 0.25)
    const S6 = Math.pow(0.25, 6);
    const bp = bodies.map(b => !b.bump ? [] : b.kind === "wire" ? [{ x: b.x, y: b.y }] : ends(b));
    for (let i = 0; i < bodies.length; i++) for (let j = i + 1; j < bodies.length; j++)
      for (const a of bp[i]) for (const c of bp[j]) {
        const dx = a.x - c.x, dy = a.y - c.y, r2 = dx * dx + dy * dy, k = 6 * bodies[i].bump * bodies[j].bump * S6 / (r2 * r2 * r2 * r2);
        push(i, a, k * dx, k * dy); push(j, c, -k * dx, -k * dy);
      }
    for (let i = 0; i < bodies.length; i++) {
      if (bodies[i].kind !== "loop") continue;
      const e = ends(bodies[i]), Il = out[i].current;
      const mine = [{ x: e[0].x, y: e[0].y, I: Il }, { x: e[1].x, y: e[1].y, I: -Il }];
      for (let j = 0; j < bodies.length; j++) {
        if (j === i) continue;
        for (const a of mine) for (const c of em[j]) { const F = on(a, c); push(i, a, F[0], F[1]); push(j, c, -F[0], -F[1]); }
      }
    }
    return out;
  }

  // The total field at a point, for drawing: every emitter, and every dead loop's two ends at the current it carries now.
  function sources(bodies) {
    const L = loads(bodies), s = [];
    bodies.forEach((b, i) => {
      if (b.kind === "loop") { const e = ends(b); s.push({ x: e[0].x, y: e[0].y, I: L[i].current }, { x: e[1].x, y: e[1].y, I: -L[i].current }); }
      else for (const w of wiresOf(b)) s.push(w);
    });
    return s;
  }

  // opts.drag: the rate at which a cart loses momentum; opts.turn false: the carts slide and do not turn
  function rate(bodies, held, opts) {
    const L = loads(bodies), drag = opts && opts.drag || 0, turn = !opts || opts.turn !== false;
    return bodies.map((b, i) => held && held[i] ? [0, 0, 0, 0, 0, 0]
      : [b.px / b.m, b.py / b.m, turn ? b.pth / b.J : 0, L[i].fx - drag * b.px, L[i].fy - drag * b.py, turn ? L[i].tq - drag * b.pth : 0]);
  }

  function moved(bodies, k, f) {
    return bodies.map((b, i) => Object.assign({}, b, { x: b.x + f * k[i][0], y: b.y + f * k[i][1], th: b.th + f * k[i][2],
      px: b.px + f * k[i][3], py: b.py + f * k[i][4], pth: b.pth + f * k[i][5] }));
  }

  // One Runge-Kutta step of duration tau, the rule the library's run uses; held[i] true keeps body i where it is.
  function step(bodies, tau, held, opts) {
    const k1 = rate(bodies, held, opts), k2 = rate(moved(bodies, k1, tau / 2), held, opts);
    const k3 = rate(moved(bodies, k2, tau / 2), held, opts), k4 = rate(moved(bodies, k3, tau), held, opts);
    const k = k1.map((r, i) => r.map((v, c) => (v + 2 * k2[i][c] + 2 * k3[i][c] + k4[i][c]) / 6));
    return moved(bodies, k, tau);
  }

  function closest(bodies) {    // the least distance between points of two different bodies where a field is made or read
    let best = Infinity;
    const pts = bodies.map(b => b.kind === "wire" ? [{ x: b.x, y: b.y }] : ends(b));
    for (let i = 0; i < bodies.length; i++) for (let j = i + 1; j < bodies.length; j++)
      for (const a of pts[i]) for (const c of pts[j]) best = Math.min(best, Math.hypot(a.x - c.x, a.y - c.y));
    return best;
  }

  const model = { g, ends, wiresOf, fieldAt, loads, sources, step, closest };
  root.magnetsModel = model;
  if (typeof module !== "undefined") module.exports = model;
})(typeof window !== "undefined" ? window : globalThis);
