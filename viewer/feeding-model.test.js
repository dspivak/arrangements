// node viewer/feeding-model.test.js — the page's model against the book's check_episode.py figures
// (t = 30: battery 8.37718, the full casing's food 0.07392, cart at -2.96609; the eye first holds on red,
// seen > 1/2, at t 5.922), and the energy identity E + H = E(0).
const F = require('./feeding-model.js');
const P = { ...F.DEFAULTS };
const places = F.PLACES;
let s = F.START.slice();
const h = 0.001;
let t = 0, H = 0, lock = null;
const E0 = F.field(s, places, P).energy;
for (let n = 0; n < 30 / h; n++) {
  const f0 = F.field(s, places, P);
  const s1 = F.rk4(s, h, places, P);
  const f1 = F.field(s1, places, P);
  H += h / 2 * (f0.loss + f1.loss);
  if (lock === null && f0.seen <= 0.5 && f1.seen > 0.5) lock = t + h * (0.5 - f0.seen) / (f1.seen - f0.seen);
  s = s1; t += h;
}
const E = F.field(s, places, P).energy;
const ok = (name, got, want, tol) => {
  const pass = Math.abs(got - want) <= tol;
  console.log(`${pass ? 'ok  ' : 'FAIL'} ${name}: ${got.toFixed(6)} (want ${want} ± ${tol})`);
  if (!pass) process.exitCode = 1;
};
ok('battery at t=30', s[0], 8.37718, 1e-3);
ok('full casing food at t=30', s[F.NR + 5], 0.07392, 1e-3);
ok('cart at t=30', s[3], -2.96609, 1e-3);
ok('eye holds on red at', lock, 5.922, 2e-3);
ok('E(30) + H(30) - E(0)', E + H - E0, 0, 1e-4);
