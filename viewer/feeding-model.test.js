// node viewer/feeding-model.test.js — the page's model against the book's check_episode.py figures
// (b 10.39890, f 0.07397 at t = 12; grip through 0 at t 3.05838), and the energy identity E + H = E(0).
const F = require('./feeding-model.js');
const P = { ...F.DEFAULTS };
let s = F.START.slice();
const h = 0.0005;
let t = 0, H = 0, cross = null;
const E0 = F.field(s, P).energy;
for (let n = 0; n < 12 / h; n++) {
  const L0 = F.field(s, P).loss;
  const s1 = F.rk4(s, h, P);
  H += h / 2 * (L0 + F.field(s1, P).loss);
  if (cross === null && s[5] > 0 && s1[5] <= 0) cross = t + h * s[5] / (s[5] - s1[5]);
  s = s1; t += h;
}
const E = F.field(s, P).energy;
const ok = (name, got, want, tol) => {
  const pass = Math.abs(got - want) <= tol;
  console.log(`${pass ? 'ok  ' : 'FAIL'} ${name}: ${got.toFixed(6)} (want ${want} ± ${tol})`);
  if (!pass) process.exitCode = 1;
};
ok('battery at t=12', s[0], 10.39890, 1e-4);
ok('food at t=12', s[1], 0.07397, 1e-4);
ok('grip crosses 0 at', cross, 3.05838, 1e-3);
ok('E(12) + H(12) - E(0)', E + H - E0, 0, 1e-4);
