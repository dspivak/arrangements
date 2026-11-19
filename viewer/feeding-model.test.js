// node viewer/feeding-model.test.js — the page's model against the book's check_episode.py figures, on David's
// screenshot (full casings at -8 and 8, an empty one at 4, the cart at 0): at t = 60 the battery is 7.32872, the
// full casings' food 0.07394 and 0.07394, the cart at -7.06319; the eye holds on red (seen rises past 1/2) at
// t 11.127 and 46.676; and the energy identity E + H = E(0).
const F = require('./feeding-model.js');
const P = { ...F.DEFAULTS };
const places = F.PLACES;
let s = F.START.slice();
const h = 0.001;
let t = 0, H = 0;
const locks = [];
const E0 = F.field(s, places, P).energy;
for (let n = 0; n < 60 / h; n++) {
  const f0 = F.field(s, places, P);
  const s1 = F.rk4(s, h, places, P);
  const f1 = F.field(s1, places, P);
  H += h / 2 * (f0.loss + f1.loss);
  if (f0.seen <= 0.5 && f1.seen > 0.5) locks.push(t + h * (0.5 - f0.seen) / (f1.seen - f0.seen));
  s = s1; t += h;
}
const E = F.field(s, places, P).energy;
const ok = (name, got, want, tol) => {
  const pass = Math.abs(got - want) <= tol;
  console.log(`${pass ? 'ok  ' : 'FAIL'} ${name}: ${got.toFixed(6)} (want ${want} ± ${tol})`);
  if (!pass) process.exitCode = 1;
};
ok('battery at t=60', s[0], 7.32872, 1e-3);
ok('food of the casing at -8, t=60', s[F.NR + F.NC * 1], 0.07394, 1e-3);
ok('food of the casing at 8, t=60', s[F.NR + F.NC * 2], 0.07394, 1e-3);
ok('cart at t=60', s[3], -7.06319, 1e-3);
ok('the eye first holds on red at', locks[0], 11.127, 2e-3);
ok('the eye holds on red again at', locks[1], 46.676, 2e-3);
ok('E(60) + H(60) - E(0)', E + H - E0, 0, 1e-4);
