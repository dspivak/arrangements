// The feeding arrangement of the book's background/feeding-imc/feeding-imc.tex, as its differential equation.
// The state is s = [b, f, x, p, a, l, z]: battery, food, casing position, its momentum, opening, grip, memory.
// Each rate is the integrand's tangent response to its own component of the summed inclination field
// (the note's section 7); `field` also returns the gates and rates so the page can show the calculation.
// Checked against the book's background/feeding-imc/check_episode.py (ordinary code, not the library):
// at the note's parameters and start, t = 12 gives b 10.399, f 0.074; see feeding-model.test.js.
// The library does not yet hold this arrangement. Claude Opus 5.5, 2ff5d861, 2026-11-18.
(function (root) {
  const DEFAULTS = {
    m: 1, g: 0.2, beta: 2, kappa: 40, x0: 0, R: 1, kappa0: 100, b0: 1, f0: 0.1,
    lgrip: 2, lpry: 2, leat: 2, a1: 0.2, a2: 0.6, alpha: 0.8, cwear: 0.02, ctear: 0.3,
  };
  const START = [3, 10, -0.01, 0, 0, 0, -0.01];

  const rho = t => (t > 0 ? Math.exp(-1 / t) : 0);
  const theta = t => { const a = rho(t), c = rho(1 - t); return a / (a + c); };
  const dtheta = t => {
    if (t <= 0 || t >= 1) return 0;
    const a = rho(t), c = rho(1 - t), da = a / (t * t), dc = c / ((1 - t) * (1 - t));
    return (da * (a + c) - a * (da - dc)) / ((a + c) * (a + c));
  };

  function field(s, P) {
    const [b, f, x, p, a, l, z] = s;
    const gap = x - P.x0;
    const hold = theta(l), dhold = dtheta(l);
    const arg = (P.R * P.R - gap * gap) / (0.75 * P.R * P.R);
    const near = theta(arg), dnear = dtheta(arg) * (-2 * gap) / (0.75 * P.R * P.R);
    const live = theta(b / P.b0);
    const keep = theta((f - P.f0) / P.f0);
    const access = theta((a - P.a1) / (P.a2 - P.a1));
    const firm = P.kappa0 * hold * near;
    const dJdx = 0.5 * P.kappa0 * hold * dnear * gap * gap + firm * gap;   // the attachment cost's x-derivative
    const dJdl = 0.5 * P.kappa0 * dhold * near * gap * gap;                 // and its l-derivative
    const join = P.lgrip * live * near * keep;
    const release = P.lgrip * live * (1 - keep);
    const grip = join * (2 - l) - release * (l + 1);
    const pry = P.lpry * live * near * hold * (1 - a);
    const eat = P.leat * live * near * hold * access * f;
    const draw = dJdl * grip + P.cwear * grip * grip;
    const delta = x - z;
    const force = -dJdx - P.m * P.g - P.kappa * delta;
    const rate = [
      P.alpha * eat - draw - P.ctear * pry, // battery
      -eat,                                 // food
      p / P.m,                              // position
      force,                                // momentum
      pry,                                  // opening
      grip,                                 // grip
      P.kappa * delta / P.beta,             // memory
    ];
    const energy = b + f + p * p / (2 * P.m) + P.m * P.g * x + 0.5 * firm * gap * gap + 0.5 * P.kappa * delta * delta;
    const loss = P.cwear * grip * grip + P.ctear * pry + (1 - P.alpha) * eat + P.kappa * P.kappa / P.beta * delta * delta;
    return { rate, gates: { live, near, hold, keep, access }, rates: { join, release, grip, pry, eat, draw },
             force, delta, energy, loss, attach: 0.5 * firm * gap * gap };
  }

  function rk4(s, h, P) {
    const add = (u, v, c) => u.map((ui, i) => ui + c * v[i]);
    const k1 = field(s, P).rate;
    const k2 = field(add(s, k1, h / 2), P).rate;
    const k3 = field(add(s, k2, h / 2), P).rate;
    const k4 = field(add(s, k3, h), P).rate;
    return s.map((si, i) => si + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }

  const api = { DEFAULTS, START, theta, field, rk4 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Feeding = api;
})(this);
