// The feeding arrangement of the book's background/feeding-imc/feeding-imc.tex, as its differential equation:
// a receiver on a cart on a line, with a lever, an eye and a patrol, and any number of casings resting on the ground.
// Receiver state [b, l, h, y, my, zc, r, s, eta, zeta]: battery, grip, lever tip height, cart position, cart momentum,
// wheel, look distance, scan, patrol point, patrol's second coordinate. Casing k adds [f, a, x, mu, z, zg, za]: food,
// opening, height, vertical momentum, air memory, the ground damper's memory, and the attachment damper's memory
// (the last is the receiver's, kept beside its casing); its place p_k on the line is fixed. Each rate is the
// integrand's tangent response to its own component of the summed inclination field; `field` also returns the gates
// and rates so the page can show the calculation.
// Checked against the book's background/feeding-imc/check_episode.py (ordinary code, not the library):
// see feeding-model.test.js for the figures. The composed library model is arrangements/feeding.py; tests/test_explore.py checks its steps and source-output loads.
// Claude Opus 5.5, 2ff5d861 (casing and receiver), d531d927 (cart, lever lift, eye) and 003367d0 (patrol, damped
// contacts), 2026-11-18.
(function (root) {
  const DEFAULTS = {
    m: 1, g: 2, beta: 0.2, kappa: 40, x0: 0, R: 0.5, kappa0: 100, b0: 1, f0: 0.1,
    lgrip: 2, lpry: 2, leat: 2, a1: 0.2, a2: 0.6, alpha: 0.8, cgrip: 0.02, ctear: 0.3,
    xg: -0.4, kg: 400, eps: 0.05, Rh: 0.5, clift: 0.02, llift: 2,
    M: 0.25, betac: 0.4, kappac: 40, ktau: 0.16,
    A: 6, omega: 0.6, gam: 0.05, lam: 3, w: 0.5, csee: 0.002, fr: 0.1,
    // the contact dampers (ground and attachment) and the patrol
    kd: 4000, bd: 40, kr: 10, epsd: 0.01,
    B: 8, nup: 0.1, gamp: 0.005, cpat: 0.002, kexp: 0.1,
  };
  const NR = 10, NC = 7;
  // David's screenshot of 2026-11-18: full casings at -8 and +8, an empty one at 4, the cart at 0
  const PLACES = [4, -8, 8];
  const FOODS = [0.05, 10, 10];
  function start(places, foods, y0 = 0, P = DEFAULTS) {
    const s = [3, -1, P.xg, y0, 0, y0, 0, P.A, 0, P.B];
    for (let k = 0; k < places.length; k++) s.push(...casing(foods[k], P));
    return s;
  }
  // a casing resting on the ground, its memories unstretched
  const casing = (food, P = DEFAULTS) => [food, 0, P.xg, 0, P.xg, 0, 0];
  const START = start(PLACES, FOODS);

  const rho = t => (t > 0 ? Math.exp(-1 / t) : 0);
  const theta = t => { const a = rho(t), c = rho(1 - t); return a / (a + c); };
  const dtheta = t => {
    if (t <= 0 || t >= 1) return 0;
    const a = rho(t), c = rho(1 - t), da = a / (t * t), dc = c / ((1 - t) * (1 - t));
    return (da * (a + c) - a * (da - dc)) / ((a + c) * (a + c));
  };
  const near = (u, R) => { const t = (R * R - u * u) / (0.75 * R * R); return [theta(t), dtheta(t) * (-2 * u) / (0.75 * R * R)]; };

  function field(s, places, P) {
    const [b, l, h, y, my, zc, r, sc, eta, zeta] = s;
    const n = places.length;
    const live = theta(b / P.b0), hold = theta(l), dhold = dtheta(l);
    // the eye: how red the place it looks at is
    let seen = 0, dseen = 0;
    const reds = [];
    for (let k = 0; k < n; k++) {
      const f = s[NR + NC * k];
      const red = theta((f - P.fr) / P.fr);
      const u = (y + r - places[k]) / P.w, e = Math.exp(-u * u);
      reds.push(red);
      seen += red * e; dseen += red * (-2 * u / P.w) * e;
    }
    const free = theta(1 - 2 * seen);
    // two sweeps: the eye's (r, s) and the patrol's (eta, zeta), each round its circle while free
    const ring = P.A * P.A - r * r - sc * sc;
    const lookR = live * (free * (P.omega * sc + P.gam * ring * r) + P.lam * dseen);
    const lookS = live * free * (-P.omega * r + P.gam * ring * sc);
    const ringp = P.B * P.B - eta * eta - zeta * zeta;
    const patE = live * free * (P.nup * zeta + P.gamp * ringp * eta);
    const patZ = live * free * (-P.nup * eta + P.gamp * ringp * zeta);
    const torque = live * (P.ktau * seen * r + P.kexp * free * (eta - y));
    // the lever: grip toward a full casing at its tip, lift toward the mouth while holding
    let K = 0;
    for (let k = 0; k < n; k++) {
      const f = s[NR + NC * k], x = s[NR + NC * k + 2];
      K += near(x - h, P.R)[0] * near(y - places[k], P.Rh)[0] * theta((f - P.f0) / P.f0);
    }
    const grip = P.lgrip * live * (K * (2 - l) - (1 - K) * (l + 1));
    const lift = P.llift * live * (hold * (P.x0 - h) + (1 - hold) * (P.xg - h));
    let dJl = 0, dJh = 0, Fy = 0, eatAll = 0, pryAll = 0, attach = 0, ground = 0, damp = 0;
    const rates = [], casings = [];
    for (let k = 0; k < n; k++) {
      const [f, a, x, mu, z, zg, za] = s.slice(NR + NC * k, NR + NC * (k + 1));
      // the attachment: a spring and a damper on the stretch u, both gated by the grip and nearness
      const u = x - h, da = u - za;
      const [nv, dnv] = near(u, P.R), [nh, dnh] = near(y - places[k], P.Rh);
      const G = hold * nv * nh, W = P.kappa0 * u * u + P.kd * da * da;
      const dJx = G * (P.kappa0 * u + P.kd * da) + 0.5 * hold * dnv * nh * W;
      dJl += 0.5 * dhold * nv * nh * W;
      dJh -= dJx;
      Fy -= 0.5 * hold * nv * dnh * W;
      attach += 0.5 * G * W + 0.5 * P.kr * za * za;
      const zaDot = (G * P.kd * da - P.kr * za) / P.bd;
      const atMouth = live * near(x - P.x0, P.R)[0] * nh * hold;
      const access = theta((a - P.a1) / (P.a2 - P.a1));
      const pry = P.lpry * atMouth * (1 - a);
      const eat = P.leat * atMouth * access * f;
      // the ground: a spring gated over eps and a damper gated over epsd, below the surface
      const v = x - P.xg, dg = v - zg;
      const tg = theta(-v / P.eps), dtg = dtheta(-v / P.eps) * (-1 / P.eps);
      const td = theta(-v / P.epsd), dtd = dtheta(-v / P.epsd) * (-1 / P.epsd);
      const dJg = P.kg * tg * v + 0.5 * P.kg * dtg * v * v + P.kd * td * dg + 0.5 * P.kd * dtd * dg * dg;
      ground += 0.5 * P.kg * tg * v * v + 0.5 * P.kd * td * dg * dg + 0.5 * P.kr * zg * zg;
      const zgDot = (P.kd * td * dg - P.kr * zg) / P.bd;
      const delta = x - z;
      const force = -dJx - P.m * P.g - P.kappa * delta - dJg;
      rates.push(-eat, pry, mu / P.m, force, P.kappa * delta / P.beta, zgDot, zaDot);
      casings.push({ red: reds[k], access, pry, eat, force, atMouth });
      eatAll += eat; pryAll += pry;
      damp += P.bd * (zgDot * zgDot + zaDot * zaDot);
    }
    const stretch = y - zc;
    const wheel = (P.kappac * stretch + torque) / P.betac;
    const pay = {
      grip: (P.cgrip * grip + dJl) * grip, lift: (P.clift * lift + dJh) * lift,
      drive: torque * wheel, eye: P.csee * (lookR * lookR + lookS * lookS), patrol: P.cpat * (patE * patE + patZ * patZ),
    };
    const bdot = P.alpha * eatAll - P.ctear * pryAll - pay.grip - pay.lift - pay.drive - pay.eye - pay.patrol;
    const rate = [bdot, grip, lift, my / P.M, Fy - P.kappac * stretch, wheel, lookR, lookS, patE, patZ].concat(rates);
    let energy = b + my * my / (2 * P.M) + 0.5 * P.kappac * stretch * stretch + attach + ground;
    let loss = P.cgrip * grip * grip + P.clift * lift * lift + P.betac * wheel * wheel + pay.eye + pay.patrol
      + P.ctear * pryAll + (1 - P.alpha) * eatAll + damp;
    for (let k = 0; k < n; k++) {
      const [f, , x, mu, z] = s.slice(NR + NC * k, NR + NC * k + 5);
      energy += f + mu * mu / (2 * P.m) + P.m * P.g * x + 0.5 * P.kappa * (x - z) * (x - z);
      loss += P.kappa * P.kappa / P.beta * (x - z) * (x - z);
    }
    return { rate, live, hold, seen, free, torque, grip, lift, wheel, pay, eatAll, pryAll, casings, energy, loss, patE };
  }

  function rk4(s, h, places, P) {
    const add = (u, v, c) => u.map((ui, i) => ui + c * v[i]);
    const k1 = field(s, places, P).rate;
    const k2 = field(add(s, k1, h / 2), places, P).rate;
    const k3 = field(add(s, k2, h / 2), places, P).rate;
    const k4 = field(add(s, k3, h), places, P).rate;
    return s.map((si, i) => si + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }

  const api = { DEFAULTS, NR, NC, PLACES, FOODS, START, start, casing, theta, field, rk4 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Feeding = api;
})(this);
