// The feeding arrangement of the book's background/feeding-imc/feeding-imc.tex, as its differential equation:
// a receiver on a cart on a line, with a lever, an eye and a patrol, and any number of casings resting on the ground.
// Receiver state [b, l, h, y, my, zc, r, s, eta, zeta]: battery, grip, lever tip height, cart position, cart momentum,
// wheel, look distance, scan, patrol point, patrol's second coordinate. Each site adds the casing's food, opening,
// height, vertical momentum, air and ground memories, followed by the external contact's stroke and latch.
// The casing's place on the line is fixed. Each rate is the
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
    B: 8, nup: 0.1, gamp: 0.005, cpat: 0.002, kexp: 0.1, latchRate: 40, latchCost: 0.0001,
  };
  const NR = 10, NC = 8;
  // David's screenshot of 2026-11-18: full casings at -8 and +8, an empty one at 4, the cart at 0
  const PLACES = [4, -8, 8];
  const FOODS = [0.05, 10, 10];
  function start(places, foods, y0 = 0, P = DEFAULTS) {
    const s = [3, -1, P.xg, y0, 0, y0, 0, P.A, 0, P.B];
    for (let k = 0; k < places.length; k++) s.push(...casing(foods[k], P));
    return s;
  }
  // a casing resting on the ground, its memories unstretched
  const casing = (food, P = DEFAULTS) => [food, 0, P.xg, 0, P.xg, 0, 0, 0];
  const START = start(PLACES, FOODS);

  const rho = t => (t > 0 ? Math.exp(-1 / t) : 0);
  const theta = t => { const a = rho(t), c = rho(1 - t); return a / (a + c); };
  const dtheta = t => {
    if (t <= 0 || t >= 1) return 0;
    const a = rho(t), c = rho(1 - t), da = a / (t * t), dc = c / ((1 - t) * (1 - t));
    return (da * (a + c) - a * (da - dc)) / ((a + c) * (a + c));
  };
  const near = (u, R) => { const t = (R * R - u * u) / (0.75 * R * R); return [theta(t), dtheta(t) * (-2 * u) / (0.75 * R * R)]; };

  function encounter(s, places, P, receiverSize = NR) {
    const tip = s[2], cart = s[3], live = theta(s[0]/P.b0);
    const stillHolding = theta((s[1]+.8)/.4);
    const grip = [], mouth = [], engaged = [], engageSlope = [], latchTold = [], latchRate = [], available = [], score = [], latch = [], firstChoice = [], unblocked = [];
    for (let k = 0; k < places.length; k++) {
      const i = receiverSize+NC*k, height = s[i+2], horizontal = near(cart-places[k], P.Rh)[0];
      grip.push(horizontal*near(height-tip, P.R)[0]);
      mouth.push(horizontal*near(height-P.x0, P.R)[0]);
      latch.push(s[i+7]);
      engaged.push(theta((latch[k]-.3)/.2));
      engageSlope.push(dtheta((latch[k]-.3)/.2)/.2);
      available.push(grip[k]*theta((s[i]-P.f0)/P.f0));
      score.push(available[k]/(1+((height-tip)/P.R)**2+((cart-places[k])/P.Rh)**2));
    }
    for (let i=0; i<places.length; i++) {
      let first = 1, free = 1;
      for (let j=0; j<places.length; j++) if (i!==j) {
        first *= theta((score[i]-score[j])/.05);
        free *= 1-theta(latch[j]/.3);
      }
      firstChoice.push(first); unblocked.push(free);
      const acquire = theta((available[i]-.15)/.15)*first*(1-stillHolding);
      const retain = 1-(1-available[i])*(1-stillHolding);
      const target = free*((1-engaged[i])*acquire+engaged[i]*retain);
      // The rate the latch tells its motor; the motor reads the battery and drives at live times it.
      latchTold.push(P.latchRate*(target-latch[i]));
      latchRate.push(live*latchTold[i]);
    }
    return {grip, mouth, engaged, engageSlope, latchTold, latchRate, available, score, firstChoice, unblocked};
  }

  // The single opener and digestor expose their available effort to the surroundings.
  function mouthEfforts(battery, grip, P) {
    const active = theta(battery/P.b0)*theta(grip);
    return {opening: P.lpry*active, digestion: P.leat*active};
  }

  function field(s, places, P, learner) {
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
    // Every motor reads the battery: told a rate (or the drive, a torque), it drives at live times it.
    // `told` holds what each motor is told; the realized rates below are live times those.
    // two sweeps: the eye's (r, s) and the patrol's (eta, zeta), each round its circle while free
    const ring = P.A * P.A - r * r - sc * sc;
    const ringp = P.B * P.B - eta * eta - zeta * zeta;
    const search = learner ? learner.pull : P.kexp * (eta - y);
    const told = {
      look: free * (P.omega * sc + P.gam * ring * r) + P.lam * dseen,
      scan: free * (-P.omega * r + P.gam * ring * sc),
      patrol: learner ? 0 : free * (P.nup * zeta + P.gamp * ringp * eta),
      patrolScan: learner ? 0 : free * (-P.nup * eta + P.gamp * ringp * zeta),
      torque: P.ktau * seen * r + free * search,
    };
    const lookR = live * told.look, lookS = live * told.scan;
    const patE = live * told.patrol, patZ = live * told.patrolScan;
    const torque = live * told.torque;
    // the lever: grip toward a full casing at its tip, lift toward the mouth while holding
    let K = 0;
    const contact = encounter(s, places, P), mouth = mouthEfforts(b, l, P);
    for (let k = 0; k < n; k++) {
      const f = s[NR + NC * k], x = s[NR + NC * k + 2];
      K += contact.engaged[k]*contact.grip[k] * theta((f - P.f0) / P.f0);
    }
    told.grip = P.lgrip * (K * (2 - l) - (1 - K) * (l + 1));
    told.lift = P.llift * (hold * (P.x0 - h) + (1 - hold) * (P.xg - h));
    const grip = live * told.grip, lift = live * told.lift;
    let dJl = 0, dJh = 0, Fy = 0, eatAll = 0, pryAll = 0, attach = 0, ground = 0, damp = 0, latchPay=0, latchLoss=0;
    const rates = [], casings = [];
    for (let k = 0; k < n; k++) {
      const [f, a, x, mu, z, zg, za] = s.slice(NR + NC * k, NR + NC * (k + 1));
      // the attachment: a spring and a damper on the stretch u, both gated by the grip and nearness
      const u = x - h, da = u - za;
      const [nv, dnv] = near(u, P.R), [nh, dnh] = near(y - places[k], P.Rh);
      const selected = contact.engaged[k], G = selected * hold * nv * nh, W = P.kappa0 * u * u + P.kd * da * da;
      const dJx = G * (P.kappa0 * u + P.kd * da) + 0.5 * selected * hold * dnv * nh * W;
      dJl += 0.5 * selected * dhold * nv * nh * W;
      dJh -= dJx;
      Fy -= 0.5 * selected * hold * nv * dnh * W;
      attach += 0.5 * G * W + 0.5 * P.kr * za * za;
      const zaDot = (G * P.kd * da - P.kr * za) / P.bd;
      const atMouth = selected * live * hold * contact.mouth[k];
      const access = theta((a - P.a1) / (P.a2 - P.a1));
      const pry = mouth.opening * selected * contact.mouth[k] * (1 - a);
      const eat = mouth.digestion * selected * contact.mouth[k] * access * f;
      const latchDot=contact.latchRate[k], friction=P.latchCost*latchDot*latchDot;
      latchLoss+=friction;
      latchPay+=friction+.5*contact.engageSlope[k]*hold*nv*nh*W*latchDot;
      // the ground: a spring gated over eps and a damper gated over epsd, below the surface
      const v = x - P.xg, dg = v - zg;
      const tg = theta(-v / P.eps), dtg = dtheta(-v / P.eps) * (-1 / P.eps);
      const td = theta(-v / P.epsd), dtd = dtheta(-v / P.epsd) * (-1 / P.epsd);
      const dJg = P.kg * tg * v + 0.5 * P.kg * dtg * v * v + P.kd * td * dg + 0.5 * P.kd * dtd * dg * dg;
      ground += 0.5 * P.kg * tg * v * v + 0.5 * P.kd * td * dg * dg + 0.5 * P.kr * zg * zg;
      const zgDot = (P.kd * td * dg - P.kr * zg) / P.bd;
      const delta = x - z;
      const force = -dJx - P.m * P.g - P.kappa * delta - dJg;
      rates.push(-eat, pry, mu / P.m, force, P.kappa * delta / P.beta, zgDot, zaDot, latchDot);
      casings.push({ red: reds[k], access, pry, eat, force, atMouth, selected });
      eatAll += eat; pryAll += pry;
      damp += P.bd * (zgDot * zgDot + zaDot * zaDot);
    }
    const stretch = y - zc;
    const wheel = (P.kappac * stretch + torque) / P.betac;
    const pay = {
      grip: (P.cgrip * grip + dJl) * grip, lift: (P.clift * lift + dJh) * lift,
      drive: torque * wheel, eye: P.csee * (lookR * lookR + lookS * lookS), patrol: P.cpat * (patE * patE + patZ * patZ),
    };
    pay.contact=latchPay;
    const bdot = P.alpha * eatAll - P.ctear * pryAll - pay.grip - pay.lift - pay.drive - pay.eye - pay.patrol-latchPay;
    const rate = [bdot, grip, lift, my / P.M, Fy - P.kappac * stretch, wheel, lookR, lookS, patE, patZ].concat(rates);
    let energy = b + my * my / (2 * P.M) + 0.5 * P.kappac * stretch * stretch + attach + ground;
    let loss = P.cgrip * grip * grip + P.clift * lift * lift + P.betac * wheel * wheel + pay.eye + pay.patrol
      + P.ctear * pryAll + (1 - P.alpha) * eatAll + damp + latchLoss;
    for (let k = 0; k < n; k++) {
      const [f, , x, mu, z] = s.slice(NR + NC * k, NR + NC * k + 5);
      energy += f + mu * mu / (2 * P.m) + P.m * P.g * x + 0.5 * P.kappa * (x - z) * (x - z);
      loss += P.kappa * P.kappa / P.beta * (x - z) * (x - z);
    }
    return { rate, live, hold, seen, free, torque, grip, lift, told, wheel, pay, eatAll, pryAll, casings, energy, loss, patE };
  }

  function rk4(s, h, places, P, evaluate = field) {
    const add = (u, v, c) => u.map((ui, i) => ui + c * v[i]);
    const k1 = evaluate(s, places, P).rate;
    const k2 = evaluate(add(s, k1, h / 2), places, P).rate;
    const k3 = evaluate(add(s, k2, h / 2), places, P).rate;
    const k4 = evaluate(add(s, k3, h), places, P).rate;
    return s.map((si, i) => si + h / 6 * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
  }

  const api = { DEFAULTS, NR, NC, PLACES, FOODS, START, start, casing, theta, dtheta, encounter, mouthEfforts, field, rk4 };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.Feeding = api;
})(this);
