// The listener of arrangements/listener.py (book plan §5.9 item 63), run in the page so its inputs can be changed.
// The step is the closed form of tests/test_listener.py, which test 1 there checks against the composite in arr
// to 1e-8; tests/test_animate.py checks this file against listener.run_script the same way. Each step, at the
// state going in: the belief's log-heights move by the heard evidence (Bayes' rule, scaled by the kernel weight
// when params.hear is "weighted", as in the library, or at full strength wherever the listener stands when it is
// "full", the page's default since 2026-10-09: plan §5.9 item 63, "leaving ends the hearing is the construction's"),
// on the exact wire also by minus the kernel weight times the losses; the placement descends the expected cost
// k(p) <y, ell> + (1 - k(p)) c, the kernel weight k(p) = exp(-p^2 / 2 tau).
(function (root) {
  // Pr(outcome | kind), a row per kind: it keeps its promise with probability keep; when it breaks one, its account
  // lines up with probability lined. The library's table, (0.98, 0.019, 0.001), (0.6, 0.38, 0.02), (0.6, 0.08, 0.32),
  // is keep (0.98, 0.6, 0.6) and lined (0.95, 0.95, 0.2).
  // The loss the listener expects per round from counting on each kind: how often that kind breaks a promise
  // times the cost of a broken promise, one cost for every kind (David, 2026-10-09: "it should just be the cost
  // of when the person is wrong ... It's just that they're unreliable more often"). The library's listener.py
  // supplies these directly as (0.4, 1.4, 8): 20 per broken promise, except that the accountable kind is padded
  // for at cost 1 and then misses one time in fifty, a stipulation David does not hold (plan §5.9 item 63,
  // narrowed 2026-10-08). An explicit params.ell overrides, which is how the test checks this file against it.
  const expectedLoss = (params) => params.ell || params.keep.map(k => (1 - k) * params.fail);
  const table = (keep, lined) => keep.map((k, i) => [k, (1 - k) * lined[i], (1 - k) * (1 - lined[i])]);
  const OUTCOME = { K: 0, L: 1, X: 2 };
  const MU = 0.1, STEPS_PER_ROUND = 10;
  const KINDS = ["reliable", "unreliable but accountable", "unaccountable"];
  const COLORS = ["#2f8f6b", "#d39b2a", "#c8473d"];
  const CHIPS = {
    K: { label: "kept the promise", color: COLORS[0] },
    L: { label: "broke it; the account lined up", color: COLORS[1] },
    X: { label: "broke it; the account did not line up", color: COLORS[2] },
  };
  const NEWS = {
    K: "the speaker kept a promise",
    L: "the speaker broke a promise, and the speaker's account of why lined up",
    X: "the speaker broke a promise, and the speaker's account of why did not line up",
  };
  const WIRES = [
    { id: "oneway", name: "One-way wire",
      note: "The valuation reads the belief and nothing comes back. The belief is Bayes' rule, scaled by how much the listener counts on the source." },
    { id: "exact", name: "Exact wire",
      note: "The valuation's cost is pulled back onto the belief. Every step it pushes the belief toward the kind of source that is cheapest to count on, more strongly the more the listener counts on it." },
  ];

  const dot = (a, b) => a.reduce((s, x, i) => s + x * b[i], 0);
  function softmax(l) {
    const m = Math.max(...l), e = l.map(x => Math.exp(x - m)), z = e.reduce((s, x) => s + x, 0);
    return e.map(x => x / z);
  }

  // The run: placements, beliefs and kernel weights at steps 0..T.
  function run(params, wire) {
    const { script, c, tau, p0 } = params;
    const ell = expectedLoss(params);
    const LOGPI = table(params.keep, params.lined).map(r => r.map(Math.log));
    // How much the listener counts on the speaker from placement p. The library's spatial parent uses a bell,
    // exp(-p^2 / 2 tau), flat at its top, so a listener beside the speaker gets no push either way whatever it
    // believes (plan §5.9 item 63, "a listener beside the source cannot leave"). The page's default is a peak,
    // exp(-p / tau), so a listener near the speaker leaves as soon as relying costs more than doing without (David,
    // 2026-10-09: "I guess so"), smoothed at the speaker as exp(-(sqrt(p^2 + eps^2) - eps) / tau), eps = 1e-3, so that
    // it is the library's (attention.spatial_kappa, kernel "peak"). A listener never stands on the speaker (David,
    // 2026-10-10: "that never happens! so there's a bound to how close they can be"): personal space, a cost
    // -sigma log(1 - k(p)) rising as the listener nears, pushes it away (listener.personal_space_box). The page's
    // defaults are the library's page_model() and checked equal to it (tests/test_animate.py).
    const EPS = 1e-3, sigma = params.sigma || 0, mu = params.mu || MU;
    const peak = params.kernel !== "gaussian";
    const rho = p => Math.sqrt(p * p + EPS * EPS);
    const kappa = peak ? p => Math.exp(-(rho(p) - EPS) / tau) : p => Math.exp(-p * p / (2 * tau));
    const dkappa = peak ? p => -kappa(p) * (p / rho(p)) / tau : p => -p / tau * kappa(p);
    let l = [0, 0, 0], p = p0;
    const ps = [], ys = [], ks = [];
    const T = STEPS_PER_ROUND * script.length;
    for (let t = 0; t < T; t++) {
      const r = Math.floor(t / STEPS_PER_ROUND), s = t % STEPS_PER_ROUND;
      const letter = s === 0 ? script[r] : null;
      const y = softmax(l), k = kappa(p);
      ps.push(p); ys.push(y); ks.push(k);
      const heard = params.hear === "weighted" ? k : 1;
      let xi = letter ? LOGPI.map(row => heard * row[OUTCOME[letter]]) : [0, 0, 0];
      if (wire === "exact") xi = xi.map((x, i) => x - k * ell[i]);
      const xp = -dkappa(p) * (dot(y, ell) - c) - (sigma ? sigma * dkappa(p) / (1 - k) : 0);
      // Bayes' rule on log-heights. A kind that gives what happened probability 0 goes to minus infinity and
      // stays there, so the heights are renormalized by their largest finite entry rather than the library's
      // belief-weighted mean (the belief is the same either way). If what happened is impossible under every kind,
      // the listener has nothing to update by, and the step leaves the belief alone.
      const nl = l.map((x, i) => x + xi[i]), finite = nl.filter(Number.isFinite);
      if (finite.length) { const top = Math.max(...finite); l = nl.map(x => x - top); }
      p = p + mu * xp;
    }
    ps.push(p); ys.push(softmax(l)); ks.push(kappa(p));
    return { ps, ys, ks };
  }

  const pct = y => `${(100 * y[0]).toFixed(0)}% reliable, ${(100 * y[1]).toFixed(0)}% unreliable but accountable, ` +
                   `${(100 * y[2]).toFixed(0)}% unaccountable`;
  const num = x => (Math.round(x * 10) / 10).toString();

  // One caption per step, in the step's own order: the move is set by the belief going in, the news arrives in
  // the same step, and the belief after it is the next step's.
  function captions(params, wire, { ps, ys, ks }) {
    const { script, c } = params, ell = expectedLoss(params), last = ps.length - 1, out = [];
    for (let t = 0; t <= last; t++) {
      if (t === last) { out.push({ step: t, text: `The run ends here. The listener believes the speaker is ${pct(ys[t])}.` }); continue; }
      const loss = dot(ys[t], ell), dp = ps[t + 1] - ps[t];
      const way = Math.abs(dp) < 1e-3 ? "so it stays about where it is"
                : dp < 0 ? "so it moves toward the speaker" : "so it moves away from the speaker";
      const move = `Going into this step it expects counting on the speaker to cost ${num(loss)}, ` +
                   `${loss < c ? "less" : "more"} than the ${num(c)} of doing without, ${way}.`;
      const r = Math.floor(t / STEPS_PER_ROUND), s = t % STEPS_PER_ROUND;
      let news;
      if (s === 0) news = `Round ${r + 1}: ${NEWS[script[r]]}. ` + (params.hear === "weighted"
        ? `The listener hears it at strength ${ks[t].toFixed(2)}.` : "The listener sees it.");
      else if (wire === "exact") news = "No news, but the cost of counting on the speaker pushes the belief toward the kind that is cheapest to count on.";
      else news = "No news, so the belief does not change.";
      out.push({ step: t, text: `${move} ${news} After this step it believes the speaker is ${pct(ys[t + 1])}.` });
    }
    return out;
  }

  function lane(params, w) {
    const tr = run(params, w.id), script = params.script;
    return {
      name: w.name, note: w.note,
      // The placement matters here only through the kernel weight, so the line is marked by that weight: how much
      // a listener standing there counts on the speaker.
      space: { label: "how much the listener counts on the speaker", min: -0.4, max: Infinity, scale: { kind: "compact", a: 1.5 },
               kernel: { center: 0, tau: params.tau, shape: params.kernel === "gaussian" ? "bell" : "peak" }, infLabel: "0%",
               ticks: [1, 0.75, 0.5, 0.25, 0.1, 0.01].map(k => ({
                 p: params.kernel === "gaussian" ? Math.sqrt(-2 * params.tau * Math.log(k)) : -params.tau * Math.log(k),
                 label: Math.round(100 * k) + "%" })) },
      boxes: [
        { id: "source", label: "the speaker", x: 0 },
        { id: "listener", label: "listener", x: tr.ps,
          sub: tr.ks.map(k => `${Math.round(100 * k)}%`),
          costs: tr.ys.map(y => dot(y, expectedLoss(params))),
          breaks: tr.ys.map(y => dot(y, params.keep.map(k => 1 - k))),
          readout: { kind: "pie", title: "belief about the speaker", labels: KINDS, colors: COLORS, values: tr.ys } },
      ],
      links: [{ from: "listener", to: "source", label: "counts on the speaker", strength: tr.ks }],
      events: [...script].map((ch, r) => ({ step: r * STEPS_PER_ROUND, box: "source", to: "listener", label: CHIPS[ch].label, color: CHIPS[ch].color,
                                             heard: params.hear === "weighted" ? tr.ks[r * STEPS_PER_ROUND] : 1 })),
      captions: captions(params, w.id, tr),
      trace: tr,
    };
  }

  const model = {
    defaults: { script: "KXKXKKXKXK", hear: "full", kernel: "peak", keep: [0.98, 0.6, 0.6], lined: [0.95, 0.95, 0.2], fail: 20,
                c: 5, tau: 0.8, p0: 0.5, sigma: 1, mu: 0.05 },
    samples: [
      { id: "A", name: "One exceptional failure", script: "KKKKLKKKKK",
        note: "The source keeps nine promises out of ten; its one failure comes with an account that lines up." },
      { id: "B", name: "Habitually late, accountable", script: "KLKLKKLKLK",
        note: "The source breaks four promises in ten, and each time its account of why lines up with what a check found." },
      { id: "C", name: "Unaccountable", script: "KXKXKKXKXK",
        note: "The same four broken promises, but no account lines up. B and C differ only in that." },
    ],
    cycle: "KLX",
    chips: CHIPS,
    wires: WIRES,
    // Grouped for the page: one group per kind of source the listener considers, then the rest.
    controls: [
      ...KINDS.flatMap((kind, i) => [
        { group: kind, color: COLORS[i], key: `keep.${i}`, label: "Keeps its promise", min: 0.01, max: 0.99, step: 0.01, format: "pct" },
        { group: kind, color: COLORS[i], key: `lined.${i}`, label: "When it breaks one, its account lines up", min: 0.01, max: 0.99, step: 0.01, format: "pct" },
      ]),
      { group: "The listener", key: "hear", type: "choice", label: "What the listener sees of the speaker",
        options: [{ value: "full", label: "Everything they do, wherever it stands" },
                  { value: "weighted", label: "Only as much as it counts on the speaker (the library's)" }] },
      // where: set on the picture itself rather than among the settings ("facts", beside the number it moves;
      // "drag", by dragging the listener).
      { group: "The listener", key: "kernel", type: "choice", label: "Shape of the hill",
        options: [{ value: "peak", label: "A peak: a listener beside the speaker can still leave" },
                  { value: "gaussian", label: "A bell, flat at the top (the library's)" }] },
      { group: "The listener", key: "fail", where: "facts", label: "Cost of a broken promise", min: 0, max: 40, step: 0.5 },
      { group: "The listener", key: "c", where: "facts", label: "Cost of doing without the speaker", min: 0, max: 10, step: 0.1 },
      { group: "The listener", key: "tau", label: "Width of the hill (how far away the listener still counts on the speaker)", min: 0.5, max: 2, step: 0.05 },
      { group: "The listener", key: "p0", where: "drag", label: "Where it starts", min: 0, max: 6, step: 0.05 },
    ],
    legend: KINDS.map((k, i) => ({ label: k, color: COLORS[i] })),
    stepsPerRound: STEPS_PER_ROUND,
    // The placement takes steps of MU times its force; the force's slope in the placement is at most
    // |<y, ell> - c| / tau (at the source, where the bell bends most). Past MU |ell_i - c| / tau = 1 a step can
    // overshoot where the listener would rest, so it jumps back and forth (past 2 it need not settle at all; the
    // library's own condition, PREDICTIONS.md, is that bound with 2).
    warning(params) {
      // the force's slope in the placement is at most |<y, ell> - c| / tau for the bell, / tau^2 for the peak
      const bend = params.kernel === "gaussian" ? params.tau : params.tau * params.tau;
      const ell = expectedLoss(params), worst = Math.max(...ell.map(x => Math.abs(x - params.c))) / bend;
      return MU * worst >= 1 ? "At these settings the listener's steps can be too big, so it may jump back and forth " +
        "around where it would rest. That comes from the size of the step the integrator takes, not from " +
        "anything the listener believes. A wider bell, or costs closer together, brings it back." : null;
    },
    kindOptions: KINDS.map((k, i) => ({ label: k, color: COLORS[i] })),
    outcomeOrder: ["K", "L", "X"],
    roundsPerPlay: 5,
    // One round drawn by chance from kind i's numbers.
    sample(params, i, rand = Math.random) {
      const u = rand(), k = params.keep[i], l = params.lined[i];
      return u < k ? "K" : u < k + (1 - k) * l ? "L" : "X";
    },
    // The model in plain sentences, with the current numbers, for the page's "Explain the math".
    explain(params) {
      const pc = x => Math.round(100 * x) + "%", ell = expectedLoss(params);
      const kinds = KINDS.map((k, i) => `<li><b>${k}</b>: keeps a promise ${pc(params.keep[i])} of the time; when it breaks one, ` +
        `its account lines up ${pc(params.lined[i])} of the time. Counting on it is expected to cost ${ell[i].toFixed(1)} a round.</li>`).join("");
      return `<p>The listener holds a <b>belief</b>: how likely it is that the speaker is each of three kinds.</p><ul>${kinds}</ul>` +
        `<p>The expected cost of a kind is how often it breaks a promise times the cost of a broken promise, ${params.fail}.</p>` +
        `<p>The listener also has a <b>place</b>, a number on the line. How much it counts on the speaker, between 0 and 1, ` +
        `is the height of the hill at that place: e<sup>&minus;place/${params.tau}</sup>, a distance from the speaker, never below 0. ` +
        `Right beside the speaker it counts on the speaker fully, and the hill still slopes there, so a listener beside the speaker can leave.</p>` +
        `<p>A round is ten steps. On a round's first step the speaker does one thing. ` + (params.hear === "weighted"
          ? `The listener hears it as strongly as it counts on the speaker, and updates its belief by Bayes' rule, scaled by that strength. ` +
            `So a listener that has stepped back hears almost nothing, and stops learning.</p>`
          : `The listener sees it wherever it stands, and updates its belief by Bayes' rule.</p>`) +
        `<p>Every step the listener moves to lower what it expects to lose: how much it counts on the speaker times the ` +
        `cost it expects from counting on the speaker, plus the rest times the cost of doing without, ${params.c}. It moves ` +
        `toward the speaker when counting on the speaker is expected to cost less than doing without, and away when more.</p>` +
        `<p class="src">From the arrangements library, <code>arrangements/listener.py</code> (book plan &sect;5.9 item 63); ` +
        `this page runs the same equations, checked against the library by <code>tests/test_animate.py</code>. ` +
        `One change: the library sets each kind's expected cost directly (0.4, 1.4, 8); this page derives it from one cost of a broken promise.</p>`;
    },
    build(params) {
      return {
        warning: this.warning(params),
        steps: STEPS_PER_ROUND * params.script.length + 1,
        ticks: { every: STEPS_PER_ROUND, label: "round", chips: [...params.script].map(ch => ({ ...CHIPS[ch], letter: ch })) },
        lanes: WIRES.map(w => lane(params, w)),
      };
    },
    run,
  };

  root.ARR_MODELS = root.ARR_MODELS || {};
  root.ARR_MODELS.listener = model;
  if (typeof module !== "undefined") module.exports = model;
})(typeof window !== "undefined" ? window : globalThis);
