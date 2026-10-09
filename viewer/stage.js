// Drawing one lane of a scene at a continuous time t (in steps), shared by player.html and live.html.
// A lane has a space (a line of placements, optionally the attention kernel's bell), boxes (fixed, or one
// placement per step), links (a strength per step) and events (a flash on the sender, and on the receiver at
// the strength it is heard). tok holds the page's theme colors and fonts.
(function (root) {
  const lerp = (a, b, u) => a + (b - a) * u;
  function at(arr, s, T) {
    if (typeof arr === "number") return arr;
    const i = Math.max(0, Math.min(Math.floor(s), T - 1)), j = Math.min(i + 1, T - 1), u = Math.max(0, s - i);
    const a = arr[i], b = arr[j];
    return Array.isArray(a) ? a.map((x, k) => lerp(x, b[k], u)) : lerp(a, b, u);
  }
  function roundRect(ctx, x, y, w, h, r) {
    ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r);
    ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath();
  }
  function hexA(hex, a) {
    const n = parseInt(hex.slice(1), 16);
    return `rgba(${(n >> 16) & 255},${(n >> 8) & 255},${n & 255},${a})`;
  }

  // Where things go on a lane W wide and H tall: X maps a placement to a pixel, toPlacement maps it back. A space
  // with scale {kind: "compact", a} is drawn squeezed, p / (|p| + a), so the whole line fits and its far end,
  // max = Infinity, is the right edge: a box that walks off forever is still on screen.
  function layout(W, H, lane) {
    const sp = lane.space, padL = 28, padR = 28, axisY = H - 28, rowFixed = 44, rowMoving = axisY - 46;
    const a = sp.scale && sp.scale.kind === "compact" ? sp.scale.a : null;
    const c = a ? (p) => (p === Infinity ? 1 : p / (Math.abs(p) + a)) : (p) => p;
    const cinv = a ? (u) => (u >= 1 ? Infinity : a * u / (1 - Math.abs(u))) : (u) => u;
    const u0 = c(sp.min), u1 = c(sp.max), w = W - padL - padR;
    const X = (p) => padL + (c(p) - u0) / (u1 - u0) * w;
    const toPlacement = (x) => cinv(u0 + (x - padL) / w * (u1 - u0));
    // Ticks: the space's own, {p, label}, else whole placements (doubling ones on a squeezed line).
    let ticks = sp.ticks ? sp.ticks : (a ? [0, 1, 2, 4, 8, 16].filter(p => p >= sp.min) : []).map(p => ({ p, label: String(p) }));
    if (!a && !sp.ticks) for (let p = Math.ceil(sp.min); p <= sp.max; p++) ticks.push({ p, label: String(p) });
    return { X, toPlacement, axisY, rowFixed, rowMoving, ticks, u0, u1, cinv, infinite: sp.max === Infinity };
  }

  function drawLane(ctx, W, H, lane, t, T, tok) {
    const sp = lane.space;
    ctx.clearRect(0, 0, W, H);
    const { X, axisY, rowFixed, rowMoving, ticks, u0, u1, cinv, infinite } = layout(W, H, lane);

    if (sp.kernel) {                       // the bell: the weight a box at each placement gives the source
      const k = sp.kernel, top = rowFixed + 26, hgt = axisY - top;
      ctx.beginPath(); ctx.moveTo(X(sp.min), axisY);
      for (let i = 0; i <= 200; i++) {
        const p = cinv(u0 + (u1 - u0) * i / 200);
        const h = p === Infinity ? 0 : k.shape === "peak" ? (p < k.center ? 0 : Math.exp(-(p - k.center) / k.tau))
                : Math.exp(-((p - k.center) ** 2) / (2 * k.tau));
        ctx.lineTo(X(p), axisY - hgt * h);
      }
      ctx.lineTo(X(sp.max), axisY); ctx.closePath();
      ctx.fillStyle = tok.faint; ctx.fill();
    }
    ctx.strokeStyle = tok.rule; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(X(sp.min), axisY + 0.5); ctx.lineTo(X(sp.max), axisY + 0.5); ctx.stroke();
    ctx.fillStyle = tok.muted; ctx.font = "11px " + tok.data; ctx.textAlign = "center"; ctx.textBaseline = "top";
    for (const tk of ticks) {
      ctx.beginPath(); ctx.moveTo(X(tk.p), axisY); ctx.lineTo(X(tk.p), axisY + 4); ctx.stroke();
      ctx.fillText(tk.label, X(tk.p), axisY + 7);
    }
    if (infinite) {
      ctx.beginPath(); ctx.moveTo(X(Infinity), axisY); ctx.lineTo(X(Infinity), axisY + 4); ctx.stroke();
      ctx.textAlign = "right";
      if (sp.infLabel) ctx.fillText(sp.infLabel, X(Infinity), axisY + 7);
      else { ctx.font = "13px " + tok.data; ctx.fillText("\u221e", X(Infinity), axisY + 5); ctx.font = "11px " + tok.data; }
    }
    if (sp.ticks) {
      // a line marked by what it means is labeled by the page, beside the picture
    } else if (infinite) { ctx.textAlign = "left"; ctx.fillText(sp.label, X(sp.min), axisY + 7); }
    else { ctx.textAlign = "right"; ctx.fillText(sp.label, X(sp.max), axisY + 7); }

    const pos = {};
    for (const b of lane.boxes) {
      const fixed = typeof b.x === "number";
      pos[b.id] = { x: X(at(b.x, t, T)), y: fixed ? rowFixed : rowMoving, size: b.readout ? 46 : 34 };
    }
    for (const b of lane.boxes) {          // a fading trail behind a box that moves
      if (typeof b.x === "number") continue;
      ctx.fillStyle = tok.muted;
      const now = Math.min(Math.floor(t), T - 1);
      for (let s = Math.max(0, now - 40); s <= now; s += 2) {
        ctx.globalAlpha = 0.08 + 0.25 * (1 - (now - s) / 40);
        ctx.beginPath(); ctx.arc(X(b.x[s]), rowMoving + 30, 1.6, 0, 2 * Math.PI); ctx.fill();
      }
      ctx.globalAlpha = 1;
    }
    for (const link of lane.links || []) {
      const a = pos[link.from], b = pos[link.to], s = at(link.strength, t, T);
      ctx.strokeStyle = tok.accent; ctx.globalAlpha = 0.12 + 0.88 * s; ctx.lineWidth = 0.75 + 3 * s;
      ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y); ctx.stroke(); ctx.globalAlpha = 1;
    }
    const flashes = {};
    for (const e of lane.events || []) {
      const age = t - e.step;
      if (age < 0 || age > 6) continue;
      const fade = Math.max(0, 1 - age / 6);
      flashes[e.box] = { color: e.color, a: fade };
      if (e.to) {
        const link = (lane.links || []).find(l => (l.from === e.to && l.to === e.box) || (l.from === e.box && l.to === e.to));
        const heard = e.heard != null ? e.heard : link ? link.strength[Math.min(e.step, T - 1)] : 1;
        flashes[e.to] = { color: e.color, a: fade * heard, ring: true };
      }
    }
    for (const b of lane.boxes) {
      const p = pos[b.id], sz = p.size, f = flashes[b.id];
      if (f && f.ring) {
        ctx.strokeStyle = hexA(f.color, f.a); ctx.lineWidth = 5;
        roundRect(ctx, p.x - sz / 2 - 5, p.y - sz / 2 - 5, sz + 10, sz + 10, 11); ctx.stroke();
      }
      roundRect(ctx, p.x - sz / 2, p.y - sz / 2, sz, sz, 8);
      ctx.fillStyle = tok.panel; ctx.fill();
      if (f && !f.ring) { ctx.fillStyle = hexA(f.color, 0.25 + 0.75 * f.a); ctx.fill(); }
      ctx.strokeStyle = tok.ink; ctx.lineWidth = 1.25; ctx.stroke();
      const r = b.readout;
      if (r && r.kind === "pie") {
        const v = at(r.values, t, T), rad = sz / 2 - 6;
        let a0 = -Math.PI / 2;
        v.forEach((x, k) => {
          const a1 = a0 + 2 * Math.PI * x;
          ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.arc(p.x, p.y, rad, a0, a1); ctx.closePath();
          ctx.fillStyle = r.colors[k]; ctx.fill(); a0 = a1;
        });
      } else if (r && r.kind === "meter") {
        const v = at(r.values, t, T), u = Math.max(0, Math.min(1, (v - r.min) / (r.max - r.min)));
        ctx.fillStyle = r.color || tok.accent;
        ctx.fillRect(p.x - sz / 2 + 6, p.y + sz / 2 - 6 - (sz - 12) * u, sz - 12, (sz - 12) * u);
      }
      ctx.fillStyle = tok.ink; ctx.font = "500 12px " + tok.body; ctx.textAlign = "center"; ctx.textBaseline = "bottom";
      ctx.fillText(b.label, p.x, p.y - sz / 2 - 6);
      if (b.sub) {                         // a short per-step reading beside the box, and an arrow for which way it is going
        const i = Math.max(0, Math.min(Math.floor(t), T - 1));
        ctx.fillStyle = tok.muted; ctx.font = "11px " + tok.data; ctx.textAlign = "left"; ctx.textBaseline = "middle";
        const pxv = typeof b.x === "number" ? 0 : X(b.x[Math.min(i + 1, T - 1)]) - X(b.x[i]);
        const len = Math.min(40, Math.abs(pxv) * 25);
        let tx = p.x + sz / 2 + 10;
        if (len > 2) {
          const dir = Math.sign(pxv), y0 = p.y, x0 = dir > 0 ? p.x + sz / 2 + 6 : p.x - sz / 2 - 6;
          ctx.strokeStyle = tok.muted; ctx.lineWidth = 2;
          ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x0 + dir * len, y0); ctx.stroke();
          ctx.beginPath(); ctx.moveTo(x0 + dir * len, y0); ctx.lineTo(x0 + dir * (len - 6), y0 - 4);
          ctx.lineTo(x0 + dir * (len - 6), y0 + 4); ctx.closePath(); ctx.fillStyle = tok.muted; ctx.fill();
          if (dir > 0) tx = x0 + len + 6;
        }
        ctx.fillStyle = tok.muted;
        const lines = b.sub[i].split("\n"), w = Math.max(...lines.map(l => ctx.measureText(l).width));
        if (tx + w > W - 4) { ctx.textAlign = "right"; tx = p.x - sz / 2 - 10 - (len > 2 && pxv < 0 ? len + 4 : 0); }
        lines.forEach((l, k) => ctx.fillText(l, tx, p.y + sz / 2 + 2 - (lines.length - 1 - k) * 14));
      }
    }
  }

  root.ArrStage = { drawLane, at, layout };
})(typeof window !== "undefined" ? window : globalThis);
