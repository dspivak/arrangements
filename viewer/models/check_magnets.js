// Checks the page's model, magnets.js, against the library's arrangement at the settings of tests/test_magnets.py and of the book's
// run notes/runs/magnets-2524b69e (its series.json, recorded from ``plane . (bodies)`` run through PhiK).
// node viewer/models/check_magnets.js PATH/TO/series.json
const fs = require("fs");
const M = require("./magnets.js");
const S = JSON.parse(fs.readFileSync(process.argv[2], "utf8"));
const HEAVY = 1e12;
let bad = 0;
function near(name, got, want, tol) {
  const ok = Math.abs(got - want) <= tol;
  if (!ok) bad++;
  console.log((ok ? "ok   " : "FAIL ") + name + ": page " + got + ", library " + want + " (within " + tol + ")");
}
const body = (kind, I, h, x, y, th, o) => Object.assign({ kind, I, h, ell: 1, x, y, th, px: 0, py: 0, pth: 0, m: 1, J: HEAVY }, o || {});
function run(bodies, tau, steps, read) {
  const out = [read(bodies)];
  for (let k = 0; k < steps; k++) { bodies = M.step(bodies, tau); out.push(read(bodies)); }
  return out;
}

// forces at rest
near("two wires, like currents: force on the left one", M.loads([body("wire", 1, 0, -1, 0, 0), body("wire", 1, 0, 1, 0, 0)])[0].fx, 1 / (4 * Math.PI), 1e-12);
for (const [name, th, I, want] of [["head to tail", Math.PI / 2, 1, 0.002868], ["head to tail, one flipped", Math.PI / 2, -1, -0.002868],
                                   ["side by side", 0, 1, -0.003032], ["side by side, one flipped", 0, -1, 0.003032]])
  near("two magnets, " + name + ": force on the left one", M.loads([body("magnet", 1, 0.25, -1.5, 0, th), body("magnet", I, 0.25, 1.5, 0, th)])[0].fx, want, 5e-7);

// the runs, step for step
for (const key of ["h2_head to tail", "h2_side by side"]) {
  const th = key.includes("head") ? Math.PI / 2 : 0;
  const r = run([body("magnet", 1, 0.25, -1.5, 0, th), body("magnet", 1, 0.25, 1.5, 0, th)], 0.05, 300, b => b[1].x - b[0].x);
  near(key + ": distance after 300 steps", r[300], S[key].r[300], 1e-9);
}
{
  const th = run([body("magnet", 1, 0.25, -1.5, 0, Math.PI / 2, { m: HEAVY }), body("magnet", 1, 0.25, 1.5, 0, Math.PI / 2 + 0.5, { m: HEAVY, J: 0.01 })],
                 0.05, 400, b => b[1].th - Math.PI / 2);
  near("a magnet turning: angle after 197 steps", th[197], S.h3.theta[197], 1e-8);
  near("a magnet turning: angle after 400 steps", th[400], S.h3.theta[400], 1e-8);
}
{
  const sp = run([body("loop", 0, 0.5, 0, 0, 0, { m: HEAVY }), body("magnet", 1, 0.25, -5, 1, 0, { px: 1 })], 0.01, 1000, b => b[1].px);
  near("a magnet past a dead loop: least speed", Math.min(...sp), Math.min(...S.h4.speed), 1e-9);
  near("a magnet past a dead loop: speed at the end", sp[1000], S.h4.speed[1000], 1e-9);
}
{
  // bumpers and drag: two carts, each a magnet with a bumper at each wire, coming to rest where pull equals push (run (h9))
  const mk = x => body("magnet", 2, 0.25, x, 0, Math.PI / 2, { bump: 1 });
  let b = [mk(-1.5), mk(1.5)];
  const r = [b[1].x - b[0].x];
  for (let k = 0; k < 6000; k++) { b = M.step(b, 0.01, null, { drag: 0.5, turn: false }); r.push(b[1].x - b[0].x); }
  // the library's series keeps every fifth step, so the least distance is compared on those steps
  near("magnets with bumpers and drag: least distance (every fifth step)", Math.min(...r.filter((_, k) => k % 5 === 0)), Math.min(...S["h9_as built"].r), 1e-9);
  near("magnets with bumpers and drag: distance after 6000 steps", r[6000], S["h9_as built"].r[1200], 1e-9);
  near("magnets with bumpers and drag: where they rest", r[6000], 0.395410, 1e-3);
}
// P76(j): recorded from the field-valued library, including external inputs.
const path = require("path");
const C = require("./magnetic_circuits.js");
const J = JSON.parse(fs.readFileSync(process.argv[3] || path.join(__dirname,"magnetic_circuits_reference.json"),"utf8"));
for (const row of J.circuits) {
  let b=row.places.map((p,i)=>({x:p[0],y:p[1],th:p[2],L:1,C:1,h:.5,turns:row.turns[i],Q:i===0?1:0,phi:0,I:0}));
  let gap=0,drift=0,residual=0;
  for(let k=0;k<=row.steps;k++) {
    if(k%row.stride===0) b.forEach((v,i)=>[v.Q,v.phi,v.I].forEach((a,j)=>{
      gap=Math.max(gap,Math.abs(a-row.states[k/row.stride][i][j]));
    }));
    drift=Math.max(drift,Math.abs(C.energy(b)-.5));residual=Math.max(residual,C.residual(b));
    if(k<row.steps)b=C.step(b,row.tau,row.K);
  }
  near(row.name+": all recorded circuit states",gap,0,1e-10);
  near(row.name+": total energy change",drift,0,1e-8);
  near(row.name+": current/flux residual",residual,0,1e-9);
}
for(const row of J.open) {
  const b=row.positions.map((p,i)=>body(i===0?"magnet":row.kind,i===0?1:.7,i===0?.25:.4,...p));
  const loads=M.loads(b);let gap=0;
  loads.forEach((v,i)=>[v.fx,v.fy,v.tq].forEach((a,k)=>gap=Math.max(gap,Math.abs(a-row.loads[i][k]))));
  near("open positions, "+row.kind+": force and torque",gap,0,1e-12);
}
{
  let b=[body("magnet",1,.25,-1,.1,.4),body("magnet",.7,.4,1,.2,-.3,{J:.2})];
  let gap=0;
  for(let k=0;k<J.held.states.length;k++) {
    const a=b[1];[a.x,a.y,a.th,a.px,a.py,a.pth].forEach((v,j)=>gap=Math.max(gap,Math.abs(v-J.held.states[k][j])));
    if(k+1<J.held.states.length)b=M.step(b,J.held.tau,[true,false],{drag:0,turn:true});
  }
  near("one held, one free: every cart state",gap,0,1e-11);
}
// Whole-plane smoothing: compare the core and boundary with library field
// values and force pushbacks, so the force tests check the chain-rule factor.
const core=JSON.parse(fs.readFileSync(path.join(__dirname,'magnetic_smoothing_reference.json'),'utf8'));
near('smoothing radius',M.r0,core.r0,0);
for(const row of core.rows){
  near('smoothed field at distance '+row.distance,M.g(row.distance,0),row.g,1e-12);
  for(const kind of ['wire','bumper']){
    const b=[body('wire',kind==='wire'?1:0,0,0,0,0,{bump:kind==='bumper'?1:0}),
             body('wire',kind==='wire'?1:0,0,row.distance,0,0,{bump:kind==='bumper'?.7:0})];
    const load=M.loads(b)[0],got=[load.fx,load.fy,load.tq],want=row[kind];
    const gap=Math.max(...got.map((v,i)=>Math.abs(v-want[i])));
    near(kind+' force in smoothed core at '+row.distance,gap,0,1e-10*Math.max(1,...want.map(Math.abs)));
  }
}
console.log(bad ? bad + " check(s) failed" : "all checks passed");
process.exit(bad ? 1 : 0);
