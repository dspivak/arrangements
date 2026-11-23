// Run a function written by scripts/export_step_js.py (or any file exposing one) on given inputs.
//   node exported-step-driver.js FILE.js INPUTS.json [--global NAME] [--fn step] [--args] [--repeat N]
// INPUTS.json is a JSON array with one entry per call: the state, or with --args the list of
// arguments (state, places, params). Prints {"results": [...], "msPerCall": ...} as JSON: one
// result per call (an array, or an object of arrays); with --repeat N, the mean time of N calls
// from the first entry, each call's result fed back as the next call's state.
const fs = require('node:fs');
const path = require('node:path');
const args = process.argv.slice(2);
const option = (flag, fallback) => {
  const i = args.indexOf(flag);
  return i < 0 ? fallback : args[i + 1];
};
const file = path.resolve(args[0]);
const inputs = JSON.parse(fs.readFileSync(args[1], 'utf8'));
const loaded = require(file);
const name = option('--global', null);
const api = name ? globalThis[name] : loaded;
const fn = api[option('--fn', 'step')];
const spread = args.includes('--args');
const call = x => (spread ? fn(...x) : fn(x));
const plain = r => (Array.isArray(r) || ArrayBuffer.isView(r) ? Array.from(r)
  : Object.fromEntries(Object.entries(r).map(([k, v]) => [k, Array.from(v)])));
const results = inputs.map(x => plain(call(x)));
let msPerCall = null;
const repeat = Number(option('--repeat', 0));
if (repeat > 0) {
  let x = inputs[0];
  const advance = () => { const r = call(x); x = spread ? [r, ...x.slice(1)] : r; };
  for (let i = 0; i < Math.min(repeat, 50); i++) advance();
  const began = process.hrtime.bigint();
  for (let i = 0; i < repeat; i++) advance();
  msPerCall = Number(process.hrtime.bigint() - began) / 1e6 / repeat;
}
console.log(JSON.stringify({results, msPerCall}));
