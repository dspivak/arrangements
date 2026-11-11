// Loop beside magnetic_circuits.py, checked against smooth.PhiK recordings.
// Each winding reads the sum of other fields at its own ends. No pair matrix.
(function(root) {
  "use strict";
  const M = typeof module !== "undefined" ? require("./magnets.js") : root.magnetsModel;
  function fluxes(bodies) {
    const emitted = bodies.map(b => ({...b, kind:"magnet", I:b.turns*b.I}));
    return bodies.map((b,i) => {
      const e=M.ends(b);
      return b.turns*(M.fieldAt(emitted,i,e[0].x,e[0].y)-M.fieldAt(emitted,i,e[1].x,e[1].y));
    });
  }
  function step(bodies,tau,K=16) {
    let trial=bodies.map(b=>({...b}));
    for(let k=0;k<K;k++) {
      const flux=fluxes(trial);
      const next=trial.map((b,i)=>({...b,
        Q:bodies[i].Q-tau*b.I/2,
        phi:bodies[i].phi+tau*b.Q/(2*b.C),
        I:(b.phi-flux[i])/b.L}));
      if(k===K-1) return next.map((b,i)=>({...b,
        Q:2*b.Q-bodies[i].Q, phi:2*b.phi-bodies[i].phi, I:2*b.I-bodies[i].I}));
      trial=next;
    }
    throw new Error("At least one round is required");
  }
  function energy(bodies) {
    const flux=fluxes(bodies);
    return bodies.reduce((sum,b,i)=>sum+b.Q*b.Q/(2*b.C)+b.L*b.I*b.I/2+b.I*flux[i]/2,0);
  }
  function residual(bodies) {
    const flux=fluxes(bodies);
    return Math.max(...bodies.map((b,i)=>Math.abs(b.phi-b.L*b.I-flux[i])));
  }
  const model={fluxes,step,energy,residual};
  root.magneticCircuitsModel=model;
  if(typeof module!=="undefined") module.exports=model;
})(typeof window!=="undefined"?window:globalThis);
