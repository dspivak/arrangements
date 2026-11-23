/* The feeding explorer's adapter, read from the library composite (book P109 (f), act (4)).
 *
 * `data` is a module written by scripts/export_feeding_explorer.py: `scene`, the diagram of
 * feeding_composed.feeding_with_slot as explore_composite.diagram reads it from the composite's
 * construction (boxes, stores, static maps, field marks with their rubric kinds and formulas,
 * one wire per value a mark or map reads); `branches`, where each wire's numbers are in the
 * output of `readings`; `states`, where each store's configuration is in `q`; and two functions
 * generated from the library's own code, `readings(state, places, params)` (every wire's value
 * and the load its reader sends back) and `step(state, places, params)` (four Runge-Kutta rounds
 * of length `h`). Nothing about feeding is typed here: this file only copies numbers into the
 * scene. The one function a formula calls that the explorer lacks, the smooth step θ, is
 * `feeding.theta`, written out below for the forms check, which compares every formula drawn
 * with the loads `readings` returns and marks a mismatch red.
 *
 * The adapter keeps the interface the explorer and the parent's feeding view share,
 * `{snapshot, step, reset}`; `state`, `places` and `params` are its fields, so a page with
 * another clock can set them and call `view.refresh()`.
 */
(function(root){
  'use strict';
  // feeding.theta: exp(-1/x) is taken as 0 below x = 1/700, as the library does.
  const rho=x=>x>1/700?Math.exp(-1/x):0;
  const theta=t=>{const a=rho(t),b=rho(1-t);return a/(a+b);};
  const thetaWords={head:'θ(t) = ρ(t)/(ρ(t) + ρ(1 − t))',
    words:'a smooth step, 0 for t\u00a0≤\u00a00 and 1 for t\u00a0≥\u00a01, where ρ(t) is exp(−1/t) for t\u00a0>\u00a00 and 0 otherwise'};
  if(root.ArrangementForms){root.ArrangementForms.functions.θ=theta;root.ArrangementForms.functionWords.θ=thetaWords;}

  function compositeAdapter(data,options={}){
    // A generated step of the whole composite takes milliseconds, so a frame takes a few at most and
    // the page runs slower than the model's time when the work would not fit.
    const maxSteps=options.maxSteps??4;
    const adapter={
      state:Float64Array.from(data.initial),places:data.places.slice(),params:data.defaults.slice(),
      time:0,pending:0,
      snapshot(){
        const r=data.readings(adapter.state,adapter.places,adapter.params);
        const pick=(a,b)=>b.width===1?a[b.eps]:Array.from(a.slice(b.eps,b.eps+b.width));
        const wires=data.scene.wires.map((w,i)=>({...w,value:pick(r.values,data.branches[i]),load:pick(r.loads,data.branches[i])}));
        // A store's inclination: the loads on its wires, added coordinate by coordinate.
        const inclination=new Map();
        data.branches.forEach((b,i)=>{
          const total=inclination.get(b.source)||[];
          b.indices.forEach((k,j)=>{total[k]=(total[k]||0)+r.loads[b.eps+j];});
          inclination.set(b.source,total);
        });
        const value=new Map(data.states.map(s=>[s.node,s.size===1?r.q[s.start]:Array.from(r.q.slice(s.start,s.start+s.size))]));
        const nodes=data.scene.nodes.map(n=>{
          if(n.kind!=='state')return n;
          const v=value.get(n.id),inc=inclination.get(n.id)||[];
          const full=Array.isArray(v)?v.map((_,k)=>inc[k]||0):(inc[0]||0);
          return {...n,value:v,configuration:v,inclination:full};
        });
        return {...data.scene,time:adapter.time,nodes,wires};
      },
      step(seconds){
        adapter.pending+=seconds;
        for(let n=0;adapter.pending>=data.h&&n<maxSteps;n++){
          adapter.state=Float64Array.from(data.step(adapter.state,adapter.places,adapter.params));
          adapter.pending-=data.h;adapter.time+=data.h;
        }
        adapter.pending=Math.min(adapter.pending,data.h);
      },
      reset(){adapter.state=Float64Array.from(data.initial);adapter.time=0;adapter.pending=0;},
    };
    if(!data.step)delete adapter.step;
    return adapter;
  }
  const api={compositeAdapter,theta,thetaWords};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.FeedingExplorerComposite=api;
})(typeof globalThis!=='undefined'?globalThis:this);
