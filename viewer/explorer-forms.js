/* The 1-form of an inclination field, written once: shown on its mark and checked against its wires.
 *
 * A field mark's `form` is a finite sum of terms (c, h), meaning  sum c d(h):  the image of the
 * list under the homomorphism of commutative monoids from the free commutative monoid on smooth
 * maps (c, h): V -> R^2 to the 1-forms on V (the feeding note's prop.push_generates, and the
 * library's feeding_parts.pushes). A cost k is the one term (-1, k); a push F dx is (F, x); a
 * Petri transition is (rate, a signed sum of what it gives and takes).
 *
 *   form = {
 *     letters:   {'Battery': 'b', 'Look distance': 'r'},  // wire label -> the letter the formulas use
 *     constants: {c_see: 0.002},                           // optional numbers, by letter
 *     where:     [['live', 'θ(b/b_0)']],                   // optional abbreviations, each from those before it
 *     terms:     [{c: 'c_see*look_r', h: 'r'}],            // the form is  sum c d(h)
 *   }
 *
 * c, h and the abbreviations are expressions in the arithmetic Python and JavaScript share
 * (+ - * / **, parentheses, numbers, names, calls), read with Python's precedences (-x**2 is
 * -(x**2)), so a library test can evaluate the same string. Names may be any letters, Greek
 * included; an underscore starts a subscript. Functions a page uses beyond the list below are
 * added to `functions` by that page (feeding adds its smooth step θ).
 *
 * `check` evaluates the form at the values on the mark's wires, the derivative of each h by
 * central differences, and compares the load it gives on every wire with the load the model
 * returned there. A formula is therefore never trusted: one that is mistyped shows as a mismatch.
 */
(function(root){
  'use strict';
  const functions={exp:Math.exp,log:Math.log,sqrt:Math.sqrt,abs:Math.abs,sin:Math.sin,cos:Math.cos,tanh:Math.tanh,
    min:Math.min,max:Math.max};
  // A wire may carry a list of numbers (a store of several). Arithmetic then goes entry by entry, a number
  // beside a list standing for each entry, a one-argument function is applied to each entry, and Σ adds a
  // list up. A term whose c and h are lists is the sum over entries of c_j d(h_j), a push of that width.
  const each=f=>(a,b)=>Array.isArray(a)?a.map((x,i)=>f(x,Array.isArray(b)?b[i]:b)):Array.isArray(b)?b.map(y=>f(a,y)):f(a,b);
  const plus=each((x,y)=>x+y),minus=each((x,y)=>x-y),times=each((x,y)=>x*y),over=each((x,y)=>x/y),raise=each((x,y)=>x**y);
  const total=x=>Array.isArray(x)?x.reduce((a,b)=>a+b,0):x;
  functions.Σ=total;functions.Σ.whole=true;
  const cache=new Map();
  function parse(text){
    if(cache.has(text))return cache.get(text);
    const tokens=String(text).match(/\s*(\*\*|[-+*/(),]|\d+\.?\d*(?:e[-+]?\d+)?|\.\d+|[\p{L}_][\p{L}\p{N}_]*)/giu)?.map(t=>t.trim())||[];
    if(tokens.join('').length!==String(text).replace(/\s+/g,'').length)throw Error('Cannot read the formula: '+text);
    let i=0;
    const peek=()=>tokens[i],take=t=>{if(tokens[i]!==t)throw Error(`Expected ${t} in the formula: ${text}`);i++;};
    function sum(){let a=product();while(peek()==='+'||peek()==='-'){const op=tokens[i++];a={op,a,b:product()};}return a;}
    function product(){let a=unary();while(peek()==='*'||peek()==='/'){const op=tokens[i++];a={op,a,b:unary()};}return a;}
    function unary(){if(peek()==='-'){i++;return {op:'neg',a:unary()};}if(peek()==='+'){i++;return unary();}return power();}
    function power(){const a=atom();if(peek()==='**'){i++;return {op:'**',a,b:unary()};}return a;}
    function atom(){
      const t=tokens[i++];
      if(t===undefined)throw Error('The formula ends early: '+text);
      if(t==='('){const a=sum();take(')');return a;}
      if(/^[\d.]/.test(t))return {op:'num',v:Number(t)};
      if(!/^[\p{L}_]/u.test(t))throw Error(`Unexpected ${t} in the formula: ${text}`);
      if(peek()!=='(')return {op:'name',n:t};
      i++;const args=[];
      if(peek()!==')')for(;;){args.push(sum());if(peek()!==',')break;i++;}
      take(')');return {op:'call',n:t,args};
    }
    const tree=sum();
    if(i<tokens.length)throw Error(`Unexpected ${tokens[i]} in the formula: ${text}`);
    cache.set(text,tree);return tree;
  }
  function value(t,env){
    switch(t.op){
      case 'num':return t.v;
      case 'name':{const v=env[t.n];if(typeof v!=='number'&&!Array.isArray(v))throw Error('No number for '+t.n);return v;}
      case 'call':{
        const f=env[t.n]??functions[t.n];if(typeof f!=='function')throw Error('No function '+t.n);
        const args=t.args.map(a=>value(a,env));
        return !f.whole&&args.length===1&&Array.isArray(args[0])?args[0].map(x=>f(x)):f(...args);
      }
      case 'neg':return times(-1,value(t.a,env));
      case '+':return plus(value(t.a,env),value(t.b,env));
      case '-':return minus(value(t.a,env),value(t.b,env));
      case '*':return times(value(t.a,env),value(t.b,env));
      case '/':return over(value(t.a,env),value(t.b,env));
      case '**':return raise(value(t.a,env),value(t.b,env));
    }
    throw Error('Unknown operation '+t.op);
  }
  // The names of a form at a point: its constants, the values by letter, then each abbreviation in turn.
  function environment(form,values){
    const env={...(form.constants||{}),...values};
    for(const [name,text] of form.where||[])env[name]=value(parse(text),env);
    return env;
  }
  // The load the form gives on each letter: sum over terms of c times the derivative of h along that letter.
  function loads(form,values){
    const out={},env=environment(form,values);
    const cs=form.terms.map(term=>value(parse(term.c),env));
    // The load along one number: a letter that is a number, or one entry of a letter that is a list.
    const along=(letter,index)=>{
      const at=index===undefined?values[letter]:values[letter][index],step=1e-5*Math.max(1,Math.abs(at));
      const moved=d=>environment(form,{...values,[letter]:index===undefined?at+d:values[letter].map((v,i)=>i===index?v+d:v)});
      const up=moved(step),down=moved(-step);
      return form.terms.reduce((sum,term,j)=>sum+total(times(cs[j],over(minus(value(parse(term.h),up),value(parse(term.h),down)),2*step))),0);
    };
    for(const letter of Object.keys(values))out[letter]=Array.isArray(values[letter])?values[letter].map((_,i)=>along(letter,i)):along(letter);
    return out;
  }
  /* readings: the mark's incoming wires, [{label, value, load}]. Returns {ok, rows, error}: one row per
   * lettered wire with the model's load (`want`) and the form's (`got`); `error` when it cannot be
   * evaluated (a wire with no letter that carries a load, an array on a lettered wire, a bad formula). */
  function check(form,readings,tolerance=1e-6){
    try{
      const values={},want={},rows=[];
      for(const r of readings){
        const letter=form.letters?.[r.label];
        if(letter===undefined){
          if(r.load&&[].concat(r.load).some(x=>Math.abs(x)>tolerance))throw Error(`The wire ${r.label} carries a load and has no letter`);
          continue;
        }
        const numbers=x=>typeof x==='number'||(Array.isArray(x)&&x.every(v=>typeof v==='number'));
        if(!numbers(r.value)||!numbers(r.load??0))throw Error(`The wire ${r.label} does not carry numbers`);
        values[letter]=r.value;want[letter]=letter in want?plus(want[letter],r.load??0):(r.load??0);
      }
      for(const letter of Object.values(form.letters||{}))if(!(letter in values))throw Error('No wire for the letter '+letter);
      const got=loads(form,values);
      for(const letter of Object.keys(values)){
        // Relative, since a load may be small (an eye's payment is thousandths); the floor is the differences' round-off.
        const g=[].concat(got[letter]),w=[].concat(want[letter]);
        const ok=g.every((x,i)=>{const y=w.length===1?w[0]:w[i];return Math.abs(x-y)<=1e-5*Math.max(Math.abs(x),Math.abs(y))+1e-8;});
        rows.push({letter,got:got[letter],want:want[letter],ok});
      }
      return {ok:rows.every(r=>r.ok),rows};
    }catch(error){return {ok:false,rows:[],error:error.message};}
  }
  /* A static map's law: {letters, constants, where}, the same fields as a form's, the map's value being the
   * last abbreviation. readings: the map's entering wires, [{label, value}]; want: the value the model put
   * on the wire that leaves it. Returns {ok, got, want, error}, so a law is no more trusted than a form. */
  function lawCheck(law,readings,want){
    try{
      const values={};
      for(const r of readings){const letter=law.letters?.[r.label];if(letter!==undefined)values[letter]=r.value;}
      for(const letter of Object.values(law.letters||{}))if(!(letter in values))throw Error('No wire for the letter '+letter);
      const got=environment(law,values)[law.where.at(-1)[0]],g=[].concat(got),w=[].concat(want);
      const ok=g.length===w.length&&g.every((x,i)=>Math.abs(x-w[i])<=1e-9*Math.max(Math.abs(x),Math.abs(w[i]))+1e-12);
      return {ok,got,want};
    }catch(error){return {ok:false,error:error.message};}
  }

  // ---- Writing a formula. A run is {text, shift} with shift 1 a superscript, -1 a subscript, and
  // `italic` for a one-letter name. `plain` joins runs into one line of text.
  const rank=t=>({'+':1,'-':1,'*':2,'/':2,neg:3,'**':4})[t.op]||5;
  const hoist=t=>{ // (-a)*b is written -(a*b)
    if((t.op==='*'||t.op==='/')){const a=hoist(t.a);if(a.op==='neg')return {op:'neg',a:{op:t.op,a:a.a,b:t.b}};return {...t,a};}
    return t;
  };
  function name(n){
    const cut=n.indexOf('_'),base=cut>0?n.slice(0,cut):n,sub=cut>0?n.slice(cut+1):'';
    const runs=[{text:base,italic:[...base].length===1}];
    if(sub)runs.push({text:sub,shift:-1});
    return runs;
  }
  const number=v=>v===.5?'½':String(+v.toPrecision(6));
  function runs(t,parent=0,right=false){
    t=hoist(t);
    const wrap=(inner,needs)=>needs?[{text:'('},...inner,{text:')'}]:inner;
    const r=rank(t);
    switch(t.op){
      case 'num':return [{text:number(t.v)}];
      case 'name':return name(t.n);
      case 'call':return [...name(t.n),{text:'('},...t.args.flatMap((a,k)=>[...(k?[{text:', '}]:[]),...runs(a)]),{text:')'}];
      // A sum under a minus sign is wrapped, a product is not: −(a + b), −a b.
      case 'neg':return wrap([{text:'−'},...runs(t.a,2)],parent>1||right);
      case '+':case '-':return wrap([...runs(t.a,1),{text:t.op==='+'?' + ':' − '},...runs(t.b,1,true)],parent>1||(parent===1&&right));
      case '*':{
        // Two letters are set side by side; a number, or a name of several letters, takes a dot, so that
        // live·(a + b) is not read as a function applied.
        const b=hoist(t.b),word=r=>r&&r.italic===false&&[...r.text].length>1;
        const dot=b.op==='num'||(b.op==='**'&&b.a.op==='num')||word(runs(t.a,2).filter(r=>!r.shift).at(-1))||word(runs(t.b,2,true)[0]);
        return wrap([...runs(t.a,2),{text:dot?'·':' '},...runs(t.b,2,true)],parent>2);
      }
      case '/':return wrap([...runs(t.a,2),{text:'/'},...runs(t.b,3,true)],parent>2);
      case '**':{
        // A sum, product or minus as the base wraps itself (its rank is below 5); a power as the base is wrapped here.
        const power=t.b.op==='num'?number(t.b.v):plain(runs(t.b));
        return [...wrap(runs(t.a,5),t.a.op==='**'),{text:power,shift:1}];
      }
    }
    return [{text:'?'}];
  }
  const plain=list=>list.map(r=>r.shift===1&&!/^[0-9]$/.test(r.text)?'^('+r.text+')':r.shift===1?'⁰¹²³⁴⁵⁶⁷⁸⁹'[+r.text]:r.shift===-1?'_'+r.text:r.text).join('');
  // One term c d(h), as {negative, runs}: the sign is taken out so a sum of terms reads a − b.
  function term(entry){
    let c=hoist(parse(entry.c)),negative=false;
    if(c.op==='neg'){negative=true;c=c.a;}else if(c.op==='num'&&c.v<0){negative=true;c={op:'num',v:-c.v};}
    const h=parse(entry.h),d=h.op==='name'?[{text:'d'},...name(h.n)]:[{text:'d('},...runs(h),{text:')'}];
    const one=c.op==='num'&&c.v===1;
    return {negative,runs:one?d:[...runs(c,2),{text:' '},...d]};
  }
  // The whole form on one line: its terms joined by their signs; the zero form is 0.
  function written(form){
    if(!form.terms?.length)return [{text:'0'}];
    return form.terms.flatMap((entry,k)=>{const t=term(entry);return [{text:k?(t.negative?' − ':' + '):(t.negative?'−':'')},...t.runs];});
  }
  // Each abbreviation as name = formula, in order.
  const definitions=form=>(form.where||[]).map(([n,text])=>[...name(n),{text:' = '},...runs(parse(text))]);
  const api={functions,parse,value,environment,loads,check,lawCheck,runs,term,written,definitions,plain};
  if(typeof module!=='undefined'&&module.exports)module.exports=api;else root.ArrangementForms=api;
})(typeof globalThis!=='undefined'?globalThis:this);
