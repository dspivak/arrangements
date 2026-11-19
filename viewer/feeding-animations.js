/* Optional library-supplied animations. Each renderer receives the same observed
 * scene and a local SVG group; it owns no model state and never steps it. */
(function(root){
  const line=(svg,g,x1,y1,x2,y2,color='#51718a',width=3)=>svg('line',{x1,y1,x2,y2,stroke:color,'stroke-width':width,'stroke-linecap':'round'},g);
  function wheel({svg,group:g,node,scene}){
    const value=typeof node.value==='number'?node.value:scene.state[5];
    const rotor=svg('g',{transform:`rotate(${value*90})`},g);
    svg('circle',{r:25,fill:'#fff',stroke:'#2b5974','stroke-width':4},rotor);
    for(let i=0;i<4;i++){const a=i*Math.PI/2;line(svg,rotor,0,0,22*Math.cos(a),22*Math.sin(a));}
    svg('circle',{r:5,fill:'#196f9d'},g);
  }
  function battery({svg,group:g,node,scene}){
    const v=typeof node.value==='number'?node.value:scene.state[0];
    svg('rect',{x:-32,y:-18,width:64,height:36,rx:5,fill:'none',stroke:'#317598','stroke-width':2},g);
    svg('rect',{x:-28,y:-14,width:56*Math.max(0,Math.min(1,v/10)),height:28,rx:3,fill:'#5aa9c5'},g);
    svg('text',{y:6,'text-anchor':'middle','font-size':14},g,v.toFixed(2));
  }
  function casing({svg,group:g,node,scene}){
    const {food,opening,height,red}=scene.casings[node.index??0];
    const dy=-Math.max(-.1,height+.4)*28,gap=Math.min(1,Math.max(0,opening))*22;
    const color=red>.5?'#c65346':'#8293a1';
    svg('path',{d:`M ${-gap} ${dy-20} L -30 ${dy-20} L -30 ${dy+18} Q 0 ${dy+35} 30 ${dy+18} L 30 ${dy-20} L ${gap} ${dy-20}`,fill:'#fff',stroke:color,'stroke-width':3},g);
    svg('rect',{x:-25,y:dy+18-30*Math.min(1,Math.max(0,food/10)),width:50,height:30*Math.min(1,Math.max(0,food/10)),rx:4,fill:color,opacity:.6},g);
    svg('text',{y:dy-28,'text-anchor':'middle','font-size':13},g,food.toFixed(2));
  }
  function receiver({svg,group:g,scene}){
    svg('rect',{x:-45,y:-20,width:90,height:33,rx:8,fill:'#dbe9f1',stroke:'#486d85','stroke-width':2},g);
    for(const x of [-27,27]){const w=svg('g',{transform:`translate(${x},25) scale(.42)`},g);wheel({svg,group:w,node:{},scene});}
    const b=svg('g',{transform:'translate(-18,-4) scale(.42)'},g);battery({svg,group:b,node:{},scene});
    line(svg,g,25,-20,25,-40,'#ae8527');line(svg,g,25,-40,25+scene.state[6]*3,-36,'#ae8527',2);
    line(svg,g,45,-6,65,25-scene.state[2]*30,'#825e9f',3);
  }
  function eye({svg,group:g,scene}){
    svg('path',{d:'M -44 0 Q 0 -35 44 0 Q 0 35 -44 0',fill:'#fff',stroke:'#b48b34','stroke-width':2},g);
    svg('circle',{cx:Math.max(-25,Math.min(25,scene.state[6]*4)),cy:0,r:11,fill:scene.seen>.5?'#b33f43':'#bd9235'},g);
    svg('circle',{cx:Math.max(-25,Math.min(25,scene.state[6]*4)),cy:0,r:4,fill:'#23394b'},g);
  }
  function spring({svg,group:g,node}){
    const stretch=node.stretch??0,length=50+Math.max(-12,Math.min(12,stretch*25));
    let d=`M ${-length} 0`;
    for(let i=0;i<9;i++)d+=` L ${-length+2*length*(i+1)/10} ${i%2?12:-12}`;
    d+=` L ${length} 0`;svg('path',{d,fill:'none',stroke:'#527d90','stroke-width':2.5},g);
  }
  function lever({svg,group:g,scene}){line(svg,g,-40,-22,0,-22,'#805898',4);line(svg,g,0,-22,35,25-scene.state[2]*30,'#805898',4);svg('circle',{cx:35,cy:25-scene.state[2]*30,r:6,fill:'#ac87bc'},g);}
  const renderers={receiver,wheels:wheel,wheel,battery,casing,eye,spring,lever,
    look:eye,vision:eye,cart:({svg,group:g,node})=>{
      const [position,momentum]=node.value,x=28*Math.tanh(position/5);
      line(svg,g,-65,22,65,22,'#cad6df',2);
      svg('rect',{x:x-22,y:-15,width:44,height:30,rx:4,fill:'#d4e5ef',stroke:'#47758f','stroke-width':2},g);
      line(svg,g,x,0,x+Math.max(-40,Math.min(40,momentum*30)),0,'#b07740',3);
      svg('text',{y:44,'text-anchor':'middle','font-size':13},g,'position '+position.toFixed(2));
    },tip:({svg,group:g,node})=>{const v=Array.isArray(node.value)?node.value[0]:node.value;line(svg,g,-35,30,35,30,'#c4b398',4);svg('circle',{cx:0,cy:12-(v+.4)*80,r:12,fill:'#6687a7'},g);},
    food:({svg,group:g,node})=>{svg('circle',{r:7+20*Math.sqrt(Math.max(0,node.value)/10),fill:'#d67c63'},g);},
    mouth:({svg,group:g,scene})=>{svg('path',{d:'M -35 -8 Q 0 35 35 -8',fill:'none',stroke:'#ae6e6e','stroke-width':5},g);const k=scene.casings.findIndex(c=>c.selected>0),c=scene.casings[k];
      svg('circle',{cy:0,r:5+Math.min(12,(c?.eat??0)*2),fill:'#c76a4e'},g);
      svg('text',{y:36,'text-anchor':'middle','font-size':12},g,c?'Casing '+(k+1):'Grip free');
      if(c)svg('text',{y:51,'text-anchor':'middle','font-size':10},g,'Open '+c.pry.toFixed(2)+' · Eat '+c.eat.toFixed(2));},
    red:({svg,group:g,node,scene})=>svg('circle',{r:18,fill:scene.casings[node.index].red>.5?'#c65346':'#a1abb4'},g)};
  root.FeedingAnimations=renderers;
})(globalThis);
