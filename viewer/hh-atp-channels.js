/* Illustrated-view renderers and the population view share these gate readouts.
 * Each drawn pore represents one equal share of the modeled population. */
(function (root) {
  'use strict';
  const species = {
    sodium: {label:'Sodium', ion:'Na⁺', color:'#007c83', pale:'#a3d5ce', formula:'m³h'},
    potassium: {label:'Potassium', ion:'K⁺', color:'#99520d', pale:'#efd09c', formula:'n⁴'}
  };
  const ns = 'http://www.w3.org/2000/svg';
  function element(tag, attributes, parent, text) {
    const node = document.createElementNS(ns, tag);
    for (const [k,v] of Object.entries(attributes)) node.setAttribute(k,v);
    if (text !== undefined) node.textContent = text;
    parent.append(node);
    return node;
  }
  function fraction(name, q) { return name==='sodium' ? q[4]**3*q[5] : q[6]**4; }
  function pore(parent, x, y, isOpen, colors, scale=1) {
    const g=element('g',{transform:'translate('+x+' '+y+') scale('+scale+')','data-open':String(isOpen)},parent);
    for(const direction of [-1,1]){
      const side=element('g',{transform:'scale('+direction+' 1)'},g);
      element('path',{d:'M 7 -10 Q 4 -10 4 -6 L 4 5 Q 3 9 7 10 L 9 10 L 9 -10 Z',
        fill:isOpen?colors.color:'#89a0a8'},side);
      element('path',{d:isOpen?'M 5 4 L 4 -2':'M 5 4 L 0 1',stroke:isOpen?colors.color:'#344e5a',
        'stroke-linecap':'round','stroke-width':2.1},side);
    }
    if(isOpen) element('circle',{cx:0,cy:0,r:1.65,fill:colors.color},g);
  }
  function population(container, name, q) {
    const c=species[name], count=24, probability=fraction(name,q);
    const open=Math.round(probability*count);
    container.setAttribute('aria-label',c.label+' channels, '+(100*probability).toFixed(2)+' percent open');
    if(container.dataset.count===String(open))return;
    container.dataset.count=String(open);
    container.replaceChildren();
    element('rect',{x:0,y:11,width:600,height:18,rx:4,fill:'#e8eeed'},container);
    for(const y of [12,28]) element('line',{x1:0,x2:600,y1:y,y2:y,stroke:'#bed1cf','stroke-width':1.5},container);
    for(let i=0;i<count;i++)pore(container,12.5+25*i,20,(i*7)%count<open,c);
  }
  function channelIcon(name) {
    return ({group,scene})=>{
      const c=species[name], p=fraction(name,scene.configuration);
      element('rect',{x:-60,y:-6,width:120,height:12,rx:3,fill:'#e4edec'},group);
      for(let i=0;i<12;i++)pore(group,-55+10*i,0,(i*5)%12<Math.round(p*12),c,.49);
      element('text',{x:0,y:24,'text-anchor':'middle',fill:c.color,'font-size':12},group,(100*p).toFixed(1)+'% open');
    };
  }
  function gateIcon(index) {
    return ({group,scene})=>{
      const x=scene.configuration[index], label=['m','h','n'][index-4];
      element('rect',{x:-43,y:-5,width:86,height:10,rx:5,fill:'#d7e5e6'},group);
      element('rect',{x:-43,y:-5,width:86*x,height:10,rx:5,fill:'#168086'},group);
      element('text',{x:0,y:23,'text-anchor':'middle',fill:'#254950','font-size':13},group,label+' = '+x.toFixed(3));
    };
  }
  root.MembraneChannels={fraction,population,renderers:{
    'na-channel':channelIcon('sodium'),'na-pore':channelIcon('sodium'),
    'k-channel':channelIcon('potassium'),'k-pore':channelIcon('potassium'),
    'gate-m':gateIcon(4),'gate-h':gateIcon(5),'gate-n':gateIcon(6)
  }};
})(globalThis);
