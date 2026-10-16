import {add,labels,results,trainingRun,forward,inputs,makeAxes,actout,unit,norm,dot,scale,sub,dist,mix,smooth,triplet} from './geometry.js?v=3d1';
import {camera,project,spherePoint,angleArc,rotateVector} from './scene3d.js?v=3d1';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const state={module:'scenarize',view:'train',step:0,token:0,playing:false,scenario:0,beta:.3,lr:.0035,selectedAxis:1,vector:[.58,.32,.46],orbit:false,overrides:{}};
let frames=trainingRun(), last=0, heroDragging=false, heroResumeAt=0, heroSpeed=0, lastHeroDraw=0, actoutFrame=frames.at(-1);
const mathCache=new Map();
function math(tex,display=false){const key=`${display}:${tex}`;if(!mathCache.has(key))mathCache.set(key,window.katex.renderToString(tex,{displayMode:display,throwOnError:true,output:'htmlAndMathml',strict:'ignore'}));return mathCache.get(key);}
function renderMath(root=document){root.querySelectorAll('[data-math]').forEach(el=>el.innerHTML=math(el.dataset.math,!!el.closest('.equation-strip,.equation-definition,.sampling-row')));}
const C={x:380,y:232,s:174};
const views={train:{yaw:.65,pitch:.38},act:{yaw:.65,pitch:.38},hero:{yaw:.65,pitch:.38}};
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
let viewCamera=camera();
const narrow=()=>matchMedia('(max-width:600px)').matches;
function activate(view,zoom){C.s=zoom;viewCamera=camera(views[view].yaw,views[view].pitch,zoom,C.x,C.y);}
function sizePlot(el){el.setAttribute('viewBox',narrow()?'100 0 560 470':'0 0 760 470');}
const xy=v=>project(v,viewCamera).slice(0,2);
const path=ps=>ps.map((p,i)=>`${i?'L':'M'}${p.map(x=>x.toFixed(2)).join(',')}`).join(' ');
function line(a,b,color='#cdd2c2',width=1,extra=''){return `<path d="${path([xy(a),xy(b)])}" fill="none" stroke="${color}" stroke-width="${width}" ${extra}/>`;}
function curve(points,color,width=1,extra=''){return `<path d="${path(points.map(xy))}" fill="none" stroke="${color}" stroke-width="${width}" ${extra}/>`;}
function circle(v,r,color,extra=''){const [x,y]=xy(v);return `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" ${extra}/>`;}
function ml(v,tex,w=150,offset=[10,-29]){const [x,y]=xy(v);return `<foreignObject x="${x+offset[0]}" y="${y+offset[1]}" width="${w}" height="42" class="plot-label"><div xmlns="http://www.w3.org/1999/xhtml" class="math-label">${math(tex)}</div></foreignObject>`;}
function diamond(v,color,r=4){const [x,y]=xy(v);return `<path d="M${x},${y-r}l${r},${r}l-${r},${r}l-${r},-${r}Z" fill="${color}" stroke="#fff" stroke-width="1"/>`;}
function base(id='plot',polar=false){
 let s=`<defs><marker id="${id}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="none" stroke="context-stroke" stroke-width="1.3"/></marker></defs>`;
 const wide=polar||id==='hero',extent=wide?1.12:.73,floor=wide?-.78:-.58,axis=wide?1.2:.82;
 for(let i=-4;i<=4;i++){
  const a=i*extent/4;
  s+=line([-extent,a,floor],[extent,a,floor],'#e9ede4',.7)+line([a,-extent,floor],[a,extent,floor],'#e9ede4',.7);
 }
 for(let i=0;i<3;i++){
  const a=[0,0,0],b=[0,0,0];a[i]=-axis;b[i]=axis;
  s+=line(a,[0,0,0],'#d4dbce',.9,'stroke-dasharray="3 5"')+line([0,0,0],b,'#c3cdb9',1,`marker-end="url(#${id}-arrow)"`);
  const label=scale(b,1.04);s+=ml(label,`x_{${i+1}}`,55,i===2?[8,-27]:[8,-7]);
 }
 if(polar){
  for(let ring=0;ring<3;ring++){
   const points=Array.from({length:81},(_,j)=>{const t=j/80*Math.PI*2,v=[0,0,0];v[(ring+1)%3]=Math.cos(t);v[(ring+2)%3]=Math.sin(t);return v;});
   s+=curve(points,'#d4dccb',.9,'stroke-dasharray="3 5"');
  }
 }
 return s;
}
function interpolatedFrame(step){const i=Math.floor(step),j=Math.min(160,i+1),t=smooth(step-i),a=frames[i],b=frames[j];return {...a,points:a.points.map((p,k)=>({...p,p:mix(p.p,b.points[k].p,t)})),centroids:a.centroids.map((p,k)=>mix(p,b.centroids[k],t)),chosen:a.chosen.map((vs,k)=>vs.map((p,l)=>mix(p,b.chosen[k][l],t)))};}
function focusPrompt(token=null){
 const parts=labels[state.scenario].prompt.split(' / ');
 $('#prompt-text').innerHTML='“'+parts.map((t,i)=>`<span class="${token===null?'':i===Math.min(3,Math.max(0,Math.floor(token)-1))?'token-highlight':i>=token?'prompt-pending':''}">${t}</span>`).join(' ')+'”';
}
function updateFlow(){
 const infer=state.view==='infer',t=Math.floor(state.token);
 $('#rnn-flow').innerHTML=`<span class="flow-block ${infer?'active':''}">${math('h_{p_t}^{(16)}')}</span><span>→</span><span class="flow-block ${infer?'active':''}">SMM</span><span>+</span><span class="flow-block ${infer?'active':''}">STM ${math('s_{t-1}')}</span><span>→</span><span class="flow-block ${infer?'current':''}">${math('\\tanh')}</span><span>→</span><span class="flow-block current">${math(infer?`s_{${t}}`:'s_{\\mathcal P}')}</span>`;
}
function drawTrain(){
 activate('train',285);
 const f=interpolatedFrame(state.step),c=state.scenario;
 let s=base('train');
 f.points.slice().sort((a,b)=>project(a.p,viewCamera)[2]-project(b.p,viewCamera)[2]).forEach(p=>s+=circle(p.p,p.label===c?3:2.2,p.label===c?'#c57647':'#aeb7a6',`opacity="${p.label===c?.45:.25}"`));
 f.centroids.forEach((p,i)=>s+=diamond(p,i===c?'#b98a6d':'#aeb7a6',i===c?4:3));
 let [a,p,n]=f.chosen[c];
 if(state.view==='train'){
  s+=line(a,p,'#7e9c6b',1.6,'stroke-dasharray="4 4"')+line(a,n,'#b9795d',1.3,'stroke-dasharray="4 4"');
  s+=circle(p,5,'#7e9c6b','stroke="#fff" stroke-width="2"')+circle(n,5,'#b9795d','stroke="#fff" stroke-width="2"');
  s+=ml(p,'s_{\\mathcal P^+}',80,[12,-33])+ml(n,'s_{\\mathcal P^-}',80,[-73,8]);
  s+=line([0,0,0],a,'#d47747',1.7,'marker-end="url(#train-arrow)"')+circle(a,15,'#d47747','opacity=".07"')+circle(a,6,'#d47747','stroke="#fff" stroke-width="2"')+ml(a,'s_{\\mathcal P}',80,[12,3]);
  const da=dist(a,p),dn=dist(a,n);$('#positive-distance').textContent=da.toFixed(3);$('#negative-distance').textContent=dn.toFixed(3);$('#loss-value').textContent=triplet(a,p,n,state.beta).toFixed(3);
  focusPrompt();
 }else{
  const states=forward(actoutFrame.model,inputs(c,0));
  const t=Math.floor(state.token),frac=smooth(state.token-t),current=mix(states[t],states[Math.min(4,t+1)],frac);
  const trail=states.slice(0,t+1);if(frac>0)trail.push(current);
  s+=`<path d="${path(trail.map(v=>xy(v)))}" fill="none" stroke="#d47747" stroke-width="2" stroke-linejoin="round"/>`;
  states.slice(0,t+1).forEach((v,i)=>{s+=circle(v,3,'#d47747',`opacity="${.25+i*.12}"`);if(i===0||i===t)s+=ml(v,`s_{${i}}`,55,[i%2?10:-42,i%2?-32:5]);});
  if(t>0){const prev=states[t-1],mapped=inputs(c,0)[t-1];const m=actoutFrame.model;const stm=m.U.map(row=>dot(row,prev)),smm=m.W.map(row=>dot(row,mapped));
   s+=line([0,0,0],stm,'#8fa071',1.2,'stroke-dasharray="4 4" marker-end="url(#train-arrow)"')+line(stm,add(stm,smm),'#7e98af',1.2,'stroke-dasharray="4 4" marker-end="url(#train-arrow)"');
  }
  s+=line([0,0,0],current,'#d47747',1.5,'marker-end="url(#train-arrow)"')+circle(current,17,'#d47747','opacity=".08"')+circle(current,6,'#d47747','stroke="#fff" stroke-width="2"');
  $('#positive-distance').textContent=current[0].toFixed(3);$('#negative-distance').textContent=current[1].toFixed(3);$('#third-coordinate').textContent=current[2].toFixed(3);$('#loss-value').textContent=norm(current).toFixed(3);focusPrompt(state.token);
 }
 $('#transfer').disabled=state.view==='infer'&&state.token<.001;
 sizePlot($('#training-space'));$('#training-space').innerHTML=s;
 const v=state.view==='train'?state.step:state.token;$('#progress').value=v;$('#step-value').textContent=state.view==='train'?Math.round(v).toString():Math.floor(v).toString();
}
function rebuild(){state.playing=false;state.step=0;state.token=0;frames=trainingRun(state.beta,state.lr);actoutFrame=frames.at(-1);state.overrides={};updatePlay();drawTrain();}
function updatePlay(){$('#play').textContent=state.playing?'Ⅱ':'▶';$('#play').setAttribute('aria-label',`${state.playing?'Pause':'Play'} ${state.view==='train'?'training':'token inference'}`);}
function setView(view){
 state.view=view;state.playing=false;updatePlay();$$('[data-view]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.view===view));
 const infer=view==='infer';if(infer)actoutFrame=frames.at(-1);$('#training-controls').hidden=infer;$('#rnn-flow').hidden=!infer;$('#third-coordinate-row').hidden=!infer;
 $('#control-title').textContent=infer?'Prompt inference':'Triplet training';
 $('#scenarize-legend').innerHTML=infer?'<span><i class="point-key"></i> Scenario estimate</span><span><i class="point-key similar"></i> STM contribution</span><span><i class="point-key smm"></i> SMM contribution</span><span><i class="centroid-key"></i> Class centroid</span>':'<span><i class="point-key"></i> Prompt embedding</span><span><i class="point-key similar"></i> Similar scenario</span><span><i class="point-key opposite"></i> Opposite scenario</span><span><i class="centroid-key"></i> Class centroid</span>';
 $('#control-desc').textContent=infer?'Each prompt chunk updates the scenario estimate through SMM and STM.':'The loss draws similar scenarios closer and separates opposite scenarios.';
 $('#positive-distance').previousElementSibling.textContent=infer?'First coordinate':'Similar distance²';$('#negative-distance').previousElementSibling.textContent=infer?'Second coordinate':'Opposite distance²';$('#loss-label').textContent=infer?'Vector norm':'Triplet loss';$('#next').setAttribute('aria-label',infer?'Next token step':'Next training step');
 $('#scenarize-equation').innerHTML=math(infer?'s_t=\\tanh(U_{\\mathrm{stm}}s_{t-1}+W_{\\mathrm{smm}}h_{p_t}^{(16)}+b)':'\\mathcal{L}=\\max\\!\\left(0,\\|s_{\\mathcal P}-s_{\\mathcal P^+}\\|^2-\\|s_{\\mathcal P}-s_{\\mathcal P^-}\\|^2+\\beta\\right)',true);
 $('#equation-source').textContent=infer?'Eq. 1':'Eq. 2';$('#progress').max=infer?4:160;$('#progress').step=infer?.01:1;$('#progress').setAttribute('aria-label',infer?'Prompt token step':'Training step');$('#progress-max').textContent=infer?'4':'160';state.step=infer?160:0;state.token=0;updateFlow();drawTrain();
}
function setModule(module){state.module=module;state.playing=false;state.orbit=false;updatePlay();$('#orbit').textContent='▶';$('#orbit').setAttribute('aria-label','Animate scenario embedding');$$('[data-module]').forEach(b=>{const active=b.dataset.module===module;b.setAttribute('aria-selected',active);b.tabIndex=active?0:-1;});$('#panel-scenarize').hidden=module!=='scenarize';$('#panel-actout').hidden=module!=='actout';if(module==='actout'){syncEndpoints();drawActout();}else drawTrain();}
function currentAxes(){return makeAxes(actoutFrame.centroids,state.overrides);}
const endpointText=v=>Math.abs(v*100-Math.round(v*100))<1e-8?v.toFixed(2):v.toFixed(3);
function syncEndpoints(){const a=currentAxes()[state.selectedAxis];$('#positive-endpoint').textContent=labels[a.a].name;$('#negative-endpoint').textContent=a.b<0?'Non-adversarial (mean)':labels[a.b].name;$('#tau-plus').value=a.plus;$('#tau-minus').value=a.minus;$('#tau-plus-value').textContent=endpointText(a.plus);$('#tau-minus-value').textContent=endpointText(a.minus);$('#axis-select').value=state.selectedAxis;}
function arc(a,b,r){return path(angleArc(a,b,r).map(xy));}
function drawActout(){
 activate('act',174);
 const axes=currentAxes(),out=actout(state.vector,axes),selected=out.axes[state.selectedAxis],u=unit(selected.vector),sn=unit(state.vector);
 let s=base('act',true);

 out.axes.forEach((a,i)=>{const v=unit(a.vector),focus=i===state.selectedAxis;s+=line(scale(v,-1.13),scale(v,1.13),focus?'#5e7699':'#cbd2c4',focus?1.7:.8,`opacity="${focus?1:.65}" marker-end="url(#act-arrow)"`);if(focus){s+=diamond(a.start,'#a5b09c')+diamond(a.end,'#a5b09c');s+=ml(scale(v,1.17),'\\hat{u}_i',70,[v[0]<0?-40:6,-19]);s+=ml(scale(v,-1.17),'-\\hat{u}_i',70,[v[0]>0?-58:5,-19]);}});
 const projection=scale(u,dot(sn,u));s+=line(sn,projection,'#b4bca3',1.2,'stroke-dasharray="4 4"')+circle(projection,3,'#5e7699');
 s+=`<path d="${arc(u,sn,.3)}" stroke="#5e7699" stroke-width="1.2" fill="none"/>`;
 const midpoint=angleArc(u,sn,.4)[18];s+=ml(midpoint,'\\theta_i',55,[-7,-16]);
 s+=line([0,0,0],sn,'#d47747',1.7,'marker-end="url(#act-arrow)"')+circle(sn,5,'#d47747','opacity=".18"');
 s+=circle(state.vector,22,'#d47747','opacity=".06"')+`<g class="drag-handle" id="embedding-handle" role="slider" tabindex="0" aria-label="3D scenario embedding. Left/right change azimuth; up/down change elevation." aria-valuemin="-180" aria-valuemax="180" aria-valuenow="${Math.round(Math.atan2(state.vector[1],state.vector[0])*180/Math.PI)}">${circle(state.vector,24,'transparent')}${circle(state.vector,7,'#d47747','stroke="white" stroke-width="2.5"')}</g>`+ml(state.vector,'s_{\\mathcal P}',75,[10,-30]);
 if(Math.abs(norm(state.vector)-1)>.12)s+=circle(sn,4,'white','stroke="#d47747" stroke-width="1.5"')+ml(sn,'\\hat{s}_{\\mathcal P}',80,[8,6]);
 sizePlot($('#actout-space'));$('#actout-space').innerHTML=s;
 $('#rho-value').textContent=(selected.rho>=0?'+':'')+selected.rho.toFixed(3);$('#angle-caption').textContent=`θ = ${(Math.acos(selected.rho)*180/Math.PI).toFixed(1)}°`;
 const temp=out.temperature;$('#temperature-value').textContent=temp===null?'—':temp.toFixed(3);
 $('#axis-contributions').innerHTML=out.axes.map((a,i)=>`<button class="contribution ${i===state.selectedAxis?'selected':''}" data-axis="${i}" title="Inspect ${a.name}; normalized weight ${(a.weight*100).toFixed(1)}%"><div><span>${a.name}</span><span>${(a.weight*100).toFixed(0)}%</span></div><div class="bar"><i style="width:${a.weight*100}%;background:${i===state.selectedAxis?'#5e7699':'#bdc8b1'}"></i></div></button>`).join('');
 $('#actout-note').hidden=state.selectedAxis!==2;
 $('#actout-note').textContent='Non-adversarial: mean of the other eight centroids. Its temperature (0.875) is a demo convention; Table 2 does not specify it.';
 const zs=[3.2,2.7,2.3,1.6,.9],names=['“the”','“a”','“this”','“one”','“our”'];const probs=zs.map(z=>Math.exp((z-3.2)/(temp||1)));const total=probs.reduce((a,b)=>a+b,0);
 $('#token-bars').innerHTML=probs.map((v,i)=>`<div class="token-bar"><output>${(v/total*100).toFixed(1)}%</output><i style="height:${v/total*92}%"></i><span>${names[i]}</span></div>`).join('');
}
function tick(now){
 const dt=Math.min(.08,last?(now-last)/1000:0);last=now;
 if(state.playing&&state.module==='scenarize'){
  if(state.view==='train'){state.step=Math.min(160,state.step+dt*12);if(state.step>=160)state.playing=false;}
  else {state.token=Math.min(4,state.token+dt*.65);const prev=Math.floor(state.token-dt*.65);if(Math.floor(state.token)!==prev)updateFlow();if(state.token>=4)state.playing=false;}
  drawTrain();updatePlay();
 }
 if(state.orbit&&state.module==='actout'){state.vector=rotateVector(state.vector,dt*.35,dt*.14*Math.cos(now/3500));drawActout();}
 const targetSpeed=(!reduced&&!heroDragging&&now>=heroResumeAt) ? .12 : 0;
 heroSpeed+=(targetSpeed-heroSpeed)*(1-Math.exp(-dt*3));
 if(!heroDragging&&heroSpeed>1e-5){
  views.hero.yaw+=heroSpeed*dt;
  if(now-lastHeroDraw>=40){drawHero();lastHeroDraw=now;}
 }
 requestAnimationFrame(tick);
}
function stopOrbit(){state.orbit=false;$('#orbit').textContent='▶';$('#orbit').setAttribute('aria-label','Animate scenario embedding');}
function drawHero(){
 activate('hero',190);
 const centroids=frames.at(-1).centroids.map(p=>scale(p,2.6));
 let s=base('hero');
 for(const [a,b,color] of [[0,1,'#95a9bc'],[2,3,'#a3b397']])s+=line(centroids[b],centroids[a],color,1.4,'marker-end="url(#hero-arrow)"');
 const focus=[0,1,2,3];
 focus.forEach((c,j)=>{
  frames.at(-1).points.filter(p=>p.label===c).forEach(p=>s+=circle(scale(p.p,2.6),2.8,labels[c].color,'opacity=".28"'));
  s+=diamond(centroids[c],labels[c].color,4);
  const q=xy(centroids[c]),label=['rigorous','brainstorming','institutional','casual'][j];
  const offsets=[[-105,-30],[12,31],[-130,30],[25,30]][j];
  s+=`<text x="${q[0]+offsets[0]}" y="${q[1]+offsets[1]}" class="svg-text">${label}</text>`;
 });
 const embedding=mix(centroids[3],centroids[6],.3);
 s+=line([0,0,0],embedding,'#c57647',1.9,'marker-end="url(#hero-arrow)"')+circle(embedding,15,'#c57647','opacity=".07"')+circle(embedding,6,'#c57647','stroke="#fff" stroke-width="2"')+ml(embedding,'s_{\\mathcal P}',90,[12,-33]);
 $('#hero-space').innerHTML=s;
}
function bindScene(svg,view){
 let gesture=null;
 const repaint=()=>view==='hero'?drawHero():view==='train'?drawTrain():drawActout();
 svg.addEventListener('pointerdown',e=>{
  const embedding=view==='act'&&!!e.target.closest('#embedding-handle');
  const c=camera(views[view].yaw,views[view].pitch,174);
  gesture={x:e.clientX,y:e.clientY,yaw:views[view].yaw,pitch:views[view].pitch,embedding,hemisphere:dot(state.vector,c.depth)>=0?1:-1,radius:Math.max(.12,norm(state.vector))};
  if(view==='hero'){heroDragging=true;heroSpeed=0;}if(embedding)stopOrbit();
  svg.setPointerCapture(e.pointerId);svg.classList.add('rotating');e.preventDefault();
 });
 svg.addEventListener('pointermove',e=>{
  if(!gesture)return;
  if(gesture.embedding){
   const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*(narrow()?560:760)+(narrow()?100:0),y=(e.clientY-r.top)/r.height*470;
   state.vector=spherePoint(x,y,gesture.radius,camera(views.act.yaw,views.act.pitch,174),gesture.hemisphere);
  }else{
   views[view].yaw=gesture.yaw+(e.clientX-gesture.x)*.008;
   views[view].pitch=Math.max(-1.15,Math.min(1.15,gesture.pitch+(e.clientY-gesture.y)*.006));
  }
  repaint();
 });
 const finish=()=>{if(!gesture)return;gesture=null;if(view==='hero'){heroDragging=false;heroResumeAt=performance.now()+2000;}svg.classList.remove('rotating');};
 svg.addEventListener('pointerup',finish);svg.addEventListener('pointercancel',finish);svg.addEventListener('lostpointercapture',finish);
 svg.addEventListener('keydown',e=>{
  if(!['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key))return;
  e.preventDefault();
  if(view==='act'&&e.target.closest('#embedding-handle')){
   stopOrbit();state.vector=rotateVector(state.vector,e.key==='ArrowLeft'?-.06:e.key==='ArrowRight'?.06:0,e.key==='ArrowUp'?.06:e.key==='ArrowDown'?-.06:0);drawActout();$('#embedding-handle').focus();
  }else{
   if(view==='hero'){heroSpeed=0;heroResumeAt=performance.now()+2000;}
   views[view].yaw+=e.key==='ArrowLeft'?-.08:e.key==='ArrowRight'?.08:0;
   views[view].pitch=Math.max(-1.15,Math.min(1.15,views[view].pitch+(e.key==='ArrowUp'?.06:e.key==='ArrowDown'?-.06:0)));repaint();
  }
 });
}
function bind(){
 $$('[data-module]').forEach(b=>{b.addEventListener('click',()=>setModule(b.dataset.module));b.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const module=['ArrowRight','End'].includes(e.key)?'actout':'scenarize';setModule(module);$(`[data-module="${module}"]`).focus();}});});
 $$('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
 $('#play').addEventListener('click',()=>{if(state.view==='train'&&state.step>=160)state.step=0;if(state.view==='infer'&&state.token>=4)state.token=0;state.playing=!state.playing;updatePlay();updateFlow();});
 $('#next').addEventListener('click',()=>{state.playing=false;state.view==='train'?state.step=Math.min(160,Math.floor(state.step)+1):state.token=Math.min(4,Math.floor(state.token)+1);updatePlay();updateFlow();drawTrain();});
 $('#progress').addEventListener('input',e=>{state.playing=false;state.view==='train'?state.step=+e.target.value:state.token=+e.target.value;updatePlay();updateFlow();drawTrain();});
 $('#scenario').addEventListener('change',e=>{state.scenario=+e.target.value;drawTrain();});
 $('#margin').addEventListener('input',e=>{state.beta=+e.target.value;$('#margin-value').textContent=state.beta.toFixed(2);rebuild();});
 $('#reset').addEventListener('click',()=>{views.train={yaw:.65,pitch:.38};views.act={yaw:.65,pitch:.38};if(state.module==='actout'){stopOrbit();state.vector=[.58,.32,.46];state.overrides={};syncEndpoints();drawActout();}else {state.playing=false;state.step=state.view==='infer'?160:0;state.token=0;updateFlow();updatePlay();drawTrain();}});
 $('#transfer').addEventListener('click',()=>{actoutFrame=state.view==='train'?frames[Math.round(state.step)]:frames.at(-1);const p=state.view==='train'?actoutFrame.chosen[state.scenario][0]:forward(actoutFrame.model,inputs(state.scenario,0))[Math.floor(state.token)];state.vector=norm(p)>1e-6?[...p]:[.58,.32,.46];state.overrides={};setModule('actout');});
 $('#axis-select').addEventListener('change',e=>{state.selectedAxis=+e.target.value;syncEndpoints();drawActout();});
 $('#axis-contributions').addEventListener('click',e=>{const b=e.target.closest('[data-axis]');if(b){state.selectedAxis=+b.dataset.axis;syncEndpoints();drawActout();}});
 for(const [id,key] of [['tau-plus','plus'],['tau-minus','minus']])$('#'+id).addEventListener('input',e=>{state.overrides[state.selectedAxis]={...state.overrides[state.selectedAxis],[key]:+e.target.value};$('#'+id+'-value').textContent=endpointText(+e.target.value);drawActout();});
 $('#restore-endpoints').addEventListener('click',()=>{state.overrides={};syncEndpoints();drawActout();});
 $('#orbit').addEventListener('click',()=>{state.orbit=!state.orbit;$('#orbit').textContent=state.orbit?'Ⅱ':'▶';$('#orbit').setAttribute('aria-label',state.orbit?'Pause scenario animation':'Animate scenario embedding');});
 bindScene($('#training-space'),'train');bindScene($('#actout-space'),'act');bindScene($('#hero-space'),'hero');
 $('#copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('#bibtex').textContent);$('#copy-status').textContent='BibTeX copied.';}catch{$('.citation-fallback').open=true;$('#copy-status').textContent='Select and copy the BibTeX below.';}});
}
function init(){
 renderMath();$('#scenario').innerHTML=labels.map((l,i)=>`<option value="${i}">${l.name}</option>`).join('');$('#axis-select').innerHTML=makeAxes(actoutFrame.centroids).map((a,i)=>`<option value="${i}" ${i===1?'selected':''}>${a.name}</option>`).join('');
 $('#results-table').innerHTML=[results[0],results[1],results[3],results[6],results[7]].map((r,i)=>`<tr class="${i===0?'ours':''}"><th scope="row">${r.name}</th><td>${r.as.toFixed(4)}</td><td>${r.ae.toFixed(2)}</td><td>${r.mt.toFixed(2)}</td></tr>`).join('');
 addEventListener('resize',()=>{state.module==='actout'?drawActout():drawTrain();drawHero();});
 bind();updateFlow();drawTrain();drawHero();requestAnimationFrame(tick);
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
