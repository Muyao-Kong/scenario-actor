import {labels,results,trainingRun,forward,inputs,makeAxes,actout,unit,norm,dot,scale,sub,dist,mix,smooth,triplet} from '../../geometry.js';
const $=s=>document.querySelector(s),$$=s=>[...document.querySelectorAll(s)];
const reduced=matchMedia('(prefers-reduced-motion: reduce)').matches;
const state={module:'scenarize',view:'train',step:0,token:0,playing:false,speed:1,scenario:0,beta:.3,lr:.0035,selectedAxis:1,vector:[.65,.38],orbit:false,showAxes:true,overrides:{}};
let frames=trainingRun(), last=0, clock=0, heroTime=0, heroLast=0, drag=false, actoutFrame=frames.at(-1);
const mathCache=new Map();
function math(tex){if(!mathCache.has(tex))mathCache.set(tex,window.katex.renderToString(tex,{throwOnError:true,output:'htmlAndMathml',strict:'ignore'}));return mathCache.get(tex);}
function renderMath(root=document){root.querySelectorAll('[data-math]').forEach(el=>el.innerHTML=math(el.dataset.math));}
const C={x:380,y:232,s:174};
const narrow=()=>matchMedia('(max-width:600px)').matches;
function sizePlot(el){el.setAttribute('viewBox',narrow()?'100 0 560 470':'0 0 760 470');}
const xy=(v,c=C)=>[c.x+v[0]*c.s,c.y-v[1]*c.s];
const path=ps=>ps.map((p,i)=>`${i?'L':'M'}${p.map(x=>x.toFixed(2)).join(',')}`).join(' ');
function line(a,b,color='#cdd2c2',width=1,extra=''){return `<path d="${path([xy(a),xy(b)])}" fill="none" stroke="${color}" stroke-width="${width}" ${extra}/>`;}
function circle(v,r,color,extra=''){const [x,y]=xy(v);return `<circle cx="${x}" cy="${y}" r="${r}" fill="${color}" ${extra}/>`;}
function textAt(v,str,extra=''){const [x,y]=xy(v);return `<text x="${x}" y="${y}" class="svg-text" ${extra}>${str}</text>`;}
function ml(v,tex,w=150,offset=[10,-29]){const [x,y]=xy(v);return `<foreignObject x="${x+offset[0]}" y="${y+offset[1]}" width="${w}" height="38" class="plot-label"><div xmlns="http://www.w3.org/1999/xhtml" class="math-label">${math(tex)}</div></foreignObject>`;}
function diamond(v,color,r=4){const [x,y]=xy(v);return `<path d="M${x},${y-r}l${r},${r}l-${r},${r}l-${r},-${r}Z" fill="${color}" stroke="#fff" stroke-width="1"/>`;}
function base(id='plot',polar=false){
 let s=`<defs><pattern id="${id}-grid" width="43.5" height="43.5" patternUnits="userSpaceOnUse" x="32" y="14.5"><path d="M43.5 0H0V43.5" fill="none" stroke="#eaede5" stroke-width=".7"/></pattern><marker id="${id}-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="none" stroke="context-stroke" stroke-width="1.3"/></marker></defs><rect x="32" y="15" width="696" height="408" fill="url(#${id}-grid)"/>`;
 const xr=(narrow()?235:322)/C.s,yr=192/C.s;
 s+=line([-xr,0],[xr,0],'#cbd0c1',.8,`marker-end="url(#${id}-arrow)"`)+line([0,-yr],[0,yr],'#cbd0c1',.8,`marker-end="url(#${id}-arrow)"`);
 s+=ml([xr-.08,-.06],'s_1',40,[0,0])+ml([.03,yr-.02],'s_2',40,[0,-25])+textAt([-.08,-.1],'0','font-size="9"');
 if(polar)s+=circle([0,0],C.s,'none','stroke="#d8ddcc" stroke-width="1" stroke-dasharray="3 5"');
 return s;
}
function interpolatedFrame(step){const i=Math.floor(step),j=Math.min(160,i+1),t=smooth(step-i),a=frames[i],b=frames[j];return {...a,points:a.points.map((p,k)=>({...p,p:mix(p.p,b.points[k].p,t)})),centroids:a.centroids.map((p,k)=>mix(p,b.centroids[k],t)),chosen:a.chosen.map((vs,k)=>vs.map((p,l)=>mix(p,b.chosen[k][l],t)))};}
function focusPrompt(token=null){
 const parts=labels[state.scenario].prompt.split(' / ');
 $('#prompt-text').innerHTML='“'+parts.map((t,i)=>`<span class="${token===null?'':i===Math.min(3,Math.max(0,Math.floor(token)-1))?'token-highlight':i>=token?'prompt-pending':''}">${t}</span>`).join(' ')+'”';
}
function updateFlow(){
 const infer=state.view==='infer',t=Math.floor(state.token);
 $('#rnn-flow').innerHTML=`<span class="svg-ref">FIG. 2</span><span class="flow-block ${infer?'active':''}">${math('h_{p_t}^{(16)}')}</span><span>→</span><span class="flow-block ${infer?'active':''}">SMM</span><span>+</span><span class="flow-block ${infer?'active':''}">STM ${math('s_{t-1}')}</span><span>→</span><span class="flow-block ${infer?'current':''}">${math('\\tanh')}</span><span>→</span><span class="flow-block current">${math(infer?`s_{${t}}`:'s_{\\mathcal P}')}</span>`;
}
function drawTrain(){
 C.s=285;
 const f=interpolatedFrame(state.step),c=state.scenario;
 let s=base('train');
 s+=`<text x="${narrow()?125:54}" y="39" class="svg-small">SCENARIO SPACE</text><text x="${narrow()?635:705}" y="39" class="svg-small" text-anchor="end">${state.view==='train'?'TRAINING STEP '+Math.round(state.step).toString().padStart(3,'0'):'TOKEN '+Math.floor(state.token)+' / 4'}</text>`;
 const axes=makeAxes(f.centroids);
 if(state.step>4)axes.forEach(a=>s+=line(a.start,a.end,a.color,.8,`opacity=".2" marker-end="url(#train-arrow)"`));
 f.points.forEach(p=>s+=circle(p.p,p.label===c?3:2.2,labels[p.label].color,`opacity="${p.label===c?.55:.22}"`));
 f.centroids.forEach((p,i)=>s+=diamond(p,labels[i].color,i===c?5:3.5));
 let [a,p,n]=f.chosen[c];
 if(state.view==='train'){
  s+=line(a,p,'#7e9c6b',1.6,'stroke-dasharray="4 4"')+line(a,n,'#b9795d',1.3,'stroke-dasharray="4 4"');
  s+=circle(p,5,'#7e9c6b','stroke="#fff" stroke-width="2"')+circle(n,5,'#b9795d','stroke="#fff" stroke-width="2"');
  s+=ml(p,'s_{\\mathcal P^+}',80,[12,-33])+ml(n,'s_{\\mathcal P^-}',80,[-73,8]);
  s+=line([0,0],a,'#d47747',1.7,'marker-end="url(#train-arrow)"')+circle(a,15,'#d47747','opacity=".07"')+circle(a,6,'#d47747','stroke="#fff" stroke-width="2"')+ml(a,'s_{\\mathcal P}',80,[12,3]);
  const da=dist(a,p),dn=dist(a,n);$('#positive-distance').textContent=da.toFixed(3);$('#negative-distance').textContent=dn.toFixed(3);$('#loss-value').textContent=triplet(a,p,n,state.beta).toFixed(3);
  s+=`<text x="${narrow()?125:54}" y="387" class="svg-small" fill="#7e9c6b">● similar</text><text x="${narrow()?275:143}" y="387" class="svg-small" fill="#b9795d">● opposite</text>`;
  focusPrompt();
 }else{
  const states=forward(actoutFrame.model,inputs(c,0));
  const t=Math.floor(state.token),frac=smooth(state.token-t),current=mix(states[t],states[Math.min(4,t+1)],frac);
  const trail=states.slice(0,t+1);if(frac>0)trail.push(current);
  s+=`<path d="${path(trail.map(v=>xy(v)))}" fill="none" stroke="#d47747" stroke-width="2" stroke-linejoin="round"/>`;
  states.slice(0,t+1).forEach((v,i)=>{s+=circle(v,3,'#d47747',`opacity="${.25+i*.12}"`);s+=ml(v,`s_{${i}}`,45,[i%2?10:-35,i%2?-29:5]);});
  if(t>0){const prev=states[t-1],mapped=inputs(c,0)[t-1];const m=actoutFrame.model;const stm=m.U.map(row=>dot(row,prev)),smm=m.W.map(row=>dot(row,mapped));
   s+=line([0,0],stm,'#8fa071',1.2,'stroke-dasharray="4 4" marker-end="url(#train-arrow)"')+line(stm,[stm[0]+smm[0],stm[1]+smm[1]],'#7e98af',1.2,'stroke-dasharray="4 4" marker-end="url(#train-arrow)"');
  }
  s+=line([0,0],current,'#d47747',1.5,'marker-end="url(#train-arrow)"')+circle(current,17,'#d47747','opacity=".08"')+circle(current,6,'#d47747','stroke="#fff" stroke-width="2"');
  s+=`<text x="${narrow()?125:54}" y="387" class="svg-small">● STM contribution</text><text x="${narrow()?335:211}" y="387" class="svg-small">● SMM contribution</text>`;
  $('#positive-distance').textContent=current[0].toFixed(3);$('#negative-distance').textContent=current[1].toFixed(3);$('#loss-value').textContent=norm(current).toFixed(3);focusPrompt(state.token);
 }
 sizePlot($('#training-space'));$('#training-space').innerHTML=s;
 const v=state.view==='train'?state.step:state.token;$('#progress').value=v;$('#step-value').textContent=state.view==='train'?Math.round(v).toString().padStart(3,'0'):Math.floor(v).toString().padStart(2,'0');
 const lossPoints=frames.slice(0,Math.floor(state.step)+1).map((f,i)=>[i/160*250,41-f.loss/Math.max(state.beta,.001)*35]);
 $('#loss-chart').innerHTML=`<path d="M0 42H250" stroke="#e0e4d6"/><path d="${path(lossPoints)}" fill="none" stroke="#8e9e6c" stroke-width="1.4"/>`;
}
function rebuild(){state.playing=false;state.step=0;state.token=0;frames=trainingRun(state.beta,state.lr);actoutFrame=frames.at(-1);state.overrides={};updatePlay();drawTrain();}
function updatePlay(){$('#play').textContent=state.playing?'Ⅱ':'▶';$('#play').setAttribute('aria-label',`${state.playing?'Pause':'Play'} ${state.view==='train'?'training':'token inference'}`);}
function setView(view){
 state.view=view;state.playing=false;updatePlay();$$('[data-view]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.view===view));
 const infer=view==='infer';if(infer)actoutFrame=frames.at(-1);$('#training-controls').hidden=infer;$('#inference-controls').hidden=!infer;
 $('#figure-ref').textContent=infer?'FIG. 2 · EQ. 1':'EQ. 2 · TRAINING';$('#control-title').textContent=infer?'Read the prompt, one step at a time.':'Learn a scenario geometry.';
 $('#control-desc').innerHTML=infer?'New context updates the estimate.<br>The final state represents the scenario.':'Similar scenarios move closer.<br>Opposite scenarios move apart.';
 $('#positive-distance').previousElementSibling.textContent=infer?'EMBEDDING s₁':'SIMILAR DISTANCE²';$('#negative-distance').previousElementSibling.textContent=infer?'EMBEDDING s₂':'OPPOSITE DISTANCE²';$('#loss-label').textContent=infer?'VECTOR NORM':'TRIPLET LOSS';$('#loss-chart').toggleAttribute('hidden',infer);$('#next').setAttribute('aria-label',infer?'Next token step':'Next training step');
 $('#model-note').textContent=infer?'Four prompt chunks stand in for tokens. Uses the trained toy RNN; the paper processes individual tokens.':'Real updates to a tiny RNN on synthetic inputs. Paper: 32D, AdamW, η = 3 × 10⁻⁴.';
 $('#scenarize-equation').innerHTML=math(infer?'s_t=\\tanh(U_{\\mathrm{stm}}s_{t-1}+W_{\\mathrm{smm}}h_{p_t}^{(16)}+b)':'\\mathcal{L}=\\max\\!\\left(0,\\|s_{\\mathcal P}-s_{\\mathcal P^+}\\|^2-\\|s_{\\mathcal P}-s_{\\mathcal P^-}\\|^2+\\beta\\right)');
 $('#equation-source').textContent=infer?'PAPER / EQ. 1':'PAPER / EQ. 2';$('#progress').max=infer?4:160;$('#progress').step=infer?.01:1;$('#progress').setAttribute('aria-label',infer?'Prompt token step':'Training step');$('#progress-max').textContent=infer?'4':'160';state.step=infer?160:0;state.token=0;updateFlow();drawTrain();
}
function setModule(module){state.module=module;state.playing=false;state.orbit=false;updatePlay();$('#orbit').textContent='▶';$('#orbit').setAttribute('aria-label','Animate scenario embedding');$$('[data-module]').forEach(b=>{const active=b.dataset.module===module;b.setAttribute('aria-selected',active);b.tabIndex=active?0:-1;});$('#panel-scenarize').hidden=module!=='scenarize';$('#panel-actout').hidden=module!=='actout';$('#module-tag').textContent=module==='scenarize'?'LEARN THE SPACE':'ACT ON ITS GEOMETRY';if(module==='actout'){syncEndpoints();drawActout();}else drawTrain();}
function currentAxes(){return makeAxes(actoutFrame.centroids,state.overrides);}
function syncEndpoints(){const a=currentAxes()[state.selectedAxis];$('#positive-endpoint').textContent=labels[a.a].name;$('#negative-endpoint').textContent=a.b<0?'Non-adversarial (mean)':labels[a.b].name;$('#tau-plus').value=a.plus;$('#tau-minus').value=a.minus;$('#tau-plus-value').textContent=a.plus.toFixed(2);$('#tau-minus-value').textContent=a.minus.toFixed(2);$('#axis-select').value=state.selectedAxis;}
function arc(a,b,r){const aa=Math.atan2(a[1],a[0]),bb=Math.atan2(b[1],b[0]);let d=bb-aa;while(d>Math.PI)d-=2*Math.PI;while(d<-Math.PI)d+=2*Math.PI;return path(Array.from({length:35},(_,i)=>xy([Math.cos(aa+d*i/34)*r,Math.sin(aa+d*i/34)*r])));}
function drawActout(){
 C.s=174;
 const axes=currentAxes(),out=actout(state.vector,axes),selected=out.axes[state.selectedAxis],u=unit(selected.vector),sn=unit(state.vector);
 let s=base('act',true);
 s+=`<text x="${narrow()?125:54}" y="39" class="svg-small">SCENARIO SPACE · NORMALIZED DIRECTIONS</text>`;
 actoutFrame.points.forEach(p=>s+=circle(p.p,2,labels[p.label].color,'opacity=".13"'));
 out.axes.forEach((a,i)=>{if(!state.showAxes&&i!==state.selectedAxis)return;const v=unit(a.vector),focus=i===state.selectedAxis;s+=line(scale(v,-1.13),scale(v,1.13),a.color,focus?1.7:.8,`opacity="${focus?1:.28}" marker-end="url(#act-arrow)"`);if(focus){s+=diamond(a.start,a.color)+diamond(a.end,a.color)+line(a.start,a.end,a.color,.9,'stroke-dasharray="3 4" opacity=".65"');s+=ml(scale(v,1.17),'\\hat{u}_i',70,[v[0]<0?-40:6,-19]);s+=ml(scale(v,-1.17),'-\\hat{u}_i',70,[v[0]>0?-58:5,-19]);}});
 const projection=scale(u,dot(sn,u));s+=line(sn,projection,'#b4bca3',1.2,'stroke-dasharray="4 4"')+circle(projection,3,selected.color);
 s+=`<path d="${arc(u,sn,.3)}" stroke="${selected.color}" stroke-width="1.2" fill="none"/>`;
 const avg=unit([u[0]+sn[0],u[1]+sn[1]]);s+=ml(scale(avg,.4),'\\theta_i',55,[-7,-16]);
 s+=line([0,0],sn,'#d47747',1.7,'marker-end="url(#act-arrow)"')+circle(sn,5,'#d47747','opacity=".18"');
 s+=circle(state.vector,22,'#d47747','opacity=".06"')+`<g class="drag-handle" id="embedding-handle" role="slider" tabindex="0" aria-label="Scenario embedding direction. Use arrow keys to rotate." aria-valuemin="-180" aria-valuemax="180" aria-valuenow="${Math.round(Math.atan2(state.vector[1],state.vector[0])*180/Math.PI)}">${circle(state.vector,14,'transparent')}${circle(state.vector,7,'#d47747','stroke="white" stroke-width="2.5"')}</g>`+ml(state.vector,'s_{\\mathcal P}',75,[10,-30]);
 if(Math.abs(norm(state.vector)-1)>.12)s+=circle(sn,4,'white','stroke="#d47747" stroke-width="1.5"')+ml(sn,'\\hat{s}_{\\mathcal P}',80,[8,6]);
 s+=`<text x="${narrow()?125:54}" y="386" class="svg-small">◆ class centroid</text><text x="${narrow()?330:190}" y="386" class="svg-small">dashed line = centroid difference</text>`;
 sizePlot($('#actout-space'));$('#actout-space').innerHTML=s;
 $('#rho-value').textContent=(selected.rho>=0?'+':'')+selected.rho.toFixed(3);$('#angle-caption').textContent=`θ = ${(Math.acos(selected.rho)*180/Math.PI).toFixed(1)}°`;
 const temp=out.temperature;$('#temperature-value').textContent=temp===null?'—':temp.toFixed(3);$('#temperature-marker').style.left=temp===null?'50%':`${(temp-.3)/1.2*100}%`;
 const angle=Math.atan2(state.vector[1],state.vector[0])*180/Math.PI;$('#angle').value=angle;$('#angle-value').textContent=Math.round(angle)+'°';
 $('#axis-contributions').innerHTML=out.axes.map((a,i)=>`<button class="contribution ${i===state.selectedAxis?'selected':''}" data-axis="${i}" title="Inspect ${a.name}; normalized weight ${(a.weight*100).toFixed(1)}%"><div><span>${a.name}</span><span>${(a.weight*100).toFixed(0)}%</span></div><div class="bar"><i style="width:${a.weight*100}%;background:${a.color}"></i></div><output>ρ ${a.rho>=0?'+':''}${a.rho.toFixed(2)}</output></button>`).join('');
 $('#actout-note').textContent=state.selectedAxis===2?'Non-adversarial centroid = mean of the other eight centroids. Its temperature uses their mean (0.875), a demo convention; Table 2 does not specify it.':'Axis directions are centroid differences. Dashed projection gives signed cosine similarity; |ρ| gives the axis weight.';
 const zs=[3.2,2.7,2.3,1.6,.9],names=['“the”','“a”','“this”','“one”','“our”'];const probs=zs.map(z=>Math.exp((z-3.2)/(temp||1)));const total=probs.reduce((a,b)=>a+b,0);
 $('#token-bars').innerHTML=probs.map((v,i)=>`<div class="token-bar"><output>${(v/total*100).toFixed(1)}%</output><i style="height:${v/total*92}%"></i><span>${names[i]}</span></div>`).join('');
}
function drawHero(){
 const c={x:270,y:205,s:145},hxy=p=>xy(p,c);const hline=(a,b,color,w=1,d='')=>`<path d="${path([hxy(a),hxy(b)])}" fill="none" stroke="${color}" stroke-width="${w}" ${d}/>`;
 let s=`<defs><pattern id="hero-grid" width="36.25" height="36.25" patternUnits="userSpaceOnUse" x="16.25" y="23.75"><path d="M36.25 0H0V36.25" fill="none" stroke="#e6e9df" stroke-width=".8"/></pattern><marker id="hero-arrow" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="5" markerHeight="5" orient="auto-start-reverse"><path d="M0 0L10 5L0 10" fill="none" stroke="context-stroke" stroke-width="1.5"/></marker></defs><rect x="16" y="24" width="522" height="326" fill="url(#hero-grid)"/>`;
 s+=hline([-1.65,0],[1.7,0],'#c9cebe',.8,'marker-end="url(#hero-arrow)"')+hline([0,-.96],[0,1.19],'#c9cebe',.8,'marker-end="url(#hero-arrow)"');
 const clusters=[{p:[-.92,.5],l:'rigorous',color:'#6682a5'},{p:[.95,-.44],l:'brainstorming',color:'#bda46d'},{p:[-.67,-.58],l:'institutional',color:'#87a098'},{p:[.87,.62],l:'casual',color:'#9ea178'}];
 clusters.forEach((cl,i)=>{for(let j=0;j<14;j++){const p=[cl.p[0]+Math.sin(j*7+i)*.14,cl.p[1]+Math.cos(j*11+i)*.13],q=hxy(p);s+=`<circle cx="${q[0]}" cy="${q[1]}" r="2.2" fill="${cl.color}" opacity=".3"/>`;}const q=hxy(cl.p);s+=`<path d="M${q[0]},${q[1]-4}l4,4l-4,4l-4,-4Z" fill="${cl.color}"/><text x="${q[0]+(i%2?12:-70)}" y="${q[1]+(i<2?-21:28)}" class="svg-text" style="font-size:10px">${cl.l}</text>`;});
 s+=hline(clusters[1].p,clusters[0].p,'#9aafbc',1.1,'marker-end="url(#hero-arrow)"')+hline(clusters[2].p,clusters[3].p,'#a3b297',1.1,'marker-end="url(#hero-arrow)"');
 const a=.68+Math.sin(heroTime*.25)*.4,v=[Math.cos(a)*.84,Math.sin(a)*.84],q=hxy(v);
 s+=`<ellipse cx="270" cy="205" rx="122" ry="122" fill="none" stroke="#cfd6c1" stroke-width=".8" stroke-dasharray="3 5"/>`+hline([0,0],v,'#d47747',1.6,'marker-end="url(#hero-arrow)"')+`<circle cx="${q[0]}" cy="${q[1]}" r="17" fill="#d47747" opacity=".07"/><circle cx="${q[0]}" cy="${q[1]}" r="5.5" fill="#d47747" stroke="#fff" stroke-width="2"/><foreignObject x="${q[0]+12}" y="${q[1]-24}" width="90" height="42"><div xmlns="http://www.w3.org/1999/xhtml" class="math-label">${math('s_{\\mathcal P}')}</div></foreignObject><text x="275" y="38" class="svg-math">s₂</text><text x="514" y="221" class="svg-math">s₁</text><text x="255" y="220" class="svg-small">0</text><foreignObject x="408" y="298" width="130" height="55"><div xmlns="http://www.w3.org/1999/xhtml" class="math-label" style="padding:10px 15px;border:1px solid #dfe4d4;background:#fafcf3">${math('T = '+actout(v,[{vector:sub(clusters[0].p,clusters[1].p),plus:.3,minus:1.4},{vector:sub(clusters[2].p,clusters[3].p),plus:.5,minus:1.1}]).temperature.toFixed(2))}</div></foreignObject>`;
 $('#hero-space').innerHTML=s;
}
function benchmark(key){
 $$('[data-benchmark]').forEach(b=>b.setAttribute('aria-pressed',b.dataset.benchmark===key));$('#benchmark-metric').textContent={ae:'LC-WR (%) ↑',as:'Goal completion ↑',mt:'GPT-4 score ↑'}[key];
 const shown=[results[0],results[3],results[6],results[7],results[1]].sort((a,b)=>b[key]-a[key]);const max={ae:35,as:1,mt:10}[key];
 $('#benchmark-chart').innerHTML=shown.map(r=>`<div class="benchmark-row ${r===results[0]?'ours':''}"><span>${r.name}</span><div class="benchmark-bar"><i style="width:${r[key]/max*100}%"></i></div><span>${r[key].toFixed(key==='as'?4:2)}</span></div>`).join('');
}
function tick(now){
 const dt=Math.min(.08,last?(now-last)/1000:0);last=now;
 if(state.playing&&state.module==='scenarize'){
  if(state.view==='train'){state.step=Math.min(160,state.step+dt*12*state.speed);if(state.step>=160)state.playing=false;}
  else {state.token=Math.min(4,state.token+dt*.65*state.speed);const prev=Math.floor(state.token-dt*.65*state.speed);if(Math.floor(state.token)!==prev)updateFlow();if(state.token>=4)state.playing=false;}
  drawTrain();updatePlay();
 }
 if(state.orbit&&state.module==='actout'){const theta=Math.atan2(state.vector[1],state.vector[0])+dt*.35;state.vector=[Math.cos(theta)*.8,Math.sin(theta)*.8];drawActout();}
 if(!reduced&&now-heroLast>45){heroTime+=dt;drawHero();heroLast=now;}
 requestAnimationFrame(tick);
}
function stopOrbit(){state.orbit=false;$('#orbit').textContent='▶';$('#orbit').setAttribute('aria-label','Animate scenario embedding');}
function setDirection(angle){state.vector=scale([Math.cos(angle),Math.sin(angle)],Math.max(.35,norm(state.vector)));drawActout();}
function bind(){
 $$('[data-module]').forEach(b=>{b.addEventListener('click',()=>setModule(b.dataset.module));b.addEventListener('keydown',e=>{if(['ArrowLeft','ArrowRight','Home','End'].includes(e.key)){e.preventDefault();const module=['ArrowRight','End'].includes(e.key)?'actout':'scenarize';setModule(module);$(`[data-module="${module}"]`).focus();}});});
 $$('[data-open]').forEach(b=>b.addEventListener('click',()=>setModule(b.dataset.open)));$$('[data-view]').forEach(b=>b.addEventListener('click',()=>setView(b.dataset.view)));
 $('#play').addEventListener('click',()=>{if(state.view==='train'&&state.step>=160)state.step=0;if(state.view==='infer'&&state.token>=4)state.token=0;state.playing=!state.playing;updatePlay();updateFlow();});
 $('#next').addEventListener('click',()=>{state.playing=false;state.view==='train'?state.step=Math.min(160,Math.floor(state.step)+1):state.token=Math.min(4,Math.floor(state.token)+1);updatePlay();updateFlow();drawTrain();});
 $('#progress').addEventListener('input',e=>{state.playing=false;state.view==='train'?state.step=+e.target.value:state.token=+e.target.value;updatePlay();updateFlow();drawTrain();});
 $('#speed').addEventListener('change',e=>state.speed=+e.target.value);
 $('#scenario').addEventListener('change',e=>{state.scenario=+e.target.value;drawTrain();});
 $('#margin').addEventListener('input',e=>{state.beta=+e.target.value;$('#margin-value').textContent=state.beta.toFixed(2);rebuild();});
 $('#learning-rate').addEventListener('input',e=>{state.lr=+e.target.value;$('#lr-value').textContent=state.lr.toFixed(4);rebuild();});
 $('#reset').addEventListener('click',()=>{if(state.module==='actout'){stopOrbit();state.vector=[.65,.38];state.overrides={};syncEndpoints();drawActout();}else {state.playing=false;state.step=state.view==='infer'?160:0;state.token=0;updateFlow();updatePlay();drawTrain();}});
 $('#transfer').addEventListener('click',()=>{actoutFrame=state.view==='train'?frames[Math.round(state.step)]:frames.at(-1);const p=state.view==='train'?actoutFrame.chosen[state.scenario][0]:forward(actoutFrame.model,inputs(state.scenario,0))[Math.floor(state.token)];state.vector=norm(p)>1e-6?[...p]:[.65,.38];state.overrides={};setModule('actout');});
 $('#details-toggle').addEventListener('click',()=>{const open=$('#method-details').hidden;$('#method-details').hidden=!open;$('#details-toggle').setAttribute('aria-expanded',open);$('#details-toggle').innerHTML=`How it works <span>${open?'−':'+'}</span>`;});
 $('#axis-select').addEventListener('change',e=>{state.selectedAxis=+e.target.value;syncEndpoints();drawActout();});
 $('#axis-contributions').addEventListener('click',e=>{const b=e.target.closest('[data-axis]');if(b){state.selectedAxis=+b.dataset.axis;syncEndpoints();drawActout();}});
 $('#show-axes').addEventListener('change',e=>{state.showAxes=e.target.checked;drawActout();});
 for(const [id,key] of [['tau-plus','plus'],['tau-minus','minus']])$('#'+id).addEventListener('input',e=>{state.overrides[state.selectedAxis]={...state.overrides[state.selectedAxis],[key]:+e.target.value};$('#'+id+'-value').textContent=(+e.target.value).toFixed(2);drawActout();});
 $('#restore-endpoints').addEventListener('click',()=>{state.overrides={};syncEndpoints();drawActout();});
 $('#angle').addEventListener('input',e=>{stopOrbit();setDirection(+e.target.value*Math.PI/180);});
 $('#orbit').addEventListener('click',()=>{state.orbit=!state.orbit;$('#orbit').textContent=state.orbit?'Ⅱ':'▶';$('#orbit').setAttribute('aria-label',state.orbit?'Pause scenario animation':'Animate scenario embedding');});
 const svg=$('#actout-space');
 const move=e=>{const r=svg.getBoundingClientRect(),x=(e.clientX-r.left)/r.width*(narrow()?560:760)+(narrow()?100:0),y=(e.clientY-r.top)/r.height*470;let p=[(x-C.x)/C.s,(C.y-y)/C.s];if(norm(p)>1.08)p=scale(unit(p),1.08);if(norm(p)<.08)p=scale(unit(norm(p)<1e-9?[1,0]:p),.08);state.vector=p;drawActout();};
 svg.addEventListener('pointerdown',e=>{if(e.target.closest('#embedding-handle')){drag=true;stopOrbit();svg.setPointerCapture(e.pointerId);e.preventDefault();}});svg.addEventListener('pointermove',e=>{if(drag)move(e);});svg.addEventListener('pointerup',()=>drag=false);svg.addEventListener('pointercancel',()=>drag=false);
 svg.addEventListener('keydown',e=>{if(e.target.closest('#embedding-handle')&&['ArrowLeft','ArrowRight','ArrowUp','ArrowDown'].includes(e.key)){e.preventDefault();stopOrbit();const theta=Math.atan2(state.vector[1],state.vector[0]);setDirection(theta+(['ArrowUp','ArrowRight'].includes(e.key)?1:-1)*Math.PI/60);$('#embedding-handle').focus();}});
 $$('[data-benchmark]').forEach(b=>b.addEventListener('click',()=>benchmark(b.dataset.benchmark)));
 $('#copy-citation').addEventListener('click',async()=>{try{await navigator.clipboard.writeText($('#bibtex').textContent);$('#copy-status').textContent='BibTeX copied.';}catch{$('.citation-fallback').open=true;$('#copy-status').textContent='Select and copy the BibTeX below.';}});
}
function init(){
 renderMath();$('#scenario').innerHTML=labels.map((l,i)=>`<option value="${i}">${l.name}</option>`).join('');$('#axis-select').innerHTML=makeAxes(actoutFrame.centroids).map((a,i)=>`<option value="${i}" ${i===1?'selected':''}>${a.name}</option>`).join('');
 $('#results-table').innerHTML=results.map((r,i)=>`<tr class="${i===0?'ours':''}"><th scope="row">${r.name}</th><td>${r.as.toFixed(4)}</td><td>${r.ae.toFixed(2)}</td><td>${r.mt.toFixed(2)}</td></tr>`).join('');
 addEventListener('resize',()=>{state.module==='actout'?drawActout():drawTrain();});
 bind();updateFlow();drawTrain();drawHero();benchmark('ae');requestAnimationFrame(tick);
}
if(document.readyState==='loading')document.addEventListener('DOMContentLoaded',init);else init();
