// The paper's equations, evaluated on a small deterministic teaching model.
export const add=(a,b)=>a.map((v,i)=>v+b[i]);
export const sub=(a,b)=>a.map((v,i)=>v-b[i]);
export const scale=(a,t)=>a.map(v=>v*t);
export const dot=(a,b)=>a.reduce((v,x,i)=>v+x*b[i],0);
export const norm=a=>Math.sqrt(dot(a,a));
export const unit=a=>norm(a)>1e-10?scale(a,1/norm(a)):a.map(()=>0);
export const dist=(a,b)=>dot(sub(a,b),sub(a,b));
export const mix=(a,b,t)=>a.map((v,i)=>v+(b[i]-v)*t);
export const triplet=(a,p,n,beta=.3)=>Math.max(0,dist(a,p)-dist(a,n)+beta);
export const smooth=t=>t*t*(3-2*t);
export function actout(s,axes){
 const sn=norm(s);
 const values=axes.map(a=>{
  const rho=sn>1e-10?Math.max(-1,Math.min(1,dot(unit(s),unit(a.vector)))):0;
  const estimate=(1+rho)/2*a.plus+(1-rho)/2*a.minus;
  return {...a,rho,estimate,weight:Math.abs(rho)};
 });
 const total=values.reduce((v,a)=>v+a.weight,0);
 return {temperature:total>1e-10?values.reduce((v,a)=>v+a.weight*a.estimate,0)/total:null,axes:values.map(a=>({...a,weight:total>1e-10?a.weight/total:0}))};
}
export const labels=[
 {id:'rigorous',name:'Rigorous',color:'#546e9e',pos:[-.8,.25],temp:.30,opposite:1,prompt:'Explain / the result / carefully / with a proof.'},
 {id:'brainstorming',name:'Brainstorming',color:'#c98a40',pos:[.65,-.2],temp:1.40,opposite:0,prompt:'Imagine / new directions / without limits / for this idea.'},
 {id:'institutional',name:'Institutional',color:'#668a87',pos:[-.55,-.4],temp:.50,opposite:3,prompt:'Review / this request / under the policy / and resolve it.'},
 {id:'casual',name:'Casual conversation',color:'#9a9472',pos:[.6,.4],temp:1.10,opposite:2,prompt:'Hey / how was / your weekend / by the way?'},
 {id:'formal',name:'Formal',color:'#8a7ab3',pos:[-.4,.65],temp:.55,opposite:5,prompt:'Please / address the committee / with care / and courtesy.'},
 {id:'playful',name:'Playful',color:'#c57e89',pos:[.65,-.65],temp:1.50,opposite:4,prompt:'Make / a silly joke / about my cat / being the boss.'},
 {id:'support_pos',name:'Positive support',color:'#90a16a',pos:[.12,.75],temp:1.00,opposite:7,prompt:'I finally / got the offer / after all that work / celebrate with me!'},
 {id:'support_neg',name:'Negative support',color:'#6a9db3',pos:[-.05,-.8],temp:.65,opposite:6,prompt:'I feel / overwhelmed today / and could use / some reassurance.'},
 {id:'adversarial',name:'Adversarial',color:'#b77050',pos:[.82,.05],temp:.75,opposite:0,prompt:'I disagree / with your claim / defend it / with evidence.'}
];
export function makeAxes(centroids,endpointOverrides={}){
 const mean=centroids.filter((_,i)=>i!==8).reduce((v,p)=>add(v,scale(p,1/8)),[0,0]);
 // Table 2 gives no non-adversarial temperature. Its mean below is a demo convention.
 const nonAdvTemp=labels.filter((_,i)=>i!==8).reduce((s,l)=>s+l.temp,0)/8;
 return [
  {name:'Duty',a:2,b:3}, {name:'Intellect',a:0,b:1},
  {name:'Adversity',a:8,b:-1}, {name:'Negativity',a:6,b:7},
  {name:'Sociality',a:4,b:5}
 ].map((a,i)=>({...a,color:['#668a87','#546e9e','#b77050','#90a16a','#8a7ab3'][i],start:a.b<0?mean:centroids[a.b],end:centroids[a.a],vector:sub(centroids[a.a],a.b<0?mean:centroids[a.b]),plus:labels[a.a].temp,minus:a.b<0?nonAdvTemp:labels[a.b].temp,...endpointOverrides[i]}));
}
function rng(seed){return()=>{seed|=0;seed=seed+0x6D2B79F5|0;let t=Math.imul(seed^seed>>>15,1|seed);t=t+Math.imul(t^t>>>7,61|t)^t;return((t^t>>>14)>>>0)/4294967296;};}
export function createModel(){return {W:[[.11,-.04,.035,.02],[.015,.09,-.025,.035]],U:[[.42,.04],[-.03,.38]],b:[0,0]};}
export function inputs(label,index=0){
 const r=rng(71+label*317+index*911),p=labels[label].pos;
 return Array.from({length:4},(_,t)=>{const strength=(t+1)/4;return [p[0]*strength+(r()-.5)*.18,p[1]*strength+(r()-.5)*.18,(r()-.5)*.35,(r()-.5)*.35];});
}
export function forward(model,xs){
 const states=[[0,0]];
 xs.forEach(x=>states.push(model.b.map((v,j)=>Math.tanh(v+dot(model.W[j],x)+dot(model.U[j],states.at(-1))))));
 return states;
}
function backward(model,xs,states,gradient,g){
 let ds=[...gradient];
 for(let t=xs.length-1;t>=0;t--){
  const dz=ds.map((v,j)=>v*(1-states[t+1][j]**2));
  for(let j=0;j<2;j++){
   g.b[j]+=dz[j];
   for(let k=0;k<4;k++)g.W[j][k]+=dz[j]*xs[t][k];
   for(let k=0;k<2;k++)g.U[j][k]+=dz[j]*states[t][k];
  }
  ds=[0,1].map(k=>dz.reduce((v,d,j)=>v+d*model.U[j][k],0));
 }
}
export function trainStep(model,step,beta=.3,lr=.0035){
 const g={W:[[0,0,0,0],[0,0,0,0]],U:[[0,0],[0,0]],b:[0,0]};let loss=0;
 for(let c=0;c<labels.length;c++){
  const seqs=[inputs(c,(step%7)+1),inputs(c,(step%7)+9),inputs(labels[c].opposite,(step%7)+5)];
  const states=seqs.map(x=>forward(model,x)),[a,p,n]=states.map(x=>x.at(-1));
  const l=triplet(a,p,n,beta);loss+=l;
  if(l>0){
   const grads=[scale(sub(n,p),2),scale(sub(p,a),2),scale(sub(a,n),2)];
   seqs.forEach((x,i)=>backward(model,x,states[i],grads[i],g));
  }
 }
 for(const name of ['W','U'])for(let i=0;i<model[name].length;i++)for(let j=0;j<model[name][i].length;j++)model[name][i][j]-=lr*g[name][i][j]/9;
 model.b=model.b.map((v,i)=>v-lr*g.b[i]/9);
 return loss/9;
}
export function trainingRun(beta=.3,lr=.0035,count=160){
 const model=createModel(),frames=[];
 for(let step=0;step<=count;step++){
  const points=labels.flatMap((_,c)=>Array.from({length:10},(_,i)=>({label:c,p:forward(model,inputs(c,i)).at(-1)})));
  const centroids=labels.map((_,c)=>points.filter(p=>p.label===c).reduce((s,p)=>add(s,scale(p.p,.1)),[0,0]));
  const chosen=labels.map((l,c)=>[forward(model,inputs(c,0)).at(-1),forward(model,inputs(c,9)).at(-1),forward(model,inputs(l.opposite,0)).at(-1)]);
  frames.push({step,points,centroids,chosen,model:JSON.parse(JSON.stringify(model)),loss:chosen.reduce((s,[a,p,n])=>s+triplet(a,p,n,beta),0)/9});
  if(step<count)trainStep(model,step,beta,lr);
 }
 return frames;
}
export const results=[
 {name:'Scenario-Actor',as:.8015,ae:30.03,mt:7.91},
 {name:'Greedy',as:.7582,ae:22.87,mt:7.79},
 {name:'Fixed T = 0.5',as:.7540,ae:23.63,mt:7.79},
 {name:'Fixed T = 0.7',as:.7801,ae:25.68,mt:7.83},
 {name:'Fixed T = 1.0',as:.7692,ae:23.42,mt:7.61},
 {name:'Fixed T = 1.5',as:.6989,ae:8.60,mt:6.76},
 {name:'EDT',as:.7671,ae:25.58,mt:6.74},
 {name:'Nucleus (p = 0.95)',as:.7557,ae:23.59,mt:7.63},
 {name:'Random-T',as:.7699,ae:20.44,mt:7.71}
];
