import test from 'node:test';
import assert from 'node:assert/strict';
import {camera,project,spherePoint,angleArc,rotateVector} from '../scene3d.js';
import {dot,norm,unit,scale,sub,dist,actout,makeAxes,trainingRun,forward,inputs} from '../geometry.js';
const close=(a,b,e=1e-9)=>assert.ok(Math.abs(a-b)<e,`${a} != ${b}`);
test('the recurrent model learns three genuine coordinates',()=>{
 const f=trainingRun(),a=f[0],b=f.at(-1);
 assert.equal(b.model.b.length,3);assert.equal(b.model.U.length,3);
 assert.ok(b.loss<a.loss*.1);
 for(const p of b.points){assert.equal(p.p.length,3);assert.ok(p.p.every(Number.isFinite));}
 assert.ok(b.centroids.some(p=>Math.abs(p[2])>.15));
 // Nonzero scalar triple product: the class geometry is not confined to a plane.
 const [u,v,w]=[0,2,6].map(i=>sub(b.centroids[i],b.centroids[8]));
 const determinant=u[0]*(v[1]*w[2]-v[2]*w[1])-u[1]*(v[0]*w[2]-v[2]*w[0])+u[2]*(v[0]*w[1]-v[1]*w[0]);
 assert.ok(Math.abs(determinant)>.001);
 assert.deepEqual(b.chosen[0][0],forward(b.model,inputs(0,0)).at(-1));
});
test('the three displayed scenario pairs keep substantial depth throughout training',()=>{
 const cross=(a,b)=>[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 for(const margin of [.1,.3,1]){
  const frames=trainingRun(margin);
  for(const frame of [frames[0],frames[40],frames.at(-1)]){
   const axes=makeAxes(frame.centroids);
   const [u,v,w]=[0,1,4].map(i=>unit(axes[i].vector));
   // Unit-vector volume is scale invariant: small but nonzero depth is insufficient.
   assert.ok(Math.abs(dot(u,cross(v,w)))>.5,`Nearly planar at margin ${margin}, step ${frame.step}`);
  }
 }
});
test('three-dimensional cosine temperature depends on the third coordinate',()=>{
 const axes=[{vector:[0,0,1],plus:.3,minus:1.5}];
 close(actout([0,0,2],axes).temperature,.3);close(actout([0,0,-2],axes).temperature,1.5);
});
test('orthographic camera has an orthonormal frame at any tested angle',()=>{
 for(const yaw of [-2,-.4,.65,2.1])for(const pitch of [-1,.38,1]){
  const c=camera(yaw,pitch);for(const v of [c.right,c.up,c.depth])close(norm(v),1);
  close(dot(c.right,c.up),0);close(dot(c.right,c.depth),0);close(dot(c.up,c.depth),0);
 }
});
test('camera rotation preserves spatial distance, cosine, and selected temperature',()=>{
 const f=trainingRun().at(-1),axes=makeAxes(f.centroids),s=[.4,-.6,.7];
 const target=actout(s,axes).temperature;
 for(const [yaw,pitch] of [[0,0],[.6,.4],[-1.3,.8]]){
  const c=camera(yaw,pitch,1,0,0),rotate=v=>[dot(v,c.right),dot(v,c.up),dot(v,c.depth)];
  close(dist(rotate(s),rotate(axes[0].vector)),dist(s,axes[0].vector));
  close(actout(rotate(s),axes.map(a=>({...a,vector:rotate(a.vector)}))).temperature,target);
 }
});
test('dragging on the projected sphere reconstructs the selected hemisphere in 3D',()=>{
 for(const c of [camera(),camera(-1.1,.7)])for(const v of [[.3,.4,.6],[-.4,.2,-.5]]){
  const [x,y]=project(v,c),recovered=spherePoint(x,y,norm(v),c,Math.sign(dot(v,c.depth)));
  close(dist(v,recovered),0);
  close(norm(spherePoint(x+10000,y-5000,norm(v),c)),norm(v));
 }
});
test('the geometric angle arc lies on the actual 3D plane of the two vectors',()=>{
 const a=unit([1,0,1]),b=unit([0,1,1]),arc=angleArc(a,b,.3),n=[a[1]*b[2]-a[2]*b[1],a[2]*b[0]-a[0]*b[2],a[0]*b[1]-a[1]*b[0]];
 close(dist(arc[0],scale(a,.3)),0);close(dist(arc.at(-1),scale(b,.3)),0);
 for(const p of arc){close(norm(p),.3);close(dot(p,n),0);}
});
test('azimuth and elevation controls preserve norm and can leave the xy plane',()=>{
 const p=rotateVector([1,0,0],.2,.3);close(norm(p),1);assert.ok(p[2]>.25);
});

test('normalized Actout handles start without a jump and stay on the unit sphere',()=>{
 const axes=makeAxes(trainingRun().at(-1).centroids);
 for(const raw of [[.58,.32,.46],[.08,.12,.04],[.8,1.1,-.6]]){
  const s=unit(raw);
  close(norm(s),1);
  close(actout(raw,axes).temperature,actout(s,axes).temperature);
  for(const c of [camera(),camera(-1.1,.7)]){
   const [x,y]=project(s,c),side=Math.sign(dot(s,c.depth));
   const initial=spherePoint(x,y,1,c,side);
   close(dist(initial,s),0);
   close(actout(initial,axes).temperature,actout(raw,axes).temperature);
   for(const [dx,dy] of [[40,25],[1000,-1000]]){
    const moved=spherePoint(x+dx,y+dy,1,c,side);
    close(norm(moved),1);
    close(norm(rotateVector(moved,.2,.1)),1);
   }
  }
 }
});
