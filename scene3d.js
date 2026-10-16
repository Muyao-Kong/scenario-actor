// Orthographic camera for the three-dimensional teaching space.
// Camera rotation changes only the view, never distances, angles, or temperature.
import {dot,norm,unit,scale,add,sub} from './geometry.js?v=3d5';
export function camera(yaw=.65,pitch=.38,zoom=174,cx=380,cy=232){
 const c=Math.cos(yaw),s=Math.sin(yaw),p=Math.sin(pitch),q=Math.cos(pitch);
 return {yaw,pitch,zoom,cx,cy,right:[c,-s,0],up:[s*p,c*p,q],depth:[-s*q,-c*q,p]};
}
export function project(v,c){return [c.cx+dot(v,c.right)*c.zoom,c.cy-dot(v,c.up)*c.zoom,dot(v,c.depth)];}
export function spherePoint(x,y,radius,c,hemisphere=1){
 let a=(x-c.cx)/c.zoom,b=(c.cy-y)/c.zoom;
 const length=Math.hypot(a,b);
 if(length>radius){a*=radius/length;b*=radius/length;}
 const depth=Math.sqrt(Math.max(0,radius*radius-a*a-b*b))*(hemisphere<0?-1:1);
 return add(add(scale(c.right,a),scale(c.up,b)),scale(c.depth,depth));
}
export function angleArc(a,b,radius=.3,segments=36){
 const u=unit(a),v=unit(b),cosine=Math.max(-1,Math.min(1,dot(u,v))),theta=Math.acos(cosine);
 let tangent=sub(v,scale(u,cosine));
 if(norm(tangent)<1e-9){const helper=Math.abs(u[2])<.8?[0,0,1]:[0,1,0];tangent=sub(helper,scale(u,dot(helper,u)));}
 tangent=unit(tangent);
 return Array.from({length:segments+1},(_,i)=>add(scale(u,radius*Math.cos(theta*i/segments)),scale(tangent,radius*Math.sin(theta*i/segments))));
}
export function rotateVector(v,azimuthDelta=0,elevationDelta=0){
 const radius=norm(v),azimuth=Math.atan2(v[1],v[0])+azimuthDelta;
 const elevation=Math.max(-Math.PI/2+.015,Math.min(Math.PI/2-.015,Math.asin(v[2]/radius)+elevationDelta));
 return [radius*Math.cos(elevation)*Math.cos(azimuth),radius*Math.cos(elevation)*Math.sin(azimuth),radius*Math.sin(elevation)];
}
