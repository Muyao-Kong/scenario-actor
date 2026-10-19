import test from 'node:test';
import assert from 'node:assert/strict';
import {edgeGesture} from '../deck.js';

test('reading within a long page cannot turn a chapter, even after reaching its edge',()=>{
 const reading=edgeGesture({top:false,bottom:false});
 assert.equal(reading(0,900),0);
 // The start boundary is immutable: another move in the same gesture stays local.
 assert.equal(reading(0,1200),0);
 assert.equal(edgeGesture({top:false,bottom:true})(0,80),1);
});
test('top and bottom permit only their outward direction',()=>{
 assert.equal(edgeGesture({top:true,bottom:false})(0,-80),-1);
 assert.equal(edgeGesture({top:true,bottom:false})(0,80),0);
 assert.equal(edgeGesture({top:false,bottom:true})(0,-80),0);
 assert.equal(edgeGesture({top:false,bottom:true})(0,80),1);
});
test('a gesture requires deliberate vertical movement and turns at most once',()=>{
 const gesture=edgeGesture({top:false,bottom:true});
 assert.equal(gesture(0,30),0);
 assert.equal(gesture(100,90),0);
 assert.equal(gesture(10,80),1);
 assert.equal(gesture(10,300),0);
});
test('short pages support either direction with a fresh gesture',()=>{
 assert.equal(edgeGesture({top:true,bottom:true})(0,80),1);
 assert.equal(edgeGesture({top:true,bottom:true})(0,-80),-1);
});
