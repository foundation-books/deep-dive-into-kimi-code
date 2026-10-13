// Independent finite arithmetic probe for the attributed WebGen-Agent estimator.
// The pinned upstream implementation was source-read, not imported or trained.
import assert from "node:assert/strict";

function coefficients(trajectories){
  const pooled=trajectories.flatMap(t=>t.map(step=>step.reward));
  const mean=pooled.reduce((a,b)=>a+b,0)/pooled.length;
  const variance=pooled.length>1?pooled.reduce((a,b)=>a+(b-mean)**2,0)/(pooled.length-1):0;
  const sd=pooled.length>1?Math.sqrt(variance):1;
  return trajectories.map(t=>t.map(({reward,editTokens,observationTokens})=>({
    value:(reward-(pooled.length>1?mean:0))/(sd+1e-6),
    actorCoefficients:Array(editTokens).fill((reward-(pooled.length>1?mean:0))/(sd+1e-6)),
    observationMask:Array(observationTokens).fill(0)
  })));
}
const unequal=coefficients([
  [{reward:1,editTokens:2,observationTokens:3},{reward:3,editTokens:1,observationTokens:2}],
  [{reward:5,editTokens:3,observationTokens:1}]
]);
const near=(a,b)=>Math.abs(a-b)<1e-5;
assert(near(unequal[0][0].value,-1));
assert(near(unequal[0][1].value,0));
assert(near(unequal[1][0].value,1));
assert.deepEqual(unequal[0][0].observationMask,[0,0,0]);
assert.equal(unequal[0][0].actorCoefficients.length,2);
assert.equal(unequal[1][0].actorCoefficients.length,3);
assert(near(coefficients([[{reward:4,editTokens:1,observationTokens:1},{reward:4,editTokens:2,observationTokens:1}]])[0][0].value,0));
assert(near(coefficients([[{reward:2,editTokens:1,observationTokens:1}]])[0][0].value,2));
console.log(JSON.stringify({pooledRewards:[1,3,5],sampleMean:3,sampleSd:2,unequal,zeroVariance:0,singleton:2},null,2));
