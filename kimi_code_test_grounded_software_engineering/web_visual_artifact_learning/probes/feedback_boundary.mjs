// Book-authored request-boundary probe, not an invocation of WebGen-Agent.
// Scripted feedback makes request construction observable without a model service.
import assert from 'node:assert/strict';
import { createHash } from 'node:crypto';
import { readFile } from 'node:fs/promises';

const evidence = new URL('../evidence/screenshots/', import.meta.url);
const states = [
  { name: 'D2 default desktop', file: 'D2-desktop-default.png', feedback: 'The chart legend is missing; restore it.' },
  { name: 'D3 default desktop', file: 'D3-desktop-default.png', feedback: 'The chart legend is visible; retain this layout.' },
];

function sha256(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function makeVisualRequest(png) {
  return {
    input: [
      { type: 'text', text: 'Describe the visible dashboard state.' },
      { type: 'image', mimeType: 'image/png', data: png.toString('base64') },
    ],
  };
}

const visualCalls = [];
const codingCalls = [];
function recordingVisualStub(request, scriptedFeedback, state) {
  const image = request.input.find((part) => part.type === 'image');
  assert.equal(image?.mimeType, 'image/png');
  const bytes = Buffer.from(image.data, 'base64');
  visualCalls.push({ state, sha256: sha256(bytes), byteLength: bytes.length,
    width: bytes.readUInt32BE(16), height: bytes.readUInt32BE(20) });
  return scriptedFeedback;
}

function recordingCodingStub(request) {
  assert.equal(request.messages.at(-1).role, 'user');
  assert.equal(typeof request.messages.at(-1).content, 'string');
  codingCalls.push(structuredClone(request));
}

for (const state of states) {
  const png = await readFile(new URL(state.file, evidence));
  assert.equal(png.subarray(0, 8).toString('hex'), '89504e470d0a1a0a');
  const feedback = recordingVisualStub(makeVisualRequest(png), state.feedback, state.name);
  assert.equal(visualCalls.at(-1).sha256, sha256(png));
  recordingCodingStub({ messages: [
    { role: 'system', content: 'Revise the dashboard using returned feedback.' },
    { role: 'user', content: feedback },
  ] });
}

assert.notEqual(visualCalls[0].sha256, visualCalls[1].sha256);
assert.equal(visualCalls[0].width, visualCalls[1].width);
assert.equal(visualCalls[0].height, visualCalls[1].height);
assert.notEqual(codingCalls[0].messages.at(-1).content, codingCalls[1].messages.at(-1).content);
for (const call of codingCalls) {
  assert.equal(JSON.stringify(call).includes('image/png'), false);
  assert.equal(JSON.stringify(call).includes('iVBOR'), false);
}

// The source and selected filter are the same here; viewport alone changes pixels.
const desktop = await readFile(new URL('D3-A-3-8.png', evidence));
const narrow = await readFile(new URL('D3-narrow-A-3-8.png', evidence));
assert.notEqual(sha256(desktop), sha256(narrow));

console.log(JSON.stringify({
  visualCalls,
  codingFeedback: codingCalls.map((call) => call.messages.at(-1).content),
  sameStateViewportChangesPixels: true,
  note: 'Stubbed request boundary only; no model interpretation or upstream execution.',
}, null, 2));
