// Memory and speed of the GeCo2 graphs in onnxruntime-web WASM (Node, 1 thread), like
// scripts/ai-mem-probe.mjs for LOCA: RSS after creating the sessions and after each run.
// WASM memory never shrinks, so the last number is what a browser tab keeps.
// Usage: node scripts/geco2/geco2-probe.mjs <onnx-dir> [size=1024] [variant=w8] [all|basic|disabled]
import * as ort from 'onnxruntime-web';
import { readFileSync } from 'node:fs';
import { join } from 'node:path';

const [dir, sizeArg = '1024', variant = 'w8', opt = 'disabled', lean] = process.argv.slice(2);
const S = Number(sizeArg);
const mb = () => Math.round(process.memoryUsage().rss / 1048576);
ort.env.wasm.numThreads = 1;
const o = { executionProviders: ['wasm'], graphOptimizationLevel: opt, ...(lean === 'lean' ? { enableCpuMemArena: false, enableMemPattern: false } : {}) };
const load = (k) => ort.InferenceSession.create(readFileSync(join(dir, `geco2-${k}-${S}-${variant}.onnx`)), o);
console.log('start rss', mb());
const b = await load('backbone');
console.log('backbone session rss', mb());
const e = await load('exemplars');
const h = await load('head');
console.log('all sessions rss', mb());
// A smooth pseudo-image (not constant, so attention is not degenerate).
const img = new Float32Array(3 * S * S);
for (let i = 0; i < img.length; i++) img[i] = Math.sin(i * 0.0007) + Math.cos(i * 0.013);
let t = Date.now();
const f = await b.run({ image: new ort.Tensor('float32', img, [1, 3, S, S]) });
console.log('backbone run rss', mb(), 'ms', Date.now() - t);
t = Date.now();
const r = S / 1024;
const boxes = new Float32Array([100, 100, 150, 150, 300, 300, 350, 350, 500, 500, 550, 550].map((v) => v * r));
const p = await e.run({ feat: f.feat, fpn1: f.fpn1, fpn0: f.fpn0, boxes: new ort.Tensor('float32', boxes, [1, 3, 4]) });
console.log('exemplars run rss', mb(), 'ms', Date.now() - t);
t = Date.now();
const out = await h.run({ feat: f.feat, fpn1: f.fpn1, fpn0: f.fpn0, proto0: p.proto0, proto1: p.proto1, proto2: p.proto2 });
console.log('head run rss', mb(), 'ms', Date.now() - t, 'out', out.objectness.dims, out.ltrb.dims);
