// Memory of the LOCA sessions (onnxruntime-web WASM, Node): RSS after creating the sessions and after
// one features + head run. WASM memory never shrinks, so 'after run' is what a browser tab keeps.
// Usage: [MODEL_DIR=dir] node scripts/ai-mem-probe.mjs [all|basic|disabled] [lean]
import * as ort from 'onnxruntime-web';
import { readFileSync } from 'node:fs';
const opt = process.argv[2] ?? 'all';
const mb = () => Math.round(process.memoryUsage().rss / 1048576);
ort.env.wasm.numThreads = 1;
const lean = process.argv[3] === 'lean';
const o = { executionProviders: ['wasm'], graphOptimizationLevel: opt, ...(lean ? { enableCpuMemArena: false, enableMemPattern: false } : {}) };
console.log('start rss', mb());
const f = await ort.InferenceSession.create(readFileSync((process.env.MODEL_DIR ?? 'public/models/loca') + '/loca-features-w8.onnx'), o);
console.log('features session rss', mb());
const h = await ort.InferenceSession.create(readFileSync((process.env.MODEL_DIR ?? 'public/models/loca') + '/loca-head-w8.onnx'), o);
console.log('head session rss', mb());
const img = new Float32Array(3 * 512 * 512).fill(0.1);
const t0 = Date.now();
const fo = await f.run({ image: new ort.Tensor('float32', img, [1, 3, 512, 512]) });
const feat = fo.features.data;
console.log('features run rss', mb(), 'ms', Date.now() - t0);
const ho = await h.run({ features: new ort.Tensor('float32', feat, [1, 256, 64, 64]), exemplar_features: new ort.Tensor('float32', feat, [1, 256, 64, 64]), bboxes: new ort.Tensor('float32', new Float32Array([100,100,150,150,200,200,250,250,300,300,350,350]), [1, 3, 4]) });
console.log('head run rss', mb(), 'ms', Date.now() - t0, 'out', ho.density.dims);
