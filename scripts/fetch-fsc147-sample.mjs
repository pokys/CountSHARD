// Downloads the small FSC-147 sample used by `npm run eval` (tests/eval/ai.eval.test.ts):
// 20 test-split images (12 "tree logs", 8 other classes, picked with a fixed seed) and their
// annotations. Nothing is committed: files go to tests/eval/.data/fsc147 (git-ignored).
//
// Sources: annotation_FSC147_384.json from github.com/cvlab-stonybrook/LearningToCountEverything,
// images (images_384_VarV2) from the Hugging Face mirror isentropic/FSC147.
// FSC-147: Ranjan et al., "Learning To Count Everything", CVPR 2021. Images come from the web;
// check the dataset's terms before using them for anything beyond evaluation.
import { mkdirSync, writeFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

export const FSC_SAMPLE = [
  '4943.jpg', '4947.jpg', '4950.jpg', '4951.jpg', '4952.jpg', '4957.jpg', '4958.jpg', '4961.jpg',
  '4964.jpg', '4966.jpg', '4967.jpg', '4970.jpg', '2227.jpg', '2916.jpg', '5839.jpg', '5852.jpg',
  '5917.jpg', '5925.jpg', '6847.jpg', '7109.jpg',
];

const ANN_URL = 'https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/annotation_FSC147_384.json';
const CLS_URL = 'https://raw.githubusercontent.com/cvlab-stonybrook/LearningToCountEverything/master/data/ImageClasses_FSC147.txt';
const IMG_URL = (n) => `https://huggingface.co/datasets/isentropic/FSC147/resolve/main/images_384_VarV2/${n}`;

const out = join(dirname(fileURLToPath(import.meta.url)), '..', 'tests', 'eval', '.data', 'fsc147');
mkdirSync(out, { recursive: true });

async function get(url) {
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  return Buffer.from(await r.arrayBuffer());
}

const ann = JSON.parse((await get(ANN_URL)).toString('utf8'));
const classes = Object.fromEntries((await get(CLS_URL)).toString('utf8').split(/\r?\n/).filter(Boolean).map((l) => l.split('\t')));
const subset = {};
for (const n of FSC_SAMPLE) {
  const a = ann[n];
  subset[n] = { class: classes[n], count: a.points.length, boxes: a.box_examples_coordinates.slice(0, 3) };
  const p = join(out, n);
  if (!existsSync(p)) writeFileSync(p, await get(IMG_URL(n)));
}
writeFileSync(join(out, 'annotations.json'), JSON.stringify(subset, null, 1));
console.log(`FSC-147 sample: ${FSC_SAMPLE.length} images in ${out}`);
