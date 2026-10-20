// Writes the Mode B evaluation cases to a folder so the Python GeCo2 scripts can run the same
// inputs as tests/eval/ai.eval.test.ts (LOCA): the 10 synthetic scenes (PNG + true circles),
// the FSC-147 sample and its 2 × 2 "tree logs" mosaics (referenced, not copied).
// Usage: npx vite-node scripts/geco2/dump-cases.ts <out-dir>
import { existsSync, mkdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { loadCv } from '../../tests/loadCv';
import { makeScene } from '../../tests/synthetic/scene';
import { writePng } from '../../tests/synthetic/png';
import { SCENES } from '../../tests/eval/scenes';

interface Case {
  name: string;
  file: string;
  exemplars: number[][]; // x1, y1, x2, y2 in image pixels
  gt: number;
  mosaic?: boolean; // 2 × 2 tile of `file`; exemplars from the top-left copy
  truth?: { x: number; y: number; r: number }[];
}

const out = resolve(process.argv[2] ?? 'geco2-cases');
mkdirSync(out, { recursive: true });
const cases: Case[] = [];

const cv = await loadCv();
for (const spec of SCENES) {
  const s = makeScene(cv, spec);
  const { width, height, data } = s.image;
  const rgb = new Uint8Array(width * height * 3);
  for (let i = 0; i < data.length; i++) rgb[3 * i] = rgb[3 * i + 1] = rgb[3 * i + 2] = data[i]!;
  const file = join(out, `${spec.name}.png`);
  writePng(file, rgb, width, height);
  // Same exemplars as the LOCA eval: the 3 true objects nearest the image centre.
  const near = [...s.truth].sort((a, b) => Math.hypot(a.x - width / 2, a.y - height / 2) - Math.hypot(b.x - width / 2, b.y - height / 2)).slice(0, 3);
  cases.push({
    name: spec.name,
    file,
    exemplars: near.map((c) => [c.x - c.r, c.y - c.r, c.x + c.r, c.y + c.r]),
    gt: s.truth.length,
    truth: s.truth.map((c) => ({ x: c.x, y: c.y, r: c.r })),
  });
}

const fsc = join(process.cwd(), 'tests', 'eval', '.data', 'fsc147');
if (existsSync(join(fsc, 'annotations.json'))) {
  const ann = JSON.parse(readFileSync(join(fsc, 'annotations.json'), 'utf8')) as Record<string, { class: string; count: number; boxes: number[][][] }>;
  const box = (b: number[][]) => {
    const xs = b.map((p) => p[0]!), ys = b.map((p) => p[1]!);
    return [Math.min(...xs), Math.min(...ys), Math.max(...xs), Math.max(...ys)];
  };
  for (const [n, a] of Object.entries(ann)) {
    cases.push({ name: `${n.replace('.jpg', '')} (${a.class})`, file: join(fsc, n), exemplars: a.boxes.map(box), gt: a.count });
  }
  for (const [n, a] of Object.entries(ann)) {
    if (a.class !== 'tree logs') continue;
    cases.push({ name: `${n.replace('.jpg', '')} ×4`, file: join(fsc, n), exemplars: a.boxes.map(box), gt: 4 * a.count, mosaic: true });
  }
}

writeFileSync(join(out, 'cases.json'), JSON.stringify(cases));
console.log(`${cases.length} cases in ${out}`);
process.exit(0);
