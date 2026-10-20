// Renders the PNG app icons (Android/Chrome install) from the same geometry as public/icon.svg.
// Run: node scripts/make-icons.mjs
import { deflateSync } from 'node:zlib';
import { writeFileSync } from 'node:fs';

const BG = [15, 23, 42];
const GREEN = [34, 197, 94];
const BLUE = [59, 130, 246];
// In the 64×64 icon coordinate system (see public/icon.svg).
const RINGS = [
  { x: 21, y: 23, c: GREEN },
  { x: 43, y: 23, c: GREEN },
  { x: 32, y: 43, c: BLUE },
];

/** Colour of one icon-space point; `maskable` = full square background, content scaled to 70 %. */
function sample(u, v, maskable) {
  if (maskable) {
    u = (u - 32) / 0.7 + 32;
    v = (v - 32) / 0.7 + 32;
  } else {
    // Rounded square, radius 14.
    const r = 14;
    const cx = Math.min(Math.max(u, r), 64 - r);
    const cy = Math.min(Math.max(v, r), 64 - r);
    if (Math.hypot(u - cx, v - cy) > r) return null;
  }
  for (const g of RINGS) {
    const d = Math.hypot(u - g.x, v - g.y);
    if (Math.abs(d - 9) <= 2 || d <= 3) return g.c;
  }
  return BG;
}

function render(size, maskable) {
  const SS = 4; // supersampling per axis
  const rgba = Buffer.alloc(size * size * 4);
  for (let y = 0; y < size; y++)
    for (let x = 0; x < size; x++) {
      let r = 0, g = 0, b = 0, a = 0;
      for (let sy = 0; sy < SS; sy++)
        for (let sx = 0; sx < SS; sx++) {
          const c = sample(((x + (sx + 0.5) / SS) / size) * 64, ((y + (sy + 0.5) / SS) / size) * 64, maskable);
          if (!c) continue;
          r += c[0]; g += c[1]; b += c[2]; a++;
        }
      const i = (y * size + x) * 4;
      if (a) {
        rgba[i] = Math.round(r / a);
        rgba[i + 1] = Math.round(g / a);
        rgba[i + 2] = Math.round(b / a);
      }
      rgba[i + 3] = Math.round((255 * a) / (SS * SS));
    }
  return png(rgba, size);
}

const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();
const crc32 = (buf) => {
  let c = 0xffffffff;
  for (const x of buf) c = CRC[(c ^ x) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
};
function chunk(type, data) {
  const len = Buffer.alloc(4);
  len.writeUInt32BE(data.length);
  const td = Buffer.concat([Buffer.from(type, 'ascii'), data]);
  const crc = Buffer.alloc(4);
  crc.writeUInt32BE(crc32(td));
  return Buffer.concat([len, td, crc]);
}
function png(rgba, size) {
  const raw = Buffer.alloc((size * 4 + 1) * size);
  for (let y = 0; y < size; y++) rgba.copy(raw, y * (size * 4 + 1) + 1, y * size * 4, (y + 1) * size * 4);
  const ihdr = Buffer.alloc(13);
  ihdr.writeUInt32BE(size, 0);
  ihdr.writeUInt32BE(size, 4);
  ihdr[8] = 8;
  ihdr[9] = 6; // RGBA
  return Buffer.concat([Buffer.from([0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw)), chunk('IEND', Buffer.alloc(0))]);
}

writeFileSync('public/icon-192.png', render(192, false));
writeFileSync('public/icon-512.png', render(512, false));
writeFileSync('public/icon-maskable-512.png', render(512, true));
writeFileSync('public/apple-touch-icon.png', render(180, true));
console.log('icons written');
