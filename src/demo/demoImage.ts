/**
 * Generates a synthetic demo photo (two pipe sizes, uneven light) so the app can be
 * tried without a real photograph. Deterministic for a given seed.
 */
export function makeDemoImage(seed = 7): Promise<Blob> {
  let s = seed;
  const rand = () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296;
  };
  const w = 1400;
  const h = 1000;
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d')!;
  ctx.fillStyle = '#2a2622';
  ctx.fillRect(0, 0, w, h);
  const placed: { x: number; y: number; r: number }[] = [];
  const sizes = [26, 36];
  for (let i = 0; i < 6000 && placed.length < 230; i++) {
    const r = sizes[rand() < 0.65 ? 0 : 1]! * (1 + (rand() - 0.5) * 0.04);
    const cx = w / 2 + (rand() - 0.5) * (w - 240);
    const cy = h / 2 + (rand() - 0.5) * (h - 160);
    // Roughly elliptical bundle outline.
    if (((cx - w / 2) / (w * 0.43)) ** 2 + ((cy - h / 2) / (h * 0.42)) ** 2 > 1) continue;
    if (placed.some((p) => Math.hypot(p.x - cx, p.y - cy) < p.r + r + 1)) continue;
    placed.push({ x: cx, y: cy, r });
  }
  for (const p of placed) {
    const light = 0.75 + 0.25 * (1 - p.x / w);
    const base = (150 + rand() * 70) * light;
    // Contact shadow
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r + 2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(0,0,0,0.55)';
    ctx.fill();
    // Wall (cut face) with metallic gradient
    const g = ctx.createRadialGradient(p.x - p.r * 0.3, p.y - p.r * 0.3, p.r * 0.2, p.x, p.y, p.r);
    g.addColorStop(0, `rgb(${base + 25},${base + 22},${base + 18})`);
    g.addColorStop(1, `rgb(${base - 20},${base - 22},${base - 26})`);
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
    ctx.fillStyle = g;
    ctx.fill();
    // Bore
    ctx.beginPath();
    ctx.arc(p.x, p.y, p.r * 0.74, 0, Math.PI * 2);
    ctx.fillStyle = `rgb(${30 + rand() * 25},${28 + rand() * 20},${26 + rand() * 18})`;
    ctx.fill();
  }
  // Sensor noise
  const img = ctx.getImageData(0, 0, w, h);
  for (let i = 0; i < img.data.length; i += 4) {
    const n = (rand() - 0.5) * 14;
    img.data[i] = img.data[i]! + n;
    img.data[i + 1] = img.data[i + 1]! + n;
    img.data[i + 2] = img.data[i + 2]! + n;
  }
  ctx.putImageData(img, 0, 0);
  return new Promise((resolve, reject) => c.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob'))), 'image/png'));
}
