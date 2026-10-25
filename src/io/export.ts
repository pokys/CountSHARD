/**
 * Browser exports: annotated PNG, downloads, data URLs.
 */
import { type AiDensityResult, type AiDetectionResult, type Project, inCountArea } from '../model/types';
import { GROUP_COLORS, computeStats, numbering, visibleInArea } from '../state/project';
import { decodeDensity, roundEstimate, summarize } from '../ai/estimate';
import { asCountedObjects, boxesOf, classGrouping, countedDetections, shownDetections } from '../ai/geco2';
import { heatRgba } from '../ai/heatmap';
import { drawAiOverlay, drawAnnotations, drawCountArea } from './draw';
import { canvasToBlob } from './image';

export interface LegendText {
  title: string;
  total: (n: number) => string;
  group: (label: string, count: number, diameter: string) => string;
  footer: string;
}

/** Renders the project image with numbered annotations and a legend strip below it. */
export async function renderAnnotatedPng(base: HTMLCanvasElement, p: Project, legend: LegendText, formatDiameter: (px: number, mm: number | null) => string): Promise<Blob> {
  const stats = computeStats(p);
  const w = base.width;
  const unit = Math.max(1, w / 1200); // legend scales with image width
  const lineH = 26 * unit;
  const lines = 2 + Math.ceil(stats.groups.length / 3);
  const legendH = Math.round(lineH * lines + 24 * unit);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = base.height + legendH;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(base, 0, 0);
  drawAnnotations(ctx, visibleInArea(p).objects, p.groups, {
    pxPerUnit: 1 / Math.max(1, w / 1600),
    numbers: numbering(p.objects, p.countArea ?? null),
    selectedId: null,
    showRejected: false,
  });
  if (p.countArea) drawCountArea(ctx, p.countArea, base.width, base.height, 1 / Math.max(1, w / 1600));
  // Legend
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, base.height, w, legendH);
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  let y = base.height + 12 * unit + lineH / 2;
  const x0 = 16 * unit;
  ctx.fillStyle = '#ffffff';
  ctx.font = `700 ${20 * unit}px system-ui, sans-serif`;
  ctx.fillText(`${legend.title} — ${legend.total(stats.total)}`, x0, y);
  y += lineH;
  ctx.font = `500 ${16 * unit}px system-ui, sans-serif`;
  const colW = (w - 2 * x0) / 3;
  stats.groups.forEach((g, i) => {
    const cx = x0 + (i % 3) * colW;
    const cy = y + Math.floor(i / 3) * lineH;
    ctx.fillStyle = g.group.color;
    ctx.beginPath();
    ctx.arc(cx + 8 * unit, cy, 7 * unit, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(legend.group(g.group.label, g.count, formatDiameter(g.medianDiameterPx, g.medianDiameterMm)), cx + 22 * unit, cy);
  });
  y += Math.ceil(stats.groups.length / 3) * lineH;
  ctx.fillStyle = '#94a3b8';
  ctx.font = `400 ${13 * unit}px system-ui, sans-serif`;
  ctx.fillText(legend.footer, x0, y);
  return canvasToBlob(canvas, 'image/png');
}

/** Heat-map image of a Mode B result (one pixel per density cell). */
export function heatCanvas(res: AiDensityResult): HTMLCanvasElement {
  const c = document.createElement('canvas');
  c.width = res.gridW;
  c.height = res.gridH;
  const rgba = heatRgba(decodeDensity(res.density), res.gridW, res.gridH);
  c.getContext('2d')!.putImageData(new ImageData(rgba, res.gridW, res.gridH), 0, 0);
  return c;
}

export interface AiLegendText {
  title: string;
  total: (n: number) => string;
  corrected: (raw: number, regions: number) => string;
  note: string;
  footer: string;
  regionLabel: (user: number, model: string) => string;
  fmt: (v: number, digits?: number) => string;
}

/**
 * Mode B PNG: photo + heat map + exemplar boxes + corrected regions, and a legend that
 * states the number is an estimate. No per-object marks are drawn.
 */
export interface AiDetectionLegendText {
  title: string;
  total: (n: number) => string;
  footer: string;
  /** Label of a class of a multi-class result (steel profiles). */
  classLabel?: (cls: number) => string;
}

/** Mode B with GeCo2: photo + numbered detection boxes + exemplars, legend with the count and the model. */
export async function renderAiDetectionsPng(base: HTMLCanvasElement, p: Project, legend: AiDetectionLegendText): Promise<Blob> {
  const res = p.ai!.result as AiDetectionResult;
  const area = p.countArea ?? null;
  const list = shownDetections(res);
  const objects = asCountedObjects(list);
  const countedList = countedDetections(list, area);
  const counted = countedList.length;
  const grouping =
    res.classes && legend.classLabel
      ? classGrouping(res.objects, countedList.map((d) => d.id), legend.classLabel, GROUP_COLORS)
      : { groups: [], byId: new Map<string, string>() };
  const groupCounts = grouping.groups.map((g) => ({ g, n: [...grouping.byId.values()].filter((id) => id === g.id).length }));
  const w = base.width;
  const unit = Math.max(1, w / 1200);
  const lineH = 26 * unit;
  const groupLines = Math.ceil(groupCounts.length / 3);
  const legendH = Math.round(lineH * (2 + groupLines) + 24 * unit);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = base.height + legendH;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(base, 0, 0);
  const ppu = 1 / Math.max(1, w / 1600);
  const shown = objects.filter((o) => inCountArea(area, o.x, o.y)).map((o) => ({ ...o, groupId: grouping.byId.get(o.id) ?? null }));
  drawAnnotations(ctx, shown, grouping.groups, {
    pxPerUnit: ppu,
    numbers: numbering(objects, area),
    selectedId: null,
    showRejected: false,
    boxes: boxesOf(res.objects),
  });
  drawAiOverlay(ctx, { heat: null, heatWidth: 0, heatHeight: 0, exemplars: res.exemplarsUsed, regions: [] }, ppu);
  if (area) drawCountArea(ctx, area, base.width, base.height, ppu);
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, base.height, w, legendH);
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  const x0 = 16 * unit;
  let y = base.height + 12 * unit + lineH / 2;
  ctx.fillStyle = '#ffffff';
  ctx.font = `700 ${20 * unit}px system-ui, sans-serif`;
  ctx.fillText(`${legend.title} — ${legend.total(counted)}`, x0, y);
  y += lineH;
  ctx.font = `500 ${16 * unit}px system-ui, sans-serif`;
  const colW = (w - 2 * x0) / 3;
  groupCounts.forEach(({ g, n }, i) => {
    const cx = x0 + (i % 3) * colW;
    const cy = y + Math.floor(i / 3) * lineH;
    ctx.fillStyle = g.color;
    ctx.beginPath();
    ctx.arc(cx + 8 * unit, cy, 7 * unit, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#e2e8f0';
    ctx.fillText(`${g.label}: ${legend.total(n)}`, cx + 22 * unit, cy);
  });
  y += groupLines * lineH;
  ctx.fillStyle = '#94a3b8';
  ctx.font = `400 ${13 * unit}px system-ui, sans-serif`;
  ctx.fillText(legend.footer, x0, y);
  return canvasToBlob(canvas, 'image/png');
}

export async function renderAiPng(base: HTMLCanvasElement, p: Project, legend: AiLegendText): Promise<Blob> {
  const ai = p.ai!;
  const res = ai.result as AiDensityResult;
  const sum = summarize(ai, p.countArea ?? null)!;
  const w = base.width;
  const unit = Math.max(1, w / 1200);
  const lineH = 26 * unit;
  const x0 = 16 * unit;
  const fonts = { title: `700 ${20 * unit}px system-ui, sans-serif`, body: `500 ${16 * unit}px system-ui, sans-serif`, small: `400 ${13 * unit}px system-ui, sans-serif` };
  const measure = document.createElement('canvas').getContext('2d')!;
  const wrap = (text: string, font: string) => {
    measure.font = font;
    const out: string[] = [];
    let line = '';
    for (const word of text.split(' ')) {
      const next = line ? `${line} ${word}` : word;
      if (line && measure.measureText(next).width > w - 2 * x0) {
        out.push(line);
        line = word;
      } else line = next;
    }
    if (line) out.push(line);
    return out;
  };
  const lines: { text: string; font: string; color: string }[] = [
    ...wrap(`${legend.title} — ${legend.total(roundEstimate(sum.corrected))}`, fonts.title).map((text) => ({ text, font: fonts.title, color: '#ffffff' })),
    ...(ai.corrections.length ? wrap(legend.corrected(roundEstimate(sum.raw), ai.corrections.length), fonts.body) : []).map((text) => ({ text, font: fonts.body, color: '#e2e8f0' })),
    ...wrap(legend.note, fonts.body).map((text) => ({ text, font: fonts.body, color: '#e2e8f0' })),
    ...wrap(legend.footer, fonts.small).map((text) => ({ text, font: fonts.small, color: '#94a3b8' })),
  ];
  const legendH = Math.round(lineH * lines.length + 24 * unit);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = base.height + legendH;
  const ctx = canvas.getContext('2d')!;
  ctx.drawImage(base, 0, 0);
  drawAiOverlay(
    ctx,
    {
      heat: heatCanvas(res),
      heatWidth: res.gridW * res.cellPx,
      heatHeight: res.gridH * res.cellPx,
      exemplars: res.exemplarsUsed,
      // Only corrections that count (inside the counting area); sums looked up by id.
      regions: ai.corrections.flatMap((c) => {
        const r = sum.regions.find((q) => q.id === c.id);
        return r ? [{ ...c, label: legend.regionLabel(c.userCount, legend.fmt(r.model, 1)) }] : [];
      }),
    },
    1 / Math.max(1, w / 1600),
  );
  if (p.countArea) drawCountArea(ctx, p.countArea, base.width, base.height, 1 / Math.max(1, w / 1600));
  ctx.fillStyle = '#0f172a';
  ctx.fillRect(0, base.height, w, legendH);
  ctx.textBaseline = 'middle';
  ctx.textAlign = 'left';
  let y = base.height + 12 * unit + lineH / 2;
  for (const l of lines) {
    ctx.font = l.font;
    ctx.fillStyle = l.color;
    ctx.fillText(l.text, x0, y);
    y += lineH;
  }
  return canvasToBlob(canvas, 'image/png');
}

export function downloadBlob(blob: Blob, filename: string) {
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10000);
}

export function blobToDataUrl(blob: Blob): Promise<string> {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result));
    r.onerror = () => reject(r.error);
    r.readAsDataURL(blob);
  });
}

export async function dataUrlToBlob(url: string): Promise<Blob> {
  return (await fetch(url)).blob();
}

export function safeFilename(name: string): string {
  return name.replace(/[\\/:*?"<>|]+/g, '_').slice(0, 80) || 'countshard';
}
