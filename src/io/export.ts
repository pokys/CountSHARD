/**
 * Browser exports: annotated PNG, downloads, data URLs.
 */
import type { Project } from '../model/types';
import { computeStats, numbering } from '../state/project';
import { drawAnnotations } from './draw';
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
  drawAnnotations(ctx, p.objects, p.groups, {
    pxPerUnit: 1 / Math.max(1, w / 1600),
    numbers: numbering(p.objects),
    selectedId: null,
    showRejected: false,
  });
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
