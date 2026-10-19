/**
 * DOM-free serialisation: CSV results and JSON project files (testable in Node).
 */
import { type Project, isCounted } from '../model/types';
import { computeStats, numbering } from '../state/project';

export interface CsvOptions {
  /** Decimal separator; Czech Excel expects a comma. Field separator is then ';'. */
  decimal: ',' | '.';
  headers: {
    number: string;
    x: string;
    y: string;
    diameterPx: string;
    diameterMm: string;
    group: string;
    source: string;
    review: string;
    score: string;
  };
  sourceLabels: Record<'auto' | 'manual', string>;
  reviewLabels: Record<'unreviewed' | 'confirmed', string>;
}

export function toCsv(p: Project, o: CsvOptions): string {
  const sep = o.decimal === ',' ? ';' : ',';
  const num = (v: number | null | undefined, digits = 1) => (v === null || v === undefined || Number.isNaN(v) ? '' : v.toFixed(digits).replace('.', o.decimal));
  const groups = new Map(p.groups.map((g) => [g.id, g]));
  const numbers = numbering(p.objects);
  const rows = p.objects
    .filter(isCounted)
    .sort((a, b) => numbers.get(a.id)! - numbers.get(b.id)!)
    .map((obj) => [
      String(numbers.get(obj.id)),
      num(obj.x),
      num(obj.y),
      num(2 * obj.r),
      p.calibration ? num(2 * obj.r * p.calibration.mmPerPx) : '',
      (obj.groupId && groups.get(obj.groupId)?.label) || '',
      o.sourceLabels[obj.source],
      o.reviewLabels[obj.review === 'confirmed' ? 'confirmed' : 'unreviewed'],
      num(obj.score, 2),
    ]);
  const h = o.headers;
  const header = [h.number, h.x, h.y, h.diameterPx, h.diameterMm, h.group, h.source, h.review, h.score];
  // Quote only what collides with the active separator (a decimal comma is fine with ';').
  const esc = (s: string) => (s.includes(sep) || /["\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  // BOM so that Excel opens UTF-8 (Czech diacritics) correctly.
  return '﻿' + [header, ...rows].map((r) => r.map(esc).join(sep)).join('\r\n') + '\r\n';
}

export interface ProjectFile {
  format: 'countshard-project';
  version: 1;
  exportedAt: string;
  project: Project;
  /** Project image as a data URL, so the file restores fully editable annotations. */
  image: string;
  summary: { total: number; groups: { label: string; count: number; medianDiameterPx: number; medianDiameterMm: number | null }[] };
}

export function toProjectFile(p: Project, imageDataUrl: string): ProjectFile {
  const s = computeStats(p);
  return {
    format: 'countshard-project',
    version: 1,
    exportedAt: new Date().toISOString(),
    project: p,
    image: imageDataUrl,
    summary: {
      total: s.total,
      groups: s.groups.map((g) => ({ label: g.group.label, count: g.count, medianDiameterPx: g.medianDiameterPx, medianDiameterMm: g.medianDiameterMm })),
    },
  };
}

export class ProjectFileError extends Error {}

/** Parses and validates a project file. Throws ProjectFileError with a short reason. */
export function parseProjectFile(text: string): ProjectFile {
  let data: unknown;
  try {
    data = JSON.parse(text);
  } catch {
    throw new ProjectFileError('json');
  }
  const f = data as Partial<ProjectFile>;
  // Files exported before the rename used 'countit-project'; accept and upgrade them.
  if (f && (f.format as string) === 'countit-project') {
    f.format = 'countshard-project';
    if (f.project) (f.project as { format: string }).format = 'countshard-project';
  }
  if (!f || f.format !== 'countshard-project') throw new ProjectFileError('format');
  if (f.version !== 1) throw new ProjectFileError('version');
  const p = f.project as Project | undefined;
  if (!p || !Array.isArray(p.objects) || !Array.isArray(p.groups) || !p.image || typeof p.image.width !== 'number') throw new ProjectFileError('structure');
  if (typeof f.image !== 'string' || !f.image.startsWith('data:image/')) throw new ProjectFileError('image');
  for (const o of p.objects) {
    if (![o.x, o.y, o.r].every((v) => typeof v === 'number' && Number.isFinite(v)) || typeof o.id !== 'string') throw new ProjectFileError('objects');
  }
  return f as ProjectFile;
}
