/**
 * DOM-free serialisation: CSV results and JSON project files (testable in Node).
 */
import { type Project, isCounted } from '../model/types';
import { computeStats, numbering } from '../state/project';
import { decodeDensity, summarize } from '../ai/estimate';

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

export interface AiCsvLabels {
  decimal: ',' | '.';
  headers: { type: string; region: string; x: string; y: string; w: string; h: string; model: string; user: string; result: string };
  rows: { whole: string; corrected: string; regionRow: string; exemplarRow: string };
}

/**
 * Mode B CSV: there are no individual objects, only the model's estimate, the user's region
 * corrections and the corrected estimate. Every estimate row says so in its type column.
 */
export function toAiCsv(p: Project, o: AiCsvLabels): string {
  const ai = p.ai;
  const sum = ai ? summarize(ai) : null;
  const sep = o.decimal === ',' ? ';' : ',';
  const num = (v: number | null | undefined, digits = 1) => (v === null || v === undefined || Number.isNaN(v) ? '' : v.toFixed(digits).replace('.', o.decimal));
  const h = o.headers;
  const rows: string[][] = [[h.type, h.region, h.x, h.y, h.w, h.h, h.model, h.user, h.result]];
  if (ai && sum) {
    rows.push([o.rows.whole, '', '', '', num(p.image.width, 0), num(p.image.height, 0), num(sum.raw), '', String(Math.round(sum.raw))]);
    ai.corrections.forEach((c, i) => {
      const r = sum.regions[i]!;
      rows.push([o.rows.regionRow, String(i + 1), num(c.x, 0), num(c.y, 0), num(c.w, 0), num(c.h, 0), num(r.model), String(c.userCount), '']);
    });
    rows.push([o.rows.corrected, '', '', '', '', '', '', '', String(Math.round(sum.corrected))]);
    ai.result!.exemplarsUsed.forEach((e, i) => rows.push([o.rows.exemplarRow, String(i + 1), num(e.x, 0), num(e.y, 0), num(e.w, 0), num(e.h, 0), '', '', '']));
  }
  const esc = (s: string) => (s.includes(sep) || /["\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  return '﻿' + rows.map((r) => r.map(esc).join(sep)).join('\r\n') + '\r\n';
}

/** Machine-readable Mode B summary in exported project files. */
export interface EstimateFileSummary {
  kind: 'density-estimate';
  /** Always true: the number is a model estimate, not a count of identified objects. */
  isEstimate: true;
  model: string;
  rawEstimate: number;
  correctedEstimate: number;
  regions: { x: number; y: number; w: number; h: number; modelEstimate: number; userCount: number }[];
}

export function estimateSummary(p: Project): EstimateFileSummary | null {
  const ai = p.ai;
  const sum = ai ? summarize(ai) : null;
  if (!ai || !sum) return null;
  return {
    kind: 'density-estimate',
    isEstimate: true,
    model: ai.result!.model,
    rawEstimate: sum.raw,
    correctedEstimate: sum.corrected,
    regions: ai.corrections.map((c, i) => ({ x: c.x, y: c.y, w: c.w, h: c.h, modelEstimate: sum.regions[i]!.model, userCount: c.userCount })),
  };
}

export interface ProjectFile {
  format: 'countshard-project';
  version: 1;
  exportedAt: string;
  project: Project;
  /** Project image as a data URL, so the file restores fully editable annotations. */
  image: string;
  summary: {
    /** Mode A: individually counted objects. */
    total: number;
    groups: { label: string; count: number; medianDiameterPx: number; medianDiameterMm: number | null }[];
    mode: 'circles' | 'ai';
    /** Mode B: the AI estimate (present only when one was computed). */
    estimate?: EstimateFileSummary;
  };
}

export function toProjectFile(p: Project, imageDataUrl: string): ProjectFile {
  const s = computeStats(p);
  const est = estimateSummary(p);
  return {
    format: 'countshard-project',
    version: 1,
    exportedAt: new Date().toISOString(),
    project: p,
    image: imageDataUrl,
    summary: {
      total: s.total,
      groups: s.groups.map((g) => ({ label: g.group.label, count: g.count, medianDiameterPx: g.medianDiameterPx, medianDiameterMm: g.medianDiameterMm })),
      mode: p.mode ?? 'circles',
      ...(est ? { estimate: est } : {}),
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
  if (p.mode !== undefined && p.mode !== 'circles' && p.mode !== 'ai') throw new ProjectFileError('mode');
  if (p.ai) validateAi(p);
  return f as ProjectFile;
}

function validateAi(p: Project) {
  const ai = p.ai!;
  const box = (b: unknown) => {
    const o = b as Record<string, unknown>;
    return !!o && typeof o.id === 'string' && ['x', 'y', 'w', 'h'].every((k) => typeof o[k] === 'number' && Number.isFinite(o[k]));
  };
  if (!Array.isArray(ai.exemplars) || !ai.exemplars.every(box)) throw new ProjectFileError('ai');
  if (!Array.isArray(ai.corrections) || !ai.corrections.every((c) => box(c) && Number.isInteger(c.userCount) && c.userCount >= 0)) throw new ProjectFileError('ai');
  const r = ai.result;
  if (r) {
    if (typeof r.density !== 'string' || !(r.gridW > 0) || !(r.gridH > 0) || !(r.cellPx > 0) || !Number.isFinite(r.rawTotal)) throw new ProjectFileError('ai');
    let data: Float32Array;
    try {
      data = decodeDensity(r.density);
    } catch {
      throw new ProjectFileError('ai');
    }
    if (data.length !== r.gridW * r.gridH) throw new ProjectFileError('ai');
  }
}
