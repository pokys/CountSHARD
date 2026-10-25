/**
 * DOM-free serialisation: CSV results and JSON project files (testable in Node).
 */
import { type Project, isCounted, isDetectionResult } from '../model/types';
import { computeStats, numbering } from '../state/project';
import { decodeDensity, summarize } from '../ai/estimate';
import { asCountedObjects, countedDetections } from '../ai/geco2';

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
  const numbers = numbering(p.objects, p.countArea ?? null);
  const rows = p.objects
    .filter((o) => isCounted(o) && numbers.has(o.id))
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
  const sum = ai ? summarize(ai, p.countArea ?? null) : null;
  const sep = o.decimal === ',' ? ';' : ',';
  const num = (v: number | null | undefined, digits = 1) => (v === null || v === undefined || Number.isNaN(v) ? '' : v.toFixed(digits).replace('.', o.decimal));
  const h = o.headers;
  const rows: string[][] = [[h.type, h.region, h.x, h.y, h.w, h.h, h.model, h.user, h.result]];
  if (ai && sum) {
    rows.push([o.rows.whole, '', '', '', num(p.image.width, 0), num(p.image.height, 0), num(sum.raw), '', String(Math.round(sum.raw))]);
    // Only corrections that count (inside the counting area, if any); sums looked up by id.
    ai.corrections.forEach((c, i) => {
      const r = sum.regions.find((q) => q.id === c.id);
      if (r) rows.push([o.rows.regionRow, String(i + 1), num(c.x, 0), num(c.y, 0), num(c.w, 0), num(c.h, 0), num(r.model), String(c.userCount), '']);
    });
    rows.push([o.rows.corrected, '', '', '', '', '', '', '', String(Math.round(sum.corrected))]);
    ai.result!.exemplarsUsed.forEach((e, i) => rows.push([o.rows.exemplarRow, String(i + 1), num(e.x, 0), num(e.y, 0), num(e.w, 0), num(e.h, 0), '', '', '']));
  }
  const esc = (s: string) => (s.includes(sep) || /["\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  return '﻿' + rows.map((r) => r.map(esc).join(sep)).join('\r\n') + '\r\n';
}

export interface AiDetectionCsvLabels {
  decimal: ',' | '.';
  headers: { number: string; x: string; y: string; w: string; h: string; source: string; review: string; score: string; model: string; type?: string };
  sourceLabels: Record<'auto' | 'manual', string>;
  reviewLabels: Record<'unreviewed' | 'confirmed', string>;
  /** Names of the classes of a multi-class model (steel profiles), by class index. */
  classLabels?: string[];
}

/**
 * Mode B with a detection model (GeCo2): one row per counted object (box in project pixels,
 * numbered like circle mode), the model on every row so a pasted extract stays traceable.
 */
export function toAiDetectionsCsv(p: Project, o: AiDetectionCsvLabels): string {
  const res = p.ai?.result;
  const sep = o.decimal === ',' ? ';' : ',';
  const num = (v: number | null | undefined, digits = 1) => (v === null || v === undefined || Number.isNaN(v) ? '' : v.toFixed(digits).replace('.', o.decimal));
  const h = o.headers;
  const typed = isDetectionResult(res) && !!res.classes;
  const rows: string[][] = [[h.number, h.x, h.y, h.w, h.h, ...(typed ? [h.type ?? 'Type'] : []), h.source, h.review, h.score, h.model]];
  const typeName = (c: number | undefined) => (c === undefined ? '' : (o.classLabels?.[c] ?? (isDetectionResult(res) ? res.classes?.[c] : undefined) ?? String(c)));
  if (isDetectionResult(res)) {
    const area = p.countArea ?? null;
    const numbers = numbering(asCountedObjects(res.objects), area);
    countedDetections(res.objects, area)
      .sort((a, b) => numbers.get(a.id)! - numbers.get(b.id)!)
      .forEach((d) =>
        rows.push([
          String(numbers.get(d.id)),
          num(d.x + d.w / 2),
          num(d.y + d.h / 2),
          num(d.w),
          num(d.h),
          ...(typed ? [typeName(d.cls)] : []),
          o.sourceLabels[d.source],
          o.reviewLabels[d.review === 'confirmed' ? 'confirmed' : 'unreviewed'],
          num(d.score, 2),
          res.model,
        ]),
      );
  }
  const esc = (s: string) => (s.includes(sep) || /["\r\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  return '﻿' + rows.map((r) => r.map(esc).join(sep)).join('\r\n') + '\r\n';
}

/** Machine-readable Mode B detection summary in exported project files. */
export interface DetectionFileSummary {
  kind: 'detections';
  /** False: individually detected objects, reviewed by the user. */
  isEstimate: false;
  model: string;
  engine: string;
  count: number;
  auto: number;
  manual: number;
  /** Multi-class model: counted items per class name. */
  byClass?: Record<string, number>;
}

export function detectionSummary(p: Project): DetectionFileSummary | null {
  const res = p.ai?.result;
  if (!isDetectionResult(res)) return null;
  const c = countedDetections(res.objects, p.countArea ?? null);
  const out: DetectionFileSummary = { kind: 'detections', isEstimate: false, model: res.model, engine: res.engine, count: c.length, auto: c.filter((d) => d.source === 'auto').length, manual: c.filter((d) => d.source === 'manual').length };
  if (res.classes) {
    const by: Record<string, number> = {};
    for (const d of c) {
      const k = d.cls === undefined ? 'unknown' : (res.classes[d.cls] ?? String(d.cls));
      by[k] = (by[k] ?? 0) + 1;
    }
    out.byClass = by;
  }
  return out;
}

/** Machine-readable Mode B summary in exported project files. */
export interface EstimateFileSummary {
  kind: 'density-estimate';
  /** Always true: the number is a model estimate, not a count of identified objects. */
  isEstimate: true;
  model: string;
  engine: string;
  rawEstimate: number;
  correctedEstimate: number;
  regions: { x: number; y: number; w: number; h: number; modelEstimate: number; userCount: number }[];
}

export function estimateSummary(p: Project): EstimateFileSummary | null {
  const ai = p.ai;
  const sum = ai ? summarize(ai, p.countArea ?? null) : null;
  if (!ai || !sum) return null;
  return {
    kind: 'density-estimate',
    isEstimate: true,
    model: ai.result!.model,
    engine: ai.result!.engine,
    rawEstimate: sum.raw,
    correctedEstimate: sum.corrected,
    regions: ai.corrections.flatMap((c) => {
      const r = sum.regions.find((q) => q.id === c.id);
      return r ? [{ x: c.x, y: c.y, w: c.w, h: c.h, modelEstimate: r.model, userCount: c.userCount }] : [];
    }),
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
    /** Mode B: the AI estimate (LOCA; present only when one was computed). */
    estimate?: EstimateFileSummary;
    /** Mode B: individually detected objects (GeCo2). */
    detections?: DetectionFileSummary;
  };
}

export function toProjectFile(p: Project, imageDataUrl: string): ProjectFile {
  const s = computeStats(p);
  const est = estimateSummary(p);
  const det = detectionSummary(p);
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
      ...(det ? { detections: det } : {}),
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
  if (isDetectionResult(r)) {
    if (!Array.isArray(r.objects) || typeof r.model !== 'string' || !(r.scale > 0)) throw new ProjectFileError('ai');
    const states = ['unreviewed', 'confirmed', 'rejected'];
    if (!r.objects.every((d) => box(d) && (d.source === 'auto' || d.source === 'manual') && states.includes(d.review))) throw new ProjectFileError('ai');
    if (r.classes !== undefined && (!Array.isArray(r.classes) || !r.classes.every((c) => typeof c === 'string'))) throw new ProjectFileError('ai');
    if (!r.objects.every((d) => d.cls === undefined || (Number.isInteger(d.cls) && d.cls >= 0))) throw new ProjectFileError('ai');
  } else if (r) {
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
