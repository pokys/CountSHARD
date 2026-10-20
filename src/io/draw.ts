/**
 * Annotation rendering shared by the interactive viewer and the PNG export.
 * The context must already be transformed so that 1 unit = 1 project-image pixel;
 * `pxPerUnit` converts desired on-screen line widths/fonts into image units.
 */
import { RESERVED_COLORS } from '../state/project';
import { type CountedObject, type SizeGroup, isCandidate, isCounted, isMultiShotFlagged } from '../model/types';

/** Multi-shot review colour (candidates and objects seen in few photos). */
const REVIEW_COLOR = RESERVED_COLORS.review;

export interface DrawOptions {
  pxPerUnit: number;
  numbers: Map<string, number> | null;
  selectedId: string | null;
  showRejected: boolean;
  /** Colour for objects without a group. */
  defaultColor?: string;
  /** Draw a small marker for unreviewed (automatic) vs confirmed objects. */
  reviewMarks?: boolean;
  /** Objects with a box (AI detections) are drawn as rectangles of this size around their centre. */
  boxes?: Map<string, { w: number; h: number }>;
}

function shapePath(ctx: CanvasRenderingContext2D, obj: CountedObject, grow: number, box?: { w: number; h: number }) {
  ctx.beginPath();
  if (box) ctx.rect(obj.x - box.w / 2 - grow, obj.y - box.h / 2 - grow, box.w + 2 * grow, box.h + 2 * grow);
  else ctx.arc(obj.x, obj.y, obj.r + grow, 0, Math.PI * 2);
}

export function drawAnnotations(ctx: CanvasRenderingContext2D, objects: CountedObject[], groups: SizeGroup[], o: DrawOptions) {
  const colors = new Map(groups.map((g) => [g.id, g.color]));
  const u = 1 / o.pxPerUnit;
  // Numbers are shown for all objects or for none: one on-screen font size from the
  // typical object size, so small and large objects are labelled consistently.
  const radii = objects.filter(isCounted).map((x) => x.r).sort((a, b) => a - b);
  const typicalR = radii.length ? radii[Math.floor(radii.length / 2)]! : 0;
  const numberPx = Math.min(16, typicalR * o.pxPerUnit * 0.8);
  const showNumbers = !!o.numbers && numberPx >= 7;
  for (const obj of objects) {
    const counted = isCounted(obj);
    const candidate = isCandidate(obj);
    if (!counted && !candidate && !o.showRejected) continue;
    const color = candidate ? REVIEW_COLOR : counted ? (obj.groupId && colors.get(obj.groupId)) || o.defaultColor || '#22c55e' : RESERVED_COLORS.rejected;
    const selected = obj.id === o.selectedId;
    const box = o.boxes?.get(obj.id);
    shapePath(ctx, obj, 0, box);
    ctx.lineWidth = Math.max(u * (selected ? 3.5 : 2), obj.r * (box ? 0.04 : 0.06));
    ctx.strokeStyle = color;
    if (!counted) ctx.setLineDash([4 * u, 4 * u]);
    ctx.stroke();
    ctx.setLineDash([]);
    if (candidate) {
      // Multi-shot candidate: not counted, marked with "?" until the user decides.
      const fontPx = Math.max(obj.r * 0.8, 10 * u);
      ctx.font = `700 ${fontPx}px system-ui, -apple-system, Segoe UI, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.lineWidth = Math.max(fontPx / 5, 1.5 * u);
      ctx.strokeStyle = 'rgba(0,0,0,0.75)';
      ctx.strokeText('?', obj.x, obj.y);
      ctx.fillStyle = REVIEW_COLOR;
      ctx.fillText('?', obj.x, obj.y);
      if (selected) {
        shapePath(ctx, obj, 4 * u, box);
        ctx.lineWidth = 2 * u;
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      }
      continue;
    }
    if (!counted) {
      // Cross for a rejected detection.
      const d = obj.r * 0.5;
      ctx.beginPath();
      ctx.moveTo(obj.x - d, obj.y - d);
      ctx.lineTo(obj.x + d, obj.y + d);
      ctx.moveTo(obj.x + d, obj.y - d);
      ctx.lineTo(obj.x - d, obj.y + d);
      ctx.stroke();
      continue;
    }
    if (selected) {
      shapePath(ctx, obj, 4 * u, box);
      ctx.lineWidth = 2 * u;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }
    const n = showNumbers ? o.numbers!.get(obj.id) : undefined;
    const fontPx = numberPx * u;
    if (n !== undefined) {
      ctx.font = `700 ${fontPx}px system-ui, -apple-system, Segoe UI, sans-serif`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.lineWidth = Math.max(fontPx / 5, 1.5 * u);
      ctx.strokeStyle = 'rgba(0,0,0,0.75)';
      ctx.strokeText(String(n), obj.x, obj.y);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(String(n), obj.x, obj.y);
    } else {
      ctx.beginPath();
      ctx.arc(obj.x, obj.y, Math.max(1.5 * u, obj.r * 0.18), 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();
    }
    if (o.reviewMarks && isMultiShotFlagged(obj)) {
      // Seen in few of the photos: small amber triangle at the rim.
      const s = Math.max(3 * u, obj.r * 0.2);
      const cx = obj.x - obj.r * 0.7, cy = obj.y - obj.r * 0.7;
      ctx.beginPath();
      ctx.moveTo(cx, cy - s);
      ctx.lineTo(cx + s, cy + s * 0.8);
      ctx.lineTo(cx - s, cy + s * 0.8);
      ctx.closePath();
      ctx.fillStyle = REVIEW_COLOR;
      ctx.fill();
    }
    if (o.reviewMarks && obj.source === 'manual') {
      ctx.beginPath();
      ctx.arc(obj.x + obj.r * 0.7, obj.y - obj.r * 0.7, Math.max(2.5 * u, obj.r * 0.14), 0, Math.PI * 2);
      ctx.fillStyle = '#3b82f6';
      ctx.fill();
    }
  }
}

// --- Mode B (AI density estimate) overlay ----------------------------------

/** Exemplar boxes: cyan, dashed. Correction regions: white. Neither is a group/review colour. */
export const AI_COLORS = { exemplar: '#22d3ee', region: '#ffffff' } as const;

export interface AiOverlay {
  /** Heat map image covering gridW·cellPx × gridH·cellPx project pixels from the origin. */
  heat: CanvasImageSource | null;
  heatWidth: number;
  heatHeight: number;
  exemplars: { id: string; x: number; y: number; w: number; h: number }[];
  regions: { id: string; x: number; y: number; w: number; h: number; label: string }[];
  /** Rectangle being drawn. */
  draft?: { x: number; y: number; w: number; h: number; kind: 'exemplar' | 'region' } | null;
  selectedRegionId?: string | null;
  /** Exemplar being edited: drawn solid, with corner handles. */
  selectedExemplarId?: string | null;
}

function label(ctx: CanvasRenderingContext2D, text: string, x: number, y: number, u: number, color: string) {
  const fontPx = 13 * u;
  ctx.font = `700 ${fontPx}px system-ui, -apple-system, Segoe UI, sans-serif`;
  ctx.textAlign = 'left';
  ctx.textBaseline = 'top';
  const w = ctx.measureText(text).width + 8 * u;
  ctx.fillStyle = 'rgba(15,23,42,0.85)';
  ctx.fillRect(x, y, w, fontPx + 6 * u);
  ctx.fillStyle = color;
  ctx.fillText(text, x + 4 * u, y + 3 * u);
}

/** Draws the heat map, exemplar boxes and correction regions (no per-object marks: it is an estimate). */
export function drawAiOverlay(ctx: CanvasRenderingContext2D, o: AiOverlay, pxPerUnit: number) {
  const u = 1 / pxPerUnit;
  if (o.heat) {
    const smooth = ctx.imageSmoothingEnabled;
    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(o.heat, 0, 0, o.heatWidth, o.heatHeight);
    ctx.imageSmoothingEnabled = smooth;
  }
  for (const r of o.regions) {
    ctx.lineWidth = (r.id === o.selectedRegionId ? 3.5 : 2) * u;
    ctx.strokeStyle = 'rgba(0,0,0,0.6)';
    ctx.strokeRect(r.x - u, r.y - u, r.w + 2 * u, r.h + 2 * u);
    ctx.strokeStyle = AI_COLORS.region;
    ctx.strokeRect(r.x, r.y, r.w, r.h);
    label(ctx, r.label, r.x + 2 * u, r.y + 2 * u, u, AI_COLORS.region);
  }
  o.exemplars.forEach((e, i) => {
    const sel = e.id === o.selectedExemplarId;
    ctx.lineWidth = (sel ? 3 : 2.5) * u;
    if (!sel) ctx.setLineDash([6 * u, 4 * u]);
    ctx.strokeStyle = AI_COLORS.exemplar;
    ctx.strokeRect(e.x, e.y, e.w, e.h);
    ctx.setLineDash([]);
    label(ctx, `${i + 1}`, e.x, Math.max(0, e.y - 20 * u), u, AI_COLORS.exemplar);
    if (sel) {
      const hs = 7 * u;
      for (const [cx, cy] of [[e.x, e.y], [e.x + e.w, e.y], [e.x, e.y + e.h], [e.x + e.w, e.y + e.h]] as const) {
        ctx.fillStyle = '#ffffff';
        ctx.fillRect(cx - hs, cy - hs, 2 * hs, 2 * hs);
        ctx.lineWidth = 2 * u;
        ctx.strokeStyle = '#0f172a';
        ctx.strokeRect(cx - hs, cy - hs, 2 * hs, 2 * hs);
      }
    }
  });
  if (o.draft && o.draft.w > 0 && o.draft.h > 0) {
    ctx.lineWidth = 2 * u;
    ctx.setLineDash([4 * u, 4 * u]);
    ctx.strokeStyle = o.draft.kind === 'exemplar' ? AI_COLORS.exemplar : AI_COLORS.region;
    ctx.strokeRect(o.draft.x, o.draft.y, o.draft.w, o.draft.h);
    ctx.setLineDash([]);
  }
}

type AreaShape = { x: number; y: number; w: number; h: number; poly?: [number, number][] };

function areaPath(ctx: CanvasRenderingContext2D, area: AreaShape) {
  if (area.poly && area.poly.length > 2) {
    ctx.moveTo(area.poly[0]![0], area.poly[0]![1]);
    for (const [x, y] of area.poly.slice(1)) ctx.lineTo(x, y);
    ctx.closePath();
  } else {
    ctx.rect(area.x, area.y, area.w, area.h);
  }
}

/**
 * Counting area (rectangle or lasso): dims everything outside it and outlines it. The context is
 * in image units; `outside` is the region to dim (the image, or more around it on screen).
 */
export function drawCountArea(
  ctx: CanvasRenderingContext2D,
  area: AreaShape,
  imgW: number,
  imgH: number,
  pxPerUnit: number,
  opts: { outside?: { x: number; y: number; w: number; h: number }; color?: string } = {},
) {
  const o = opts.outside ?? { x: 0, y: 0, w: imgW, h: imgH };
  ctx.save();
  ctx.beginPath();
  ctx.rect(o.x, o.y, o.w, o.h);
  areaPath(ctx, area);
  ctx.fillStyle = 'rgba(2, 6, 23, 0.6)';
  ctx.fill('evenodd');
  ctx.beginPath();
  areaPath(ctx, area);
  ctx.setLineDash([8 / pxPerUnit, 6 / pxPerUnit]);
  ctx.lineWidth = 2.5 / pxPerUnit;
  ctx.strokeStyle = opts.color ?? '#facc15';
  ctx.stroke();
  ctx.restore();
}
