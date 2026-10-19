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
}

export function drawAnnotations(ctx: CanvasRenderingContext2D, objects: CountedObject[], groups: SizeGroup[], o: DrawOptions) {
  const colors = new Map(groups.map((g) => [g.id, g.color]));
  const u = 1 / o.pxPerUnit;
  for (const obj of objects) {
    const counted = isCounted(obj);
    const candidate = isCandidate(obj);
    if (!counted && !candidate && !o.showRejected) continue;
    const color = candidate ? REVIEW_COLOR : counted ? (obj.groupId && colors.get(obj.groupId)) || o.defaultColor || '#22c55e' : RESERVED_COLORS.rejected;
    const selected = obj.id === o.selectedId;
    ctx.beginPath();
    ctx.arc(obj.x, obj.y, obj.r, 0, Math.PI * 2);
    ctx.lineWidth = Math.max(u * (selected ? 3.5 : 2), obj.r * 0.06);
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
        ctx.beginPath();
        ctx.arc(obj.x, obj.y, obj.r + 4 * u, 0, Math.PI * 2);
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
      ctx.beginPath();
      ctx.arc(obj.x, obj.y, obj.r + 4 * u, 0, Math.PI * 2);
      ctx.lineWidth = 2 * u;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
    }
    const n = o.numbers?.get(obj.id);
    const fontPx = Math.min(obj.r * 0.9, Math.max(obj.r * 0.55, 11 * u));
    if (n !== undefined && fontPx * o.pxPerUnit >= 7) {
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
