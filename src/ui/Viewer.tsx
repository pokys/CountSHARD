import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from 'react';
import type { CountedObject, SizeGroup } from '../model/types';
import { type AiOverlay, drawAiOverlay, drawAnnotations, drawCountArea } from '../io/draw';
import { hitTest } from '../state/project';

/** Mode A: edit/select. Mode B (AI): draw exemplar boxes, draw correction regions, or just pan. */
export type Tool = 'edit' | 'select' | 'exemplar' | 'region' | 'pan' | 'area';

export interface ViewerHandle {
  zoomBy(factor: number): void;
  fit(): void;
}

interface Props {
  image: HTMLCanvasElement;
  objects: CountedObject[];
  groups: SizeGroup[];
  numbers: Map<string, number> | null;
  selectedId: string | null;
  tool: Tool;
  showOverlay: boolean;
  showRejected: boolean;
  /** Mode B overlay. When set, the photo shows the density estimate instead of circles. */
  ai?: AiOverlay | null;
  /** Mode B: a rectangle was drawn with the exemplar or region tool (image coordinates). */
  onDrawRect?(kind: 'exemplar' | 'region' | 'area', x1: number, y1: number, x2: number, y2: number): void;
  /** Counting area (both modes): the outside is dimmed. */
  area?: { x: number; y: number; w: number; h: number; poly?: [number, number][] } | null;
  /** Area tool: the finger/mouse path, in image coordinates (lasso or diagonal). */
  onDrawArea?(points: [number, number][]): void;
  onSelectRegion?(id: string | null): void;
  onAdd(x: number, y: number): void;
  onRemove(id: string): void;
  onSelect(id: string | null): void;
  /** `final` is false while dragging (coalesced into one undo step). */
  onMove(id: string, x: number, y: number, final: boolean): void;
  onResize(id: string, r: number, final: boolean): void;
}

interface View {
  s: number; // CSS px per image px
  tx: number;
  ty: number;
}

type Gesture =
  | { kind: 'none' }
  | { kind: 'tap-or-pan'; startX: number; startY: number; view: View; t0: number; moved: boolean; hitId: string | null }
  | { kind: 'move'; id: string; dx: number; dy: number; moved: boolean }
  | { kind: 'resize'; id: string; moved: boolean }
  | { kind: 'pinch'; d0: number; cx0: number; cy0: number; view: View }
  | { kind: 'rect'; tool: 'exemplar' | 'region' | 'area'; x0: number; y0: number; x1: number; y1: number }
  | { kind: 'lasso'; pts: [number, number][] };

const TAP_SLOP = 7;

export const Viewer = forwardRef<ViewerHandle, Props>(function Viewer(props, ref) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const view = useRef<View>({ s: 1, tx: 0, ty: 0 });
  /** Scale of the "whole photo" view; at (about) this scale vertical swipes scroll the page. */
  const fitScale = useRef(1);
  const pointers = useRef(new Map<number, { x: number; y: number }>());
  const gesture = useRef<Gesture>({ kind: 'none' });
  const propsRef = useRef(props);
  propsRef.current = props;
  const frame = useRef(0);

  const draw = useCallback(() => {
    frame.current = 0;
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;
    const p = propsRef.current;
    const dpr = window.devicePixelRatio || 1;
    const cw = wrap.clientWidth;
    const ch = wrap.clientHeight;
    if (canvas.width !== Math.round(cw * dpr) || canvas.height !== Math.round(ch * dpr)) {
      canvas.width = Math.round(cw * dpr);
      canvas.height = Math.round(ch * dpr);
    }
    const ctx = canvas.getContext('2d')!;
    // Not zoomed in: let a vertical swipe scroll the page (phones), otherwise the photo
    // would capture every swipe. Zoomed in: all gestures pan/zoom the photo.
    const zoomed = view.current.s > fitScale.current * 1.02;
    // Drawing a box must not scroll the page either.
    const drawing = p.tool === 'area' || (!!p.ai && (p.tool === 'exemplar' || p.tool === 'region'));
    canvas.style.touchAction = zoomed || drawing ? 'none' : 'pan-y';
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const v = view.current;
    ctx.setTransform(v.s * dpr, 0, 0, v.s * dpr, v.tx * dpr, v.ty * dpr);
    ctx.imageSmoothingEnabled = v.s < 2;
    ctx.drawImage(p.image, 0, 0);
    if (p.ai) {
      if (p.showOverlay) {
        const g = gesture.current;
        const draft =
          g.kind === 'rect' && g.tool !== 'area'
            ? { x: Math.min(g.x0, g.x1), y: Math.min(g.y0, g.y1), w: Math.abs(g.x1 - g.x0), h: Math.abs(g.y1 - g.y0), kind: g.tool }
            : null;
        drawAiOverlay(ctx, { ...p.ai, draft }, v.s);
      }
    } else if (p.showOverlay) {
      drawAnnotations(ctx, p.objects, p.groups, {
        pxPerUnit: v.s,
        numbers: p.numbers,
        selectedId: p.selectedId,
        showRejected: p.showRejected,
        reviewMarks: true,
      });
      if (p.tool === 'select' && p.selectedId) {
        const o = p.objects.find((x) => x.id === p.selectedId);
        if (o) {
          // Resize handle on the rim (east).
          ctx.beginPath();
          ctx.arc(o.x + o.r, o.y, 7 / v.s, 0, Math.PI * 2);
          ctx.fillStyle = '#ffffff';
          ctx.fill();
          ctx.lineWidth = 2 / v.s;
          ctx.strokeStyle = '#0f172a';
          ctx.stroke();
        }
      }
    }
    // Counting area: dim everything outside it; while drawing, show the stroke.
    const g = gesture.current;
    const { width: iw, height: ih } = p.image;
    const outside = { x: -iw, y: -ih, w: iw * 3, h: ih * 3 };
    if (g.kind === 'lasso' && g.pts.length > 1) {
      ctx.save();
      ctx.beginPath();
      ctx.moveTo(g.pts[0]![0], g.pts[0]![1]);
      for (const [x, y] of g.pts.slice(1)) ctx.lineTo(x, y);
      ctx.setLineDash([8 / v.s, 6 / v.s]);
      ctx.lineWidth = 2.5 / v.s;
      ctx.strokeStyle = '#ffffff';
      ctx.stroke();
      ctx.restore();
    } else if (p.area) {
      drawCountArea(ctx, p.area, iw, ih, v.s, { outside });
    }
  }, []);

  const requestDraw = useCallback(() => {
    if (!frame.current) frame.current = requestAnimationFrame(draw);
  }, [draw]);

  const fit = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const { width: iw, height: ih } = propsRef.current.image;
    const s = Math.min(wrap.clientWidth / iw, wrap.clientHeight / ih) * 0.98;
    fitScale.current = s;
    view.current = { s, tx: (wrap.clientWidth - iw * s) / 2, ty: (wrap.clientHeight - ih * s) / 2 };
    requestDraw();
  }, [requestDraw]);

  const zoomAt = useCallback(
    (factor: number, cx: number, cy: number) => {
      const v = view.current;
      const wrap = wrapRef.current!;
      const minS = Math.min(wrap.clientWidth / props.image.width, wrap.clientHeight / props.image.height) * 0.5;
      const s = Math.min(16, Math.max(minS, v.s * factor));
      const f = s / v.s;
      view.current = { s, tx: cx - (cx - v.tx) * f, ty: cy - (cy - v.ty) * f };
      requestDraw();
    },
    [props.image, requestDraw],
  );

  useImperativeHandle(ref, () => ({
    zoomBy: (f) => {
      const wrap = wrapRef.current!;
      zoomAt(f, wrap.clientWidth / 2, wrap.clientHeight / 2);
    },
    fit,
  }));

  // Fit when a new image arrives; redraw on any prop change.
  useEffect(() => fit(), [props.image, fit]);
  useEffect(() => requestDraw());
  useEffect(() => {
    const wrap = wrapRef.current!;
    const ro = new ResizeObserver(() => {
      // Keep showing the whole photo when the area resizes, unless the user zoomed in.
      if (view.current.s <= fitScale.current * 1.02) fit();
      else requestDraw();
    });
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [requestDraw, fit]);

  const toImage = (clientX: number, clientY: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const v = view.current;
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    return { sx, sy, x: (sx - v.tx) / v.s, y: (sy - v.ty) / v.s };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    try {
      canvasRef.current!.setPointerCapture(e.pointerId);
    } catch {
      /* not capturable (e.g. synthetic or already-released pointer): the gesture still works */
    }
    const pt = toImage(e.clientX, e.clientY);
    pointers.current.set(e.pointerId, { x: pt.sx, y: pt.sy });
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()] as [{ x: number; y: number }, { x: number; y: number }];
      gesture.current = { kind: 'pinch', d0: Math.hypot(a.x - b.x, a.y - b.y), cx0: (a.x + b.x) / 2, cy0: (a.y + b.y) / 2, view: { ...view.current } };
      return;
    }
    if (pointers.current.size > 2) return;
    const p = propsRef.current;
    if (p.tool === 'area') {
      gesture.current = { kind: 'lasso', pts: [[pt.x, pt.y]] };
      return;
    }
    if (p.ai) {
      if ((p.tool === 'exemplar' || p.tool === 'region') && p.showOverlay) {
        gesture.current = { kind: 'rect', tool: p.tool, x0: pt.x, y0: pt.y, x1: pt.x, y1: pt.y };
        return;
      }
      gesture.current = { kind: 'tap-or-pan', startX: pt.sx, startY: pt.sy, view: { ...view.current }, t0: performance.now(), moved: false, hitId: null };
      return;
    }
    if (p.tool === 'select' && p.showOverlay) {
      const sel = p.selectedId ? p.objects.find((o) => o.id === p.selectedId) : null;
      if (sel) {
        const handleD = Math.hypot(pt.x - (sel.x + sel.r), pt.y - sel.y) * view.current.s;
        const rimD = Math.abs(Math.hypot(pt.x - sel.x, pt.y - sel.y) - sel.r) * view.current.s;
        if (handleD < 16 || (rimD < 8 && Math.hypot(pt.x - sel.x, pt.y - sel.y) > sel.r * 0.6)) {
          gesture.current = { kind: 'resize', id: sel.id, moved: false };
          return;
        }
      }
      const hit = hitTest(p.objects, pt.x, pt.y);
      if (hit) {
        if (hit.id !== p.selectedId) p.onSelect(hit.id);
        gesture.current = { kind: 'move', id: hit.id, dx: pt.x - hit.x, dy: pt.y - hit.y, moved: false };
        return;
      }
    }
    const hit = p.showOverlay ? hitTest(p.objects, pt.x, pt.y) : null;
    gesture.current = { kind: 'tap-or-pan', startX: pt.sx, startY: pt.sy, view: { ...view.current }, t0: performance.now(), moved: false, hitId: hit?.id ?? null };
  };

  const onPointerMove = (e: React.PointerEvent) => {
    if (!pointers.current.has(e.pointerId)) return;
    const pt = toImage(e.clientX, e.clientY);
    pointers.current.set(e.pointerId, { x: pt.sx, y: pt.sy });
    const g = gesture.current;
    const p = propsRef.current;
    if (g.kind === 'pinch' && pointers.current.size >= 2) {
      const [a, b] = [...pointers.current.values()] as [{ x: number; y: number }, { x: number; y: number }];
      const d = Math.hypot(a.x - b.x, a.y - b.y);
      const cx = (a.x + b.x) / 2;
      const cy = (a.y + b.y) / 2;
      const s = Math.min(16, Math.max(0.02, g.view.s * (d / Math.max(1, g.d0))));
      const f = s / g.view.s;
      view.current = { s, tx: cx - (g.cx0 - g.view.tx) * f, ty: cy - (g.cy0 - g.view.ty) * f };
      requestDraw();
    } else if (g.kind === 'tap-or-pan') {
      const dx = pt.sx - g.startX;
      const dy = pt.sy - g.startY;
      if (!g.moved && Math.hypot(dx, dy) > TAP_SLOP) g.moved = true;
      if (g.moved) {
        view.current = { s: g.view.s, tx: g.view.tx + dx, ty: g.view.ty + dy };
        requestDraw();
      }
    } else if (g.kind === 'lasso') {
      const last = g.pts[g.pts.length - 1]!;
      // Keep a point every ~3 screen pixels.
      if (Math.hypot(pt.x - last[0], pt.y - last[1]) * view.current.s >= 3) {
        g.pts.push([pt.x, pt.y]);
        requestDraw();
      }
    } else if (g.kind === 'rect') {
      g.x1 = pt.x;
      g.y1 = pt.y;
      requestDraw();
    } else if (g.kind === 'move') {
      g.moved = true;
      p.onMove(g.id, pt.x - g.dx, pt.y - g.dy, false);
    } else if (g.kind === 'resize') {
      const o = p.objects.find((x) => x.id === g.id);
      if (o) {
        g.moved = true;
        p.onResize(g.id, Math.max(2, Math.hypot(pt.x - o.x, pt.y - o.y)), false);
      }
    }
  };

  const onPointerUp = (e: React.PointerEvent) => {
    if (!pointers.current.has(e.pointerId)) return;
    const pt = toImage(e.clientX, e.clientY);
    pointers.current.delete(e.pointerId);
    const g = gesture.current;
    const p = propsRef.current;
    if (g.kind === 'pinch') {
      if (pointers.current.size === 0) gesture.current = { kind: 'none' };
      return;
    }
    gesture.current = { kind: 'none' };
    if (e.type === 'pointercancel') return;
    if (g.kind === 'lasso') {
      requestDraw();
      g.pts.push([pt.x, pt.y]);
      p.onDrawArea?.(g.pts);
      return;
    }
    if (g.kind === 'rect') {
      requestDraw();
      if (Math.abs(pt.x - g.x0) * view.current.s > TAP_SLOP && Math.abs(pt.y - g.y0) * view.current.s > TAP_SLOP) p.onDrawRect?.(g.tool, g.x0, g.y0, pt.x, pt.y);
      return;
    }
    if (g.kind === 'tap-or-pan' && !g.moved && p.ai) {
      const r = p.ai.regions.find((q) => pt.x >= q.x && pt.x <= q.x + q.w && pt.y >= q.y && pt.y <= q.y + q.h);
      p.onSelectRegion?.(r?.id ?? null);
      return;
    }
    if (g.kind === 'tap-or-pan' && !g.moved) {
      if (p.tool === 'edit') {
        if (!p.showOverlay) return;
        if (g.hitId) p.onRemove(g.hitId);
        else p.onAdd(pt.x, pt.y);
      } else {
        p.onSelect(g.hitId);
      }
    } else if (g.kind === 'move' && g.moved) {
      p.onMove(g.id, pt.x - g.dx, pt.y - g.dy, true);
    } else if (g.kind === 'resize' && g.moved) {
      const o = p.objects.find((x) => x.id === g.id);
      if (o) p.onResize(g.id, Math.max(2, Math.hypot(pt.x - o.x, pt.y - o.y)), true);
    }
  };

  // Native, non-passive wheel listener so zooming does not also scroll the page.
  useEffect(() => {
    const canvas = canvasRef.current!;
    const onWheel = (e: WheelEvent) => {
      e.preventDefault();
      const rect = canvas.getBoundingClientRect();
      zoomAt(Math.exp(-e.deltaY * (e.deltaMode === 1 ? 0.05 : 0.0015)), e.clientX - rect.left, e.clientY - rect.top);
    };
    canvas.addEventListener('wheel', onWheel, { passive: false });
    return () => canvas.removeEventListener('wheel', onWheel);
  }, [zoomAt]);

  return (
    <div ref={wrapRef} className="viewer">
      <canvas
        ref={canvasRef}
        className={`viewer-canvas tool-${props.tool}`}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onPointerCancel={onPointerUp}
        onContextMenu={(e) => e.preventDefault()}
        data-testid="viewer"
      />
    </div>
  );
});
