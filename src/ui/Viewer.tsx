import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef } from 'react';
import type { CountedObject, SizeGroup } from '../model/types';
import { drawAnnotations } from '../io/draw';
import { hitTest } from '../state/project';

export type Tool = 'edit' | 'select';

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
  | { kind: 'pinch'; d0: number; cx0: number; cy0: number; view: View };

const TAP_SLOP = 7;

export const Viewer = forwardRef<ViewerHandle, Props>(function Viewer(props, ref) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const view = useRef<View>({ s: 1, tx: 0, ty: 0 });
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
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.fillStyle = '#0b1220';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    const v = view.current;
    ctx.setTransform(v.s * dpr, 0, 0, v.s * dpr, v.tx * dpr, v.ty * dpr);
    ctx.imageSmoothingEnabled = v.s < 2;
    ctx.drawImage(p.image, 0, 0);
    if (p.showOverlay) {
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
  }, []);

  const requestDraw = useCallback(() => {
    if (!frame.current) frame.current = requestAnimationFrame(draw);
  }, [draw]);

  const fit = useCallback(() => {
    const wrap = wrapRef.current;
    if (!wrap) return;
    const { width: iw, height: ih } = propsRef.current.image;
    const s = Math.min(wrap.clientWidth / iw, wrap.clientHeight / ih) * 0.98;
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
    const ro = new ResizeObserver(() => requestDraw());
    ro.observe(wrap);
    return () => ro.disconnect();
  }, [requestDraw]);

  const toImage = (clientX: number, clientY: number) => {
    const rect = canvasRef.current!.getBoundingClientRect();
    const v = view.current;
    const sx = clientX - rect.left;
    const sy = clientY - rect.top;
    return { sx, sy, x: (sx - v.tx) / v.s, y: (sy - v.ty) / v.s };
  };

  const onPointerDown = (e: React.PointerEvent) => {
    canvasRef.current!.setPointerCapture(e.pointerId);
    const pt = toImage(e.clientX, e.clientY);
    pointers.current.set(e.pointerId, { x: pt.sx, y: pt.sy });
    if (pointers.current.size === 2) {
      const [a, b] = [...pointers.current.values()] as [{ x: number; y: number }, { x: number; y: number }];
      gesture.current = { kind: 'pinch', d0: Math.hypot(a.x - b.x, a.y - b.y), cx0: (a.x + b.x) / 2, cy0: (a.y + b.y) / 2, view: { ...view.current } };
      return;
    }
    if (pointers.current.size > 2) return;
    const p = propsRef.current;
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
