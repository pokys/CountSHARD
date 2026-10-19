/** Generic undo/redo history over immutable snapshots. */
export interface History<T> {
  past: T[];
  present: T;
  future: T[];
}

export const HISTORY_LIMIT = 100;

export function initHistory<T>(present: T): History<T> {
  return { past: [], present, future: [] };
}

/** Records a new state. `coalesce` replaces the present without adding an undo step (e.g. during a drag). */
export function push<T>(h: History<T>, next: T, coalesce = false): History<T> {
  if (next === h.present) return h;
  if (coalesce) return { ...h, present: next, future: [] };
  return { past: [...h.past, h.present].slice(-HISTORY_LIMIT), present: next, future: [] };
}

export function undo<T>(h: History<T>): History<T> {
  if (!h.past.length) return h;
  return { past: h.past.slice(0, -1), present: h.past[h.past.length - 1]!, future: [h.present, ...h.future] };
}

export function redo<T>(h: History<T>): History<T> {
  if (!h.future.length) return h;
  return { past: [...h.past, h.present], present: h.future[0]!, future: h.future.slice(1) };
}
