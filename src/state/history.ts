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

/**
 * Drag tracking for callers that send intermediate drag states with `coalesce` = true and the
 * final state with false. Returns whether this change merges into the current undo step: the
 * first intermediate state opens a step, later ones and the final state replace it. A drag
 * without a final state (keyboard on a slider) ends after `idleMs` without changes.
 */
export function dragMerge(d: { active: boolean; at: number }, coalesce: boolean, now: number, idleMs = 1000): boolean {
  const merge = d.active && now - d.at < idleMs;
  d.active = coalesce;
  d.at = now;
  return merge;
}

export function undo<T>(h: History<T>): History<T> {
  if (!h.past.length) return h;
  return { past: h.past.slice(0, -1), present: h.past[h.past.length - 1]!, future: [h.present, ...h.future] };
}

export function redo<T>(h: History<T>): History<T> {
  if (!h.future.length) return h;
  return { past: [...h.past, h.present], present: h.future[0]!, future: h.future.slice(1) };
}
