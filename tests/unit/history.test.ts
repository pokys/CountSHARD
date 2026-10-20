import { describe, expect, it } from 'vitest';
import { HISTORY_LIMIT, dragMerge, initHistory, push, redo, undo } from '../../src/state/history';

describe('history', () => {
  it('undoes and redoes', () => {
    let h = initHistory(0);
    h = push(h, 1);
    h = push(h, 2);
    h = undo(h);
    expect(h.present).toBe(1);
    h = redo(h);
    expect(h.present).toBe(2);
  });

  it('drops the redo branch on a new change', () => {
    let h = push(push(initHistory(0), 1), 2);
    h = push(undo(h), 5);
    expect(h.future).toEqual([]);
    expect(redo(h).present).toBe(5);
  });

  it('coalesces intermediate drag states into one step', () => {
    let h = push(initHistory(0), 1);
    h = push(h, 2, true);
    h = push(h, 3, true);
    expect(undo(h).present).toBe(0);
  });

  it('a whole drag (intermediate states, then the final one) is exactly one undo step', () => {
    const d = { active: false, at: 0 };
    let h = push(initHistory(0), 1); // state before the drag
    // Callers: intermediate states coalesce = true, final state false.
    for (const [v, coalesce, t] of [[2, true, 10], [3, true, 20], [4, false, 30]] as const) h = push(h, v, dragMerge(d, coalesce, t));
    expect(h.present).toBe(4);
    expect(undo(h).present).toBe(1);
    // The next ordinary change is its own step again.
    h = push(h, 5, dragMerge(d, false, 40));
    expect(undo(h).present).toBe(4);
  });

  it('a drag without a final state ends after a second of inactivity', () => {
    const d = { active: false, at: 0 };
    let h = initHistory(0);
    h = push(h, 1, dragMerge(d, true, 0)); // slider moved with the keyboard
    h = push(h, 2, dragMerge(d, true, 100));
    h = push(h, 3, dragMerge(d, false, 5000)); // unrelated change later
    expect(undo(h).present).toBe(2);
    expect(undo(undo(h)).present).toBe(0);
  });

  it('is bounded', () => {
    let h = initHistory(0);
    for (let i = 1; i <= HISTORY_LIMIT + 50; i++) h = push(h, i);
    expect(h.past.length).toBe(HISTORY_LIMIT);
  });
});
