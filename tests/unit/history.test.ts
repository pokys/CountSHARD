import { describe, expect, it } from 'vitest';
import { HISTORY_LIMIT, initHistory, push, redo, undo } from '../../src/state/history';

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

  it('is bounded', () => {
    let h = initHistory(0);
    for (let i = 1; i <= HISTORY_LIMIT + 50; i++) h = push(h, i);
    expect(h.past.length).toBe(HISTORY_LIMIT);
  });
});
