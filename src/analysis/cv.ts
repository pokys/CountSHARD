/**
 * Minimal structural typing of the OpenCV.js surface we use. Keeping the analysis
 * code typed against this interface (instead of the 900 kB generated typings)
 * lets the same functions run inside the Web Worker and in Node tests.
 */
/* eslint-disable @typescript-eslint/no-explicit-any */
export type CV = any;
export type Mat = any;

/** Runs `fn` and deletes every Mat allocated via `track` afterwards (OpenCV.js has no GC for Mats). */
export function withMats<T>(fn: (track: <M extends { delete(): void }>(m: M) => M) => T): T {
  const mats: { delete(): void }[] = [];
  const track = <M extends { delete(): void }>(m: M): M => {
    mats.push(m);
    return m;
  };
  try {
    return fn(track);
  } finally {
    for (const m of mats) {
      try {
        m.delete();
      } catch {
        /* already deleted */
      }
    }
  }
}
