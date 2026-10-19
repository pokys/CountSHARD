import { createRequire } from 'node:module';
import type { CV } from '../src/analysis/cv';

const require = createRequire(import.meta.url);
let cvPromise: Promise<CV> | null = null;

/** Loads the same OpenCV.js build the app ships, in Node. */
export function loadCv(): Promise<CV> {
  cvPromise ??= (async () => {
    // The module export is a Promise resolving to the initialised runtime.
    const cv = await require('@techstark/opencv-js');
    return cv as CV;
  })();
  return cvPromise;
}
