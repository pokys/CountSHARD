// Quick capability probe for the bundled OpenCV.js build (run: node scripts/cv-probe.mjs)
import { createRequire } from 'node:module';
const require = createRequire(import.meta.url);
const t0 = Date.now();
const cv = await require('@techstark/opencv-js');
console.log('load ms', Date.now() - t0, 'HOUGH_GRADIENT_ALT', cv.HOUGH_GRADIENT_ALT);
const m = new cv.Mat(400, 400, cv.CV_8UC1, new cv.Scalar(30));
for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) cv.circle(m, new cv.Point(50 + i * 75, 50 + j * 75), 30, new cv.Scalar(200), -1);
cv.GaussianBlur(m, m, new cv.Size(5, 5), 0);
const out = new cv.Mat();
cv.HoughCircles(m, out, cv.HOUGH_GRADIENT, 1.2, 40, 120, 20, 20, 40);
console.log('HOUGH_GRADIENT', out.rows, out.cols, out.channels());
const alt = new cv.Mat();
cv.HoughCircles(m, alt, cv.HOUGH_GRADIENT_ALT, 1.5, 40, 300, 0.8, 20, 40);
console.log('HOUGH_GRADIENT_ALT', alt.rows, alt.cols, alt.channels(), Array.from(alt.data32F.slice(0, 3)));
for (const k of ['distanceTransform', 'findContours', 'fitEllipse', 'connectedComponentsWithStats', 'ORB', 'AKAZE', 'BFMatcher', 'findHomography', 'estimateAffinePartial2D', 'CLAHE', 'watershed', 'Sobel']) console.log(k, typeof cv[k]);
