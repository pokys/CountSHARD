/**
 * Baseline for comparison: a textbook single-range HoughCircles pass with the
 * object size estimated from the peak of an Otsu/distance-transform map. This
 * mirrors the *approach* observed on the reference site (see
 * docs/research/loreatec-analysis.md); it is written independently from the
 * OpenCV documentation and is used only in evaluation.
 */
import type { CV } from '../../src/analysis/cv';
import type { GrayImage } from '../../src/analysis/circles';

export function baselineDetect(cv: CV, img: GrayImage, sensitivity = 40) {
  const src = new cv.Mat(img.height, img.width, cv.CV_8UC1);
  src.data.set(img.data);
  const blur = new cv.Mat();
  const bin = new cv.Mat();
  const dist = new cv.Mat();
  const out = new cv.Mat();
  try {
    cv.GaussianBlur(src, blur, new cv.Size(5, 5), 0);
    cv.threshold(blur, bin, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU);
    if (cv.countNonZero(bin) > bin.rows * bin.cols * 0.55) cv.bitwise_not(bin, bin);
    cv.distanceTransform(bin, dist, cv.DIST_L2, 5);
    const r = Math.min(80, Math.max(3, cv.minMaxLoc(dist).maxVal * 0.95));
    cv.medianBlur(src, blur, r > 10 ? 5 : 3);
    cv.HoughCircles(blur, out, cv.HOUGH_GRADIENT, 1.2, Math.max(6, 1.5 * r), 120, 67 - sensitivity, Math.max(3, Math.round(0.6 * r)), Math.round(1.45 * r));
    const circles: { x: number; y: number; r: number }[] = [];
    for (let i = 0; i < out.cols; i++) circles.push({ x: out.data32F[i * 3], y: out.data32F[i * 3 + 1], r: out.data32F[i * 3 + 2] });
    return { circles, estimatedR: r };
  } finally {
    src.delete(); blur.delete(); bin.delete(); dist.delete(); out.delete();
  }
}
