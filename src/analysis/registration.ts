/**
 * Registration of photos taken from nearly the same viewpoint (multi-shot, Phase 6).
 *
 * ORB keypoints on the whole frame → brute-force Hamming kNN matching with Lowe's
 * ratio test → homography by RANSAC (cv.findHomography). The whole frame is used on
 * purpose: pipe faces alone are too self-similar to match reliably, the surroundings
 * are what pins the transform down.
 *
 * The result is judged before anyone may use it. A registration is refused when
 * there are too few inliers, the inliers cover only a small part of the image, the
 * reprojection residual is high, or the transform is not a "small viewpoint change"
 * (large viewpoint changes are out of scope; a wrong homography would silently
 * duplicate or merge objects, which is worse than not fusing at all).
 */
import { type CV, withMats } from './cv';
import type { GrayImage } from './circles';
import { type Mat3, apply3 } from './homography';

export type RegistrationFailure =
  | 'too-few-features'
  | 'too-few-matches'
  | 'too-few-inliers'
  | 'low-inlier-ratio'
  | 'poor-coverage'
  | 'high-residual'
  | 'implausible-transform';

export interface RegistrationResult {
  ok: boolean;
  /** Maps pixel coordinates of the moving image into the reference image. `null` when no estimate exists. */
  H: Mat3 | null;
  keypoints: [number, number];
  /** Matches surviving the ratio test. */
  matches: number;
  inliers: number;
  inlierRatio: number;
  /** RMS reprojection error of the inliers, reference pixels. */
  residualPx: number;
  /** Fraction of the reference image covered by the inliers' bounding box. */
  coverage: number;
  /** Largest displacement of an image corner under H, as a fraction of the image diagonal. */
  cornerShift: number;
  reason?: RegistrationFailure;
}

export interface RegistrationOptions {
  /** Features are computed on a copy scaled to this longest side (speed). */
  featureMaxSide?: number;
  nFeatures?: number;
  ratio?: number;
  /** RANSAC reprojection threshold, reference pixels. */
  ransacPx?: number;
  minInliers?: number;
  minInlierRatio?: number;
  minCoverage?: number;
  maxResidualPx?: number;
  /** Max corner displacement / diagonal accepted as a "nearly the same viewpoint". */
  maxCornerShift?: number;
}

export const REGISTRATION_DEFAULTS: Required<RegistrationOptions> = {
  featureMaxSide: 1200,
  nFeatures: 5000,
  ratio: 0.8,
  ransacPx: 3,
  minInliers: 40,
  minInlierRatio: 0.15,
  minCoverage: 0.2,
  maxResidualPx: 2.5,
  maxCornerShift: 0.2,
};

interface Features {
  pts: Float32Array;
  des: unknown;
  count: number;
}

function features(cv: CV, img: GrayImage, o: Required<RegistrationOptions>, track: <M extends { delete(): void }>(m: M) => M): Features {
  const src = track(new cv.Mat(img.height, img.width, cv.CV_8UC1));
  src.data.set(img.data);
  const s = Math.min(1, o.featureMaxSide / Math.max(img.width, img.height));
  let work = src;
  if (s < 1) {
    work = track(new cv.Mat());
    cv.resize(src, work, new cv.Size(Math.round(img.width * s), Math.round(img.height * s)), 0, 0, cv.INTER_AREA);
  }
  const orb = track(new cv.ORB(o.nFeatures));
  const kp = track(new cv.KeyPointVector());
  const des = track(new cv.Mat());
  const noMask = track(new cv.Mat());
  orb.detectAndCompute(work, noMask, kp, des);
  const n = kp.size();
  const pts = new Float32Array(2 * n);
  for (let i = 0; i < n; i++) {
    const p = kp.get(i).pt;
    pts[2 * i] = p.x / s;
    pts[2 * i + 1] = p.y / s;
  }
  return { pts, des, count: n };
}

/** Registers `moving` onto `reference`. Pure function of the two images. */
export function registerImages(cv: CV, reference: GrayImage, moving: GrayImage, opts: RegistrationOptions = {}): RegistrationResult {
  const o = { ...REGISTRATION_DEFAULTS, ...opts };
  return withMats((track) => {
    const fr = features(cv, reference, o, track);
    const fm = features(cv, moving, o, track);
    const base: RegistrationResult = {
      ok: false,
      H: null,
      keypoints: [fr.count, fm.count],
      matches: 0,
      inliers: 0,
      inlierRatio: 0,
      residualPx: Infinity,
      coverage: 0,
      cornerShift: Infinity,
    };
    if (fr.count < o.minInliers || fm.count < o.minInliers) return { ...base, reason: 'too-few-features' };

    const bf = track(new cv.BFMatcher(cv.NORM_HAMMING, false));
    const knn = track(new cv.DMatchVectorVector());
    bf.knnMatch(fm.des, fr.des, knn, 2);
    const src: number[] = [];
    const dst: number[] = [];
    for (let i = 0; i < knn.size(); i++) {
      const pair = knn.get(i);
      if (pair.size() < 2) continue;
      const a = pair.get(0), b = pair.get(1);
      if (a.distance >= o.ratio * b.distance) continue;
      src.push(fm.pts[2 * a.queryIdx]!, fm.pts[2 * a.queryIdx + 1]!);
      dst.push(fr.pts[2 * a.trainIdx]!, fr.pts[2 * a.trainIdx + 1]!);
    }
    const matches = src.length / 2;
    if (matches < o.minInliers) return { ...base, matches, reason: 'too-few-matches' };

    const srcM = track(cv.matFromArray(matches, 1, cv.CV_32FC2, src));
    const dstM = track(cv.matFromArray(matches, 1, cv.CV_32FC2, dst));
    const mask = track(new cv.Mat());
    const Hm = track(cv.findHomography(srcM, dstM, cv.RANSAC, o.ransacPx, mask, 4000, 0.999));
    if (Hm.empty() || Hm.rows !== 3) return { ...base, matches, reason: 'too-few-inliers' };
    const H = Array.from(Hm.data64F as Float64Array) as Mat3;

    let inliers = 0, se = 0;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < matches; i++) {
      if (!mask.data[i]) continue;
      inliers++;
      const [u, v] = apply3(H, src[2 * i]!, src[2 * i + 1]!);
      const dx = u - dst[2 * i]!, dy = v - dst[2 * i + 1]!;
      se += dx * dx + dy * dy;
      x0 = Math.min(x0, dst[2 * i]!); x1 = Math.max(x1, dst[2 * i]!);
      y0 = Math.min(y0, dst[2 * i + 1]!); y1 = Math.max(y1, dst[2 * i + 1]!);
    }
    const residualPx = inliers ? Math.sqrt(se / inliers) : Infinity;
    const coverage = inliers ? ((x1 - x0) * (y1 - y0)) / (reference.width * reference.height) : 0;
    const diag = Math.hypot(moving.width, moving.height);
    const corners: [number, number][] = [[0, 0], [moving.width, 0], [0, moving.height], [moving.width, moving.height]];
    let cornerShift = 0;
    for (const [x, y] of corners) {
      const [u, v] = apply3(H, x, y);
      const d = Math.hypot(u - x, v - y) / diag;
      cornerShift = Math.max(cornerShift, Number.isFinite(d) ? d : Infinity);
    }
    const res: RegistrationResult = { ...base, H, matches, inliers, inlierRatio: inliers / matches, residualPx, coverage, cornerShift };

    if (inliers < o.minInliers) return { ...res, reason: 'too-few-inliers' };
    if (res.inlierRatio < o.minInlierRatio) return { ...res, reason: 'low-inlier-ratio' };
    if (coverage < o.minCoverage) return { ...res, reason: 'poor-coverage' };
    if (residualPx > o.maxResidualPx) return { ...res, reason: 'high-residual' };
    if (!(cornerShift <= o.maxCornerShift)) return { ...res, reason: 'implausible-transform' };
    return { ...res, ok: true };
  });
}
