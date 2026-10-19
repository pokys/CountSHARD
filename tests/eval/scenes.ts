import type { SceneSpec } from '../synthetic/scene';

/** Evaluation suite. Sizes are in analysis pixels (the app analyses at ≤1600 px). */
export const SCENES: SceneSpec[] = [
  { name: 'uniform-rods', width: 900, height: 700, radii: [22], layout: 'hex', noiseSigma: 6, blur: 1, seed: 1 },
  { name: 'uniform-tubes', width: 900, height: 700, radii: [26], layout: 'hex', hollow: true, noiseSigma: 6, blur: 1, seed: 2 },
  { name: 'dense-small', width: 1000, height: 800, radii: [9], layout: 'hex', noiseSigma: 5, blur: 0.8, seed: 3 },
  {
    name: 'mixed-4-sizes', width: 1200, height: 900, radii: [20, 26.9, 33.7, 42.4].map((d) => d * 0.75),
    layout: 'random', count: 260, noiseSigma: 6, blur: 1, radiusJitter: 0.015, seed: 4,
  },
  {
    name: 'mixed-tubes', width: 1200, height: 900, radii: [18, 27], layout: 'random', count: 220,
    hollow: true, noiseSigma: 6, blur: 1, seed: 5,
  },
  { name: 'uneven-light', width: 900, height: 700, radii: [20], layout: 'hex', lightGradient: 0.75, noiseSigma: 6, blur: 1, seed: 6 },
  { name: 'noisy-blurred', width: 900, height: 700, radii: [20], layout: 'hex', noiseSigma: 18, blur: 2.2, seed: 7 },
  { name: 'clutter', width: 1000, height: 800, radii: [24], layout: 'random', count: 120, clutter: true, noiseSigma: 8, blur: 1, seed: 8 },
  { name: 'perspective-0.88', width: 900, height: 700, radii: [22], layout: 'hex', squash: 0.88, noiseSigma: 6, blur: 1, seed: 9 },
  { name: 'occluded', width: 900, height: 700, radii: [22], layout: 'hex', occluder: true, noiseSigma: 6, blur: 1, seed: 10 },
];
