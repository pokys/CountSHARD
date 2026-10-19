/**
 * Image acquisition and normalisation. The *project image* is the decoded photo,
 * EXIF-oriented and capped at PROJECT_MAX_SIDE; every annotation coordinate refers to it.
 */
import type { ImageInfo } from '../model/types';

/** Longest side of the stored project image. Keeps memory bounded on phones (≈ 28 MB RGBA). */
export const PROJECT_MAX_SIDE = 3072;

export class ImageLoadError extends Error {
  constructor(public code: 'unsupported' | 'decode', message: string) {
    super(message);
  }
}

export interface LoadedImage {
  canvas: HTMLCanvasElement;
  info: ImageInfo;
  blob: Blob;
}

async function decode(file: Blob): Promise<ImageBitmap | HTMLImageElement> {
  if ('createImageBitmap' in window) {
    try {
      // Applies EXIF orientation (default in current browsers; explicit for older ones).
      return await createImageBitmap(file, { imageOrientation: 'from-image' });
    } catch {
      /* fall back to <img> (e.g. older Safari) */
    }
  }
  const url = URL.createObjectURL(file);
  try {
    const img = new Image();
    img.decoding = 'async';
    img.src = url;
    await img.decode();
    return img;
  } finally {
    URL.revokeObjectURL(url);
  }
}

export async function loadImageFile(file: Blob, name: string): Promise<LoadedImage> {
  if (file.type && !file.type.startsWith('image/')) throw new ImageLoadError('unsupported', file.type);
  let src: ImageBitmap | HTMLImageElement;
  try {
    src = await decode(file);
  } catch {
    // Typical case: HEIC/HEIF outside Safari.
    throw new ImageLoadError('decode', name);
  }
  const ow = 'naturalWidth' in src ? src.naturalWidth : src.width;
  const oh = 'naturalHeight' in src ? src.naturalHeight : src.height;
  if (!ow || !oh) throw new ImageLoadError('decode', name);
  const scale = Math.min(1, PROJECT_MAX_SIDE / Math.max(ow, oh));
  const w = Math.round(ow * scale);
  const h = Math.round(oh * scale);
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  const ctx = canvas.getContext('2d')!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(src, 0, 0, w, h);
  if ('close' in src) src.close();
  const blob = scale < 1 || !/^image\/(jpeg|png|webp)$/.test(file.type) ? await canvasToBlob(canvas, 'image/jpeg', 0.92) : file;
  return {
    canvas,
    blob,
    info: { name, width: w, height: h, originalWidth: ow, originalHeight: oh, mime: blob.type || 'image/jpeg' },
  };
}

export function canvasToBlob(canvas: HTMLCanvasElement, type: string, quality?: number): Promise<Blob> {
  return new Promise((resolve, reject) =>
    canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('toBlob failed'))), type, quality),
  );
}

/** Downscales the project image for analysis. Returns pixels and the analysis→project scale factor. */
export function analysisPixels(canvas: HTMLCanvasElement, maxSide: number): { image: ImageData; toProject: number } {
  const s = Math.min(1, maxSide / Math.max(canvas.width, canvas.height));
  const w = Math.max(1, Math.round(canvas.width * s));
  const h = Math.max(1, Math.round(canvas.height * s));
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(canvas, 0, 0, w, h);
  return { image: ctx.getImageData(0, 0, w, h), toProject: canvas.width / w };
}

export async function blobToCanvas(blob: Blob): Promise<HTMLCanvasElement> {
  const src = await decode(blob);
  const w = 'naturalWidth' in src ? src.naturalWidth : src.width;
  const h = 'naturalHeight' in src ? src.naturalHeight : src.height;
  const canvas = document.createElement('canvas');
  canvas.width = w;
  canvas.height = h;
  canvas.getContext('2d')!.drawImage(src, 0, 0);
  if ('close' in src) src.close();
  return canvas;
}
