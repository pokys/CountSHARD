/**
 * Local persistence in IndexedDB. Projects (annotations) and images (blobs) are
 * stored separately so the project list stays cheap to read.
 */
import { type Project, isDetectionResult } from '../model/types';
import { computeStats } from '../state/project';
import { countedDetections, shownDetections } from '../ai/geco2';
import { roundEstimate, summarize } from '../ai/estimate';

const DB_NAME = 'countshard';
const DB_VERSION = 1;

export interface ProjectSummary {
  id: string;
  name: string;
  updatedAt: string;
  /** Count of the project's current mode; null before anything was counted. */
  total: number | null;
  /** The total is an AI density estimate, not counted items. */
  estimate: boolean;
  groups: number;
}

/** What the project list shows: the count of the mode the project was left in. */
export function projectTotal(p: Project): { total: number | null; estimate: boolean } {
  const area = p.countArea ?? null;
  if ((p.mode ?? 'circles') === 'circles') return { total: p.analysis || p.objects.length ? computeStats(p).total : null, estimate: false };
  const r = p.ai?.result;
  if (isDetectionResult(r)) return { total: countedDetections(shownDetections(r), area).length, estimate: false };
  const s = p.ai ? summarize(p.ai, area) : null;
  return s ? { total: roundEstimate(s.corrected), estimate: true } : { total: null, estimate: false };
}

let dbPromise: Promise<IDBDatabase> | null = null;

function openDb(): Promise<IDBDatabase> {
  dbPromise ??= new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, DB_VERSION);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains('projects')) db.createObjectStore('projects', { keyPath: 'id' });
      if (!db.objectStoreNames.contains('images')) db.createObjectStore('images', { keyPath: 'id' });
    };
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => {
      dbPromise = null;
      reject(req.error);
    };
  });
  return dbPromise;
}

function tx<T>(stores: string[], mode: IDBTransactionMode, fn: (t: IDBTransaction) => IDBRequest<T> | void): Promise<T | undefined> {
  return openDb().then(
    (db) =>
      new Promise<T | undefined>((resolve, reject) => {
        const t = db.transaction(stores, mode);
        const req = fn(t);
        t.oncomplete = () => resolve(req ? req.result : undefined);
        t.onerror = () => reject(t.error);
        t.onabort = () => reject(t.error ?? new Error('aborted'));
      }),
  );
}

/** Closes the connection so the database can be deleted. */
export async function closeDb(): Promise<void> {
  if (!dbPromise) return;
  try {
    (await dbPromise).close();
  } catch {
    /* never opened */
  }
  dbPromise = null;
}

export async function deleteAllData(): Promise<void> {
  await closeDb();
  // Also the database of the app's earlier name, which may still exist on early users' devices.
  for (const name of [DB_NAME, 'countit']) {
    await new Promise<void>((resolve) => {
      const req = indexedDB.deleteDatabase(name);
      req.onsuccess = req.onerror = req.onblocked = () => resolve();
    });
  }
}

export async function saveProject(p: Project, image?: Blob): Promise<void> {
  await tx(['projects', 'images'], 'readwrite', (t) => {
    t.objectStore('projects').put(p);
    if (image) t.objectStore('images').put({ id: p.id, blob: image });
  });
}

export async function loadProject(id: string): Promise<{ project: Project; image: Blob } | null> {
  const project = await tx<Project>(['projects'], 'readonly', (t) => t.objectStore('projects').get(id));
  const img = await tx<{ id: string; blob: Blob }>(['images'], 'readonly', (t) => t.objectStore('images').get(id));
  if (!project || !img) return null;
  return { project, image: img.blob };
}

export async function listProjects(): Promise<ProjectSummary[]> {
  const all = (await tx<Project[]>(['projects'], 'readonly', (t) => t.objectStore('projects').getAll())) ?? [];
  return all
    .map((p) => {
      const s = computeStats(p);
      return { id: p.id, name: p.name, updatedAt: p.updatedAt, ...projectTotal(p), groups: s.groups.length };
    })
    .sort((a, b) => b.updatedAt.localeCompare(a.updatedAt));
}

export async function deleteProject(id: string): Promise<void> {
  await tx(['projects', 'images'], 'readwrite', (t) => {
    t.objectStore('projects').delete(id);
    t.objectStore('images').delete(id);
  });
}

/** Asks the browser not to evict our storage under pressure (best effort). */
export async function requestPersistence(): Promise<boolean> {
  try {
    if (navigator.storage?.persisted && (await navigator.storage.persisted())) return true;
    return (await navigator.storage?.persist?.()) ?? false;
  } catch {
    return false;
  }
}

export async function storageEstimate(): Promise<{ usage: number; quota: number } | null> {
  try {
    const e = await navigator.storage?.estimate?.();
    return e && e.quota ? { usage: e.usage ?? 0, quota: e.quota } : null;
  } catch {
    return null;
  }
}
