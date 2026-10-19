/**
 * Local persistence in IndexedDB. Projects (annotations) and images (blobs) are
 * stored separately so the project list stays cheap to read.
 */
import type { Project } from '../model/types';
import { computeStats } from '../state/project';

const DB_NAME = 'countshard';
const DB_VERSION = 1;

export interface ProjectSummary {
  id: string;
  name: string;
  updatedAt: string;
  total: number;
  groups: number;
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
      return { id: p.id, name: p.name, updatedAt: p.updatedAt, total: s.total, groups: s.groups.length };
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
