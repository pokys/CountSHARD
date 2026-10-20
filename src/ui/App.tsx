import { useCallback, useEffect, useState } from 'react';
import { useI18n } from '../i18n';
import type { Project } from '../model/types';
import { createProject, relabel } from '../state/project';
import { ImageLoadError, blobToCanvas, loadImageFile } from '../io/image';
import { deleteProject, listProjects, loadProject, requestPersistence, saveProject, type ProjectSummary } from '../io/db';
import { dataUrlToBlob } from '../io/export';
import { ProjectFileError, parseProjectFile } from '../io/serialize';
import { makeDemoImage } from '../demo/demoImage';
import { analysisClient } from '../analysis/client';
import { takeAiCrashReport } from '../ai/client';
import { StartScreen } from './StartScreen';
import { Workspace } from './Workspace';
import { OfflineBadge } from './OfflineBadge';

export interface Session {
  project: Project;
  canvas: HTMLCanvasElement;
  imageBlob: Blob;
  /** Run automatic analysis right after opening (new photos only). */
  autoAnalyze: boolean;
}

/** Brings projects saved by older versions up to date (e.g. group colours that are now reserved). */
function normalize(p: Project): Project {
  return { ...p, groups: relabel(p.objects, p.groups) };
}

export function App() {
  const { t, lang, setLang } = useI18n();
  const [session, setSession] = useState<Session | null>(null);
  const [recent, setRecent] = useState<ProjectSummary[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [engineLoaded, setEngineLoaded] = useState(false);

  const refreshRecent = useCallback(() => {
    listProjects().then(setRecent).catch(() => setRecent([]));
  }, []);

  // A Mode B run that killed the page (out of memory on phones) is reported instead of a silent reset.
  useEffect(() => {
    const crash = takeAiCrashReport();
    if (crash) setError(crash.model === 'geco2' ? t.ai_crashed_geco2(crash.backend === 'webgpu') : t.ai_crashed(crash.backend === 'webgpu'));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    refreshRecent();
    // Warm up the detection engine in the background so the first analysis is quick.
    const id = setTimeout(() => {
      analysisClient.warmup().then(() => setEngineLoaded(true)).catch(() => setEngineLoaded(false));
    }, 300);
    return () => clearTimeout(id);
  }, [refreshRecent]);

  const openFile = async (file: Blob, name: string) => {
    setError(null);
    setBusy(true);
    try {
      const img = await loadImageFile(file, name);
      const project = createProject(img.info);
      try {
        await saveProject(project, img.blob);
        void requestPersistence();
      } catch {
        setError(t.err_storage);
      }
      setSession({ project, canvas: img.canvas, imageBlob: img.blob, autoAnalyze: true });
    } catch (e) {
      if (e instanceof ImageLoadError) setError(e.code === 'unsupported' ? t.err_unsupported : t.err_decode(name));
      else setError(String(e));
    } finally {
      setBusy(false);
    }
  };

  const openDemo = async () => {
    const blob = await makeDemoImage();
    await openFile(blob, `${t.demo_name}.png`);
  };

  const openSaved = async (id: string) => {
    setError(null);
    const loaded = await loadProject(id);
    if (!loaded) return;
    const canvas = await blobToCanvas(loaded.image);
    setSession({ project: normalize(loaded.project), canvas, imageBlob: loaded.image, autoAnalyze: false });
  };

  const importProject = async (file: File) => {
    setError(null);
    try {
      const f = parseProjectFile(await file.text());
      const blob = await dataUrlToBlob(f.image);
      const canvas = await blobToCanvas(blob);
      const project = { ...f.project, updatedAt: new Date().toISOString() };
      await saveProject(project, blob).catch(() => setError(t.err_storage));
      setSession({ project: normalize(project), canvas, imageBlob: blob, autoAnalyze: false });
    } catch (e) {
      setError(t.err_import(e instanceof ProjectFileError ? e.message : String(e)));
    }
  };

  const removeSaved = async (p: ProjectSummary) => {
    if (!confirm(t.delete_confirm(p.name))) return;
    await deleteProject(p.id);
    refreshRecent();
  };

  return (
    <div className="app">
      <header className="topbar">
        <button
          className="brand"
          onClick={() => {
            setSession(null);
            refreshRecent();
          }}
          aria-label={t.appName}
        >
          <img src="./icon.svg" alt="" width={28} height={28} />
          <span>{t.appName}</span>
        </button>
        <div className="topbar-right">
          <OfflineBadge engineLoaded={engineLoaded} />
          <button className="btn ghost small" onClick={() => setLang(lang === 'cs' ? 'en' : 'cs')} aria-label={t.lang}>
            {lang === 'cs' ? 'EN' : 'CS'}
          </button>
        </div>
      </header>
      {error && (
        <div className="banner error" role="alert">
          {error}
          <button className="btn ghost small" onClick={() => setError(null)} aria-label="×">
            ×
          </button>
        </div>
      )}
      {session ? (
        <Workspace
          key={session.project.id}
          session={session}
          onEngineLoaded={() => setEngineLoaded(true)}
          onEngineUnloaded={() => setEngineLoaded(false)}
          onError={setError}
          onNewPhoto={() => {
            setSession(null);
            refreshRecent();
          }}
        />
      ) : (
        <StartScreen busy={busy} recent={recent} onFile={openFile} onDemo={openDemo} onOpen={openSaved} onDelete={removeSaved} onImport={importProject} />
      )}
    </div>
  );
}
