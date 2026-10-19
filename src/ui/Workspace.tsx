import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useI18n } from '../i18n';
import { type MultiShotInfo, type Project, isCounted, isSelectable } from '../model/types';
import { median } from '../analysis/clustering';
import { CancelledError, analysisClient } from '../analysis/client';
import { ImageLoadError, analysisPixels, loadImageFile } from '../io/image';
import { saveProject } from '../io/db';
import { initHistory, push, redo, undo, type History } from '../state/history';
import {
  addManualObject,
  applyGrouping,
  computeStats,
  mergeDetections,
  numbering,
  removeObject,
  setReview,
  updateObject,
} from '../state/project';
import type { Session } from './App';
import { Viewer, type Tool, type ViewerHandle } from './Viewer';
import { CalibrationPanel, ExportPanel, type ExtraPhoto, GroupsPanel, MultiShotPanel, ResultPanel, SelectedPanel, SettingsPanel } from './Panels';

interface Props {
  session: Session;
  onEngineLoaded(): void;
  onError(msg: string | null): void;
  onNewPhoto(): void;
}

export type Commit = (fn: (p: Project) => Project, coalesce?: boolean) => void;

interface Progress {
  phase: 'engine' | 'detect' | 'fuse';
  fraction: number;
}

export function Workspace({ session, onEngineLoaded, onError, onNewPhoto }: Props) {
  const { t } = useI18n();
  const [hist, setHist] = useState<History<Project>>(() => initHistory(session.project));
  const project = hist.present;
  const [tool, setTool] = useState<Tool>('edit');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [showOverlay, setShowOverlay] = useState(true);
  const [showNumbers, setShowNumbers] = useState(true);
  const [showRejected, setShowRejected] = useState(false);
  const [comparing, setComparing] = useState(false);
  const [progress, setProgress] = useState<Progress | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  // Multi-shot: extra photos live only in memory; the fused result is saved with the project.
  const [extraPhotos, setExtraPhotos] = useState<ExtraPhoto[]>([]);
  const [fusionReport, setFusionReport] = useState<MultiShotInfo | null>(() => session.project.analysis?.multiShot ?? null);
  const viewerRef = useRef<ViewerHandle>(null);

  const commit: Commit = useCallback((fn, coalesce = false) => {
    setHist((h) => push(h, { ...fn(h.present), updatedAt: new Date().toISOString() }, coalesce));
  }, []);

  // Autosave (debounced). The image blob was stored when the project was created.
  useEffect(() => {
    const id = setTimeout(() => {
      saveProject(project).catch(() => onError(t.err_storage));
    }, 600);
    return () => clearTimeout(id);
  }, [project, onError, t]);

  // Set when a newer run supersedes the current one, so its cancellation is not reported.
  const superseded = useRef(false);

  const runAnalysis = useCallback(async (override?: Partial<Project['params']>) => {
    onError(null);
    setNotice(null);
    const params = { ...hist.present.params, ...override };
    setProgress({ phase: 'engine', fraction: 0 });
    try {
      await analysisClient.warmup();
      onEngineLoaded();
    } catch {
      setProgress(null);
      onError(t.err_engine);
      return;
    }
    try {
      setProgress({ phase: 'detect', fraction: 0 });
      const { image, toProject } = analysisPixels(session.canvas, params.analysisMaxSide);
      const manual = params.sizeMode === 'manual';
      const res = await analysisClient.detect(
        {
          image,
          sensitivity: params.sensitivity,
          minRadius: manual ? params.minDiameterPx / 2 / toProject : undefined,
          maxRadius: manual ? params.maxDiameterPx / 2 / toProject : undefined,
        },
        (f) => setProgress({ phase: 'detect', fraction: f }),
      );
      const circles = res.circles.map((c) => ({ x: c.x * toProject, y: c.y * toProject, r: c.r * toProject, score: c.score }));
      commit((p) => {
        const objects = mergeDetections(p.objects, circles);
        const grouped = applyGrouping(objects, p.groups, p.grouping);
        const diam = circles.map((c) => 2 * c.r).sort((a, b) => a - b);
        return {
          ...p,
          ...grouped,
          analysis: {
            algorithm: 'circles-multiband-hough-v1',
            engine: 'OpenCV.js 5.0.0',
            durationMs: res.durationMs,
            analysisWidth: image.width,
            analysisHeight: image.height,
            estimatedDiameterPx: diam.length ? [diam[Math.floor(diam.length * 0.05)]!, diam[Math.floor(diam.length * 0.95)]!] : null,
            at: new Date().toISOString(),
          },
        };
      });
      // A single-photo run replaces the unreviewed fused objects, so the fusion summary no longer applies.
      setFusionReport(null);
      if (!circles.length) setNotice(t.none_found);
    } catch (e) {
      if (e instanceof CancelledError) {
        if (!superseded.current) setNotice(t.cancelled);
        superseded.current = false;
        return;
      }
      else onError(t.err_analysis(e instanceof Error ? e.message : String(e)));
    } finally {
      setProgress(null);
    }
  }, [hist.present.params, session.canvas, commit, onEngineLoaded, onError, t]);

  const addExtraPhotos = useCallback(
    async (files: File[]) => {
      onError(null);
      const room = Math.max(0, 4 - extraPhotos.length);
      if (files.length > room) setNotice(t.ms_max);
      const loaded: ExtraPhoto[] = [];
      for (const file of files.slice(0, room)) {
        try {
          loaded.push({ name: file.name, canvas: (await loadImageFile(file, file.name)).canvas });
        } catch (e) {
          if (e instanceof ImageLoadError) onError(e.code === 'unsupported' ? t.err_unsupported : t.err_decode(file.name));
          else onError(t.err_analysis(e instanceof Error ? e.message : String(e)));
        }
      }
      setExtraPhotos((list) => [...list, ...loaded].slice(0, 4));
    },
    [extraPhotos.length, onError, t],
  );

  const runFusion = useCallback(async () => {
    onError(null);
    setNotice(null);
    const params = hist.present.params;
    setProgress({ phase: 'engine', fraction: 0 });
    try {
      await analysisClient.warmup();
      onEngineLoaded();
    } catch {
      setProgress(null);
      onError(t.err_engine);
      return;
    }
    try {
      setProgress({ phase: 'fuse', fraction: 0 });
      const ref = analysisPixels(session.canvas, params.analysisMaxSide);
      const others = extraPhotos.map((p) => analysisPixels(p.canvas, params.analysisMaxSide).image);
      const manual = params.sizeMode === 'manual';
      const { fusion, durationMs } = await analysisClient.fuse(
        [ref.image, ...others],
        {
          sensitivity: params.sensitivity,
          minRadius: manual ? params.minDiameterPx / 2 / ref.toProject : undefined,
          maxRadius: manual ? params.maxDiameterPx / 2 / ref.toProject : undefined,
        },
        (f) => setProgress({ phase: 'fuse', fraction: f }),
      );
      const names = [hist.present.image.name, ...extraPhotos.map((p) => p.name)];
      const info: MultiShotInfo = {
        status: fusion.status,
        photos: fusion.views.map((v) => ({
          name: names[v.index] ?? `#${v.index + 1}`,
          status: v.status,
          reason: v.reason,
          detections: v.detections,
          inliers: v.inliers,
          residualPx: Number.isFinite(v.residualPx) ? v.residualPx : 0,
          agreement: v.agreement,
          added: v.added,
        })),
        referenceCount: fusion.referenceCount,
        countedTotal: fusion.countedTotal,
        addedCount: fusion.addedCount,
        flaggedCount: fusion.flaggedCount,
        uncertainCount: fusion.uncertainCount,
        droppedFromReference: fusion.droppedFromReference,
      };
      setFusionReport(info);
      // Refused: the current result stays untouched — a fused one would not be more reliable.
      if (fusion.status !== 'fused') return;
      const k = ref.toProject;
      const circles = fusion.objects.map((o) => ({
        x: o.x * k,
        y: o.y * k,
        r: o.r * k,
        score: o.score,
        multiShot: { seenIn: o.seenIn, inFrame: o.inFrame, agreement: o.agreement },
      }));
      commit((p) => {
        const objects = mergeDetections(p.objects, circles);
        const grouped = applyGrouping(objects, p.groups, p.grouping);
        const diam = circles.filter((c) => c.multiShot.agreement !== 'uncertain').map((c) => 2 * c.r).sort((a, b) => a - b);
        return {
          ...p,
          ...grouped,
          analysis: {
            algorithm: 'circles-multiband-hough-v1 + orb-ransac-fusion-v1',
            engine: 'OpenCV.js 5.0.0',
            durationMs,
            analysisWidth: ref.image.width,
            analysisHeight: ref.image.height,
            estimatedDiameterPx: diam.length ? [diam[Math.floor(diam.length * 0.05)]!, diam[Math.floor(diam.length * 0.95)]!] : null,
            at: new Date().toISOString(),
            multiShot: info,
          },
        };
      });
    } catch (e) {
      if (e instanceof CancelledError) setNotice(t.cancelled);
      else onError(t.err_analysis(e instanceof Error ? e.message : String(e)));
    } finally {
      setProgress(null);
    }
  }, [hist.present.params, hist.present.image.name, session.canvas, extraPhotos, commit, onEngineLoaded, onError, t]);

  // New photos are analysed immediately.
  const autoRan = useRef(false);
  useEffect(() => {
    if (session.autoAnalyze && !autoRan.current) {
      autoRan.current = true;
      void runAnalysis();
    }
  }, [session.autoAnalyze, runAnalysis]);

  const stats = useMemo(() => computeStats(project), [project]);
  const numbers = useMemo(() => numbering(project.objects), [project.objects]);
  const selected = project.objects.find((o) => o.id === selectedId && isSelectable(o)) ?? null;

  const defaultRadius = (x: number, y: number): number => {
    const counted = project.objects.filter(isCounted);
    if (!counted.length) return Math.min(project.image.width, project.image.height) * 0.025;
    const near = [...counted].sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y)).slice(0, 6);
    return median(near.map((o) => o.r));
  };

  const regroup = (p: Project): Project => ({ ...p, ...applyGrouping(p.objects, p.groups, p.grouping) });

  const handlers = {
    onAdd: (x: number, y: number) => {
      const r = defaultRadius(x, y);
      commit((p) => regroup({ ...p, objects: addManualObject(p.objects, x, y, r, null).objects }));
    },
    onRemove: (id: string) => {
      commit((p) => ({ ...p, objects: removeObject(p.objects, id) }));
      if (id === selectedId) setSelectedId(null);
    },
    onSelect: setSelectedId,
    onMove: (id: string, x: number, y: number, final: boolean) => commit((p) => ({ ...p, objects: updateObject(p.objects, id, { x, y }) }), !final),
    onResize: (id: string, r: number, final: boolean) =>
      commit((p) => {
        const next = { ...p, objects: updateObject(p.objects, id, { r }) };
        return final ? regroup(next) : next;
      }, !final),
  };

  // Keyboard shortcuts.
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement)?.closest('input, textarea, select')) return;
      const mod = e.ctrlKey || e.metaKey;
      if (mod && e.key.toLowerCase() === 'z') {
        e.preventDefault();
        setHist((h) => (e.shiftKey ? redo(h) : undo(h)));
      } else if (mod && e.key.toLowerCase() === 'y') {
        e.preventDefault();
        setHist(redo);
      } else if ((e.key === 'Delete' || e.key === 'Backspace') && selectedId) {
        handlers.onRemove(selectedId);
      } else if (e.key === 'Escape') {
        setSelectedId(null);
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  // Sensitivity slider under the photo: re-counts shortly after the user stops moving it.
  const [sens, setSens] = useState(project.params.sensitivity);
  useEffect(() => setSens(project.params.sensitivity), [project.params.sensitivity]);
  const sensTimer = useRef<ReturnType<typeof setTimeout>>();
  const onSensitivity = (v: number) => {
    setSens(v);
    clearTimeout(sensTimer.current);
    sensTimer.current = setTimeout(() => {
      commit((p) => ({ ...p, params: { ...p.params, sensitivity: v } }));
      if (progress) {
        superseded.current = true;
        analysisClient.cancel();
      }
      void runAnalysis({ sensitivity: v });
    }, 450);
  };
  useEffect(() => () => clearTimeout(sensTimer.current), []);

  const confirmAll = () =>
    commit((p) => ({ ...p, objects: setReview(p.objects, p.objects.filter((o) => isCounted(o)).map((o) => o.id), 'confirmed') }));

  return (
    <main className="workspace">
      <section className="stage">
        <div className="toolbar" role="toolbar">
          <div className="seg" role="radiogroup">
            <button className={tool === 'edit' ? 'on' : ''} onClick={() => setTool('edit')} role="radio" aria-checked={tool === 'edit'}>
              {t.tool_edit}
            </button>
            <button className={tool === 'select' ? 'on' : ''} onClick={() => setTool('select')} role="radio" aria-checked={tool === 'select'}>
              {t.tool_select}
            </button>
          </div>
          <div className="row">
            <button className="icon" onClick={() => setHist(undo)} disabled={!hist.past.length} title={t.undo} aria-label={t.undo}>
              ↶
            </button>
            <button className="icon" onClick={() => setHist(redo)} disabled={!hist.future.length} title={t.redo} aria-label={t.redo}>
              ↷
            </button>
            <button className="icon" onClick={() => viewerRef.current?.zoomBy(1.4)} title={t.zoom_in} aria-label={t.zoom_in}>
              +
            </button>
            <button className="icon" onClick={() => viewerRef.current?.zoomBy(1 / 1.4)} title={t.zoom_out} aria-label={t.zoom_out}>
              −
            </button>
            <button className="icon" onClick={() => viewerRef.current?.fit()} title={t.zoom_fit} aria-label={t.zoom_fit}>
              ⤢
            </button>
          </div>
        </div>
        <div className="viewer-wrap">
        <Viewer
          ref={viewerRef}
          image={session.canvas}
          objects={project.objects}
          groups={project.groups}
          numbers={showNumbers ? numbers : null}
          selectedId={tool === 'select' ? selectedId : null}
          tool={tool}
          showOverlay={showOverlay && !comparing}
          showRejected={showRejected}
          {...handlers}
        />
        {progress && (
          <div className="progress-overlay" role="status" aria-live="polite">
            <div className="progress-card">
              <p>{progress.phase === 'engine' ? t.loading_engine : progress.phase === 'fuse' ? t.ms_analyzing : t.analyzing}</p>
              <progress value={progress.phase === 'engine' ? undefined : progress.fraction} max={1} />
              <button className="btn small" onClick={() => analysisClient.cancel()}>
                {t.cancel}
              </button>
            </div>
          </div>
        )}
        </div>
        <div className="sensbar">
          <label htmlFor="sens-quick">{t.sensitivity}</label>
          <span className="tiny muted-dark">{t.sens_less}</span>
          <input
            id="sens-quick"
            type="range"
            min={0}
            max={100}
            value={sens}
            onChange={(e) => onSensitivity(Number(e.target.value))}
            aria-describedby="sens-quick-hint"
            data-testid="sens-quick"
          />
          <span className="tiny muted-dark">{t.sens_more}</span>
          <output htmlFor="sens-quick">{sens}</output>
          <span id="sens-quick-hint" hidden>
            {t.sensitivity_hint} {t.rerun_hint}
          </span>
        </div>
        <div className="toolbar bottom">
          <span className="hint">{tool === 'edit' ? t.tool_edit_hint : t.tool_select_hint}</span>
          <div className="row">
            <label className="toggle">
              <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} /> {t.overlay}
            </label>
            <label className="toggle">
              <input type="checkbox" checked={showNumbers} onChange={(e) => setShowNumbers(e.target.checked)} /> {t.numbers}
            </label>
            <label className="toggle">
              <input type="checkbox" checked={showRejected} onChange={(e) => setShowRejected(e.target.checked)} /> {t.show_rejected}
            </label>
            <button
              className="btn ghost small"
              onPointerDown={() => setComparing(true)}
              onPointerUp={() => setComparing(false)}
              onPointerLeave={() => setComparing(false)}
              onPointerCancel={() => setComparing(false)}
            >
              {t.compare}
            </button>
          </div>
        </div>
      </section>

      <aside className="side">
        {notice && <div className="banner info">{notice}</div>}
        <ResultPanel stats={stats} onConfirmAll={confirmAll} />
        {selected && (
          <SelectedPanel
            obj={selected}
            number={numbers.get(selected.id) ?? 0}
            project={project}
            commit={commit}
            onRemove={() => handlers.onRemove(selected.id)}
          />
        )}
        <GroupsPanel project={project} stats={stats} commit={commit} />
        <CalibrationPanel project={project} commit={commit} />
        <SettingsPanel project={project} commit={commit} running={!!progress} onRun={() => runAnalysis()} />
        <MultiShotPanel
          photos={extraPhotos}
          report={fusionReport}
          running={!!progress}
          onAdd={(files) => void addExtraPhotos(files)}
          onRemove={(i) => setExtraPhotos((list) => list.filter((_, j) => j !== i))}
          onRun={runFusion}
        />
        <ExportPanel project={project} canvas={session.canvas} commit={commit} />
        <button className="btn ghost" onClick={onNewPhoto}>
          {t.new_photo}
        </button>
      </aside>
    </main>
  );
}

