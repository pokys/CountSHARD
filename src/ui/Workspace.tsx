import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useI18n } from '../i18n';
import { type AiDetection, type CountArea, type CountMode, type MultiShotInfo, type Project, type SizeGroup, inCountArea, isCounted, isDensityResult, isDetectionResult, isSelectable } from '../model/types';
import { clusterRadii, median } from '../analysis/clustering';
import { CancelledError, analysisClient } from '../analysis/client';
import { ImageLoadError, analysisPixels, loadImageFile } from '../io/image';
import { saveProject } from '../io/db';
import { dragMerge, initHistory, push, redo, undo, type History } from '../state/history';
import {
  addManualObject,
  applyGrouping,
  GROUP_COLORS,
  computeStats,
  groupLabel,
  makeCountAreaFromPath,
  mergeDetections,
  newId,
  numbering,
  removeGroup,
  switchMode,
  autoProjectName,
  removeObject,
  setObjectGroup,
  setReview,
  updateObject,
  visibleInArea,
} from '../state/project';
import type { Session } from './App';
import { Viewer, type Tool, type ViewerHandle } from './Viewer';
import { CalibrationPanel, ExportPanel, type ExtraPhoto, GroupsPanel, MultiShotPanel, ResultPanel, SelectedPanel, SettingsPanel, useExportPng } from './Panels';
import { AiDetectionResultPanel, AiPanel, AiResultPanel, type PendingRegion } from './AiPanels';
import { type AiStatus, aiClient } from '../ai/client';
import { AI_MODEL_ID, AiDownloadError, ORT_VERSION, PROFILES_MODEL_ID, geco2ModelId } from '../ai/modelStore';
import { SCORE_MIN, YOLO_CLASSES, YOLO_SIZE, minScoreFromSensitivity, sensitivityFromMinScore } from '../ai/yolo';
import {
  asCountedObjects,
  boxesOf,
  classGrouping,
  countedDetections,
  detectionClassAt,
  isUncertain,
  shownDetections,
  geco2Boxes,
  geco2ContentSize,
  geco2Scale,
  mergeAiDetections,
  newManualDetection,
  removeAiDetection,
  toProjectDetections,
  typicalBox,
} from '../ai/geco2';
import { planTiles } from '../ai/tiling';
import { GRID_CELL } from '../ai/runner';
import { addCorrection, addExemplar, decodeDensity, emptyAiState, encodeDensity, normalizeRect, roundEstimate, sumInRect, summarize, estimateTooLow, exemplarSizeRatio, exemplarsTooDifferent } from '../ai/estimate';
import { heatCanvas } from '../io/export';
import type { AiOverlay } from '../io/draw';

interface Props {
  session: Session;
  onEngineLoaded(): void;
  /** The OpenCV worker was terminated (AI mode frees its memory). */
  onEngineUnloaded(): void;
  onError(msg: string | null): void;
  onNewPhoto(): void;
}

export type Commit = (fn: (p: Project) => Project, coalesce?: boolean) => void;

interface Progress {
  phase: 'engine' | 'detect' | 'fuse' | 'ai-load' | 'ai-run';
  fraction: number;
}

/** Resizes the photo to the model scale chosen by the tile plan. */
/** Longer side of the photo given to the steel-profile model (4 tiles × 640 px at most). */
const PROFILES_MAX_SIDE = 2560;

function scaledPixels(canvas: HTMLCanvasElement, w: number, h: number): ImageData {
  const c = document.createElement('canvas');
  c.width = w;
  c.height = h;
  const ctx = c.getContext('2d', { willReadFrequently: true })!;
  ctx.imageSmoothingQuality = 'high';
  ctx.drawImage(canvas, 0, 0, w, h);
  return ctx.getImageData(0, 0, w, h);
}

export function Workspace({ session, onEngineLoaded, onEngineUnloaded, onError, onNewPhoto }: Props) {
  const { t, fmt } = useI18n();
  const [hist, setHist] = useState<History<Project>>(() => initHistory(session.project));
  const project = hist.present;
  const [tool, setTool] = useState<Tool>((session.project.mode ?? 'circles') !== 'circles' && !isDetectionResult(session.project.ai?.result) ? 'pan' : 'edit');
  const [selectedId, setSelectedId] = useState<string | null>(null);
  /** AI modes: detection selected by a long press (confirm / change type / remove). */
  const [selDetId, setSelDetId] = useState<string | null>(null);
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
  const exportPng = useExportPng(hist.present, session.canvas);
  // Short "Added · Undo" notice after a tap edit, so an accidental tap is one tap away from undone.
  const [toast, setToast] = useState<{ text: string; id: number; undoable: boolean } | null>(null);
  const toastTimer = useRef<ReturnType<typeof setTimeout>>();
  const showToast = (text: string, undoable = true) => {
    clearTimeout(toastTimer.current);
    setToast({ text, id: Date.now(), undoable });
    toastTimer.current = setTimeout(() => setToast(null), 3500);
  };
  useEffect(() => () => clearTimeout(toastTimer.current), []);
  const resultRef = useRef<HTMLDivElement>(null);

  // One drag (move/resize, exemplar handles, radius slider) = one undo step, see dragMerge.
  const drag = useRef({ active: false, at: 0 });
  const commit: Commit = useCallback((fn, coalesce = false) => {
    const merge = dragMerge(drag.current, coalesce, Date.now());
    setHist((h) => push(h, { ...fn(h.present), updatedAt: new Date().toISOString() }, merge));
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
        const grouped = applyGrouping(objects, p.groups, p.grouping, p.countArea ?? null);
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
        const grouped = applyGrouping(objects, p.groups, p.grouping, p.countArea ?? null);
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
    if (session.autoAnalyze && !autoRan.current && (session.project.mode ?? 'circles') === 'circles') {
      autoRan.current = true;
      void runAnalysis();
    }
  }, [session.autoAnalyze, session.project.mode, runAnalysis]);

  // --- Mode B (generic objects, AI density estimate) ---------------------------
  const mode = project.mode ?? 'circles';
  /** One of the AI modes (generic objects or steel profiles). */
  const aiMode = mode !== 'circles';
  const [aiStatus, setAiStatus] = useState<AiStatus>(aiClient.status);
  useEffect(() => aiClient.subscribe(setAiStatus), []);
  useEffect(() => void aiClient.refresh(), []);
  useEffect(() => {
    if (mode !== 'circles') void aiClient.setMode(mode);
  }, [mode]);
  const [aiDownloadError, setAiDownloadError] = useState<string | null>(null);
  const [pendingRegion, setPendingRegion] = useState<PendingRegion | null>(null);
  const [regionError, setRegionError] = useState<string | null>(null);
  const [selectedRegionId, setSelectedRegionId] = useState<string | null>(null);
  const [selectedExemplarId, setSelectedExemplarId] = useState<string | null>(null);
  const onExemplarChange = (id: string, box: { x: number; y: number; w: number; h: number }, final: boolean) =>
    commit((p) => (p.ai ? { ...p, ai: { ...p.ai, exemplars: p.ai.exemplars.map((e) => (e.id === id ? { ...e, ...box } : e)) } } : p), !final);
  const aiSummary = useMemo(() => (project.ai ? summarize(project.ai, project.countArea ?? null) : null), [project.ai, project.countArea]);
  const anyAiResult = project.ai?.result ?? null;
  /** LOCA density estimate (null for a GeCo2 result). */
  const aiResult = isDensityResult(anyAiResult) ? anyAiResult : null;
  /** GeCo2 detections (null for a LOCA result). */
  const detResult = isDetectionResult(anyAiResult) ? anyAiResult : null;
  const heat = useMemo(() => (aiResult ? heatCanvas(aiResult) : null), [aiResult]);
  /** Detections in play (the sensitivity slider hides unsure automatic ones). */
  const detList = useMemo(() => (detResult ? shownDetections(detResult) : []), [detResult]);
  const detObjects = useMemo(() => asCountedObjects(detList), [detList]);
  /** Steel profiles: automatic items the model was unsure about. */
  const detUncertain = useMemo(
    () => (detResult?.classes ? new Set(countedDetections(detList, project.countArea ?? null).filter(isUncertain).map((d) => d.id)) : null),
    [detResult, detList, project.countArea],
  );
  const detShown = useMemo(() => detObjects.filter((o) => inCountArea(project.countArea, o.x, o.y)), [detObjects, project.countArea]);
  const selDet = selDetId ? (detList.find((d) => d.id === selDetId && d.review !== 'rejected') ?? null) : null;
  const detNumbers = useMemo(() => numbering(detObjects, project.countArea ?? null), [detObjects, project.countArea]);
  const detBoxes = useMemo(() => (detResult ? boxesOf(detResult.objects) : null), [detResult]);
  const detCount = useMemo(() => (detResult ? countedDetections(detList, project.countArea ?? null).length : 0), [detResult, detList, project.countArea]);
  const aiModel = aiStatus.active;
  const [showHeat, setShowHeat] = useState(true);

  // Phones kill the tab when OpenCV's WASM heap and the AI model are both resident, so the
  // OpenCV worker is terminated in AI mode; it is restarted on the next circle analysis.
  const freeOpenCv = useCallback(() => {
    analysisClient.cancel();
    onEngineUnloaded();
  }, [onEngineUnloaded]);

  const setMode = (m: CountMode) => {
    if (m === mode) return;
    if (progress) return;
    const next = switchMode(project, m);
    commit((p) => {
      const q = switchMode(p, m);
      return m === 'circles' ? q : { ...q, ai: q.ai ?? emptyAiState() };
    });
    if (m !== 'circles') freeOpenCv();
    const nextDet = isDetectionResult(next.ai?.result);
    setTool(m === 'circles' || nextDet ? 'edit' : m === 'profiles' || (next.ai?.exemplars.length ?? 0) > 0 ? 'pan' : 'exemplar');
    setPendingRegion(null);
    setSelectedId(null);
  };

  const downloadModel = useCallback(async () => {
    setAiDownloadError(null);
    try {
      await aiClient.download();
    } catch (e) {
      const code = e instanceof AiDownloadError ? e.code : 'http';
      setAiDownloadError(t.ai_download_err(t.ai_download_reason[code] ?? code));
    }
  }, [t]);

  const deleteModel = useCallback(async () => {
    const m = aiClient.model;
    await aiClient.deleteModel(m === 'loca' || m === 'profiles' ? m : aiClient.status.geco2?.size === 1024 ? 'geco2_1024' : 'geco2_768');
  }, []);

  /** GeCo2: individual detections, merged with the user's earlier edits. */
  const runGeco2 = useCallback(async () => {
    const ai = hist.present.ai!;
    const exemplars = ai.exemplars;
    const size = aiClient.status.geco2?.size ?? 768;
    const W = session.canvas.width;
    const H = session.canvas.height;
    const scale = geco2Scale(W, H, exemplars, size);
    const content = geco2ContentSize(W, H, scale, size);
    // The scale actually applied (content is rounded down to whole pixels).
    const sx = content.w / W;
    const scaled = scaledPixels(session.canvas, content.w, content.h);
    const out = await aiClient.detect(
      scaled,
      geco2Boxes(exemplars, sx),
      (f) => setProgress({ phase: 'ai-run', fraction: f }),
      () => {
        setNotice(t.ai_fallback_wasm);
        setProgress({ phase: 'ai-load', fraction: 0 });
      },
    );
    const fresh = toProjectDetections(out, sx);
    commit((p) => {
      const prev = isDetectionResult(p.ai?.result) && p.ai!.result.modelId === 'geco2' ? p.ai!.result.objects : [];
      return {
        ...p,
        ai: {
          ...(p.ai ?? emptyAiState()),
          result: {
            kind: 'detections',
            modelId: 'geco2',
            model: geco2ModelId(out.size),
            engine: `onnxruntime-web ${ORT_VERSION} · ${out.backend === 'webgpu' ? 'WebGPU' : 'WASM'}`,
            inputSize: out.size,
            scale: sx,
            objects: mergeAiDetections(prev, fresh),
            exemplarsUsed: exemplars,
            inputExemplarIds: exemplars.map((e) => e.id),
            durationMs: out.durationMs,
            at: new Date().toISOString(),
          },
        },
      };
    });
    setTool('edit');
  }, [hist.present.ai, session.canvas, commit, t]);

  /** Steel profiles: detections with a class each, no exemplars; merged with earlier edits. */
  const runProfiles = useCallback(async () => {
    const W = session.canvas.width;
    const H = session.canvas.height;
    const k = Math.min(1, PROFILES_MAX_SIDE / Math.max(W, H));
    const w = Math.max(1, Math.round(W * k));
    const h = Math.max(1, Math.round(H * k));
    const sx = w / W;
    const out = await aiClient.profiles(
      scaledPixels(session.canvas, w, h),
      (f) => setProgress({ phase: 'ai-run', fraction: f }),
      () => {
        setNotice(t.ai_fallback_wasm);
        setProgress({ phase: 'ai-load', fraction: 0 });
      },
    );
    const fresh = toProjectDetections(out, sx);
    commit((p) => {
      const prev = isDetectionResult(p.ai?.result) && p.ai!.result.modelId === 'profiles' ? p.ai!.result.objects : [];
      return {
        ...p,
        ai: {
          ...(p.ai ?? emptyAiState()),
          result: {
            kind: 'detections',
            modelId: 'profiles',
            model: PROFILES_MODEL_ID,
            engine: `onnxruntime-web ${ORT_VERSION} · ${out.backend === 'webgpu' ? 'WebGPU' : 'WASM'}`,
            inputSize: YOLO_SIZE,
            scale: sx,
            objects: mergeAiDetections(prev, fresh),
            classes: [...YOLO_CLASSES],
            minScore: isDetectionResult(p.ai?.result) && p.ai!.result.modelId === 'profiles' ? (p.ai!.result.minScore ?? SCORE_MIN) : SCORE_MIN,
            exemplarsUsed: [],
            inputExemplarIds: [],
            durationMs: out.durationMs,
            at: new Date().toISOString(),
          },
        },
      };
    });
    setTool('edit');
  }, [session.canvas, commit, t]);

  const runAi = useCallback(async () => {
    const ai = hist.present.ai;
    if (aiClient.model !== 'profiles' && (!ai || !ai.exemplars.length)) {
      setNotice(t.ai_need_exemplar);
      return;
    }
    onError(null);
    setNotice(null);
    setPendingRegion(null);
    freeOpenCv();
    setProgress({ phase: 'ai-load', fraction: 0 });
    try {
      await aiClient.load();
    } catch {
      setProgress(null);
      onError(t.ai_err_load);
      return;
    }
    try {
      setProgress({ phase: 'ai-run', fraction: 0 });
      if (aiClient.model === 'geco2') {
        await runGeco2();
        return;
      }
      if (aiClient.model === 'profiles') {
        await runProfiles();
        return;
      }
      if (!ai) return;
      const exemplars = ai.exemplars;
      const plan = planTiles(session.canvas.width, session.canvas.height, exemplars);
      const scaled = scaledPixels(session.canvas, plan.width, plan.height);
      const out = await aiClient.estimate(
        scaled,
        plan,
        (f) => setProgress({ phase: 'ai-run', fraction: f }),
        () => {
          setNotice(t.ai_fallback_wasm);
          setProgress({ phase: 'ai-load', fraction: 0 });
        },
      );
      commit((p) => ({
        ...p,
        ai: {
          ...(p.ai ?? emptyAiState()),
          result: {
            kind: 'density-estimate',
            model: AI_MODEL_ID,
            engine: `onnxruntime-web ${ORT_VERSION} · ${out.backend === 'webgpu' ? 'WebGPU' : 'WASM'}`,
            cellPx: GRID_CELL / plan.scale,
            gridW: out.gridW,
            gridH: out.gridH,
            density: encodeDensity(out.grid),
            rawTotal: out.total,
            tiles: plan.tiles.length,
            scale: plan.scale,
            limitedByTiles: plan.limitedByTiles,
            exemplarsUsed: plan.exemplarsUsed.map((i) => exemplars[i]!),
            inputExemplarIds: exemplars.map((e) => e.id),
            durationMs: out.durationMs,
            at: new Date().toISOString(),
          },
        },
      }));
      setTool('pan');
    } catch (e) {
      if (e instanceof CancelledError) setNotice(t.cancelled);
      else onError(t.ai_err_run(e instanceof Error ? e.message : String(e)));
    } finally {
      setProgress(null);
    }
  }, [hist.present.ai, session.canvas, commit, onError, freeOpenCv, runGeco2, runProfiles, t]);

  // A new photo in the steel-profiles mode is counted as soon as the model is available (at once,
  // or right after its first download).
  const autoRanProfiles = useRef(false);
  useEffect(() => {
    if (!session.autoAnalyze || autoRanProfiles.current || mode !== 'profiles' || progress) return;
    if (aiStatus.active !== 'profiles' || !aiStatus.downloaded || isDetectionResult(hist.present.ai?.result)) return;
    autoRanProfiles.current = true;
    void runAi();
  }, [session.autoAnalyze, mode, progress, aiStatus.active, aiStatus.downloaded, hist.present.ai, runAi]);

  /** Sets (or clears) the counting area; size groups are recomputed for the objects inside it. */
  const setCountArea = (area: CountArea | null) =>
    commit((p) => ({ ...p, countArea: area, ...applyGrouping(p.objects, p.groups, p.grouping, area) }));

  const onDrawArea = (points: [number, number][]) => {
    const a = makeCountAreaFromPath(points, project.image.width, project.image.height);
    if (!a) return;
    setCountArea(a);
    setTool(mode === 'circles' || detResult ? 'edit' : 'pan');
  };

  const onDrawRect = (kind: 'exemplar' | 'region' | 'area', x1: number, y1: number, x2: number, y2: number) => {
    const r = normalizeRect(x1, y1, x2, y2, project.image.width, project.image.height);
    if (r.w < 4 || r.h < 4) return;
    if (kind === 'exemplar') {
      commit((p) => ({ ...p, ai: addExemplar(p.ai ?? emptyAiState(), { id: newId('e'), ...r }) }));
      // Not selected automatically: the next box often starts at this one's corner (neighbouring
      // objects) and must draw a new box, not resize this one. A tap selects it for adjusting.
      setSelectedExemplarId(null);
      return;
    }
    if (!aiResult) return;
    setRegionError(null);
    setPendingRegion({ ...r, model: sumInRect(aiResult, decodeDensity(aiResult.density), r) });
  };

  const finishRegion = (count: number | null) => {
    const r = pendingRegion;
    setPendingRegion(null);
    if (!r || count === null) return;
    const res = addCorrection(project.ai ?? emptyAiState(), { id: newId('r'), x: r.x, y: r.y, w: r.w, h: r.h, userCount: count });
    if (res.error) {
      setRegionError(res.error === 'overlap' ? t.ai_err_overlap : t.ai_err_small);
      return;
    }
    setRegionError(null);
    commit((p) => ({ ...p, ai: addCorrection(p.ai ?? emptyAiState(), res.ai.corrections[res.ai.corrections.length - 1]!).ai }));
  };

  const aiOverlay: AiOverlay | null =
    aiMode
      ? {
          heat: showHeat ? heat : null,
          heatWidth: aiResult ? aiResult.gridW * aiResult.cellPx : 0,
          heatHeight: aiResult ? aiResult.gridH * aiResult.cellPx : 0,
          exemplars: project.ai?.exemplars ?? [],
          regions: [
            // Count corrections belong to a LOCA estimate only; a GeCo2 result is corrected item by item.
            ...(aiResult ? (project.ai?.corrections ?? []) : []).map((c) => ({ ...c, label: t.ai_region_label(c.userCount, fmt(aiSummary?.regions.find((q) => q.id === c.id)?.model ?? 0, 1)) })),
            ...(pendingRegion ? [{ id: 'pending', x: pendingRegion.x, y: pendingRegion.y, w: pendingRegion.w, h: pendingRegion.h, label: '?' }] : []),
          ],
          selectedRegionId,
          selectedExemplarId: tool === 'exemplar' ? selectedExemplarId : null,
        }
      : null;

  const stats = useMemo(() => computeStats(project), [project]);

  // Generated name ("what, how many – time"), kept up to date with the result until the user
  // names the project. Not an undo step: it follows the counts, which undo restores.
  const generatedName = useMemo(() => {
    let what = t.name_what_none[mode];
    if (mode === 'circles') {
      if (project.analysis || project.objects.length) what = t.name_what_pcs(t.name_round, stats.total);
    } else if (detResult?.classes) {
      const per = new Map<number, number>();
      for (const d of countedDetections(detList, project.countArea ?? null)) if (d.cls !== undefined) per.set(d.cls, (per.get(d.cls) ?? 0) + 1);
      const parts = [...per].sort((a, b) => b[1] - a[1]).map(([c, n]) => t.name_what_pcs(t.ai_profiles_classes[c] ?? t.ai_profiles_unknown, n));
      what = parts.length ? parts.join(', ') : t.name_what_pcs(t.name_what_none.profiles, 0);
    } else if (detResult) what = t.name_what_pcs(t.name_items, detCount);
    else if (aiResult && aiSummary) what = t.name_estimate(roundEstimate(aiSummary.corrected));
    return autoProjectName(what, project.createdAt);
  }, [t, mode, project.analysis, project.objects.length, project.countArea, project.createdAt, stats.total, detResult, detList, detCount, aiResult, aiSummary]);
  useEffect(() => {
    if (!project.nameAuto || project.name === generatedName) return;
    setHist((h) => (h.present.nameAuto ? { ...h, present: { ...h.present, name: generatedName } } : h));
  }, [project.nameAuto, project.name, generatedName]);
  const numbers = useMemo(() => numbering(project.objects, project.countArea ?? null), [project.objects, project.countArea]);
  const shownObjects = useMemo(() => visibleInArea(project).objects, [project]);
  const selected = project.objects.find((o) => o.id === selectedId && isSelectable(o)) ?? null;

  const defaultRadius = (x: number, y: number): number => {
    const counted = project.objects.filter(isCounted);
    if (!counted.length) return Math.min(project.image.width, project.image.height) * 0.025;
    const near = [...counted].sort((a, b) => Math.hypot(a.x - x, a.y - y) - Math.hypot(b.x - x, b.y - y)).slice(0, 6);
    return median(near.map((o) => o.r));
  };

  const regroup = (p: Project): Project => ({ ...p, ...applyGrouping(p.objects, p.groups, p.grouping, p.countArea ?? null) });

  // GeCo2 size groups: derived from the box sizes of the counted detections (not stored), so the
  // per-group counts are visible at once, like in circle mode.
  const detGrouping = useMemo(() => {
    const counted = detShown.filter(isCounted);
    const none = { groups: [] as SizeGroup[], byId: new Map<string, string>() };
    // Steel profiles: one group per profile type found (when there are at least two types).
    if (detResult?.classes) {
      return classGrouping(
        detResult.objects,
        counted.map((o) => o.id),
        (c) => t.ai_profiles_classes[c] ?? t.ai_profiles_unknown,
        GROUP_COLORS,
      );
    }
    if (counted.length < 2) return none;
    const res = clusterRadii(counted.map((o) => o.r), { mode: 'auto', minSeparation: project.grouping.minSeparation });
    if (res.centers.length < 2) return none;
    const order = res.centers.map((c, i) => ({ c, i })).sort((a, b) => a.c - b.c);
    const groups: SizeGroup[] = order.map((o, k) => ({ id: `dg${o.i}`, label: groupLabel(k), color: GROUP_COLORS[k % GROUP_COLORS.length]! }));
    return { groups, byId: new Map(counted.map((o, j) => [o.id, `dg${res.labels[j]}`])) };
  }, [detShown, detResult, project.grouping.minSeparation, t]);
  const detShownGrouped = useMemo(() => detShown.map((o) => ({ ...o, groupId: detGrouping.byId.get(o.id) ?? null })), [detShown, detGrouping]);
  const detGroupStats = useMemo(
    () =>
      detGrouping.groups.map((g) => {
        const members = detShownGrouped.filter((o) => o.groupId === g.id);
        return { group: g, count: members.length, medianSidePx: 2 * median(members.map((o) => o.r)), ids: members.map((o) => o.id) };
      }),
    [detGrouping, detShownGrouped],
  );

  /** Edits a GeCo2 result's detections (Mode B); the result's other fields stay. */
  const editDetections = (fn: (list: AiDetection[]) => AiDetection[]) =>
    commit((p) => (isDetectionResult(p.ai?.result) ? { ...p, ai: { ...p.ai!, result: { ...p.ai!.result, objects: fn(p.ai!.result.objects) } } } : p));

  const detHandlers = {
    onAdd: (x: number, y: number) => {
      if (!detResult) return;
      if (!inCountArea(project.countArea, x, y)) {
        showToast(t.area_outside, false);
        return;
      }
      const ex = project.ai?.exemplars ?? [];
      const fallback = ex.length ? ex.reduce((a, e) => a + (e.w + e.h) / 2, 0) / ex.length : Math.min(project.image.width, project.image.height) * 0.05;
      const b = typicalBox(detResult.objects, x, y, fallback);
      editDetections((list) => [...list, newManualDetection(x, y, b.w, b.h, detResult.classes ? detectionClassAt(list, x, y) : undefined)]);
      showToast(t.toast_added);
    },
    onRemove: (id: string) => {
      if (id === selDetId) setSelDetId(null);
      editDetections((list) => removeAiDetection(list, id));
      showToast(t.toast_removed);
    },
    onSelect: (id: string | null) => setSelDetId(id),
    onMove: () => undefined,
    onResize: () => undefined,
  };

  const handlers = {
    onAdd: (x: number, y: number) => {
      if (!inCountArea(project.countArea, x, y)) {
        showToast(t.area_outside, false);
        return;
      }
      const r = defaultRadius(x, y);
      commit((p) => regroup({ ...p, objects: addManualObject(p.objects, x, y, r, null).objects }));
      showToast(t.toast_added);
    },
    onRemove: (id: string) => {
      commit((p) => ({ ...p, objects: removeObject(p.objects, id) }));
      if (id === selectedId) setSelectedId(null);
      showToast(t.toast_removed);
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

  // --- layout: photo over the whole stage, tools in a floating pill, everything else in the sheet ---
  /** The tool used when no drawing tool is active. */
  const defaultTool: Tool = mode === 'circles' || detResult ? 'edit' : 'pan';
  const drawingTool = tool === 'exemplar' || tool === 'area' || tool === 'region';
  const pickTool = (next: Tool) => {
    setSelectedId(null);
    setSelectedExemplarId(null);
    // Tapping the active drawing tool again ends drawing.
    setTool(next === tool && next !== defaultTool ? defaultTool : next);
  };
  const finishDrawing = () => {
    setSelectedExemplarId(null);
    setTool(defaultTool);
  };
  const [viewMenu, setViewMenu] = useState(false);
  // Project name, edited here; saved on Enter / leaving the field, empty = generated again.
  const [nameOpen, setNameOpen] = useState(false);
  const [nameDraft, setNameDraft] = useState(project.name);
  useEffect(() => setNameDraft(project.name), [project.name]);
  const saveName = () => {
    const v = nameDraft.trim();
    if (!v) {
      if (!project.nameAuto) commit((p) => ({ ...p, nameAuto: true }));
      else setNameDraft(project.name);
    } else if (v !== project.name) commit((p) => ({ ...p, name: v, nameAuto: false }));
  };
  const [sheet, setSheet] = useState<'peek' | 'half' | 'full'>('peek');
  const sheetDrag = useRef<{ y: number; moved: boolean } | null>(null);
  const SHEET_ORDER = ['peek', 'half', 'full'] as const;
  const moveSheet = (dir: 1 | -1) => setSheet((s) => SHEET_ORDER[Math.min(2, Math.max(0, SHEET_ORDER.indexOf(s) + dir))]!);
  const peekRef = useRef<HTMLDivElement>(null);
  const mainRef = useRef<HTMLElement>(null);
  // The photo fits above the collapsed sheet (phones): the peek height is a CSS variable.
  useEffect(() => {
    const el = peekRef.current;
    const main = mainRef.current;
    if (!el || !main) return;
    const ro = new ResizeObserver(() => main.style.setProperty('--peek-h', `${el.offsetHeight}px`));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const toolButtons: { id: Tool; label: string; testId: string; disabled?: boolean }[] =
    mode === 'circles'
      ? [
          { id: 'edit', label: t.ai_tool_edit, testId: 'tool-edit' },
          { id: 'area', label: t.tool_area, testId: 'tool-area' },
        ]
      : [
          ...(mode === 'profiles' ? [] : [{ id: 'exemplar' as Tool, label: t.ai_tool_exemplar, testId: 'tool-exemplar' }]),
          ...(detResult
            ? [{ id: 'edit' as Tool, label: t.ai_tool_edit, testId: 'tool-ai-edit' }]
            : aiResult
              ? [{ id: 'region' as Tool, label: t.ai_tool_region, testId: 'tool-region' }]
              : []),
          { id: 'area', label: t.tool_area, testId: 'tool-area' },
        ];

  const taskText =
    tool === 'exemplar' ? t.task_exemplar : tool === 'area' ? t.task_area : tool === 'region' ? t.task_region : null;
  /** Longer help for the current tool (in the sheet, not over the photo). */
  const toolHelp =
    tool === 'exemplar'
      ? t.task_hint_exemplar
      : tool === 'area'
        ? t.tool_area_hint
        : tool === 'region'
          ? t.task_hint_region
          : mode === 'circles'
            ? t.tool_edit_hint_long
            : tool === 'edit'
              ? t.ai_tool_edit_hint
              : t.ai_tool_pan_hint;
  const [regionCount, setRegionCount] = useState('');
  useEffect(() => setRegionCount(pendingRegion?.model != null ? String(roundEstimate(pendingRegion.model)) : ''), [pendingRegion]);
  const modeSwitch = (
    <div className="modebar seg" role="radiogroup" aria-label={t.mode_label}>
      <button className={mode === 'circles' ? 'on' : ''} onClick={() => setMode('circles')} role="radio" aria-checked={mode === 'circles'} disabled={!!progress} data-testid="mode-circles">
        {t.mode_circles}
      </button>
      <button className={mode === 'profiles' ? 'on' : ''} onClick={() => setMode('profiles')} role="radio" aria-checked={mode === 'profiles'} disabled={!!progress} data-testid="mode-profiles">
        {t.mode_profiles}
      </button>
      <button className={mode === 'ai' ? 'on' : ''} onClick={() => setMode('ai')} role="radio" aria-checked={mode === 'ai'} disabled={!!progress} data-testid="mode-ai">
        {t.mode_ai}
      </button>
    </div>
  );
  const dl = aiStatus.downloading;
  const canRunAi = !progress && aiStatus.downloaded && (aiModel === 'profiles' || !!project.ai?.exemplars.length);
  /** The chosen model finds individual items (GeCo2, steel profiles) rather than a density. */
  const detModel = aiModel === 'geco2' || aiModel === 'profiles';

  const discardDetGroup = (g: (typeof detGroupStats)[number]) => {
    const d = `${fmt(g.medianSidePx, 0)} px`;
    if (!confirm(t.qb_discard_confirm(g.group.label, g.count, d))) return;
    editDetections((list) => g.ids.reduce((l, id) => removeAiDetection(l, id), list));
    showToast(t.toast_group_discarded(g.group.label, g.count));
  };

  const groupChips =
    mode === 'circles'
      ? stats.groups.length > 1 &&
        stats.groups.map((g) => {
          const d = g.medianDiameterMm !== null ? `${fmt(g.medianDiameterMm, 1)} mm` : `${fmt(g.medianDiameterPx, 0)} px`;
          return (
            <button
              key={g.group.id}
              className="qb-group"
              role="listitem"
              style={{ ['--gc' as string]: g.group.color }}
              title={`${t.group} ${g.group.label} · Ø ${d} · ${t.qb_group_tap}`}
              disabled={!!progress}
              onClick={() => {
                if (!confirm(t.qb_discard_confirm(g.group.label, g.count, d))) return;
                commit((p) => ({ ...p, ...removeGroup(p.objects, p.groups, g.group.id) }));
                showToast(t.toast_group_discarded(g.group.label, g.count));
              }}
              data-testid={`quick-group-${g.group.label}`}
            >
              <span className="qb-dot" aria-hidden="true" />
              {g.group.label}
              <span className="qb-size">Ø {d}</span>
              <strong>{g.count}</strong>
            </button>
          );
        })
      : detGroupStats.length > 1 &&
        detGroupStats.map((g) => (
          <button
            key={g.group.id}
            className="qb-group"
            role="listitem"
            style={{ ['--gc' as string]: g.group.color }}
            title={`${t.group} ${g.group.label} · ${fmt(g.medianSidePx, 0)} px · ${t.qb_group_tap}`}
            disabled={!!progress}
            onClick={() => discardDetGroup(g)}
            data-testid={`quick-group-${g.group.label}`}
          >
            <span className="qb-dot" aria-hidden="true" />
            {g.group.label}
            <span className="qb-size">{fmt(g.medianSidePx, 0)} px</span>
            <strong>{g.count}</strong>
          </button>
        ));

  const totalButton =
    mode === 'circles' ? (
      <span className="qb-total" aria-label={t.qb_total}>
        <span className="qb-num" data-testid="quick-total">{stats.total}</span> {t.pcs}
      </span>
    ) : detResult || (!aiResult && detModel) ? (
      <span className="qb-total" aria-label={t.ai_det_tag}>
        <span className="qb-num" data-testid="quick-det-total">{detResult ? detCount : '–'}</span> {t.pcs}
        <span className="qb-sub">{t.ai_det_tag}</span>
      </span>
    ) : (
      <span className="qb-total" aria-label={t.ai_estimate_label}>
        <span className="estimate-tag">{t.ai_estimate_label}</span>{' '}
        <span className="qb-num" data-testid="quick-estimate">{aiSummary ? t.ai_estimate(roundEstimate(aiSummary.corrected)) : '–'}</span> {t.pcs}
      </span>
    );

  return (
    <main className={`workspace ws-sheet-${sheet}`} ref={mainRef}>
      <section className="stage">
        <div className="viewer-wrap">
          <Viewer
            ref={viewerRef}
            image={session.canvas}
            objects={aiMode ? detShownGrouped : shownObjects}
            groups={aiMode ? detGrouping.groups : project.groups}
            area={project.countArea ?? null}
            numbers={showNumbers ? (aiMode ? detNumbers : numbers) : null}
            selectedId={mode === 'circles' ? selectedId : selDetId}
            tool={tool}
            showOverlay={showOverlay && !comparing}
            showRejected={showRejected}
            ai={aiOverlay}
            boxes={aiMode ? detBoxes : null}
            uncertain={aiMode ? detUncertain : null}
            onDrawRect={onDrawRect}
            onDrawArea={onDrawArea}
            onSelectRegion={setSelectedRegionId}
            onSelectExemplar={setSelectedExemplarId}
            onExemplarChange={onExemplarChange}
            {...(aiMode ? detHandlers : handlers)}
          />

          <div className="stage-top">
            <div className="pill" role="toolbar" aria-label={t.mode_label}>
              <div className="seg" role="radiogroup">
                {toolButtons.map((b) => (
                  <button
                    key={b.id}
                    className={tool === b.id ? 'on' : ''}
                    onClick={() => pickTool(b.id)}
                    role="radio"
                    aria-checked={tool === b.id}
                    disabled={b.disabled}
                    data-testid={b.testId}
                  >
                    {b.label}
                  </button>
                ))}
              </div>
              <button className="icon" onClick={() => setHist(undo)} disabled={!hist.past.length} title={t.undo} aria-label={t.undo}>
                ↶
              </button>
              <button className="icon" onClick={() => setHist(redo)} disabled={!hist.future.length} title={t.redo} aria-label={t.redo}>
                ↷
              </button>
              <button className="icon desktop-only" onClick={() => viewerRef.current?.zoomBy(1.4)} title={t.zoom_in} aria-label={t.zoom_in}>
                +
              </button>
              <button className="icon desktop-only" onClick={() => viewerRef.current?.zoomBy(1 / 1.4)} title={t.zoom_out} aria-label={t.zoom_out}>
                −
              </button>
              <button className="icon" onClick={() => viewerRef.current?.fit()} title={t.zoom_fit} aria-label={t.zoom_fit}>
                ⤢
              </button>
              <button
                className={`icon${viewMenu || comparing || !showOverlay ? ' on' : ''}`}
                onClick={() => setViewMenu((v) => !v)}
                aria-expanded={viewMenu}
                title={t.view_menu}
                aria-label={t.view_menu}
                data-testid="view-menu"
              >
                👁
              </button>
            </div>
            {viewMenu && (
              <div className="view-menu" role="dialog" aria-label={t.view_menu}>
                <label className="toggle">
                  <input type="checkbox" checked={showOverlay} onChange={(e) => setShowOverlay(e.target.checked)} /> {t.overlay}
                </label>
                {aiMode && !detResult ? (
                  <label className="toggle">
                    <input type="checkbox" checked={showHeat} onChange={(e) => setShowHeat(e.target.checked)} /> {t.ai_heat}
                  </label>
                ) : (
                  <>
                    <label className="toggle">
                      <input type="checkbox" checked={showNumbers} onChange={(e) => setShowNumbers(e.target.checked)} /> {t.numbers}
                    </label>
                    <label className="toggle">
                      <input type="checkbox" checked={showRejected} onChange={(e) => setShowRejected(e.target.checked)} /> {t.show_rejected}
                    </label>
                  </>
                )}
                <label className="toggle">
                  <input type="checkbox" checked={comparing} onChange={(e) => setComparing(e.target.checked)} data-testid="compare" /> {t.compare}
                </label>
              </div>
            )}
            {drawingTool && taskText && (
              <div className="task" role="status" data-testid="task-banner">
                <span>{taskText}</span>
                <div className="row">
                  {tool === 'area' && project.countArea && (
                    <button className="btn small" onClick={() => setCountArea(null)} title={t.area_clear_hint} data-testid="area-clear">
                      {t.area_reset}
                    </button>
                  )}
                  {tool === 'exemplar' && (
                    <button className="btn small primary" disabled={!canRunAi} onClick={runAi} data-testid="task-run">
                      {detModel ? (detResult ? t.ai_rerun_detect : t.ai_run_detect) : aiResult ? t.ai_rerun : t.ai_run}
                    </button>
                  )}
                  <button className="btn small" onClick={finishDrawing} data-testid="task-done">
                    {t.done}
                  </button>
                </div>
              </div>
            )}
            {pendingRegion && (
              <div className="task" role="dialog" aria-label={t.ai_region_user} data-testid="region-ask">
                <span>{pendingRegion.model !== null ? t.region_ask(fmt(pendingRegion.model, 1)) : t.ai_region_user}</span>
                <div className="row">
                  <input
                    inputMode="numeric"
                    value={regionCount}
                    onChange={(e) => setRegionCount(e.target.value.replace(/[^0-9]/g, ''))}
                    size={5}
                    aria-label={t.ai_region_user}
                    data-testid="region-count"
                  />
                  <button className="btn small primary" disabled={regionCount === ''} onClick={() => finishRegion(Number(regionCount))} data-testid="region-save">
                    {t.save}
                  </button>
                  <button className="btn small" onClick={() => finishRegion(null)}>
                    {t.cancel}
                  </button>
                </div>
              </div>
            )}
            {toast && (
              <div className="toast" role="status" key={toast.id}>
                <span>{toast.text}</span>
                {toast.undoable && (
                  <button
                    className="btn small"
                    onClick={() => {
                      setHist(undo);
                      setToast(null);
                    }}
                  >
                    {t.undo}
                  </button>
                )}
              </div>
            )}
          </div>

          {progress && (
            <div className="progress-overlay" role="status" aria-live="polite">
              <div className="progress-card">
                <p>
                  {progress.phase === 'engine'
                    ? t.loading_engine
                    : progress.phase === 'fuse'
                      ? t.ms_analyzing
                      : progress.phase === 'ai-load'
                        ? t.ai_loading_model
                        : progress.phase === 'ai-run'
                          ? detModel
                            ? t.ai_detecting
                            : t.ai_estimating
                          : t.analyzing}
                </p>
                <progress value={progress.phase === 'engine' || progress.phase === 'ai-load' ? undefined : progress.fraction} max={1} />
                <button className="btn small" onClick={() => (progress.phase.startsWith('ai') ? aiClient.cancel() : analysisClient.cancel())}>
                  {t.cancel}
                </button>
              </div>
            </div>
          )}
        </div>
      </section>

      <aside className="side sheet" aria-label={t.sheet_label}>
        <div ref={peekRef} className="sheet-peek">
          <button
            className="sheet-handle"
            aria-label={sheet === 'peek' ? t.sheet_open : t.sheet_close}
            onClick={() => {
              if (sheetDrag.current?.moved) return;
              setSheet((s) => (s === 'peek' ? 'half' : 'peek'));
            }}
            onPointerDown={(e) => {
              sheetDrag.current = { y: e.clientY, moved: false };
            }}
            onPointerMove={(e) => {
              const d = sheetDrag.current;
              if (!d || d.moved || Math.abs(e.clientY - d.y) < 30) return;
              d.moved = true;
              moveSheet(e.clientY < d.y ? 1 : -1);
            }}
            onPointerUp={() => setTimeout(() => (sheetDrag.current = null), 0)}
            data-testid="sheet-handle"
          >
            <span aria-hidden="true" />
          </button>

          {selDet && aiMode && !progress && (
            <div className="selbar" role="group" aria-label={t.selected} data-testid="det-selbar">
              <span className="selbar-title">
                {t.det_sel_title(detNumbers.get(selDet.id) ?? 0, selDet.cls !== undefined ? (t.ai_profiles_classes[selDet.cls] ?? '') : '')}
                {selDet.score !== undefined && detResult?.classes && <span className="tiny muted"> · {t.det_sel_score(`${Math.round(selDet.score * 100)} %`)}</span>}
              </span>
              {detResult?.classes && (
                <select
                  aria-label={t.det_sel_type}
                  value={selDet.cls ?? ''}
                  onChange={(e) => {
                    const c = Number(e.target.value);
                    editDetections((list) => list.map((d) => (d.id === selDet.id ? { ...d, cls: c, review: d.review === 'rejected' ? d.review : 'confirmed' } : d)));
                  }}
                  data-testid="det-type"
                >
                  {t.ai_profiles_classes.map((name, c) => (
                    <option key={c} value={c}>
                      {name}
                    </option>
                  ))}
                </select>
              )}
              {selDet.review !== 'confirmed' && (
                <button
                  className="btn small"
                  onClick={() => editDetections((list) => list.map((d) => (d.id === selDet.id ? { ...d, review: 'confirmed' } : d)))}
                  data-testid="det-confirm"
                >
                  {t.confirm}
                </button>
              )}
              <button className="btn small danger" onClick={() => detHandlers.onRemove(selDet.id)}>
                {t.remove}
              </button>
              <button className="icon" onClick={() => setSelDetId(null)} aria-label={t.sel_close} title={t.sel_close}>
                ×
              </button>
            </div>
          )}

          {selected && mode === 'circles' && !progress && (
            <div className="selbar" role="group" aria-label={t.selected}>
              <span className="selbar-title">
                {t.sel_title(numbers.get(selected.id) ?? 0, project.calibration ? `${fmt(2 * selected.r * project.calibration.mmPerPx, 1)} mm` : `${fmt(2 * selected.r, 0)} px`)}
              </span>
              {project.groups.length > 1 && (
                <select
                  aria-label={t.group}
                  value={selected.groupId ?? ''}
                  onChange={(e) => e.target.value && commit((p) => ({ ...p, objects: setObjectGroup(p.objects, [selected.id], e.target.value) }))}
                >
                  {project.groups.map((g) => (
                    <option key={g.id} value={g.id}>
                      {g.label}
                    </option>
                  ))}
                </select>
              )}
              {selected.review !== 'confirmed' && (
                <button className="btn small" onClick={() => commit((p) => ({ ...p, objects: setReview(p.objects, [selected.id], 'confirmed') }))}>
                  {t.confirm}
                </button>
              )}
              <button className="btn small danger" onClick={() => handlers.onRemove(selected.id)}>
                {t.remove}
              </button>
              <button className="icon" onClick={() => setSelectedId(null)} aria-label={t.sel_close} title={t.sel_close}>
                ×
              </button>
            </div>
          )}

          {modeSwitch}
          {notice && <div className="banner info small">{notice}</div>}
          <div className="quickbar">
            {totalButton}
            {mode === 'circles' ? (
              <>
                {stats.total > 0 && (
                  <button className="btn small" onClick={() => void exportPng()} title={t.export_png} data-testid="quick-png">
                    ⬇ PNG
                  </button>
                )}
              </>
            ) : (
              <>
                {aiStatus.decision && !aiStatus.downloaded ? (
                  // First use: the model download is the next step, not a disabled count button.
                  <button className="btn small primary" disabled={!!dl} onClick={() => void downloadModel()} data-testid="quick-ai-download">
                    {dl ? t.ai_downloading(Math.floor((100 * dl.loaded) / Math.max(1, dl.total))) : t.ai_download(fmt(aiStatus.missingBytes / 1e6, 0))}
                  </button>
                ) : (
                  <button className="btn small primary" disabled={!canRunAi} onClick={runAi} data-testid="quick-ai-run">
                    {detModel ? (detResult ? t.ai_rerun_detect : t.ai_run_detect) : aiResult ? t.ai_rerun : t.ai_run}
                  </button>
                )}
                {anyAiResult && (
                  <button className="btn small" onClick={() => void exportPng()} title={t.export_png} data-testid="quick-png">
                    ⬇ PNG
                  </button>
                )}
              </>
            )}
            <button className="btn small" onClick={() => setNameOpen((o) => !o)} aria-expanded={nameOpen} title={project.name} data-testid="quick-name">
              ✎ {t.name_button}
            </button>
            {nameOpen && (
              <label className="qb-caption">
                <span className="tiny muted">
                  {t.name_label} · {t.name_hint}
                </span>
                <input
                  value={nameDraft}
                  maxLength={120}
                  onChange={(e) => setNameDraft(e.target.value)}
                  onBlur={saveName}
                  onKeyDown={(e) => {
                    if (e.key === 'Enter') {
                      saveName();
                      setNameOpen(false);
                    }
                  }}
                  onFocus={(e) => e.target.select()}
                  autoFocus
                  data-testid="name-input"
                />
              </label>
            )}
            {groupChips && (
              <div className="qb-groups" role="list" aria-label={t.groups_title} data-testid="quick-groups">
                {groupChips}
              </div>
            )}
            {mode === 'ai' && !project.ai?.exemplars.length && !progress && <p className="qb-hint">{t.ai_need_exemplar}</p>}
            {aiDownloadError && aiMode && <p className="qb-warn">{aiDownloadError}</p>}
            {mode === 'profiles' && !!detUncertain?.size && !progress && (
              <p className="qb-warn" data-testid="quick-uncertain">
                ⚠ {t.ai_uncertain_short(detUncertain.size)}
              </p>
            )}
            {mode === 'ai' && (estimateTooLow(aiSummary) || exemplarsTooDifferent(project.ai?.exemplars ?? [])) && (
              <p className="qb-warn" data-testid="quick-ai-warn">
                ⚠ {estimateTooLow(aiSummary) ? t.ai_warn_low_short : t.ai_warn_sizes_short(fmt(exemplarSizeRatio(project.ai?.exemplars ?? []), 1))}
              </p>
            )}
          </div>
          {mode === 'profiles' && detResult?.classes && (
            <div className="sensbar">
              <label htmlFor="sens-profiles">{t.sensitivity}</label>
              <span className="tiny muted">{t.sens_less}</span>
              <input
                id="sens-profiles"
                type="range"
                min={0}
                max={100}
                value={sensitivityFromMinScore(detResult.minScore ?? SCORE_MIN)}
                onChange={(e) => {
                  const m = minScoreFromSensitivity(Number(e.target.value));
                  commit((p) => (isDetectionResult(p.ai?.result) ? { ...p, ai: { ...p.ai!, result: { ...p.ai!.result, minScore: m } } } : p), true);
                }}
                aria-describedby="sens-profiles-hint"
                data-testid="sens-profiles"
              />
              <span className="tiny muted">{t.sens_more}</span>
              <output htmlFor="sens-profiles">{sensitivityFromMinScore(detResult.minScore ?? SCORE_MIN)}</output>
              <span id="sens-profiles-hint" hidden>
                {t.profiles_sens_hint}
              </span>
            </div>
          )}
          {mode === 'circles' && (
            // Always visible: the marks on the photo show the effect while the slider moves.
            <div className="sensbar">
              <label htmlFor="sens-quick">{t.sensitivity}</label>
              <span className="tiny muted">{t.sens_less}</span>
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
              <span className="tiny muted">{t.sens_more}</span>
              <output htmlFor="sens-quick">{sens}</output>
              <span id="sens-quick-hint" hidden>
                {t.sensitivity_hint} {t.rerun_hint}
              </span>
            </div>
          )}
        </div>

        <div className="sheet-body">
          <p className="muted tiny tool-hint">{toolHelp}</p>
          {project.countArea && (
            <div className="row">
              <span className="small">{t.area_active}</span>
              <button className="btn small" onClick={() => setCountArea(null)} title={t.area_clear_hint}>
                {t.area_reset}
              </button>
            </div>
          )}
          {aiMode ? (
            <>
              <div ref={resultRef}>
                {detResult ? (
                  <AiDetectionResultPanel project={project} count={detCount} />
                ) : aiResult || aiModel === 'loca' ? (
                  <AiResultPanel project={project} summary={aiSummary} />
                ) : null}
              </div>
              <AiPanel
                project={project}
                commit={commit}
                status={aiStatus}
                downloadError={aiDownloadError}
                running={!!progress}
                onDownload={() => void downloadModel()}
                onDeleteModel={() => void deleteModel()}
                onChoice={(c) => void aiClient.setChoice(c)}
                onRun={() => void runAi()}
                summary={aiSummary}
                pending={pendingRegion}
                onPendingDone={finishRegion}
                regionError={regionError}
                selectedRegionId={selectedRegionId}
                onSelectRegion={setSelectedRegionId}
              />
            </>
          ) : (
            <>
              <div ref={resultRef}>
                <ResultPanel stats={stats} onConfirmAll={confirmAll} />
              </div>
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
              <details className="more" data-testid="more-options">
                <summary>{t.more_options}</summary>
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
              </details>
            </>
          )}
          <ExportPanel project={project} canvas={session.canvas} commit={commit} />
          <button className="btn ghost" onClick={onNewPhoto}>
            {t.new_photo}
          </button>
        </div>
      </aside>
    </main>
  );
}
