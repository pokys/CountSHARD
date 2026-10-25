import { useEffect, useState } from 'react';
import { useI18n } from '../i18n';
import { type AiState, type Project, inCountArea, isDensityResult, isDetectionResult } from '../model/types';
import { type AiStatus, aiClient } from '../ai/client';
import { type EstimateSummary, estimateTooLow, exemplarSizeRatio, exemplarsTooDifferent, removeCorrection, removeExemplar, resultIsStale, roundEstimate, setCorrectionCount } from '../ai/estimate';
import { type AiModelChoice, heavyModelWarning } from '../ai/device';
import { EXEMPLAR_FIT_BIG, EXEMPLAR_FIT_SMALL, GECO2_MIN_EXEMPLAR_PX, exemplarFit, geco2ExemplarSide } from '../ai/geco2';
import type { Rect } from '../ai/estimate';
import type { Commit } from './Workspace';

export interface PendingRegion extends Rect {
  /** Model's density sum inside the drawn region. */
  model: number | null;
}

const withAi = (p: Project, fn: (ai: AiState) => AiState): Project => ({ ...p, ai: fn(p.ai ?? { exemplars: [], result: null, corrections: [] }) });

/** Result card for a GeCo2 run: individually detected (and user-reviewed) items. */
export function AiDetectionResultPanel({ project, count }: { project: Project; count: number }) {
  const { t, fmt } = useI18n();
  const res = project.ai?.result;
  if (!isDetectionResult(res)) return null;
  const inArea = res.objects.filter((d) => inCountArea(project.countArea, d.x + d.w / 2, d.y + d.h / 2));
  const counted = inArea.filter((d) => d.review !== 'rejected');
  const profiles = !!res.classes;
  const small = !profiles && geco2ExemplarSide(res.exemplarsUsed, res.scale) < GECO2_MIN_EXEMPLAR_PX;
  const fit = profiles ? null : exemplarFit(res.exemplarsUsed, res.objects);
  const perClass = profiles
    ? t.ai_profiles_classes
        .map((name, c) => ({ name, n: counted.filter((d) => d.cls === c).length }))
        .filter((x) => x.n > 0)
    : [];
  return (
    <section className="card result" aria-live="polite">
      <div className="total">
        <span className="label">{t.ai_det_tag}</span>
        <span className="value" data-testid="ai-det-total">
          {count}
        </span>
        <span className="unit">{t.pcs}</span>
      </div>
      <p className="small">
        {t.ai_det_counts(counted.filter((d) => d.source === 'auto').length, counted.filter((d) => d.source === 'manual').length, inArea.filter((d) => d.review === 'rejected').length)}
      </p>
      {perClass.length > 0 && (
        <p className="small" data-testid="ai-profiles-classes">
          {perClass.map((x) => `${x.name}: ${x.n}`).join(' · ')}
        </p>
      )}
      {!profiles && project.ai && resultIsStale(project.ai) && <p className="small warn">{t.ai_stale}</p>}
      {small && <p className="small warn">{t.ai_det_small}</p>}
      {fit !== null && fit > EXEMPLAR_FIT_BIG && (
        <p className="small warn" data-testid="ai-fit-warn">
          {t.ai_det_ex_big(fmt(fit, 1))}
        </p>
      )}
      {fit !== null && fit < EXEMPLAR_FIT_SMALL && (
        <p className="small warn" data-testid="ai-fit-warn">
          {t.ai_det_ex_small(fmt(1 / fit, 1))}
        </p>
      )}
      <p className="muted tiny">{t.ai_det_note}</p>
      <p className="muted tiny">{t.ai_det_info(fmt(res.durationMs / 1000, 1), res.engine, res.inputSize)}</p>
    </section>
  );
}

/** Result card for Mode B: the corrected estimate, clearly labelled as an estimate. */
export function AiResultPanel({ project, summary }: { project: Project; summary: EstimateSummary | null }) {
  const { t, fmt } = useI18n();
  const res = isDensityResult(project.ai?.result) ? project.ai!.result : null;
  return (
    <section className="card result" aria-live="polite">
      <div className="total">
        <span className="label estimate-tag">{t.ai_estimate_label}</span>
        <span className="value" data-testid="ai-estimate">
          {summary ? t.ai_estimate(roundEstimate(summary.corrected)) : '–'}
        </span>
        <span className="unit">{t.pcs}</span>
      </div>
      {summary && project.ai!.corrections.length > 0 && (
        <p className="small">{t.ai_raw_vs_corrected(roundEstimate(summary.raw), roundEstimate(summary.corrected))}</p>
      )}
      {res && project.ai && resultIsStale(project.ai) && <p className="small warn">{t.ai_stale}</p>}
      {res && res.exemplarsUsed.length < res.inputExemplarIds.length && <p className="small warn">{t.ai_exemplars_partial(res.exemplarsUsed.length, res.inputExemplarIds.length)}</p>}
      {res?.limitedByTiles && <p className="small warn">{t.ai_limited}</p>}
      {estimateTooLow(summary) && <p className="small warn" data-testid="ai-warn-low">{t.ai_warn_low}</p>}
      {res && project.ai && exemplarsTooDifferent(project.ai.exemplars) && (
        <p className="small warn">{t.ai_warn_sizes(fmt(exemplarSizeRatio(project.ai.exemplars), 1))}</p>
      )}
      <p className="muted tiny">{t.ai_result_note}</p>
      {res && (
        <p className="muted tiny">
          {t.ai_info(fmt(res.durationMs / 1000, 1), res.tiles, res.engine)}
        </p>
      )}
    </section>
  );
}

export function AiPanel({
  project,
  commit,
  status,
  downloadError,
  running,
  onDownload,
  onDeleteModel,
  onChoice,
  onRun,
  summary,
  pending,
  onPendingDone,
  regionError,
  selectedRegionId,
  onSelectRegion,
}: {
  project: Project;
  commit: Commit;
  status: AiStatus;
  downloadError: string | null;
  running: boolean;
  onDownload(): void;
  onDeleteModel(): void;
  onChoice(c: AiModelChoice): void;
  onRun(): void;
  summary: EstimateSummary | null;
  pending: PendingRegion | null;
  /** count = null → discard the drawn region. */
  onPendingDone(count: number | null): void;
  regionError: string | null;
  selectedRegionId: string | null;
  onSelectRegion(id: string | null): void;
}) {
  const { t, fmt } = useI18n();
  const ai = project.ai;
  const exemplars = ai?.exemplars ?? [];
  const mb = fmt(status.missingBytes / 1e6, 0);
  const dl = status.downloading;
  const [count, setCount] = useState('');
  useEffect(() => setCount(pending?.model !== null && pending?.model !== undefined ? String(roundEstimate(pending.model)) : ''), [pending]);
  const countVal = Number(count);
  const geco2 = status.active === 'geco2';
  const profiles = status.active === 'profiles';
  const modelName = (m: string | undefined) => (m === 'geco2' ? 'GeCo2' : m === 'profiles' ? t.ai_model_profiles_short : 'LOCA');
  const plan = status.geco2;
  const backendName = (b: 'webgpu' | 'wasm') => (b === 'webgpu' ? 'WebGPU' : 'WASM');
  const sec = (s: number | null | undefined) => fmt(s ?? 0, 0);
  const why = status.decision?.reason;
  const autoText =
    status.choice !== 'auto' || !status.decision
      ? null
      : why === 'geco2-fast' && plan
        ? t.ai_auto_geco2(sec(plan.seconds), backendName(plan.backend))
        : why === 'geco2-slow'
          ? t.ai_auto_loca_slow(sec(plan?.seconds))
          : why === 'low-memory'
            ? t.ai_auto_loca_memory
            : t.ai_auto_loca_unknown;
  const warn = status.choice === 'geco2' && status.profile ? heavyModelWarning(status.profile) : null;
  const densityResult = isDensityResult(ai?.result) ? ai!.result : null;

  return (
    <section className="card">
      <h2>{profiles ? t.ai_title_profiles : geco2 ? t.ai_title_geco2 : t.ai_title}</h2>
      <p className="muted small">{profiles ? t.ai_intro_profiles : geco2 ? t.ai_intro_geco2 : t.ai_intro}</p>

      {status.crash?.model === 'geco2' && !profiles && (
        <div className="banner warn small" data-testid="ai-crash-geco2">
          <p>{t.ai_crash_geco2}</p>
          <div className="row wrap">
            {status.choice !== 'loca' && (
              <button
                className="btn small primary"
                onClick={() => {
                  aiClient.dismissCrash();
                  onChoice('loca');
                }}
              >
                {t.ai_switch_loca}
              </button>
            )}
            <button className="btn small ghost" onClick={() => aiClient.dismissCrash()}>
              {t.ai_dismiss}
            </button>
          </div>
        </div>
      )}

      {/* One line by default (the automatic choice usually fits); the choice opens on tap. */}
      {!profiles && (
        <details className="model-pick">
          <summary className="small" data-testid="ai-model-summary">
            {t.ai_model_line(status.choice === 'auto' ? t.ai_model_auto : modelName(status.choice), modelName(status.decision?.model), status.choice === 'auto')}
          </summary>
          <label className="stack">
            <span className="small">{t.ai_model_label}</span>
            <select value={status.choice} disabled={running || !!dl} onChange={(e) => onChoice(e.target.value as AiModelChoice)} data-testid="ai-model">
              <option value="auto">{t.ai_model_auto}</option>
              <option value="loca">{t.ai_model_loca}</option>
              <option value="geco2">{t.ai_model_geco2}</option>
            </select>
          </label>
          {status.choice === 'auto' && !status.decision && <p className="muted tiny">{t.ai_auto_checking}</p>}
          {autoText && <p className="muted tiny" data-testid="ai-auto-reason">{autoText}</p>}
          {status.choice === 'geco2' && plan?.seconds !== null && plan?.seconds !== undefined && !warn && (
            <p className="muted tiny">{t.ai_geco2_time(sec(plan.seconds), plan.measured)}</p>
          )}
          <p className="muted tiny">{t.ai_model_note}</p>
        </details>
      )}
      {warn && (
        <p className="small warn" data-testid="ai-heavy-warn">
          {warn.kind === 'memory' ? t.ai_geco2_warn_memory : t.ai_geco2_warn_slow(sec(warn.seconds))}
        </p>
      )}

      {!status.downloaded && status.decision && (
        <div className="ai-download">
          <button className="btn primary" disabled={!!dl} onClick={onDownload} data-testid="ai-download">
            {dl ? t.ai_downloading(Math.floor((100 * dl.loaded) / Math.max(1, dl.total))) : t.ai_download(mb)}
          </button>
          {dl && <progress value={dl.loaded} max={dl.total} />}
          <p className="muted tiny">{t.ai_download_note}</p>
          {downloadError && <p className="small warn">{downloadError}</p>}
        </div>
      )}

      {!profiles && <h3 className="small">{t.ai_step_exemplars}</h3>}
      {profiles ? null : exemplars.length === 0 ? (
        <p className="muted small">{t.ai_tool_exemplar_hint}</p>
      ) : (
        <ol className="ms-list small">
          {exemplars.map((e, i) => (
            <li key={e.id}>
              {t.ai_exemplar(i + 1)} · {fmt(e.w, 0)} × {fmt(e.h, 0)} px{' '}
              <button className="btn ghost small" disabled={running} onClick={() => commit((p) => withAi(p, (a) => removeExemplar(a, e.id)))}>
                {t.ai_remove}
              </button>
            </li>
          ))}
        </ol>
      )}
      {!profiles && exemplarsTooDifferent(exemplars) && (
        <p className="small warn" data-testid="ai-warn-sizes">
          {t.ai_warn_sizes(fmt(exemplarSizeRatio(exemplars), 1))}
        </p>
      )}
      <button className="btn primary" disabled={running || !status.downloaded || (!profiles && exemplars.length === 0)} onClick={onRun} data-testid="ai-run">
        {geco2 || profiles ? (isDetectionResult(ai?.result) ? t.ai_rerun_detect : t.ai_run_detect) : densityResult ? t.ai_rerun : t.ai_run}
      </button>
      {!profiles && status.downloaded && exemplars.length === 0 && <p className="muted tiny">{t.ai_need_exemplar}</p>}
      {!geco2 && !profiles && status.backend === 'wasm' && <p className="muted tiny">{t.ai_time_hint}</p>}

      {densityResult && ai && (
        <>
          <h3 className="small">{t.ai_corrections_title}</h3>
          <p className="muted tiny">{t.ai_corrections_hint}</p>
          {pending && (
            <div className="ai-pending" data-testid="ai-pending">
              <p className="small">{pending.model !== null ? t.ai_region_model(fmt(pending.model, 1)) : ''}</p>
              <label className="inline">
                <span>{t.ai_region_user}</span>
                <input inputMode="numeric" value={count} onChange={(e) => setCount(e.target.value.replace(/[^0-9]/g, ''))} size={6} data-testid="ai-region-count" />
              </label>
              <div className="row wrap">
                <button className="btn small primary" disabled={count === '' || !Number.isInteger(countVal)} onClick={() => onPendingDone(countVal)} data-testid="ai-region-save">
                  {t.confirm}
                </button>
                <button className="btn small ghost" onClick={() => onPendingDone(null)}>
                  {t.cancel}
                </button>
              </div>
            </div>
          )}
          {regionError && <p className="small warn">{regionError}</p>}
          {ai.corrections.length > 0 && (
            <ol className="ms-list small">
              {ai.corrections.map((c, i) => (
                <li key={c.id} className={c.id === selectedRegionId ? 'sel' : ''}>
                  <button className="linklike" onClick={() => onSelectRegion(c.id)}>
                    {t.ai_region(i + 1)}
                  </button>{' '}
                  · {t.ai_region_model(fmt(summary?.regions.find((q) => q.id === c.id)?.model ?? 0, 1))} ·{' '}
                  <input
                    aria-label={t.ai_region_user}
                    inputMode="numeric"
                    size={5}
                    value={c.userCount}
                    onChange={(e) => {
                      const v = Number(e.target.value.replace(/[^0-9]/g, '') || '0');
                      commit((p) => withAi(p, (a) => setCorrectionCount(a, c.id, v)));
                    }}
                  />{' '}
                  {t.pcs}{' '}
                  <button className="btn ghost small" onClick={() => commit((p) => withAi(p, (a) => removeCorrection(a, c.id)))}>
                    {t.ai_remove}
                  </button>
                </li>
              ))}
            </ol>
          )}
        </>
      )}

      <p className="muted tiny">{profiles ? t.ai_source_profiles : geco2 ? t.ai_licence_geco2 : t.ai_licence}</p>
      {status.downloaded && !running && (
        <button className="btn ghost small" onClick={onDeleteModel}>
          {t.ai_delete}
        </button>
      )}
    </section>
  );
}
