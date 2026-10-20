import { useEffect, useState } from 'react';
import { useI18n } from '../i18n';
import type { AiState, Project } from '../model/types';
import type { AiStatus } from '../ai/client';
import { type EstimateSummary, estimateTooLow, exemplarSizeRatio, exemplarsTooDifferent, removeCorrection, removeExemplar, resultIsStale, roundEstimate, setCorrectionCount } from '../ai/estimate';
import { totalBytes } from '../ai/modelStore';
import type { Rect } from '../ai/estimate';
import type { Commit } from './Workspace';

export interface PendingRegion extends Rect {
  /** Model's density sum inside the drawn region. */
  model: number | null;
}

const withAi = (p: Project, fn: (ai: AiState) => AiState): Project => ({ ...p, ai: fn(p.ai ?? { exemplars: [], result: null, corrections: [] }) });

/** Result card for Mode B: the corrected estimate, clearly labelled as an estimate. */
export function AiResultPanel({ project, summary }: { project: Project; summary: EstimateSummary | null }) {
  const { t, fmt } = useI18n();
  const res = project.ai?.result;
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
  const mb = fmt(totalBytes() / 1e6, 0);
  const dl = status.downloading;
  const [count, setCount] = useState('');
  useEffect(() => setCount(pending?.model !== null && pending?.model !== undefined ? String(roundEstimate(pending.model)) : ''), [pending]);
  const countVal = Number(count);

  return (
    <section className="card">
      <h2>{t.ai_title}</h2>
      <p className="muted small">{t.ai_intro}</p>

      {!status.downloaded && (
        <div className="ai-download">
          <button className="btn primary" disabled={!!dl} onClick={onDownload} data-testid="ai-download">
            {dl ? t.ai_downloading(Math.floor((100 * dl.loaded) / Math.max(1, dl.total))) : t.ai_download(mb)}
          </button>
          {dl && <progress value={dl.loaded} max={dl.total} />}
          <p className="muted tiny">{t.ai_download_note}</p>
          {downloadError && <p className="small warn">{downloadError}</p>}
        </div>
      )}

      <h3 className="small">{t.ai_step_exemplars}</h3>
      {exemplars.length === 0 ? (
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
      {exemplarsTooDifferent(exemplars) && (
        <p className="small warn" data-testid="ai-warn-sizes">
          {t.ai_warn_sizes(fmt(exemplarSizeRatio(exemplars), 1))}
        </p>
      )}
      <button className="btn primary" disabled={running || !status.downloaded || exemplars.length === 0} onClick={onRun} data-testid="ai-run">
        {ai?.result ? t.ai_rerun : t.ai_run}
      </button>
      {status.downloaded && exemplars.length === 0 && <p className="muted tiny">{t.ai_need_exemplar}</p>}
      {status.backend === 'wasm' && <p className="muted tiny">{t.ai_time_hint}</p>}

      {ai?.result && (
        <>
          <h3 className="small">{t.ai_corrections_title}</h3>
          <p className="muted tiny">{t.ai_corrections_hint}</p>
          {pending && (
            <div className="ai-pending" data-testid="ai-pending">
              <p className="small">{pending.model !== null ? t.ai_region_model(fmt(pending.model, 1)) : ''}</p>
              <label className="inline">
                <span>{t.ai_region_user}</span>
                <input inputMode="numeric" value={count} onChange={(e) => setCount(e.target.value.replace(/[^0-9]/g, ''))} size={6} autoFocus data-testid="ai-region-count" />
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
                  · {t.ai_region_model(fmt(summary?.regions[i]?.model ?? 0, 1))} ·{' '}
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

      <p className="muted tiny">{t.ai_licence}</p>
      {status.downloaded && !running && (
        <button className="btn ghost small" onClick={onDeleteModel}>
          {t.ai_delete}
        </button>
      )}
    </section>
  );
}
