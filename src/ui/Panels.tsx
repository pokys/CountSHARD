import { useEffect, useState } from 'react';
import { useI18n } from '../i18n';
import { type CountedObject, type Project, isCounted } from '../model/types';
import { MIN_GROUP_SIZE } from '../analysis/clustering';
import {
  type ProjectStats,
  applyGrouping,
  calibrateFromGroup,
  calibrateFromObject,
  mergeGroups,
  setObjectGroup,
  setReview,
  splitGroup,
  unlockAllGroups,
  updateObject,
} from '../state/project';
import { blobToDataUrl, downloadBlob, renderAnnotatedPng, safeFilename } from '../io/export';
import { canvasToBlob } from '../io/image';
import { toCsv, toProjectFile } from '../io/serialize';
import type { Commit } from './Workspace';

/** Spread above which a size group is flagged for review (MAD/median). */
const SPREAD_WARN = 0.08;

function useDiameterFormat() {
  const { fmt } = useI18n();
  return (px: number, mm: number | null) => (mm !== null ? `${fmt(mm, 1)} mm` : `${fmt(px, 0)} px`);
}

export function ResultPanel({ stats, onConfirmAll }: { stats: ProjectStats; onConfirmAll(): void }) {
  const { t } = useI18n();
  return (
    <section className="card result" aria-live="polite">
      <div className="total">
        <span className="label">{t.total}</span>
        <span className="value" data-testid="total-count">
          {stats.total}
        </span>
        <span className="unit">{t.pcs}</span>
      </div>
      <p className="muted small">{t.breakdown(stats.auto, stats.manual, stats.rejected)}</p>
      <div className="row between">
        <span className="small">{t.review_status(stats.confirmed, stats.total)}</span>
        {stats.unreviewed > 0 && (
          <button className="btn ghost small" onClick={onConfirmAll}>
            {t.confirm_all}
          </button>
        )}
      </div>
      <p className="muted tiny">{t.result_note}</p>
    </section>
  );
}

export function GroupsPanel({ project, stats, commit }: { project: Project; stats: ProjectStats; commit: Commit }) {
  const { t, fmt } = useI18n();
  const fd = useDiameterFormat();
  const g = project.grouping;
  const regroup = (grouping = project.grouping) => commit((p) => ({ ...p, grouping, ...applyGrouping(p.objects, p.groups, grouping) }));
  if (stats.total === 0) return null;
  return (
    <section className="card">
      <h2>{t.groups_title}</h2>
      <table className="groups">
        <thead>
          <tr>
            <th>{t.group}</th>
            <th className="num">{t.count}</th>
            <th className="num">{t.diameter}</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {stats.groups.map((s) => (
            <tr key={s.group.id}>
              <td>
                <span className="swatch" style={{ background: s.group.color }} /> {s.group.label}
              </td>
              <td className="num">{s.count}</td>
              <td className="num">
                {fd(s.medianDiameterPx, s.medianDiameterMm)}
                {s.spread > SPREAD_WARN && <div className="warn tiny">{t.spread_warn}</div>}
                {s.count < MIN_GROUP_SIZE && <div className="warn tiny">{t.small_group}</div>}
              </td>
              <td className="actions">
                {stats.groups.length > 1 && (
                  <select
                    aria-label={t.merge_into}
                    value=""
                    onChange={(e) => e.target.value && commit((p) => ({ ...p, ...mergeGroups(p.objects, p.groups, s.group.id, e.target.value) }))}
                  >
                    <option value="">{t.merge_into}</option>
                    {stats.groups
                      .filter((o) => o.group.id !== s.group.id)
                      .map((o) => (
                        <option key={o.group.id} value={o.group.id}>
                          {o.group.label}
                        </option>
                      ))}
                  </select>
                )}
                {s.count >= 4 && (
                  <button className="btn ghost small" onClick={() => commit((p) => ({ ...p, ...splitGroup(p.objects, p.groups, s.group.id) }))}>
                    {t.split}
                  </button>
                )}
              </td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td>{t.total}</td>
            <td className="num">{stats.total}</td>
            <td colSpan={2} />
          </tr>
        </tfoot>
      </table>
      <div className="form-grid">
        <label>
          <span>{t.grouping_auto}</span>
          <select
            value={g.mode}
            onChange={(e) => regroup({ ...g, mode: e.target.value as 'auto' | 'fixed', k: Math.max(1, stats.groups.length) })}
          >
            <option value="auto">{t.grouping_auto}</option>
            <option value="fixed">{t.grouping_fixed}</option>
          </select>
        </label>
        {g.mode === 'fixed' ? (
          <label>
            <span>{t.grouping_k}</span>
            <input type="number" min={1} max={10} value={g.k} onChange={(e) => regroup({ ...g, k: Math.max(1, Math.min(10, Number(e.target.value) || 1)) })} />
          </label>
        ) : (
          <label>
            <span>
              {t.grouping_sep}: {fmt(g.minSeparation * 100, 0)} %
            </span>
            <input type="range" min={4} max={30} value={Math.round(g.minSeparation * 100)} onChange={(e) => regroup({ ...g, minSeparation: Number(e.target.value) / 100 })} />
          </label>
        )}
      </div>
      <div className="row wrap">
        <button className="btn small" onClick={() => regroup()} title={t.regroup_hint}>
          {t.regroup}
        </button>
        {project.objects.some((o) => o.groupLocked) && (
          <button className="btn ghost small" onClick={() => commit((p) => ({ ...p, ...applyGrouping(unlockAllGroups(p.objects), p.groups, p.grouping) }))}>
            {t.unlock_groups}
          </button>
        )}
      </div>
    </section>
  );
}

export function SelectedPanel({ obj, number, project, commit, onRemove }: { obj: CountedObject; number: number; project: Project; commit: Commit; onRemove(): void }) {
  const { t, fmt } = useI18n();
  const fd = useDiameterFormat();
  const [mm, setMm] = useState('');
  const mmVal = Number(mm.replace(',', '.'));
  return (
    <section className="card selected">
      <h2>{t.obj_number(number)}</h2>
      <p className="small">
        Ø {fd(2 * obj.r, project.calibration ? 2 * obj.r * project.calibration.mmPerPx : null)} · {obj.source === 'auto' ? t.source_auto : t.source_manual} ·{' '}
        {obj.review === 'confirmed' ? t.review_confirmed : t.review_unreviewed}
        {obj.score !== undefined && (
          <span title={t.score_hint}>
            {' '}
            · {t.score} {fmt(obj.score * 100, 0)} %
          </span>
        )}
      </p>
      <div className="form-grid">
        <label>
          <span>{t.group}</span>
          <select value={obj.groupId ?? ''} onChange={(e) => e.target.value && commit((p) => ({ ...p, objects: setObjectGroup(p.objects, [obj.id], e.target.value) }))}>
            <option value="" disabled>
              –
            </option>
            {project.groups.map((g) => (
              <option key={g.id} value={g.id}>
                {g.label}
              </option>
            ))}
          </select>
        </label>
        <label>
          <span>{t.radius}</span>
          <input
            type="range"
            min={Math.max(2, obj.r * 0.5)}
            max={obj.r * 1.5}
            step={0.5}
            value={obj.r}
            onChange={(e) => commit((p) => ({ ...p, objects: updateObject(p.objects, obj.id, { r: Number(e.target.value) }) }), true)}
            onPointerUp={() => commit((p) => ({ ...p, ...applyGrouping(p.objects, p.groups, p.grouping) }))}
          />
        </label>
      </div>
      <div className="row wrap">
        <button className="btn small danger" onClick={onRemove}>
          {t.remove}
        </button>
        {obj.review !== 'confirmed' && (
          <button className="btn small" onClick={() => commit((p) => ({ ...p, objects: setReview(p.objects, [obj.id], 'confirmed') }))}>
            {t.confirm}
          </button>
        )}
      </div>
      <div className="row wrap">
        <label className="inline">
          <span>{t.reference_mm}</span>
          <input inputMode="decimal" value={mm} onChange={(e) => setMm(e.target.value)} placeholder="26,9" size={6} />
        </label>
        <button className="btn small" disabled={!(mmVal > 0)} onClick={() => commit((p) => ({ ...p, calibration: calibrateFromObject(obj, mmVal) }))}>
          {t.set_reference}
        </button>
      </div>
    </section>
  );
}

export function CalibrationPanel({ project, commit }: { project: Project; commit: Commit }) {
  const { t, fmt } = useI18n();
  const [groupId, setGroupId] = useState('');
  const [mm, setMm] = useState('');
  const mmVal = Number(mm.replace(',', '.'));
  const cal = project.calibration;
  if (!project.objects.some(isCounted)) return null;
  return (
    <section className="card">
      <h2>{t.calib_title}</h2>
      {cal ? (
        <>
          <p className="small">{t.calib_active(fmt(cal.mmPerPx, 4), cal.referenceObjectId ? t.calib_ref_object : t.calib_ref_group)}</p>
          <p className="warn tiny">{t.calib_warning}</p>
          <button className="btn ghost small" onClick={() => commit((p) => ({ ...p, calibration: null }))}>
            {t.calib_reset}
          </button>
        </>
      ) : (
        <>
          <p className="muted small">{t.calib_none}</p>
          <p className="muted tiny">{t.calib_hint}</p>
          <div className="row wrap">
            <select value={groupId} onChange={(e) => setGroupId(e.target.value)} aria-label={t.group}>
              <option value="">{t.group}…</option>
              {project.groups.map((g) => (
                <option key={g.id} value={g.id}>
                  {g.label}
                </option>
              ))}
            </select>
            <input inputMode="decimal" value={mm} onChange={(e) => setMm(e.target.value)} placeholder={t.nominal} aria-label={t.nominal} size={10} />
            <button
              className="btn small"
              disabled={!groupId || !(mmVal > 0)}
              onClick={() =>
                commit((p) => {
                  const c = calibrateFromGroup(p.objects, groupId, mmVal);
                  return c ? { ...p, calibration: c, groups: p.groups.map((g) => (g.id === groupId ? { ...g, nominalMm: mmVal } : g)) } : p;
                })
              }
            >
              {t.calibrate_group}
            </button>
          </div>
        </>
      )}
    </section>
  );
}

export function SettingsPanel({ project, commit, running, onRun }: { project: Project; commit: Commit; running: boolean; onRun(): void }) {
  const { t } = useI18n();
  const p = project.params;
  const set = (patch: Partial<Project['params']>) => commit((pr) => ({ ...pr, params: { ...pr.params, ...patch } }));
  const est = project.analysis?.estimatedDiameterPx;
  return (
    <section className="card">
      <h2>{t.settings_title}</h2>
      <label className="stack">
        <span>
          {t.sensitivity}: {p.sensitivity}
        </span>
        <input type="range" min={0} max={100} value={p.sensitivity} onChange={(e) => set({ sensitivity: Number(e.target.value) })} />
        <span className="muted tiny">{t.sensitivity_hint}</span>
      </label>
      <div className="form-grid">
        <label>
          <span>{t.size_mode}</span>
          <select
            value={p.sizeMode}
            onChange={(e) => {
              const mode = e.target.value as 'auto' | 'manual';
              set(mode === 'manual' && est ? { sizeMode: mode, minDiameterPx: Math.round(est[0] * 0.8), maxDiameterPx: Math.round(est[1] * 1.25) } : { sizeMode: mode });
            }}
          >
            <option value="auto">{t.size_auto}</option>
            <option value="manual">{t.size_manual}</option>
          </select>
        </label>
        <label>
          <span>{t.resolution}</span>
          <select value={p.analysisMaxSide} onChange={(e) => set({ analysisMaxSide: Number(e.target.value) })}>
            <option value={1600}>{t.res_standard}</option>
            <option value={2400}>{t.res_high}</option>
          </select>
        </label>
        {p.sizeMode === 'manual' && (
          <>
            <label>
              <span>{t.min_d}</span>
              <input type="number" min={6} value={p.minDiameterPx} onChange={(e) => set({ minDiameterPx: Math.max(6, Number(e.target.value) || 6) })} />
            </label>
            <label>
              <span>{t.max_d}</span>
              <input type="number" min={8} value={p.maxDiameterPx} onChange={(e) => set({ maxDiameterPx: Math.max(p.minDiameterPx + 2, Number(e.target.value) || 8) })} />
            </label>
          </>
        )}
      </div>
      <button className="btn primary" disabled={running} onClick={onRun} data-testid="rerun">
        {t.rerun}
      </button>
      <p className="muted tiny">{t.rerun_hint}</p>
      {project.analysis && <p className="muted tiny">{t.analysis_info(project.analysis.durationMs, project.analysis.analysisWidth, project.analysis.analysisHeight)}</p>}
    </section>
  );
}

export function ExportPanel({ project, canvas, commit }: { project: Project; canvas: HTMLCanvasElement; commit: Commit }) {
  const { t, lang } = useI18n();
  const fd = useDiameterFormat();
  const [name, setName] = useState(project.name);
  useEffect(() => setName(project.name), [project.name]);
  const base = safeFilename(project.name);
  const date = new Date().toLocaleString(lang === 'cs' ? 'cs-CZ' : 'en-GB');

  const png = async () => {
    const blob = await renderAnnotatedPng(
      canvas,
      project,
      { title: t.legend_title, total: t.legend_total, group: t.legend_group, footer: t.legend_footer(date) },
      fd,
    );
    downloadBlob(blob, `${base}-${project.objects.filter(isCounted).length}.png`);
  };
  const csv = () => {
    const text = toCsv(project, {
      decimal: lang === 'cs' ? ',' : '.',
      headers: t.csv,
      sourceLabels: { auto: t.source_auto, manual: t.source_manual },
      reviewLabels: { unreviewed: t.review_unreviewed, confirmed: t.review_confirmed },
    });
    downloadBlob(new Blob([text], { type: 'text/csv;charset=utf-8' }), `${base}.csv`);
  };
  const json = async () => {
    const dataUrl = await blobToDataUrl(await canvasToBlob(canvas, 'image/jpeg', 0.92));
    const file = toProjectFile(project, dataUrl);
    downloadBlob(new Blob([JSON.stringify(file)], { type: 'application/json' }), `${base}.countshard.json`);
  };

  return (
    <section className="card">
      <h2>{t.export_title}</h2>
      <label className="stack">
        <span>{t.project_name}</span>
        <input value={name} onChange={(e) => setName(e.target.value)} onBlur={() => name.trim() && name !== project.name && commit((p) => ({ ...p, name: name.trim() }))} />
      </label>
      <div className="row wrap">
        <button className="btn primary" onClick={png} data-testid="export-png">
          {t.export_png}
        </button>
        <button className="btn" onClick={csv}>
          {t.export_csv}
        </button>
        <button className="btn" onClick={json}>
          {t.export_json}
        </button>
      </div>
      <p className="muted tiny">{t.saved_locally}</p>
    </section>
  );
}
