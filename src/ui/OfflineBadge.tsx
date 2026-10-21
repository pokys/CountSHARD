import { useEffect, useState } from 'react';
import { useI18n } from '../i18n';
import { checkOffline, isFullyOffline, type OfflineStatus } from '../pwa/offline';
import { registerServiceWorker } from '../pwa/register';
import { iosBrowserTab, refreshApp, wipeEverything } from '../pwa/maintenance';
import { type AiStatus, aiClient } from '../ai/client';
import { aiAssets, geco2Assets, modelBytes } from '../ai/modelStore';

export function OfflineBadge({ engineLoaded }: { engineLoaded: boolean }) {
  const { t, fmt } = useI18n();
  const [status, setStatus] = useState<OfflineStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [update, setUpdate] = useState<null | (() => void)>(null);
  const [ai, setAi] = useState<AiStatus>(aiClient.status);
  useEffect(() => aiClient.subscribe(setAi), []);

  useEffect(() => {
    registerServiceWorker({
      onReady: () => checkOffline(engineLoaded).then(setStatus),
      onNeedRefresh: (apply) => setUpdate(() => apply),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let alive = true;
    const run = () => {
      void aiClient.refresh();
      return checkOffline(engineLoaded).then((s) => alive && setStatus(s));
    };
    run();
    const id = setInterval(run, 15000);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [engineLoaded]);

  const label = !status ? t.offline_checking : status.unsupported ? t.offline_dev : isFullyOffline(status) ? t.offline_ready : t.offline_partial;
  const tone = !status ? '' : status.unsupported ? 'warn' : isFullyOffline(status) ? 'ok' : 'warn';
  const mb = (b: number) => `${fmt(b / 1048576, 0)} MB`;
  const Item = ({ ok, children }: { ok: boolean; children: React.ReactNode }) => (
    <li className={ok ? 'ok' : 'no'}>
      <span aria-hidden="true">{ok ? '✓' : '–'}</span> {children}
    </li>
  );

  return (
    <div className="offline">
      {update && (
        <button className="btn small primary" onClick={update} title={t.update_available}>
          {t.update_now}
        </button>
      )}
      <button className={`chip ${tone}`} onClick={() => setOpen(!open)} aria-expanded={open} aria-label={label} title={label} data-testid="offline-badge">
        <span className="dot" aria-hidden="true" /> <span className="chip-label">{label}</span>
      </button>
      {open && status && (
        <div className="popover" role="dialog" aria-label={t.offline_title}>
          <h3>{t.offline_title}</h3>
          <ul className="checklist">
            <Item ok={status.shellCached}>{t.offline_shell}</Item>
            <Item ok={status.engineCached}>{t.offline_engine_cached}</Item>
            <Item ok={status.engineLoaded}>{t.offline_engine_loaded}</Item>
            {/* Mode B is optional: downloaded on first use, then cached for offline use. */}
            <Item ok={ai.downloaded}>{ai.downloaded ? t.offline_ai_downloaded : t.offline_ai_not_downloaded}</Item>
            <Item ok={ai.ready}>
              {t.offline_ai_ready}
              {ai.ready && ai.readyModel ? ` (${ai.readyModel === 'geco2' ? 'GeCo2' : 'LOCA'}, ${ai.backend === 'webgpu' ? 'WebGPU' : 'WASM'})` : ''}
            </Item>
            <Item ok={status.persisted}>{t.offline_persist}</Item>
          </ul>
          <h4 className="small">{t.offline_ai_models}</h4>
          {!ai.models.loca && !ai.models.geco2_768 && !ai.models.geco2_1024 ? (
            <p className="muted tiny">{t.offline_model_none}</p>
          ) : (
            <ul className="checklist" data-testid="offline-models">
              {(
                [
                  ['loca', ai.models.loca, t.offline_model_loca(mb(modelBytes(aiAssets())))],
                  ['geco2_768', ai.models.geco2_768, t.offline_model_geco2(768, mb(modelBytes(geco2Assets(768))))],
                  ['geco2_1024', ai.models.geco2_1024, t.offline_model_geco2(1024, mb(modelBytes(geco2Assets(1024))))],
                ] as const
              )
                .filter(([, ok]) => ok)
                .map(([key, , label]) => (
                  <li key={key} className="ok">
                    <span aria-hidden="true">✓</span> {label}{' '}
                    <button className="btn ghost small" onClick={() => void aiClient.deleteModel(key)} data-testid={`offline-delete-${key}`}>
                      {t.offline_model_delete}
                    </button>
                  </li>
                ))}
            </ul>
          )}
          {status.usage !== null && status.quota !== null && (
            <p className="muted small">
              {t.offline_storage}: {mb(status.usage)} / {mb(status.quota)}
            </p>
          )}
          <p className="muted small">{t.offline_note}</p>
          {!status.persisted && iosBrowserTab() && <p className="small warn">{t.maint_ios_home}</p>}
          <details className="maint">
            <summary className="small">{t.maint_title}</summary>
            <p className="muted tiny">{t.maint_note}</p>
            <div className="row wrap">
              <button className="btn small" onClick={() => void refreshApp()} title={t.maint_refresh_hint}>
                {t.maint_refresh}
              </button>
              <button className="btn small danger" onClick={() => confirm(t.maint_wipe_confirm) && void wipeEverything()}>
                {t.maint_wipe}
              </button>
            </div>
            <p className="muted tiny">{t.maint_refresh_hint}</p>
          </details>
        </div>
      )}
    </div>
  );
}
