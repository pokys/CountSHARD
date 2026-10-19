import { useEffect, useState } from 'react';
import { useI18n } from '../i18n';
import { checkOffline, isFullyOffline, type OfflineStatus } from '../pwa/offline';
import { registerServiceWorker } from '../pwa/register';

export function OfflineBadge({ engineLoaded }: { engineLoaded: boolean }) {
  const { t, fmt } = useI18n();
  const [status, setStatus] = useState<OfflineStatus | null>(null);
  const [open, setOpen] = useState(false);
  const [update, setUpdate] = useState<null | (() => void)>(null);

  useEffect(() => {
    registerServiceWorker({
      onReady: () => checkOffline(engineLoaded).then(setStatus),
      onNeedRefresh: (apply) => setUpdate(() => apply),
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    let alive = true;
    const run = () => checkOffline(engineLoaded).then((s) => alive && setStatus(s));
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
      <button className={`chip ${tone}`} onClick={() => setOpen(!open)} aria-expanded={open} data-testid="offline-badge">
        <span className="dot" aria-hidden="true" /> {label}
      </button>
      {open && status && (
        <div className="popover" role="dialog" aria-label={t.offline_title}>
          <h3>{t.offline_title}</h3>
          <ul className="checklist">
            <Item ok={status.shellCached}>{t.offline_shell}</Item>
            <Item ok={status.engineCached}>{t.offline_engine_cached}</Item>
            <Item ok={status.engineLoaded}>{t.offline_engine_loaded}</Item>
            <Item ok={false}>
              {t.offline_ai}: {t.offline_ai_na}
            </Item>
            <Item ok={status.persisted}>{t.offline_persist}</Item>
          </ul>
          {status.usage !== null && status.quota !== null && (
            <p className="muted small">
              {t.offline_storage}: {mb(status.usage)} / {mb(status.quota)}
            </p>
          )}
          <p className="muted small">{t.offline_note}</p>
        </div>
      )}
    </div>
  );
}
