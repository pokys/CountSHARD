import { useRef, useState } from 'react';
import { useI18n } from '../i18n';
import type { ProjectSummary } from '../io/db';

interface Props {
  busy: boolean;
  recent: ProjectSummary[];
  onFile(file: Blob, name: string): void;
  onDemo(): void;
  onOpen(id: string): void;
  onDelete(p: ProjectSummary): void;
  onImport(file: File): void;
}

export function StartScreen({ busy, recent, onFile, onDemo, onOpen, onDelete, onImport }: Props) {
  const { t, lang } = useI18n();
  const [drag, setDrag] = useState(false);
  const fileRef = useRef<HTMLInputElement>(null);
  const camRef = useRef<HTMLInputElement>(null);
  const jsonRef = useRef<HTMLInputElement>(null);

  const pick = (files: FileList | null) => {
    const f = files?.[0];
    if (!f) return;
    if (f.name.toLowerCase().endsWith('.json')) onImport(f);
    else onFile(f, f.name);
  };

  return (
    <main className="start">
      <p className="tagline">{t.tagline}</p>
      <section
        className={`dropzone ${drag ? 'drag' : ''}`}
        onDragOver={(e) => {
          e.preventDefault();
          setDrag(true);
        }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => {
          e.preventDefault();
          setDrag(false);
          pick(e.dataTransfer.files);
        }}
        aria-busy={busy}
      >
        <svg width="56" height="56" viewBox="0 0 24 24" aria-hidden="true" className="drop-icon">
          <circle cx="7" cy="8" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="15" cy="8" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="11" cy="15" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.5" />
          <circle cx="19" cy="15" r="3.2" fill="none" stroke="currentColor" strokeWidth="1.5" />
        </svg>
        <h1>{t.drop_title}</h1>
        <p className="muted">{t.drop_hint}</p>
        <div className="row wrap center">
          <button className="btn primary" disabled={busy} onClick={() => fileRef.current?.click()}>
            {t.pick_file}
          </button>
          <button className="btn" disabled={busy} onClick={() => camRef.current?.click()}>
            {t.take_photo}
          </button>
          <button className="btn ghost" disabled={busy} onClick={onDemo}>
            {t.try_demo}
          </button>
        </div>
        <input ref={fileRef} type="file" accept="image/*,.heic,.heif" hidden onChange={(e) => pick(e.target.files)} data-testid="file-input" />
        <input ref={camRef} type="file" accept="image/*" capture="environment" hidden onChange={(e) => pick(e.target.files)} />
        <input ref={jsonRef} type="file" accept="application/json,.json" hidden onChange={(e) => pick(e.target.files)} />
      </section>

      <div className="start-grid">
        <section className="card">
          <h2>{t.recent}</h2>
          {recent.length === 0 ? (
            <p className="muted">{t.recent_empty}</p>
          ) : (
            <ul className="recent">
              {recent.slice(0, 12).map((p) => (
                <li key={p.id}>
                  <button className="link" onClick={() => onOpen(p.id)}>
                    <strong>{p.name}</strong>
                    <span className="muted">
                      {p.total} {t.pcs} · {new Date(p.updatedAt).toLocaleString(lang === 'cs' ? 'cs-CZ' : 'en-GB', { dateStyle: 'short', timeStyle: 'short' })}
                    </span>
                  </button>
                  <button className="btn ghost small" onClick={() => onDelete(p)} aria-label={`${t.delete} ${p.name}`}>
                    {t.delete}
                  </button>
                </li>
              ))}
            </ul>
          )}
          <button className="btn ghost small" onClick={() => jsonRef.current?.click()}>
            {t.import_project}
          </button>
        </section>
        <section className="card">
          <h2>{t.tips_title}</h2>
          <ul className="tips">
            {t.tips.map((tip) => (
              <li key={tip}>{tip}</li>
            ))}
          </ul>
        </section>
      </div>
    </main>
  );
}
