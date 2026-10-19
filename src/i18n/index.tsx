import { createContext, useContext, useMemo, useState, type ReactNode } from 'react';
import { cs } from './cs';
import { en } from './en';

export type Lang = 'cs' | 'en';
export type Messages = typeof cs;
const MESSAGES: Record<Lang, Messages> = { cs, en };

const KEY = 'countit.lang';

function initialLang(): Lang {
  try {
    const saved = localStorage.getItem(KEY);
    if (saved === 'cs' || saved === 'en') return saved;
  } catch {
    /* storage unavailable */
  }
  return 'cs';
}

interface I18n {
  lang: Lang;
  t: Messages;
  setLang: (l: Lang) => void;
  /** Number formatting for the active locale. */
  fmt: (v: number, digits?: number) => string;
}

const Ctx = createContext<I18n | null>(null);

export function I18nProvider({ children }: { children: ReactNode }) {
  const [lang, setLangState] = useState<Lang>(initialLang);
  const value = useMemo<I18n>(() => {
    const locale = lang === 'cs' ? 'cs-CZ' : 'en-GB';
    return {
      lang,
      t: MESSAGES[lang],
      setLang: (l) => {
        setLangState(l);
        document.documentElement.lang = l;
        try {
          localStorage.setItem(KEY, l);
        } catch {
          /* ignore */
        }
      },
      fmt: (v, digits = 0) => v.toLocaleString(locale, { minimumFractionDigits: digits, maximumFractionDigits: digits }),
    };
  }, [lang]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

export function useI18n(): I18n {
  const v = useContext(Ctx);
  if (!v) throw new Error('I18nProvider missing');
  return v;
}
