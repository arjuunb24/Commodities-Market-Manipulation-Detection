import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, ChevronDown, Crosshair, Database, Lightbulb, RotateCcw, Sparkles, User } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import ReactMarkdown, { type Components } from "react-markdown";
import remarkGfm from "remark-gfm";
import { Callout, PageHeader } from "@/components/ui";
import { askAnalyst, useAiStatus, type ChatContext, type ChatResponse } from "@/lib/api";
import { runLabel } from "@/lib/format";
import { marketLabel, useSelection } from "@/lib/selection";
import { useSuggestions, type Suggestion } from "./analyst/suggestions";

interface Turn {
  role: "user" | "assistant";
  content: string;
  sources?: ChatResponse["sources"];
  error?: boolean;
  context?: string;
}

/** Markdown with GitHub tables; wide tables scroll inside the bubble instead of breaking it. */
const MD: Components = {
  table: ({ children }) => (
    <div className="my-3 overflow-x-auto rounded-lg border border-ink-700">
      <table className="w-full border-collapse text-left text-[12.5px]">{children}</table>
    </div>
  ),
  thead: ({ children }) => <thead className="bg-ink-800/80">{children}</thead>,
  th: ({ children }) => <th className="whitespace-nowrap border-b border-ink-600 px-3 py-2 font-semibold text-fg">{children}</th>,
  td: ({ children }) => <td className="border-b border-ink-700/70 px-3 py-1.5 align-top text-fg-2">{children}</td>,
  code: ({ children }) => <code className="rounded bg-ink-700/80 px-1 py-0.5 font-mono text-[11.5px] text-fg">{children}</code>,
  a: ({ children, href }) => (
    <a href={href} className="text-accent underline-offset-2 hover:underline" target="_blank" rel="noreferrer">
      {children}
    </a>
  ),
  h1: ({ children }) => <div className="mb-1 mt-2 text-[14px] font-semibold text-fg">{children}</div>,
  h2: ({ children }) => <div className="mb-1 mt-2 text-[14px] font-semibold text-fg">{children}</div>,
  h3: ({ children }) => <div className="mb-1 mt-2 text-[13.5px] font-semibold text-fg">{children}</div>,
  hr: () => null,
};

const TOOL_LABEL: Record<string, string> = {
  summarize_cases: "Cases by account",
  run_overview: "Run overview",
  get_strategist_history: "Strategist history",
  query_flags: "Flagged windows",
  query_round_metrics: "Round metrics",
  get_trader_history: "Trader activity",
  list_rounds: "Rounds",
};

function Sources({ sources }: { sources: ChatResponse["sources"] }) {
  const [open, setOpen] = useState(false);
  return (
    <div className="mt-3">
      <button onClick={() => setOpen((o) => !o)} className="focus-ring inline-flex items-center gap-1.5 rounded-md border border-ink-600 px-2 py-1 text-[11.5px] text-fg-2 hover:bg-tint/[0.03]">
        <Database size={12} /> {sources.length ? `${sources.length} data lookup${sources.length > 1 ? "s" : ""}` : "No data lookups"}
        <ChevronDown size={12} className={clsx("transition-transform", open && "rotate-180")} />
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="mt-2 space-y-1.5">
              {!sources.length && <div className="text-[12px] text-fg-muted">The assistant answered without reading run data.</div>}
              {sources.map((s, i) => {
                const args = s.args && typeof s.args === "object" ? Object.entries(s.args as Record<string, unknown>) : [];
                return (
                  <div key={i} className="flex flex-wrap items-center gap-x-2 gap-y-1 rounded-md border border-ink-700 bg-ink-900 px-2.5 py-1.5 text-[11.5px]">
                    <span className="font-medium text-fg">{TOOL_LABEL[s.tool] ?? s.tool}</span>
                    <span className="font-mono text-[10.5px] text-fg-muted">{s.tool}</span>
                    {args.map(([k, v]) => (
                      <span key={k} className="rounded bg-ink-700/70 px-1.5 py-px font-mono text-[10.5px] text-fg-2">
                        {k}={String(v)}
                      </span>
                    ))}
                    {s.error && <span className="text-critical">returned an error</span>}
                  </div>
                );
              })}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function SuggestionChips({ items, onPick, disabled, compact }: { items: Suggestion[]; onPick: (s: Suggestion) => void; disabled?: boolean; compact?: boolean }) {
  if (!items.length) return null;
  return (
    <div className={clsx("flex flex-wrap gap-2", compact ? "" : "justify-center")}>
      {items.map((s, i) => (
        <motion.button
          key={s.prompt}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0, transition: { delay: 0.04 * i } }}
          disabled={disabled}
          onClick={() => onPick(s)}
          title={s.prompt}
          className={clsx(
            "focus-ring rounded-xl border border-ink-600 bg-ink-800/60 text-left text-fg-2 transition hover:-translate-y-0.5 hover:border-accent/30 hover:text-fg disabled:pointer-events-none disabled:opacity-40",
            compact ? "px-3 py-1.5 text-[12px]" : "px-4 py-3 text-[12.5px]",
          )}
        >
          {s.label}
        </motion.button>
      ))}
    </div>
  );
}

function storageGet(key: string) {
  try {
    return sessionStorage.getItem(key);
  } catch {
    return null;
  }
}
function storageSet(key: string, v: string | null) {
  try {
    if (v === null) sessionStorage.removeItem(key);
    else sessionStorage.setItem(key, v);
  } catch {
    /* storage unavailable: the conversation simply is not restored */
  }
}

export default function Analyst() {
  const { run, round, market, roundInfo, pooled } = useSelection();
  const ai = useAiStatus();
  const suggestions = useSuggestions();
  const key = `vx-chat-${run}`;
  const [turns, setTurns] = useState<Turn[]>([]);
  const [session, setSession] = useState<string | null>(null);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const end = useRef<HTMLDivElement>(null);
  const box = useRef<HTMLTextAreaElement>(null);

  const nMarkets = roundInfo?.markets.length ?? 1;
  const mLabel = market ? marketLabel(market, nMarkets) : "";
  const contextText = `Round ${round ?? "-"}${nMarkets > 1 ? ` · ${mLabel}` : ""}${pooled ? " (metrics pooled)" : ""}`;
  const context: ChatContext = {
    round,
    market: nMarkets > 1 && mLabel.startsWith("Market ") ? mLabel.slice(-1) : undefined,
    scope: pooled ? "pooled" : "market",
    page: "analyst",
  };

  useEffect(() => {
    const saved = storageGet(key);
    if (saved) {
      try {
        const s = JSON.parse(saved) as { turns: Turn[]; session: string | null };
        setTurns(s.turns);
        setSession(s.session);
        return;
      } catch {
        /* ignore a corrupt entry */
      }
    }
    setTurns([]);
    setSession(null);
  }, [key]);

  useEffect(() => {
    storageSet(key, JSON.stringify({ turns, session }));
    end.current?.scrollIntoView({ behavior: "smooth", block: "end" });
  }, [turns, session, key]);

  // auto-grow the input
  useEffect(() => {
    const el = box.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${Math.min(160, el.scrollHeight)}px`;
  }, [input]);

  const send = async (text: string) => {
    const q = text.trim();
    if (!q || busy || !run) return;
    setInput("");
    setTurns((t) => [...t, { role: "user", content: q, context: contextText }]);
    setBusy(true);
    try {
      const r = await askAnalyst(run, q, session, context);
      setSession(r.session_id);
      setTurns((t) => [...t, { role: "assistant", content: r.answer || "(empty answer)", sources: r.sources }]);
    } catch (e) {
      setTurns((t) => [...t, { role: "assistant", content: e instanceof Error ? e.message : String(e), error: true }]);
    } finally {
      setBusy(false);
      box.current?.focus();
    }
  };

  const reset = () => {
    setTurns([]);
    setSession(null);
    storageSet(key, null);
  };

  const available = ai.data?.analyst_assistant.available;
  const lbl = run ? runLabel(run) : null;
  const asked = new Set(turns.filter((t) => t.role === "user").map((t) => t.content));
  const followUps = suggestions.filter((s) => !asked.has(s.prompt)).slice(0, 3);

  return (
    <div className="flex min-h-[calc(100vh-190px)] flex-col">
      <PageHeader
        eyebrow="Analyst"
        title="Ask the run"
        description={`Answers come only from this run's logged detector output, through data lookups you can inspect. Run of ${lbl?.date ?? ""}. Simulation data, not real traders.`}
        right={
          turns.length > 0 && (
            <button onClick={reset} className="focus-ring inline-flex h-8 items-center gap-1.5 rounded-lg border border-ink-600 px-3 text-[12px] text-fg-2 hover:bg-tint/[0.03]">
              <RotateCcw size={13} /> New conversation
            </button>
          )
        }
      />

      {ai.data && !available && (
        <div className="mb-4">
          <Callout kind="warning" title="AI features are unavailable">
            {ai.data.analyst_assistant.reason}. Everything else works; the Surveillance view shows the SHAP evidence for every case.
          </Callout>
        </div>
      )}

      <div className="panel flex flex-1 flex-col overflow-hidden">
        <div className="flex items-center gap-2 border-b border-ink-700 px-5 py-2.5 text-[12px] text-fg-muted">
          <Crosshair size={13} className="text-accent" />
          Asking about <span className="font-medium text-fg-2">{contextText}</span>
          <span className="hidden sm:inline">· change the round or market in the top bar</span>
        </div>
        <div className="flex-1 space-y-5 overflow-y-auto p-5" style={{ maxHeight: "calc(100vh - 360px)", minHeight: 380 }}>
          {!turns.length ? (
            <div className="flex h-full flex-col items-center justify-center py-8 text-center">
              <motion.div initial={{ scale: 0.8, opacity: 0 }} animate={{ scale: 1, opacity: 1 }} className="flex h-12 w-12 items-center justify-center rounded-2xl border border-accent/30 bg-accent/10 text-accent">
                <Sparkles size={20} />
              </motion.div>
              <div className="mt-4 text-[15px] font-semibold text-fg">What do you want to know about {contextText}?</div>
              <div className="mt-1 text-[12.5px] text-fg-muted">Suggestions come from this round's cases, metrics and the strategist's history.</div>
              <div className="mt-5 w-full max-w-4xl">
                <SuggestionChips items={suggestions.slice(0, 6)} onPick={(s) => send(s.prompt)} disabled={!available || busy} />
              </div>
            </div>
          ) : (
            <AnimatePresence initial={false}>
              {turns.map((t, i) => (
                <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className={clsx("flex gap-3", t.role === "user" && "flex-row-reverse")}>
                  <div
                    className={clsx(
                      "flex h-8 w-8 shrink-0 items-center justify-center rounded-lg border",
                      t.role === "user" ? "border-ink-600 bg-ink-800 text-fg-2" : "border-accent/30 bg-accent/10 text-accent",
                    )}
                  >
                    {t.role === "user" ? <User size={15} /> : <Sparkles size={15} />}
                  </div>
                  <div
                    className={clsx(
                      "min-w-0 rounded-2xl px-4 py-3",
                      t.role === "user" ? "max-w-[min(720px,80%)] rounded-tr-md bg-ink-700 text-[13.5px] text-fg" : "max-w-[min(980px,92%)] rounded-tl-md border border-ink-700 bg-ink-850",
                      t.error && "border-critical/30",
                    )}
                  >
                    {t.role === "user" ? (
                      <>
                        <div>{t.content}</div>
                        {t.context && <div className="mt-1 text-[10.5px] uppercase tracking-[0.08em] text-fg-muted">{t.context}</div>}
                      </>
                    ) : (
                      <>
                        <div className={clsx("prose-vx text-[13.5px]", t.error && "text-critical")}>
                          <ReactMarkdown remarkPlugins={[remarkGfm]} components={MD}>
                            {t.content}
                          </ReactMarkdown>
                        </div>
                        {t.sources && <Sources sources={t.sources} />}
                      </>
                    )}
                  </div>
                </motion.div>
              ))}
              {busy && (
                <motion.div key="typing" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="flex gap-3">
                  <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-accent/30 bg-accent/10 text-accent">
                    <Sparkles size={15} />
                  </div>
                  <div className="flex items-center gap-1.5 rounded-2xl rounded-tl-md border border-ink-700 bg-ink-850 px-4 py-3">
                    {[0, 1, 2].map((d) => (
                      <motion.span
                        key={d}
                        className="h-1.5 w-1.5 rounded-full bg-fg-muted"
                        animate={{ opacity: [0.3, 1, 0.3], y: [0, -2, 0] }}
                        transition={{ duration: 1, repeat: Infinity, delay: d * 0.15 }}
                      />
                    ))}
                    <span className="ml-2 text-[12px] text-fg-muted">Reading the run's logs</span>
                  </div>
                </motion.div>
              )}
              {!busy && turns[turns.length - 1]?.role === "assistant" && followUps.length > 0 && (
                <motion.div key={`follow-${turns.length}`} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="pl-11">
                  <div className="mb-2 flex items-center gap-1.5 text-[11px] uppercase tracking-[0.1em] text-fg-muted">
                    <Lightbulb size={12} /> Ask next
                  </div>
                  <SuggestionChips compact items={followUps} onPick={(s) => send(s.prompt)} disabled={!available} />
                </motion.div>
              )}
            </AnimatePresence>
          )}
          <div ref={end} />
        </div>

        <form
          onSubmit={(e) => {
            e.preventDefault();
            send(input);
          }}
          className="border-t border-ink-700 p-3"
        >
          <div className="flex items-end gap-2 rounded-xl border border-ink-600 bg-ink-900 p-2 focus-within:border-accent/40">
            <textarea
              ref={box}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  send(input);
                }
              }}
              rows={1}
              maxLength={2000}
              disabled={!available}
              placeholder={available ? `Ask about ${contextText}: accounts, personas, detector performance, the strategist` : "AI features are unavailable"}
              className="max-h-40 min-h-[36px] flex-1 resize-none bg-transparent px-2 py-1.5 text-[13.5px] text-fg placeholder:text-fg-muted focus:outline-none disabled:opacity-50"
            />
            <button
              type="submit"
              disabled={!input.trim() || busy || !available}
              className="focus-ring flex h-9 w-9 items-center justify-center rounded-lg bg-accent text-accent-ink transition hover:brightness-110 disabled:opacity-30"
              aria-label="Send"
            >
              <ArrowUp size={16} strokeWidth={2.4} />
            </button>
          </div>
          <div className="mt-1.5 px-1 text-[11px] text-fg-muted">Enter to send, Shift + Enter for a new line. Questions use the round and market shown above unless you name others.</div>
        </form>
      </div>
    </div>
  );
}
