import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { ChevronDown, Eye, FileSearch, Search, ShieldAlert, ShieldCheck, ShieldQuestion, Siren } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import {
  Callout,
  Chip,
  DataTable,
  EmptyState,
  ErrorNote,
  PageHeader,
  PersonaMark,
  Skeleton,
  StatGrid,
  StatTile,
  StatusBadge,
} from "@/components/ui";
import { useCases, type CaseRow } from "@/lib/api";
import { fmtInt, tickClock } from "@/lib/format";
import { STATUS_TEXT } from "@/lib/glossary";
import { useSelection } from "@/lib/selection";
import { COMMODITY_META, PERSONA_META } from "@/lib/theme";
import { CaseDetail } from "./surveillance/CaseDetail";

const PAGE = 14;
const ORDER = ["spoofing", "layering", "wash_trading", "pump_and_dump", "anomaly"];

export default function Surveillance() {
  const { run, round, market } = useSelection();
  const q = useCases(run, round, market);
  const [params, setParams] = useSearchParams();
  const data = q.data;

  const personas = useMemo(() => ORDER.filter((p) => data?.cases.some((c) => c.persona === p)), [data]);
  const [personaFilter, setPersonaFilter] = useState<string[] | null>(null);
  const [statusFilter, setStatusFilter] = useState<string[]>(["confirmed", "other_persona", "near_episode", "no_episode"]);
  const [showBrief, setShowBrief] = useState(false);
  const [minScore, setMinScore] = useState(0);
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(0);
  const [showMissed, setShowMissed] = useState(false);

  const activePersonas = personaFilter ?? personas.filter((p) => p !== "anomaly");
  useEffect(() => {
    setPersonaFilter(null);
    setPage(0);
  }, [run, round, market]);

  const view = useMemo(() => {
    const s = search.trim().toLowerCase();
    return (data?.cases ?? []).filter(
      (c) =>
        activePersonas.includes(c.persona) &&
        statusFilter.includes(c.status) &&
        (showBrief || c.tier !== "brief") &&
        c.score >= minScore &&
        (!s || c.trader_id.toLowerCase().includes(s) || c.commodity.includes(s)),
    );
  }, [data, activePersonas, statusFilter, showBrief, minScore, search]);

  const pages = Math.max(1, Math.ceil(view.length / PAGE));
  const pageRows = view.slice(page * PAGE, page * PAGE + PAGE);
  const paramCase = params.get("case");
  const selectedId = (paramCase && data?.cases.some((c) => c.id === paramCase) ? paramCase : pageRows[0]?.id) ?? null;
  const detailRef = useRef<HTMLDivElement>(null);
  const select = (id: string) => {
    setParams((p) => {
      const n = new URLSearchParams(p);
      n.set("case", id);
      return n;
    });
    // On one-column layouts the detail sits below the whole list: bring it into view.
    if (window.innerWidth < 1280) window.setTimeout(() => detailRef.current?.scrollIntoView({ behavior: "smooth", block: "start" }), 60);
  };

  useEffect(() => {
    if (page >= pages) setPage(0);
  }, [page, pages]);

  const k = data?.kpi;
  const toggle = (list: string[], v: string) => (list.includes(v) ? list.filter((x) => x !== v) : [...list, v]);

  return (
    <div>
      <PageHeader
        eyebrow="Surveillance"
        title="Detector cases"
        description="What the retrained detector flagged in this round's test market. Consecutive flagged windows of one account are merged into a case, then checked against the simulator's ground truth. Detector belief, ground truth and the AI note are three different levels of certainty and are kept apart."
      />

      {q.error ? (
        <ErrorNote error={q.error} />
      ) : !data ? (
        <div className="space-y-4">
          <div className="grid grid-cols-2 gap-3 lg:grid-cols-5">
            {Array.from({ length: 5 }).map((_, i) => (
              <Skeleton key={i} className="h-24" />
            ))}
          </div>
          <Skeleton className="h-[520px]" />
        </div>
      ) : !data.has_flags ? (
        <EmptyState title="No detector output for this round" icon={<FileSearch size={22} />}>
          flags.parquet is missing: the round did not finish, or the run predates trader-level flags.
        </EmptyState>
      ) : (
        <>
          <StatGrid className="grid-cols-2 lg:grid-cols-5">
            <StatTile label="Persona cases" value={k!.persona_cases} format={fmtInt} icon={<Siren size={15} />} sub={`${k!.min_case_windows}+ consecutive flagged windows`} />
            <StatTile label="Confirmed" value={k!.confirmed} format={fmtInt} tone="good" icon={<ShieldCheck size={15} />} sub="Account, persona and span match" />
            <StatTile label="No matching episode" value={k!.no_episode} format={fmtInt} tone={k!.no_episode ? "warn" : "good"} icon={<ShieldAlert size={15} />} sub="Each costs an investigation" />
            <StatTile label="Missed episodes" value={k!.missed} format={fmtInt} tone={k!.missed ? "critical" : "good"} icon={<ShieldQuestion size={15} />} sub="Never flagged with the right persona" />
            <StatTile label="Anomaly-only" value={k!.anomalies} format={fmtInt} icon={<Eye size={15} />} sub={`${k!.anomalies_on_episode} overlap a real episode`} />
          </StatGrid>

          <div className="mt-4 grid gap-3 lg:grid-cols-2">
            {k!.review_windows > 0 && (
              <Callout title={`${fmtInt(k!.review_windows)} review-tier windows`}>
                The IsolationForest found them unusual, but neither a classifier nor a strong deviation from the account's own history backed it up, so they
                are not alerts. These are typically legitimate HFT firms and execution algorithms.
              </Callout>
            )}
            {k!.brief > 0 && (
              <Callout title={`${fmtInt(k!.brief)} brief alerts not counted as cases`}>
                Flags lasting fewer than {k!.min_case_windows} consecutive windows. Real schemes persist, so a one-window blip is not worth an
                investigation: {fmtInt(k!.brief_no_episode)} of these had no ground-truth episode and {fmtInt(k!.brief_confirmed)} were confirmed
                {k!.missed_brief > 0 ? `; ${k!.missed_brief} missed episodes were caught only by a brief alert` : ""}. Use the Brief alerts filter to list them.
              </Callout>
            )}
            {k!.sibling_label > 0 && (
              <Callout kind="warning" title={`${k!.sibling_label} cases carry the sibling label`}>
                Labelled spoofing while the episode was layering, or the reverse. Both are fake-depth schemes (layering is a form of spoofing under CEA
                4c(a)(5)(C)), and split spoof orders on consecutive levels look like a ladder, so read the label as the fake-depth family.
              </Callout>
            )}
          </div>

          {data.missed.length > 0 && (
            <div className="panel mt-4">
              <button onClick={() => setShowMissed((s) => !s)} className="focus-ring flex w-full items-center justify-between px-5 py-3.5 text-left">
                <span className="text-[13.5px] font-semibold text-fg">Missed ground-truth episodes ({data.missed.length})</span>
                <ChevronDown size={16} className={clsx("text-fg-muted transition-transform", showMissed && "rotate-180")} />
              </button>
              <AnimatePresence initial={false}>
                {showMissed && (
                  <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
                    <div className="px-5 pb-5">
                      <DataTable
                        dense
                        spec={{
                          columns: [
                            { key: "event_id", label: "Episode" },
                            { key: "persona", label: "Persona", format: (v) => PERSONA_META[String(v)]?.label ?? String(v) },
                            { key: "commodity", label: "Commodity", format: (v) => COMMODITY_META[String(v)]?.label ?? String(v) },
                            { key: "start_tick", label: "Start", format: (v) => tickClock(v as number) },
                            { key: "end_tick", label: "End", format: (v) => tickClock(v as number) },
                            { key: "trader_ids", label: "Accounts", format: (v) => (v as string[]).join(", ") },
                          ],
                          rows: data.missed as unknown as Record<string, unknown>[],
                        }}
                      />
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>
            </div>
          )}

          <div className="panel mt-4 flex flex-wrap items-center gap-x-6 gap-y-3 px-4 py-3">
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="eyebrow mr-1">Persona</span>
              {personas.map((p) => (
                <Chip key={p} active={activePersonas.includes(p)} onClick={() => { setPersonaFilter(toggle(activePersonas, p)); setPage(0); }}>
                  <PersonaMark persona={p} /> {PERSONA_META[p].label}
                </Chip>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="eyebrow mr-1">Ground truth</span>
              {Object.entries(STATUS_TEXT).map(([s, t]) => (
                <Chip key={s} active={statusFilter.includes(s)} onClick={() => { setStatusFilter(toggle(statusFilter, s)); setPage(0); }}>
                  {t.label.replace("Ground truth confirms", "Confirmed")}
                </Chip>
              ))}
            </div>
            <div className="flex flex-wrap items-center gap-1.5">
              <span className="eyebrow mr-1">Length</span>
              <Chip active={showBrief} onClick={() => { setShowBrief(!showBrief); setPage(0); }}>
                Brief alerts ({fmtInt(k!.brief)})
              </Chip>
            </div>
            <label className="flex items-center gap-2">
              <span className="eyebrow">Min score</span>
              <input
                type="range"
                min={0}
                max={1}
                step={0.05}
                value={minScore}
                onChange={(e) => { setMinScore(+e.target.value); setPage(0); }}
                className="h-1 w-28 cursor-pointer accent-[var(--accent)]"
              />
              <span className="num w-9 font-mono text-[12px] text-fg-2">{minScore.toFixed(2)}</span>
            </label>
            <label className="ml-auto flex h-8 items-center gap-2 rounded-lg border border-ink-600 bg-ink-900 px-2.5">
              <Search size={13} className="text-fg-muted" />
              <input
                value={search}
                onChange={(e) => { setSearch(e.target.value); setPage(0); }}
                placeholder="Account or commodity"
                className="w-40 bg-transparent text-[12.5px] text-fg placeholder:text-fg-muted focus:outline-none"
              />
            </label>
          </div>

          {!data.cases.length ? (
            <div className="mt-4">
              <EmptyState title="No flags in this test market">The detector raised no alerts here. That is a valid result, not an error.</EmptyState>
            </div>
          ) : (
            <div className="mt-4 grid items-start gap-4 xl:grid-cols-[400px_1fr]">
              <div className="panel overflow-hidden">
                <div className="flex items-center justify-between border-b border-ink-700 px-4 py-2.5">
                  <span className="text-[12.5px] text-fg-2">
                    <span className="font-semibold text-fg">{view.length}</span> cases
                  </span>
                  <span className="font-mono text-[11px] text-fg-muted">
                    page {page + 1} / {pages}
                  </span>
                </div>
                {view.length === 0 ? (
                  <div className="px-4 py-10 text-center text-[12.5px] text-fg-muted">No cases match the filters.</div>
                ) : (
                  <motion.ul layout className="max-h-[1060px] divide-y divide-ink-700/70 overflow-auto">
                    <AnimatePresence initial={false} mode="popLayout">
                      {pageRows.map((c, i) => (
                        <CaseItem key={c.id} c={c} active={c.id === selectedId} onClick={() => select(c.id)} index={i} />
                      ))}
                    </AnimatePresence>
                  </motion.ul>
                )}
                {pages > 1 && (
                  <div className="flex items-center justify-between border-t border-ink-700 px-3 py-2">
                    <button disabled={page === 0} onClick={() => setPage((p) => p - 1)} className="focus-ring rounded-md px-2.5 py-1 text-[12px] text-fg-2 hover:bg-tint/5 disabled:opacity-30">
                      Previous
                    </button>
                    <div className="flex gap-1">
                      {Array.from({ length: Math.min(pages, 7) }).map((_, i) => {
                        const n = pages <= 7 ? i : Math.min(Math.max(0, page - 3), pages - 7) + i;
                        return (
                          <button key={n} onClick={() => setPage(n)} className={clsx("h-6 min-w-6 rounded font-mono text-[11px]", n === page ? "bg-ink-700 text-fg" : "text-fg-muted hover:text-fg")}>
                            {n + 1}
                          </button>
                        );
                      })}
                    </div>
                    <button disabled={page >= pages - 1} onClick={() => setPage((p) => p + 1)} className="focus-ring rounded-md px-2.5 py-1 text-[12px] text-fg-2 hover:bg-tint/5 disabled:opacity-30">
                      Next
                    </button>
                  </div>
                )}
              </div>
              <div ref={detailRef} className="scroll-mt-[120px] xl:sticky xl:top-[72px]">
                {selectedId ? <CaseDetail id={selectedId} /> : <EmptyState title="Select a case" />}
              </div>
            </div>
          )}
          <p className="mt-4 text-[11.5px] text-fg-muted">
            Detector flags are model output on synthetic data, not findings about real traders. Figures cover the test market selected in the top bar.
          </p>
        </>
      )}
    </div>
  );
}

function CaseItem({ c, active, onClick, index }: { c: CaseRow; active: boolean; onClick: () => void; index: number }) {
  const st = STATUS_TEXT[c.status];
  return (
    <motion.li
      layout
      initial={{ opacity: 0, x: -8 }}
      animate={{ opacity: 1, x: 0, transition: { delay: index * 0.02 } }}
      exit={{ opacity: 0 }}
    >
      <button
        onClick={onClick}
        className={clsx("focus-ring relative block w-full px-4 py-3 text-left transition-colors", active ? "bg-tint/[0.04]" : "hover:bg-tint/[0.02]")}
      >
        {active && <motion.span layoutId="case-active" className="absolute inset-y-0 left-0 w-[2px] bg-accent" />}
        <div className="flex items-center justify-between gap-2">
          <span className="flex min-w-0 items-center gap-2">
            <PersonaMark persona={c.persona} size={10} />
            <span className="truncate font-mono text-[12.5px] font-medium text-fg">{c.trader_id}</span>
          </span>
          <span className="num font-mono text-[12.5px] text-fg">{c.score.toFixed(2)}</span>
        </div>
        <div className="mt-1.5 flex items-center justify-between gap-2">
          <span className="truncate text-[11.5px] text-fg-muted">
            {PERSONA_META[c.persona]?.label} · {COMMODITY_META[c.commodity]?.label ?? c.commodity} · {tickClock(c.start_tick)}
          </span>
          <StatusBadge kind={st.kind}>
            {c.status === "confirmed" ? "Confirmed" : c.status === "other_persona" ? "Other persona" : c.status === "near_episode" ? "Near episode" : "No episode"}
            {c.tier === "brief" ? " · brief" : ""}
          </StatusBadge>
        </div>
        <div className="mt-2 h-[3px] w-full rounded-full bg-tint/[0.05]">
          <div className="h-full rounded-full" style={{ width: `${Math.max(4, c.score * 100)}%`, background: PERSONA_META[c.persona]?.color }} />
        </div>
      </button>
    </motion.li>
  );
}
