import clsx from "clsx";
import { AnimatePresence, motion } from "framer-motion";
import { Check, ChevronsUpDown, Search } from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { fmtNum, runLabel } from "@/lib/format";
import { useSelection } from "@/lib/selection";
import { Kbd } from "../ui";

function Spark({ values, width = 64, height = 18 }: { values: (number | null)[]; width?: number; height?: number }) {
  const v = values.filter((x): x is number => typeof x === "number");
  if (v.length < 2) return <span className="inline-block" style={{ width }} />;
  const lo = Math.min(...v) - 0.01;
  const hi = Math.max(...v) + 0.01;
  const pts = v.map((x, i) => `${(i / (v.length - 1)) * width},${height - ((x - lo) / (hi - lo)) * height}`).join(" ");
  return (
    <svg width={width} height={height} className="overflow-visible" aria-hidden>
      <polyline points={pts} fill="none" stroke="var(--accent)" strokeWidth={1.5} strokeLinejoin="round" strokeLinecap="round" opacity={0.85} />
    </svg>
  );
}

export function RunPicker() {
  const { runs, run, setRun, runInfo } = useSelection();
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [cursor, setCursor] = useState(0);
  const box = useRef<HTMLDivElement>(null);
  const input = useRef<HTMLInputElement>(null);

  const list = useMemo(() => {
    const q = query.trim().toLowerCase();
    return runs.filter((r) => !q || r.run.toLowerCase().includes(q) || runLabel(r.run).date.toLowerCase().includes(q));
  }, [runs, query]);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.key === "k" && (e.metaKey || e.ctrlKey)) || (e.key === "/" && !(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLTextAreaElement))) {
        e.preventDefault();
        setOpen(true);
      }
      if (e.key === "Escape") setOpen(false);
    };
    const onClick = (e: MouseEvent) => {
      if (box.current && !box.current.contains(e.target as Node)) setOpen(false);
    };
    window.addEventListener("keydown", onKey);
    window.addEventListener("mousedown", onClick);
    return () => {
      window.removeEventListener("keydown", onKey);
      window.removeEventListener("mousedown", onClick);
    };
  }, []);

  useEffect(() => {
    if (open) {
      setQuery("");
      setCursor(Math.max(0, runs.findIndex((r) => r.run === run)));
      setTimeout(() => input.current?.focus(), 30);
    }
  }, [open, runs, run]);

  const lbl = run ? runLabel(run) : null;
  const pick = (r: string) => {
    setRun(r);
    setOpen(false);
  };

  return (
    <div className="relative" ref={box}>
      <button
        onClick={() => setOpen((o) => !o)}
        className="focus-ring group flex h-9 items-center gap-3 rounded-lg border border-ink-600 bg-ink-850 px-3 text-left transition hover:border-ink-600 hover:bg-ink-800"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <span className="relative flex h-2 w-2">
          <span className="absolute inline-flex h-full w-full animate-pulseDot rounded-full bg-accent/60" />
          <span className="relative inline-flex h-2 w-2 rounded-full bg-accent" />
        </span>
        <span className="min-w-0">
          <span className="block truncate text-[12.5px] font-medium text-fg">{lbl ? `${lbl.date} · ${lbl.time}` : "Select a run"}</span>
        </span>
        <span className="hidden font-mono text-[11px] text-fg-muted md:inline">
          {runInfo ? `${runInfo.rounds.length} rounds · seed ${lbl?.seed}` : ""}
        </span>
        <ChevronsUpDown size={14} className="text-fg-muted" />
      </button>

      <AnimatePresence>
        {open && (
          <motion.div
            initial={{ opacity: 0, y: -6, scale: 0.98 }}
            animate={{ opacity: 1, y: 0, scale: 1 }}
            exit={{ opacity: 0, y: -4, scale: 0.98 }}
            transition={{ duration: 0.16 }}
            className="absolute left-0 top-11 z-50 w-[min(560px,calc(100vw-24px))] overflow-hidden rounded-xl border border-ink-600 bg-ink-850/95 shadow-2xl backdrop-blur-xl"
          >
            <div className="flex items-center gap-2 border-b border-ink-700 px-3 py-2.5">
              <Search size={15} className="text-fg-muted" />
              <input
                ref={input}
                value={query}
                onChange={(e) => {
                  setQuery(e.target.value);
                  setCursor(0);
                }}
                onKeyDown={(e) => {
                  if (e.key === "ArrowDown") {
                    e.preventDefault();
                    setCursor((c) => Math.min(list.length - 1, c + 1));
                  } else if (e.key === "ArrowUp") {
                    e.preventDefault();
                    setCursor((c) => Math.max(0, c - 1));
                  } else if (e.key === "Enter" && list[cursor]) {
                    pick(list[cursor].run);
                  }
                }}
                placeholder="Search adversarial runs"
                className="w-full bg-transparent text-[13px] text-fg placeholder:text-fg-muted focus:outline-none"
              />
              <Kbd>Esc</Kbd>
            </div>
            <ul className="max-h-[420px] overflow-auto p-1.5" role="listbox">
              {list.map((r, i) => {
                const l = runLabel(r.run);
                const selected = r.run === run;
                return (
                  <li key={r.run}>
                    <button
                      onMouseEnter={() => setCursor(i)}
                      onClick={() => pick(r.run)}
                      className={clsx(
                        "flex w-full items-center gap-3 rounded-lg px-2.5 py-2 text-left transition-colors",
                        i === cursor ? "bg-tint/[0.05]" : "",
                      )}
                      role="option"
                      aria-selected={selected}
                    >
                      <span className="w-4">{selected && <Check size={14} className="text-accent" />}</span>
                      <span className="min-w-0 flex-1">
                        <span className="block text-[12.5px] font-medium text-fg">
                          {l.date} <span className="text-fg-muted">· {l.time}</span>
                          {l.tag && <span className="ml-2 rounded border border-ink-600 px-1.5 py-px text-[10.5px] text-fg-2">{l.tag}</span>}
                        </span>
                        <span className="block truncate font-mono text-[10.5px] text-fg-muted">{r.run}</span>
                      </span>
                      <span className="hidden text-right sm:block">
                        <span className="block font-mono text-[11px] text-fg-2">{r.rounds.length} rounds</span>
                        <span className="block font-mono text-[10.5px] text-fg-muted">F1 {fmtNum(r.avg_f1, 3)}</span>
                      </span>
                      <Spark values={r.f1_trend ?? []} />
                    </button>
                  </li>
                );
              })}
              {!list.length && <li className="px-3 py-6 text-center text-[12.5px] text-fg-muted">No runs match.</li>}
            </ul>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
