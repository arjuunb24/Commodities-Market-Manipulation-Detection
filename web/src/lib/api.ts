import { useMutation, useQuery } from "@tanstack/react-query";

// ── Types (mirror api/*.py payloads) ─────────────────────────────────────────

export interface RunSummary {
  run: string;
  rounds: number[];
  latest_round?: number;
  avg_f1?: number | null;
  f1_trend?: (number | null)[];
  n_test_markets?: number;
  llm_calls?: number;
  llm_cost_usd?: number;
  mutation_sources?: Record<string, number>;
}

export interface RoundInfo {
  round: number;
  markets: { name: string; tick_range: [number, number] | null }[];
}

export interface PersonaMetrics {
  precision?: number | null;
  recall?: number | null;
  f1?: number | null;
  fpr?: number | null;
  support?: number | null;
  events_total?: number | null;
  events_detected?: number | null;
  detection_latency_ticks?: number | null;
  episode_coverage?: number | null;
  caught_coverage?: number | null;
  caught_events_detected?: number | null;
  caught_latency_ticks?: number | null;
  caught_weak_share?: number | null;
  family_coverage?: number | null;
  /** spoofing + layering scored as one fake-depth class (label swaps between them are not errors) */
  family_precision?: number | null;
  family_recall?: number | null;
  family_f1?: number | null;
}

export interface RoundSummary {
  personas?: Record<string, PersonaMetrics>;
  pre_retrain?: Record<string, PersonaMetrics>;
  unsupervised?: { precision?: number; recall?: number; f1?: number; fpr?: number; support?: number };
  thresholds?: Record<string, number>;
  n_flags?: number;
  n_test_markets?: number;
  params_used?: Record<string, Record<string, unknown>>;
  mutation_source?: Record<string, string>;
  effectiveness?: Record<string, Record<string, unknown>>;
  effectiveness_ratio?: Record<string, number | null>;
  mutation_score?: Record<string, number | null>;
  scope?: string;
  source?: "saved" | "derived";
  frozen_by_market?: boolean;
}

export interface MarketEvent {
  event_id: string;
  persona: string;
  commodity: string;
  start_tick: number;
  end_tick: number;
  trader_ids: string[];
}

export interface Kpi {
  trades: number;
  volume: number;
  traders: number;
  events: number;
}

export interface PersonaF1 {
  persona: string;
  f1?: number | null;
  precision?: number | null;
  recall?: number | null;
  fpr?: number | null;
}

export interface Overview {
  commodities: string[];
  kpi: Kpi;
  persona_f1?: PersonaF1[];
  snapshot: { commodity: string; first: number; last: number; pct: number; volume: number; trades: number; events: number; spark: number[] }[];
  indexed?: { ticks: number[]; series: Record<string, number[]> };
  prices?: { ticks: number[]; series: Record<string, number[]> };
  correlation?: { labels: string[]; matrix: number[][] };
  volume_by_commodity?: { commodity: string; volume: number; trades: number }[];
  events_by_commodity?: ({ commodity: string } & Record<string, number | string>)[];
  events?: MarketEvent[];
}

export interface CommodityView {
  commodity: string;
  empty?: boolean;
  kpi: Kpi & { last: number; first: number; pct: number; high: number; low: number };
  persona_f1?: PersonaF1[];
  events: MarketEvent[];
  candles: { tick: number; open: number; high: number; low: number; close: number; volume: number; vwap: number }[];
  flow: { tick: number; buy: number; sell: number; trades: number; imbalance: number; cancel_rate: number; otr: number; placed: number; cancelled: number }[];
  profile: { price: number; volume: number }[];
  returns: { centers: number[]; density: number[]; normal: number[]; mean: number; std: number; excess_kurtosis: number; n: number } | null;
  heatmap: { traders: string[]; buckets: number[]; values: number[][] };
  bucket_ticks: { bar: number; flow: number; heat: number };
}

export type CaseStatus = "confirmed" | "other_persona" | "near_episode" | "no_episode";

export interface CaseRow {
  id: string;
  commodity: string;
  trader_id: string;
  persona: string;
  start_tick: number;
  end_tick: number;
  score: number;
  n_windows: number;
  triggered_by: string[];
  status: CaseStatus;
  gt_event: string | null;
  gt_persona: string | null;
  /** "brief" = fewer consecutive flagged windows than kpi.min_case_windows; not counted in the KPIs. */
  tier: "case" | "brief";
  near_event: string | null;
}

export interface CasesResponse {
  has_flags: boolean;
  kpi: {
    persona_cases: number;
    confirmed: number;
    no_episode: number;
    other_persona: number;
    near_episode: number;
    missed: number;
    missed_brief: number;
    brief: number;
    brief_confirmed: number;
    brief_no_episode: number;
    min_case_windows: number;
    anomalies: number;
    anomalies_on_episode: number;
    sibling_label: number;
    review_windows: number;
    windows: number;
    alert_windows: number;
  };
  cases: CaseRow[];
  missed: MarketEvent[];
  thresholds: Record<string, number>;
}

export interface CaseDetail extends CaseRow {
  iso_score: number;
  persona_scores: Record<string, number>;
  thresholds: Record<string, number>;
  shap: { feature: string; value: number }[];
  evidence: Record<string, unknown>;
  price: { ticks: number[]; prices: number[] };
  buys: { tick: number; price: number; qty: number }[];
  sells: { tick: number; price: number; qty: number }[];
}

export interface CaseRaw {
  orders_total: number;
  trades_total: number;
  orders: Record<string, unknown>[];
  trades: Record<string, unknown>[];
}

export interface PerfRow extends PersonaMetrics {
  round: number;
  persona: string;
  phase: "pre" | "post";
}

export interface Performance {
  rows: PerfRow[];
  headline: string;
  rounds: number[];
  scope?: string;
  frozen_by_market?: boolean;
  thresholds: Record<string, Record<string, number> | null>;
  unsupervised: Record<string, { precision?: number; recall?: number; f1?: number; fpr?: number } | null>;
  effectiveness: { round: number; persona: string; ratio: number | null; score: number | null; primary_metric: string | null; primary: number | null }[];
}

export interface ParamsHistory {
  rounds: number[];
  rows: { persona: string; parameter: string; values: Record<string, unknown>; changed: string[] }[];
}

export interface KnobChange {
  knob: string;
  old: unknown;
  new: unknown;
}

export interface StrategistPersona {
  source: string;
  changes: KnobChange[];
  reasoning: { points: string[]; notes: string[] };
  frozen_coverage?: number | null;
  own_label_coverage?: number | null;
  family_coverage?: number | null;
  weak_share?: number | null;
  episodes_caught?: number | null;
  episodes_total?: number | null;
  frozen_delay?: number | null;
  frozen_recall?: number | null;
  retrained_recall?: number | null;
  effect_ratio?: number | null;
  score?: number | null;
  recovered_from_round?: number | null;
  /** client-side: frozen figures taken from the pooled scope (run predates per-market recording) */
  frozen_from_pooled?: boolean;
  guard_reverts: KnobChange[];
  probe: {
    chosen: number | null;
    candidates: { changes: Record<string, unknown>; coverage: number | null; effect_ratio: number | null; score: number | null; origin?: string | null; source?: string | null }[];
  } | null;
}

export interface StrategistRound {
  round: number;
  personas: Record<string, StrategistPersona>;
}

export interface AiStatus {
  narrative_generator: { available: boolean; reason: string | null };
  analyst_assistant: { available: boolean; reason: string | null };
}

export interface ChatResponse {
  session_id: string;
  answer: string;
  sources: { tool: string; args: unknown; error?: boolean }[];
}

// ── Fetching ─────────────────────────────────────────────────────────────────

export class ApiError extends Error {
  constructor(public status: number, message: string) {
    super(message);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, { headers: { "Content-Type": "application/json" }, ...init });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = await res.json();
      detail = typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail ?? body);
    } catch {
      /* body was not JSON */
    }
    throw new ApiError(res.status, detail);
  }
  return res.json() as Promise<T>;
}

const enc = encodeURIComponent;
const q = (market?: string | null) => (market ? `?market=${enc(market)}` : "");

export const useRuns = () => useQuery({ queryKey: ["runs"], queryFn: () => request<RunSummary[]>("/api/runs"), staleTime: 60_000 });

export const useRounds = (run?: string) =>
  useQuery({ queryKey: ["rounds", run], enabled: !!run, queryFn: () => request<RoundInfo[]>(`/api/runs/${enc(run!)}/rounds`) });

export const useRoundSummary = (run?: string, round?: number, market?: string | null) =>
  useQuery({
    queryKey: ["round-summary", run, round, market ?? "pooled"],
    enabled: !!run && round !== undefined,
    queryFn: () => request<RoundSummary>(`/api/runs/${enc(run!)}/rounds/${round}/summary${q(market)}`),
    placeholderData: (prev) => prev,
  });

export const useOverview = (run?: string, round?: number, market?: string | null) =>
  useQuery({
    queryKey: ["overview", run, round, market],
    enabled: !!run && round !== undefined,
    queryFn: () => request<Overview>(`/api/runs/${enc(run!)}/rounds/${round}/market/overview${q(market)}`),
    placeholderData: (prev) => prev,
  });

export const useCommodity = (run?: string, round?: number, market?: string | null, commodity?: string) =>
  useQuery({
    queryKey: ["commodity", run, round, market, commodity],
    enabled: !!run && round !== undefined && !!commodity,
    queryFn: () => request<CommodityView>(`/api/runs/${enc(run!)}/rounds/${round}/market/commodity/${enc(commodity!)}${q(market)}`),
    placeholderData: (prev) => prev,
  });

export const useCases = (run?: string, round?: number, market?: string | null) =>
  useQuery({
    queryKey: ["cases", run, round, market],
    enabled: !!run && round !== undefined,
    queryFn: () => request<CasesResponse>(`/api/runs/${enc(run!)}/rounds/${round}/cases${q(market)}`),
    placeholderData: (prev) => prev,
  });

export const useCaseDetail = (run?: string, round?: number, market?: string | null, id?: string | null) =>
  useQuery({
    queryKey: ["case", run, round, market, id],
    enabled: !!run && round !== undefined && !!id,
    queryFn: () => request<CaseDetail>(`/api/runs/${enc(run!)}/rounds/${round}/cases/${enc(id!)}${q(market)}`),
  });

export const useCaseRaw = (run?: string, round?: number, market?: string | null, id?: string | null, enabled = false) =>
  useQuery({
    queryKey: ["case-raw", run, round, market, id],
    enabled: enabled && !!run && round !== undefined && !!id,
    queryFn: () => request<CaseRaw>(`/api/runs/${enc(run!)}/rounds/${round}/cases/${enc(id!)}/raw${q(market)}`),
  });

export const useNarrative = (run?: string, round?: number, market?: string | null, id?: string | null) =>
  useMutation({
    mutationFn: () =>
      request<{ narrative_text?: string; grounding_check_passed?: boolean; error?: string }>(
        `/api/runs/${enc(run!)}/rounds/${round}/cases/${enc(id!)}/narrative${q(market)}`,
        { method: "POST" },
      ),
  });

export const usePerformance = (run?: string, market?: string | null) =>
  useQuery({
    queryKey: ["performance", run, market ?? "pooled"],
    enabled: !!run,
    queryFn: () => request<Performance>(`/api/runs/${enc(run!)}/performance${q(market)}`),
    placeholderData: (prev) => prev,
  });

export const useParams = (run?: string) =>
  useQuery({ queryKey: ["params", run], enabled: !!run, queryFn: () => request<ParamsHistory>(`/api/runs/${enc(run!)}/params`) });

export const useStrategist = (run?: string, market?: string | null) =>
  useQuery({
    queryKey: ["strategist", run, market ?? "pooled"],
    enabled: !!run,
    queryFn: () => request<StrategistRound[]>(`/api/runs/${enc(run!)}/strategist${q(market)}`),
    placeholderData: (prev) => prev,
  });

export const useAiStatus = () =>
  useQuery({ queryKey: ["ai-status"], queryFn: () => request<AiStatus>("/api/ai/status"), staleTime: 120_000 });

export interface ChatContext {
  round?: number;
  market?: string;
  scope?: "market" | "pooled";
  page?: string;
}

export const askAnalyst = (run: string, question: string, session_id?: string | null, context?: ChatContext) =>
  request<ChatResponse>(`/api/runs/${enc(run)}/chat`, { method: "POST", body: JSON.stringify({ question, session_id, context }) });
