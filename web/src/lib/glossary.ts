/** Plain-language meaning of every detector feature (ported from dashboard/components/tab_flags.py). */
export const FEATURES: Record<string, string> = {
  n_orders: "Number of orders the account placed in the window",
  n_trades: "Number of trades the account took part in",
  exec_vol: "Volume the account traded (bought + sold)",
  gross_volume_share: "Account's share of all market volume in the window",
  activity_vs_history: "Activity now vs. the account's own usual level (log ratio; high = sudden burst)",
  avg_order_size_rel: "Average order size vs. the market's median order size",
  max_order_size_rel: "Largest order vs. the market's median order size",
  order_size_cv: "How much the account's order sizes vary (coefficient of variation)",
  large_order_share: "Share of placed volume in large orders (4x the market median or more)",
  layered_order_share: "Share of placed volume in same-tick, same-side batches on 3+ price levels (layering)",
  cancel_replace_ratio: "Share of cancels immediately replaced by a same-side order in the same tick (repricing)",
  cancel_ratio: "Cancelled volume / placed volume",
  fast_cancel_ratio: "Share of order volume cancelled within 10 s of placement",
  large_cancel_ratio: "Share of the account's large orders that were cancelled",
  n_large_cancels: "Number of large orders cancelled in the window",
  mean_cancel_lifetime: "Average time a cancelled order rested before cancellation (ticks)",
  passive_distance_bp: "How far behind the market price the account's orders were placed (bp)",
  fill_ratio: "Traded volume / placed volume (low = orders mostly not meant to trade)",
  cancel_side_imbalance: "Whether cancellations were concentrated on the buy (+) or sell (-) side",
  cancel_exec_opposition: "Cancelling on one side while trading on the other (spoofing pattern)",
  spoof_signature: "Share of large cancelled orders during whose life the account was filled on the opposite side",
  exec_side_imbalance: "Net buying (+) or selling (-) as a share of the account's traded volume",
  aggressor_ratio: "Share of the account's volume where it crossed the spread (took liquidity)",
  prev_net_ratio: "Net buying (+) or selling (-) in the previous ~6 minutes",
  long_net_ratio: "Position built over ~12 minutes as a share of volume",
  reversal_score: "Net buyer before, net seller now (or vice versa): a position flip",
  signed_flow_return: "Own net direction x market return (high = trading with the price move it pushes)",
  coordinated_peers: "Other accounts bursting in the same direction in this window",
  net_flatness: "1 - |net position / gross volume| (1 = lots of volume, no net position change)",
  top_cpty_share: "Share of the account's volume done with a single counterparty",
  reciprocal_ratio: "Volume bought from and sold back to the same counterparties",
  n_counterparties: "Number of different accounts it traded with",
  trade_size_cv: "How much its trade sizes vary (low = scripted, uniform sizes)",
  price_dev_bp: "Average distance of its trade prices from the market VWAP (bp)",
  mkt_return_bp: "Market price change over the window (bp)",
  mkt_volatility_bp: "Market volatility in the window (bp)",
  mkt_range_bp: "High-low price range of the market in the window (bp)",
  mkt_volume_spike: "Market volume vs. its recent average",
  mkt_trade_imbalance: "Market-wide aggressive buying minus selling, as a share of volume",
  mkt_order_imbalance: "Market-wide placed buy volume minus sell volume, as a share",
  mkt_cancel_ratio: "Market-wide cancelled / placed orders",
  mkt_abnormal_return_z: "How unusual the market's price move was vs. recent history (z-score)",
  mkt_anomaly_score: "How unusual the whole market looked in this window (percentile)",
};
for (const b of ["cancel_ratio", "fast_cancel_ratio", "avg_order_size_rel", "max_order_size_rel", "gross_volume_share", "reciprocal_ratio", "aggressor_ratio", "large_order_share"]) {
  FEATURES[`peer_z_${b}`] = `${FEATURES[b]}, compared with all other accounts in the same window (robust z-score)`;
}
for (const b of ["avg_order_size_rel", "max_order_size_rel", "large_order_share", "cancel_ratio", "fast_cancel_ratio", "n_orders", "gross_volume_share", "layered_order_share"]) {
  FEATURES[`self_z_${b}`] = `${FEATURES[b] ?? b}, compared with this account's own earlier windows (robust z-score)`;
}

export const featureLabel = (k: string) =>
  k
    .replace(/^peer_z_/, "peer z · ")
    .replace(/^self_z_/, "self z · ")
    .replace(/^mkt_/, "market · ")
    .replace(/_/g, " ");

export const EVIDENCE: Record<string, string> = {
  span_length_ticks: "Flagged span (ticks)",
  orders_placed: "Orders placed in the span",
  large_order_size_vs_market_median_x: "Large order size vs market median (x)",
  median_large_order_lifetime_ticks: "Median life of large orders before cancel (ticks)",
  large_orders_with_opposite_side_fill_while_resting: "Large orders during which the account was filled on the other side",
  avg_price_move_toward_filled_side_bp: "Average price move toward the filled side while the large order rested (bp)",
  account_cancelled_volume_pct: "Share of placed volume cancelled (%)",
  net_position_vs_gross_pct: "Net position as % of gross volume",
  volume_with_top_counterparty_pct: "Volume with top counterparty (%)",
  pair_share_of_market_volume_pct: "Pair's share of all market volume (%)",
  median_price_distance_from_vwap_bp: "Median trade price distance from VWAP (bp)",
  aggressive_share_pct: "Volume taken aggressively (%)",
  burst_share_of_market_volume_pct: "Burst share of market volume (%)",
  other_accounts_flagged_for_pump_and_dump_in_same_span: "Other accounts flagged for pump-and-dump in the same span",
  large_orders_cancelled_and_replaced_same_tick: "Large orders cancelled and re-placed in the same tick (repricing)",
  aggressive_opposite_trades_right_after_cancel: "Aggressive trades the other way right after a cancel (flips)",
  multi_level_batches: "Batches of orders stacked on 3+ price levels",
  layer_size_vs_market_median_x: "Layer size vs market median (x)",
  batches_with_opposite_side_fill_while_resting: "Batches during which the account was filled on the other side",
};

export const evidenceLabel = (k: string) => EVIDENCE[k] ?? k.replace(/_/g, " ").replace(/^\w/, (c) => c.toUpperCase());

export const confidenceLabel = (margin: number) => (margin >= 0.4 ? "Very high" : margin >= 0.2 ? "High" : margin >= 0.05 ? "Moderate" : "Borderline");

export const STATUS_TEXT: Record<string, { kind: "good" | "warning" | "serious"; label: string }> = {
  confirmed: { kind: "good", label: "Ground truth confirms" },
  other_persona: { kind: "warning", label: "Episode of another persona" },
  no_episode: { kind: "serious", label: "No ground-truth episode" },
};
