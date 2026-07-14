// Pure, stateless preprocessing types/helpers shared between the "New Preprocessed Dataset"
// builder (web/app/data/preprocessed/new/page.tsx) and anywhere else that needs to reason about
// indicator/clustering config without owning the React state for it.

export type IndicatorType = "sma" | "ema" | "rsi" | "macd" | "bbands" | "atr" | "returns" | "volatility";

export interface IndicatorCfg {
  type: IndicatorType;
  period?: number;
  fast?: number;
  slow?: number;
  signal?: number;
  std?: number;
}

export interface ClusteringCfg {
  enabled: boolean;
  n_clusters: number;
  on_cols: string[];
}

export const INDICATOR_DEFAULTS: Record<IndicatorType, Partial<IndicatorCfg>> = {
  sma:        { period: 20 },
  ema:        { period: 20 },
  rsi:        { period: 14 },
  macd:       { fast: 12, slow: 26, signal: 9 },
  bbands:     { period: 20, std: 2 },
  atr:        { period: 14 },
  returns:    { period: 1 },
  volatility: { period: 20 },
};

export const INDICATOR_LABELS: Record<IndicatorType, string> = {
  sma: "SMA", ema: "EMA", rsi: "RSI", macd: "MACD",
  bbands: "Bollinger Bands", atr: "ATR", returns: "Returns", volatility: "Volatility",
};

export function getOutputCols(cfg: IndicatorCfg): string[] {
  const p = cfg.period;
  switch (cfg.type) {
    case "sma":        return [`sma_${p ?? 20}`];
    case "ema":        return [`ema_${p ?? 20}`];
    case "rsi":        return [`rsi_${p ?? 14}`];
    case "macd":       return ["macd", "macd_signal", "macd_hist"];
    case "bbands":     return [`bb_upper_${p ?? 20}`, `bb_mid_${p ?? 20}`, `bb_lower_${p ?? 20}`, `bb_width_${p ?? 20}`];
    case "atr":        return [`atr_${p ?? 14}`];
    case "returns":    return [`returns_${p ?? 1}`];
    case "volatility": return [`vol_${p ?? 20}`];
    default:           return [];
  }
}

export const BASE_COLS = ["open", "high", "low", "close", "volume"];

export function getAllAvailableCols(indicators: IndicatorCfg[], clustering: ClusteringCfg): string[] {
  const cols = [...BASE_COLS];
  for (const ind of indicators) cols.push(...getOutputCols(ind));
  if (clustering.enabled) cols.push(`cluster_${clustering.n_clusters}`);
  return [...new Set(cols)];
}

export const DEFAULT_CLUSTERING: ClusteringCfg = { enabled: false, n_clusters: 5, on_cols: ["close"] };

/** Short human-readable summary of a preprocessing config, e.g. "RSI(14) + MACD, clustering(k=5)". */
export function summarizePreprocessing(preprocessing: { indicators?: IndicatorCfg[]; clustering?: ClusteringCfg | null } | null | undefined): string {
  const indicators = preprocessing?.indicators ?? [];
  const parts = indicators.map((i) => `${INDICATOR_LABELS[i.type] ?? i.type}${i.period != null ? `(${i.period})` : ""}`);
  if (preprocessing?.clustering?.enabled) parts.push(`clustering(k=${preprocessing.clustering.n_clusters})`);
  return parts.length > 0 ? parts.join(" + ") : "none";
}
