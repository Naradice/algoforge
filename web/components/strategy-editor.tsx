"use client";

import { useState, useEffect, useRef, lazy, Suspense } from "react";

const StrategyChartImport = lazy(() =>
  import("@/components/strategy-chart").then((m) => ({ default: m.StrategyChart }))
);

function StrategyChartLazy(props: React.ComponentProps<typeof import("@/components/strategy-chart").StrategyChart>) {
  return (
    <Suspense fallback={<div className="text-xs text-gray-500 py-4 text-center">Loading chart…</div>}>
      <StrategyChartImport {...props} />
    </Suspense>
  );
}

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type IndicatorCategory = "technical" | "ml" | "economic";

type IndicatorDef = {
  id: string;
  category: IndicatorCategory;
  type: string;
  params: Record<string, string | number>;
};

type ComparisonCondition = {
  left: string;
  op: string;
  right: string | number;
};
type StreakCondition = {
  type: "streak";
  left: string;
  op: string;
  right: string | number;
  min_streak: number;
};
type MlCondition = {
  type: "ml_signal";
  model_id: number;
  direction: string;
  step: number;
  min_confidence: number;
};
type LlmCondition = {
  type: "llm_signal";
  direction: string;
  lookback: number;
  model: string;
  columns: string[];
  cache: boolean;
};
type RegimeCondition = {
  type: "regime";
  mode: "range" | "trend" | "bull" | "bear";
  indicator: string;
  threshold: number;
};
type Condition = ComparisonCondition | StreakCondition | MlCondition | LlmCondition | RegimeCondition | GroupRefCondition;

type ColMeta = {
  name: string;
  kind: "float" | "enum";
  values?: { label: string; value: number | string }[];
  desc: string;
};

type DirectionalBlock = {
  enabled: boolean;
  conditions: Condition[];
  logic: "and" | "or";
};

type GroupDef = {
  id: string;
  name: string;
  conditions: Condition[];
  logic: "and" | "or";
};

type GroupRefCondition = {
  type: "group_ref";
  group_id: string;
};

type FormState = {
  symbol: string;
  indicators: IndicatorDef[];
  groups: GroupDef[];
  long_entry: DirectionalBlock;
  long_exit: DirectionalBlock;
  short_entry: DirectionalBlock;
  short_exit: DirectionalBlock;
};

// ---------------------------------------------------------------------------
// Indicator catalogue
// ---------------------------------------------------------------------------

const CATEGORIES: { value: IndicatorCategory; label: string }[] = [
  { value: "technical", label: "Technical" },
  { value: "ml",        label: "ML Output" },
  { value: "economic",  label: "Economic" },
];

const CATEGORY_TYPES: Record<IndicatorCategory, string[]> = {
  technical: ["macd", "rsi", "atr", "ema", "sma", "bb", "slope", "cci", "adx", "stochastic", "sar", "donchian", "rangetrend", "renko", "streak", "roc", "candle"],
  ml:        ["ml_inference"],
  economic:  ["macro_series"],
};

type ParamSpec = { key: string; label: string; inputType: "number" | "text" | "column"; default: number | string; step?: number };

const INDICATOR_PARAMS: Record<string, ParamSpec[]> = {
  macd:         [{ key: "fast", label: "Fast", inputType: "number", default: 12 }, { key: "slow", label: "Slow", inputType: "number", default: 26 }, { key: "signal_period", label: "Signal", inputType: "number", default: 9 }],
  rsi:          [{ key: "period", label: "Period", inputType: "number", default: 14 }],
  atr:          [{ key: "period", label: "Period", inputType: "number", default: 14 }],
  ema:          [{ key: "period", label: "Period", inputType: "number", default: 20 }],
  sma:          [{ key: "period", label: "Period", inputType: "number", default: 20 }],
  bb:           [{ key: "period", label: "Period", inputType: "number", default: 20 }, { key: "std_dev", label: "Std Dev", inputType: "number", default: 2 }],
  slope:        [{ key: "period", label: "Period", inputType: "number", default: 5 }, { key: "column", label: "Column", inputType: "column", default: "close" }],
  cci:          [{ key: "period", label: "Period", inputType: "number", default: 14 }],
  adx:          [{ key: "period", label: "Period", inputType: "number", default: 14 }],
  stochastic:   [{ key: "k_period", label: "%K Period", inputType: "number", default: 14 }, { key: "d_period", label: "%D Period", inputType: "number", default: 3 }],
  sar:          [{ key: "af_start", label: "AF Start", inputType: "number", default: 0.02 }, { key: "af_max", label: "AF Max", inputType: "number", default: 0.2 }],
  donchian:     [{ key: "period", label: "Period", inputType: "number", default: 20 }],
  rangetrend:   [],  // method-specific params rendered separately via RANGETREND_METHOD_PARAMS
  roc:          [{ key: "period", label: "Period", inputType: "number", default: 10 }, { key: "column", label: "Column", inputType: "column", default: "close" }],
  renko:        [{ key: "atr_window", label: "ATR Window", inputType: "number", default: 14 }, { key: "brick_size", label: "Brick Size (0=ATR)", inputType: "number", default: 0 }],
  streak:       [],  // left / op / right rendered specially (same shape as a comparison condition)
  candle:       [{ key: "ratio", label: "Pin Ratio", inputType: "number", default: 2.0 }],
  ml_inference: [{ key: "model_id", label: "Model ID", inputType: "number", default: 1 }, { key: "output_col", label: "Output column", inputType: "text", default: "ml_score" }],
  macro_series: [{ key: "source", label: "Source", inputType: "text", default: "fred" }, { key: "series", label: "Series ID", inputType: "text", default: "CPIAUCSL" }, { key: "lag", label: "Lag bars", inputType: "number", default: 0 }],
};

// ---------------------------------------------------------------------------
// Rangetrend method catalogue
// ---------------------------------------------------------------------------

const RANGETREND_METHODS = ["bband", "atr", "bollinger", "swing", "adx", "ma_deviation"] as const;
type RangeTrendMethod = typeof RANGETREND_METHODS[number];

const RANGETREND_METHOD_LABELS: Record<RangeTrendMethod, string> = {
  bband:        "BB Width (continuous)",
  atr:          "ATR vs Mean",
  bollinger:    "Bollinger Std",
  swing:        "Swing Width",
  adx:          "ADX Threshold",
  ma_deviation: "MA Deviation",
};

const RANGETREND_METHOD_PARAMS: Record<RangeTrendMethod, ParamSpec[]> = {
  bband:        [{ key: "slope_window", label: "Slope Window", inputType: "number", default: 4 }, { key: "bb_period", label: "BB Period", inputType: "number", default: 20 }],
  atr:          [{ key: "mean_window", label: "Mean Window", inputType: "number", default: 100 }, { key: "atr_window", label: "ATR Window", inputType: "number", default: 14 }, { key: "range_threshold", label: "Threshold", inputType: "number", default: 0.7, step: 0.05 }],
  bollinger:    [{ key: "std_window", label: "Std Window", inputType: "number", default: 200 }, { key: "window", label: "BB Period", inputType: "number", default: 20 }, { key: "std_threshold", label: "Threshold", inputType: "number", default: 0.6, step: 0.05 }],
  swing:        [{ key: "window", label: "Window", inputType: "number", default: 50 }, { key: "width_threshold", label: "Width Threshold", inputType: "number", default: 0.015, step: 0.001 }],
  adx:          [{ key: "adx_window", label: "ADX Window", inputType: "number", default: 14 }, { key: "adx_threshold", label: "ADX Threshold", inputType: "number", default: 25 }],
  ma_deviation: [{ key: "short_window", label: "Short MA", inputType: "number", default: 10 }, { key: "long_window", label: "Long MA", inputType: "number", default: 50 }, { key: "deviation_threshold", label: "Deviation", inputType: "number", default: 0.005, step: 0.001 }],
};

function defaultParams(type: string): Record<string, string | number> {
  if (type === "rangetrend") {
    const methodParams = RANGETREND_METHOD_PARAMS["bband"];
    return { method: "bband", ...Object.fromEntries(methodParams.map((s) => [s.key, s.default])) };
  }
  if (type === "streak") return { left: "close", op: ">", right: "open" };
  return Object.fromEntries((INDICATOR_PARAMS[type] ?? []).map((s) => [s.key, s.default]));
}

function defaultTypeForCategory(cat: IndicatorCategory): string {
  return CATEGORY_TYPES[cat][0];
}

/** Auto-generate id from type and existing list */
function suggestId(type: string, indicators: IndicatorDef[], excludeIndex?: number): string {
  const others = excludeIndex !== undefined ? indicators.filter((_, i) => i !== excludeIndex) : indicators;
  const sameType = others.filter((i) => i.type === type).length;
  return sameType === 0 ? type : `${type}${sameType + 1}`;
}

/** True if the id looks auto-generated for oldType */
function isAutoId(id: string, oldType: string): boolean {
  return id === oldType || /^\d+$/.test(id.slice(oldType.length));
}

/** Derive column metadata (name + value type) from indicators */
const TYPE_DESC: Record<string, string> = {
  rsi:          "Relative Strength Index (0–100; >70 overbought, <30 oversold)",
  ema:          "Exponential Moving Average",
  sma:          "Simple Moving Average",
  atr:          "Average True Range — measures bar-to-bar volatility",
  slope:        "Linear regression slope of a column over the period",
  ml_inference: "ML model output score",
  macro_series: "Macroeconomic data series value",
};

function deriveColumnMeta(indicators: IndicatorDef[]): ColMeta[] {
  const f = (name: string, desc: string): ColMeta => ({ name, kind: "float", desc });
  const e = (name: string, desc: string, values: { label: string; value: number | string }[]): ColMeta => ({ name, kind: "enum", desc, values });

  const base: ColMeta[] = [
    f("open",   "Bar open price"),
    f("high",   "Bar high price"),
    f("low",    "Bar low price"),
    f("close",  "Bar close price"),
    f("volume", "Bar volume"),
  ];
  const derived: ColMeta[] = [];

  for (const ind of indicators) {
    switch (ind.type) {
      case "macd":
        derived.push(
          f(`${ind.id}_line`,   "MACD line (fast EMA − slow EMA)"),
          f(`${ind.id}_signal`, "Signal line (EMA of the MACD line)"),
          f(`${ind.id}_hist`,   "Histogram (MACD line − signal line)"),
        ); break;
      case "bb":
        derived.push(
          f(`${ind.id}_upper`,  `Upper band (middle + ${ind.params.std_dev ?? 2}× std dev)`),
          f(`${ind.id}_middle`, "Middle band (SMA)"),
          f(`${ind.id}_lower`,  `Lower band (middle − ${ind.params.std_dev ?? 2}× std dev)`),
        ); break;
      case "adx":
        derived.push(
          f(ind.id,                  "ADX — trend strength (0–100; >25 = strong trend)"),
          f(`${ind.id}_plus_di`,     "+DI — upward directional indicator"),
          f(`${ind.id}_minus_di`,    "−DI — downward directional indicator"),
        ); break;
      case "stochastic":
        derived.push(
          f(`${ind.id}_k`, "%K — raw stochastic oscillator (0–100)"),
          f(`${ind.id}_d`, "%D — smoothed %K signal line"),
        ); break;
      case "donchian":
        derived.push(
          f(`${ind.id}_upper`, `Highest high over ${ind.params.period ?? "N"} bars`),
          f(`${ind.id}_lower`, `Lowest low over ${ind.params.period ?? "N"} bars`),
          f(`${ind.id}_mid`,   "Midpoint (upper + lower) / 2"),
        ); break;
      case "renko":
        derived.push(
          e(`${ind.id}_direction`, "Current brick direction", [{ label: "Long (+1)", value: 1 }, { label: "Short (−1)", value: -1 }]),
          e(`${ind.id}_flip`,      "Brick flip this bar",     [{ label: "No flip (0)", value: 0 }, { label: "Up flip (+1)", value: 1 }, { label: "Down flip (−1)", value: -1 }]),
          f(`${ind.id}_momentum`,  "Forward-filled Renko diff used to match legacy consecutive-brick MACDRenko thresholds"),
          f(`${ind.id}_bricksize`, `Brick size in price units (${ind.params.brick_size > 0 ? "fixed" : "ATR-based"})`),
        ); break;
      case "rangetrend":
        if (!ind.params.method || ind.params.method === "bband") {
          derived.push(
            f(`${ind.id}_trend`, "Trend score (positive = uptrend, negative = downtrend)"),
            f(`${ind.id}_range`, "Range probability (0–1; higher = more mean-reverting)"),
          );
        } else {
          derived.push(e(`${ind.id}_is_range`, "Market regime", [{ label: "Trending (0)", value: 0 }, { label: "Range (1)", value: 1 }]));
        }
        break;
      case "candle": {
        const CANDLE_DESC: Record<string, string> = {
          bull_engulf:   "Bullish engulfing candle",
          bear_engulf:   "Bearish engulfing candle",
          bull_pin:      "Bullish pin bar (hammer)",
          bear_pin:      "Bearish pin bar (shooting star)",
          bull_outside:  "Bullish outside bar",
          bear_outside:  "Bearish outside bar",
        };
        for (const [pat, desc] of Object.entries(CANDLE_DESC)) {
          derived.push(e(`${ind.id}_${pat}`, desc, [{ label: "No (0)", value: 0 }, { label: "Yes (1)", value: 1 }]));
        }
        break;
      }
      case "ml_inference":
        derived.push(f(String(ind.params.output_col ?? ind.id), `ML model #${ind.params.model_id ?? "?"} output score`)); break;
      default:
        derived.push(f(ind.id, TYPE_DESC[ind.type] ?? ind.type));
    }
  }
  return [...base, ...derived];
}

/** Derive all available column names from indicators (used where only names are needed) */
function deriveColumns(indicators: IndicatorDef[]): string[] {
  return deriveColumnMeta(indicators).map((m) => m.name);
}

/** Human-readable summary line for a collapsed indicator row */
function indicatorSummary(ind: IndicatorDef): string {
  const pairs = Object.entries(ind.params).map(([k, v]) => `${k}=${v}`).join(", ");
  return `${ind.type}${pairs ? ` (${pairs})` : ""}`;
}

// ---------------------------------------------------------------------------
// Condition helpers
// ---------------------------------------------------------------------------

type CondType = "comparison" | "streak" | "ml" | "llm" | "regime" | "group_ref";

function condType(c: Condition): CondType {
  const t = (c as any).type;
  if (t === "ml_signal")  return "ml";
  if (t === "llm_signal") return "llm";
  if (t === "streak")     return "streak";
  if (t === "regime")     return "regime";
  if (t === "group_ref")  return "group_ref";
  return "comparison";
}

const defaultComparison = (): ComparisonCondition => ({ left: "close", op: ">", right: "open" });
const defaultStreak = (): StreakCondition => ({ type: "streak", left: "close", op: ">", right: "open", min_streak: 3 });
const defaultGroupRef = (groups: GroupDef[]): GroupRefCondition => ({ type: "group_ref", group_id: groups[0]?.id ?? "" });
const defaultMl = (): MlCondition => ({ type: "ml_signal", model_id: 1, direction: "buy", step: 1, min_confidence: 0.0 });
const defaultLlm = (): LlmCondition => ({ type: "llm_signal", direction: "buy", lookback: 10, model: "gemini-2.0-flash", columns: ["close", "volume"], cache: true });
const defaultRegime = (): RegimeCondition => ({ type: "regime", mode: "range", indicator: "rt", threshold: 0.5 });
const defaultBlock = (): DirectionalBlock => ({ enabled: false, conditions: [defaultComparison()], logic: "and" });

// ---------------------------------------------------------------------------
// Parse / serialise
// ---------------------------------------------------------------------------

function parseBlock(raw: any): DirectionalBlock {
  if (!raw) return defaultBlock();
  return {
    enabled: true,
    conditions: Array.isArray(raw.conditions) ? raw.conditions : [defaultComparison()],
    logic: raw.logic ?? "and",
  };
}

function parseDefinition(def: any): FormState {
  const indicators: IndicatorDef[] = (def.indicators ?? []).map((i: any) => ({
    id:       i.id ?? "",
    category: (i.category as IndicatorCategory) ?? "technical",
    type:     i.type ?? "rsi",
    params:   i.params ?? {},
  }));

  const rawGroups: Record<string, any> = def.groups ?? {};
  const groups: GroupDef[] = Object.entries(rawGroups).map(([id, g]: [string, any]) => ({
    id,
    name:       g.name ?? id,
    conditions: Array.isArray(g.conditions) ? g.conditions : [defaultComparison()],
    logic:      g.logic ?? "and",
  }));

  // Support old format { entry: { direction: "buy", ... }, exit: ... }
  const oldEntry = def.entry && !def.long && !def.short ? def.entry : null;
  const oldExit  = def.exit  && !def.long && !def.short ? def.exit  : null;
  const oldDir   = oldEntry?.direction ?? "buy";

  const long_entry  = def.long?.entry  ? parseBlock(def.long.entry)  : (oldEntry && oldDir === "buy"  ? parseBlock(oldEntry) : defaultBlock());
  const long_exit   = def.long?.exit   ? parseBlock(def.long.exit)   : (oldExit  && oldDir === "buy"  ? parseBlock(oldExit)  : defaultBlock());
  const short_entry = def.short?.entry ? parseBlock(def.short.entry) : (oldEntry && oldDir === "sell" ? parseBlock(oldEntry) : defaultBlock());
  const short_exit  = def.short?.exit  ? parseBlock(def.short.exit)  : (oldExit  && oldDir === "sell" ? parseBlock(oldExit)  : defaultBlock());

  return {
    symbol: def.symbol ?? "AAPL",
    indicators,
    groups,
    long_entry,
    long_exit,
    short_entry,
    short_exit,
  };
}

function serialiseBlock(b: DirectionalBlock, direction: string): object | null {
  if (!b.enabled) return null;
  return { direction, conditions: b.conditions.map(serialiseCondition), logic: b.logic };
}

function serialiseCondition(c: Condition): object {
  const t = condType(c);
  if (t === "ml") {
    const m = c as MlCondition;
    return { type: "ml_signal", model_id: m.model_id, direction: m.direction, step: m.step, min_confidence: m.min_confidence };
  }
  if (t === "llm") {
    const l = c as LlmCondition;
    return { type: "llm_signal", direction: l.direction, lookback: l.lookback, model: l.model, columns: l.columns, cache: l.cache };
  }
  if (t === "streak") {
    const s = c as StreakCondition;
    const right = isNaN(Number(s.right)) || String(s.right).trim() === "" ? s.right : Number(s.right);
    return { type: "streak", left: s.left, op: s.op, right, min_streak: s.min_streak };
  }
  if (t === "group_ref") {
    const g = c as GroupRefCondition;
    return { type: "group_ref", group_id: g.group_id };
  }
  const cmp = c as ComparisonCondition;
  const right = isNaN(Number(cmp.right)) || String(cmp.right).trim() === "" ? cmp.right : Number(cmp.right);
  return { left: cmp.left, op: cmp.op, right };
}

function serialiseForm(f: FormState): object {
  const long: any = {};
  if (f.long_entry.enabled)  long.entry = serialiseBlock(f.long_entry, "buy");
  if (f.long_exit.enabled)   long.exit  = serialiseBlock(f.long_exit, "buy");
  const short: any = {};
  if (f.short_entry.enabled) short.entry = serialiseBlock(f.short_entry, "sell");
  if (f.short_exit.enabled)  short.exit  = serialiseBlock(f.short_exit, "sell");

  const groups: Record<string, object> = {};
  for (const g of f.groups) {
    groups[g.id] = { name: g.name, conditions: g.conditions.map(serialiseCondition), logic: g.logic };
  }

  return {
    symbol: f.symbol,
    indicators: f.indicators.map((i) => ({ id: i.id, category: i.category, type: i.type, params: i.params })),
    ...(f.groups.length ? { groups } : {}),
    ...(Object.keys(long).length  ? { long }  : {}),
    ...(Object.keys(short).length ? { short } : {}),
  };
}

// ---------------------------------------------------------------------------
// Primitive inputs
// ---------------------------------------------------------------------------

function Sel({ value, onChange, options, className }: { value: string; onChange: (v: string) => void; options: { value: string; label: string }[]; className?: string }) {
  return (
    <select value={value} onChange={(e) => onChange(e.target.value)}
      className={`rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500 ${className ?? ""}`}>
      {options.map((o) => <option key={o.value} value={o.value}>{o.label}</option>)}
    </select>
  );
}

function TxtIn({ value, onChange, placeholder, className }: { value: string; onChange: (v: string) => void; placeholder?: string; className?: string }) {
  return (
    <input value={value} onChange={(e) => onChange(e.target.value)} placeholder={placeholder}
      className={`rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500 ${className ?? ""}`} />
  );
}

function NumIn({ value, onChange, step, min, max, className }: { value: number; onChange: (v: number) => void; step?: number; min?: number; max?: number; className?: string }) {
  return (
    <input type="number" value={value} step={step ?? 1} min={min} max={max} onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
      className={`rounded border border-gray-700 bg-gray-800 px-2 py-1 text-xs text-white w-20 focus:outline-none focus:ring-1 focus:ring-brand-500 ${className ?? ""}`} />
  );
}

// ---------------------------------------------------------------------------
// Indicator section
// ---------------------------------------------------------------------------

function IndicatorExpandedEditor({
  ind, allIndicators, index, onChange, onRemove,
}: {
  ind: IndicatorDef; allIndicators: IndicatorDef[]; index: number;
  onChange: (updated: IndicatorDef) => void; onRemove: () => void;
}) {
  const paramSpecs = INDICATOR_PARAMS[ind.type] ?? [];
  const columns = deriveColumns(allIndicators);

  function changeCategory(cat: IndicatorCategory) {
    const newType = defaultTypeForCategory(cat);
    const newId = suggestId(newType, allIndicators, index);
    onChange({ ...ind, category: cat, type: newType, id: newId, params: defaultParams(newType) });
  }

  function changeType(newType: string) {
    const newId = isAutoId(ind.id, ind.type) ? suggestId(newType, allIndicators, index) : ind.id;
    onChange({ ...ind, type: newType, id: newId, params: defaultParams(newType) });
  }

  return (
    <div className="rounded border border-brand-500/40 bg-gray-900 p-3 space-y-3">
      <div className="flex flex-wrap gap-2 items-center">
        {/* Category */}
        <Sel value={ind.category} onChange={(v) => changeCategory(v as IndicatorCategory)}
          options={CATEGORIES} />
        {/* Type within category */}
        <Sel value={ind.type} onChange={changeType}
          options={(CATEGORY_TYPES[ind.category] ?? []).map((t) => ({ value: t, label: t }))} />
        {/* ID */}
        <span className="text-xs text-gray-500">id →</span>
        <TxtIn value={ind.id} onChange={(v) => onChange({ ...ind, id: v })} placeholder="column name" className="w-28" />
        <button onClick={onRemove} className="ml-auto text-gray-600 hover:text-red-400 text-xs">Remove</button>
      </div>

      {/* Rangetrend method selector + method-specific params */}
      {ind.type === "rangetrend" && (() => {
        const method = (String(ind.params.method ?? "bband")) as RangeTrendMethod;
        const methodSpecs = RANGETREND_METHOD_PARAMS[method] ?? [];
        return (
          <div className="flex flex-wrap gap-3 items-center">
            <label className="flex items-center gap-1">
              <span className="text-xs text-gray-400">Method</span>
              <Sel value={method} onChange={(v) => {
                const newSpecs = RANGETREND_METHOD_PARAMS[v as RangeTrendMethod] ?? [];
                const newMethodParams = Object.fromEntries(newSpecs.map((s) => [s.key, s.default]));
                onChange({ ...ind, params: { method: v, ...newMethodParams } });
              }} options={RANGETREND_METHODS.map((m) => ({ value: m, label: RANGETREND_METHOD_LABELS[m] }))} />
            </label>
            {methodSpecs.map((spec) => (
              <label key={spec.key} className="flex items-center gap-1">
                <span className="text-xs text-gray-400">{spec.label}</span>
                <NumIn value={Number(ind.params[spec.key] ?? spec.default)} step={spec.step}
                  onChange={(v) => onChange({ ...ind, params: { ...ind.params, [spec.key]: v } })}
                  className="w-20" />
              </label>
            ))}
          </div>
        );
      })()}

      {/* Streak: condition whose consecutive-true count becomes the output column */}
      {ind.type === "streak" && (() => {
        const left  = String(ind.params.left  ?? "close");
        const op    = String(ind.params.op    ?? ">");
        const right = ind.params.right ?? 0;
        const rightIsCol = columns.includes(String(right));
        const colOpts = columns.map((c) => ({ value: c, label: c }));
        return (
          <div className="flex flex-wrap items-center gap-2">
            <span className="text-xs text-gray-500">Count consecutive bars where</span>
            <Sel value={left} onChange={(v) => onChange({ ...ind, params: { ...ind.params, left: v } })} options={colOpts} />
            <Sel value={op}   onChange={(v) => onChange({ ...ind, params: { ...ind.params, op: v } })}
              options={OPS.map((o) => ({ value: o, label: o }))} />
            {rightIsCol
              ? <Sel value={String(right)} onChange={(v) => onChange({ ...ind, params: { ...ind.params, right: v } })} options={colOpts} />
              : <TxtIn value={String(right)} onChange={(v) => onChange({ ...ind, params: { ...ind.params, right: v } })} placeholder="value" className="w-24" />}
            <button className="text-[10px] text-gray-500 hover:text-gray-300 border border-gray-700 rounded px-1"
              onClick={() => onChange({ ...ind, params: { ...ind.params, right: rightIsCol ? 0 : (columns[0] ?? "close") } })}
              title={rightIsCol ? "Switch to numeric" : "Switch to column"}>
              {rightIsCol ? "→ num" : "→ col"}
            </button>
            <span className="text-xs text-gray-500">is true</span>
          </div>
        );
      })()}

      {/* Params for all other indicator types */}
      {ind.type !== "rangetrend" && ind.type !== "streak" && paramSpecs.length > 0 && (
        <div className="flex flex-wrap gap-3">
          {paramSpecs.map((spec) => (
            <label key={spec.key} className="flex items-center gap-1">
              <span className="text-xs text-gray-400">{spec.label}</span>
              {spec.inputType === "number" ? (
                <NumIn value={Number(ind.params[spec.key] ?? spec.default)} step={spec.step}
                  onChange={(v) => onChange({ ...ind, params: { ...ind.params, [spec.key]: v } })}
                  className="w-16" />
              ) : spec.inputType === "column" ? (
                <Sel
                  value={String(ind.params[spec.key] ?? spec.default)}
                  onChange={(v) => onChange({ ...ind, params: { ...ind.params, [spec.key]: v } })}
                  options={columns.map((c) => ({ value: c, label: c }))}
                />
              ) : (
                <TxtIn value={String(ind.params[spec.key] ?? spec.default)}
                  onChange={(v) => onChange({ ...ind, params: { ...ind.params, [spec.key]: v } })}
                  className="w-24" />
              )}
            </label>
          ))}
        </div>
      )}
    </div>
  );
}

function IndicatorsSection({
  indicators, onChange,
}: {
  indicators: IndicatorDef[];
  onChange: (inds: IndicatorDef[]) => void;
}) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);
  const [showColumns, setShowColumns] = useState(false);
  const columnMeta = deriveColumnMeta(indicators);

  function addIndicator() {
    const cat: IndicatorCategory = "technical";
    const type = defaultTypeForCategory(cat);
    const ind: IndicatorDef = { id: suggestId(type, indicators), category: cat, type, params: defaultParams(type) };
    const next = [...indicators, ind];
    onChange(next);
    setExpandedIndex(next.length - 1);
  }

  function updateAt(i: number, ind: IndicatorDef) {
    const next = [...indicators];
    next[i] = ind;
    onChange(next);
  }

  function removeAt(i: number) {
    onChange(indicators.filter((_, j) => j !== i));
    setExpandedIndex(null);
  }

  return (
    <section>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-gray-400">Indicators / Data sources</h3>
        <button type="button" onClick={addIndicator} className="text-xs text-gray-400 hover:text-white">+ Add</button>
      </div>

      {indicators.length === 0 && (
        <p className="text-xs text-gray-600">No indicators. Conditions can still use OHLCV base columns.</p>
      )}

      <div className="space-y-1">
        {indicators.map((ind, i) => (
          <div key={i}>
            <div
              onClick={() => setExpandedIndex(expandedIndex === i ? null : i)}
              className={`flex items-center gap-2 px-3 py-2 rounded cursor-pointer text-xs border ${
                expandedIndex === i
                  ? "border-brand-500/40 bg-gray-900"
                  : "border-gray-700 bg-gray-900/50 hover:bg-gray-900"
              }`}
            >
              <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${
                ind.category === "technical" ? "bg-sky-900 text-sky-300" :
                ind.category === "ml"        ? "bg-purple-900 text-purple-300" :
                                               "bg-amber-900 text-amber-300"
              }`}>
                {ind.category === "ml" ? "ML" : ind.category === "economic" ? "ECON" : "TA"}
              </span>
              <span className="font-mono text-gray-200 font-medium">{ind.id}</span>
              <span className="text-gray-500">=</span>
              <span className="text-gray-400">{indicatorSummary(ind)}</span>
              <span className="ml-auto text-gray-600">{expandedIndex === i ? "▲" : "▼"}</span>
            </div>

            {expandedIndex === i && (
              <div className="mt-1">
                <IndicatorExpandedEditor
                  ind={ind}
                  allIndicators={indicators}
                  index={i}
                  onChange={(updated) => updateAt(i, updated)}
                  onRemove={() => removeAt(i)}
                />
              </div>
            )}
          </div>
        ))}
      </div>

      {/* Available columns reference */}
      <div className="mt-3">
        <button
          type="button"
          onClick={() => setShowColumns(!showColumns)}
          className="text-[10px] text-gray-500 hover:text-gray-300"
        >
          {showColumns ? "▲ Hide" : "▼ Show"} available columns ({columnMeta.length})
        </button>

        {showColumns && (
          <div className="mt-2 rounded border border-gray-800 bg-gray-950 divide-y divide-gray-800">
            {columnMeta.map((col) => (
              <div key={col.name} className="flex items-start gap-3 px-3 py-1.5">
                <span className="font-mono text-xs text-gray-200 shrink-0 w-40">{col.name}</span>
                <span className="text-[10px] text-gray-500 leading-tight">{col.desc}</span>
                {col.kind === "enum" && col.values && (
                  <span className="ml-auto text-[10px] text-gray-600 shrink-0">
                    {col.values.map((v) => v.label).join(" / ")}
                  </span>
                )}
              </div>
            ))}
          </div>
        )}
      </div>
    </section>
  );
}

// ---------------------------------------------------------------------------
// Condition frequency stats hook
// ---------------------------------------------------------------------------
// Condition section
// ---------------------------------------------------------------------------

const OPS = [">", "<", ">=", "<=", "==", "!="];
const LLM_MODELS = ["gemini-2.0-flash", "gemini-1.5-pro", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"];

function ConditionRow({
  cond, columnMeta, groups, onChange, onRemove,
}: {
  cond: Condition; columnMeta: ColMeta[]; groups: GroupDef[];
  onChange: (c: Condition) => void; onRemove: () => void;
}) {
  const t = condType(cond);
  const columns = columnMeta.map((m) => m.name);
  const colOpts = columnMeta.map((m) => ({ value: m.name, label: m.name }));

  return (
    <div className="rounded border border-gray-700 bg-gray-900 p-2 space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        <Sel value={t} onChange={(newT) => {
          if (newT === "ml")           onChange(defaultMl());
          else if (newT === "llm")     onChange(defaultLlm());
          else if (newT === "regime")  onChange(defaultRegime());
          else if (newT === "streak")  onChange(defaultStreak());
          else if (newT === "group_ref") onChange(defaultGroupRef(groups));
          else                         onChange(defaultComparison());
        }} options={[
          { value: "comparison", label: "Compare" },
          { value: "streak",     label: "Streak" },
          { value: "group_ref",  label: "Group" },
          { value: "regime",     label: "Regime" },
          { value: "ml",         label: "ML Signal" },
          { value: "llm",        label: "LLM Signal" },
        ]} />

        {t === "comparison" && (() => {
          const c = cond as ComparisonCondition;
          const leftMeta = columnMeta.find((m) => m.name === c.left);
          const leftIsEnum = leftMeta?.kind === "enum";
          const rightIsCol = !leftIsEnum && columns.includes(String(c.right));

          function handleLeftChange(v: string) {
            const newMeta = columnMeta.find((m) => m.name === v);
            // Auto-set right to first enum value when switching to an enum column
            const newRight = newMeta?.kind === "enum" ? (newMeta.values?.[0]?.value ?? 0) : c.right;
            onChange({ ...c, left: v, right: newRight });
          }

          return (
            <>
              <Sel value={c.left} onChange={handleLeftChange} options={colOpts} />
              <Sel value={c.op}   onChange={(v) => onChange({ ...c, op: v })}   options={OPS.map((o) => ({ value: o, label: o }))} />
              {leftIsEnum
                ? (
                  <Sel
                    value={String(c.right)}
                    onChange={(v) => onChange({ ...c, right: isNaN(Number(v)) ? v : Number(v) })}
                    options={(leftMeta!.values ?? []).map((opt) => ({ value: String(opt.value), label: opt.label }))}
                  />
                )
                : rightIsCol
                  ? <Sel value={String(c.right)} onChange={(v) => onChange({ ...c, right: v })} options={colOpts} />
                  : <TxtIn value={String(c.right)} onChange={(v) => onChange({ ...c, right: v })} placeholder="number" className="w-24" />
              }
              {!leftIsEnum && (
                <button type="button" className="text-[10px] text-gray-500 hover:text-gray-300 border border-gray-700 rounded px-1"
                  onClick={() => onChange({ ...c, right: rightIsCol ? 0 : (columns[0] ?? "close") })}
                  title={rightIsCol ? "Switch to numeric" : "Switch to column"}>
                  {rightIsCol ? "→ num" : "→ col"}
                </button>
              )}
            </>
          );
        })()}

        {t === "streak" && (() => {
          const s = cond as StreakCondition;
          const leftMeta = columnMeta.find((m) => m.name === s.left);
          const leftIsEnum = leftMeta?.kind === "enum";
          const rightIsCol = !leftIsEnum && columns.includes(String(s.right));

          function handleLeftChange(v: string) {
            const newMeta = columnMeta.find((m) => m.name === v);
            const newRight = newMeta?.kind === "enum" ? (newMeta.values?.[0]?.value ?? 0) : s.right;
            onChange({ ...s, left: v, right: newRight });
          }

          return (
            <>
              <Sel value={s.left} onChange={handleLeftChange} options={colOpts} />
              <Sel value={s.op}   onChange={(v) => onChange({ ...s, op: v })}   options={OPS.map((o) => ({ value: o, label: o }))} />
              {leftIsEnum
                ? (
                  <Sel
                    value={String(s.right)}
                    onChange={(v) => onChange({ ...s, right: isNaN(Number(v)) ? v : Number(v) })}
                    options={(leftMeta!.values ?? []).map((opt) => ({ value: String(opt.value), label: opt.label }))}
                  />
                )
                : rightIsCol
                  ? <Sel value={String(s.right)} onChange={(v) => onChange({ ...s, right: v })} options={colOpts} />
                  : <TxtIn value={String(s.right)} onChange={(v) => onChange({ ...s, right: v })} placeholder="number" className="w-24" />
              }
              {!leftIsEnum && (
                <button type="button" className="text-[10px] text-gray-500 hover:text-gray-300 border border-gray-700 rounded px-1"
                  onClick={() => onChange({ ...s, right: rightIsCol ? 0 : (columns[0] ?? "close") })}>
                  {rightIsCol ? "→ num" : "→ col"}
                </button>
              )}
              <span className="text-xs text-gray-500">for ≥</span>
              <NumIn value={s.min_streak} step={1} min={1} onChange={(v) => onChange({ ...s, min_streak: Math.max(1, Math.round(v)) })} className="w-14" />
              <span className="text-xs text-gray-500">bars</span>
            </>
          );
        })()}

        {t === "ml" && (() => {
          const m = cond as MlCondition;
          return (
            <>
              <span className="text-xs text-gray-500">model:</span>
              <NumIn value={m.model_id} onChange={(v) => onChange({ ...m, model_id: v })} className="w-14" />
              <Sel value={m.direction} onChange={(v) => onChange({ ...m, direction: v })}
                options={[{ value: "buy", label: "Buy" }, { value: "sell", label: "Sell" }]} />
              <span className="text-xs text-gray-500">step:</span>
              <NumIn value={m.step} onChange={(v) => onChange({ ...m, step: v })} className="w-12" />
              <span className="text-xs text-gray-500">min_conf:</span>
              <NumIn value={m.min_confidence} step={0.01} min={0} onChange={(v) => onChange({ ...m, min_confidence: v })} className="w-14" />
            </>
          );
        })()}

        {t === "llm" && (() => {
          const l = cond as LlmCondition;
          return (
            <>
              <Sel value={l.direction} onChange={(v) => onChange({ ...l, direction: v })}
                options={[{ value: "buy", label: "Buy" }, { value: "sell", label: "Sell" }]} />
              <span className="text-xs text-gray-500">lookback:</span>
              <NumIn value={l.lookback} onChange={(v) => onChange({ ...l, lookback: v })} className="w-14" />
              <Sel value={l.model} onChange={(v) => onChange({ ...l, model: v })}
                options={LLM_MODELS.map((m) => ({ value: m, label: m }))} />
            </>
          );
        })()}

        {t === "regime" && (() => {
          const r = cond as RegimeCondition;
          return (
            <>
              <Sel value={r.mode} onChange={(v) => onChange({ ...r, mode: v as RegimeCondition["mode"] })}
                options={[{ value: "range", label: "Range" }, { value: "trend", label: "Trend" }, { value: "bull", label: "Bull" }, { value: "bear", label: "Bear" }]} />
              <span className="text-xs text-gray-500">indicator:</span>
              <TxtIn value={r.indicator} onChange={(v) => onChange({ ...r, indicator: v })} placeholder="rt" className="w-20" />
              <span className="text-xs text-gray-500">threshold:</span>
              <NumIn value={r.threshold} step={0.05} min={0} max={1} onChange={(v) => onChange({ ...r, threshold: v })} className="w-16" />
            </>
          );
        })()}

        {t === "group_ref" && (() => {
          const g = cond as GroupRefCondition;
          if (groups.length === 0) {
            return <span className="text-xs text-amber-400">No groups defined yet — add one above.</span>;
          }
          return (
            <Sel
              value={g.group_id}
              onChange={(v) => onChange({ ...g, group_id: v })}
              options={groups.map((grp) => ({ value: grp.id, label: grp.name }))}
            />
          );
        })()}

        <button onClick={onRemove} className="text-gray-600 hover:text-red-400 text-xs">✕</button>
      </div>

      {t === "llm" && (() => {
        const l = cond as LlmCondition;
        return (
          <div className="flex flex-wrap gap-x-3 gap-y-1 pl-2">
            <span className="text-[10px] text-gray-500 self-center">columns:</span>
            {columns.map((col) => (
              <label key={col} className="flex items-center gap-1 text-[10px] text-gray-300 cursor-pointer">
                <input type="checkbox" checked={l.columns.includes(col)}
                  onChange={(e) => onChange({ ...l, columns: e.target.checked ? [...l.columns, col] : l.columns.filter((c) => c !== col) })}
                  className="accent-sky-500" />
                {col}
              </label>
            ))}
            <label className="flex items-center gap-1 text-[10px] text-gray-500 cursor-pointer ml-2">
              <input type="checkbox" checked={l.cache}
                onChange={(e) => onChange({ ...l, cache: e.target.checked })} className="accent-sky-500" />
              cache
            </label>
          </div>
        );
      })()}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Condition groups section
// ---------------------------------------------------------------------------

function ConditionGroupsSection({
  groups, columnMeta, onChange,
}: {
  groups: GroupDef[];
  columnMeta: ColMeta[];
  onChange: (groups: GroupDef[]) => void;
}) {
  const [expandedIndex, setExpandedIndex] = useState<number | null>(null);

  function addGroup() {
    const id = `group${groups.length + 1}`;
    const next = [...groups, { id, name: `Group ${groups.length + 1}`, conditions: [defaultComparison()], logic: "and" as const }];
    onChange(next);
    setExpandedIndex(next.length - 1);
  }

  function updateAt(i: number, g: GroupDef) {
    const next = [...groups]; next[i] = g; onChange(next);
  }

  function removeAt(i: number) {
    onChange(groups.filter((_, j) => j !== i));
    setExpandedIndex(null);
  }

  return (
    <section>
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-xs font-medium uppercase tracking-wide text-gray-400">Condition Groups</h3>
        <button type="button" onClick={addGroup} className="text-xs text-gray-400 hover:text-white">+ Add group</button>
      </div>

      {groups.length === 0 && (
        <p className="text-xs text-gray-600">No groups. Add one to reuse a set of conditions across entry and exit blocks.</p>
      )}

      <div className="space-y-1">
        {groups.map((g, i) => (
          <div key={i}>
            <div
              onClick={() => setExpandedIndex(expandedIndex === i ? null : i)}
              className={`flex items-center gap-2 px-3 py-2 rounded cursor-pointer text-xs border ${
                expandedIndex === i ? "border-brand-500/40 bg-gray-900" : "border-gray-700 bg-gray-900/50 hover:bg-gray-900"
              }`}
            >
              <span className="rounded bg-indigo-900 text-indigo-300 px-1.5 py-0.5 text-[10px] font-medium">GRP</span>
              <span className="font-mono text-gray-200 font-medium">{g.name}</span>
              <span className={`text-[10px] rounded px-1.5 py-0.5 ml-1 ${
                g.logic === "and" ? "bg-sky-900/50 text-sky-300" : "bg-amber-900/50 text-amber-300"
              }`}>{g.logic.toUpperCase()}</span>
              <span className="text-gray-600 text-[10px] ml-1">{g.conditions.length} condition{g.conditions.length !== 1 ? "s" : ""}</span>
              <span className="ml-auto text-gray-600">{expandedIndex === i ? "▲" : "▼"}</span>
            </div>

            {expandedIndex === i && (
              <div className="mt-1 rounded border border-brand-500/40 bg-gray-900 p-3 space-y-3">
                <div className="flex flex-wrap gap-2 items-center">
                  <span className="text-xs text-gray-500">Name</span>
                  <TxtIn value={g.name} onChange={(v) => updateAt(i, { ...g, name: v })} className="w-40" />
                  <span className="text-xs text-gray-500 ml-2">ID</span>
                  <TxtIn value={g.id} onChange={(v) => updateAt(i, { ...g, id: v })} placeholder="group_id" className="w-28" />
                  <div className="flex items-center gap-1 ml-auto">
                    <span className="text-xs text-gray-500">Logic:</span>
                    <Sel value={g.logic} onChange={(v) => updateAt(i, { ...g, logic: v as "and" | "or" })}
                      options={[{ value: "and", label: "AND (all)" }, { value: "or", label: "OR (any)" }]} />
                  </div>
                  <button onClick={() => removeAt(i)} className="text-gray-600 hover:text-red-400 text-xs ml-2">Remove</button>
                </div>
                <div className="space-y-2">
                  {g.conditions.map((c, ci) => (
                    <ConditionRow key={ci} cond={c} columnMeta={columnMeta} groups={[]}
                      onChange={(updated) => {
                        const next = [...g.conditions]; next[ci] = updated;
                        updateAt(i, { ...g, conditions: next });
                      }}
                      onRemove={() => updateAt(i, { ...g, conditions: g.conditions.filter((_, j) => j !== ci) })}
                    />
                  ))}
                </div>
                <button type="button"
                  onClick={() => updateAt(i, { ...g, conditions: [...g.conditions, defaultComparison()] })}
                  className="text-xs text-gray-400 hover:text-white">
                  + Add condition
                </button>
              </div>
            )}
          </div>
        ))}
      </div>
    </section>
  );
}

function DirectionalBlockEditor({
  label, block, columnMeta, groups, onChange,
}: {
  label: string; block: DirectionalBlock; columnMeta: ColMeta[]; groups: GroupDef[];
  onChange: (b: DirectionalBlock) => void;
}) {
  return (
    <div className={`rounded border p-3 space-y-3 ${block.enabled ? "border-gray-700" : "border-gray-800 opacity-60"}`}>
      <div className="flex items-center gap-3">
        <label className="flex items-center gap-2 cursor-pointer">
          <input type="checkbox" checked={block.enabled}
            onChange={(e) => onChange({ ...block, enabled: e.target.checked })}
            className="accent-sky-500" />
          <span className="text-xs font-medium text-gray-300">{label}</span>
        </label>
        {block.enabled && (
          <div className="flex items-center gap-1 ml-auto">
            <span className="text-xs text-gray-500">Logic:</span>
            <Sel value={block.logic} onChange={(v) => onChange({ ...block, logic: v as "and" | "or" })}
              options={[{ value: "and", label: "AND (all)" }, { value: "or", label: "OR (any)" }]} />
          </div>
        )}
      </div>

      {block.enabled && (
        <>
          <div className="space-y-2">
            {block.conditions.map((c, i) => (
              <ConditionRow key={i} cond={c} columnMeta={columnMeta} groups={groups}
                onChange={(updated) => {
                  const next = [...block.conditions]; next[i] = updated;
                  onChange({ ...block, conditions: next });
                }}
                onRemove={() => onChange({ ...block, conditions: block.conditions.filter((_, j) => j !== i) })}
              />
            ))}
          </div>
          <button type="button"
            onClick={() => onChange({ ...block, conditions: [...block.conditions, defaultComparison()] })}
            className="text-xs text-gray-400 hover:text-white">
            + Add condition
          </button>
        </>
      )}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Read-only definition view
// ---------------------------------------------------------------------------

function conditionLabel(c: Condition): string {
  const t = condType(c);
  if (t === "ml") {
    const m = c as MlCondition;
    return `ML Model #${m.model_id} → ${m.direction.toUpperCase()} (conf ≥ ${(m.min_confidence * 100).toFixed(0)}%)`;
  }
  if (t === "llm") {
    const l = c as LlmCondition;
    return `LLM (${l.model}) → ${l.direction.toUpperCase()}, lookback ${l.lookback}`;
  }
  if (t === "streak") {
    const s = c as StreakCondition;
    return `${s.left} ${s.op} ${s.right}  ≥ ${s.min_streak} bars`;
  }
  if (t === "group_ref") {
    const g = c as GroupRefCondition;
    return `→ group: ${g.group_id}`;
  }
  const cmp = c as ComparisonCondition;
  return `${cmp.left} ${cmp.op} ${cmp.right}`;
}

function conditionLabelWithGroups(c: Condition, groups: GroupDef[]): string {
  if (condType(c) === "group_ref") {
    const g = c as GroupRefCondition;
    const found = groups.find((grp) => grp.id === g.group_id);
    return `→ ${found ? found.name : g.group_id}`;
  }
  return conditionLabel(c);
}

function ViewConditionList({ block, label, color, groups }: { block: DirectionalBlock; label: string; color: string; groups: GroupDef[] }) {
  if (!block.enabled) return null;
  return (
    <div>
      <div className="flex items-center gap-2 mb-1">
        <span className={`text-[10px] font-medium uppercase tracking-wide ${color}`}>{label}</span>
        <span className={`text-[10px] rounded px-1.5 py-0.5 font-mono ${
          block.logic === "and" ? "bg-sky-900/50 text-sky-300" : "bg-amber-900/50 text-amber-300"
        }`}>
          {block.logic.toUpperCase()}
        </span>
      </div>
      <div className="space-y-1 pl-2 border-l border-gray-800">
        {block.conditions.map((c, i) => (
          <p key={i} className={`text-xs font-mono ${condType(c) === "group_ref" ? "text-indigo-400" : "text-gray-300"}`}>
            {conditionLabelWithGroups(c, groups)}
          </p>
        ))}
      </div>
    </div>
  );
}

export function StrategyDefinitionView({ definition }: { definition: any }) {
  const form = parseDefinition(definition ?? {});
  const hasLong  = form.long_entry.enabled  || form.long_exit.enabled;
  const hasShort = form.short_entry.enabled || form.short_exit.enabled;

  const INDICATOR_COLOR: Record<IndicatorCategory, string> = {
    technical: "bg-sky-900 text-sky-300",
    ml:        "bg-purple-900 text-purple-300",
    economic:  "bg-amber-900 text-amber-300",
  };
  const INDICATOR_LABEL: Record<IndicatorCategory, string> = {
    technical: "TA",
    ml:        "ML",
    economic:  "ECON",
  };

  return (
    <div className="rounded border border-gray-800 bg-gray-900/60 p-4 space-y-4">
      {/* Symbol */}
      <div className="flex items-center gap-3">
        <span className="text-xs text-gray-500 uppercase tracking-wide">Symbol</span>
        <span className="font-mono font-semibold text-white">{form.symbol}</span>
      </div>

      {/* Indicators */}
      {form.indicators.length > 0 && (
        <div>
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-500">Indicators</p>
          <div className="flex flex-wrap gap-2">
            {form.indicators.map((ind, i) => (
              <div key={i} className="flex items-center gap-1.5 rounded border border-gray-700 bg-gray-800 px-2 py-1">
                <span className={`text-[10px] font-medium rounded px-1 py-0.5 ${INDICATOR_COLOR[ind.category]}`}>
                  {INDICATOR_LABEL[ind.category]}
                </span>
                <span className="font-mono text-xs text-gray-200">{ind.id}</span>
                <span className="text-[10px] text-gray-500">=</span>
                <span className="text-[10px] text-gray-400">{indicatorSummary(ind)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Condition Groups */}
      {form.groups.length > 0 && (
        <div>
          <p className="mb-2 text-[10px] font-medium uppercase tracking-wide text-gray-500">Condition Groups</p>
          <div className="space-y-2">
            {form.groups.map((g) => (
              <div key={g.id} className="rounded border border-indigo-900/50 bg-gray-800/50 px-3 py-2">
                <div className="flex items-center gap-2 mb-1">
                  <span className="rounded bg-indigo-900 text-indigo-300 px-1.5 py-0.5 text-[10px] font-medium">GRP</span>
                  <span className="text-xs font-mono text-gray-200 font-medium">{g.name}</span>
                  <span className={`text-[10px] rounded px-1.5 py-0.5 ${
                    g.logic === "and" ? "bg-sky-900/50 text-sky-300" : "bg-amber-900/50 text-amber-300"
                  }`}>{g.logic.toUpperCase()}</span>
                </div>
                <div className="space-y-0.5 pl-2 border-l border-indigo-900/40">
                  {g.conditions.map((c, i) => (
                    <p key={i} className="text-xs font-mono text-gray-400">{conditionLabel(c)}</p>
                  ))}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Long */}
      {hasLong && (
        <div className="space-y-3">
          <p className="text-[10px] font-medium uppercase tracking-wide text-green-500">Long (buy)</p>
          <ViewConditionList block={form.long_entry} label="Entry" color="text-gray-400" groups={form.groups} />
          <ViewConditionList block={form.long_exit}  label="Exit"  color="text-gray-400" groups={form.groups} />
        </div>
      )}

      {/* Short */}
      {hasShort && (
        <div className="space-y-3">
          <p className="text-[10px] font-medium uppercase tracking-wide text-red-400">Short (sell)</p>
          <ViewConditionList block={form.short_entry} label="Entry" color="text-gray-400" groups={form.groups} />
          <ViewConditionList block={form.short_exit}  label="Exit"  color="text-gray-400" groups={form.groups} />
        </div>
      )}

    </div>
  );
}

// ---------------------------------------------------------------------------
// Main component
// ---------------------------------------------------------------------------

export interface StrategyEditorProps {
  initialDefinition?: any;
  onChange: (def: object) => void;
  strategyId?: string | number;
  datasets?: { id: number; name: string; row_count: number }[];
}

export function StrategyEditor({ initialDefinition, onChange, strategyId, datasets }: StrategyEditorProps) {
  const [mode, setMode] = useState<"visual" | "json">("visual");
  const [form, setForm] = useState<FormState>(() => parseDefinition(initialDefinition ?? {}));
  const [jsonText, setJsonText] = useState<string>(() =>
    JSON.stringify(serialiseForm(parseDefinition(initialDefinition ?? {})), null, 2)
  );
  const [jsonError, setJsonError] = useState<string | null>(null);

  // Propagate visual changes upward
  useEffect(() => {
    if (mode === "visual") onChange(serialiseForm(form));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [form, mode]);

  function switchToJson() {
    setJsonText(JSON.stringify(serialiseForm(form), null, 2));
    setJsonError(null);
    setMode("json");
  }

  function switchToVisual() {
    try {
      const parsed = JSON.parse(jsonText);
      setForm(parseDefinition(parsed));
      setJsonError(null);
      setMode("visual");
    } catch {
      setJsonError("Fix JSON errors before switching to visual mode");
    }
  }

  function handleJsonChange(text: string) {
    setJsonText(text);
    setJsonError(null);
    try { onChange(JSON.parse(text)); } catch { /* keep last valid */ }
  }

  function patch(p: Partial<FormState>) { setForm((prev) => ({ ...prev, ...p })); }

  const columnMeta = deriveColumnMeta(form.indicators);
  const columns = columnMeta.map((m) => m.name);

  const [validateDatasetId, setValidateDatasetId] = useState("");
  const [validateLimitBars, setValidateLimitBars] = useState("3000");
  const [disabledConds, setDisabledConds] = useState<Record<string, number[]>>({});
  const [validating, setValidating] = useState(false);
  const [validateCandles, setValidateCandles] = useState<any[]>([]);
  const [validateIndicators, setValidateIndicators] = useState<any>({});
  const [validateMarkers, setValidateMarkers] = useState<any[]>([]);
  const [validateCounts, setValidateCounts] = useState<Record<string, number>>({});
  const [validateError, setValidateError] = useState<string | null>(null);
  const validateAbortRef = useRef<AbortController | null>(null);
  const validateCandlesRef = useRef(validateCandles);
  validateCandlesRef.current = validateCandles;

  // Auto-rerun (soft) when condition toggles change, but only after first successful run
  useEffect(() => {
    if (!strategyId || !validateDatasetId || validateCandlesRef.current.length === 0) return;
    const t = setTimeout(() => runValidate(true), 350);
    return () => clearTimeout(t);
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [disabledConds]);

  function clearValidate(soft = false) {
    if (!soft) { setValidateCandles([]); setValidateIndicators({}); }
    setValidateMarkers([]); setValidateCounts({}); setValidateError(null);
  }

  function signalsToMarkers(signals: Record<string, number[]>): any[] {
    const defs: Array<{ key: string; color: string; position: "aboveBar" | "belowBar"; label: string }> = [
      { key: "long_entry",  color: "#22c55e", position: "belowBar", label: "Long entry" },
      { key: "long_exit",   color: "#38bdf8", position: "aboveBar", label: "Long exit" },
      { key: "short_entry", color: "#ef4444", position: "aboveBar", label: "Short entry" },
      { key: "short_exit",  color: "#f97316", position: "belowBar", label: "Short exit" },
    ];
    const markers: any[] = [];
    for (const { key, color, position, label } of defs) {
      for (const ts of (signals[key] ?? [])) {
        markers.push({ time: ts, position, color, shape: "circle", text: label });
      }
    }
    markers.sort((a, b) => a.time - b.time);
    return markers;
  }

  function toggleCond(blockKey: string, idx: number) {
    setDisabledConds((prev) => {
      const arr = prev[blockKey] ?? [];
      const next = arr.includes(idx) ? arr.filter((i) => i !== idx) : [...arr, idx];
      return { ...prev, [blockKey]: next };
    });
  }

  function buildFilteredDef(): Record<string, any> {
    const def = serialiseForm(form) as any;
    const blockMap: Record<string, [string, string]> = {
      long_entry: ["long", "entry"], long_exit: ["long", "exit"],
      short_entry: ["short", "entry"], short_exit: ["short", "exit"],
    };
    const out = JSON.parse(JSON.stringify(def));
    for (const [key, [dir, phase]] of Object.entries(blockMap)) {
      const disabled = disabledConds[key] ?? [];
      if (disabled.length > 0 && out[dir]?.[phase]?.conditions) {
        out[dir][phase].conditions = out[dir][phase].conditions.filter((_: any, i: number) => !disabled.includes(i));
      }
    }
    return out;
  }

  async function runValidate(soft = false) {
    if (!strategyId || !validateDatasetId) return;
    validateAbortRef.current?.abort();
    setValidating(true);
    clearValidate(soft);
    const abort = new AbortController();
    validateAbortRef.current = abort;
    try {
      const limitBars = validateLimitBars === "all" ? undefined : parseInt(validateLimitBars);
      const res = await fetch(`/api/v1/strategies/${strategyId}/validate/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          dataset_id: parseInt(validateDatasetId),
          definition: buildFilteredDef(),
          ...(limitBars ? { limit_bars: limitBars } : {}),
        }),
        signal: abort.signal,
      });
      if (!res.ok || !res.body) {
        const errBody = await res.json().catch(() => ({}));
        setValidateError(errBody.detail ?? `Error ${res.status}`);
        return;
      }
      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buf = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buf += decoder.decode(value, { stream: true });
        const lines = buf.split("\n");
        buf = lines.pop() ?? "";
        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (!raw) continue;
          let evt: any;
          try { evt = JSON.parse(raw); } catch { continue; }
          if (evt.type === "init") {
            setValidateCandles(evt.candles ?? []);
            setValidateIndicators(evt.indicators ?? {});
          } else if (evt.type === "signals") {
            const markers = signalsToMarkers(evt);
            setValidateMarkers(markers);
            setValidateCounts({
              long_entry: (evt.long_entry ?? []).length,
              long_exit: (evt.long_exit ?? []).length,
              short_entry: (evt.short_entry ?? []).length,
              short_exit: (evt.short_exit ?? []).length,
            });
          } else if (evt.type === "error") {
            setValidateError(evt.message ?? "Unknown error");
            break;
          }
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") setValidateError(e.message ?? "Unknown error");
    } finally {
      setValidating(false);
      validateAbortRef.current = null;
    }
  }

  return (
    <div className="space-y-5">
      {/* Mode toggle */}
      <div className="flex gap-2 items-center">
        <span className="text-xs text-gray-500">Edit as:</span>
        <button type="button" onClick={() => mode === "json" ? switchToVisual() : undefined}
          className={`rounded px-2 py-1 text-xs ${mode === "visual" ? "bg-brand-500 text-white" : "text-gray-400 hover:text-white"}`}>
          Visual
        </button>
        <button type="button" onClick={() => mode === "visual" ? switchToJson() : undefined}
          className={`rounded px-2 py-1 text-xs ${mode === "json" ? "bg-brand-500 text-white" : "text-gray-400 hover:text-white"}`}>
          JSON
        </button>
      </div>

      {mode === "json" && (
        <div className="space-y-2">
          <textarea value={jsonText} onChange={(e) => handleJsonChange(e.target.value)} rows={20}
            className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 font-mono text-xs text-white focus:outline-none focus:ring-1 focus:ring-brand-500" />
          {jsonError && <p className="text-xs text-red-400">{jsonError}</p>}
        </div>
      )}

      {mode === "visual" && (
        <div className="space-y-6">

          {/* Indicators */}
          <IndicatorsSection
            indicators={form.indicators}
            onChange={(inds) => patch({ indicators: inds })}
          />

          {/* Condition Groups */}
          <ConditionGroupsSection
            groups={form.groups}
            columnMeta={columnMeta}
            onChange={(grps) => patch({ groups: grps })}
          />

          {/* Long / Short entry + exit */}
          <section className="space-y-3">
            <h3 className="text-xs font-medium uppercase tracking-wide text-gray-400">Long (buy)</h3>
            <DirectionalBlockEditor label="Entry conditions" block={form.long_entry} columnMeta={columnMeta} groups={form.groups}
              onChange={(b) => patch({ long_entry: b, ...(!form.long_exit.enabled && b.enabled ? { long_exit: { ...form.long_exit, enabled: true } } : {}) })} />
            <DirectionalBlockEditor label="Exit conditions" block={form.long_exit} columnMeta={columnMeta} groups={form.groups}
              onChange={(b) => patch({ long_exit: b })} />
          </section>

          <section className="space-y-3">
            <h3 className="text-xs font-medium uppercase tracking-wide text-gray-400">Short (sell)</h3>
            <DirectionalBlockEditor label="Entry conditions" block={form.short_entry} columnMeta={columnMeta} groups={form.groups}
              onChange={(b) => patch({ short_entry: b, ...(!form.short_exit.enabled && b.enabled ? { short_exit: { ...form.short_exit, enabled: true } } : {}) })} />
            <DirectionalBlockEditor label="Exit conditions" block={form.short_exit} columnMeta={columnMeta} groups={form.groups}
              onChange={(b) => patch({ short_exit: b })} />
          </section>

          {/* Validate — only shown when strategyId + datasets are provided */}
          {strategyId && datasets && (
            <section className="space-y-3 border-t border-gray-800 pt-4">
              <h3 className="text-xs font-medium uppercase tracking-wide text-gray-400">Validate conditions</h3>

              {/* Condition toggles */}
              {(() => {
                const blocks = [
                  { key: "long_entry",  label: "Long entry",  color: "#22c55e", conds: form.long_entry.enabled  ? form.long_entry.conditions  : [] },
                  { key: "long_exit",   label: "Long exit",   color: "#38bdf8", conds: form.long_exit.enabled   ? form.long_exit.conditions   : [] },
                  { key: "short_entry", label: "Short entry", color: "#ef4444", conds: form.short_entry.enabled ? form.short_entry.conditions : [] },
                  { key: "short_exit",  label: "Short exit",  color: "#f97316", conds: form.short_exit.enabled  ? form.short_exit.conditions  : [] },
                ].filter((b) => b.conds.length > 0);
                if (blocks.length === 0) return null;
                return (
                  <div className="rounded border border-gray-700 bg-gray-900 p-3 space-y-2">
                    {blocks.map(({ key, label, color, conds }) => (
                      <div key={key}>
                        <p className="text-xs font-medium mb-1" style={{ color }}>{label}</p>
                        <div className="space-y-0.5 pl-2">
                          {conds.map((c, i) => {
                            const disabled = (disabledConds[key] ?? []).includes(i);
                            return (
                              <label key={i} className="flex items-center gap-2 text-xs cursor-pointer select-none">
                                <input type="checkbox" checked={!disabled}
                                  onChange={() => toggleCond(key, i)}
                                  className="accent-sky-500 shrink-0" />
                                <span className={disabled ? "text-gray-600 line-through" : "text-gray-300"}>
                                  {conditionLabel(c)}
                                </span>
                              </label>
                            );
                          })}
                        </div>
                      </div>
                    ))}
                  </div>
                );
              })()}

              <div className="flex items-center gap-2 flex-wrap">
                <select
                  value={validateDatasetId}
                  onChange={(e) => { setValidateDatasetId(e.target.value); clearValidate(); }}
                  className="rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                >
                  <option value="">Select dataset…</option>
                  {datasets.map((d) => (
                    <option key={d.id} value={d.id}>{d.name} ({d.row_count?.toLocaleString()} rows)</option>
                  ))}
                </select>
                <select
                  value={validateLimitBars}
                  onChange={(e) => setValidateLimitBars(e.target.value)}
                  className="rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
                >
                  <option value="1000">Last 1 000 bars</option>
                  <option value="3000">Last 3 000 bars</option>
                  <option value="10000">Last 10 000 bars</option>
                  <option value="all">All bars</option>
                </select>
                <button type="button" onClick={() => runValidate()} disabled={!validateDatasetId || validating}
                  className="rounded bg-sky-700 px-3 py-1.5 text-xs text-white hover:bg-sky-600 disabled:opacity-50">
                  {validating ? "Loading…" : "Run"}
                </button>
              </div>
              {validating && <div className="h-0.5 w-full rounded bg-gray-700 overflow-hidden"><div className="h-full bg-sky-500 animate-pulse w-full" /></div>}
              {validateError && <p className="text-xs text-red-400">{validateError}</p>}
              {Object.keys(validateCounts).length > 0 && (
                <p className="text-xs text-gray-400 flex gap-3 flex-wrap">
                  {validateCounts.long_entry > 0 && <span><span className="inline-block w-2 h-2 rounded-full bg-green-500 mr-1" />Long entry: {validateCounts.long_entry}</span>}
                  {validateCounts.long_exit > 0 && <span><span className="inline-block w-2 h-2 rounded-full bg-sky-400 mr-1" />Long exit: {validateCounts.long_exit}</span>}
                  {validateCounts.short_entry > 0 && <span><span className="inline-block w-2 h-2 rounded-full bg-red-500 mr-1" />Short entry: {validateCounts.short_entry}</span>}
                  {validateCounts.short_exit > 0 && <span><span className="inline-block w-2 h-2 rounded-full bg-orange-500 mr-1" />Short exit: {validateCounts.short_exit}</span>}
                </p>
              )}
              {validateCandles.length > 0 && (
                <StrategyChartLazy
                  candles={validateCandles}
                  indicators={validateIndicators}
                  markers={validateMarkers}
                  defaultZoom={1}
                />
              )}
            </section>
          )}

        </div>
      )}
    </div>
  );
}
