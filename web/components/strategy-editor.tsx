"use client";

import { useState, useEffect } from "react";

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
type Condition = ComparisonCondition | MlCondition | LlmCondition;

type DirectionalBlock = {
  enabled: boolean;
  conditions: Condition[];
  logic: "and" | "or";
};

type RiskDef = {
  risk_type: "fixed" | "percent_equity" | "atr";
  position_size: number;
  risk_pct: number;
  atr_multiplier: number;
  sl_pct: number;
  tp_pct: number;
  slippage_pct: number;
  commission_pct: number;
  max_positions: number;
  daily_loss_limit_pct: number;
  cooldown_bars: number;
};

type FormState = {
  symbol: string;
  indicators: IndicatorDef[];
  long_entry: DirectionalBlock;
  long_exit: DirectionalBlock;
  short_entry: DirectionalBlock;
  short_exit: DirectionalBlock;
  risk: RiskDef;
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
  technical: ["macd", "rsi", "atr", "ema", "sma", "bb", "slope"],
  ml:        ["ml_inference"],
  economic:  ["macro_series"],
};

type ParamSpec = { key: string; label: string; inputType: "number" | "text"; default: number | string };

const INDICATOR_PARAMS: Record<string, ParamSpec[]> = {
  macd:         [{ key: "fast", label: "Fast", inputType: "number", default: 12 }, { key: "slow", label: "Slow", inputType: "number", default: 26 }, { key: "signal_period", label: "Signal", inputType: "number", default: 9 }],
  rsi:          [{ key: "period", label: "Period", inputType: "number", default: 14 }],
  atr:          [{ key: "period", label: "Period", inputType: "number", default: 14 }],
  ema:          [{ key: "period", label: "Period", inputType: "number", default: 20 }],
  sma:          [{ key: "period", label: "Period", inputType: "number", default: 20 }],
  bb:           [{ key: "period", label: "Period", inputType: "number", default: 20 }, { key: "std_dev", label: "Std Dev", inputType: "number", default: 2 }],
  slope:        [{ key: "period", label: "Period", inputType: "number", default: 5 }, { key: "column", label: "Column", inputType: "text", default: "close" }],
  ml_inference: [{ key: "model_id", label: "Model ID", inputType: "number", default: 1 }, { key: "output_col", label: "Output column", inputType: "text", default: "ml_score" }],
  macro_series: [{ key: "source", label: "Source", inputType: "text", default: "fred" }, { key: "series", label: "Series ID", inputType: "text", default: "CPIAUCSL" }, { key: "lag", label: "Lag bars", inputType: "number", default: 0 }],
};

function defaultParams(type: string): Record<string, string | number> {
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

/** Derive all available column names from indicators */
function deriveColumns(indicators: IndicatorDef[]): string[] {
  const base = ["open", "high", "low", "close", "volume"];
  const derived: string[] = [];
  for (const ind of indicators) {
    switch (ind.type) {
      case "macd": derived.push(`${ind.id}_line`, `${ind.id}_signal`, `${ind.id}_hist`); break;
      case "bb":   derived.push(`${ind.id}_upper`, `${ind.id}_middle`, `${ind.id}_lower`); break;
      case "ml_inference": derived.push(String(ind.params.output_col ?? ind.id)); break;
      default:     derived.push(ind.id);
    }
  }
  return [...base, ...derived];
}

/** Human-readable summary line for a collapsed indicator row */
function indicatorSummary(ind: IndicatorDef): string {
  const pairs = Object.entries(ind.params).map(([k, v]) => `${k}=${v}`).join(", ");
  return `${ind.type}${pairs ? ` (${pairs})` : ""}`;
}

// ---------------------------------------------------------------------------
// Condition helpers
// ---------------------------------------------------------------------------

type CondType = "comparison" | "ml" | "llm";

function condType(c: Condition): CondType {
  const t = (c as any).type;
  if (t === "ml_signal")  return "ml";
  if (t === "llm_signal") return "llm";
  return "comparison";
}

const defaultComparison = (): ComparisonCondition => ({ left: "close", op: ">", right: "open" });
const defaultMl = (): MlCondition => ({ type: "ml_signal", model_id: 1, direction: "buy", step: 1, min_confidence: 0.0 });
const defaultLlm = (): LlmCondition => ({ type: "llm_signal", direction: "buy", lookback: 10, model: "gemini-2.0-flash", columns: ["close", "volume"], cache: true });
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
    long_entry,
    long_exit,
    short_entry,
    short_exit,
    risk: {
      risk_type:            (def.risk?.risk_type         ?? "fixed") as RiskDef["risk_type"],
      position_size:        def.risk?.position_size        ?? 1.0,
      risk_pct:             def.risk?.risk_pct             ?? 0.01,
      atr_multiplier:       def.risk?.atr_multiplier       ?? 2.0,
      sl_pct:               def.risk?.sl_pct               ?? 0.02,
      tp_pct:               def.risk?.tp_pct               ?? 0.04,
      slippage_pct:         def.risk?.slippage_pct         ?? 0.0005,
      commission_pct:       def.risk?.commission_pct       ?? 0.001,
      max_positions:        def.risk?.max_positions        ?? 1,
      daily_loss_limit_pct: def.risk?.daily_loss_limit_pct ?? 0.0,
      cooldown_bars:        def.risk?.cooldown_bars        ?? 0,
    },
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

  return {
    symbol: f.symbol,
    indicators: f.indicators.map((i) => ({ id: i.id, category: i.category, type: i.type, params: i.params })),
    ...(Object.keys(long).length  ? { long }  : {}),
    ...(Object.keys(short).length ? { short } : {}),
    risk: f.risk,
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

function NumIn({ value, onChange, step, min, className }: { value: number; onChange: (v: number) => void; step?: number; min?: number; className?: string }) {
  return (
    <input type="number" value={value} step={step ?? 1} min={min} onChange={(e) => onChange(parseFloat(e.target.value) || 0)}
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

      {/* Params */}
      {paramSpecs.length > 0 && (
        <div className="flex flex-wrap gap-3">
          {paramSpecs.map((spec) => (
            <label key={spec.key} className="flex items-center gap-1">
              <span className="text-xs text-gray-400">{spec.label}</span>
              {spec.inputType === "number" ? (
                <NumIn value={Number(ind.params[spec.key] ?? spec.default)}
                  onChange={(v) => onChange({ ...ind, params: { ...ind.params, [spec.key]: v } })}
                  className="w-16" />
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
            {/* Collapsed row */}
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

            {/* Expanded editor */}
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
    </section>
  );
}

// ---------------------------------------------------------------------------
// Condition section
// ---------------------------------------------------------------------------

const OPS = [">", "<", ">=", "<=", "==", "!="];
const LLM_MODELS = ["gemini-2.0-flash", "gemini-1.5-pro", "claude-sonnet-4-6", "claude-haiku-4-5-20251001"];

function ConditionRow({
  cond, columns, onChange, onRemove,
}: {
  cond: Condition; columns: string[];
  onChange: (c: Condition) => void; onRemove: () => void;
}) {
  const t = condType(cond);
  const colOpts = columns.map((c) => ({ value: c, label: c }));

  return (
    <div className="rounded border border-gray-700 bg-gray-900 p-2 space-y-2">
      <div className="flex items-center gap-2 flex-wrap">
        <Sel value={t} onChange={(newT) => {
          if (newT === "ml")  onChange(defaultMl());
          else if (newT === "llm") onChange(defaultLlm());
          else onChange(defaultComparison());
        }} options={[{ value: "comparison", label: "Compare" }, { value: "ml", label: "ML Signal" }, { value: "llm", label: "LLM Signal" }]} />

        {t === "comparison" && (() => {
          const c = cond as ComparisonCondition;
          const rightIsCol = columns.includes(String(c.right));
          return (
            <>
              <Sel value={c.left} onChange={(v) => onChange({ ...c, left: v })} options={colOpts} />
              <Sel value={c.op}   onChange={(v) => onChange({ ...c, op: v })}   options={OPS.map((o) => ({ value: o, label: o }))} />
              {rightIsCol
                ? <Sel value={String(c.right)} onChange={(v) => onChange({ ...c, right: v })} options={colOpts} />
                : <TxtIn value={String(c.right)} onChange={(v) => onChange({ ...c, right: v })} placeholder="number" className="w-24" />}
              <button className="text-[10px] text-gray-500 hover:text-gray-300 border border-gray-700 rounded px-1"
                onClick={() => onChange({ ...c, right: rightIsCol ? 0 : (columns[0] ?? "close") })}
                title={rightIsCol ? "Switch to numeric" : "Switch to column"}>
                {rightIsCol ? "→ num" : "→ col"}
              </button>
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

        <button onClick={onRemove} className="ml-auto text-gray-600 hover:text-red-400 text-xs">✕</button>
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

function DirectionalBlockEditor({
  label, block, columns, onChange,
}: {
  label: string; block: DirectionalBlock; columns: string[];
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
              <ConditionRow key={i} cond={c} columns={columns}
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
// Main component
// ---------------------------------------------------------------------------

export interface StrategyEditorProps {
  initialDefinition?: any;
  onChange: (def: object) => void;
}

export function StrategyEditor({ initialDefinition, onChange }: StrategyEditorProps) {
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

  const columns = deriveColumns(form.indicators);

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

          {/* Symbol */}
          <section>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-400">Symbol</h3>
            <TxtIn value={form.symbol} onChange={(v) => patch({ symbol: v })}
              placeholder="AAPL, EURUSD=X, BTC-USD" className="w-48" />
          </section>

          {/* Indicators */}
          <IndicatorsSection
            indicators={form.indicators}
            onChange={(inds) => patch({ indicators: inds })}
          />

          {/* Long / Short entry + exit */}
          <section className="space-y-3">
            <h3 className="text-xs font-medium uppercase tracking-wide text-gray-400">Long (buy)</h3>
            <DirectionalBlockEditor label="Entry conditions" block={form.long_entry} columns={columns}
              onChange={(b) => patch({ long_entry: b })} />
            <DirectionalBlockEditor label="Exit conditions" block={form.long_exit} columns={columns}
              onChange={(b) => patch({ long_exit: b })} />
          </section>

          <section className="space-y-3">
            <h3 className="text-xs font-medium uppercase tracking-wide text-gray-400">Short (sell)</h3>
            <DirectionalBlockEditor label="Entry conditions" block={form.short_entry} columns={columns}
              onChange={(b) => patch({ short_entry: b })} />
            <DirectionalBlockEditor label="Exit conditions" block={form.short_exit} columns={columns}
              onChange={(b) => patch({ short_exit: b })} />
          </section>

          {/* Risk */}
          <section>
            <h3 className="mb-2 text-xs font-medium uppercase tracking-wide text-gray-400">Risk</h3>
            <div className="space-y-3">

              {/* Position sizing mode */}
              <div className="flex items-center gap-2">
                <span className="w-32 shrink-0 text-xs text-gray-400">Sizing method</span>
                <Sel value={form.risk.risk_type}
                  onChange={(v) => patch({ risk: { ...form.risk, risk_type: v as RiskDef["risk_type"] } })}
                  options={[
                    { value: "fixed",         label: "Fixed lot" },
                    { value: "percent_equity", label: "% of equity per trade" },
                    { value: "atr",            label: "ATR-based" },
                  ]} />
              </div>

              {form.risk.risk_type === "fixed" && (
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-400">Position size</span>
                  <NumIn value={form.risk.position_size} step={0.1} min={0}
                    onChange={(v) => patch({ risk: { ...form.risk, position_size: v } })} />
                  <span className="text-xs text-gray-500">fraction of equity</span>
                </div>
              )}

              {(form.risk.risk_type === "percent_equity" || form.risk.risk_type === "atr") && (
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-400">Risk per trade</span>
                  <NumIn value={form.risk.risk_pct} step={0.005} min={0}
                    onChange={(v) => patch({ risk: { ...form.risk, risk_pct: v } })} />
                  <span className="text-xs text-gray-500">({(form.risk.risk_pct * 100).toFixed(1)}% of equity)</span>
                </div>
              )}

              {form.risk.risk_type === "atr" && (
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-400">ATR multiplier</span>
                  <NumIn value={form.risk.atr_multiplier} step={0.5} min={0.5}
                    onChange={(v) => patch({ risk: { ...form.risk, atr_multiplier: v } })} />
                  <span className="text-xs text-gray-500">stop = ATR × multiplier</span>
                </div>
              )}

              <div className="border-t border-gray-800 pt-3 space-y-2">
                <p className="text-xs text-gray-500 uppercase tracking-wide">Stop / Target</p>
                {[
                  { label: "Stop-loss",   key: "sl_pct" as const, step: 0.005 },
                  { label: "Take-profit", key: "tp_pct" as const, step: 0.005 },
                ].map(({ label, key, step }) => (
                  <div key={key} className="flex items-center gap-2">
                    <span className="w-32 shrink-0 text-xs text-gray-400">{label}</span>
                    <NumIn value={form.risk[key]} step={step} min={0}
                      onChange={(v) => patch({ risk: { ...form.risk, [key]: v } })} />
                    <span className="text-xs text-gray-500">({(form.risk[key] * 100).toFixed(2)}%)</span>
                  </div>
                ))}
              </div>

              <div className="border-t border-gray-800 pt-3 space-y-2">
                <p className="text-xs text-gray-500 uppercase tracking-wide">Execution costs</p>
                {[
                  { label: "Slippage",   key: "slippage_pct" as const,   step: 0.0001 },
                  { label: "Commission", key: "commission_pct" as const, step: 0.0001 },
                ].map(({ label, key, step }) => (
                  <div key={key} className="flex items-center gap-2">
                    <span className="w-32 shrink-0 text-xs text-gray-400">{label}</span>
                    <NumIn value={form.risk[key]} step={step} min={0}
                      onChange={(v) => patch({ risk: { ...form.risk, [key]: v } })} />
                    <span className="text-xs text-gray-500">({(form.risk[key] * 100).toFixed(3)}%)</span>
                  </div>
                ))}
              </div>

              <div className="border-t border-gray-800 pt-3 space-y-2">
                <p className="text-xs text-gray-500 uppercase tracking-wide">Guards</p>
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-400">Max positions</span>
                  <NumIn value={form.risk.max_positions} step={1} min={1}
                    onChange={(v) => patch({ risk: { ...form.risk, max_positions: Math.round(v) } })} className="w-14" />
                  <span className="text-xs text-gray-500">simultaneous open trades</span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-400">Daily loss limit</span>
                  <NumIn value={form.risk.daily_loss_limit_pct} step={0.005} min={0}
                    onChange={(v) => patch({ risk: { ...form.risk, daily_loss_limit_pct: v } })} />
                  <span className="text-xs text-gray-500">
                    {form.risk.daily_loss_limit_pct === 0 ? "disabled" : `${(form.risk.daily_loss_limit_pct * 100).toFixed(1)}% of equity`}
                  </span>
                </div>
                <div className="flex items-center gap-2">
                  <span className="w-32 shrink-0 text-xs text-gray-400">Cooldown after loss</span>
                  <NumIn value={form.risk.cooldown_bars} step={1} min={0}
                    onChange={(v) => patch({ risk: { ...form.risk, cooldown_bars: Math.round(v) } })} className="w-14" />
                  <span className="text-xs text-gray-500">bars</span>
                </div>
              </div>

            </div>
          </section>
        </div>
      )}
    </div>
  );
}
