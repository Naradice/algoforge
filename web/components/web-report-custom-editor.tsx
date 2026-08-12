"use client";

import { useState } from "react";
import {
  WebReportCustomStep,
  WebReportTarget,
  WEB_REPORT_FETCH_TYPES,
  WEB_REPORT_UNIQUE_TYPES,
  newWebReportStep,
  newWebReportTarget,
} from "@/lib/datasource-types";

// Structured editor for the web_report "custom" config array (link_parse / element_parse steps,
// each with one or more regex-matched targets). Mirrors the shape backend/data/collectors/web_report.py
// expects — see docs/data-layer.md for the full field semantics. Falls back to a raw JSON textarea
// for cases the visual form doesn't cover (nested `custom` recursion, hand-tuned edge cases).
export function CustomStepsEditor({
  steps,
  onChange,
  defaultExt = "pdf",
  defaultType = "load",
}: {
  steps: WebReportCustomStep[];
  onChange: (steps: WebReportCustomStep[]) => void;
  defaultExt?: string;
  defaultType?: string;
}) {
  const [jsonMode, setJsonMode] = useState(false);
  const [jsonText, setJsonText] = useState("");
  const [jsonError, setJsonError] = useState<string | null>(null);

  function enterJsonMode() {
    setJsonText(JSON.stringify(steps, null, 2));
    setJsonError(null);
    setJsonMode(true);
  }

  function applyJson() {
    try {
      const parsed = JSON.parse(jsonText);
      if (!Array.isArray(parsed)) throw new Error("Must be a JSON array of steps");
      onChange(parsed);
      setJsonError(null);
      setJsonMode(false);
    } catch (e: any) {
      setJsonError(e.message ?? "Invalid JSON");
    }
  }

  function updateStep(i: number, patch: Partial<WebReportCustomStep>) {
    onChange(steps.map((s, idx) => (idx === i ? { ...s, ...patch } : s)));
  }

  function removeStep(i: number) {
    onChange(steps.filter((_, idx) => idx !== i));
  }

  function addStep() {
    onChange([...steps, newWebReportStep({ ext: defaultExt, type: defaultType })]);
  }

  function updateTarget(stepIdx: number, targetIdx: number, patch: Partial<WebReportTarget>) {
    onChange(
      steps.map((s, si) =>
        si !== stepIdx ? s : { ...s, targets: s.targets.map((t, ti) => (ti === targetIdx ? { ...t, ...patch } : t)) }
      )
    );
  }

  function removeTarget(stepIdx: number, targetIdx: number) {
    onChange(
      steps.map((s, si) => (si !== stepIdx ? s : { ...s, targets: s.targets.filter((_, ti) => ti !== targetIdx) }))
    );
  }

  function addTarget(stepIdx: number) {
    onChange(
      steps.map((s, si) =>
        si !== stepIdx ? s : { ...s, targets: [...s.targets, newWebReportTarget({ ext: defaultExt, type: defaultType })] }
      )
    );
  }

  if (jsonMode) {
    return (
      <div className="space-y-2">
        <textarea
          className="md-input w-full h-64 font-mono text-xs"
          value={jsonText}
          onChange={(e) => setJsonText(e.target.value)}
          spellCheck={false}
        />
        {jsonError && <p className="text-xs text-red-400">{jsonError}</p>}
        <div className="flex gap-2">
          <button type="button" onClick={applyJson} className="rounded bg-brand-500 px-3 py-1 text-xs text-white hover:bg-sky-400">
            Apply
          </button>
          <button type="button" onClick={() => setJsonMode(false)} className="rounded bg-gray-700 px-3 py-1 text-xs text-white hover:bg-gray-600">
            Cancel
          </button>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-3">
      {steps.length === 0 && (
        <p className="text-xs text-gray-500">No custom targets configured. Add a step to match specific links on the page individually.</p>
      )}

      {steps.map((step, si) => (
        <div key={si} className="rounded border border-gray-700 bg-gray-800/40 p-3 space-y-3">
          <div className="flex items-center justify-between gap-2">
            <select
              className="md-input text-xs"
              value={step.type}
              onChange={(e) => updateStep(si, { type: e.target.value as WebReportCustomStep["type"] })}
            >
              <option value="link_parse">link_parse — scan &lt;a&gt; links</option>
              <option value="element_parse">element_parse — CSS selector</option>
            </select>
            <button type="button" onClick={() => removeStep(si)} className="text-xs text-red-400 hover:text-red-300 shrink-0">
              Remove step
            </button>
          </div>

          {step.type === "element_parse" && (
            <div className="space-y-1">
              <label className="text-xs text-gray-400 uppercase">CSS Selector</label>
              <input
                className="md-input w-full font-mono text-xs"
                value={step.selector ?? ""}
                onChange={(e) => updateStep(si, { selector: e.target.value })}
                placeholder="a.report-link"
              />
            </div>
          )}

          <div className="space-y-2">
            {step.targets.map((target, ti) => (
              <div key={ti} className="rounded border border-gray-700 p-2 space-y-2">
                <div className="flex items-center justify-between">
                  <span className="text-xs text-gray-500">Target {ti + 1}</span>
                  <button type="button" onClick={() => removeTarget(si, ti)} className="text-xs text-red-400 hover:text-red-300">
                    Remove
                  </button>
                </div>
                <div className="grid grid-cols-2 gap-2">
                  <MiniField label="Match regex (value)" hint="Tested against the full link href">
                    <input
                      className="md-input w-full font-mono text-xs"
                      value={target.value}
                      onChange={(e) => updateTarget(si, ti, { value: e.target.value })}
                      placeholder="/pdf/w_.*pdf"
                    />
                  </MiniField>
                  <MiniField label="Filename template" hint="{YYYYMMDD} {YYMMDD} {filename} {basefilename}">
                    <input
                      className="md-input w-full font-mono text-xs"
                      value={target.filename}
                      onChange={(e) => updateTarget(si, ti, { filename: e.target.value })}
                      placeholder="{filename}"
                    />
                  </MiniField>
                  <MiniField label="File type (ext)">
                    <select className="md-input w-full text-xs" value={target.ext} onChange={(e) => updateTarget(si, ti, { ext: e.target.value })}>
                      {["pdf", "html", "mp3", "txt"].map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </MiniField>
                  <MiniField label="Fetch method">
                    <select className="md-input w-full text-xs" value={target.type} onChange={(e) => updateTarget(si, ti, { type: e.target.value })}>
                      {WEB_REPORT_FETCH_TYPES.map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </MiniField>
                  <MiniField label="Deduplication">
                    <select className="md-input w-full text-xs" value={target.unique} onChange={(e) => updateTarget(si, ti, { unique: e.target.value })}>
                      {WEB_REPORT_UNIQUE_TYPES.map((o) => <option key={o} value={o}>{o}</option>)}
                    </select>
                  </MiniField>
                  <MiniField label="Interval (days)" hint="Blank = download once only">
                    <input
                      type="number"
                      min="1"
                      className="md-input w-full text-xs"
                      value={target.interval_days ?? ""}
                      onChange={(e) =>
                        updateTarget(si, ti, {
                          interval_days: e.target.value === "" ? null : Math.max(1, Number(e.target.value) || 1),
                        })
                      }
                      placeholder="1"
                    />
                  </MiniField>
                </div>
              </div>
            ))}
            <button type="button" onClick={() => addTarget(si)} className="text-xs text-brand-400 hover:text-brand-300">
              + Add target
            </button>
          </div>
        </div>
      ))}

      <div className="flex items-center justify-between">
        <button type="button" onClick={addStep} className="rounded border border-gray-700 px-3 py-1 text-xs text-gray-300 hover:border-gray-400 hover:text-white">
          + Add step
        </button>
        <button type="button" onClick={enterJsonMode} className="text-xs text-gray-500 hover:text-gray-300">
          Edit as JSON
        </button>
      </div>
    </div>
  );
}

function MiniField({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-0.5">
      <label className="text-[10px] text-gray-500 uppercase">{label}</label>
      {children}
      {hint && <p className="text-[10px] text-gray-600">{hint}</p>}
    </div>
  );
}
