"use client";

import { useState, useRef, useEffect } from "react";
import { apiFetch } from "@/lib/fetcher";

type EventItem =
  | { type: "start"; content: string }
  | { type: "thinking"; content: string }
  | { type: "tool_call"; tool: string }
  | { type: "tool_result"; tool: string; content: string }
  | { type: "done"; content: string }
  | { type: "error"; content: string };

interface Props {
  strategyId: string;
  datasets: { id: number; name: string; row_count: number }[];
}

export function InvestigationPanel({ strategyId, datasets }: Props) {
  const [datasetId, setDatasetId] = useState("");
  const [running, setRunning] = useState(false);
  const [events, setEvents] = useState<EventItem[]>([]);
  const abortRef = useRef<AbortController | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [events]);

  async function startInvestigation() {
    if (!datasetId || running) return;
    abortRef.current?.abort();
    const ctrl = new AbortController();
    abortRef.current = ctrl;
    setEvents([]);
    setRunning(true);

    try {
      const res = await apiFetch(`/api/v1/strategies/${strategyId}/investigate`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ dataset_id: parseInt(datasetId) }),
        signal: ctrl.signal,
      });

      if (!res.ok || !res.body) {
        setEvents([{ type: "error", content: `HTTP ${res.status}` }]);
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
          try {
            const evt = JSON.parse(line.slice(6)) as EventItem;
            setEvents((prev) => {
              // Merge consecutive thinking events into the last one
              if (evt.type === "thinking" && prev.length > 0) {
                const last = prev[prev.length - 1];
                if (last.type === "thinking") {
                  return [
                    ...prev.slice(0, -1),
                    { type: "thinking", content: last.content + evt.content },
                  ];
                }
              }
              return [...prev, evt];
            });
          } catch { /* skip malformed lines */ }
        }
      }
    } catch (e: any) {
      if (e.name !== "AbortError") {
        setEvents((prev) => [...prev, { type: "error", content: e.message ?? "Stream error" }]);
      }
    } finally {
      setRunning(false);
    }
  }

  function stop() {
    abortRef.current?.abort();
    setRunning(false);
  }

  const isDone = events.some((e) => e.type === "done" || e.type === "error");

  return (
    <section className="space-y-3">
      <div className="flex items-center gap-2 flex-wrap">
        <select
          value={datasetId}
          onChange={(e) => setDatasetId(e.target.value)}
          disabled={running}
          className="rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500 disabled:opacity-50"
        >
          <option value="">Select dataset…</option>
          {datasets.map((d) => (
            <option key={d.id} value={d.id}>
              {d.name} ({d.row_count?.toLocaleString()} rows)
            </option>
          ))}
        </select>

        {!running ? (
          <button
            onClick={startInvestigation}
            disabled={!datasetId}
            className="rounded bg-violet-600 px-3 py-1.5 text-xs text-white hover:bg-violet-500 disabled:opacity-50"
          >
            Investigate with AI
          </button>
        ) : (
          <button
            onClick={stop}
            className="rounded border border-red-700 px-3 py-1.5 text-xs text-red-400 hover:bg-red-900/30"
          >
            Stop
          </button>
        )}

        {running && (
          <span className="flex items-center gap-1.5 text-xs text-violet-400">
            <span className="h-1.5 w-1.5 rounded-full bg-violet-400 animate-pulse" />
            AI investigating…
          </span>
        )}
      </div>

      {events.length > 0 && (
        <div className="max-h-[32rem] overflow-y-auto rounded border border-gray-700 bg-gray-950 p-4 space-y-3 text-sm">
          {events.map((evt, i) => {
            if (evt.type === "start") {
              return (
                <p key={i} className="text-xs text-gray-500 italic">{evt.content}</p>
              );
            }
            if (evt.type === "thinking") {
              return (
                <div key={i} className="text-gray-200 leading-relaxed whitespace-pre-wrap">
                  {evt.content}
                </div>
              );
            }
            if (evt.type === "tool_call") {
              return (
                <div key={i} className="flex items-center gap-2">
                  <span className="rounded bg-violet-900/50 px-2 py-0.5 text-xs text-violet-300 font-mono">
                    ⚙ {evt.tool}
                  </span>
                  <span className="text-xs text-gray-500">running…</span>
                </div>
              );
            }
            if (evt.type === "tool_result") {
              return (
                <details key={i} className="rounded border border-gray-800 bg-gray-900">
                  <summary className="cursor-pointer px-3 py-1.5 text-xs text-gray-400 hover:text-gray-200 select-none">
                    <span className="text-violet-400 font-mono">{evt.tool}</span> result
                  </summary>
                  <pre className="px-3 pb-2 text-xs text-gray-400 whitespace-pre-wrap overflow-auto max-h-40">
                    {evt.content}
                  </pre>
                </details>
              );
            }
            if (evt.type === "done") {
              return (
                <p key={i} className="text-xs text-green-400 font-medium">{evt.content}</p>
              );
            }
            if (evt.type === "error") {
              return (
                <p key={i} className="text-xs text-red-400">{evt.content}</p>
              );
            }
            return null;
          })}
          <div ref={bottomRef} />
        </div>
      )}

      {isDone && events.length > 0 && (
        <button
          onClick={() => setEvents([])}
          className="text-xs text-gray-500 hover:text-gray-300"
        >
          Clear
        </button>
      )}
    </section>
  );
}
