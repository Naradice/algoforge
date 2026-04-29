"use client";

import ReactECharts from "echarts-for-react";
import { useMemo, useRef, useCallback, useState, useEffect } from "react";

export interface Candle {
  time: number;
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface IndicatorSeries {
  type: "line" | "histogram";
  pane: "overlay" | "separate";
  color: string;
  group: string;
  data: { time: number; value: number }[];
  line_style?: "solid" | "dashed" | "step";
}

export interface TradeMarker {
  time: number;
  position: "aboveBar" | "belowBar";
  color: string;
  shape: "arrowUp" | "arrowDown" | "circle";
  text: string;
}

interface StrategyChartProps {
  candles: Candle[];
  indicators: Record<string, IndicatorSeries>;
  markers: TradeMarker[];
  defaultZoom?: number | null; // months to zoom on mount; null = show all
}

function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${alpha})`;
}


const PRESETS = [
  { label: "1M", months: 1 },
  { label: "3M", months: 3 },
  { label: "6M", months: 6 },
  { label: "1Y", months: 12 },
  { label: "2Y", months: 24 },
  { label: "All", months: null },
] as const;

export function StrategyChart({ candles, indicators, markers, defaultZoom }: StrategyChartProps) {
  const chartRef = useRef<ReactECharts>(null);
  // Persist zoom across option recomputes that don't change the candle set
  const zoomStateRef = useRef<{ start: number; end: number } | null>(null);
  const candlesSigRef = useRef<string>("");

  // Background shading indicator selector
  const indicatorKeys = Object.keys(indicators);
  // Synthetic BG keys for combined condition masks
  const COND_LE_KEY = "__cond_long_entry__";
  const COND_SE_KEY = "__cond_short_entry__";
  const defaultBgKey = indicatorKeys.find((k) => k.endsWith("_is_range")) ?? "";
  const [bgKey, setBgKey] = useState(defaultBgKey);
  // Reset when indicators change (new run selected)
  useEffect(() => {
    setBgKey(indicatorKeys.find((k) => k.endsWith("_is_range")) ?? "");
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Object.keys(indicators).join(",")]);

  // Compute combined condition masks (AND of all step series per side) — used both in chart and dropdown
  const { leMask, seMask } = useMemo(() => {
    const expand = (s: IndicatorSeries): boolean[] => {
      const result = new Array(candles.length).fill(false);
      if (!s.data.length) return result;
      let cur = 0, di = 0;
      for (let i = 0; i < candles.length; i++) {
        while (di + 1 < s.data.length && s.data[di + 1].time <= candles[i].time) di++;
        if (s.data[di].time <= candles[i].time) cur = s.data[di].value;
        result[i] = cur >= 0.5;
      }
      return result;
    };
    const mask = (suffix: string): boolean[] => {
      const matching = Object.values(indicators).filter(
        (s) => s.line_style === "step" && s.group?.endsWith(suffix)
      );
      if (!matching.length) return new Array(candles.length).fill(false);
      const expanded = matching.map(expand);
      return candles.map((_, i) => expanded.every((arr) => arr[i]));
    };
    return { leMask: mask("long_entry"), seMask: mask("short_entry") };
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [candles, indicators]);

  // Sub-graph group visibility
  const allOscGroups = useMemo(() => {
    const groups = new Map<string, string>(); // group → representative color
    for (const s of Object.values(indicators)) {
      if (s.pane === "separate" && !groups.has(s.group)) groups.set(s.group, s.color);
    }
    return groups;
  }, [indicators]);
  const [hiddenGroups, setHiddenGroups] = useState<Set<string>>(new Set());
  // Reset hidden groups when the indicator set changes
  useEffect(() => {
    setHiddenGroups(new Set());
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [Object.keys(indicators).join(",")]);
  const toggleGroup = useCallback((g: string) => {
    setHiddenGroups((prev) => {
      const next = new Set(prev);
      next.has(g) ? next.delete(g) : next.add(g);
      return next;
    });
  }, []);

  const zoomTo = useCallback((months: number | null) => {
    const instance = (chartRef.current as any)?.getEchartsInstance?.();
    if (!instance || candles.length === 0) return;
    if (months === null) {
      instance.dispatchAction({ type: "dataZoom", dataZoomIndex: 0, start: 0, end: 100 });
      return;
    }
    const lastTs = candles[candles.length - 1].time;
    const fromTs = lastTs - months * 30 * 24 * 3600;
    const firstTs = candles[0].time;
    const range = lastTs - firstTs;
    const startPct = Math.max(0, ((fromTs - firstTs) / range) * 100);
    instance.dispatchAction({ type: "dataZoom", dataZoomIndex: 0, start: startPct, end: 100 });
  }, [candles]);

  const option = useMemo(() => {
    if (candles.length === 0) return null;

    // Determine zoom start/end: preserve user zoom unless the candle set changed
    const candlesSig = candles.length > 0
      ? `${candles[0].time}-${candles[candles.length - 1].time}-${candles.length}`
      : "";
    const candlesChanged = candlesSig !== candlesSigRef.current;
    candlesSigRef.current = candlesSig;

    let zoomStart = 0;
    let zoomEnd = 100;
    if (!candlesChanged && zoomStateRef.current) {
      zoomStart = zoomStateRef.current.start;
      zoomEnd = zoomStateRef.current.end;
    } else if (defaultZoom != null && candles.length > 0) {
      const lastTs = candles[candles.length - 1].time;
      const firstTs = candles[0].time;
      const fromTs = lastTs - defaultZoom * 30 * 24 * 3600;
      const range = lastTs - firstTs;
      zoomStart = range > 0 ? Math.max(0, ((fromTs - firstTs) / range) * 100) : 0;
    }

    const times = candles.map((c) => new Date(c.time * 1000).toISOString());

    // Separate overlay and oscillator groups (hidden groups excluded)
    const overlayEntries = Object.entries(indicators).filter(([, s]) => s.pane === "overlay");
    const oscillatorGroups: Record<string, [string, IndicatorSeries][]> = {};
    for (const [name, s] of Object.entries(indicators)) {
      if (s.pane === "separate" && !hiddenGroups.has(s.group)) {
        if (!oscillatorGroups[s.group]) oscillatorGroups[s.group] = [];
        oscillatorGroups[s.group].push([name, s]);
      }
    }
    const oscGroupNames = Object.keys(oscillatorGroups);

    // Grid layout: price pane + one pane per oscillator group
    const priceHeightPct = oscGroupNames.length === 0 ? 78 : Math.max(40, 78 - oscGroupNames.length * 15);
    const oscHeightPct = oscGroupNames.length > 0 ? Math.floor((78 - priceHeightPct) / oscGroupNames.length) : 0;

    const grids: object[] = [
      { left: 60, right: 60, top: 16, height: `${priceHeightPct}%` },
    ];
    let topOffset = priceHeightPct + 4;
    for (let i = 0; i < oscGroupNames.length; i++) {
      grids.push({ left: 60, right: 60, top: `${topOffset}%`, height: `${oscHeightPct}%` });
      topOffset += oscHeightPct + 2;
    }

    // X-axes — one per grid, all linked
    const xAxes = grids.map((_, i) => ({
      type: "category",
      data: times,
      gridIndex: i,
      axisLine: { lineStyle: { color: "#374151" } },
      axisLabel: i === grids.length - 1 ? { color: "#6b7280", fontSize: 10, formatter: (v: string) => v.slice(0, 10) } : { show: false },
      axisTick: { show: false },
      splitLine: { show: false },
      axisPointer: { label: { show: false } },
    }));

    // Y-axes — one per grid
    const yAxes = grids.map((_, i) => ({
      scale: true,
      gridIndex: i,
      splitNumber: 4,
      axisLine: { show: false },
      axisTick: { show: false },
      axisLabel: { color: "#6b7280", fontSize: 10 },
      splitLine: { lineStyle: { color: "#1f2937" } },
    }));

    // Series
    const series: object[] = [];

    // Candlestick series (grid 0)
    const candleData = candles.map((c) => [c.open, c.close, c.low, c.high]);

    // Trade markers as markPoint triangles on the candlestick series
    // Also build a per-candle-index lookup so the axis tooltip can show marker details.
    const markersByIdx = new Map<number, { text: string; color: string }[]>();
    const markPointData = markers.map((m) => {
      const idx = candles.findIndex((c) => c.time === m.time);
      if (idx < 0) return null;
      const candle = candles[idx];
      const belowBar = m.position === "belowBar";
      const price = belowBar ? candle.low * 0.997 : candle.high * 1.003;
      if (!markersByIdx.has(idx)) markersByIdx.set(idx, []);
      markersByIdx.get(idx)!.push({ text: m.text, color: m.color });
      const isCircle = m.shape === "circle";
      return {
        coord: [idx, price],
        name: m.text,
        itemStyle: { color: m.color },
        symbol: isCircle ? "circle" : "triangle",
        symbolRotate: isCircle ? 0 : (belowBar ? 0 : 180),
        symbolSize: isCircle ? 7 : 10,
      };
    }).filter(Boolean);

    // Direct lookup from candle index → candle, used in tooltip to bypass ECharts p.value quirks
    const candleByIdx = new Map(candles.map((c, i) => [i, c]));

    // Background shading: either from user-selected indicator key or combined condition masks
    const markAreaData: [object, object][] = [];
    const addMaskShading = (mask: boolean[], color: string) => {
      let start: number | null = null;
      for (let i = 0; i < mask.length; i++) {
        if (mask[i] && start === null) { start = i; }
        else if (!mask[i] && start !== null) {
          markAreaData.push([{ xAxis: start, itemStyle: { color: hexToRgba(color, 0.18) } }, { xAxis: i - 1 }]);
          start = null;
        }
      }
      if (start !== null)
        markAreaData.push([{ xAxis: start, itemStyle: { color: hexToRgba(color, 0.18) } }, { xAxis: mask.length - 1 }]);
    };

    if (bgKey === COND_LE_KEY) {
      addMaskShading(leMask, "#22c55e");
    } else if (bgKey === COND_SE_KEY) {
      addMaskShading(seMask, "#ef4444");
    } else if (bgKey && indicators[bgKey]) {
      const bgSeries = indicators[bgKey];
      const timeToVal = new Map(bgSeries.data.map((d) => [d.time, d.value]));
      const bgMask = candles.map((c) => {
        const v = timeToVal.get(c.time);
        return v != null && v >= 0.5;
      });
      addMaskShading(bgMask, "#60a5fa");
    }

    // Condition-signal background: one shaded column per circle marker
    for (const m of markers) {
      if (m.shape !== "circle") continue;
      const idx = candles.findIndex((c) => c.time === m.time);
      if (idx < 0) continue;
      markAreaData.push([
        { xAxis: idx, itemStyle: { color: hexToRgba(m.color, 0.18) } },
        { xAxis: idx },
      ]);
    }

    series.push({
      name: "Price",
      type: "candlestick",
      xAxisIndex: 0,
      yAxisIndex: 0,
      data: candleData,
      itemStyle: {
        color: "#22c55e",
        color0: "#ef4444",
        borderColor: "#22c55e",
        borderColor0: "#ef4444",
      },
      markPoint: markPointData.length > 0 ? {
        data: markPointData,
        animation: false,
        label: { show: false },
      } : undefined,
      markArea: markAreaData.length > 0 ? {
        silent: true,
        animation: false,
        itemStyle: { color: "rgba(245, 158, 11, 0.08)" },
        data: markAreaData,
      } : undefined,
    });

    // Overlay indicator series (grid 0)
    for (const [name, s] of overlayEntries) {
      const timeMap = new Map(s.data.map((d) => [d.time, d.value]));
      const data = candles.map((c) => {
        const v = timeMap.get(c.time);
        return v == null ? null : v;
      });
      series.push({
        name,
        type: "line",
        xAxisIndex: 0,
        yAxisIndex: 0,
        data,
        lineStyle: { color: s.color, width: 1, type: s.line_style === "dashed" ? "dashed" : "solid" },
        itemStyle: { color: s.color },
        symbol: "none",
        connectNulls: s.line_style === "dashed" || s.line_style === "step",
        smooth: false,
      });
    }

    // Oscillator series — each group in its own grid
    for (let gi = 0; gi < oscGroupNames.length; gi++) {
      const group = oscGroupNames[gi];
      const gridIdx = gi + 1;
      for (const [name, s] of oscillatorGroups[group]) {
        const timeMap = new Map(s.data.map((d) => [d.time, d.value]));
        const data = candles.map((c) => {
          const v = timeMap.get(c.time);
          return v == null ? null : v;
        });
        if (s.type === "histogram") {
          series.push({
            name,
            type: "bar",
            xAxisIndex: gridIdx,
            yAxisIndex: gridIdx,
            data,
            itemStyle: {
              color: (params: { value: number }) =>
                params.value >= 0 ? "#22c55e" : "#ef4444",
            },
            barMaxWidth: 4,
          });
        } else {
          const isStep = s.line_style === "step";
          const isDashed = s.line_style === "dashed";
          series.push({
            name,
            type: "line",
            xAxisIndex: gridIdx,
            yAxisIndex: gridIdx,
            data,
            lineStyle: { color: s.color, width: isDashed ? 1 : isStep ? 1.5 : 1, type: isDashed ? "dashed" : "solid" },
            itemStyle: { color: s.color },
            symbol: "none",
            connectNulls: isStep || isDashed,
            ...(isStep ? { step: "end" } : {}),
          });
        }
      }
    }

    // DataZoom — shared across all grids
    const dataZoom = [
      {
        type: "inside",
        xAxisIndex: grids.map((_, i) => i),
        start: zoomStart,
        end: zoomEnd,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
      },
      {
        type: "slider",
        xAxisIndex: grids.map((_, i) => i),
        bottom: 4,
        height: 20,
        start: zoomStart,
        end: zoomEnd,
        fillerColor: "rgba(55, 65, 81, 0.5)",
        borderColor: "#374151",
        handleStyle: { color: "#6b7280" },
        textStyle: { color: "#6b7280", fontSize: 10 },
        dataBackground: {
          areaStyle: { color: "#1f2937" },
          lineStyle: { color: "#374151" },
        },
      },
    ];

    return {
      backgroundColor: "#111827",
      animation: false,
      grid: grids,
      xAxis: xAxes,
      yAxis: yAxes,
      series,
      dataZoom,
      tooltip: {
        trigger: "axis",
        axisPointer: { type: "cross", link: grids.map((_, i) => ({ xAxisIndex: i })) },
        backgroundColor: "#1f2937",
        borderColor: "#374151",
        textStyle: { color: "#e5e7eb", fontSize: 11 },
        formatter: (params: object[]) => {
          if (!Array.isArray(params) || params.length === 0) return "";
          type P = { seriesName: string; seriesType: string; value: number | number[]; color: string; dataIndex: number; axisValue: string };
          const ps = params as P[];
          // Use the candlestick entry's dataIndex so it's correct even when hovering an oscillator pane
          const candleEntry = ps.find((p) => p.seriesType === "candlestick") ?? ps[0];
          const dataIndex = candleEntry.dataIndex;
          const time = (ps[0].axisValue ?? "").slice(0, 16).replace("T", " ");
          const lines = [`<b>${time}</b>`];
          // OHLC from our own array — avoids any ECharts internal value reordering
          const candle = candleByIdx.get(dataIndex);
          if (candle) {
            lines.push(
              `<span style="color:#9ca3af">●</span> Price: ` +
              `O:${candle.open.toFixed(4)} H:${candle.high.toFixed(4)} L:${candle.low.toFixed(4)} C:${candle.close.toFixed(4)}`
            );
          }
          // Indicator values (skip the candlestick series — already shown above)
          for (const p of ps) {
            if (p.seriesType === "candlestick") continue;
            if (typeof p.value !== "number") continue;
            lines.push(`<span style="color:${p.color}">●</span> ${p.seriesName}: ${p.value.toFixed(4)}`);
          }
          const barMarkers = markersByIdx.get(dataIndex);
          if (barMarkers && barMarkers.length > 0) {
            lines.push('<hr style="border-color:#374151;margin:4px 0"/>');
            for (const bm of barMarkers) {
              lines.push(`<span style="color:${bm.color}">▶</span> ${bm.text}`);
            }
          }
          return lines.join("<br/>");
        },
      },
      axisPointer: {
        link: grids.map((_, i) => ({ xAxisIndex: i })),
        label: { backgroundColor: "#374151" },
      },
      legend: {
        top: 0,
        right: 60,
        textStyle: { color: "#9ca3af", fontSize: 10 },
        itemHeight: 8,
      },
    };
  }, [candles, indicators, markers, bgKey, hiddenGroups]);

  if (!option || candles.length === 0) {
    return (
      <div className="flex items-center justify-center text-gray-500 text-sm" style={{ height: 500 }}>
        No chart data available
      </div>
    );
  }

  const oscCount = Object.values(indicators).filter((s) => s.pane === "separate")
    .map((s) => s.group)
    .filter((v, i, a) => a.indexOf(v) === i).length;
  const totalHeight = 420 + oscCount * 120;

  return (
    <div>
      <div className="flex items-center gap-2 mb-1 flex-wrap">
        <div className="flex gap-1">
          {PRESETS.map(({ label, months }) => (
            <button
              key={label}
              onClick={() => zoomTo(months)}
              className="rounded px-2 py-0.5 text-xs text-gray-400 hover:text-white hover:bg-gray-700 border border-gray-700"
            >
              {label}
            </button>
          ))}
        </div>
        {indicatorKeys.length > 0 && (
          <div className="flex items-center gap-1 ml-auto">
            <span className="text-xs text-gray-500">BG:</span>
            <select
              value={bgKey}
              onChange={(e) => setBgKey(e.target.value)}
              className="rounded border border-gray-700 bg-gray-900 px-1.5 py-0.5 text-xs text-gray-300 focus:outline-none focus:ring-1 focus:ring-brand-500"
            >
              <option value="">None</option>
              {leMask.some(Boolean) && <option value={COND_LE_KEY}>All Long Entry conditions</option>}
              {seMask.some(Boolean) && <option value={COND_SE_KEY}>All Short Entry conditions</option>}
              {indicatorKeys.filter((k) => !k.startsWith("__cond_")).map((k) => (
                <option key={k} value={k}>{k}</option>
              ))}
            </select>
          </div>
        )}
      </div>
      {allOscGroups.size > 0 && (
        <div className="flex items-center gap-1.5 flex-wrap mb-1">
          {Array.from(allOscGroups.entries()).map(([group, color]) => {
            const hidden = hiddenGroups.has(group);
            return (
              <button
                key={group}
                onClick={() => toggleGroup(group)}
                className={`flex items-center gap-1 rounded px-2 py-0.5 text-xs border transition-opacity ${
                  hidden
                    ? "border-gray-700 text-gray-600 opacity-40"
                    : "border-gray-600 text-gray-300"
                }`}
              >
                <span
                  className="inline-block w-2 h-2 rounded-full flex-shrink-0"
                  style={{ backgroundColor: hidden ? "#4b5563" : color }}
                />
                {group}
              </button>
            );
          })}
        </div>
      )}
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: totalHeight, width: "100%" }}
        opts={{ renderer: "canvas" }}
        notMerge={true}
        onEvents={{
          datazoom: () => {
            const instance = (chartRef.current as any)?.getEchartsInstance?.();
            if (!instance) return;
            const opt = instance.getOption();
            const dz = opt?.dataZoom?.[0];
            if (dz != null) zoomStateRef.current = { start: dz.start ?? 0, end: dz.end ?? 100 };
          },
        }}
      />
    </div>
  );
}
