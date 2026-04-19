"use client";

import ReactECharts from "echarts-for-react";
import { useMemo, useRef, useCallback } from "react";

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

    // Compute initial zoom start percentage from defaultZoom prop
    let initialStartPct = 0;
    if (defaultZoom != null && candles.length > 0) {
      const lastTs = candles[candles.length - 1].time;
      const firstTs = candles[0].time;
      const fromTs = lastTs - defaultZoom * 30 * 24 * 3600;
      const range = lastTs - firstTs;
      initialStartPct = range > 0 ? Math.max(0, ((fromTs - firstTs) / range) * 100) : 0;
    }

    const times = candles.map((c) => new Date(c.time * 1000).toISOString());

    // Separate overlay and oscillator groups
    const overlayEntries = Object.entries(indicators).filter(([, s]) => s.pane === "overlay");
    const oscillatorGroups: Record<string, [string, IndicatorSeries][]> = {};
    for (const [name, s] of Object.entries(indicators)) {
      if (s.pane === "separate") {
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

    // Range/trend background shading from any *_is_range indicator
    const isRangeKey = Object.keys(indicators).find((k) => k.endsWith("_is_range"));
    const markAreaData: [object, object][] = [];
    if (isRangeKey) {
      const isRangeSeries = indicators[isRangeKey];
      const timeToIsRange = new Map(isRangeSeries.data.map((d) => [d.time, d.value]));
      let spanStart: number | null = null;
      for (let i = 0; i < candles.length; i++) {
        const val = timeToIsRange.get(candles[i].time);
        const isRange = val != null && val >= 0.5;
        if (isRange && spanStart === null) {
          spanStart = i;
        } else if (!isRange && spanStart !== null) {
          markAreaData.push([{ xAxis: spanStart }, { xAxis: i - 1 }]);
          spanStart = null;
        }
      }
      if (spanStart !== null) {
        markAreaData.push([{ xAxis: spanStart }, { xAxis: candles.length - 1 }]);
      }
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
        lineStyle: { color: s.color, width: 1 },
        symbol: "none",
        connectNulls: false,
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
          series.push({
            name,
            type: "line",
            xAxisIndex: gridIdx,
            yAxisIndex: gridIdx,
            data,
            lineStyle: { color: s.color, width: 1 },
            symbol: "none",
            connectNulls: false,
          });
        }
      }
    }

    // DataZoom — shared across all grids
    const dataZoom = [
      {
        type: "inside",
        xAxisIndex: grids.map((_, i) => i),
        start: initialStartPct,
        end: 100,
        zoomOnMouseWheel: true,
        moveOnMouseMove: true,
      },
      {
        type: "slider",
        xAxisIndex: grids.map((_, i) => i),
        bottom: 4,
        height: 20,
        start: initialStartPct,
        end: 100,
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
  }, [candles, indicators, markers]);

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
      <div className="flex gap-1 mb-1">
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
      <ReactECharts
        ref={chartRef}
        option={option}
        style={{ height: totalHeight, width: "100%" }}
        opts={{ renderer: "canvas" }}
        notMerge={true}
      />
    </div>
  );
}
