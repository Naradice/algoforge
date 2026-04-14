"use client";

import { useEffect, useRef } from "react";

export interface Candle {
  time: number; // Unix seconds
  open: number;
  high: number;
  low: number;
  close: number;
}

export interface IndicatorSeries {
  type: "line" | "histogram";
  pane: "overlay" | "separate";
  color: string;
  group: string; // e.g. "macd", "rsi" — series sharing a group share a pane
  data: { time: number; value: number }[];
}

export interface TradeMarker {
  time: number;
  position: "aboveBar" | "belowBar";
  color: string;
  shape: "arrowUp" | "arrowDown";
  text: string;
}

export interface MarketEvent {
  time: number;       // Unix seconds
  indicator: string;  // e.g. "NONFARM_PAYROLL"
  value: number;
  unit: string;
}

interface OHLCChartProps {
  candles: Candle[];
  indicators: Record<string, IndicatorSeries>;
  markers: TradeMarker[];
  events?: MarketEvent[];
  height?: number;
}

export function OHLCChart({ candles, indicators, markers, events = [], height = 420 }: OHLCChartProps) {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!containerRef.current || candles.length === 0) return;

    let removed = false;
    let cleanupFns: (() => void)[] = [];

    import("lightweight-charts").then((lc) => {
      if (removed || !containerRef.current) return;

      const { createChart, ColorType, CrosshairMode } = lc;

      const chart = createChart(containerRef.current, {
        width: containerRef.current.clientWidth,
        height,
        layout: {
          background: { type: ColorType.Solid, color: "#111827" },
          textColor: "#9ca3af",
        },
        grid: {
          vertLines: { color: "#1f2937" },
          horzLines: { color: "#1f2937" },
        },
        crosshair: { mode: CrosshairMode.Normal },
        timeScale: {
          borderColor: "#374151",
          timeVisible: true,
          secondsVisible: false,
        },
        rightPriceScale: { borderColor: "#374151" },
        leftPriceScale: { visible: false },
        handleScroll: true,
        handleScale: true,
      });

      // ── Candlestick series (main pane) ──────────────────────────────────────
      const candleSeries = chart.addCandlestickSeries({
        upColor: "#22c55e",
        downColor: "#ef4444",
        borderVisible: false,
        wickUpColor: "#22c55e",
        wickDownColor: "#ef4444",
      });
      candleSeries.setData(candles);

      // ── Overlay indicators (EMA, SMA, BB — same price scale) ───────────────
      const overlayEntries = Object.entries(indicators).filter(
        ([, s]) => s.pane === "overlay" && s.data.length > 0
      );
      for (const [, series] of overlayEntries) {
        const ls = chart.addLineSeries({
          color: series.color,
          lineWidth: 1,
          lastValueVisible: false,
          priceLineVisible: false,
          crosshairMarkerVisible: false,
        });
        ls.setData(series.data);
      }

      // ── All markers: economic events + trades (sorted by time) ─────────────
      const allMarkers: {
        time: number;
        position: "aboveBar" | "belowBar" | "inBar";
        color: string;
        shape: "circle" | "square" | "arrowUp" | "arrowDown";
        text: string;
        size?: number;
      }[] = [];

      // Economic events → small circles below price bars
      for (const ev of events) {
        const label = ev.unit
          ? `${ev.indicator}: ${ev.value} ${ev.unit}`
          : `${ev.indicator}: ${ev.value}`;
        allMarkers.push({
          time: ev.time,
          position: "belowBar",
          color: "#facc15",   // yellow — visually distinct from trade markers
          shape: "circle",
          text: label,
          size: 1,
        });
      }

      // Trade entry / exit markers
      for (const m of markers) {
        allMarkers.push(m);
      }

      if (allMarkers.length > 0) {
        allMarkers.sort((a, b) => a.time - b.time);
        candleSeries.setMarkers(allMarkers);
      }

      chart.timeScale().fitContent();

      // ── Resize observer ─────────────────────────────────────────────────────
      const ro = new ResizeObserver(() => {
        if (containerRef.current && !removed) {
          chart.applyOptions({ width: containerRef.current.clientWidth });
        }
      });
      ro.observe(containerRef.current);

      cleanupFns = [
        () => ro.disconnect(),
        () => chart.remove(),
      ];
    });

    return () => {
      removed = true;
      cleanupFns.forEach((fn) => fn());
    };
  }, [candles, indicators, markers, height]);

  if (candles.length === 0) {
    return (
      <div
        className="flex items-center justify-center text-gray-500 text-sm"
        style={{ height }}
      >
        No OHLC data available
      </div>
    );
  }

  return <div ref={containerRef} style={{ width: "100%", height }} />;
}
