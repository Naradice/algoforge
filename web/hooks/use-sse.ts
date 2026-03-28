"use client";

import { useEffect } from "react";

export function useSSE(url: string | null, onEvent: (data: unknown) => void) {
  useEffect(() => {
    if (!url) return;
    const es = new EventSource(url);
    es.onmessage = (e) => {
      try {
        onEvent(JSON.parse(e.data));
      } catch {
        // ignore malformed events
      }
    };
    es.onerror = () => {
      // SSE will auto-reconnect; no action needed
    };
    return () => es.close();
  }, [url]); // eslint-disable-line react-hooks/exhaustive-deps
}
