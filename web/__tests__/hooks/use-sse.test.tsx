import { act, renderHook } from "@testing-library/react";
import { useSSE } from "@/hooks/use-sse";

// ── EventSource mock ───────────────────────────────────────────────────────────

class MockEventSource {
  static instances: MockEventSource[] = [];
  url: string;
  onmessage: ((e: { data: string }) => void) | null = null;
  onerror: ((e: unknown) => void) | null = null;
  close = jest.fn();

  constructor(url: string) {
    this.url = url;
    MockEventSource.instances.push(this);
  }

  emit(data: string) {
    this.onmessage?.({ data });
  }

  static reset() {
    MockEventSource.instances = [];
  }
}

(global as unknown as Record<string, unknown>).EventSource = MockEventSource;

beforeEach(() => MockEventSource.reset());

// ── Tests ──────────────────────────────────────────────────────────────────────

describe("useSSE", () => {
  it("creates EventSource when url is a non-null string", () => {
    renderHook(() => useSSE("/api/v1/runs/1/events", () => {}));
    expect(MockEventSource.instances).toHaveLength(1);
    expect(MockEventSource.instances[0].url).toBe("/api/v1/runs/1/events");
  });

  it("does not create EventSource when url is null", () => {
    renderHook(() => useSSE(null, () => {}));
    expect(MockEventSource.instances).toHaveLength(0);
  });

  it("calls onEvent with parsed JSON payload", () => {
    const onEvent = jest.fn();
    renderHook(() => useSSE("/api/v1/runs/1/events", onEvent));

    const es = MockEventSource.instances[0];
    act(() => es.emit('{"type":"progress","pct":50}'));

    expect(onEvent).toHaveBeenCalledWith({ type: "progress", pct: 50 });
  });

  it("silently ignores malformed JSON", () => {
    const onEvent = jest.fn();
    renderHook(() => useSSE("/api/v1/runs/1/events", onEvent));

    const es = MockEventSource.instances[0];
    act(() => es.emit("not valid json {{{"));

    expect(onEvent).not.toHaveBeenCalled();
  });

  it("calls onEvent multiple times for multiple messages", () => {
    const onEvent = jest.fn();
    renderHook(() => useSSE("/api/v1/runs/1/events", onEvent));

    const es = MockEventSource.instances[0];
    act(() => {
      es.emit('{"epoch":1}');
      es.emit('{"epoch":2}');
      es.emit('{"epoch":3}');
    });

    expect(onEvent).toHaveBeenCalledTimes(3);
    expect(onEvent).toHaveBeenNthCalledWith(3, { epoch: 3 });
  });

  it("closes EventSource on unmount", () => {
    const { unmount } = renderHook(() => useSSE("/api/v1/runs/1/events", () => {}));
    const es = MockEventSource.instances[0];
    unmount();
    expect(es.close).toHaveBeenCalledTimes(1);
  });

  it("does not create EventSource when url switches from string to null", () => {
    let url: string | null = "/api/v1/runs/1/events";
    const { rerender } = renderHook(() => useSSE(url, () => {}));
    expect(MockEventSource.instances).toHaveLength(1);

    url = null;
    rerender();
    // The old instance should be closed (cleanup runs on re-render)
    expect(MockEventSource.instances[0].close).toHaveBeenCalled();
  });
});
