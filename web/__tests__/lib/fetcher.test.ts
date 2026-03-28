import { fetcher, fetcherWithMeta } from "@/lib/fetcher";

const mockFetch = jest.fn();
global.fetch = mockFetch;

beforeEach(() => mockFetch.mockReset());

// ── Helper builders ────────────────────────────────────────────────────────────

function okResponse(body: unknown) {
  return { ok: true, json: async () => body };
}

function errResponse(status: number, statusText: string, body: unknown) {
  return { ok: false, status, statusText, json: async () => body };
}

// ── fetcher ────────────────────────────────────────────────────────────────────

describe("fetcher", () => {
  it("unwraps data envelope { data, meta }", async () => {
    mockFetch.mockResolvedValueOnce(okResponse({ data: { id: 1 }, meta: {} }));
    expect(await fetcher("/api/v1/strategies/1")).toEqual({ id: 1 });
  });

  it("returns raw body when no data key present", async () => {
    mockFetch.mockResolvedValueOnce(okResponse({ status: "ok", version: "0.1.0" }));
    expect(await fetcher("/api/v1/health")).toEqual({ status: "ok", version: "0.1.0" });
  });

  it("returns empty array when data is []", async () => {
    mockFetch.mockResolvedValueOnce(okResponse({ data: [], meta: { total: 0 } }));
    expect(await fetcher("/api/v1/strategies")).toEqual([]);
  });

  it("throws with message from error.message envelope", async () => {
    mockFetch.mockResolvedValueOnce(
      errResponse(404, "Not Found", { error: { code: "NOT_FOUND", message: "Strategy not found" } })
    );
    await expect(fetcher("/api/v1/strategies/99")).rejects.toThrow("Strategy not found");
  });

  it("falls back to body.detail string when no error envelope", async () => {
    mockFetch.mockResolvedValueOnce(
      errResponse(422, "Unprocessable Entity", { detail: "Validation failed" })
    );
    await expect(fetcher("/api/v1/test")).rejects.toThrow("Validation failed");
  });

  it("falls back to statusText when body is empty", async () => {
    mockFetch.mockResolvedValueOnce(errResponse(500, "Internal Server Error", {}));
    await expect(fetcher("/api/v1/test")).rejects.toThrow("Internal Server Error");
  });

  it("attaches .status to thrown error", async () => {
    mockFetch.mockResolvedValueOnce(
      errResponse(404, "Not Found", { error: { code: "NOT_FOUND", message: "Not found" } })
    );
    try {
      await fetcher("/api/v1/test");
    } catch (err: unknown) {
      expect((err as { status: number }).status).toBe(404);
    }
  });

  it("attaches .code to thrown error", async () => {
    mockFetch.mockResolvedValueOnce(
      errResponse(404, "Not Found", { error: { code: "STRATEGY_NOT_FOUND", message: "Not found" } })
    );
    try {
      await fetcher("/api/v1/test");
    } catch (err: unknown) {
      expect((err as { code: string }).code).toBe("STRATEGY_NOT_FOUND");
    }
  });

  it("uses API_ERROR as default code when none provided", async () => {
    mockFetch.mockResolvedValueOnce(errResponse(500, "Error", {}));
    try {
      await fetcher("/api/v1/test");
    } catch (err: unknown) {
      expect((err as { code: string }).code).toBe("API_ERROR");
    }
  });

  it("handles malformed JSON response body gracefully", async () => {
    mockFetch.mockResolvedValueOnce({
      ok: false,
      status: 500,
      statusText: "Server Error",
      json: async () => { throw new Error("invalid json"); },
    });
    await expect(fetcher("/api/v1/test")).rejects.toThrow("Server Error");
  });
});

// ── fetcherWithMeta ────────────────────────────────────────────────────────────

describe("fetcherWithMeta", () => {
  it("returns data and full meta", async () => {
    mockFetch.mockResolvedValueOnce(
      okResponse({ data: [{ id: 1 }, { id: 2 }], meta: { total: 2, page: 1, page_size: 20 } })
    );
    const result = await fetcherWithMeta("/api/v1/strategies");
    expect(result.data).toEqual([{ id: 1 }, { id: 2 }]);
    expect(result.meta).toEqual({ total: 2, page: 1, page_size: 20 });
  });

  it("returns empty meta when not in response", async () => {
    mockFetch.mockResolvedValueOnce(okResponse({ data: [] }));
    const result = await fetcherWithMeta("/api/v1/strategies");
    expect(result.meta).toEqual({});
  });

  it("falls back to raw body as data when no data key", async () => {
    mockFetch.mockResolvedValueOnce(okResponse([1, 2, 3]));
    const result = await fetcherWithMeta("/api/v1/test");
    expect(result.data).toEqual([1, 2, 3]);
  });

  it("throws on non-ok response", async () => {
    mockFetch.mockResolvedValueOnce(
      errResponse(403, "Forbidden", { error: { message: "Forbidden" } })
    );
    await expect(fetcherWithMeta("/api/v1/test")).rejects.toThrow("Forbidden");
  });
});
