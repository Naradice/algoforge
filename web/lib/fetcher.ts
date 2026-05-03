const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "");

export function resolveApiUrl(url: string): string {
  // Browser: leave relative URLs as-is — Next.js rewrites proxy /api/v1/* to the backend.
  // SSR: prepend API_BASE_URL so the Next.js server can reach the backend directly.
  if (!url.startsWith("/") || typeof window !== "undefined" || !API_BASE_URL) return url;
  return `${API_BASE_URL}${url}`;
}

export async function apiFetch(url: string, init?: RequestInit) {
  return fetch(resolveApiUrl(url), init);
}

export async function fetcher(url: string) {
  const res = await apiFetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const message = body.error?.message ?? body.detail ?? res.statusText;
    const code = body.error?.code ?? "API_ERROR";
    throw Object.assign(new Error(message), { status: res.status, code, info: body });
  }
  const body = await res.json();
  // Unwrap the { data, meta } envelope; fall back to raw body for non-enveloped responses
  return "data" in body ? body.data : body;
}

/** Fetcher that also returns pagination meta alongside the data. */
export async function fetcherWithMeta(url: string): Promise<{ data: unknown; meta: { total?: number; page?: number; page_size?: number } }> {
  const res = await apiFetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const message = body.error?.message ?? body.detail ?? res.statusText;
    throw Object.assign(new Error(message), { status: res.status, info: body });
  }
  const body = await res.json();
  return { data: body.data ?? body, meta: body.meta ?? {} };
}
