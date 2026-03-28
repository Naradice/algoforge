export async function fetcher(url: string) {
  const res = await fetch(url);
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
  const res = await fetch(url);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    const message = body.error?.message ?? body.detail ?? res.statusText;
    throw Object.assign(new Error(message), { status: res.status, info: body });
  }
  const body = await res.json();
  return { data: body.data ?? body, meta: body.meta ?? {} };
}
