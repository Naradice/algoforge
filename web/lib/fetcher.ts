export async function fetcher(url: string) {
  const res = await fetch(url);
  if (!res.ok) {
    const error = await res.json().catch(() => ({ detail: res.statusText }));
    throw Object.assign(new Error(error.detail ?? "API error"), { status: res.status, info: error });
  }
  return res.json();
}
