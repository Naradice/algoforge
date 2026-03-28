"use client";

import useSWR from "swr";
import { fetcher, fetcherWithMeta } from "@/lib/fetcher";

export default function DashboardPage() {
  const { data: health } = useSWR("/api/v1/health", fetcher);
  const { data: strategies } = useSWR("/api/v1/strategies?page_size=1", fetcherWithMeta);
  const { data: models } = useSWR("/api/v1/models?page_size=1", fetcherWithMeta);
  const { data: datasets } = useSWR("/api/v1/datasets?page_size=1", fetcherWithMeta);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-white">Dashboard</h1>

      <div className="grid grid-cols-3 gap-4">
        <SummaryCard title="Strategies" href="/strategy" count={strategies?.meta.total} />
        <SummaryCard title="ML Models" href="/model" count={models?.meta.total} />
        <SummaryCard title="Datasets" href="/data" count={datasets?.meta.total} />
      </div>

      {health && (
        <p className="text-xs text-gray-500">
          API {health.status} · v{health.version}
        </p>
      )}
    </div>
  );
}

function SummaryCard({ title, href, count }: { title: string; href: string; count?: number }) {
  return (
    <a
      href={href}
      className="block rounded-lg border border-gray-800 bg-gray-900 p-5 hover:border-brand-500 transition-colors"
    >
      <p className="text-xs text-gray-400 uppercase tracking-wide">{title}</p>
      <p className="mt-2 text-3xl font-bold text-white">{count ?? "—"}</p>
    </a>
  );
}
