"use client";

import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";

export default function DashboardPage() {
  const { data: health } = useSWR("/api/v1/health", fetcher);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-semibold text-white">Dashboard</h1>

      <div className="grid grid-cols-3 gap-4">
        <SummaryCard title="Strategies" href="/strategy" />
        <SummaryCard title="ML Models" href="/model" />
        <SummaryCard title="Datasets" href="/data" />
      </div>

      {health && (
        <p className="text-xs text-gray-500">
          API {health.status} · v{health.version}
        </p>
      )}
    </div>
  );
}

function SummaryCard({ title, href }: { title: string; href: string }) {
  return (
    <a
      href={href}
      className="block rounded-lg border border-gray-800 bg-gray-900 p-5 hover:border-brand-500 transition-colors"
    >
      <p className="text-xs text-gray-400 uppercase tracking-wide">{title}</p>
      <p className="mt-2 text-3xl font-bold text-white">—</p>
    </a>
  );
}
