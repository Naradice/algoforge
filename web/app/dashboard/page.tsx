"use client";

import useSWR from "swr";
import { fetcher, fetcherWithMeta } from "@/lib/fetcher";
import { StatusBadge } from "@/components/status-badge";

type ActivityItem = {
  id: string; label: string; sublabel: string; status: string; ts: string; href: string;
};

export default function DashboardPage() {
  const { data: health } = useSWR("/api/v1/health", fetcher);
  const { data: strategies } = useSWR("/api/v1/strategies?page_size=1", fetcherWithMeta);
  const { data: models } = useSWR("/api/v1/models?page_size=1", fetcherWithMeta);
  const { data: datasets } = useSWR("/api/v1/datasets?page_size=1", fetcherWithMeta);

  const { data: recentDatasets } = useSWR("/api/v1/datasets?page_size=5", fetcher, { refreshInterval: 10000 });
  const { data: recentJobs } = useSWR("/api/v1/collection-jobs?page_size=5", fetcher, { refreshInterval: 5000 });
  const { data: recentModels } = useSWR("/api/v1/models?page_size=5", fetcher, { refreshInterval: 10000 });
  const { data: recentStrategies } = useSWR("/api/v1/strategies?page_size=5", fetcher, { refreshInterval: 10000 });

  const activity: ActivityItem[] = [];

  (recentDatasets as any[] | undefined)?.forEach((d: any) => {
    activity.push({
      id: `ds-${d.id}`, label: d.name,
      sublabel: `Dataset · ${d.row_count?.toLocaleString() ?? "?"} rows${d.symbol ? ` · ${d.symbol}` : ""}${d.timeframe ? ` ${d.timeframe}` : ""}`,
      status: d.status, ts: d.created_at, href: `/data/datasets/${d.id}`,
    });
  });
  (recentJobs as any[] | undefined)?.forEach((j: any) => {
    if (j.last_run_at) {
      activity.push({
        id: `job-${j.id}`, label: `Collection job #${j.id}`,
        sublabel: j.last_error ? j.last_error.slice(0, 80) : `Datasource ${j.datasource_id}`,
        status: j.status, ts: j.last_run_at, href: `/data/datasources/${j.datasource_id}`,
      });
    }
  });
  (recentModels as any[] | undefined)?.forEach((m: any) => {
    activity.push({ id: `model-${m.id}`, label: m.name, sublabel: `Model · ${m.architecture}`, status: m.status, ts: m.created_at, href: `/model/${m.id}` });
  });
  (recentStrategies as any[] | undefined)?.forEach((s: any) => {
    activity.push({ id: `strat-${s.id}`, label: s.name, sublabel: `Strategy${s.description ? ` · ${s.description}` : ""}`, status: s.status, ts: s.created_at, href: `/strategy/${s.id}` });
  });

  activity.sort((a, b) => new Date(b.ts).getTime() - new Date(a.ts).getTime());
  const recentActivity = activity.slice(0, 10);

  return (
    <div className="space-y-8 max-w-4xl">
      {/* Header */}
      <div>
        <h1 className="md-title-lg">Dashboard</h1>
        {health && <p className="md-body-sm mt-1">API {health.status} · v{health.version}</p>}
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-3 gap-4">
        <SummaryCard title="Strategies" href="/strategy" count={strategies?.meta.total} icon="📈" />
        <SummaryCard title="ML Models" href="/model" count={models?.meta.total} icon="🧠" />
        <SummaryCard title="Datasets" href="/data" count={datasets?.meta.total} icon="💾" />
      </div>

      {/* Quick actions */}
      <div className="flex flex-wrap gap-3">
        <a href="/strategy/new" className="md-btn md-btn-outlined md-btn-sm">+ New Strategy</a>
        <a href="/model/new" className="md-btn md-btn-outlined md-btn-sm">+ New Model</a>
        <a href="/data/new" className="md-btn md-btn-outlined md-btn-sm">+ New Datasource</a>
      </div>

      {/* Recent activity */}
      <section>
        <h2 className="md-label-md mb-4">Recent Activity</h2>
        {recentActivity.length === 0 ? (
          <div className="md-empty-state">
            <p className="text-gray-200 font-medium">No activity yet.</p>
            <p className="md-body-sm mt-1">Create a datasource, train a model, or run a strategy to get started.</p>
          </div>
        ) : (
          <div className="md-card divide-y divide-gray-800 overflow-hidden">
            {recentActivity.map((item) => (
              <a key={item.id} href={item.href}
                className="flex items-center justify-between px-5 py-4 hover:bg-gray-800/50 transition-colors">
                <div className="min-w-0 flex-1">
                  <p className="md-label-lg truncate">{item.label}</p>
                  <p className="md-body-sm truncate mt-0.5">{item.sublabel}</p>
                </div>
                <div className="ml-6 flex flex-col items-end gap-1.5 shrink-0">
                  <StatusBadge status={item.status} />
                  <span className="text-xs text-gray-500">
                    {new Date(item.ts).toLocaleString(undefined, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" })}
                  </span>
                </div>
              </a>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function SummaryCard({ title, href, count, icon }: { title: string; href: string; count?: number; icon: string }) {
  return (
    <a href={href} className="md-card block p-6 hover:border-brand-500/60 hover:shadow-elevation-2 transition-all group">
      <div className="flex items-start justify-between mb-3">
        <p className="md-label-md">{title}</p>
        <span className="text-xl">{icon}</span>
      </div>
      <p className="text-4xl font-bold text-gray-50 tabular-nums">{count ?? "—"}</p>
      <p className="md-body-sm mt-2 text-brand-400 opacity-0 group-hover:opacity-100 transition-opacity">View all →</p>
    </a>
  );
}
