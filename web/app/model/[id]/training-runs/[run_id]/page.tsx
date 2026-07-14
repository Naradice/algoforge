"use client";

import { useState, useCallback } from "react";
import { useParams, useRouter } from "next/navigation";
import useSWR from "swr";
import { apiFetch, fetcher } from "@/lib/fetcher";
import { LossChart } from "@/components/loss-chart";
import { useSSE } from "@/hooks/use-sse";

interface EpochMetric {
  epoch: number;
  train_loss: number;
  val_loss: number;
  lr: number | null;
}

interface TrainingRunDetail {
  hyperparams: Record<string, unknown>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  preprocessed_characteristics: Record<string, any> | null;
}

// ── Data Characteristics (as trained) ───────────────────────────────────────

function StatGrid({ stats }: { stats: { label: string; value: string }[] }) {
  return (
    <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-3">
      {stats.map(({ label, value }) => (
        <div key={label} className="rounded border border-gray-800 bg-gray-950 p-2">
          <p className="text-xs text-gray-500 uppercase">{label}</p>
          <p className="mt-0.5 text-sm font-bold text-white font-mono">{value}</p>
        </div>
      ))}
    </div>
  );
}

function preprocessingSummary(hp: Record<string, unknown>): string {
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const pp = (hp.preprocessing ?? {}) as Record<string, any>;
  const parts: string[] = [];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const indicators = (pp.indicators ?? []) as any[];
  if (indicators.length > 0) {
    parts.push(indicators.map((i) => `${i.type}${i.period != null ? `(${i.period})` : ""}`).join(", "));
  }
  if (pp.clustering?.enabled) parts.push(`clustering(k=${pp.clustering.n_clusters ?? "?"})`);
  if (parts.length === 0) parts.push("none");
  const featureCols = Array.isArray(hp.feature_cols) ? (hp.feature_cols as string[]).join(", ") : "close";
  const normalize = (hp.normalize as string) ?? "returns";
  return `indicators: ${parts.join(" + ")} · features: [${featureCols}] · normalize: ${normalize}`;
}

function DataCharacteristicsCard({ hyperparams, characteristics }: {
  hyperparams: Record<string, unknown>;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  characteristics: Record<string, any> | null;
}) {
  if (!characteristics) {
    return (
      <div className="rounded border border-gray-700 bg-gray-900 p-4">
        <h3 className="mb-2 text-sm font-medium text-gray-300">Data Characteristics (as trained)</h3>
        <p className="text-xs text-gray-500">
          Not yet recorded for this run — computed once training starts building the model.
        </p>
      </div>
    );
  }

  const lrd = characteristics.long_range_dependence ?? {};
  const sp = characteristics.spectral_periodicity ?? {};
  const mw = characteristics.multiscale_wavelet ?? {};
  const cn = characteristics.complexity_nonlinearity ?? {};
  const rc = characteristics.regime_changes ?? {};

  const num = (v: unknown, fmt: (n: number) => string) =>
    typeof v === "number" && !isNaN(v) ? fmt(v) : "—";

  const stats = [
    { label: "Hurst", value: num(lrd.hurst, (v) => v.toFixed(4)) },
    { label: "Behavior", value: lrd.interpretation ?? "—" },
    { label: "Memory length", value: num(lrd.memory_length, (v) => String(v)) },
    { label: "Periodicity strength", value: num(sp.periodicity_strength, (v) => v.toFixed(2)) },
    { label: "Spectral entropy", value: num(sp.spectral_entropy, (v) => v.toFixed(3)) },
    { label: "Wavelet flatness", value: num(mw.flatness_score, (v) => v.toFixed(3)) },
    { label: "Permutation entropy", value: num(cn.permutation_entropy, (v) => v.toFixed(3)) },
    { label: "Sample entropy", value: num(cn.sample_entropy, (v) => v.toFixed(3)) },
    { label: "Nonlinear (BDS)?", value: cn.nonlinear === true ? "Yes" : cn.nonlinear === false ? "No" : "—" },
    { label: "Changepoints", value: num(rc.n_changepoints, (v) => String(v)) },
  ];

  return (
    <div className="rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
      <h3 className="text-sm font-medium text-gray-300">Data Characteristics (as trained)</h3>
      <p className="text-xs text-gray-500 font-mono">{preprocessingSummary(hyperparams)}</p>
      <StatGrid stats={stats} />
      <p className="text-xs text-gray-600">
        Computed on the primary feature column after preprocessing and the row cap, before
        normalization — the same structure indicators as the dataset &quot;Structure&quot; tab,
        applied to what this run actually trained on.
      </p>
    </div>
  );
}

export default function TrainingRunDetailPage() {
  const params = useParams<{ id: string; run_id: string }>();
  const router = useRouter();
  const modelId = params.id;
  const runId = params.run_id;

  const [liveMetrics, setLiveMetrics] = useState<EpochMetric[]>([]);
  const [stopping, setStopping] = useState(false);

  const modelUrl = `/api/v1/models/${modelId}`;
  const runUrl = `/api/v1/training-runs/${runId}/status`;
  const metricsUrl = `/api/v1/training-runs/${runId}/metrics`;
  const runDetailUrl = `/api/v1/models/${modelId}/training-runs/${runId}`;

  const { data: model } = useSWR(modelUrl, fetcher);
  const { data: runStatus, mutate: mutateStatus } = useSWR(runUrl, fetcher, {
    refreshInterval: (data) => (data?.status === "running" || data?.status === "pending") ? 3000 : 0,
  });
  const { data: runDetail } = useSWR<TrainingRunDetail>(runDetailUrl, fetcher, {
    refreshInterval: (data) => !data?.preprocessed_characteristics && (runStatus?.status === "running" || runStatus?.status === "pending") ? 5000 : 0,
  });
  const { data: epochMetrics, mutate: mutateMetrics } = useSWR<EpochMetric[]>(metricsUrl, fetcher, {
    refreshInterval: runStatus?.status === "running" ? 5000 : 0,
  });

  const onSSEEvent = useCallback((data: unknown) => {
    const evt = data as { type: string; epoch?: number; train_loss?: number; val_loss?: number; lr?: number };
    if (evt.type === "epoch_completed" && evt.epoch !== undefined) {
      setLiveMetrics((prev) => {
        const existing = prev.find((m) => m.epoch === evt.epoch);
        if (existing) return prev;
        return [...prev, { epoch: evt.epoch!, train_loss: evt.train_loss ?? 0, val_loss: evt.val_loss ?? 0, lr: evt.lr ?? null }];
      });
      mutateMetrics();
    }
  }, [mutateMetrics]);

  useSSE(
    runStatus?.status === "running" ? `/api/v1/training-runs/${runId}/events` : null,
    onSSEEvent
  );

  async function handleStop() {
    setStopping(true);
    try {
      await apiFetch(`/api/v1/training-runs/${runId}/stop`, { method: "POST" });
      mutateStatus();
    } finally {
      setStopping(false);
    }
  }

  const chartData = epochMetrics && epochMetrics.length > 0 ? epochMetrics : liveMetrics;
  const isActive = runStatus?.status === "running" || runStatus?.status === "pending";
  const totalEpochs = runStatus?.total_epochs;
  const currentEpoch = runStatus?.current_epoch ?? 0;
  const progressPct = totalEpochs ? (currentEpoch / totalEpochs) * 100 : 0;

  return (
    <div className="space-y-6">
      {/* Breadcrumb */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2 text-sm text-gray-400">
          <button onClick={() => router.push("/model")} className="hover:text-white">Models</button>
          <span>/</span>
          <button onClick={() => router.push(`/model/${modelId}`)} className="hover:text-white">
            {model?.name ?? `Model ${modelId}`}
          </button>
          <span>/</span>
          <span className="text-white">Training Run #{runId}</span>
        </div>
        <div className="flex items-center gap-3">
          {runStatus && (
            <span className={`rounded px-2 py-1 text-xs font-medium ${
              runStatus.status === "completed" ? "bg-green-900 text-green-300" :
              runStatus.status === "running" ? "bg-blue-900 text-blue-300" :
              runStatus.status === "error" ? "bg-red-900 text-red-300" :
              "bg-gray-700 text-gray-300"
            }`}>
              {runStatus.status}
            </span>
          )}
          {isActive && (
            <button
              onClick={handleStop}
              disabled={stopping || runStatus?.stop_requested}
              className="rounded bg-red-600 px-3 py-1.5 text-sm text-white hover:bg-red-500 disabled:opacity-50"
            >
              {runStatus?.stop_requested ? "Stop requested…" : stopping ? "Stopping…" : "Stop"}
            </button>
          )}
        </div>
      </div>

      {/* Progress */}
      {(isActive || totalEpochs) && (
        <div className="rounded border border-gray-700 bg-gray-900 p-4 space-y-3">
          <div className="flex justify-between text-sm">
            <span className="text-gray-400">Epoch {currentEpoch}{totalEpochs ? ` / ${totalEpochs}` : ""}</span>
            <span className="text-gray-400">
              {runStatus?.elapsed_seconds != null && `${Math.round(runStatus.elapsed_seconds)}s elapsed`}
              {runStatus?.eta_seconds != null && runStatus.eta_seconds > 0 && ` · ${Math.round(runStatus.eta_seconds)}s ETA`}
            </span>
          </div>
          {totalEpochs && (
            <div className="h-2 rounded bg-gray-700">
              <div className="h-2 rounded bg-brand-500 transition-all" style={{ width: `${progressPct}%` }} />
            </div>
          )}
          {runStatus?.val_loss != null && (
            <div className="text-xs text-gray-400">Best val loss: <span className="text-white">{runStatus.val_loss.toFixed(6)}</span></div>
          )}
        </div>
      )}

      {/* Loss Chart */}
      <div className="rounded border border-gray-700 bg-gray-900 p-4">
        <h3 className="mb-3 text-sm font-medium text-gray-300">Training Loss</h3>
        <LossChart data={chartData ?? []} />
      </div>

      {/* Best epoch card */}
      {runStatus?.best_epoch != null && (
        <div className="rounded border border-gray-700 bg-gray-900 p-4 flex gap-6">
          <div>
            <div className="text-xs text-gray-400">Best Epoch</div>
            <div className="text-2xl font-semibold text-white">{runStatus.best_epoch}</div>
          </div>
          {runStatus?.val_loss != null && (
            <div>
              <div className="text-xs text-gray-400">Best Val Loss</div>
              <div className="text-2xl font-semibold text-green-400">{runStatus.val_loss.toFixed(6)}</div>
            </div>
          )}
        </div>
      )}

      {/* Data characteristics as actually trained on (preprocessing + row cap applied) */}
      {runDetail && (
        <DataCharacteristicsCard
          hyperparams={runDetail.hyperparams}
          characteristics={runDetail.preprocessed_characteristics}
        />
      )}
    </div>
  );
}
