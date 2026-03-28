"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

const ARCHITECTURES = [
  { value: "seq2seq_transformer", label: "Seq2Seq Transformer" },
  { value: "lstm", label: "LSTM" },
  { value: "timegan", label: "TimeGAN" },
];

const DEFAULT_CONFIGS: Record<string, object> = {
  seq2seq_transformer: {
    d_model: 64,
    nhead: 4,
    num_encoder_layers: 2,
    num_decoder_layers: 2,
    dim_feedforward: 256,
    dropout: 0.1,
    input_size: 1,
    output_size: 1,
  },
  lstm: {
    input_size: 1,
    hidden_size: 128,
    num_layers: 2,
    output_size: 1,
    dropout: 0.1,
  },
  timegan: {
    input_size: 1,
    hidden_size: 64,
    latent_dim: 32,
    pred_len: 10,
  },
};

export default function NewModelPage() {
  const router = useRouter();
  const [name, setName] = useState("");
  const [architecture, setArchitecture] = useState("seq2seq_transformer");
  const [configText, setConfigText] = useState(
    JSON.stringify(DEFAULT_CONFIGS["seq2seq_transformer"], null, 2)
  );
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  function handleArchChange(arch: string) {
    setArchitecture(arch);
    setConfigText(JSON.stringify(DEFAULT_CONFIGS[arch] ?? {}, null, 2));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);

    let config: object;
    try {
      config = JSON.parse(configText);
    } catch {
      setError("Config is not valid JSON");
      return;
    }

    setSaving(true);
    try {
      const res = await fetch("/api/v1/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name, architecture, config }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        setError(body.error?.message ?? body.detail ?? `Error ${res.status}`);
        return;
      }
      const body = await res.json();
      const model = body.data ?? body;
      router.push(`/model/${model.id}`);
    } finally {
      setSaving(false);
    }
  }

  return (
    <div className="max-w-xl space-y-6">
      <h1 className="text-2xl font-semibold text-white">New ML Model</h1>

      <form onSubmit={handleSubmit} className="space-y-4">
        <div>
          <label className="mb-1 block text-sm text-gray-400">Name</label>
          <input
            required
            value={name}
            onChange={(e) => setName(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            placeholder="My LSTM Model"
          />
        </div>

        <div>
          <label className="mb-1 block text-sm text-gray-400">Architecture</label>
          <select
            value={architecture}
            onChange={(e) => handleArchChange(e.target.value)}
            className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
          >
            {ARCHITECTURES.map((a) => (
              <option key={a.value} value={a.value}>
                {a.label}
              </option>
            ))}
          </select>
        </div>

        <div>
          <label className="mb-1 block text-sm text-gray-400">
            Architecture Config (JSON)
          </label>
          <textarea
            value={configText}
            onChange={(e) => setConfigText(e.target.value)}
            rows={10}
            className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 font-mono text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
          />
        </div>

        {error && <p className="text-sm text-red-400">{error}</p>}

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={saving}
            className="rounded bg-brand-500 px-4 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-50"
          >
            {saving ? "Creating…" : "Create Model"}
          </button>
          <a href="/model" className="rounded border border-gray-700 px-4 py-2 text-sm text-gray-400 hover:text-white">
            Cancel
          </a>
        </div>
      </form>
    </div>
  );
}
