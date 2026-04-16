"use client";

import { useState } from "react";
import { useRouter } from "next/navigation";

// ---------------------------------------------------------------------------
// Types
// ---------------------------------------------------------------------------

type ConfigField = {
  key: string;
  label: string;
  type: "integer" | "number" | "boolean" | "select";
  default: number | string | boolean;
  options?: { value: string; label: string }[];
};

type ModelTemplate = {
  id: string;
  architecture: string;
  name: string;
  description: string;
  tags: string[];
  config: Record<string, unknown>;
  fields: ConfigField[];
};

// ---------------------------------------------------------------------------
// Architecture templates
// ---------------------------------------------------------------------------

const LSTM_FIELDS: ConfigField[] = [
  { key: "input_size",    label: "Input Size",    type: "integer", default: 1 },
  { key: "hidden_size",   label: "Hidden Size",   type: "integer", default: 128 },
  { key: "num_layers",    label: "Num Layers",    type: "integer", default: 2 },
  { key: "output_size",   label: "Output Size",   type: "integer", default: 1 },
  { key: "dropout",       label: "Dropout",       type: "number",  default: 0.1 },
  { key: "bidirectional", label: "Bidirectional", type: "boolean", default: false },
];

const TRANSFORMER_FIELDS: ConfigField[] = [
  { key: "d_model",            label: "Model Dimension",  type: "integer", default: 64 },
  { key: "nhead",              label: "Attention Heads",  type: "integer", default: 4 },
  { key: "num_encoder_layers", label: "Encoder Layers",   type: "integer", default: 2 },
  { key: "num_decoder_layers", label: "Decoder Layers",   type: "integer", default: 2 },
  { key: "dim_feedforward",    label: "Feedforward Dim",  type: "integer", default: 256 },
  { key: "dropout",            label: "Dropout",          type: "number",  default: 0.1 },
  { key: "input_size",         label: "Input Size",       type: "integer", default: 1 },
  { key: "output_size",        label: "Output Size",      type: "integer", default: 1 },
];

const RL_FIELDS: ConfigField[] = [
  {
    key: "algorithm", label: "Algorithm", type: "select", default: "ppo",
    options: [
      { value: "ppo", label: "PPO — Proximal Policy Optimization" },
      { value: "dqn", label: "DQN — Deep Q-Network" },
      { value: "a2c", label: "A2C — Advantage Actor-Critic" },
    ],
  },
  {
    key: "policy", label: "Policy Network", type: "select", default: "MlpPolicy",
    options: [
      { value: "MlpPolicy", label: "MlpPolicy (fully connected)" },
      { value: "CnnPolicy", label: "CnnPolicy (convolutional)" },
    ],
  },
  { key: "learning_rate", label: "Learning Rate", type: "number", default: 0.0003 },
];

const TEMPLATES: ModelTemplate[] = [
  // ── Scratch ──────────────────────────────────────────────────────────────
  {
    id: "scratch",
    architecture: "lstm",
    name: "",
    description: "Choose architecture and set all parameters manually.",
    tags: [],
    config: {},
    fields: [],
  },

  // ── LSTM variants ─────────────────────────────────────────────────────────
  {
    id: "lstm",
    architecture: "lstm",
    name: "LSTM Predictor",
    description: "LSTM autoencoder/predictor suited for short-term price movement forecasting.",
    tags: ["forecasting", "sequential"],
    config: { input_size: 1, hidden_size: 128, num_layers: 2, output_size: 1, dropout: 0.1, bidirectional: false },
    fields: LSTM_FIELDS,
  },
  {
    id: "lstm_bidirectional",
    architecture: "lstm",
    name: "Bidirectional LSTM",
    description: "Reads the sequence both forward and backward for richer pattern extraction. Best for feature generation, not live inference.",
    tags: ["forecasting", "sequential", "feature-extraction"],
    config: { input_size: 1, hidden_size: 256, num_layers: 2, output_size: 1, dropout: 0.2, bidirectional: true },
    fields: LSTM_FIELDS,
  },
  {
    id: "lstm_deep",
    architecture: "lstm",
    name: "Deep LSTM Stack",
    description: "Four LSTM layers with higher dropout to capture long-range market regime patterns.",
    tags: ["forecasting", "sequential", "deep"],
    config: { input_size: 1, hidden_size: 64, num_layers: 4, output_size: 1, dropout: 0.3, bidirectional: false },
    fields: LSTM_FIELDS,
  },
  {
    id: "lstm_lightweight",
    architecture: "lstm",
    name: "Lightweight LSTM",
    description: "Single-layer, small hidden size — fast training and inference when data or compute is limited.",
    tags: ["forecasting", "sequential", "fast"],
    config: { input_size: 1, hidden_size: 32, num_layers: 1, output_size: 1, dropout: 0.0, bidirectional: false },
    fields: LSTM_FIELDS,
  },

  // ── Transformer variants ──────────────────────────────────────────────────
  {
    id: "seq2seq_transformer",
    architecture: "seq2seq_transformer",
    name: "Seq2Seq Transformer",
    description: "Transformer encoder-decoder for multi-step sequence prediction with attention.",
    tags: ["forecasting", "transformer", "multi-step"],
    config: { d_model: 64, nhead: 4, num_encoder_layers: 2, num_decoder_layers: 2, dim_feedforward: 256, dropout: 0.1, input_size: 1, output_size: 1 },
    fields: TRANSFORMER_FIELDS,
  },
  {
    id: "transformer_large",
    architecture: "seq2seq_transformer",
    name: "Large Transformer",
    description: "High-capacity model for multi-feature or multi-asset inputs requiring more expressive representations.",
    tags: ["forecasting", "transformer", "multi-step", "high-capacity"],
    config: { d_model: 128, nhead: 8, num_encoder_layers: 4, num_decoder_layers: 4, dim_feedforward: 512, dropout: 0.1, input_size: 1, output_size: 1 },
    fields: TRANSFORMER_FIELDS,
  },
  {
    id: "transformer_tiny",
    architecture: "seq2seq_transformer",
    name: "Tiny Transformer",
    description: "Minimal footprint — fast training, low-latency inference. Good baseline to compare against before scaling up.",
    tags: ["forecasting", "transformer", "fast"],
    config: { d_model: 32, nhead: 2, num_encoder_layers: 1, num_decoder_layers: 1, dim_feedforward: 128, dropout: 0.1, input_size: 1, output_size: 1 },
    fields: TRANSFORMER_FIELDS,
  },

  // ── CNN-LSTM ──────────────────────────────────────────────────────────────
  {
    id: "cnn_lstm",
    architecture: "cnn_lstm",
    name: "CNN + LSTM Hybrid",
    description: "1-D CNN extracts local chart patterns (supports, resistances, candle clusters); LSTM captures the resulting temporal sequence.",
    tags: ["forecasting", "hybrid", "pattern-detection"],
    config: { cnn_filters: 64, kernel_size: 3, cnn_layers: 2, lstm_hidden: 128, lstm_layers: 1, dropout: 0.2, input_size: 1, output_size: 1 },
    fields: [
      { key: "cnn_filters",  label: "CNN Filters",   type: "integer", default: 64 },
      { key: "kernel_size",  label: "Kernel Size",   type: "integer", default: 3 },
      { key: "cnn_layers",   label: "CNN Layers",    type: "integer", default: 2 },
      { key: "lstm_hidden",  label: "LSTM Hidden",   type: "integer", default: 128 },
      { key: "lstm_layers",  label: "LSTM Layers",   type: "integer", default: 1 },
      { key: "dropout",      label: "Dropout",       type: "number",  default: 0.2 },
      { key: "input_size",   label: "Input Size",    type: "integer", default: 1 },
      { key: "output_size",  label: "Output Size",   type: "integer", default: 1 },
    ],
  },

  // ── TCN ───────────────────────────────────────────────────────────────────
  {
    id: "tcn",
    architecture: "tcn",
    name: "Temporal Conv Network",
    description: "Dilated causal convolutions with residual connections. Parallelizable — trains faster than LSTM with comparable accuracy on many financial series.",
    tags: ["forecasting", "convolutional", "fast"],
    config: { num_channels: 64, num_levels: 4, kernel_size: 3, dropout: 0.2, input_size: 1, output_size: 1 },
    fields: [
      { key: "num_channels", label: "Channels",      type: "integer", default: 64 },
      { key: "num_levels",   label: "Num Levels",    type: "integer", default: 4 },
      { key: "kernel_size",  label: "Kernel Size",   type: "integer", default: 3 },
      { key: "dropout",      label: "Dropout",       type: "number",  default: 0.2 },
      { key: "input_size",   label: "Input Size",    type: "integer", default: 1 },
      { key: "output_size",  label: "Output Size",   type: "integer", default: 1 },
    ],
  },

  // ── VAE ───────────────────────────────────────────────────────────────────
  {
    id: "vae",
    architecture: "vae",
    name: "Seq-VAE",
    description: "Encodes market history into a probabilistic latent space. Useful for regime detection, anomaly scoring, and uncertainty-aware forecasting.",
    tags: ["generative", "anomaly-detection", "probabilistic"],
    config: { latent_dim: 32, encoder_hidden: 128, decoder_hidden: 128, encoder_layers: 2, dropout: 0.1, input_size: 1, output_size: 1 },
    fields: [
      { key: "latent_dim",     label: "Latent Dim",      type: "integer", default: 32 },
      { key: "encoder_hidden", label: "Encoder Hidden",  type: "integer", default: 128 },
      { key: "decoder_hidden", label: "Decoder Hidden",  type: "integer", default: 128 },
      { key: "encoder_layers", label: "Encoder Layers",  type: "integer", default: 2 },
      { key: "dropout",        label: "Dropout",         type: "number",  default: 0.1 },
      { key: "input_size",     label: "Input Size",      type: "integer", default: 1 },
      { key: "output_size",    label: "Output Size",     type: "integer", default: 1 },
    ],
  },

  // ── N-BEATS ───────────────────────────────────────────────────────────────
  {
    id: "nbeats",
    architecture: "nbeats",
    name: "N-BEATS",
    description: "Pure MLP forecaster with backcast residual connections. No recurrence, no attention — fast training and interpretable trend/seasonality decomposition.",
    tags: ["forecasting", "interpretable", "MLP"],
    config: { hidden_units: 256, nb_blocks: 3, theta_dim: 64, obs_len: 60, input_size: 1, output_size: 1 },
    fields: [
      { key: "hidden_units", label: "Hidden Units",  type: "integer", default: 256 },
      { key: "nb_blocks",    label: "Num Blocks",    type: "integer", default: 3 },
      { key: "theta_dim",    label: "Theta Dim",     type: "integer", default: 64 },
      { key: "obs_len",      label: "Obs Window",    type: "integer", default: 60 },
      { key: "input_size",   label: "Input Size",    type: "integer", default: 1 },
      { key: "output_size",  label: "Output Size",   type: "integer", default: 1 },
    ],
  },

  // ── TimeGAN ───────────────────────────────────────────────────────────────
  {
    id: "timegan",
    architecture: "timegan",
    name: "TimeGAN",
    description: "Generative adversarial network for producing synthetic financial time-series data.",
    tags: ["generative", "synthetic-data", "GAN"],
    config: { hidden_dim: 24, num_layer: 3, seq_len: 24, batch_size: 128 },
    fields: [
      { key: "hidden_dim",  label: "Hidden Dim",      type: "integer", default: 24 },
      { key: "num_layer",   label: "Num Layers",      type: "integer", default: 3 },
      { key: "seq_len",     label: "Sequence Length", type: "integer", default: 24 },
      { key: "batch_size",  label: "Batch Size",      type: "integer", default: 128 },
    ],
  },

  // ── RL Agent variants ─────────────────────────────────────────────────────
  {
    id: "rl_ppo",
    architecture: "rl_agent",
    name: "PPO Agent",
    description: "Proximal Policy Optimization — stable on-policy algorithm; good default for continuous action spaces.",
    tags: ["reinforcement-learning", "adaptive", "on-policy"],
    config: { algorithm: "ppo", policy: "MlpPolicy", learning_rate: 0.0003 },
    fields: RL_FIELDS,
  },
  {
    id: "rl_dqn",
    architecture: "rl_agent",
    name: "DQN Agent",
    description: "Deep Q-Network — off-policy Q-learning for discrete action spaces (buy / hold / sell).",
    tags: ["reinforcement-learning", "adaptive", "off-policy"],
    config: { algorithm: "dqn", policy: "MlpPolicy", learning_rate: 0.0001 },
    fields: RL_FIELDS,
  },
  {
    id: "rl_a2c",
    architecture: "rl_agent",
    name: "A2C Agent",
    description: "Advantage Actor-Critic — faster convergence than PPO, useful for quick policy exploration with lower compute.",
    tags: ["reinforcement-learning", "adaptive", "on-policy"],
    config: { algorithm: "a2c", policy: "MlpPolicy", learning_rate: 0.0007 },
    fields: RL_FIELDS,
  },
];

// ---------------------------------------------------------------------------
// Tag colors
// ---------------------------------------------------------------------------

const TAG_COLORS: Record<string, string> = {
  forecasting:              "bg-blue-900 text-blue-300",
  sequential:               "bg-purple-900 text-purple-300",
  "multi-step":             "bg-sky-900 text-sky-300",
  transformer:              "bg-indigo-900 text-indigo-300",
  generative:               "bg-green-900 text-green-300",
  "synthetic-data":         "bg-teal-900 text-teal-300",
  GAN:                      "bg-yellow-900 text-yellow-300",
  "reinforcement-learning": "bg-orange-900 text-orange-300",
  adaptive:                 "bg-pink-900 text-pink-300",
  "feature-extraction":     "bg-violet-900 text-violet-300",
  deep:                     "bg-slate-800 text-slate-300",
  fast:                     "bg-emerald-900 text-emerald-300",
  "high-capacity":          "bg-red-900 text-red-300",
  hybrid:                   "bg-cyan-900 text-cyan-300",
  "pattern-detection":      "bg-amber-900 text-amber-300",
  convolutional:            "bg-lime-900 text-lime-300",
  "anomaly-detection":      "bg-rose-900 text-rose-300",
  probabilistic:            "bg-fuchsia-900 text-fuchsia-300",
  interpretable:            "bg-stone-700 text-stone-300",
  MLP:                      "bg-zinc-700 text-zinc-300",
  "on-policy":              "bg-orange-800 text-orange-300",
  "off-policy":             "bg-yellow-800 text-yellow-300",
};

function TagBadge({ tag }: { tag: string }) {
  const cls = TAG_COLORS[tag] ?? "bg-gray-700 text-gray-300";
  return <span className={`rounded px-1.5 py-0.5 text-[10px] font-medium ${cls}`}>{tag}</span>;
}

// ---------------------------------------------------------------------------
// Config field editor
// ---------------------------------------------------------------------------

function ConfigEditor({
  fields,
  values,
  onChange,
}: {
  fields: ConfigField[];
  values: Record<string, unknown>;
  onChange: (key: string, value: unknown) => void;
}) {
  if (fields.length === 0) return null;
  return (
    <div className="grid grid-cols-2 gap-4">
      {fields.map((field) => (
        <div key={field.key}>
          <label className="mb-1 block text-sm text-gray-400">{field.label}</label>
          {field.type === "select" || field.type === "boolean" ? (
            <select
              value={String(values[field.key] ?? field.default)}
              onChange={(e) => {
                const raw = e.target.value;
                onChange(field.key, field.type === "boolean" ? raw === "true" : raw);
              }}
              className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            >
              {field.type === "boolean" ? (
                <>
                  <option value="false">False</option>
                  <option value="true">True</option>
                </>
              ) : (
                field.options!.map((o) => (
                  <option key={o.value} value={o.value}>{o.label}</option>
                ))
              )}
            </select>
          ) : (
            <input
              type="number"
              step={field.type === "number" ? "any" : "1"}
              value={String(values[field.key] ?? field.default)}
              onChange={(e) => {
                const v = field.type === "integer"
                  ? parseInt(e.target.value, 10)
                  : parseFloat(e.target.value);
                onChange(field.key, isNaN(v) ? field.default : v);
              }}
              className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            />
          )}
        </div>
      ))}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Template groups
// ---------------------------------------------------------------------------

type TemplateGroup = { label: string; ids: string[] };

const TEMPLATE_GROUPS: TemplateGroup[] = [
  { label: "LSTM",                   ids: ["lstm", "lstm_bidirectional", "lstm_deep", "lstm_lightweight"] },
  { label: "Transformer",            ids: ["seq2seq_transformer", "transformer_large", "transformer_tiny"] },
  { label: "Convolutional",          ids: ["cnn_lstm", "tcn"] },
  { label: "Generative & Latent",    ids: ["timegan", "vae"] },
  { label: "Interpretable",          ids: ["nbeats"] },
  { label: "Reinforcement Learning", ids: ["rl_ppo", "rl_dqn", "rl_a2c"] },
];

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function NewModelPage() {
  const router = useRouter();
  const [step, setStep] = useState<"pick" | "edit">("pick");
  const [templateId, setTemplateId] = useState<string | null>(null);
  const [name, setName] = useState("");
  const [description, setDescription] = useState("");
  const [config, setConfig] = useState<Record<string, unknown>>({});
  const [error, setError] = useState<string | null>(null);
  const [saving, setSaving] = useState(false);

  // active template for the editor step (may differ from templateId when scratch switches arch)
  const [activeArchId, setActiveArchId] = useState<string>("lstm");

  function selectTemplate(tpl: ModelTemplate) {
    setTemplateId(tpl.id);
    const archId = tpl.id === "scratch" ? "lstm" : tpl.id;
    setActiveArchId(archId);
    if (tpl.id !== "scratch" && !name) {
      setName(tpl.name);
      setDescription(tpl.description);
    }
    const defaultTpl = TEMPLATES.find((t) => t.id === archId)!;
    setConfig({ ...defaultTpl.config });
    setStep("edit");
  }

  function handleArchSwitch(archId: string) {
    setActiveArchId(archId);
    const tpl = TEMPLATES.find((t) => t.id === archId)!;
    setConfig({ ...tpl.config });
  }

  function handleConfigChange(key: string, value: unknown) {
    setConfig((prev) => ({ ...prev, [key]: value }));
  }

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setError(null);
    setSaving(true);
    try {
      const archTpl = TEMPLATES.find((t) => t.id === activeArchId)!;
      const res = await fetch("/api/v1/models", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name,
          description,
          architecture: archTpl.architecture,
          config,
        }),
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

  const isScratch = templateId === "scratch";
  const selectedTemplate = TEMPLATES.find((t) => t.id === templateId);
  const activeTemplate = TEMPLATES.find((t) => t.id === activeArchId)!;

  // ---- Step 1: architecture picker ----
  if (step === "pick") {
    return (
      <div className="max-w-4xl space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-white">New ML Model</h1>
          <p className="mt-1 text-sm text-gray-400">
            Choose an architecture to start with — you can adjust all parameters after.
          </p>
        </div>

        {/* Scratch card */}
        <div>
          <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">From Scratch</h2>
          <button
            onClick={() => selectTemplate(TEMPLATES[0])}
            className="w-full rounded-lg border border-dashed border-gray-600 bg-gray-900 px-5 py-4 text-left transition hover:border-brand-500 hover:bg-gray-800"
          >
            <p className="font-medium text-white">Blank Model</p>
            <p className="mt-0.5 text-sm text-gray-400">Choose architecture and set all parameters manually.</p>
          </button>
        </div>

        {/* Template cards grouped by type */}
        {TEMPLATE_GROUPS.map((group) => {
          const groupTemplates = group.ids
            .map((id) => TEMPLATES.find((t) => t.id === id)!)
            .filter(Boolean);
          return (
            <div key={group.label}>
              <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-gray-500">{group.label}</h2>
              <div className="grid grid-cols-1 gap-3 sm:grid-cols-2">
                {groupTemplates.map((tpl) => (
                  <button
                    key={tpl.id}
                    onClick={() => selectTemplate(tpl)}
                    className="rounded-lg border border-gray-700 bg-gray-900 px-5 py-4 text-left transition hover:border-brand-500 hover:bg-gray-800"
                  >
                    <p className="font-medium text-white">{tpl.name}</p>
                    <p className="mt-1 text-sm text-gray-400">{tpl.description}</p>
                    {tpl.tags.length > 0 && (
                      <div className="mt-3 flex flex-wrap gap-1">
                        {tpl.tags.map((tag) => <TagBadge key={tag} tag={tag} />)}
                      </div>
                    )}
                  </button>
                ))}
              </div>
            </div>
          );
        })}
      </div>
    );
  }

  // ---- Step 2: config editor ----
  return (
    <div className="max-w-3xl space-y-6">
      <div className="flex items-center gap-3">
        <button
          onClick={() => setStep("pick")}
          className="text-sm text-gray-400 hover:text-white"
        >
          ← Back
        </button>
        <h1 className="text-2xl font-semibold text-white">
          {!isScratch && selectedTemplate ? (
            <>New Model <span className="text-base font-normal text-gray-400">— {selectedTemplate.name}</span></>
          ) : (
            "New ML Model"
          )}
        </h1>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        <div className="space-y-4">
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
            <label className="mb-1 block text-sm text-gray-400">Description</label>
            <input
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              placeholder="Optional description"
            />
          </div>

          {/* Architecture selector — always visible so users can switch */}
          <div>
            <label className="mb-1 block text-sm text-gray-400">Architecture</label>
            <select
              value={activeArchId}
              onChange={(e) => handleArchSwitch(e.target.value)}
              className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-2 text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
            >
              {TEMPLATES.filter((t) => t.id !== "scratch").map((t) => (
                <option key={t.id} value={t.id}>{t.name}</option>
              ))}
            </select>
          </div>
        </div>

        {activeTemplate.fields.length > 0 && (
          <div>
            <label className="mb-2 block text-sm text-gray-400">Architecture Config</label>
            <div className="rounded border border-gray-700 bg-gray-950 p-4">
              <ConfigEditor
                fields={activeTemplate.fields}
                values={config}
                onChange={handleConfigChange}
              />
            </div>
          </div>
        )}

        {error && <p className="text-sm text-red-400">{error}</p>}

        <div className="flex gap-3">
          <button
            type="submit"
            disabled={saving}
            className="rounded bg-brand-500 px-4 py-2 text-sm text-white hover:bg-sky-400 disabled:opacity-50"
          >
            {saving ? "Creating…" : "Create Model"}
          </button>
          <button
            type="button"
            onClick={() => setStep("pick")}
            className="rounded border border-gray-700 px-4 py-2 text-sm text-gray-400 hover:text-white"
          >
            Change Template
          </button>
          <a
            href="/model"
            className="rounded border border-gray-700 px-4 py-2 text-sm text-gray-400 hover:text-white"
          >
            Cancel
          </a>
        </div>
      </form>
    </div>
  );
}
