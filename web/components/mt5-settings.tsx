"use client";

import { useState } from "react";
import useSWR from "swr";
import { fetcher } from "@/lib/fetcher";
import { useToast } from "@/lib/toast";

interface MT5Account {
  login: number;
  server: string;
  company: string;
  currency: string;
  balance: number;
  equity: number;
  profit: number;
  margin: number;
  margin_free: number;
  margin_level: number;
  leverage: number;
  trade_mode: number;
}

interface MT5Tick {
  symbol: string;
  time: string;
  bid: number;
  ask: number;
  spread: number;
}

interface MT5AccountStatus {
  connected: boolean;
  account: MT5Account | null;
  tick: MT5Tick | null;
  dry_run: boolean;
}

interface MT5Config {
  login: number | null;
  password_masked: string;
  server: string | null;
  path: string | null;
  magic: number;
  default_lot: number;
  slippage: number;
  dry_run: boolean;
}

export function MT5Settings({ onClose }: { onClose?: () => void }) {
  const { toast } = useToast();
  const { data: status, mutate: mutateStatus } = useSWR<MT5AccountStatus>("/api/v1/mt5/account", fetcher, {
    refreshInterval: 5000,
  });
  const { data: config, mutate: mutateConfig } = useSWR<MT5Config>("/api/v1/mt5/config", fetcher);

  const [login, setLogin] = useState<string>("");
  const [password, setPassword] = useState<string>("");
  const [server, setServer] = useState<string>("");
  const [path, setPath] = useState<string>("");
  const [defaultLot, setDefaultLot] = useState<number>(0.1);
  const [dryRun, setDryRun] = useState<boolean>(false);

  const [saving, setSaving] = useState(false);
  const [testing, setTesting] = useState(false);
  const [testResult, setTestResult] = useState<{ success: boolean; message: string } | null>(null);

  // Sync state when config loads
  useState(() => {
    if (config) {
      if (config.login) setLogin(String(config.login));
      if (config.server) setServer(config.server);
      if (config.path) setPath(config.path);
      if (config.default_lot) setDefaultLot(config.default_lot);
      if (config.dry_run !== undefined) setDryRun(config.dry_run);
    }
  });

  async function handleSave(e: React.FormEvent) {
    e.preventDefault();
    setSaving(true);
    setTestResult(null);

    try {
      const payload: any = {
        server: server || null,
        path: path || null,
        default_lot: Number(defaultLot) || 0.1,
        dry_run: dryRun,
      };
      if (login) payload.login = Number(login);
      if (password) payload.password = password;

      const res = await fetch("/api/v1/mt5/config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      if (!res.ok) throw new Error("Failed to save MT5 configuration");
      const data = await res.json();
      toast("MT5 configuration saved successfully", "success");
      setPassword(""); // clear password field
      mutateConfig();
      mutateStatus();
    } catch (err: any) {
      toast(err.message || "Failed to save configuration", "error");
    } finally {
      setSaving(false);
    }
  }

  async function handleTest() {
    setTesting(true);
    setTestResult(null);

    try {
      const payload: any = {
        server: server || config?.server || null,
        path: path || config?.path || null,
      };
      if (login) payload.login = Number(login);
      else if (config?.login) payload.login = config.login;
      if (password) payload.password = password;

      const res = await fetch("/api/v1/mt5/test", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });

      const data = await res.json();
      setTestResult(data);
      if (data.success) {
        toast("MT5 connection test successful!", "success");
        mutateStatus();
      } else {
        toast(`MT5 connection test failed: ${data.message}`, "error");
      }
    } catch (err: any) {
      setTestResult({ success: false, message: err.message || "Connection error" });
      toast("Test failed", "error");
    } finally {
      setTesting(false);
    }
  }

  return (
    <div className="space-y-6">
      {/* Live Status Card */}
      <div className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 backdrop-blur-sm">
        <div className="flex items-center justify-between pb-4 border-b border-gray-800">
          <div className="flex items-center gap-3">
            <div className={`h-3 w-3 rounded-full ${status?.connected ? "bg-emerald-400 shadow-sm shadow-emerald-500/50 animate-pulse" : "bg-red-500"}`} />
            <div>
              <h3 className="font-medium text-white text-sm">MetaTrader 5 (MT5) Connection</h3>
              <p className="text-xs text-gray-400">
                {status?.connected
                  ? `${status.account?.company} · ${status.account?.server}`
                  : "Disconnected or not initialized"}
              </p>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {status?.dry_run && (
              <span className="rounded bg-amber-900/40 border border-amber-700/50 px-2 py-0.5 text-xs text-amber-300 font-medium">
                DRY RUN MODE
              </span>
            )}
            <span
              className={`rounded px-2.5 py-1 text-xs font-medium border ${
                status?.connected
                  ? "bg-emerald-950/60 border-emerald-800/60 text-emerald-300"
                  : "bg-red-950/60 border-red-800/60 text-red-300"
              }`}
            >
              {status?.connected ? "ONLINE" : "OFFLINE"}
            </span>
          </div>
        </div>

        {status?.connected && status.account && (
          <div className="mt-4 grid grid-cols-2 gap-4 sm:grid-cols-4 text-xs font-mono">
            <div className="rounded-lg bg-gray-950/50 p-2.5 border border-gray-800/60">
              <span className="text-gray-500 block text-[10px] font-sans">Login Account</span>
              <span className="text-gray-200 font-semibold">{status.account.login}</span>
            </div>
            <div className="rounded-lg bg-gray-950/50 p-2.5 border border-gray-800/60">
              <span className="text-gray-500 block text-[10px] font-sans">Balance</span>
              <span className="text-emerald-400 font-semibold">{status.account.balance.toLocaleString()} {status.account.currency}</span>
            </div>
            <div className="rounded-lg bg-gray-950/50 p-2.5 border border-gray-800/60">
              <span className="text-gray-500 block text-[10px] font-sans">Free Margin / Lev</span>
              <span className="text-gray-200">{status.account.margin_free.toLocaleString()} · 1:{status.account.leverage}</span>
            </div>
            <div className="rounded-lg bg-gray-950/50 p-2.5 border border-gray-800/60">
              <span className="text-gray-500 block text-[10px] font-sans">USD/JPY Live Rate</span>
              <span className="text-sky-300 font-semibold">
                {status.tick ? `${status.tick.bid.toFixed(3)} / ${status.tick.ask.toFixed(3)}` : "—"}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Configuration Form */}
      <form onSubmit={handleSave} className="rounded-xl border border-gray-800 bg-gray-900/60 p-5 space-y-4">
        <h3 className="text-sm font-medium text-white border-b border-gray-800 pb-3 flex items-center justify-between">
          <span>Account Credentials & Parameters</span>
          <span className="text-xs font-normal text-gray-400">※ PCでMT5が起動中なら空欄でも自動接続可能</span>
        </h3>

        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          <div>
            <label className="block text-xs text-gray-400 mb-1">Login (Account Number)</label>
            <input
              type="text"
              value={login || config?.login || ""}
              onChange={(e) => setLogin(e.target.value)}
              placeholder="e.g. 12345678"
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Password</label>
            <input
              type="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              placeholder={config?.password_masked || "Leave blank to keep current password"}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Server Name</label>
            <input
              type="text"
              value={server || config?.server || ""}
              onChange={(e) => setServer(e.target.value)}
              placeholder="e.g. OANDA-Japan MT5 Live"
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Terminal Executable Path (Optional)</label>
            <input
              type="text"
              value={path || config?.path || ""}
              onChange={(e) => setPath(e.target.value)}
              placeholder="e.g. C:\Program Files\MetaTrader 5\terminal64.exe"
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
            />
          </div>

          <div>
            <label className="block text-xs text-gray-400 mb-1">Default Order Volume (Lots)</label>
            <input
              type="number"
              step="0.01"
              min="0.01"
              value={defaultLot || config?.default_lot || 0.1}
              onChange={(e) => setDefaultLot(parseFloat(e.target.value))}
              className="w-full rounded-lg border border-gray-700 bg-gray-950 px-3 py-2 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
            />
            <span className="text-[10px] text-gray-500 mt-1 block">1 standard lot = 100,000 units (0.1 = 10,000 units)</span>
          </div>

          <div className="flex items-center pt-5">
            <label className="relative flex items-center gap-2 cursor-pointer text-xs text-gray-300">
              <input
                type="checkbox"
                checked={dryRun || config?.dry_run || false}
                onChange={(e) => setDryRun(e.target.checked)}
                className="rounded border-gray-700 bg-gray-950 text-brand-500 h-4 w-4"
              />
              <span>Dry Run Mode (Simulate without sending real orders to broker)</span>
            </label>
          </div>
        </div>

        {testResult && (
          <div
            className={`p-3 rounded-lg text-xs font-mono border ${
              testResult.success
                ? "bg-emerald-950/40 border-emerald-800/60 text-emerald-300"
                : "bg-red-950/40 border-red-800/60 text-red-300"
            }`}
          >
            {testResult.success ? "✓ " : "✗ "}
            {testResult.message}
          </div>
        )}

        <div className="flex items-center justify-between pt-3 border-t border-gray-800">
          <button
            type="button"
            onClick={handleTest}
            disabled={testing}
            className="rounded-lg border border-gray-700 bg-gray-800 px-4 py-2 text-xs font-medium text-gray-300 hover:bg-gray-700 hover:text-white transition disabled:opacity-50"
          >
            {testing ? "Testing Connection…" : "⚡ Test Connection"}
          </button>

          <div className="flex items-center gap-3">
            {onClose && (
              <button
                type="button"
                onClick={onClose}
                className="rounded-lg border border-gray-700 px-4 py-2 text-xs font-medium text-gray-400 hover:bg-gray-800 hover:text-white transition"
              >
                Close
              </button>
            )}
            <button
              type="submit"
              disabled={saving}
              className="rounded-lg bg-brand-500 px-5 py-2 text-xs font-medium text-white hover:bg-brand-600 transition shadow-sm disabled:opacity-50"
            >
              {saving ? "Saving…" : "Save Configuration"}
            </button>
          </div>
        </div>
      </form>
    </div>
  );
}
