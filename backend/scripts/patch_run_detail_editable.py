path = r"c:\Users\ksato\workspace\finance\algoforge\web\app\strategy\[id]\runs\[run_id]\page.tsx"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Update states and openDeployModal
old_states = '''  const [stopping, setStopping] = useState(false);
  const { toast } = useToast();
  const [deployModalMode, setDeployModalMode] = useState<"paper" | "live" | null>(null);
  const [deploying, setDeploying] = useState(false);
  const { data: mt5Status } = useSWR("/api/v1/mt5/account", fetcher);

  async function handleDeploy(targetMode: "paper" | "live") {
    setDeploying(true);
    try {
      const body: any = {
        mode: targetMode,
        risk_override: run?.risk_override || null,
        window_size: run?.window_size || null,
        starting_capital: run?.starting_capital || null,
      };
      const res = await apiFetch(`/api/v1/strategies/${strategyId}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast(err.detail || err.error?.message || "Failed to start run", "error");
        return;
      }
      const json = await res.json();
      const newRunId = json.data.id;
      toast(`${targetMode === "live" ? "Live" : "Paper"} run #${newRunId} started!`, "success");
      setDeployModalMode(null);
      router.push(`/strategy/${strategyId}/runs/${newRunId}`);
    } catch (e: any) {
      toast(e.message || "Failed to start run", "error");
    } finally {
      setDeploying(false);
    }
  }'''

new_states = '''  const [stopping, setStopping] = useState(false);
  const { toast } = useToast();
  const [deployModalMode, setDeployModalMode] = useState<"paper" | "live" | null>(null);
  const [deploying, setDeploying] = useState(false);
  const { data: mt5Status } = useSWR("/api/v1/mt5/account", fetcher);

  // Editable parameters for deployment
  const [overrideLotSize, setOverrideLotSize] = useState<number>(0.1);
  const [overrideSlPips, setOverrideSlPips] = useState<string>("25");
  const [overrideTpPips, setOverrideTpPips] = useState<string>("100");
  const [overrideTolPips, setOverrideTolPips] = useState<string>("10");
  const [overrideCapital, setOverrideCapital] = useState<number>(1000000);
  const [overrideSlPct, setOverrideSlPct] = useState<number>(2.0);
  const [overrideTpPct, setOverrideTpPct] = useState<number>(4.0);

  function resetDeployParams() {
    const p = strategy?.definition?.parameters || {};
    const ro = run?.risk_override || {};
    setOverrideLotSize(p.lot_size !== undefined ? Number(p.lot_size) : 0.1);
    setOverrideSlPips(p.sl_pips !== undefined ? String(p.sl_pips) : "25");
    setOverrideTpPips(p.tp_pips !== undefined ? String(p.tp_pips) : "100");
    setOverrideTolPips(p.tolerance_pips !== undefined ? String(p.tolerance_pips) : "10");
    setOverrideCapital(run?.starting_capital || 1000000);
    setOverrideSlPct(ro.sl_pct !== undefined ? ro.sl_pct * 100 : 2.0);
    setOverrideTpPct(ro.tp_pct !== undefined ? ro.tp_pct * 100 : 4.0);
  }

  function openDeployModal(mode: "paper" | "live") {
    resetDeployParams();
    setDeployModalMode(mode);
  }

  async function handleDeploy(targetMode: "paper" | "live") {
    setDeploying(true);
    try {
      const isAnalyst = !!strategy?.definition?.strategy_type?.startsWith("analyst_");
      const riskOverridePayload: Record<string, any> = {
        ...(run?.risk_override || {}),
        lot_size: overrideLotSize,
      };

      if (isAnalyst) {
        riskOverridePayload.sl_pips = overrideSlPips.trim() !== "" ? Number(overrideSlPips) : null;
        riskOverridePayload.tp_pips = overrideTpPips.trim() !== "" ? (isNaN(Number(overrideTpPips)) ? overrideTpPips : Number(overrideTpPips)) : null;
        riskOverridePayload.tolerance_pips = overrideTolPips.trim() !== "" ? Number(overrideTolPips) : 0;
      } else {
        riskOverridePayload.sl_pct = overrideSlPct / 100;
        riskOverridePayload.tp_pct = overrideTpPct / 100;
      }

      const body: any = {
        mode: targetMode,
        risk_override: riskOverridePayload,
        window_size: run?.window_size || null,
        starting_capital: overrideCapital > 0 ? overrideCapital : null,
      };

      const res = await apiFetch(`/api/v1/strategies/${strategyId}/runs`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        toast(err.detail || err.error?.message || "Failed to start run", "error");
        return;
      }
      const json = await res.json();
      const newRunId = json.data.id;
      toast(`${targetMode === "live" ? "Live" : "Paper"} run #${newRunId} started!`, "success");
      setDeployModalMode(null);
      router.push(`/strategy/${strategyId}/runs/${newRunId}`);
    } catch (e: any) {
      toast(e.message || "Failed to start run", "error");
    } finally {
      setDeploying(false);
    }
  }'''

assert old_states in text, "old_states target not found"
text = text.replace(old_states, new_states, 1)

# 2. Update button onClick calls to openDeployModal
old_buttons = '''              <button
                onClick={() => setDeployModalMode("paper")}
                className="rounded border border-sky-600/70 bg-sky-950/40 hover:bg-sky-900/60 text-sky-300 px-3 py-1.5 text-xs font-medium flex items-center gap-1.5 transition-colors shadow-sm"
                title="Start paper trading with this backtest's settings"
              >
                <span>📝</span> Paper Trade
              </button>
              <button
                onClick={() => setDeployModalMode("live")}
                className="rounded border border-emerald-600/70 bg-emerald-950/40 hover:bg-emerald-900/60 text-emerald-300 px-3 py-1.5 text-xs font-medium flex items-center gap-1.5 transition-colors shadow-sm"
                title="Start live trading on MT5 broker using this backtest's settings"
              >
                <span>⚡</span> Live Trade
              </button>'''

new_buttons = '''              <button
                onClick={() => openDeployModal("paper")}
                className="rounded border border-sky-600/70 bg-sky-950/40 hover:bg-sky-900/60 text-sky-300 px-3 py-1.5 text-xs font-medium flex items-center gap-1.5 transition-colors shadow-sm"
                title="Configure and start paper trading using this backtest's settings"
              >
                <span>📝</span> Paper Trade
              </button>
              <button
                onClick={() => openDeployModal("live")}
                className="rounded border border-emerald-600/70 bg-emerald-950/40 hover:bg-emerald-900/60 text-emerald-300 px-3 py-1.5 text-xs font-medium flex items-center gap-1.5 transition-colors shadow-sm"
                title="Configure and start live real trading on MT5 using this backtest's settings"
              >
                <span>⚡</span> Live Trade
              </button>'''

assert old_buttons in text, "old_buttons target not found"
text = text.replace(old_buttons, new_buttons, 1)

# 3. Replace the entire Modal with an editable form modal
modal_target_start = '{/* Deploy to Paper/Live Modal */}'
modal_idx = text.find(modal_target_start)
assert modal_idx != -1, "modal_target_start not found"

new_modal_code = '''{/* Deploy to Paper/Live Modal with Editable Parameters */}
      {deployModalMode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4 overflow-y-auto">
          <div className="relative w-full max-w-xl rounded-xl border border-gray-700 bg-gray-900 p-6 shadow-2xl space-y-4 my-8">
            <div className="flex items-center justify-between border-b border-gray-800 pb-3">
              <div className="flex items-center gap-2">
                <span className="text-lg">
                  {deployModalMode === "live" ? "⚡" : "📝"}
                </span>
                <h3 className="text-base font-semibold text-white">
                  {deployModalMode === "live" ? "Deploy to Live Trading (MT5 Real)" : "Deploy to Paper Trading (Simulated)"}
                </h3>
              </div>
              <button
                onClick={() => setDeployModalMode(null)}
                className="text-gray-400 hover:text-white text-lg font-bold"
              >
                ✕
              </button>
            </div>

            <div className="flex items-center justify-between text-xs text-gray-400 bg-gray-950/60 p-2.5 rounded-lg border border-gray-800">
              <div>
                <span>Inherited from: </span>
                <span className="text-white font-medium">Run #{runId}</span>
                <span className="text-gray-500"> ({strategy?.name})</span>
              </div>
              <button
                type="button"
                onClick={resetDeployParams}
                className="text-brand-400 hover:text-sky-300 underline"
                title="Reset values to original backtest settings"
              >
                Reset to Backtest defaults
              </button>
            </div>

            {/* Editable Parameter Form */}
            <div className="space-y-4 text-xs">
              {/* Order Sizing & Capital */}
              <div className="rounded-lg border border-gray-800 bg-gray-950 p-4 space-y-3">
                <p className="font-semibold text-gray-300 uppercase tracking-wider text-[11px]">
                  1. Execution & Capital Sizing
                </p>
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                  <div>
                    <label className="block text-gray-400 mb-1">
                      Order Lot Size (MT5 Lots)
                    </label>
                    <input
                      type="number"
                      step={0.01}
                      min={0.01}
                      max={100}
                      value={overrideLotSize}
                      onChange={(e) => setOverrideLotSize(parseFloat(e.target.value) || 0.01)}
                      className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
                    />
                    <p className="mt-0.5 text-[10px] text-gray-500">1.0 lot = 100,000 units (0.1 = 10,000 units)</p>
                  </div>

                  <div>
                    <label className="block text-gray-400 mb-1">
                      Starting Capital (JPY)
                    </label>
                    <input
                      type="number"
                      step={10000}
                      min={1000}
                      value={overrideCapital}
                      onChange={(e) => setOverrideCapital(parseFloat(e.target.value) || 0)}
                      className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
                    />
                    <p className="mt-0.5 text-[10px] text-gray-500">Initial balance for tracking PnL and returns</p>
                  </div>
                </div>
              </div>

              {/* Strategy Parameters (Analyst Range Reversion / Breakout) */}
              {strategy?.definition?.strategy_type?.startsWith("analyst_") ? (
                <div className="rounded-lg border border-gray-800 bg-gray-950 p-4 space-y-3">
                  <div className="flex items-center justify-between">
                    <p className="font-semibold text-gray-300 uppercase tracking-wider text-[11px]">
                      2. Analyst Strategy Parameters
                    </p>
                    <span className="text-[11px] text-sky-400 font-mono">
                      {strategy.definition.strategy_type}
                    </span>
                  </div>
                  <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
                    <div>
                      <label className="block text-gray-400 mb-1">
                        Stop Loss (pips)
                      </label>
                      <input
                        type="text"
                        value={overrideSlPips}
                        onChange={(e) => setOverrideSlPips(e.target.value)}
                        placeholder="e.g. 25"
                        className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
                      />
                      <p className="mt-0.5 text-[10px] text-gray-500">25 pips = 0.25 JPY (or empty/none)</p>
                    </div>

                    <div>
                      <label className="block text-gray-400 mb-1">
                        Take Profit (pips)
                      </label>
                      <input
                        type="text"
                        value={overrideTpPips}
                        onChange={(e) => setOverrideTpPips(e.target.value)}
                        placeholder="e.g. 100 or opposite"
                        className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
                      />
                      <p className="mt-0.5 text-[10px] text-gray-500">100 pips = 1.00 JPY (or opposite/none)</p>
                    </div>

                    <div>
                      <label className="block text-gray-400 mb-1">
                        Entry Tolerance (pips)
                      </label>
                      <input
                        type="text"
                        value={overrideTolPips}
                        onChange={(e) => setOverrideTolPips(e.target.value)}
                        placeholder="e.g. 10"
                        className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
                      />
                      <p className="mt-0.5 text-[10px] text-gray-500">Buffer from forecast boundary</p>
                    </div>
                  </div>
                </div>
              ) : (
                /* General Rule Strategy Parameters */
                <div className="rounded-lg border border-gray-800 bg-gray-950 p-4 space-y-3">
                  <p className="font-semibold text-gray-300 uppercase tracking-wider text-[11px]">
                    2. Risk & Exit Controls
                  </p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
                    <div>
                      <label className="block text-gray-400 mb-1">
                        Stop Loss (%)
                      </label>
                      <input
                        type="number"
                        step={0.1}
                        min={0.1}
                        value={overrideSlPct}
                        onChange={(e) => setOverrideSlPct(parseFloat(e.target.value) || 0)}
                        className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
                      />
                    </div>
                    <div>
                      <label className="block text-gray-400 mb-1">
                        Take Profit (%)
                      </label>
                      <input
                        type="number"
                        step={0.1}
                        min={0.1}
                        value={overrideTpPct}
                        onChange={(e) => setOverrideTpPct(parseFloat(e.target.value) || 0)}
                        className="w-full rounded border border-gray-700 bg-gray-900 px-3 py-1.5 text-sm text-white font-mono focus:border-brand-500 focus:outline-none"
                      />
                    </div>
                  </div>
                </div>
              )}

              {/* MT5 Broker Status */}
              <div className="flex items-center justify-between rounded-lg border border-gray-800 bg-gray-950 px-3.5 py-2.5">
                <span className="text-gray-400">Target Broker Connection:</span>
                <span className="font-mono">
                  {mt5Status?.connected ? (
                    <span className="text-emerald-400 font-medium">
                      🟢 {mt5Status.account?.server} (Bal: {mt5Status.account?.balance?.toLocaleString()} JPY)
                    </span>
                  ) : (
                    <span className="text-red-400">🔴 Disconnected (Check MT5 Desktop App)</span>
                  )}
                </span>
              </div>
            </div>

            {/* Mode Warning */}
            {deployModalMode === "live" ? (
              <div className="rounded border border-amber-800/60 bg-amber-950/40 p-3 text-xs text-amber-300 space-y-1">
                <p className="font-semibold">⚠️ Live Trading Confirmation:</p>
                <p>
                  MT5の接続口座に対して<strong>実際の成行発注・決済</strong>が行われます。
                  上記の設定値（Lot数、SL/TP）で問題ないことをご確認ください。
                </p>
              </div>
            ) : (
              <div className="rounded border border-sky-800/60 bg-sky-950/40 p-3 text-xs text-sky-300 space-y-1">
                <p className="font-semibold">ℹ️ Paper Trading Mode:</p>
                <p>
                  MT5から取得するリアルタイムレートを用いて、上記設定でシミュレーション（Dry-run）取引を行います。ブローカーへの実発注は行われません。
                </p>
              </div>
            )}

            {/* Actions */}
            <div className="flex justify-end gap-3 pt-2">
              <button
                type="button"
                onClick={() => setDeployModalMode(null)}
                disabled={deploying}
                className="rounded border border-gray-700 px-4 py-2 text-xs font-medium text-gray-300 hover:bg-gray-800 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                type="button"
                onClick={() => handleDeploy(deployModalMode)}
                disabled={deploying}
                className={`rounded px-5 py-2 text-xs font-semibold text-white shadow-md disabled:opacity-50 transition-colors ${
                  deployModalMode === "live"
                    ? "bg-emerald-600 hover:bg-emerald-500"
                    : "bg-sky-600 hover:bg-sky-500"
                }`}
              >
                {deploying
                  ? "Starting Run…"
                  : deployModalMode === "live"
                  ? "🚀 Start Live Trading"
                  : "🚀 Start Paper Trading"}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}'''

# Replace from modal_idx to end
text = text[:modal_idx] + new_modal_code

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Successfully updated RunDetailPage with full editable parameter modal!")
