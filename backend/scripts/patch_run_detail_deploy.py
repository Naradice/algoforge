path = r"c:\Users\ksato\workspace\finance\algoforge\web\app\strategy\[id]\runs\[run_id]\page.tsx"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# 1. Add toast import
old_import = 'import { useSSE } from "@/hooks/use-sse";'
new_import = 'import { useSSE } from "@/hooks/use-sse";\nimport { useToast } from "@/lib/toast";'
assert old_import in text, "import target not found"
text = text.replace(old_import, new_import, 1)

# 2. Add toast, modal states and handleDeploy inside RunDetailPage
old_state_anchor = '  const [stopping, setStopping] = useState(false);'
new_state_anchor = '''  const [stopping, setStopping] = useState(false);
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
assert old_state_anchor in text, "state anchor not found"
text = text.replace(old_state_anchor, new_state_anchor, 1)

# 3. Add deploy buttons in header next to mode/status badges
old_header_actions = '''          {run && (
            <span className="rounded px-2 py-1 text-xs bg-gray-700 text-gray-300">
              {run.mode}
            </span>
          )}
          {isActive && ('''

new_header_actions = '''          {run && (
            <span className="rounded px-2 py-1 text-xs bg-gray-700 text-gray-300 uppercase">
              {run.mode}
            </span>
          )}
          {!isActive && (
            <div className="flex items-center gap-2">
              <button
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
              </button>
            </div>
          )}
          {isActive && ('''
assert old_header_actions in text, "header actions not found"
text = text.replace(old_header_actions, new_header_actions, 1)

# 4. Insert deploy modal before the final `    </div>\n  );\n}`
old_modal_anchor = '    </div>\n  );\n}'
last_idx = text.rfind(old_modal_anchor)
assert last_idx != -1, "modal anchor not found"

modal_code = '''      {/* Deploy to Paper/Live Modal */}
      {deployModalMode && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/75 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-lg rounded-xl border border-gray-700 bg-gray-900 p-6 shadow-2xl space-y-5">
            <div className="flex items-center justify-between border-b border-gray-800 pb-3">
              <h3 className="text-base font-semibold text-white flex items-center gap-2">
                {deployModalMode === "live" ? (
                  <>
                    <span className="text-emerald-400">⚡</span> Deploy to Live Trading
                  </>
                ) : (
                  <>
                    <span className="text-sky-400">📝</span> Deploy to Paper Trading
                  </>
                )}
              </h3>
              <button
                onClick={() => setDeployModalMode(null)}
                className="text-gray-400 hover:text-white text-lg font-bold"
              >
                ✕
              </button>
            </div>

            <p className="text-xs text-gray-400">
              Run #{runId}（{run?.message || "Backtest"}）の設定を引き継いでリアルタイム取引を開始します。
            </p>

            {/* Settings Summary */}
            <div className="rounded-lg border border-gray-800 bg-gray-950 p-3 space-y-2 text-xs">
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-500">Strategy</span>
                <span className="text-white font-medium">{strategy?.name}</span>
              </div>
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-500">Type & Symbol</span>
                <span className="text-white font-mono">
                  {strategy?.definition?.strategy_type || "Rules"} ({strategy?.definition?.symbol || "USDJPY"})
                </span>
              </div>
              {strategy?.definition?.parameters && (
                <div className="flex justify-between py-1 border-b border-gray-800/60">
                  <span className="text-gray-500">Strategy Parameters</span>
                  <span className="text-white font-mono">
                    SL={strategy.definition.parameters.sl_pips ?? "—"}pips, TP={strategy.definition.parameters.tp_pips ?? "—"}pips, Tol={strategy.definition.parameters.tolerance_pips ?? 0}pips
                  </span>
                </div>
              )}
              <div className="flex justify-between py-1 border-b border-gray-800/60">
                <span className="text-gray-500">Starting Capital</span>
                <span className="text-white font-mono">
                  {run?.starting_capital ? `${run.starting_capital.toLocaleString()} JPY` : "1,000,000 JPY (Default)"}
                </span>
              </div>
              <div className="flex justify-between py-1">
                <span className="text-gray-500">MT5 Broker Account</span>
                <span className="text-white font-mono">
                  {mt5Status?.connected
                    ? `🟢 ${mt5Status.account?.server} (Bal: ${mt5Status.account?.balance?.toLocaleString()} JPY)`
                    : "🔴 Disconnected"}
                </span>
              </div>
            </div>

            {deployModalMode === "live" ? (
              <div className="rounded border border-amber-800/60 bg-amber-950/40 p-3 text-xs text-amber-300 space-y-1">
                <p className="font-semibold">⚠️ Live モードの確認:</p>
                <p>
                  MT5の接続口座に対して<strong>実際の成行発注・決済</strong>が行われます。
                  事前にMT5 Broker設定および口座残高をご確認ください。
                </p>
              </div>
            ) : (
              <div className="rounded border border-sky-800/60 bg-sky-950/40 p-3 text-xs text-sky-300 space-y-1">
                <p className="font-semibold">ℹ️ Paper モード（シミュレーション）:</p>
                <p>
                  MT5から取得するリアルタイムレートを用いて、AlgoForge内部で擬似取引（Dry-run）を行います。ブローカー口座への実発注は行われません。
                </p>
              </div>
            )}

            {/* Actions */}
            <div className="flex justify-end gap-3 pt-2">
              <button
                onClick={() => setDeployModalMode(null)}
                disabled={deploying}
                className="rounded border border-gray-700 px-4 py-2 text-xs font-medium text-gray-300 hover:bg-gray-800 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDeploy(deployModalMode)}
                disabled={deploying}
                className={`rounded px-4 py-2 text-xs font-semibold text-white shadow-md disabled:opacity-50 transition-colors ${
                  deployModalMode === "live"
                    ? "bg-emerald-600 hover:bg-emerald-500"
                    : "bg-sky-600 hover:bg-sky-500"
                }`}
              >
                {deploying
                  ? "Deploying…"
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

text = text[:last_idx] + modal_code + text[last_idx + len(old_modal_anchor):]

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Successfully patched RunDetailPage with Paper/Live Deploy actions!")
