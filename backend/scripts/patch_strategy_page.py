import sys

path = r"c:\Users\ksato\workspace\finance\algoforge\web\app\strategy\[id]\page.tsx"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Normalize to LF for editing
text = text.replace("\r\n", "\n")

# 1. Add MT5Settings import
old_import = 'import { OptimizationPanel } from "@/components/optimization-panel";'
new_import = 'import { OptimizationPanel } from "@/components/optimization-panel";\nimport { MT5Settings } from "@/components/mt5-settings";'
if old_import in text:
    text = text.replace(old_import, new_import, 1)
    print("1. import patched")
else:
    print("1. import skipped (already present or not found)")

# 2. Add showMT5Modal state
old_state = '  const [copying, setCopying] = useState(false);'
new_state = '  const [copying, setCopying] = useState(false);\n  const [showMT5Modal, setShowMT5Modal] = useState(false);'
if old_state in text:
    text = text.replace(old_state, new_state, 1)
    print("2. state patched")
else:
    print("2. state skipped")

# 3. Add MT5 Broker button in Runs header
old_runs_header = '<h2 className="text-sm font-medium uppercase tracking-wide text-gray-400">Runs</h2>\n          <div className="flex gap-2">'
new_runs_header = old_runs_header + '''
            <button
              onClick={() => setShowMT5Modal(true)}
              className="rounded border border-gray-700 bg-gray-800 px-3 py-1 text-xs text-gray-300 hover:bg-gray-700 flex items-center gap-1.5 transition-colors"
              title="Configure MT5 broker connection"
            >
              <span>⚙️</span> MT5 Broker
            </button>'''
if old_runs_header in text:
    text = text.replace(old_runs_header, new_runs_header, 1)
    print("3. runs header patched")
else:
    print("3. runs header skipped")

# 4. Update Mode select and descriptions
old_mode_select = '''              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="backtest">Backtest</option>
                <option value="paper">Paper (live yfinance, simulated orders)</option>
                <option value="live">Live (broker — not implemented)</option>
              </select>'''

new_mode_select = '''              <select
                value={mode}
                onChange={(e) => setMode(e.target.value)}
                className="w-full rounded border border-gray-700 bg-gray-800 px-2 py-1.5 text-sm text-white focus:outline-none focus:ring-1 focus:ring-brand-500"
              >
                <option value="backtest">Backtest (Historical data)</option>
                <option value="paper">Paper (MT5 Simulated Demo / Dry-run)</option>
                <option value="live">Live (MT5 Real Order Execution)</option>
              </select>
              {mode === "live" && (
                <div className="mt-2 rounded bg-amber-950/40 border border-amber-800/60 p-2 text-xs text-amber-300">
                  ⚠️ <strong>Live Mode:</strong> MT5の接続口座に対して実際の成行発注と決済を行います。事前に「MT5 Broker」ボタンより接続テストおよび口座残高をご確認ください。
                </div>
              )}
              {mode === "paper" && (
                <div className="mt-2 rounded bg-sky-950/40 border border-sky-800/60 p-2 text-xs text-sky-300">
                  ℹ️ <strong>Paper Mode:</strong> MT5から取得するリアルタイムレートを用いて、擬似取引シミュレーション（Dry-run）を行います。ブローカー口座への実発注は行われません。
                </div>
              )}'''
if old_mode_select in text:
    text = text.replace(old_mode_select, new_mode_select, 1)
    print("4. mode select patched")
else:
    print("4. mode select skipped")

# 5. Add Stop button in run row
old_run_actions = '''<span className="text-gray-500 mr-3">{selectedRun === run.id ? "▲ Hide" : "▼ Show"}</span>
                    {!["pending", "running"].includes(run.status) && ('''

new_run_actions = '''<span className="text-gray-500 mr-3">{selectedRun === run.id ? "▲ Hide" : "▼ Show"}</span>
                    {["pending", "running"].includes(run.status) && (
                      <button
                        onClick={(e) => { e.stopPropagation(); stopRun(run.id); }}
                        disabled={stopping === run.id}
                        className="text-amber-400 hover:text-amber-300 font-medium disabled:opacity-50 mr-3"
                      >
                        {stopping === run.id ? "Stopping…" : "⏹ Stop"}
                      </button>
                    )}
                    {!["pending", "running"].includes(run.status) && ('''
if old_run_actions in text:
    text = text.replace(old_run_actions, new_run_actions, 1)
    print("5. run stop button patched")
else:
    print("5. run stop button skipped")

# 6. Add MT5 Modal at bottom of component
old_modal_anchor = '    </div>\n  );\n}'
new_modal_code = '''
      {/* MT5 Settings Modal */}
      {showMT5Modal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-3xl rounded-xl border border-gray-700 bg-gray-900 shadow-2xl p-6 max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowMT5Modal(false)}
              className="absolute top-4 right-4 text-gray-400 hover:text-white text-lg font-bold"
            >
              ✕
            </button>
            <MT5Settings />
          </div>
        </div>
      )}
    </div>
  );
}'''
if old_modal_anchor in text:
    text = text.replace(old_modal_anchor, new_modal_code, 1)
    print("6. modal patched")
else:
    print("6. modal skipped")

with open(path, "w", encoding="utf-8") as f:
    f.write(text)

print("Done patching page.tsx!")
