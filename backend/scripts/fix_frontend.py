# Fix mt5-settings.tsx
mt5_settings_path = r"c:\Users\ksato\workspace\finance\algoforge\web\components\mt5-settings.tsx"
with open(mt5_settings_path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace('import { toast } from "@/lib/toast";', 'import { useToast } from "@/lib/toast";')
text = text.replace(
    'export function MT5Settings({ onClose }: { onClose?: () => void }) {\n  const { data: status',
    'export function MT5Settings({ onClose }: { onClose?: () => void }) {\n  const { toast } = useToast();\n  const { data: status'
)
with open(mt5_settings_path, "w", encoding="utf-8") as f:
    f.write(text)
print("Fixed mt5-settings.tsx")

# Fix page.tsx
page_path = r"c:\Users\ksato\workspace\finance\algoforge\web\app\strategy\[id]\page.tsx"
with open(page_path, "r", encoding="utf-8") as f:
    ptext = f.read()

# Remove the misplaced MT5 Settings Modal from MetricCard
misplaced = '''      {/* MT5 Settings Modal */}
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
'''
if misplaced in ptext:
    ptext = ptext.replace(misplaced, "")
    print("Removed misplaced modal")
else:
    print("Misplaced modal not found as exact match, searching...")

# Append modal right before the final `    </div>\n  );\n}` in page.tsx
# Find the last occurrence of `    </div>\n  );\n}`
target = '    </div>\n  );\n}'
last_idx = ptext.rfind(target)
if last_idx != -1:
    modal_code = '''      {/* MT5 Settings Modal */}
      {showMT5Modal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="relative w-full max-w-3xl rounded-xl border border-gray-700 bg-gray-900 shadow-2xl p-6 max-h-[90vh] overflow-y-auto">
            <button
              onClick={() => setShowMT5Modal(false)}
              className="absolute top-4 right-4 text-gray-400 hover:text-white text-lg font-bold"
            >
              ✕
            </button>
            <MT5Settings onClose={() => setShowMT5Modal(false)} />
          </div>
        </div>
      )}
    </div>
  );
}'''
    ptext = ptext[:last_idx] + modal_code + ptext[last_idx + len(target):]
    print("Inserted modal at the end of StrategyDetailPage")

with open(page_path, "w", encoding="utf-8") as f:
    f.write(ptext)
print("Fixed page.tsx")
