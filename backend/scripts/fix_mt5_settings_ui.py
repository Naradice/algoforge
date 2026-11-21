path = r"c:\Users\ksato\workspace\finance\algoforge\web\components\mt5-settings.tsx"
with open(path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace placeholder
target = 'placeholder="e.g. 900009506"'
replacement = 'placeholder="e.g. 12345678"'
assert target in text, "Target placeholder not found"
text = text.replace(target, replacement)

# Add clear helpful note above credentials
form_header_target = '<h3 className="text-sm font-medium text-white border-b border-gray-800 pb-3">\n          Account Credentials & Parameters\n        </h3>'
form_header_replacement = '''<h3 className="text-sm font-medium text-white border-b border-gray-800 pb-3 flex items-center justify-between">
          <span>Account Credentials & Parameters</span>
          <span className="text-xs font-normal text-gray-400">※ PCでMT5が起動中なら空欄でも自動接続可能</span>
        </h3>'''
if form_header_target in text:
    text = text.replace(form_header_target, form_header_replacement)

with open(path, "w", encoding="utf-8") as f:
    f.write(text)
print("Updated mt5-settings.tsx placeholder and note")
