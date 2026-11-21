"""
Patch execution.py to add entry_reason support.
"""
from pathlib import Path

path = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend\strategy\engine\execution.py")
content = path.read_text(encoding="utf-8")

old_pos = '''@dataclass
class OpenPosition:
    direction: str
    entry_price: float
    sl_price: float | None
    tp_price: float | None
    volume: float
    opened_at: datetime
    bar_index: int = 0
    mae: float = 0.0
    mfe: float = 0.0'''

new_pos = '''@dataclass
class OpenPosition:
    direction: str
    entry_price: float
    sl_price: float | None
    tp_price: float | None
    volume: float
    opened_at: datetime
    bar_index: int = 0
    mae: float = 0.0
    mfe: float = 0.0
    entry_reason: str = "signal"'''

assert old_pos in content, "old_pos not found in execution.py"
content = content.replace(old_pos, new_pos)

old_trade = '''        "opened_at": pos.opened_at,
        "closed_at": closed_at,
        "exit_reason": reason,'''

new_trade = '''        "opened_at": pos.opened_at,
        "closed_at": closed_at,
        "entry_reason": getattr(pos, "entry_reason", "signal"),
        "exit_reason": reason,'''

assert old_trade in content, "old_trade not found in execution.py"
content = content.replace(old_trade, new_trade)

path.write_text(content, encoding="utf-8")
print("Successfully patched execution.py!")
