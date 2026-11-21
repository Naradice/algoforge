# 1. Update live_runner.py
live_path = r"c:\Users\ksato\workspace\finance\algoforge\backend\strategy\live_runner.py"
with open(live_path, "r", encoding="utf-8") as f:
    text = f.read()

target = '''async def run_live(
    run_id: int,
    strategy_definition: dict,
    model_cache: dict | None = None,
    mode: str = "live",  # "live" or "paper"
) -> dict[str, Any]:
    """
    Main live/paper execution loop.
    Connects to MT5, monitors market rates, evaluates signals, and places trades.
    """
    strategy_type = strategy_definition.get("strategy_type", "")
    symbol = strategy_definition.get("symbol", "USDJPY")
    params = strategy_definition.get("parameters", {})'''

replacement = '''async def run_live(
    run_id: int,
    strategy_definition: dict,
    model_cache: dict | None = None,
    mode: str = "live",  # "live" or "paper"
    risk_override: dict | None = None,
) -> dict[str, Any]:
    """
    Main live/paper execution loop.
    Connects to MT5, monitors market rates, evaluates signals, and places trades.
    """
    strategy_type = strategy_definition.get("strategy_type", "")
    symbol = strategy_definition.get("symbol", "USDJPY")
    params = dict(strategy_definition.get("parameters", {}))

    # Apply any risk_override parameters
    if risk_override:
        for k, v in risk_override.items():
            if v is not None:
                params[k] = v'''

assert target in text, "Target not found in live_runner.py"
text = text.replace(target, replacement)
with open(live_path, "w", encoding="utf-8") as f:
    f.write(text)
print("Updated live_runner.py with risk_override support")

# 2. Update executor.py
exec_path = r"c:\Users\ksato\workspace\finance\algoforge\backend\strategy\executor.py"
with open(exec_path, "r", encoding="utf-8") as f:
    etext = f.read()

etarget = '''    if run.mode in ("paper", "live"):
        from strategy.live_runner import run_live
        return await run_live(run_id, definition, model_cache=model_cache, mode=run.mode)'''

ereplacement = '''    if run.mode in ("paper", "live"):
        from strategy.live_runner import run_live
        return await run_live(run_id, definition, model_cache=model_cache, mode=run.mode, risk_override=run.risk_override)'''

assert etarget in etext, "Target not found in executor.py"
etext = etext.replace(etarget, ereplacement)
with open(exec_path, "w", encoding="utf-8") as f:
    f.write(etext)
print("Updated executor.py to pass risk_override to run_live")
