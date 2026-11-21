from pathlib import Path

p = Path(r"c:\Users\ksato\workspace\finance\algoforge\backend\strategy\executor.py")
content = p.read_text(encoding="utf-8")

old_code = """    if run.mode == "paper":
        from strategy.live_runner import run_paper
        return await run_paper(run_id, definition, model_cache)

    if run.mode == "live":
        async with session_factory() as db:
            await db.execute(
                update(StrategyRun).where(StrategyRun.id == run_id).values(
                    status="error", message="Live broker execution not yet implemented"
                )
            )
            await dispatch(db, "run.error", {
                "run_id": run_id, "strategy_id": run.strategy_id, "mode": run.mode,
                "error": "live_not_implemented",
            })
            await db.commit()
        return {"error": "live_not_implemented"}"""

new_code = """    if run.mode in ("paper", "live"):
        from strategy.live_runner import run_live
        return await run_live(run_id, definition, model_cache=model_cache, mode=run.mode)"""

assert old_code in content, "old_code not found in executor.py"
content = content.replace(old_code, new_code)
p.write_text(content, encoding="utf-8")
print("Successfully patched executor.py for live trading!")
