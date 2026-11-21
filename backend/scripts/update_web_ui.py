import pathlib

def update_run_page():
    path = pathlib.Path(r"c:\Users\ksato\workspace\finance\algoforge\web\app\strategy\[id]\runs\[run_id]\page.tsx")
    content = path.read_text(encoding="utf-8")

    # 1. Trade interface
    if "entry_reason" not in content:
        content = content.replace(
            "  exit_reason: string | null;",
            "  entry_reason: string | null;\n  exit_reason: string | null;"
        )

    # 2. Table headers
    target_header = """                      <th className="pb-2 pr-3">Opened</th>
                      <th className="pb-2 pr-3">Closed</th>
                      <th className="pb-2">Reason</th>"""

    replacement_header = """                      <th className="pb-2 pr-3">Opened</th>
                      <th className="pb-2 pr-3">Closed</th>
                      <th className="pb-2 pr-3">Signal / Entry</th>
                      <th className="pb-2">Exit Reason</th>"""

    content = content.replace(target_header, replacement_header)

    # 3. Table cells
    target_cells = """                        <td className="py-2 pr-3 text-gray-400 text-xs whitespace-nowrap">{t.closed_at ? new Date(t.closed_at).toLocaleString() : "—"}</td>
                        <td className="py-2 text-gray-500 text-xs">{t.exit_reason ?? "—"}</td>"""

    replacement_cells = """                        <td className="py-2 pr-3 text-gray-400 text-xs whitespace-nowrap">{t.closed_at ? new Date(t.closed_at).toLocaleString() : "—"}</td>
                        <td className="py-2 pr-3 text-xs whitespace-nowrap">
                          {t.entry_reason ? (
                            <span className="inline-block px-1.5 py-0.5 rounded bg-sky-950/60 text-sky-300 font-mono text-[11px] border border-sky-800/40">
                              {t.entry_reason}
                            </span>
                          ) : (
                            <span className="text-gray-600">—</span>
                          )}
                        </td>
                        <td className="py-2 text-gray-400 text-xs font-mono whitespace-nowrap">
                          {t.exit_reason ? (
                            <span className="inline-block px-1.5 py-0.5 rounded bg-gray-800/80 text-gray-300 font-mono text-[11px] border border-gray-700/50">
                              {t.exit_reason}
                            </span>
                          ) : (
                            <span className="text-gray-600">—</span>
                          )}
                        </td>"""

    content = content.replace(target_cells, replacement_cells)
    path.write_text(content, encoding="utf-8")
    print("Updated run_id page.tsx successfully.")


def update_strategy_page():
    path = pathlib.Path(r"c:\Users\ksato\workspace\finance\algoforge\web\app\strategy\[id]\page.tsx")
    content = path.read_text(encoding="utf-8")

    # Table headers
    target_header = """                      <th className="py-1 pr-3">Reason</th>
                      <th className="py-1">Opened</th>"""

    replacement_header = """                      <th className="py-1 pr-3">Signal / Entry</th>
                      <th className="py-1 pr-3">Exit Reason</th>
                      <th className="py-1">Opened</th>"""

    content = content.replace(target_header, replacement_header)

    # Table cells
    target_cell = """                        <td className="py-1 pr-3 text-gray-500 text-[10px]">{t.exit_reason ?? "—"}</td>
                        <td className="py-1 text-gray-400">
                          {new Date(t.opened_at).toLocaleDateString()}
                        </td>"""

    replacement_cell = """                        <td className="py-1 pr-3 text-[10px] whitespace-nowrap">
                          {t.entry_reason ? (
                            <span className="inline-block px-1.5 py-0.5 rounded bg-sky-950/60 text-sky-300 font-mono text-[10px] border border-sky-800/40">
                              {t.entry_reason}
                            </span>
                          ) : (
                            <span className="text-gray-600">—</span>
                          )}
                        </td>
                        <td className="py-1 pr-3 text-gray-400 text-[10px] font-mono whitespace-nowrap">{t.exit_reason ?? "—"}</td>
                        <td className="py-1 text-gray-400">
                          {new Date(t.opened_at).toLocaleDateString()}
                        </td>"""

    content = content.replace(target_cell, replacement_cell)
    path.write_text(content, encoding="utf-8")
    print("Updated strategy page.tsx successfully.")

if __name__ == "__main__":
    update_run_page()
    update_strategy_page()
