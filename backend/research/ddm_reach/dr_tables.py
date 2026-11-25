"""Markdown tables from results/*.json -> results/tables.md (plan v0.6)."""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import dr_env as E  # noqa: E402

L = []


def f(x, n=3):
    if x is None:
        return "—"
    if isinstance(x, (list, tuple)):
        return "[" + ", ".join(f(v, n) for v in x) + "]"
    if isinstance(x, bool):
        return "はい" if x else "いいえ"
    if isinstance(x, (int, np.integer)):
        return str(x)
    return f"{x:.{n}f}"


def tab(h, rows):
    L.append("| " + " | ".join(h) + " |")
    L.append("|" + "|".join(["---"] * len(h)) + "|")
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("")


def J(name):
    p = os.path.join(E.RESULTS, name)
    return json.load(open(p, encoding="utf-8")) if os.path.exists(p) else None


a0, g, a1, b, fr, c, d = J("a0.json"), J("a1_gen.json"), J("a1_analysis.json"), J("b_windows_summary.json"), J("frozen_rules.json"), J("c_min.json"), J("d_robustness.json")
if a0:
    L.append("## A0 既存アトラスの再解釈\n")
    tab(["統計量", "最小", "中央値", "最大", "点間 SD", "seed SE の中央値", "SE ÷ 点間 SD の中央値", "SE ≤ 0.25 SD の点の割合"],
        [[k, f(v["min"], 4), f(v["median"], 4), f(v["max"], 4), f(v["sd_across_points"], 4), f(v["median_se_seed"], 4), f(v["median_se_over_sd_across_points"]), f(v["frac_points_se_le_0.25_sd_across_points"])] for k, v in a0["stats"].items()])
    tab(["q", "実 acf_1", "実 尖度", "実 volclust_50", "アトラスの範囲内(acf_1/尖度/volclust)", "acf_1 箱内の点", "尖度 箱内", "volclust 箱内", "3 次元の箱内", "acf_1 と尖度の 2 次元の箱内"],
        [[r["q"], f(r["real_acf_1"]), f(r["real_kurtosis"], 1), f(r["real_volclust_50"]), f"{f(r['atlas_range_contains_acf_1'])}/{f(r['atlas_range_contains_kurtosis'])}/{f(r['atlas_range_contains_volclust_50'])}",
          r["n_atlas_in_half_sd_box_acf_1"], r["n_atlas_in_half_sd_box_kurtosis"], r["n_atlas_in_half_sd_box_volclust_50"], r["n_atlas_in_3d_box"], r["n_atlas_in_2d_box_acf1_kurt"]] for r in a0["real_buckets_vs_atlas"]])
if a1:
    L.append("## A1 DDM の格子\n")
    tab(["項目", "値"], [["run 数 / 成功", f"{a1['n_runs']} / {a1['n_runs_success']}"], ["N_grid(全 10 seed 成功の点)", a1["N_grid"]], ["SD_grid", f({k: v for k, v in a1["SD_grid"].items()}.values() and list(a1["SD_grid"].values()), 4)],
                        ["DDM 精度資格を満たす点(全窓 / 最終)", f"{a1['n_DDM_precision_eligible_all_windows']} / {a1['n_DDM_precision_eligible_final']}"],
                        ["尖度が精度資格を満たさない点の割合 → 主の裾の指標", f"{f(a1['fallback_rule']['fraction_of_N_grid_points_failing_kurtosis_precision'])} → {a1['fallback_rule']['primary_tail_measure']}"],
                        ["初期過渡: 最初の窓を除外するか", f(a1["transient"]["exclude_first_window"])], ["SA/SB の一致割合(基準 0.80)", f(a1["SA_SB_agreement"]["fraction_all_components"])]])
    tab(["統計量(SA 点平均)", "最小", "中央値", "最大"], [[k, f(v["min"], 4), f(v["median"], 4), f(v["max"], 4)] for k, v in a1["M_ranges_SA"].items()])
    tab(["TPC", "成分", "中央値", "最小", "最大"], [[t, c_, f(v["median"], 4), f(v["min"], 4), f(v["max"], 4)] for t, dd in a1["tpc_sensitivity_SA"].items() for c_, v in dd.items()])
if b and fr:
    L.append("## B 実市場の窓と凍結した規則\n")
    p = fr["primary"]
    tab(["項目", "値"], [["R-A / R-B の窓数", f"{p['n_RA']} / {p['n_RB']}"], ["R-A / R-B の週数", f"{p['n_weeks_RA']} / {p['n_weeks_RB']}"], ["M", ", ".join(p["M"])], ["SD_j", f(p["SD"], 4)], ["caliper c_j", f(p["c"], 4)],
                        ["E", ", ".join(p["E"])], ["σ_j", f(p["sigma"], 4)], ["τ_j", f(p["tau"], 3)], ["f_0j(クロスフィット平均)", f(p["f0"], 4)], ["f_0j の分割間 SD(記述的)", f(p["f0_split_sd"], 4)],
                        ["ノイズ床に使える R-B 窓(参照窓 3 以上)", p["n_RB_valid_ref_ge3"]], ["参照窓不足で除外した割合", f(p["frac_RB_unreferenceable"])], ["f_0j が [0.03, 0.08] の外", f(p["f0_outside_0.03_0.08"])]])
    tab(["ATR 群", "R-A 窓", "R-B 窓"], [[k, v["RA"], v["RB"]] for k, v in fr["atr_group_counts"].items()])
if c:
    L.append("## C-min 到達可能性\n")
    r = c["reachability"]
    tab(["項目", "値"], [["マッチング資格を満たす点 / 全 10 seed 成功の点", f"{r['n_points_matching_eligible']} / {r['n_points_all10']}"], ["到達可能な R-A 窓(箱型)", f"{r['n_reachable_windows']} / {r['n_RA_windows']} ({f(r['fraction_reachable_box'])})"],
                        ["到達可能な週数", f"{r['n_weeks_reachable']} / {r['n_weeks_RA']}"], ["ユークリッド 0.5(より厳しい)", f(r["fraction_reachable_euclid_0.5"])],
                        ["最近傍点までのチェビシェフ距離(SD 単位): 最小/中央値/最大", f"{f(r['nearest_chebyshev_distance_SD_units']['min'])} / {f(r['nearest_chebyshev_distance_SD_units']['median'])} / {f(r['nearest_chebyshev_distance_SD_units']['max'])}"],
                        ["Jaccard(SA 基準 vs SB 基準の到達可能な点)", f(r["jaccard_reachable_points_SA_vs_SB"]["jaccard"])]])
    tab(["成分", "その成分の caliper を外したときの到達可能割合", "最近傍点で最も外れている成分の窓数", "DDM の範囲", "実市場の範囲(最小/中央値/最大)", "実窓の周辺カバー率"],
        [[m, f(r["fraction_reachable_if_caliper_removed_for"][m]), r["binding_component_at_nearest_point_counts"][m], f([r["marginal_coverage"][m]["ddm_min"], r["marginal_coverage"][m]["ddm_max"]], 4),
          f([r["marginal_coverage"][m]["real_min"], r["marginal_coverage"][m]["real_median"], r["marginal_coverage"][m]["real_max"]], 4), f(r["marginal_coverage"][m]["fraction_real_inside_ddm_range"])] for m in fr["primary"]["M"]])
    tab(["ATR 群", "窓数", "到達可能割合"], [[k, v["n"], f(v["fraction_reachable"])] for k, v in r["by_atr_group"].items()])
    L.append(f"残差分析を実行: {f(c['residual_analysis_run'])}。{c.get('residual_skipped_reason', '')}\n")
    if c.get("residual"):
        res = c["residual"]
        En = fr["primary"]["E"]
        tab(["E", "D_j", "f_j", "f_0j", "g_j", "seed 層(パーセンタイル)g の 90% 区間", "seed 層(t)g の 90% 区間", "実市場層 g の 90% 区間(τ・f_0 再推定)", "条件付き f の区間(τ・f_0 固定)"],
            [[e, f(res["point"]["D"][j]), f(res["point"]["f"][j]), f(fr["primary"]["f0"][j]), f(res["point"]["g"][j]), f([res["seed_layer"]["pct"]["g_p5"][j], res["seed_layer"]["pct"]["g_p95"][j]]),
              f([res["seed_layer"]["t"]["g_p5"][j], res["seed_layer"]["t"]["g_p95"][j]]), f([res["real_layer"]["g_rep"]["p5"][j], res["real_layer"]["g_rep"]["p95"][j]]),
              f([res["real_layer"]["f_cond"]["p5"][j], res["real_layer"]["f_cond"]["p95"][j]])] for j, e in enumerate(En)])
if d:
    L.append("## D 頑健性(凍結した規則)\n")
    rows = [["検証期間(凍結した規則)", f(d["validation_frozen"]["reach"]["fraction_reachable_box"]), d["validation_frozen"]["reach"]["n_reachable_windows"], d["validation_frozen"]["reach"]["n_RA_windows"]],
            ["SA/SB を入れ替え(R-A)", f(d["swap_SA_SB"]["fraction_reachable_box"]), d["swap_SA_SB"]["n_reachable_windows"], d["swap_SA_SB"]["n_RA_windows"]],
            ["08〜17 時開始の R-A 窓", f(d["hours_08_17_RA"]["fraction_reachable_box"]), d["hours_08_17_RA"]["n_reachable_windows"], d["hours_08_17_RA"]["n_RA_windows"]]]
    for nme, v in d["partitions"].items():
        rows.append([f"分割 {nme}(M = {', '.join(v['M'])})", f(v["fraction_reachable_box"]), v["n_reachable_windows"], v["n_RA_windows"]])
    for nme, v in d["unit_pairing_L1000"].items():
        rows.append([f"単位の組み合わせ {nme}(L=1000)", f(v["fraction_reachable_box"]), v["n_reachable_windows"], v["n_RA"]])
    for nme in ("TPC10", "TPC20"):
        v = d["L4000_bar1"][nme]
        rows.append([f"L=4000 {nme} bar1", f(v["fraction_reachable_box"]), v["n_reachable_windows"], v["n_RA_windows"]])
    for y, v in d["by_year_RA"].items():
        rows.append([f"R-A の暦年 {y}", f(v["fraction_reachable"]), "—", v["n"]])
    tab(["条件", "到達可能割合(箱型)", "到達可能な窓", "窓数"], rows)
open(os.path.join(E.RESULTS, "tables.md"), "w", encoding="utf-8").write("\n".join(L))
print("tables written")
