"""P5-B stage 3: markdown tables from results/analysis3.json -> results/s3_tables.md"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import s3_env as E  # noqa: E402

d = json.load(open(os.path.join(E.RESULTS, "analysis3.json"), encoding="utf-8"))
L = []


def f(x, n=3):
    return "—" if x is None else f"{x:.{n}f}"


def tab(h, rows):
    L.append("| " + " | ".join(h) + " |")
    L.append("|" + "|".join(["---"] * len(h)) + "|")
    for r in rows:
        L.append("| " + " | ".join(str(x) for x in r) + " |")
    L.append("")


v = d["V0"]
L.append("## V0a 判定\n")
tab(["条件", "結果"], [[k, "満たす" if x else "満たさない"] for k, x in v["pass_flags"].items()] + [["V0 全体", "合格" if v["pass"] else "**不合格**"]])
c = v["C3"]
tab(["項目", "値"], [["観測数", c["n_obs"]], ["abs(e) ≤ log 1.25 の観測数(代理量が 0 以下で e が未定義の観測は範囲外として数える)", c["n_within_25pct"]],
                    ["代理量が 0 以下の観測数", c["n_proxy_nonpositive_e_undefined_counted_outside"]], ["Spearman(VRproxy, VR_win)", f(c["spearman"])],
                    ["DDM バケット(4)のうち範囲内", c["buckets_within"]], ["異なる (delta, wma) の数 / 重複観測数", f"{v['supplement']['n_unique_delta_wma']} / {v['supplement']['n_duplicate_observations']}"],
                    ["補足: 重複除去後の Spearman", f(v["supplement"]["unique_spearman"])], ["補足: 重複除去後の範囲内の割合", f(v["supplement"]["unique_fraction_within_25pct"])]])
L.append("## V0b 32 観測(C3)\n")
tab(["q", "delta", "wma", "VRproxy(60)", "VR_win(60)", "e"], [[o["q"], o["delta"], o["wma"], f(o["vrproxy"]), f(o["vrwin"]), "未定義" if o["e"] is None else f(o["e"])] for o in v["C3_observations"]])
L.append("## V0c バケット単位\n")
tab(["側", "q", "VRproxy(60)", "VR_win(60)", "e"], [["DDM(C3)", b["q"], f(b["vrproxy"]), f(b["vrwin"]), "未定義" if b["e"] is None else f(b["e"])] for b in v["C3_buckets"]] +
    [["実現値", b["q"], f(b["vrproxy"]), f(b["vrwin"]), f(b["e"])] for b in v["real_buckets"]])
L.append("## A3 候補空間の被覆(全 20 バケット、有効点 170)\n")
tab(["q", "実現 acf_1", "acf_1 窓間SD", "std 帯(±1SD)の点数", "std ±25% の点数", "acf_1 帯のみの点数", "std 帯(SD)かつ acf_1 帯", "std ±25% かつ acf_1 帯", "全有効点の最小 |Δacf_1|", "std ±25% 内の最小 |Δacf_1|"],
    [[q, f(a["real_acf_1"]), f(a["acf1_window_sd"]), a["n_std_sd"], a["n_std_pct25"], a["n_acf1_band_only"], a["n_std_sd_and_acf1"], a["n_std_pct25_and_acf1"], f(a["min_acf1_dist_all_valid"]),
      f(a["min_acf1_dist_in_std_pct25"])] for q, a in d["A3"].items()])
L.append("## B3a 現行選択(8 点)の位置\n")
tab(["q", "選択 8 点の acf_1 − 実現(平均)", "選択 8 点の acf_5 − 実現(平均)", "30 点短リスト内の acf_1 近さの順位(8 点)", "std ±25% の点数", "std ±25% 内の順位(8 点)", "短リスト内の acf_1 帯の点数", "短リストの |Δacf_1| 最小", "全有効点の |Δacf_1| 最小"],
    [[q, f(b["sel_acf1_minus_real_mean"]), f(b["sel_acf5_minus_real_mean"]), b["rank_acf1_in_shortlist30"], b["n_std_pct25_points"], b["rank_acf1_in_std_pct25"] if b["n_std_pct25_points"] else "—",
      b["shortlist_acf1_band_count"], f(b["shortlist_acf1_dist"]["min"]), f(b["valid_acf1_dist"]["min"])] for q, b in d["B3"].items()])
L.append("## B3b 反実仮想 CF1(診断のみ。採用候補ではない)\n")
tab(["q", "現行との重なり(8 点中)", "現行: std 比の中央値", "現行: max |log std 比|", "現行: 平均 |Δacf_1|", "CF1: std 比の中央値", "CF1: max |log std 比|", "CF1: 平均 |Δacf_1|"],
    [[q, b["CF1"]["overlap_with_current"], f(b["current"]["median_std_ratio"], 2), f(b["current"]["max_abs_log_std_ratio"], 2), f(b["current"]["mean_acf1_dist"]),
      f(b["CF1"]["median_std_ratio"], 2), f(b["CF1"]["max_abs_log_std_ratio"], 2), f(b["CF1"]["mean_acf1_dist"])] for q, b in d["B3"].items()])
C = d["C3"]
L.append("## C3a アトラスの尖度区分と seed 間の相対ばらつき(有効 170 点)\n")
tab(["尖度", "点数", "std_std/std の中央値", "P90", "最大"], [[k, x["n"], f(x["ratio_median"]), f(x["ratio_p90"]), f(x["ratio_max"])] for k, x in C["kurtosis_classes"].items()])
for cond, title in (("C3", "C3b 32 観測(主分析 C3)"), ("C0s", "C3c 32 観測(補助 C0s)")):
    L.append(f"## {title}\n")
    cc = C["conds"][cond]
    tab(["q", "delta", "wma", "アトラス尖度", "seed SD/std", "経路間 SD(std)", "P95−P5", "seed SD ÷ 経路間 SD", "seed SD ÷ (P95−P5)", "log(V_ens/σ²_atlas)", "位置ブロック std 最大/最小(1〜1000 本)"],
        [[r["q"], r["delta"], r["wma"], f(r["atlas_kurtosis"], 1), f(r["rel_seed_sd"]), f(r["path_std_sd"], 7), f(r["path_std_p95_minus_p5"], 7), f(r["seed_sd_over_path_sd"], 2),
          f(r["seed_sd_over_p95p5_range"], 2), f(r["log_V_ens_over_sigma2_atlas"], 2), f(r["pos_block_max_over_min_first_three"], 2)] for r in cc["rows"]])
    tab(["順位相関(32 観測)", "値"], [["log(V_ens/σ²_atlas) と アトラス尖度", f(cc["spearman_logratio_vs_atlas_kurtosis"])], ["log(V_ens/σ²_atlas) と seed SD/std", f(cc["spearman_logratio_vs_rel_seed_sd"])],
                                 ["log(V_ens/σ²_atlas) と アトラス acf_1", f(cc["spearman_logratio_vs_atlas_acf1"])]])
open(os.path.join(E.RESULTS, "s3_tables.md"), "w", encoding="utf-8").write("\n".join(L))
print("tables written")
