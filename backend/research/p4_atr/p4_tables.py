"""P4: render the result tables (markdown) from OUT/analysis_*.json. Usage: python p4_tables.py > p4_tables.md"""
import json
import os
import sys

import p4_env as E

HZ = (1, 4, 12, 24)
PERS = ("estimation", "validation", "final")


def load(name):
    return json.load(open(os.path.join(E.OUT, name), encoding="utf-8"))


def ci(x, s, f="{:+.2f}"):
    return f"{f.format(x['est'] * s)} [{f.format(x['ci'][0] * s)}, {f.format(x['ci'][1] * s)}]"


def decomp_table(A, metric, scale, unit):
    print(f"| 比較 | h | 全体の差({unit}) | 構成の寄与(Shapley) | 群内の寄与(Shapley) | 構成(順序1/順序2) | 群内(順序1/順序2) |\n|---|---|---|---|---|---|---|")
    for pair in ("validation->final", "estimation->validation", "estimation->final"):
        for h in HZ:
            d = A["decomposition"][f"{metric}|h{h}|{pair}"]
            print(f"| {pair} | {h} | {ci(d['total'], scale)} | {ci(d['composition_shapley'], scale)} | {ci(d['within_shapley'], scale)} | "
                  f"{d['composition_order1']['est'] * scale:+.2f} / {d['composition_order2']['est'] * scale:+.2f} | "
                  f"{d['within_order1']['est'] * scale:+.2f} / {d['within_order2']['est'] * scale:+.2f} |")


def main():
    A = load("analysis_A.json")
    B = load("analysis_B.json")["cells"]
    Cc = load("analysis_C.json")["cells"]
    print("### 表 A1a: 90% 被覆率の期間間の差の分解(単位 pp。[ ] は週ブロック bootstrap の記述的な 95% 区間)\n")
    decomp_table(A, "cov90", 100, "pp")
    print("\n### 表 A1b: 平均 CRPS の期間間の差の分解(単位 1e-5。CRPS は価格変動の尺度に比例するため、構成の寄与には ATR 水準の違いが機械的に含まれる)\n")
    decomp_table(A, "crps", 1e5, "1e-5")
    print("\n### 表 A2: DDM(C0s の 3 seed 平均)の ATR 群別の指標(n / 週数、90% 被覆率、k90、平均 CRPS ×1e5)\n")
    for h in HZ:
        print(f"\n**h{h}**\n\n| ATR 群 | 推定 | 検証 | final |\n|---|---|---|---|")
        for g in "01234":
            cells = []
            for per in PERS:
                t = A["group_tables"][f"{per}|h{h}"][g]
                cells.append(f"{t['n']} / {t['weeks']}週、{t['cov90']:.3f}、{t['k90']:.2f}、{t['crps'] * 1e5:.1f}")
            print(f"| {g} | " + " | ".join(cells) + " |")
    print("\n### 表 A3: 同じ ATR 群内の実現値 y の 90% 幅(q95−q05)の期間比\n\n| h | 比較 | 群 0 | 群 1 | 群 2 | 群 3 | 群 4 |\n|---|---|---|---|---|---|---|")
    for h in HZ:
        for pa, pb in (("validation", "estimation"), ("final", "validation"), ("final", "estimation")):
            r = A["y_quantile_ratios"][f"h{h}|{pa} / {pb}"]
            print(f"| {h} | {pa}/{pb} | " + " | ".join(f"{r[g]['q95/q05 width ratio']:.2f}" for g in "01234") + " |")
    print("\n### 表 B1: ATR バケットだけの経験分布ベースラインと DDM(B-1: 推定期間のみで当てはめ。推定期間は標本内)\n")
    print("| ベースライン | 評価期間 | h | DDM 被覆率 / k90 | ベースライン 被覆率 / k90 | ベースライン P(y>m) | ベースライン − DDM の相対 CRPS [95% 区間] | F1 統計モデル − DDM の相対 CRPS |\n|---|---|---|---|---|---|---|---|")
    for kind in ("E1a", "E1b", "E1c"):
        for per in PERS:
            for h in HZ:
                v = B[f"B-1|{kind}|{per}|h{h}"]
                st = f"{v['stat_model']['crps'] / v['ddm']['crps'] - 1:+.2%}" if "stat_model" in v else "—"
                tag = per + ("(標本内)" if per == "estimation" else "")
                print(f"| {kind} | {tag} | {h} | {v['ddm']['cov90']:.3f} / {v['ddm']['k90']:.2f} | {v['baseline']['cov90']:.3f} / {v['baseline']['k90']:.2f} | "
                      f"{v['baseline']['P_up']:.3f} | {ci(v['crps_rel_vs_ddm'], 100, '{:+.2f}')}% | {st} |")
    print("\n### 表 B2: B-2(推定+検証期間で当てはめ、final で評価)\n")
    print("| ベースライン | h | DDM 被覆率 / k90 | ベースライン 被覆率 / k90 | ベースライン − DDM の相対 CRPS [95% 区間] | B-1 の同じ値 |\n|---|---|---|---|---|---|")
    for kind in ("E1a", "E1b", "E1c"):
        for h in HZ:
            v = B[f"B-2|{kind}|final|h{h}"]
            v1 = B[f"B-1|{kind}|final|h{h}"]
            print(f"| {kind} | {h} | {v['ddm']['cov90']:.3f} / {v['ddm']['k90']:.2f} | {v['baseline']['cov90']:.3f} / {v['baseline']['k90']:.2f} | "
                  f"{ci(v['crps_rel_vs_ddm'], 100, '{:+.2f}')}% | {v1['crps_rel_vs_ddm']['est'] * 100:+.2f}% (被覆率 {v1['baseline']['cov90']:.3f}) |")
    print("\n### 表 B3: ATR 群別(E1a、B-1)。90% 被覆率(DDM / E1a)と、E1a − DDM の相対 CRPS\n")
    print("| 期間 | h | 群 0 | 群 1 | 群 2 | 群 3 | 群 4 |\n|---|---|---|---|---|---|---|")
    for per in ("validation", "final"):
        for h in HZ:
            v = B[f"B-1|E1a|{per}|h{h}"]
            print(f"| {per} | {h} | " + " | ".join(f"{v['ddm_by_group'][g]['cov90']:.3f} / {v['baseline_by_group'][g]['cov90']:.3f}、{v['crps_rel_vs_ddm_by_group'][g]['est'] * 100:+.2f}%"
                                                  for g in "01234") + " |")
    print("\n### 表 C: 追加の条件づけ変数 1 つ(ATR バケット × 3 区分)と E1a(同じ評価原点集合)\n")
    print("| 変数 | 期間 | h | 使用した評価原点(除外率)/ 週数 | 推定セル最小 n | 条件づけ 被覆率 / k90 | E1a 被覆率 / k90 | 条件づけ − E1a の相対 CRPS [95% 区間] | DDM − E1a の相対 CRPS |\n|---|---|---|---|---|---|---|---|---|")
    for k, v in Cc.items():
        var, per, h = k.split("|")
        print(f"| {var} | {per} | {h[1:]} | {v['n_usable']}({v['excluded_fraction']:.0%})/ {v['weeks_usable']} | {v['fit_cell_n']['min']} | "
              f"{v['cond']['cov90']:.3f} / {v['cond']['k90']:.2f} | {v['E1a']['cov90']:.3f} / {v['E1a']['k90']:.2f} | {ci(v['crps_rel_cond_vs_E1a'], 100, '{:+.2f}')}% | "
              f"{v['crps_rel_ddm_vs_E1a']['est'] * 100:+.2f}% |")
    print("\n### 表 D: P3 の保存済み DDM 変種への再適用(任意)。検証 → final の 90% 被覆率差の分解(pp)と、E1a(B-1)− 変種の相対 CRPS\n")
    print("| 変種 | h | 全体 | 構成(Shapley) | 群内(Shapley) | E1a − 変種 CRPS(検証) | E1a − 変種 CRPS(final) | 変種 被覆率(検証 / final) |\n|---|---|---|---|---|---|---|---|")
    for var in ("C0s", "C2_60", "C3"):
        suf = "" if var == "C0s" else f"_{var}"
        Av = load(f"analysis_A{suf}.json")
        Bv = load(f"analysis_B{suf}.json")["cells"]
        for h in HZ:
            d = Av["decomposition"][f"cov90|h{h}|validation->final"]
            bv, bf = Bv[f"B-1|E1a|validation|h{h}"], Bv[f"B-1|E1a|final|h{h}"]
            print(f"| {var} | {h} | {ci(d['total'], 100)} | {ci(d['composition_shapley'], 100)} | {ci(d['within_shapley'], 100)} | "
                  f"{bv['crps_rel_vs_ddm']['est'] * 100:+.2f}% | {bf['crps_rel_vs_ddm']['est'] * 100:+.2f}% | {bv['ddm']['cov90']:.3f} / {bf['ddm']['cov90']:.3f} |")


if __name__ == "__main__":
    main()
