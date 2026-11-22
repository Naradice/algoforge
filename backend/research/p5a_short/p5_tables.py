"""P5-A: render the result tables (markdown) from OUT/analysis_*.json. Usage: python p5_tables.py > p5_tables.md"""
import json
import os

import p5_env as E

HZ = (1, 4, 12, 24)
QS = (0.01, 0.025, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.975, 0.99)
IX = {p: i for i, p in enumerate(QS)}
KINDS = ("T0", "T1", "T2", "T3", "T4", "E1a")
NAMES = {"T0": "T0 DDM", "T1": "T1 共通尺度", "T2": "T2 バケット別尺度", "T3": "T3 PIT(共通)", "T4": "T4 PIT(バケット別)", "E1a": "E1a"}


def load(name):
    return json.load(open(os.path.join(E.OUT, name), encoding="utf-8"))


def f(x, spec="{:.2f}", na="—"):
    return na if x is None else spec.format(x)


def fci(v, lo, hi, spec="{:.2f}"):
    return f(v, spec) + ("" if lo is None or hi is None else f" [{spec.format(lo)}, {spec.format(hi)}]")


def rci(rec, key, p, ci_key):
    i = IX[p]
    c = rec[ci_key]
    return fci(rec[key][i], c["lo"][i], c["hi"][i])


def pct(x):
    return "—" if x is None else f"{x * 100:+.2f}%"


def pci(d):
    return f"{d['est'] * 100:+.2f}% [{d['ci'][0] * 100:+.2f}, {d['ci'][1] * 100:+.2f}]"


def main():
    AB = load("analysis_AB.json")
    C = load("analysis_C.json")
    D = load("analysis_D.json")
    print("### 表 A1: バケット別(推定期間)の DDM と E1a の分位点の比較(DDM は C0s の 3 seed をプール。[ ] は E1a 側の週ブロック bootstrap による記述的な 95% 区間)\n")
    print("上側は `r+(p) = |Q_DDM(p)−Q_DDM(.5)| / |Q_E1a(p)−Q_E1a(.5)|`(p>0.5)、下側は `r−(p)`(p<0.5)。形状は上側が `r+(p)/r+(0.75)`、下側が `r−(p)/r−(0.25)`。位置は `(med_DDM−med_E1a)/IQR_E1a`。\n")
    for h in HZ:
        print(f"\n**h{h}**{'(補助)' if h > 4 else ''}\n\n| バケット | n | 位置 | r−(0.05) | r−(0.25) | r+(0.75) | r+(0.95) | 形状−(0.05) | 形状+(0.95) | 未定義 |\n|---|---|---|---|---|---|---|---|---|---|")
        for k in range(20):
            r = AB["buckets"][str(k)][f"h{h}"]
            print(f"| {k} | {r['n_e1a']} | {f(r['loc'], '{:+.2f}')} | {rci(r, 'r', 0.05, 'r_ci')} | {f(r['r'][IX[0.25]])} | {f(r['r'][IX[0.75]])} | {rci(r, 'r', 0.95, 'r_ci')} | "
                  f"{f(r['shape'][IX[0.05]])} | {f(r['shape'][IX[0.95]])} | {r['n_undefined_r']} |")
    print("\n### 表 A2: ATR 群別・期間別の DDM(混合)と実現値 y の分位点の比較(r は DDM/実現値。[ ] は実現値側の週ブロック bootstrap による記述的な 95% 区間)\n")
    for h in HZ:
        print(f"\n**h{h}**{'(補助)' if h > 4 else ''}\n\n| 期間 | 群 | n / 週 | 位置 | r−(0.05) | r−(0.25) | r+(0.75) | r+(0.95) | 形状−(0.05) | 形状+(0.95) |\n|---|---|---|---|---|---|---|---|---|---|")
        for per in ("estimation", "validation", "final"):
            for g in range(5):
                r = AB["groups"][f"{per}|{g}"][f"h{h}"]
                v = r["ddm_vs_real"]
                print(f"| {per} | {g} | {r['n']} / {r['weeks']} | {f(v['loc'], '{:+.2f}')} | {rci(v, 'r', 0.05, 'r_ci')} | {f(v['r'][IX[0.25]])} | {f(v['r'][IX[0.75]])} | {rci(v, 'r', 0.95, 'r_ci')} | "
                      f"{f(v['shape'][IX[0.05]])} | {f(v['shape'][IX[0.95]])} |")
    print("\n### 表 B1: 90% 幅の対数をホライズンの対数に対して見た傾き(ATR 群別・期間別。DDM / E1a(混合)/ 実現値 y)。差は DDM − 実現値([ ] は記述的な 95% 区間)\n")
    print("| 期間 | 群 | h1→h4 DDM / 実現値(差) | h4→h12 DDM / 実現値(差) | h12→h24 DDM / 実現値(差) |\n|---|---|---|---|---|")
    for per in ("estimation", "validation", "final"):
        for g in range(5):
            s = AB["groups"][f"{per}|{g}"]["slopes"]
            cells = []
            for pair in ("h1->h4", "h4->h12", "h12->h24"):
                x = s[pair]
                cells.append(f"{x['ddm']:.2f} / {x['real']:.2f}(差 {fci(x['diff_ddm_minus_real'], x['diff_ci']['ci'][0] if x['diff_ci']['ci'] else None, x['diff_ci']['ci'][1] if x['diff_ci']['ci'] else None, '{:+.2f}')})")
            print(f"| {per} | {g} | " + " | ".join(cells) + " |")
    print("\n### 表 B2: バケット別(推定期間)の傾き。DDM と E1a(差 DDM − E1a、[ ] は E1a 側の記述的な 95% 区間)\n")
    print("| バケット | h1→h4 DDM / E1a(差) | h4→h12 DDM / E1a(差) | h12→h24 DDM / E1a(差) |\n|---|---|---|---|")
    for k in range(20):
        s = AB["buckets"][str(k)]["slopes"]
        cells = []
        for pair in ("h1->h4", "h4->h12", "h12->h24"):
            x = s[pair]
            ci = x["diff_ci"]["ci"]
            cells.append(f"{x['ddm']:.2f} / {x['e1a']:.2f}(差 {fci(x['diff_ddm_minus_e1a'], ci[0] if ci else None, ci[1] if ci else None, '{:+.2f}')})")
        print(f"| {k} | " + " | ".join(cells) + " |")
    print("\n### 表 C0: 推定期間のみで決めた係数(診断用。再較正ではない)\n")
    print("| h | T1 共通尺度 s_h | T2 バケット別 s_{b,h}(バケット 0〜19) |\n|---|---|---|")
    for h in HZ:
        print(f"| {h} | {C['fits']['s1'][str(h)]:.2f} | " + ", ".join(f"{x:.2f}" for x in C["fits"]["s2"][str(h)]) + " |")
    print(f"\nT0g(格子表現)と T0(標本表現)の平均 CRPS の相対誤差: 最大 {C['approx_error_T0g_vs_T0']['max']:.2e}(許容値 {C['approx_error_T0g_vs_T0']['tolerance']:.0e} を超えたセル数 {C['approx_error_T0g_vs_T0']['n_over']})。\n")
    print("### 表 C1: 変換 T0〜T4 と E1a(プール)。90% 被覆率 / k90、DDM(T0)に対する相対 CRPS [記述的 95% 区間]、E1a との差のうち閉じた割合(DDM と E1a の差が 1% 未満なら表示しない)\n")
    for per in ("validation", "final", "estimation"):
        print(f"\n**{per}**{'(標本内。独立な評価ではない)' if per == 'estimation' else ''}\n")
        print("| h | T0−E1a の差 | " + " | ".join(NAMES[k] for k in KINDS) + " |\n|---|---|" + "---|" * len(KINDS))
        for h in HZ:
            row = C["cells"][f"{per}|h{h}"]
            cells = []
            for k in KINDS:
                r = row[k]
                s = f"{f(r['pooled']['cov90'], '{:.3f}')} / {f(r['pooled']['k90'])}"
                if k not in ("T0", "E1a"):
                    s += f"<br>{pci(r['crps_rel_vs_T0'])}<br>閉じた割合 {f(r.get('gap_closed_pooled'))}"
                cells.append(s)
            print(f"| {h} | {pct(row['T0']['gap_T0_E1a_rel'])} | " + " | ".join(cells) + " |")
    print("\n### 表 C2: ATR 群別の 90% 被覆率(期間外。T0 / T1 / T2 / T3 / T4 / E1a)と、E1a との差のうち閉じた割合(T1 / T2 / T3 / T4)\n")
    print("| 期間 | h | 群 | 被覆率 T0 / T1 / T2 / T3 / T4 / E1a | 閉じた割合 T1 / T2 / T3 / T4 |\n|---|---|---|---|---|")
    for per in ("validation", "final"):
        for h in (1, 4, 12, 24):
            row = C["cells"][f"{per}|h{h}"]
            for g in range(5):
                cov = " / ".join(f(row[k]["by_group"][str(g)]["cov90"], "{:.3f}") for k in KINDS)
                gc = " / ".join(f(row[k]["gap_closed_by_group"][str(g)]) for k in ("T1", "T2", "T3", "T4"))
                print(f"| {per} | {h} | {g} | {cov} | {gc} |")
    print("\n### 表 D1: 候補ごとの 90% 幅と E1a の比(推定期間、3 seed プール)。混合は現在の 8 候補混合、候補の幅の最小・最大、E1a より広い候補の数\n")
    for h in (1, 4, 12, 24):
        print(f"\n**h{h}**{'(補助)' if h > 4 else ''}\n\n| バケット | 混合 / E1a | 候補の最小 / 最大(E1a 比) | E1a より広い候補数 | 選ばれた単一候補 | 選ばれた 2 候補 | 推定期間の CRPS 改善(単一 / 2 候補) |\n|---|---|---|---|---|---|---|")
        for k in range(20):
            c = D["candidates"][f"h{h}"][k]
            s = D["selection"][f"h{h}"][k]
            print(f"| {k} | {c['mix8_ratio_to_e1a']:.2f} | {min(c['cand_ratio_to_e1a']):.2f} / {max(c['cand_ratio_to_e1a']):.2f} | {c['n_candidates_wider_than_e1a']} | {s['single']} | "
                  f"{tuple(s['pair'])} | {pct(s['est_improvement_single'])} / {pct(s['est_improvement_pair'])} |")
    print("\n### 表 D2: 候補選択の安定性(seed 別に同じ規則で選んだ場合の一致率。偶然の一致率は単一 0.125、2 候補の完全一致 0.036)\n")
    print("| h | 単一: seed 間の一致 | 単一: seed 対 プール | 2 候補: seed 間の完全一致 | 2 候補: seed 対 プールの完全一致 | 2 候補: 1 候補以上共有 |\n|---|---|---|---|---|---|")
    for h in HZ:
        s = D["stability"][f"h{h}"]
        print(f"| {h} | {s['single_seed_pairs_equal_frac']:.2f} | {s['single_seed_vs_pooled_equal_frac']:.2f} | {s['pair_seed_pairs_equal_frac']:.2f} | {s['pair_seed_vs_pooled_equal_frac']:.2f} | "
              f"{s['pair_seed_vs_pooled_share_a_candidate_frac']:.2f} |")
    print("\n### 表 D3: 推定期間で選んだ部分アンサンブルの評価(3 seed プールの表現。平均 CRPS の 8 候補混合に対する相対差 [記述的 95% 区間] と 90% 被覆率)\n")
    print("| 期間 | h | 8 候補混合の被覆率 | 単一候補 | 2 候補 | E1a |\n|---|---|---|---|---|---|")
    for per in ("validation", "final", "estimation"):
        for h in HZ:
            r = D["eval"][f"{per}|h{h}"]
            cells = [f"{r['mix8']['pooled']['cov90']:.3f}"]
            for k in ("single", "pair", "E1a"):
                cells.append(f"{r[k]['pooled']['cov90']:.3f}<br>{pci(r[k]['crps_rel_vs_mix8'])}")
            print(f"| {per}{'(標本内)' if per == 'estimation' else ''} | {h} | " + " | ".join(cells) + " |")
    print("\n### 表 D4: S5 — バケット別の幅の比(DDM / E1a の 90% 幅、推定期間)とマッピングの記述子の順位相関(記述のみ、n = 20、検定しない)\n")
    for h in (1, 4):
        s = D["s5"][f"h{h}"]
        print(f"\n**h{h}**: 幅の比 = " + ", ".join(f"{x:.2f}" for x in s["width90_ratio_ddm_over_e1a"]) + f"。定数のため相関を計算しない記述子: {', '.join(s['constant_descriptors'])}\n")
        print("| 記述子 | Spearman の順位相関 |\n|---|---|")
        for k, v in s["spearman_rho"].items():
            print(f"| {k} | {f(v, '{:+.2f}')} |")
    print("\n### 表 E: P3 の保存済み変種への再適用(A・C)。最終期間・検証期間の E1a との差のうち閉じた割合(T1 / T2 / T3 / T4。差が 1% 未満なら表示しない)と T0 の被覆率\n")
    print("| 変種 | 期間 | h | T0 被覆率 | T0 − E1a の差 | 閉じた割合 T1 / T2 / T3 / T4 | 係数 s_h(T1) |\n|---|---|---|---|---|---|---|")
    for v in ("C2_60", "C3"):
        Ev = load(f"analysis_E_{v}.json")["C"]
        for per in ("validation", "final"):
            for h in HZ:
                row = Ev["cells"][f"{per}|h{h}"]
                gc = " / ".join(f(row[k].get("gap_closed_pooled")) for k in ("T1", "T2", "T3", "T4"))
                print(f"| {v} | {per} | {h} | {row['T0']['pooled']['cov90']:.3f} | {pct(row['T0']['gap_T0_E1a_rel'])} | {gc} | {Ev['fits']['s1'][str(h)]:.2f} |")


if __name__ == "__main__":
    main()
