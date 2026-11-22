"""P5-B: render the result tables (markdown) from OUT/analysis.json. Usage: python p5b_tables.py > p5b_tables.md"""
import json
import os

import p5b_env as E

HZ = (1, 4, 12, 24)


def f(x, spec="{:.2f}", na="—"):
    return na if x is None else spec.format(x)


def fci(v, ci, spec="{:.2f}"):
    return f(v, spec) + ("" if not ci else f" [{spec.format(ci[0])}, {spec.format(ci[1])}]")


def main():
    A = json.load(open(os.path.join(E.OUT, "analysis.json"), encoding="utf-8"))
    print("### 表 A1: マッピングの M1 レベルの検査(推定期間、バケット別)\n")
    print("`real_std` はマッピングが使った窓ごとの標準偏差の平均(×1e-5)。`√V/real_std` は窓ごとの分散の平均の平方根との比。`log(σ_atlas/real_std)` は 8 候補のアトラスの std の符号つきの対数比の平均。"
          "`log(σ_mix/√V)` は分解の M1 項(候補内分散に基づく代理量)。被覆は、アトラスの有効点 {n} のうち実現値の std の ±10%・±25% 以内の点数。\n")
    print("| バケット | real_std ×1e5 | √V/real_std | 候補の log(σ_atlas/real_std) 平均 | M1 項 log(σ_mix/√V) | ±10% / ±25% の点数 | 実現値の std がアトラスの最小未満 | 30 点の shortlist の log 比の範囲 | best_std_ratio |\n|---|---|---|---|---|---|---|---|---|")
    for b in range(20):
        a = A["A"][str(b)]
        c = a["atlas_coverage"]
        print(f"| {b} | {a['real_stats']['std'] * 1e5:.2f} | {a['sqrtV_primary_over_real_std']:.3f} | {a['mean_log_std_ratio_vs_real_std']:+.3f} | {a['log_sigma_mix_over_sqrtV']:+.3f} | "
              f"{c['n_within_10pct']} / {c['n_within_25pct']} | {'はい' if c['real_std_below_atlas_min'] else 'いいえ'} | {c['shortlist_log_ratio_range'][0]:+.2f} 〜 {c['shortlist_log_ratio_range'][1]:+.2f} | {a['best_std_ratio']:.2f} |")
    ab = A["A"]["across_buckets"]
    print(f"\nバケット 0〜16 の順位相関(記述のみ): バケット番号と M1 項 {ab['spearman_bucket_vs_log_sigma_mix_over_sqrtV_0_16']:+.2f}、バケット番号と候補の log 比の平均 {ab['spearman_bucket_vs_mean_log_std_ratio_0_16']:+.2f}。"
          f"アトラスの有効点数 {A['A']['0']['atlas_coverage']['n_valid_points']}、アトラスの std の範囲 {A['A']['0']['atlas_coverage']['atlas_std_min']:.2e} 〜 {A['A']['0']['atlas_coverage']['atlas_std_max']:.2e}。\n")
    print("### 表 A2: 実現値の M1 統計量と 8 候補のアトラスの統計量(平均)\n")
    print("| バケット | acf_1 実/候補 | acf_5 実/候補 | acf_20 実/候補 | 尖度 実/候補 | volclust_50 実/候補 |\n|---|---|---|---|---|---|")
    for b in range(20):
        a = A["A"][str(b)]
        rs = a["real_stats"]
        cd = a["candidates"]
        m = lambda k: sum(x[k] for x in cd) / len(cd)  # noqa: E731
        print(f"| {b} | {rs['acf_1']:+.3f} / {m('acf_1'):+.3f} | {rs['acf_5']:+.3f} / {m('acf_5'):+.3f} | {rs['acf_20']:+.3f} / {m('acf_20'):+.3f} | "
              f"{rs['kurtosis']:.1f} / {m('kurtosis'):.1f} | {rs['volclust_50']:+.3f} / {m('volclust_50'):+.3f} |")
    print("\n### 表 A3: M5 — 現在の ATR と前向き M1 統計量の関係\n")
    print(f"バケットの ATR 中央値と前向き窓の標準偏差(√V)の両対数の傾き(20 バケット): {ab['loglog_slope_sqrtV_vs_atr_median']:.2f}(1 なら比例)。以下は、バケット内の原点ごとの、現在の ATR と前向き窓の標準偏差の順位相関(記述のみ)。\n")
    print("| バケット | " + " | ".join(str(b) for b in range(20)) + " |\n|---|" + "---|" * 20)
    print("| 順位相関 | " + " | ".join(f(A["A"][str(b)]["M5"]["within_bucket_spearman_atr_vs_window_std"]) for b in range(20)) + " |")
    print("| √V(全原点の窓)/√V(使用した 300 窓) | " + " | ".join(f"{A['A'][str(b)]['sqrtV_all_over_sqrtV_primary']:.3f}" for b in range(20)) + " |")
    print("\n### 表 B1: 実現値の分散比 VR_real(推定期間、バケット別。V_real,M1 は使用した 300 窓。[ ] は y の分散側の週ブロック bootstrap による記述的な 95% 区間、V は固定)\n")
    print("| バケット | h1 | h4 | h12(補助) | h24(規格化指標) |\n|---|---|---|---|---|")
    for b in range(20):
        r = A["B"][str(b)]
        print(f"| {b} | {fci(r['h1']['VR_real'], r['h1']['VR_real_ci'])} | {fci(r['h4']['VR_real'], r['h4']['VR_real_ci'])} | {f(r['h12']['VR_real'])} | {f(r['h24']['VR_real'])} |")
    print("\n### 表 B2: ATR 群別・期間別の VR_real(各期間の窓。検証・final は期間外の文脈。h12・h24 は規格化指標)\n")
    print("| 期間 | 群 | n(原点)| h1 | h4 | h12 | h24 |\n|---|---|---|---|---|---|---|")
    for per in ("estimation", "validation", "final"):
        for g in range(5):
            r = A["B"]["groups"][f"{per}|{g}"]
            print(f"| {per} | {g} | {r['n_origins']} | {fci(r['h1']['VR_real'], r['h1']['VR_real_ci'])} | {fci(r['h4']['VR_real'], r['h4']['VR_real_ci'])} | {f(r['h12']['VR_real'])} | {f(r['h24']['VR_real'])} |")
    print("\n### 表 C1: ショックあり・なしの保存済みアンサンブル間の観測差(C0s / C3)と E1a との比(推定期間、バケット別。90% 幅と標準偏差。seed 範囲は seed ごとの比の最小〜最大)\n")
    for h in (1, 4):
        print(f"\n**h{h}**\n\n| バケット | 幅比 C0s/C3 | 標準偏差比 C0s/C3(seed 範囲) | 幅 C0s/E1a | 幅 C3/E1a | 標準偏差 C0s/E1a | 標準偏差 C3/E1a |\n|---|---|---|---|---|---|---|")
        for b in range(20):
            c = A["C"][str(b)][f"h{h}"]
            print(f"| {b} | {c['w90_ratio_C0s_over_C3']:.2f} | {c['sd_ratio_C0s_over_C3']:.2f}({c['sd_ratio_seed_range'][0]:.2f}〜{c['sd_ratio_seed_range'][1]:.2f}) | {c['w90_ratio_C0s_over_E1a']:.2f} | "
                  f"{c['w90_ratio_C3_over_E1a']:.2f} | {c['sd_ratio_C0s_over_E1a']:.2f} | {c['sd_ratio_C3_over_E1a']:.2f} |")
    print("\n### 表 C2: DDM 側の暗黙の分散比(C3、仮定 A_config のもとで)と実現値の VR_real(推定期間、バケット別。候補別の範囲は 8 候補)\n")
    print("| バケット | h1 VR_C3(混合) | 候補の範囲 | VR_real | h4 VR_C3(混合) | 候補の範囲 | VR_real |\n|---|---|---|---|---|---|---|")
    for b in range(20):
        cells = []
        for h in (1, 4):
            c = A["C"][str(b)][f"h{h}"]
            vc = c["VR_candidate"]["C3"]
            cells += [f"{c['VR_mix_C3']:.2f}", f"{min(vc):.2f}〜{max(vc):.2f}", f"{A['B'][str(b)][f'h{h}']['VR_real']:.2f}"]
        print(f"| {b} | " + " | ".join(cells) + " |")
    mc = max(A["C"][str(b)]["h1"]["mean_check"]["C3"]["max_abs_mean_over_sd_candidates"] for b in range(20))
    print(f"\n候補の 60 分足の累積収益の平均の絶対値 / 標準偏差の最大(h1、C3): {mc:.2f}(候補間の平均の違いの大きさの間接的な目安)。\n")
    print("### 表 D1: 対数尺度比の分解 log(sd_C0s / sd_E1a) = M1 + M3 + M2(推定期間、バケット別。A_config のもとでの代数的な分解)\n")
    for h in (1, 4):
        print(f"\n**h{h}**\n\n| バケット | 全体 [区間] | M1(M1 レベルのずれ) | M3(時間集約のずれ)[区間] | M2(ショックあり・なしの観測差) | VR_real | VR_C3 |\n|---|---|---|---|---|---|---|")
        for b in range(20):
            d = A["D"][str(b)][f"h{h}"]
            print(f"| {b} | {fci(d['total'], d['total_ci'], '{:+.2f}')} | {d['M1']:+.2f} | {fci(d['M3'], d['M3_ci'], '{:+.2f}')} | {d['M2']:+.2f} | {d['VR_real']:.2f} | {d['VR_C3']:.2f} |")
    print("\n### 表 D2: バケット 0〜16 での各項の ATR 依存(バケット番号に対する OLS の傾きと順位相関。記述のみ)\n")
    print("| h | 傾き 全体 | 傾き M1 | 傾き M3 | 傾き M2 | 傾きの比 M1 / M3 / M2 | 順位相関 全体 / M1 / M3 / M2 |\n|---|---|---|---|---|---|---|")
    for h in HZ:
        s = A["D"]["across_buckets"][f"h{h}"]
        o, sh, sp = s["ols_slope_vs_bucket_0_16"], s["slope_share_of_total_0_16"], s["spearman_vs_bucket_0_16"]
        print(f"| {h} | {o['total']:+.3f} | {o['M1']:+.3f} | {o['M3']:+.3f} | {o['M2']:+.3f} | {sh['M1']:+.2f} / {sh['M3']:+.2f} / {sh['M2']:+.2f} | "
              f"{f(sp['total'])} / {f(sp['M1'])} / {f(sp['M3'])} / {f(sp['M2'])} |")
    print("\n### 表 D3: 90% 幅の対数比(推定期間、h1・h4。M1・M3 は幅では加法的に分解しない)\n")
    print("| バケット | h1 log(C0s/E1a) | h1 log(C3/E1a) | h1 log(C0s/C3) | h4 log(C0s/E1a) | h4 log(C3/E1a) | h4 log(C0s/C3) |\n|---|---|---|---|---|---|---|")
    for b in range(20):
        w1, w4 = A["D"][str(b)]["h1"]["w90"], A["D"][str(b)]["h4"]["w90"]
        print(f"| {b} | {fci(w1['log_C0s_over_E1a'], w1['log_C0s_over_E1a_ci'], '{:+.2f}')} | {fci(w1['log_C3_over_E1a'], w1['log_C3_over_E1a_ci'], '{:+.2f}')} | {w1['log_C0s_over_C3']:+.2f} | "
              f"{fci(w4['log_C0s_over_E1a'], w4['log_C0s_over_E1a_ci'], '{:+.2f}')} | {fci(w4['log_C3_over_E1a'], w4['log_C3_over_E1a_ci'], '{:+.2f}')} | {w4['log_C0s_over_C3']:+.2f} |")
    print("\n### 表 E: 候補集合の被覆(h1・h4。8 候補の 90% 幅が E1a の幅より広い候補の数、E1a の幅が候補の範囲内か)\n")
    print("| バケット | C0s h1 広い候補数 | C0s h1 範囲内 | C3 h1 広い候補数 | C3 h1 範囲内 | C0s h4 広い候補数 | C0s h4 範囲内 | C3 h4 広い候補数 | C3 h4 範囲内 |\n|---|---|---|---|---|---|---|---|---|")
    for b in range(20):
        cells = []
        for h in (1, 4):
            for cond in ("C0s", "C3"):
                e = A["E"][str(b)][cond][f"h{h}"]
                cells += [str(e["n_wider_than_E1a"]), "はい" if e["E1a_inside_candidate_range"] else "いいえ"]
        order = [0, 1, 4, 5, 2, 3, 6, 7]
        print(f"| {b} | " + " | ".join(cells[i] for i in order) + " |")


if __name__ == "__main__":
    main()
