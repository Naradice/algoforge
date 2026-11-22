"""P5-B stage 2: render the result tables (markdown) from OUT/analysis2.json. Usage: python s2_tables.py > s2_tables.md"""
import json
import math
import os

import s2_env as E

HZ = (1, 4, 12, 24)
BK = (0, 8, 15, 19)


def f(x, spec="{:.3f}", na="—"):
    return na if x is None else spec.format(x)


def main():
    A = json.load(open(os.path.join(E.OUT, "analysis2.json"), encoding="utf-8"))["cells"]
    print("### 表 A2a: C3 の候補別: 最初の 1,000 本の M1 分散とアトラスの σ²_atlas、アトラスの統計量との比較(経路平均 ± 標準誤差。アトラスは平均 ± seed 間の標準偏差)\n")
    print("`log(V_ens/σ²_atlas)` の標準誤差は、経路間の標準誤差とアトラスの seed 間の標準誤差(delta 法の近似)の合成。\n")
    print("| バケット | 候補 (delta, wma) | n | log(V_ens/σ²_atlas) ± SE | log(経路平均 std / アトラス std) ± SE | acf_1 DDM / アトラス | acf_5 DDM / アトラス | 尖度 DDM / アトラス | volclust_50 DDM / アトラス |\n|---|---|---|---|---|---|---|---|")
    for q in BK:
        for c in A[str(q)]["candidates"]:
            e = c["C3"]
            am, ase = e["atlas_stats_mean"], e["atlas_stats_se"]
            at, sa = c["atlas"], c["atlas_std_across_seeds"]
            m_, se_ = am["std"], ase["std"]
            lr_std = math.log(m_ / at["std"])
            se_std = math.sqrt((se_ / m_) ** 2 + (sa["std"] / math.sqrt(5) / at["std"]) ** 2)
            print(f"| {q} | ({c['delta']}, {c['wma']}) | {c['n']} | {c['A2_log_ratio_var_vs_atlas']:+.3f} ± {c['A2_se_log_ratio']:.3f} | {lr_std:+.3f} ± {se_std:.3f} | "
                  f"{am['acf_1']:+.3f}±{ase['acf_1']:.3f} / {at['acf_1']:+.3f}±{sa['acf_1']:.3f} | {am['acf_5']:+.3f}±{ase['acf_5']:.3f} / {at['acf_5']:+.3f}±{sa['acf_5']:.3f} | "
                  f"{am['kurtosis']:.1f}±{ase['kurtosis']:.1f} / {at['kurtosis']:.1f}±{sa['kurtosis']:.1f} | {am['volclust_50']:+.3f} / {at['volclust_50']:+.3f} |")
    print("\n### 表 A2b: バケットの混合(C3)の M1 分散と、位置ブロック別の標準偏差(初期の過渡の確認)\n")
    print("`V_pooled` は全経路の収益をプールした分散(経路間の平均の違いを含む)、`V_within` は経路内分散の平均。`σ²_atlas,mix` は第 1 段階の代理量(候補内分散の等重み平均)。\n")
    print("| バケット | σ²_atlas,mix ×1e-10 | V_pooled ×1e-10 | V_within ×1e-10 | log(V_pooled/σ²_atlas,mix) | log(V_within/σ²_atlas,mix) | V_pooled/V_within | 位置ブロック別 std(1〜100 / 101〜200 / 201〜1000 / 1001〜1440、×1e-5) |\n|---|---|---|---|---|---|---|---|")
    for q in BK:
        b = A[str(q)]["bucket"]
        c3 = b["C3"]
        print(f"| {q} | {b['sigma2_atlas_mix_proxy'] * 1e10:.2f} | {c3['V_mix_pooled'] * 1e10:.2f} | {c3['V_mix_within_mean'] * 1e10:.2f} | {b['A2_log_ratio_mix_var_vs_atlas']['pooled']:+.3f} | "
              f"{b['A2_log_ratio_mix_var_vs_atlas']['within_mean']:+.3f} | {c3['V_pooled_over_within']:.3f} | " + " / ".join(f"{x * 1e5:.2f}" for x in c3["pos_std_mean"]) + " |")
    print("\n### 表 B2: ショックあり(C0s)と C3 の M1 レベルの観測差(バケットの混合。観測差であり純粋なショック効果ではない)\n")
    print("| バケット | V_pooled C0s/C3 | acf_1 C3 / C0s | acf_5 C3 / C0s | 尖度 C3 / C0s | volclust_50 C3 / C0s | 位置ブロック別 std の比(C0s/C3、1〜100 / 101〜200 / 201〜1000 / 1001〜1440) |\n|---|---|---|---|---|---|---|")
    for q in BK:
        b = A[str(q)]["bucket"]
        a3, a0 = b["C3"]["atlas_stats_mean"], b["C0s"]["atlas_stats_mean"]
        pr = [x / y for x, y in zip(b["C0s"]["pos_std_mean"], b["C3"]["pos_std_mean"])]
        print(f"| {q} | {b['C0s']['V_mix_pooled'] / b['C3']['V_mix_pooled']:.3f} | {a3['acf_1']:+.3f} / {a0['acf_1']:+.3f} | {a3['acf_5']:+.3f} / {a0['acf_5']:+.3f} | "
              f"{a3['kurtosis']:.1f} / {a0['kurtosis']:.1f} | {a3['volclust_50']:+.3f} / {a0['volclust_50']:+.3f} | " + " / ".join(f"{x:.3f}" for x in pr) + " |")
    print("\n### 表 C2a: 窓内(経路内)の分散比 VR_win(最初の 1,000 本)と、ACF から導かれる理論値(弱定常性を仮定した診断)\n")
    print("| バケット | 系列 | VR_win(5) 測定 / ACF 理論 | VR_win(20) 測定 / 理論 | VR_win(60) 測定 / 理論 | ACF ラグ 1 / 5 / 20 / 40 / 60 |\n|---|---|---|---|---|---|")
    for q in BK:
        cell = A[str(q)]
        rows = [("実現値(300 窓)", cell["real"]["vr_win_mean"], cell["real"]["implied_vr"], cell["real"]["acf_mean"]),
                ("DDM C3", cell["bucket"]["C3"]["vr_win_mean"], cell["bucket"]["C3"]["implied_vr"], cell["bucket"]["C3"]["acf_mean"]),
                ("DDM C0s", cell["bucket"]["C0s"]["vr_win_mean"], cell["bucket"]["C0s"]["implied_vr"], cell["bucket"]["C0s"]["acf_mean"])]
        for name, vr, im, acf in rows:
            print(f"| {q} | {name} | {vr['5']:.3f} / {im['5']:.3f} | {vr['20']:.3f} / {im['20']:.3f} | {vr['60']:.3f} / {im['60']:.3f} | "
                  + " / ".join(f"{acf[j - 1]:+.3f}" for j in (1, 5, 20, 40, 60)) + " |")
    print("\n### 表 C2b: 経路間の分散比 VR_ens(バケットの混合。D2 の分解に使う量)。分母はプール分散(主)と経路内分散の平均(変種)\n")
    print("| バケット | 条件 | h1 プール / 経路内平均 | h4 プール / 経路内平均 | h12 プール / 経路内平均 | h24 プール / 経路内平均 | 実現値 VR_real h1 / h4(第 1 段階) | 代理量 VR_C3 h1 / h4(第 1 段階) |\n|---|---|---|---|---|---|---|---|")
    s1 = json.load(open(os.path.join(E.STAGE1_RESULTS, "analysis.json"), encoding="utf-8"))
    for q in BK:
        for cond in ("C3", "C0s"):
            b = A[str(q)]["bucket"][cond]
            cells = [f"{b['VR_ens_mix_pooled'][str(h)]:.3f} / {b['VR_ens_mix_within'][str(h)]:.3f}" for h in HZ]
            vr_real = " / ".join(f"{s1['B'][str(q)][f'h{h}']['VR_real']:.3f}" for h in (1, 4))
            vr_proxy = " / ".join(f"{s1['D'][str(q)][f'h{h}']['VR_C3']:.3f}" for h in (1, 4))
            print(f"| {q} | {cond} | " + " | ".join(cells) + f" | {vr_real} | {vr_proxy if cond == 'C3' else '—'} |")
    print("\n### 表 C2c: 経路間の分散比の開始位置による変化(混合、プール分散で規格化。60 本・240 本の和の経路間分散 / (k × V_pooled)。初期状態・過渡の診断)\n")
    print("| バケット | 条件 | k=60: 開始 0 / 120 / 360 / 720 | k=240: 開始 0 / 120 / 360 / 720 |\n|---|---|---|---|")
    for q in BK:
        for cond in ("C3", "C0s"):
            o = A[str(q)]["bucket"][cond]["cross_path_vr_by_start_offset"]
            print(f"| {q} | {cond} | " + " / ".join(f"{o['60'][t]:.3f}" for t in ('0', '120', '360', '720')) + " | " + " / ".join(f"{o['240'][t]:.3f}" for t in ('0', '120', '360', '720') if t in o['240']) + " |")
    print("\n### 表 D2: 分解の再評価(測定した M1 分散による M1'・M3' と、第 1 段階の代理量による M1・M3。和 M1+M3 と M2 は不変。対数比)\n")
    print("| バケット | h | M1(代理) | M1'(プール) | M1'(経路内平均) | M3(代理) | M3'(プール) | M3'(経路内平均) | M2 | M2 の M1 レベル(プール) | M2 の集約(プール) | VR_C3 測定(プール) |\n|---|---|---|---|---|---|---|---|---|---|---|---|")
    for q in BK:
        for h in HZ:
            d = A[str(q)]["D2"][f"h{h}"]
            s, p, w = d["stage1"], d["pooled"], d["within_mean"]
            print(f"| {q} | {h} | {s['M1']:+.3f} | {p['M1']:+.3f} | {w['M1']:+.3f} | {s['M3']:+.3f} | {p['M3']:+.3f} | {w['M3']:+.3f} | {s['M2']:+.3f} | {p['M2_M1level']:+.3f} | {p['M2_aggregation']:+.3f} | {p['VR_C3_cross_path']:.3f} |")
    print("\n### 表 E2: 候補別の時間集約の診断(C3。経路間 VR_ens、窓内 VR_win(60)、ACF のラグ 1〜60 の和、尖度、volclust_50、経路ごとの平均収益のばらつきの比)\n")
    print("`平均のばらつきの比` は、経路ごとの最初の 1,000 本の平均収益の標準偏差を、独立増分の場合の期待値 `√V/√1000` で割った値(1 より大きければ、経路間の低周波の変動が大きい)。\n")
    print("| バケット | 候補 (delta, wma) | VR_ens h1 / h4 | VR_win(60) | Σ ACF(1〜60) | 尖度 | volclust_50 | 平均のばらつきの比 |\n|---|---|---|---|---|---|---|---|")
    for q in BK:
        for c in A[str(q)]["candidates"]:
            e = c["C3"]
            print(f"| {q} | ({c['delta']}, {c['wma']}) | {e['VR_ens']['1']:.2f} / {e['VR_ens']['4']:.2f} | {e['vr_win_mean']['60']:.2f} | {sum(e['acf_mean']):+.2f} | "
                  f"{e['atlas_stats_mean']['kurtosis']:.1f} | {e['atlas_stats_mean']['volclust_50']:+.3f} | {e['mean1000_sd_over_expected']:.2f} |")


if __name__ == "__main__":
    main()
