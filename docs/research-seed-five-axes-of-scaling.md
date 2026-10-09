# 研究シードデータ — "Five Axes of Scaling" investigation

> **これは何か:** [research-agent-service.md §6](research-agent-service.md#6-データモデルresearch-agent-service側独自db)
> のデータモデルに沿って、既存レポート
> [Five Axes of Scaling](https://claude.ai/code/artifact/55721b06-586f-4b3b-8f72-9078682888c4)
> （33セクション・900+訓練実行・5データセット、§33で凍結）から
> **(1) `IMPORTED`な履歴として登録する内容**と**(2) `seed_brief_json`付きDRAFTとして
> 投入する未着手の研究課題5件**を構造化して抽出したもの。この2つはライフサイクル
> （[§3](research-agent-service.md#3-ライフサイクル)、特に§3.1/§3.2）上まったく別の経路を通る
> ——(1)はBRIEFING/PENDING_APPROVAL/RUNNINGを一度も経由しない独立した第三の状態、
> (2)はDRAFTのまま止まっており、本物のBRIEFING（実DB・実LLMでのPrior-Art Search再実行）を
> まだ経由していない、という点に注意。サービス実装後、このファイルの内容をそのまま
> `research_questions` / `research_briefs` / `reports` の初期データとして投入する想定。
>
> 副次的な効果として、これは[§5.3(A) Prior-Art Search](research-agent-service.md#a-関連研究の事前検索prior-art-search)
> が実際に検索対象とするデータの形の具体例にもなっている — 以下の5件の新規研究課題は、
> BRIEFINGが実行されたときにこの`IMPORTED`な研究(`seed-q0`)を`follow_up_candidate`として
> 検索でヒットすることを狙って`related_questions`を`seed_brief_json`内に先埋めしてある
> （ただしこれはBRIEFING実行時に再検証されるべきヒントであり、確定情報ではない — §3.1）。

**出典:**
- レポート本体: https://claude.ai/code/artifact/55721b06-586f-4b3b-8f72-9078682888c4
- 付随の全実験ログ: https://claude.ai/code/artifact/f118f74a-934d-4598-ae26-c502d2e21c95
- 凍結時点: §33（"investigation phase closed" — ユーザーが能動的にexperimentationを止めた時点）

---

## 1. IMPORTEDな履歴として登録する内容（§3.2）

このinvestigationはこのサービスのAgent Loopを一度も経由していないため、
`status: "imported"` として登録する — `completed`ではない（`completed`は
`research_sessions`側の、Agent Loopが実際に走った記録に対してのみ使う。§6参照）。

### 1.1 `research_questions` / `research_briefs` / `research_sessions`

```jsonc
// research_questions
{
  "id": "seed-q0",
  "raw_text": "DDM（エージェントベース市場シミュレータ）でTransformerがボラティリティを高精度に予測できる理由を明らかにしたい。個々のトークン、およびトークン対はターゲットとほぼ無相関(mutual informationがほぼゼロ)なのに、なぜモデルは高精度(R²≈0.98)に予測できるのか。深さ・幅・データ量それぞれへのスケーリング応答は、データセットの性質によってどう変わるのか。",
  "seed_brief_json": null,   // IMPORTED経路はDRAFTを経由しないのでseed_brief_jsonは使わない
  "status": "imported",
  "created_at": "pre-service"  // このサービス導入前に実施された investigation のため開始日時は本レポートに記載なし
}

// research_sessions — Agent Loopを実行していない履歴の器（§3.2, §6）
{
  "id": "seed-s0",
  "brief_id": "seed-b0",
  "status": "imported",
  "started_at": "pre-service",
  "ended_at": "pre-service",
  "budget_used_json": null   // Agent Loopが消費した予算という概念が存在しない
}
// 対応するagent_stepsは0件（ターン単位のツール呼び出し履歴が存在しない）

// research_briefs — §33時点のsnapshotとして1件のみ登録
// (実際には§01〜§33の間に仮説は繰り返し改訂されているが、
//  サービス導入前の investigation なので単一の最終snapshotとして扱う)
{
  "id": "seed-b0",
  "question_id": "seed-q0",
  "approved_at": null,   // 人間の承認イベントは実際には発生していない（IMPORTED経路のため）
  "approved_by": null,
  "brief_json": {
    "title": "Five Axes of Scaling",
    "hypothesis": "データセットが深さ・幅・データ量にどう応答するかは、単一の性質(トークン複雑度・生の予測可能性・生成過程の"種類")では決まらない。5つの独立した性質が共同で決める。",
    "scope": {
      "datasets": ["DDM", "USDJPY", "delay_mackey_glass", "xor_temporal", "lfsr_8bit"],
      "symbols": ["USDJPY"],
      "architectures": ["transformer", "deep_sets", "histogram_baseline"],
      "allow_new_data_collection": false
    },
    "success_criteria": [
      { "metric": "n/a", "note": "探索的な統合研究のため事前の機械的成功基準は設定されていない。§33で5軸フレームワークとして統合されたことをもって収束とみなす" }
    ],
    "stopping_policy": { "note": "ユーザーが§33到達時点で能動的に凍結（budget切れではない）" },
    "llm_config": { "provider": "unknown", "model": "unknown", "note": "本サービス導入前に実施されたためAgentRunner経由ではない" }
  },
  "version": 33
}
```

### 1.2 `reports`

```jsonc
{
  "id": "seed-r0",
  "session_id": "seed-s0",  // §1.1のimportedセッション（seed-s0）に紐づく
  "format": "markdown",
  "content_path": "https://claude.ai/code/artifact/55721b06-586f-4b3b-8f72-9078682888c4",
  "summary": "個々のトークン・トークン対がターゲットとほぼ無相関でも、Transformerは高精度に予測できる。この現象は5つのデータセット（DDM/USDJPY/delay_mackey_glass/xor_temporal/lfsr_8bit）に共通して現れるが、深さ・幅・データ量それぞれへの応答の仕方はデータセットごとに大きく異なり、それを説明するのは単一の指標ではなく5つの独立した軸（トークン複雑度／order依存残差／階層構造の形状／容量の余地／使用スケールでの定常性）である。6つ目の候補軸（アーキテクチャの帰納バイアスとの相性＝learnability）も示唆されるが未検証。",
  "recommendation": "5軸フレームワークの実データ(USDJPY)での直接検証が最優先(§7 open question #1、ユーザー自身がhighest-valueと明言)。詳細はseed-q1〜q5参照。",
  "created_at": "history-import"
}
```

### 1.3 主要findings一覧（`reports`本文に対応する構造化データ）

レポート§3の各`.claim`ブロックを1行ずつに圧縮したもの。`status`はレポート内のbadge
（Confirmed / Falsified / Partial / Exploratory）をそのまま踏襲。

| # | § | Theme | Finding | Status | 要点 |
|---|---|---|---|---|---|
| 1 | §01→§05 | A | More training data degrades DDM prediction quality | Falsified | 16.8×の劣化は正規化バグ＋50,000行の暗黙capが原因。データ量単独では影響ほぼ無し(1.006×差) |
| 2 | §06 | A | 残差ギャップはcalendar-window難易度 | Falsified | 4条件ともR²≈0（"平均を予測"を超えていない）。ギャップは分散追跡の話であってスキルの話ではなかった |
| 3 | §07 | A | DDMには実構造がある — returnではなくvolatilityに | Confirmed·3 seeds | return→volatility予測でR²=0.905–0.977。以降全データセットの目的変数の標準に |
| 4 | §08 | B | トークン化方式はvolatility信号を損なわない | Confirmed·3 seeds | raw/diff/quantize_diff/SAX/clusterでR²=0.856–0.977 |
| 5 | §10,§11 | B | Transformerの優位性はorderとattentionそのもの | Confirmed·3 seeds | 順列不変Deep Sets(3,164 params)はR²=0.053(ヒストグラム同等)。40倍小さいTransformerでもR²=0.987 |
| 6 | §12 | B | 実市場データ(USDJPY)でも定性的に再現 | Confirmed | USDJPY: histogram R²=0.762 vs Transformer R²=0.991。ただしDDMのhistogramは信号の5.8%しか説明せずUSDJPYは76.2% |
| 7 | §13 | B | 分布統計・全ペア相互作用はギャップを説明しない | Falsified | skew/kurtosis/Gini、3,600全順序対のDeep Setsともにhistogram baseline圏内 |
| 8 | §14 | B | 信号はdistanceに集中、隣接ではない | Partial（交絡あり） | lag 1→32でR²上昇後lag=59で崩壊。プールペア数の交絡がTheme Cで解消される |
| 9 | §15,§16 | C | lag=40の"壁"はobs_len起因、実際の天井ではない | Partial→§17で解決 | obs_lenを200に拡張すると壁は消失 |
| 10 | §17 | C | 生lagはほぼ無関係、n_validとcoverage比は独立に効く | Confirmed·3seeds(ExpA) / Exploratory(ExpB) | coverage比25%→100%でR²が6.5–7.4倍。n_valid倍増で別途ほぼ半減 |
| 11 | §18 | D | coverage比はfull modelでも効くが、連続サンプリングが勝る | Partial（自説と矛盾しつつ確認） | 低coverageで連続サンプリングが分散サンプリングを2.5×上回る(DDM) |
| 12 | §20,§21 | D | 受容野は深さで拡大 — 階層的、層フラットではない | Confirmed·3seeds, RF一致 | 同一RF=33で深さが幅に勝る: DDM +44.4%, USDJPY +2.4% |
| 13 | §22,§25,§26 | D | 深さは階層を作る — が完全な組立ラインではない | Partial | 局所統計は層1で早期出現・全体統計は層5–7で段階的出現。一部情報(トークン識別等)はむしろ層を追うごとに減衰 |
| 14 | §23,§24 | D | attention感度の"resonance"パターンは因果的に負荷を持たない | 相関は確認・因果は否定 | 勾配感度で見えたresonanceパターンをホップ単位でablationしても性能ジャンプなし |
| 15 | §27 | E | 階層構造はDDM固有ではない — が形状は大きく異なる | Partial（5中3で確認、2で弱い） | 深さ利得: lfsr_8bit+24.7%, DDM+44.4%, xor_temporal+5.6%, USDJPY+2.4%, delay_mackey_glass+1.5% |
| 16 | §28 | E | 表面統計だけでは全く異なる挙動のデータセットを区別できない | Confirmed | xor_temporalとlfsr_8bitはtoken MI等が統計的に同一なのに最終R²は0.763 vs 0.984 |
| 17 | §29 | F | （方法論の教訓の再発見）DDMは大規模データでは非定常 | 直接確認 | R²が0.910(5K行)→0.097(1.28M行)に崩壊。原因は価格水準の2.6×ドリフト(undertrainingではない) |
| 18 | §31 | F | 幅スケーリングはボトルネックの種類でデータセットを分ける | Confirmed·3seeds, 8点フィット | DDM(α=0.220)とlfsr_8bit(α=0.699)はd=128でも容量制限、delay_mackey_glassは飽和(L∞=0.00885) |
| 19 | §30 | G | regime不均一性が§29崩壊の主因 — 部分的に | Partial | "early" chunkは単調改善(0.149→0.415)、"late" chunkは非単調でノイジー(0.099→−0.174→0.115) |
| 20 | §32 | G | 局所予測可能性は乱高下と相関、水準とは相関しない | Exploratory n=6 | volatility-of-volatility(corr=−0.574)とkurtosis(corr=−0.679)が最も強い相関。§30の非単調カーブ形状は完全には再現できず |

### 1.4 反証済み仮説（今後の研究で繰り返さないための"知っておくべき落とし穴"）

レポート§4より。将来のBRIEFING時、この投資調査に関連する新規研究課題のシステムプロンプトに
含める価値がある（同じ罠を再発見させないため）。

```jsonc
"known_pitfalls": [
  {
    "was": "訓練データが多いほどDDM予測が悪化する（§01, 16.8×）",
    "actually": "非定常な生価格正規化 と 暗黙の50,000行capの2つの交絡が原因。データ量単独ではほぼ無関係（§05）"
  },
  {
    "was": "(1)(2)修正後の残差ギャップはcalendar-regimeの真の難易度差（§05）",
    "actually": "対象タスク(return→return)自体に実信号がほぼゼロ。ギャップは分散追跡の話（§06）"
  },
  {
    "was": "リッチな分布統計や全ペア相互作用がbaselineとTransformerのギャップを埋める（§13の当初仮説）",
    "actually": "どちらも効かない。attentionが計算しているものは固定の1次・2次対称関数では表現できない"
  },
  {
    "was": "attentionの優位性はcoverage（窓の多くを見ること）に由来し、分散サンプリングも連続サンプリングと同程度効くはず（§18の当初仮説）",
    "actually": "低coverageでは連続サンプリングが分散サンプリングを2.5×上回る — 逆方向"
  },
  {
    "was": "pairwise-distance sweepのlag=40の"壁"は信号到達距離の真の天井（§15）",
    "actually": "obs_lenのartifact。窓長を伸ばすと壁は消える（§16、ただし厳密なdeconfoundには§17が必要だった）"
  },
  {
    "was": "attention感度の"resonance"パターン（migrating peak）はモデルが依拠する負荷のかかったメカニズム（§23の仮説候補）",
    "actually": "因果ablation(§24)では予測されたcrossoverで性能ジャンプ無し。パターンは実在するが機能的に必須とは確認されなかった"
  },
  {
    "was": "DDMは他3つの合成データセットと同様に定常であり、固定ベンチマークとして扱ってよい（§18–§28の暗黙の前提）",
    "actually": "十分な履歴量を使うと価格水準・volatility regimeとも大きくドリフトする（§29）。他3つの合成データセットは真に定常"
  }
]
```

> **既視感の注記:** この6件目の教訓（DDMの非定常性が`max_rows`の暗黙capと同じ形で2度顕在化した、
> というレポート冒頭のcallout）は、このワークスペースの過去のメモリー
> （*DDM Simulator — 小規模agent数でのWMA乖離*、*Always pass max_rows explicitly*）と
> 直接一致する。既に個別の教訓としては記録済みだったものが、この investigation の中で
> 独立に2度再発見されている — Prior-Art Search（§5.3-A）が実際に防ぎたい典型例そのもの。

---

## 2. 未着手の研究課題（§7 Open Questionsからの抽出、5件）

レポート§7で列挙された5項目を、`research_questions.status = "draft"` として、
[§4 Research Brief スキーマ](research-agent-service.md#4-research-brief構造化案スキーマ)の形に
沿った`seed_brief_json`（[§3.1](research-agent-service.md#31-draftへの種の持ち込みseed_brief_json)
参照 — BRIEFINGへのヒントであり、承認済みBriefではない）を添えて構造化。`budget`は人間が
承認前に確定させる前提の**仮の初期値**。すべて`related_questions`で`seed-q0`（本investigation）
への`follow_up_candidate`を明示しているが、これも本物のBRIEFINGが実DB・実LLM判定で
再検証するまでは未確定の候補にすぎない。

### 2.1 q1 — 五軸フレームワークは実データ(USDJPY)でも成立するか【最優先】

> レポート原文: "The user has flagged this as the highest-value external validation" — ユーザー自身が最優先と明言している唯一の項目。

```jsonc
{
  "id": "seed-q1",
  "raw_text": "五軸フレームワーク(トークン複雑度／order依存残差／階層構造／容量の余地／定常性)は、USDJPYの実データでも同じように深さ・幅・データ量スケーリングを予測できるか？ 幅スケーリング(§31)とデータ量スケーリング(§29)はDDM/合成データセットにしか直接適用されておらず、USDJPY自体には未実施。",
  "status": "draft",
  "seed_brief_json": {
    "title": "五軸フレームワークのUSDJPY実データ検証",
    "hypothesis": "§31(幅スケーリング)・§29(データ量スケーリング)と同じ測定方法をUSDJPYに適用したとき、五軸フレームワークが予測する挙動（DDMに近い部分定常性の影響、depth>width、hierarchy形成等）が実際に観測される",
    "scope": {
      "symbols": ["USDJPY"],
      "timeframes": ["M1"],
      "datasets": ["USDJPY"],
      "allow_new_data_collection": false
    },
    "success_criteria": [
      { "metric": "width_scaling_exponent_fitted", "op": "==", "value": true, "note": "§31と同じ8点d_model(8→128)フィットが3seedsでUSDJPYに対して完了していること" },
      { "metric": "data_volume_scaling_curve_fitted", "op": "==", "value": true, "note": "§29と同じmatched-steps手法でUSDJPYのデータ量カーブが得られていること" }
    ],
    "budget": { "max_training_runs": 30, "max_wall_clock_minutes": 480, "max_hparam_search_jobs": 2, "deadline": "要人間確定" },
    "related_questions": [
      { "question_id": "seed-q0", "relation_type": "follow_up_candidate",
        "similarity_note": "五軸フレームワーク(seed-q0の結論)の未検証の外的妥当性チェック。レポート著者自身が次の最優先事項として明記" }
    ],
    "priority_note": "レポート内でユーザーが明示的にhighest-valueと指定した唯一の項目",
    "review_notes": "AIレビュー(F1/F2, research-review-five-axes-of-scaling.md)より: §12のUSDJPY基礎数値のseed数が未確認。本課題を実行する前に、まずF1(§12の3数値を3+ seedsで再計測)を前提条件として満たすことを推奨"
  }
}
```

### 2.2 q2 — xor_temporalの平坦な幅スケーリングは本当にアーキテクチャの帰納バイアス起因か

```jsonc
{
  "id": "seed-q2",
  "raw_text": "xor_temporalの幅スケーリングカーブが統計的にはfitが良い(R²=0.996)のに実質的にほぼ動かない(214倍のパラメータ増で14%しか変化しない)のは、パリティ的・組合せ的な依存関係とこのアーキテクチャの相性が悪いからか？ より相性の良いアーキテクチャや、XOR的依存関係を扱える既知のモデルクラスとの比較で直接検証したい。",
  "status": "draft",
  "seed_brief_json": {
    "title": "xor_temporalにおけるlearnability軸(第6軸候補)の直接検証",
    "hypothesis": "xor_temporalの幅スケーリング無反応性は容量・データ・深さの制約ではなく、現行Transformerアーキテクチャがパリティ様の組合せ依存関係を学習しにくい構造的帰納バイアスに起因する",
    "scope": {
      "datasets": ["xor_temporal"],
      "architectures": ["transformer", "candidate_parity_friendly_architecture"],
      "allow_new_data_collection": false
    },
    "success_criteria": [
      { "metric": "candidate_architecture_r2", "op": ">", "value": "transformer_r2_at_matched_params", "note": "パリティ相性の良い候補アーキテクチャが同等パラメータ数のTransformerを明確に上回れば帰納バイアス仮説を支持" }
    ],
    "budget": { "max_training_runs": 20, "max_wall_clock_minutes": 300, "max_new_datasources": 0, "deadline": "要人間確定" },
    "related_questions": [
      { "question_id": "seed-q0", "relation_type": "follow_up_candidate",
        "similarity_note": "seed-q0で識別されたが未検証の"第6軸(learnability)"候補を直接テストする" }
    ],
    "note": "AlgoForgeのCLAUDE.md「新しいML architectureの追加」手順が前提になる — 新アーキテクチャの追加が必要"
  }
}
```

### 2.3 q3 — DDMとlfsr_8bitの真の容量天井はどこか

```jsonc
{
  "id": "seed-q3",
  "raw_text": "§31の幅スケーリングフィットはDDMとlfsr_8bitのどちらもd_model=128時点で有限のfloor(L∞)を解決できなかった。d≥256までスイープを広げれば、実用的な範囲内で飽和が観測できるか？",
  "status": "draft",
  "seed_brief_json": {
    "title": "DDM/lfsr_8bit 幅スケーリング拡張スイープ (d_model≥256)",
    "hypothesis": "d_model=128を超えて十分に広いスイープを行えば、DDMとlfsr_8bitのいずれも有限のL∞に収束する（無限に容量制限が続くわけではない）",
    "scope": { "datasets": ["DDM", "lfsr_8bit"], "allow_new_data_collection": false },
    "success_criteria": [
      { "metric": "l_inf_resolved", "op": "==", "value": true, "note": "§31と同じ手法(3 seeds、曲線フィット)でL∞が有限値として収束すること。収束しない場合もそれ自体が結果として報告価値あり" }
    ],
    "budget": { "max_training_runs": 24, "max_wall_clock_minutes": 600, "note": "d_model拡大は1runあたりのコストが増えるため他のqより時間予算を厚めに", "deadline": "要人間確定" },
    "related_questions": [
      { "question_id": "seed-q0", "relation_type": "follow_up_candidate",
        "similarity_note": "§31(seed-q0内)の未解決の境界条件を直接引き継ぐ" }
    ]
  }
}
```

### 2.4 q4 — DDMのlate-regime不安定性を完全に説明する

```jsonc
{
  "id": "seed-q4",
  "raw_text": "§32のより細かい分解(6セグメント)ではvolatility-of-volatilityとreturn kurtosisという実在する相関を見つけたが、§30の正確な非単調カーブ形状は再現できなかった。2つの評価設計の違いを直接和解させる必要がある。",
  "status": "draft",
  "seed_brief_json": {
    "title": "DDM late-regime不安定性の§30/§32評価設計の統合",
    "hypothesis": "§30(300K行chunk・volume sweep)と§32(6×50K行segment・standalone訓練)の評価設計の違い自体が非単調カーブ形状のズレの原因であり、両者を揃えた第3の設計で再測定すればvolatility-of-volatility/kurtosis相関から§30のカーブ形状を再構成できる",
    "scope": { "datasets": ["DDM"], "note": "'late' regime chunkに限定", "allow_new_data_collection": false },
    "success_criteria": [
      { "metric": "reconciled_curve_matches_s30", "op": "==", "value": "qualitative", "note": "定量一致までは求めず、非単調性の方向・大まかな形状が再現できれば成功" }
    ],
    "budget": { "max_training_runs": 18, "max_wall_clock_minutes": 240, "deadline": "要人間確定" },
    "related_questions": [
      { "question_id": "seed-q0", "relation_type": "follow_up_candidate",
        "similarity_note": "seed-q0内で「完全な矛盾ではなく明示的な未解決ギャップ」として報告された箇所の直接の続き" }
    ],
    "note": "5件の中で最も仮説の輪郭が曖昧 — BRIEFING時に人間が success_criteria をより厳密に絞り込む必要がある可能性が高い",
    "review_notes": "AIレビュー(F3, research-review-five-axes-of-scaling.md)より: §30の負R²runsが訓練の収束失敗である可能性を先に切り分けるべき。もし収束失敗なら、この課題自体の問題設定（2つの評価設計を"和解させる"）が変わりうる — BRIEFING時にsuccess_criteriaへ収束診断の確認を追加することを推奨"
  }
}
```

### 2.5 q5 — 深さ・文脈長スケーリングの独立軸としての測定

```jsonc
{
  "id": "seed-q5",
  "raw_text": "§29の段階的ロールアウトから先送りされていた深さスケーリングと文脈長スケーリングが、§31で完了した幅スケーリングとは独立の軸として未検証のまま残っている。",
  "status": "draft",
  "seed_brief_json": {
    "title": "深さ・文脈長スケーリング則の測定（§31幅スケーリングの姉妹実験）",
    "hypothesis": "§31が幅(d_model)についてL(N)=L∞+a·N⁻ᵅの形でフィットしたのと同じ手法を深さ(層数)・文脈長(obs_len)に適用すると、データセットごとに異なるボトルネックタイプ（幅制限／深さ制限／文脈長制限）が識別できる",
    "scope": {
      "datasets": ["DDM", "USDJPY", "delay_mackey_glass", "xor_temporal", "lfsr_8bit"],
      "note": "理想的には5データセット全てだが、budget次第でDDM+USDJPYに絞ってもよい",
      "allow_new_data_collection": false
    },
    "success_criteria": [
      { "metric": "depth_scaling_exponent_fitted", "op": "==", "value": true },
      { "metric": "context_length_scaling_exponent_fitted", "op": "==", "value": true }
    ],
    "budget": { "max_training_runs": 40, "max_wall_clock_minutes": 720, "note": "5データセット×2軸のため5件中最大のbudgetになりやすい", "deadline": "要人間確定" },
    "related_questions": [
      { "question_id": "seed-q0", "relation_type": "follow_up_candidate",
        "similarity_note": "seed-q0 §31で完了した幅スケーリングの姉妹実験、同じ手法論の横展開" },
      { "question_id": "seed-q3", "relation_type": "shares_resources",
        "similarity_note": "同じDDM/lfsr_8bitモデル群・同じ曲線フィット手法を使う可能性が高く、実装順序次第では合流できる" }
    ]
  }
}
```

### 2.6 q6 — 五軸フレームワークを追加データセットで検証できるか（AIレビューF4より昇格）

```jsonc
{
  "id": "seed-q6",
  "raw_text": "5軸フレームワークは5データセットに対して5〜6個の説明軸を当てはめており、自由度がほぼデータ点数と同じ。追加の、独立に選定されたデータセットで検証しない限り、軸の選び方自体が正しいという主張はほぼ無内容。",
  "status": "draft",
  "seed_brief_json": {
    "title": "五軸フレームワークの追加データセットでの汎化検証",
    "hypothesis": "seed-q0で選定された5データセットとは独立に選ばれた新しいデータセット（複数）に対しても、5軸フレームワークの各軸の測定値が深さ・幅・データ量への応答を事前予測できる",
    "scope": {
      "datasets": ["要選定 — seed-q0の5データセットとは独立な基準で選ぶこと自体がこの課題の一部"],
      "note": "データセット選定バイアスを避けるため、選定プロセス自体をBRIEFING/PENDING_APPROVALで明示的にレビューする必要がある",
      "allow_new_data_collection": true
    },
    "success_criteria": [
      { "metric": "axes_measured_before_scaling_sweep", "op": "==", "value": true, "note": "5軸の値をスケーリング実験の前に測定し、事後的に辻褄合わせしないこと（事前予測として扱う）" },
      { "metric": "prediction_direction_matches", "op": ">=", "value": "3/5 datasets", "note": "少なくとも新データセットの過半数で軸の予測方向が当たること" }
    ],
    "budget": { "max_training_runs": 50, "max_wall_clock_minutes": 900, "max_new_datasources": 3, "note": "新規データセット選定・収集を含むため5件中最大級のbudget", "deadline": "要人間確定" },
    "related_questions": [
      { "question_id": "seed-q0", "relation_type": "raised_by_review",
        "similarity_note": "AIレビュー F4（research-review-five-axes-of-scaling.md）より昇格。seed-q0の§33結論は自由度不足で統計的に無内容に近いという指摘への直接対応" }
    ],
    "note": "5件の中で唯一データセット選定という新しい種類の意思決定を含む。BRIEFING時にPrior-Art Search（§5.3A）が働くよう、選定候補は必ずAlgoForgeの既存list_datasets結果も含めて検討すること"
  }
}
```

### 2.7 q7 — 主要claimをwalk-forward評価で再検証できるか（AIレビューF5より昇格）

```jsonc
{
  "id": "seed-q7",
  "raw_text": "ほぼ全てのclaimはregime-controlled splitting（時系列順ではない層別80/20分割）で計測されている。唯一chronological splitが試されたDDM(§30/§32)だけが最も不安定な結果を出している。他の主要claimも真の意味でのwalk-forward評価では未検証。",
  "status": "draft",
  "seed_brief_json": {
    "title": "主要claimのwalk-forward（時系列順）再評価",
    "hypothesis": "Theme B(attentionの優位性)とTheme D(depth>width)の中心的なclaimは、regime-controlled splittingではなく真のwalk-forward split（学習期間より後の期間のみを検証に使う）でも同程度の効果量で再現する",
    "scope": {
      "datasets": ["DDM", "USDJPY", "delay_mackey_glass", "xor_temporal", "lfsr_8bit"],
      "note": "予算が限られる場合はDDM+USDJPYに絞ってもよいが、非定常性が最も強いDDMは必ず含めること（§29で確認済みの非定常性が影響を受けやすいはず）",
      "allow_new_data_collection": false
    },
    "success_criteria": [
      { "metric": "attention_advantage_walk_forward", "op": ">=", "value": "0.5 * regime_controlled_effect_size", "note": "§10-11のattention優位性(regime-controlled)の効果量の半分以上がwalk-forwardでも残ること" },
      { "metric": "depth_gt_width_walk_forward", "op": "==", "value": true, "note": "§20-21のdepth>widthの方向がwalk-forwardでも逆転しないこと" }
    ],
    "budget": { "max_training_runs": 36, "max_wall_clock_minutes": 600, "deadline": "要人間確定" },
    "related_questions": [
      { "question_id": "seed-q0", "relation_type": "raised_by_review",
        "similarity_note": "AIレビュー F5（research-review-five-axes-of-scaling.md）より昇格。§2の方法論的選択（regime-controlled split）が開示されているのに、そこから生じる"見出し数値は時系列的に未検証"という含意はseed-q0の§7 open questionsに含まれていなかった" },
      { "question_id": "seed-q4", "relation_type": "shares_resources",
        "similarity_note": "DDMのchronological split自体は§30/§32で既に一部使われている手法・データを再利用できる" }
    ]
  }
}
```

---

## 3. 優先順位についてのメモ

- **q1が最優先** — レポート著者（ユーザー本人）が明示的に"highest-value"と述べている唯一の項目。
  BRIEFINGを実装したら最初にこれを走らせるのが妥当。ただしAIレビュー(F1)により、q1着手前に
  §12のUSDJPY基礎数値のseed数確認を前提条件として挟むことを推奨。
- **q4は最も仮説の輪郭が曖昧** — 「2つの評価設計を和解させる」というのは機械的な成功基準に
  落としにくい。[research-agent-service.md §4](research-agent-service.md#4-research-brief構造化案スキーマ)
  の「`success_criteria`は機械的に判定できる形にする」という原則と摩擦がある実例 — BRIEFING時の
  AI構造化案生成がこの手のあいまいな課題をどこまでうまく機械的基準に変換できるかの、良いテストケースになる。
  AIレビュー(F3)により、この課題自体の前提（負R²=regimeの難しさ）が訓練失敗で説明できてしまう
  可能性も先に切り分ける必要がある。
- **q3とq5は`resource_claims`（§5.3-B）が効いてくる可能性がある** — どちらもDDM/lfsr_8bitの
  幅スケーリング系の実験を含み、実装順序によっては同じmodelを取り合う。q5のBriefに
  `shares_resources`として先に明記した。
- **q6とq7はAIレビュー（[research-review-five-axes-of-scaling.md](research-review-five-axes-of-scaling.md)）
  から昇格した課題** — q1〜q5がレポート自身の§7 open questionsの延長線上にあるのに対し、
  q6・q7はレポートが自己開示していなかった、より基礎的な問い（軸の統計的な無内容さ、
  評価split方式の妥当性）。§33の結論そのものへの依存度がq1〜q5より高く、優先度としては
  q1の次に検討する価値がある。
