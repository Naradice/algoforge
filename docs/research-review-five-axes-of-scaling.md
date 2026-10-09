# AIレビュー例 — "Five Axes of Scaling"

> [research-agent-service.md §7.1](research-agent-service.md#71-既存研究へのaiレビュー問題点疑問点の指摘--重要度付け)
> で設計したレビュー機能の実例。対象は
> [Five Axes of Scaling](https://claude.ai/code/artifact/55721b06-586f-4b3b-8f72-9078682888c4)
> （[research-seed-five-axes-of-scaling.md](research-seed-five-axes-of-scaling.md)の`seed-r0`）。
>
> **前提の制約:** レポート自身が既にConfirmed/Falsified/Partial/Exploratoryのbadgeと
> Evidence Ledger（§5）で自己評価している項目は再指摘していない。以下はすべて、
> レポートが自己開示していない、または開示の粒度が甘い問題・疑問のみ。
>
> **検証状況の注記:** この一連の呼び出しでは対象investigationを実行した当時のAlgoForge
> training_run/validation記録に対する実際のMCPアクセスがない。したがって`verified_via_mcp`
> が`false`のfindingは「要検証の疑い」であって「確定した欠陥」ではない — 実運用時、
> レビュー機能は該当するMCPツール（`get_model_validations`, `list_training_runs`等）で
> これらを実際に突き合わせて確定させる設計（§7.1）。

```jsonc
{
  "id": "seed-review0",
  "target_type": "report",
  "target_id": "seed-r0",
  "reviewer_provider": "anthropic", "reviewer_model": "claude-sonnet-5",
  "status": "completed"
}
```

---

## Findings

### F1 — issue · importance: high

| | |
|---|---|
| target_ref | §12; Evidence Ledger row「Qualitative pattern replicates on real FX」 |
| verified_via_mcp | false（要検証） |
| promoted_question_id | なし — 既存`seed-q1`への前提条件として付記 |

**要旨:** 投資調査全体で唯一の実市場データ検証（USDJPY, §12）を支えるseed数が開示されていない。

**詳細:** Evidence Ledgerの§12行は`Seeds`列が明示的に`n/a`。一方DDM系のほぼ全claimは
3-seed規律が守られている（§07, §08, §10-11, §17, §20-21, §27-28, §31）。§12の見出し数値
（histogram R²=0.762, Deep Sets R²=0.761, Transformer R²=0.991）が複数seedで取られたとは
claim本文にもledgerにも書かれていない。このワークスペース自身の標準規約
（[CLAUDE.md「Comparing model training runs」](../CLAUDE.md#comparing-model-training-runs) —
「Unseeded runs on this pipeline can land 2×+ apart on init/shuffle luck alone」）に照らすと、
"実データでも再現した"という、フレームワーク全体の外的妥当性を支える唯一の主張が、
他のclaimより明らかに弱い形の証拠に基づいている可能性がある。

**推奨アクション:** §12の3数値を3+ seedsで再計測し分散を報告。分散が小さければ確信度を
上げてよいが、大きければ"replicates on real FX"の主張に留保が必要。

---

### F2 — issue · importance: medium

| | |
|---|---|
| target_ref | §28（order-dependent residualによるデータセット横断ランキング） |
| verified_via_mcp | n/a（F1の解決待ち） |
| promoted_question_id | なし — `seed-q1`への前提条件としてF1と合わせて付記 |

**要旨:** データセット横断の順位付けに使われる派生指標が、精度の異なる測定値を同列に比較している。

**詳細:** 「final R² − histogram R²」というorder-dependent residualは、lfsr_8bit +0.671から
USDJPY +0.075まで幅があるが、USDJPYが最小（=最も僅差）であり、かつF1の通りUSDJPYの
基礎数値のseed数が不明。もしF1が実際に1-seedだった場合、この0.075という最小値自体が
seed-to-seedのノイズの範囲内で入れ替わりうる。レポートはこの派生指標の精度がデータセット間で
一様であることを暗黙の前提にしている。

**推奨アクション:** F1解決後、order-dependent residualのランキングを点推定ではなく信頼区間付きで再計算する。

---

### F3 — question · importance: high

| | |
|---|---|
| target_ref | §30（"late" chunk, R² = −0.174, −0.177） |
| verified_via_mcp | false（要検証） |
| promoted_question_id | なし — 既存`seed-q4`への追加検討事項として付記 |

**要旨:** 負のR²が「regimeの本質的な難しさ」なのか「訓練の収束失敗」なのか切り分けられていない。

**詳細:** 負のR²は「平均予測より悪い」ことを意味し、regimeが真に異なる/難しいことの表れにも、
単純な最適化の失敗（初期化・学習率・収束不足）の表れにもなりうる。レポートはこれらの
具体的なrunについて訓練損失カーブや勾配ノルムなどの収束診断を確認したとは述べていない。
これは既存の`seed-q4`（§30/§32の評価設計の統合）に直接関わる — もし負のR²runsが実際には
訓練失敗だった場合、§32が"6セグメントに分割すると不安定性が消える"と報告している現象は、
「2つの評価設計を和解させる」という枠組みで語るべき話ではなく、単に「セグメントが小さく
均質になったことで訓練が安定した」というもっと平凡な説明かもしれない——`seed-q4`が
前提としている問題設定そのものを変えうる。

**推奨アクション:** §30のlate-chunk負R²を出した具体的なcheckpointの訓練カーブを再確認（または
再実行）し、収束の兆候の有無を先に切り分けてから§32の分析に進む。

---

### F4 — issue · importance: high

| | |
|---|---|
| target_ref | §33（capstone: 五軸フレームワーク） |
| verified_via_mcp | n/a（統計的・方法論的な指摘であり事実確認の対象ではない） |
| promoted_question_id | **`seed-q6`（新規昇格）** |

**要旨:** 「5つの軸が独立に示されたわけではないが、共同で説明する」という結論は、5データセットしかない時点で統計的にほぼ無内容な主張になっている。

**詳細:** レポート自身、5軸が統計的に独立であるとは示していないと正直に書いている。しかし
その一歩先——**5データセットに対して5〜6個の説明軸を当てはめる時点で、自由度がほぼ
データ点数と同じであり、この枠組みが「5つのデータセットの挙動を説明できる」という事実
自体が、軸の選び方が正しいことをほとんど支持しない**（もっともらしい別の5軸の組を
事後的に選んでも同程度に"説明"できてしまう可能性が高い）——までは踏み込んでいない。
さらに5データセットは§2の方法論で「意図的に互いに最大限異なるように選ばれた」ものであり、
ランダムサンプルではない。これは文書全体が積み上げてきた§33という到達点そのものに
関わる、この文書で最もレバレッジの大きい弱点。

**推奨アクション:** 五軸フレームワークを「確認された」ではなく「仮説生成された」ものとして
再フレーミングする。追加の、独立に選定されたデータセットでのテストが本質的に必要
（データセットを1つ足すより、この文書の主張のトーン自体を弱めることの方が優先度が高い）。

---

### F5 — question · importance: high

| | |
|---|---|
| target_ref | §2（regime-controlled splitting）— §30/§32を除くほぼ全claimに適用 |
| verified_via_mcp | n/a（テキスト自体からの方法論的指摘） |
| promoted_question_id | **`seed-q7`（新規昇格）** |

**要旨:** ほぼ全ての見出し数値は時系列的な将来予測(out-of-time)ではなく、層別80/20分割で計測されている。この含意が§7の未解決事項として扱われていない。

**詳細:** §2は「regime-controlled splitting」（target値のdecile内でpoolして80/20に再分割、
時系列順ではない）を、§30/§32以外の「forward-in-timeが本質的に重要な場合を除き」標準手法
として明記している。これは方法論として開示されているが、**そこから生じる問い**——
文書全体の見出し数値（abstractのR²≈0.98を含む）が、真の意味でのwalk-forward評価では
未検証のまま残っている、ということ——は§7のどこにも挙げられていない。しかも唯一
chronological splitが試されたデータセット（DDM, §30/§32）こそ、文書内で最も未解決・
不安定な結果が出ている場所であり、これは仮説的な懸念ではない。

**推奨アクション:** 最もレバレッジの大きいclaim（Theme Bのattention vs baseline gap、
Theme Dのdepth vs width）だけでも、DDMだけでなく5データセット全てでwalk-forward再評価を行う。

---

### F6 — question · importance: low

| | |
|---|---|
| target_ref | masthead meta-row「900+ training runs」 |
| verified_via_mcp | n/a |
| promoted_question_id | なし（対応不要 — 情報提供のみ） |

**要旨:** 900以上の訓練実行に対して、集計されたcompute/コストの記録がどこにもない。

**詳細:** 研究の科学的な欠陥ではないが、[research-agent-service.md §11](research-agent-service.md#11-オープンな論点要検討)
のコスト見積もり論点（AlgoForge側の計算コストのドル換算が未実装）にとって参考になる —
将来q1/q3/q5/q6/q7のいずれかをこのサービス経由で再実行する際のbudget初期値の目安として使える。

---

## 昇格した新規研究課題

F4・F5はそれぞれ既存のq1〜q5では十分にカバーされない、独立した課題として
`research-seed-five-axes-of-scaling.md`に**q6・q7として追記**した。F1〜F3は新規課題を
起こすのではなく、既存の`seed-q1`・`seed-q4`に前提条件/追加検討事項として付記している
（詳細はそちらを参照）。
