# Research Agent Service — 設計ドキュメント（Draft / 提案段階）

> **ステータス:** 設計中。まだコードは存在しない。AlgoForge本体の一部ではなく、AlgoForgeのMCPサーバーを
> 外部クライアントとして利用する**独立サービス**として計画している。
> 実装が始まったら、このドキュメントは新しいプロジェクトディレクトリ（例: `finance/research_agent/`）配下の
> README/CLAUDE.mdに引き継ぐことを想定。それまでの間、設計の集積地としてここに置く。

---

## 1. モチベーション

AlgoForgeのMCPサーバー（`docs/mcp-guide.md`）は、データセット・モデル・戦略のCRUDと実行を
AIエージェントに公開している。現状これは「人間がClaude Codeなどの対話セッションでMCPツールを呼ぶ」
使い方を主に想定したものだが、これをもう一段自動化し、

> 研究課題（例:「USDJPY H1でLSTMとTransformerの予測精度を比較し、より良いモデルを見つける」）
> を投げると、AIが自律的にデータ調査 → モデル訓練 → 評価 → （必要なら追加訓練を繰り返す）→
> 最終的にレポートにまとめる、まで一気通貫でやってくれるWebサービス

を作りたい、というのが今回の要望。ロードマップ上は複数の項目（後述 §8）を束ねる形になる。

> **AlgoForge側の不足機能について:** この設計を詰める過程でAlgoForge本体に見つかった具体的な
> ギャップ（コード上で未実装/未配線と確認できたもの）は、都度[`requirements.md`](requirements.md)
> に追記している。特に**R-1（Webhookは登録されるが一度も発火しない）**と**R-3（`/mcp`に認証が無い）**
> はP0扱い — §5の「Webhook駆動が基本」という前提と、§3の「予算内フル自律」という前提それぞれの
> 土台になっている部分なので、実装着手前に確認すること。

---

## 2. AlgoForgeとの関係

```
┌─────────────────────────────┐        ┌───────────────────────────────┐
│  Research Agent Service      │  MCP   │  AlgoForge                     │
│  (新規・独立リポジトリ)        │ ─────▶ │  backend/mcp_server/           │
│                               │  (SSE) │  (既存・変更不要)               │
│  - 研究課題の受付/構造化       │        │                               │
│  - 自律実行ループ（Agent Loop）│ ◀───── │  webhooks (run.completed 等)   │
│  - 予算/ガードレール管理       │  push  │                               │
│  - レポート生成               │        │                               │
│  - 自分専用のDB/UI            │        └───────────────────────────────┘
└──────────────────────────────┘
```

- AlgoForge側の変更は**基本的に不要**（既存のMCPツール・Webhookをそのまま使う）。
  Three-layer separationの原則（layerはHTTP/MCP経由でのみ連携）とも自然に整合する —
  この新サービスは「もう一つの外部MCPクライアント」に過ぎない。
- 自分専用のDB（研究セッション・エージェントの思考ログ・レポート）を持つ。AlgoForgeのDBには触れない。
- 自分専用のUI（研究課題の投稿・進捗の閲覧・レポート閲覧）を持つ。AlgoForgeのUIには手を入れない。
- 唯一のAlgoForge側ギャップ: `create_preprocessed_dataset` がMCPツールとしてまだ無い（`mcp-guide.md`に
  既知のギャップとして明記あり）。前処理レシピを自律的に作りたい場合は、MCPではなくAlgoForgeのREST
  `POST /preprocessed-datasets` を直接叩く（同じ問題を将来AlgoForge側にMCPツール追加として還元してもよい）。

---

## 3. ライフサイクル

```
 [DRAFT]
    │  ユーザーが自由記述で研究課題を投稿
    │  （§3.1: 過去の研究を種として持つ場合は seed_brief_json を伴って始まる）
    ▼
 [BRIEFING]                       ← AIがAlgoForgeの現況を読み取り(read-only)、
    │                                Prior-Art Search(§5.3A)含め Research Brief（構造化案）を生成
    ▼
 [PENDING_APPROVAL]               ← 人間がBriefを確認・編集
    │  承認
    ▼
 [RUNNING]  ⇄ [PAUSED]            ← 予算内は完全自律でAgent Loopを実行
    │
    ├─ 成功基準を満たした/収束した ─▶ [REPORTING] ─▶ [COMPLETED]
    ├─ 予算を使い切った           ─▶ [REPORTING] ─▶ [BUDGET_EXCEEDED]（レポートは出す）
    └─ 回復不能なエラー           ─▶ [FAILED]

 [IMPORTED]  ← §3.2: このライフサイクルを一度も通らずに履歴として直接登録される、
                独立した第三の経路（DRAFTからは分岐しない — 最初からIMPORTED）
```

- **PENDING_APPROVALは必ず経由する**（今回の要件: 自由記述→AI構造化案→人間確認）。
- 承認後の`RUNNING`中は、予算内であれば`start_training_run`のようなコストの発生する
  ツール呼び出しの**都度承認は挟まない**（今回の要件: 予算内は完全自律）。
- `PAUSED`はユーザーがいつでも人間側から止められる状態（自律の暴走に対するキルスイッチ）。

### 3.1 DRAFTへの「種」の持ち込み（seed_brief_json）

[research-seed-five-axes-of-scaling.md](research-seed-five-axes-of-scaling.md)のq1〜q5のように、
既存の研究から導かれた課題を投入する場合、人間（またはこのサービス外のAI）があらかじめ
Brief案の下書きを用意していることがある。これは`research_briefs.brief_json`そのものでは
**ない** — `research_questions.seed_brief_json`という別フィールドに「BRIEFINGへの強いヒント」
として保持し、BRIEFINGフェーズはこれを鵜呑みにせず必ず自分で以下をやり直す:

- Prior-Art Search（§5.3A）を実DB・実LLM判定で再実行する（種に埋め込まれた`related_questions`は
  あくまで参考情報 — 陳腐化している可能性がある）
- `budget`の妥当性を再検討する（種の予算値は多くの場合プレースホルダ）
- `success_criteria`が本当に機械的に判定可能かを再検証する（§11の`q4`のように、種の時点では
  曖昧なまま残っている場合がある）

つまり`seed_brief_json`があっても**BRIEFING〜PENDING_APPROVALは省略されない**。「下書き済みの
DRAFT」であって「承認待ちのBrief」ではない。

### 3.2 IMPORTED — ライフサイクルを通らない履歴登録

`research-seed-five-axes-of-scaling.md`の`seed-q0`（Five Axes of Scaling investigation本体）は、
このサービスのAgent Loopが一度も実行していない — AlgoForge導入前/このサービス構築前に
人間とAIが手動で行った投資調査を、後から履歴として取り込んだものである。

これを`research_questions.status = "completed"`として登録するのは**モデル上誤り**だった
（§6参照）— `completed`は本来`research_sessions`（Agent Loopが実際に走った記録）の終端状態で
あり、Agent Loopを一度も回していない研究に付けるべきステータスではない。正しくは
`status = "imported"`という、他の5状態（draft/briefing/pending_approval/approved、そして
研究sessionsの running/paused/completed/failed/budget_exceeded）と並ぶ**第三の独立した経路**
として扱う:

- `IMPORTED`な`research_questions`は`DRAFT`からもBRIEFINGからも遷移してこない。登録された
  瞬間から`IMPORTED`。
- 対応する`research_sessions`行も作るが、`status="imported"`という専用値を持ち、
  `budget_used_json`は空（Agent Loopが実際に消費した予算という概念自体が無い）、
  `agent_steps`は0件（ターン単位のツール呼び出し履歴が存在しない）。
- `reports`は持ってよい（このケースのように、既存レポートをそのまま`content_path`として
  参照できる）。
- 唯一の存在意義は**§5.3(A) Prior-Art Searchの検索対象になること**。将来のBRIEFINGが
  `list related research`する際、`IMPORTED`な行も`approved`/`completed`な行と同列に
  検索対象へ含める。

---

## 4. Research Brief（構造化案）スキーマ

BRIEFINGフェーズでAIが生成し、人間が承認/編集するオブジェクト。承認後はこれが
Agent Loopの「憲法」になり、予算・停止条件の判定はすべてここに書かれた値を参照する。

```jsonc
{
  "title": "USDJPY H1: LSTM vs Transformer 予測精度比較",
  "hypothesis": "TransformerはLSTMよりdirectional accuracyが高い（USDJPY H1, 2019-2024）",
  "scope": {
    "symbols": ["USDJPY"],
    "timeframes": ["H1"],
    "dataset_ids": [12],          // 既存を使う場合。空なら新規収集を許可するか下のflagで指定
    "allow_new_data_collection": false
  },
  "success_criteria": [
    { "metric": "directional_accuracy", "op": ">=", "value": 0.55 },
    { "metric": "val_loss_improvement_vs_baseline", "op": ">=", "value": 0.05 }
  ],
  "budget": {
    "max_training_runs": 8,
    "max_wall_clock_minutes": 240,
    "max_hparam_search_jobs": 1,
    "max_new_datasources": 0,
    "max_llm_cost_usd": 5.00,        // LLM API呼び出しの累計コスト上限（§5.1, §6のllm_pricing参照）
    "deadline": "2026-07-27T00:00:00+09:00"
  },
  "stopping_policy": {
    "no_improvement_patience": 3,     // 連続N回改善なしで打ち切り検討
    "min_confidence_to_conclude": "medium"
  },
  "llm_config": { "provider": "anthropic", "model": "claude-sonnet-5" },  // 未指定ならSettingsのデフォルト
  "related_questions": [                 // §5.3(A) Prior-Art Searchの結果。BRIEFING時にAIが埋める
    { "question_id": 41, "relation_type": "follow_up_candidate",
      "similarity_note": "同じUSDJPY H1で2024-06に類似仮説を検証済み。val_lossは改善したがdirectional accuracyは未達だった" }
  ],
  "report_audience": "quant_researcher"  // レポートの詳細度/トーン調整に使う
}
```

- `success_criteria`は機械的に判定できる形にする（曖昧な自然文のままループの停止条件にしない）。
- `budget`が唯一の「フル自律の境界線」。ここを超えるアクションは実行せず、その時点で
  `REPORTING`に遷移して「予算切れ」として理由を明記する。

---

## 5. Agent Loop（自律実行ロジック）

1周（サイクル）の処理:

```
1. Survey   — list_datasets / list_models / get_model_validations / compare_model_runs /
              get_dataset_characteristics などread-only呼び出しで現況を把握
2. Reason   — Briefの仮説・成功基準・これまでのAgentStep履歴と突き合わせ、
              「次に何をすべきか」をLLMに判断させる
3. Decide   — 選択肢:
              a) train      — 新しいhyperparamsや前処理レシピでstart_training_run
              b) backtest   — 学習済みモデルをdeploy_modelしてstart_strategy_run（backtest）
              c) collect    — allow_new_data_collectionがtrueならcollect_data
              d) conclude   — 成功基準達成 or 収束 or 予算切れ → レポートへ
4. Act      — 選んだMCPツールを実行し、戻り値(job_id等)をAgentStepとして永続化
5. Wait     — Webhook（training.completed等）を購読して非同期待ち。
              フォールバックとしてポーリング（get_training_status）も実装
6. Evaluate — 結果を記録し、budget_usedを更新。stopping_policyと突き合わせ、
              満たせば5へ戻らずconclude
```

- **Webhook駆動が基本**。BRIEFING承認時にResearch Agent Service自身のエンドポイントを
  AlgoForgeの`POST /webhooks`に登録し、`training.completed` / `run.completed` /
  `collection.completed`（および対応する`.error`）をHMAC検証つきで受信する。
  ポーリングは通信断からの復旧用のフォールバックに限定する。
- **セッションは自分のDBで再開可能に**する。プロセスが落ちても、`AgentStep`の最後の
  コミット地点から再開できるようにする（`agentic_trade`の「DB経由でmonitorとpipelineを疎結合にする」
  設計と同じ思想 — 実装時に参考にする価値がある）。
- LLM本体は`AgentRunner`という薄いインターフェースの背後に置き、プロバイダを差し替え
  可能にしておく。具体的な要件は§5.1参照（OpenAI / Gemini / Anthropicの3社をサポート）。

### 5.1 LLMプロバイダの抽象化（マルチベンダー対応）

Anthropic・OpenAI・Geminiの契約を既に持っているため、この3社を**サポートし、研究課題
（Research Brief）ごとに切り替えられる**ようにする。

```
              ┌─────────────────────────────┐
Agent Loop ──▶│  AgentRunner (interface)      │
              │   generate_plan(history,      │
              │     tool_schemas) -> Action    │
              └───────────────┬───────────────┘
                               │ 実装を選択 (brief.llm_config.provider)
        ┌──────────────────────┼──────────────────────┐
        ▼                      ▼                      ▼
  AnthropicAdapter        OpenAIAdapter           GeminiAdapter
  (Messages API /         (Responses API          (Gemini API /
   Claude Agent SDK,       function calling)        Google ADK)
   MCPをネイティブ利用)
        └──────────────────────┬──────────────────────┘
                               ▼
                    ToolSchemaTranslator
        (MCPの JSON Schema を各社のtool定義形式へ変換:
         Anthropic: input_schema / OpenAI: function.parameters /
         Gemini: function_declarations)
                               │
                               ▼
                     AlgoForge MCP Server（共通・不変）
```

- MCPのツール一覧・スキーマはAlgoForge側で1つしかない。各アダプタは`ToolSchemaTranslator`
  を通して同じスキーマをそれぞれのtool-calling形式に変換するだけで、MCP呼び出しの実体
  （実際にツールを叩く処理）はプロバイダに関わらず共通。
- プロバイダ/モデルの選択は`research_briefs.brief_json.llm_config`（Brief単位）で保持し、
  未指定時はSettingsのシステムデフォルトを使う。
  ```jsonc
  "llm_config": { "provider": "anthropic", "model": "claude-sonnet-5" }
  ```
- **セッション途中でのプロバイダ切り替えは不可**とする（会話履歴・tool-call形式の互換性が
  プロバイダをまたいで保証できないため）。切り替えたい場合はセッションを`PAUSED`にして
  新しいBrief（新セッション）を起こす。レポートには使用したprovider/modelを明記する。
- 障害時（レート制限・障害）の挙動はMVPでは**自動フェイルオーバーしない** — セッションを
  `PAUSED`にしてエラー内容を記録し、人間の判断を仰ぐ。プロバイダをまたぐ自動切り替えは
  推論の一貫性が壊れるリスクがあるため、オプトインの後回し機能とする（§11）。
- APIキーはこのサービス独自の`.env`/Settings→API Keysページで管理
  （`ANTHROPIC_API_KEY` / `OPENAI_API_KEY` / `GOOGLE_API_KEY` — ワークスペース内の他プロジェクト
  と同じ命名規約に合わせる）。

### 5.2 同時実行数の管理（Web設定から変更可能）

AlgoForge側のCeleryキューはconcurrencyが固定（`collection`=3, `characteristics`=12,
`training`=2, `backtest`=5）。複数のResearch Sessionが自律的にジョブを投げ続けると、
この共有キューを占有してAlgoForge上の他の利用（人間の手動実行や他のセッション）を
詰まらせる恐れがある。そのため同時実行数は**このサービスのSettings画面から変更できる
グローバル設定**として持つ。

```
system_settings
  max_concurrent_sessions          -- 同時にRUNNINGにできるResearch Sessionの数
  max_inflight_jobs_per_session    -- 1セッションが同時に持てるAlgoForgeジョブ数（既定: 1 = 逐次実行）
  max_inflight_jobs_per_queue      -- ジョブ種別(training/backtest/collection)ごとの全セッション合算の上限
                                      -- AlgoForge実際のワーカーconcurrency以下に設定するのが安全
```

- 判定は`Act`ステップの直前に`Scheduler`コンポーネント（カウンタ/セマフォ、DB行ロックか
  Redisで実装）が行う。上限に達している場合、`Reason`/`Decide`自体は進めてよいが実際の
  ツール呼び出しは保留し、セッション状態を`WAITING_FOR_SLOT`にする。空きはWebhookの
  `*.completed`イベントで解放されたときに再チェックする。
- 設定変更は次回の`Act`判定から即時反映（実行中セッションの再起動不要）。
- MVPでは「AlgoForge側の実際のキューconcurrency」はこのサービスの設定として**人間が
  手動で入力**する（AlgoForge側にそれを問い合わせるMCPツールは現状無い）。将来的に
  AlgoForge側へ`get_queue_status()`のようなMCPツールが追加されれば自動同期できる（§11）。

### 5.3 複数研究課題間の調整（重複検出とリソース競合の防止）

これは性質の異なる2つの問題に分けて考える。**(A) 同じことを重複して調べてしまう問題**は
「関連する過去研究の検索」で防げるが、**(B) 別の仮説の研究同士がたまたま同じリソースを
書き換えてしまう問題**は検索では防げない（仮説が全く違っても、対象のmodel_idやstrategy_idが
偶然重なることはある）。

#### (A) 関連研究の事前検索（Prior-Art Search）

BRIEFINGフェーズで、Brief案を人間に見せる**前**に実行する。

```
1. Scope絞り込み（SQL、LLM不要・低コスト）
   新規質問のscope（symbols/timeframes/dataset_ids/architecture）と重なる
   research_briefs / reports を抽出。母数が少ないうちはこれだけで十分実用的。

2. LLM判定（BRIEFING呼び出しの一部として1発で）
   ステップ1の候補（タイトル・仮説・reportのsummary）をプロンプトに含め、
   「これらは今回の質問とどう関係するか」をLLMに判定させる。
   → related_questions: [{ question_id, relation_type, similarity_note }]
      relation_type ∈ likely_duplicate | follow_up_candidate |
                       shares_resources | related_but_distinct | raised_by_review
      -- raised_by_reviewは§7.1のレビュー機能がfindingをDRAFT研究課題に昇格させたときのみ付く。
      -- BRIEFINGが自分でPrior-Art Searchした結果ではなく人間の昇格操作に由来するので、
      -- BRIEFING側の再検証（§3.1）はこの関係についても他の候補と同じ扱いで行う
```

- 候補数が増えて1プロンプトに収まらなくなったら埋め込みベクトル検索（pgvector等）に
  切り替える — が、MVPでは過剰設計になるので**まずはSQLフィルタ＋LLM判定のみ**とする
  （§10 Phase 0）。埋め込み導入はPhase 3以降の課題（§9）。
- 結果は`research_briefs.brief_json.related_questions`として保存し、
  **PENDING_APPROVAL画面に必ず表示**する（今回の要件どおり、人間が確認するBrief案の一部）。
  `likely_duplicate`かつ元研究が`COMPLETED`の場合、AIは承認案の中で「新規に走らせるより
  既存レポートを見ることを推奨」と明記する。それでも続行するかどうかは人間の判断。
- `follow_up_candidate`の場合、新Briefの`hypothesis`に前回の結論を出発点として自動で
  織り込む（ゼロから同じ探索をやり直さないようにする）。

#### (B) 実行中セッション間のリソース書き込み競合の防止

§5.2のSchedulerを拡張し、共有リソースへの**書き込み**を伴うAct（特に`deploy_model` —
[R-6](requirements.md#r-6-deploy_modelに競合制御が無い--p2)でAlgoForge側に競合制御が
無いことを確認済み）の前に「今このリソースを別のRUNNINGセッションが書き込み中でないか」
をチェックする。

```
resource_claims(session_id, resource_type[model|dataset|strategy],
                 resource_id, mode[read|write], claimed_at, released_at)
```

- **読み取り専用の利用（datasetの参照、modelを対象にした新規`start_training_run`など）は
  claim不要** — parquetは並行読み取り安全だし、training_runは行が増えるだけで衝突しない。
  同じmodel_idを複数セッションが別々のhyperparamsで訓練すること自体はむしろ許容する
  （並行探索として有用）。
- **`deploy_model`のような「どのtraining_runを正とするか」を書き換えるAct**の直前だけ
  `write` claimを取る。既に別セッションが同じresource_idにwrite claimを持っていたら、
  即エラーにはせずDecideへ「busy」を返し、LLMに(i)新しいmodelとしてforkする
  (ii) `WAITING_FOR_RESOURCE`として待つ (iii) 待つと予算を超えるならconcludeする、
  のいずれかを選ばせる。
- claimはActの完了（成功/失敗いずれも）で即release。長時間の待ちロックにはしない。
- これはあくまで**このサービス側の自衛策**。AlgoForge側でR-6（`ml_models`への
  versionカラム＋楽観ロック）が入れば根本対策になるので、`requirements.md`のR-6は
  この設計の直接の依存として優先度を上げてよい。

> **実データでの検証例:** [research-seed-five-axes-of-scaling.md](research-seed-five-axes-of-scaling.md)に、
> 既存レポート「Five Axes of Scaling」から抽出した実例を置いている（IMPORTED 1件・
> seed_brief_json付きDRAFT 5件 — 詳細は§3.1/3.2）。

---

## 6. データモデル（Research Agent Service側・独自DB）

```
research_questions(id, raw_text, seed_brief_json,
                    status[draft|briefing|pending_approval|approved|imported],
                    created_at)
        -- seed_brief_json: §3.1。BRIEFINGへのヒントであり、research_briefs.brief_jsonの代わりにはならない
        -- status="imported"は他の4値と並ぶ独立経路（§3.2）。draftからimportedへは遷移しない

research_briefs(id, question_id, brief_json, approved_at, approved_by, version)
        -- imported由来のquestionに付くbriefはapproved_at/approved_byがnull
        -- （人間の承認イベントが実際には発生していないことを表す）

research_sessions(id, brief_id,
                   status[running|paused|completed|failed|budget_exceeded|imported],
                   started_at, ended_at, budget_used_json)
        -- status="imported": Agent Loopを実行していない履歴の器（§3.2）。
        --   budget_used_jsonは常にnull、agent_stepsは常に0件

agent_steps(id, session_id, seq, phase[survey|reason|decide|act|evaluate],
            reasoning_text, mcp_calls_json, mcp_results_json,
            provider, model, input_tokens, output_tokens, estimated_cost_usd,
            created_at)

decisions(id, session_id, step_id, decision[train|backtest|collect|conclude|abort],
          rationale, linked_algoforge_ids_json)   -- training_run_id / run_id 等の相互参照

reports(id, session_id, format[markdown|pdf], content_path, summary, recommendation,
        created_at)

budget_ledger(id, session_id, dimension, amount_used, amount_limit)
        -- dimension例: training_runs / wall_clock_minutes / hparam_search_jobs /
        --              new_datasources / llm_cost_usd

llm_pricing(id, provider, model, input_price_per_mtok_usd, output_price_per_mtok_usd,
            cached_input_price_per_mtok_usd, updated_at)
        -- 手動メンテナンス。各社の価格改定に追従して更新する
        -- (Anthropicの最新価格は claude-api スキルが参照元になる)

system_settings(id, key, value_json, updated_at)
        -- max_concurrent_sessions / max_inflight_jobs_per_session /
        -- max_inflight_jobs_per_queue / default_llm_config など

resource_claims(id, session_id, resource_type[model|dataset|strategy], resource_id,
                 mode[read|write], claimed_at, released_at)
        -- §5.3(B)。deploy_model等の書き込みActの直前だけwrite claimを取る
```

`agent_steps.estimated_cost_usd`は`input_tokens`/`output_tokens`と`llm_pricing`の単価から
その場で計算して書き込む。`budget_ledger`の`llm_cost_usd`はセッション内の`agent_steps`
の合計として更新され、`budget.max_llm_cost_usd`に達したら以後の`Act`はconclude一択に
制限される。

`decisions.linked_algoforge_ids_json`でAlgoForge側のtraining_run_id/strategy_run_idを
記録しておくことで、レポートから「AlgoForge UI上でこのrunを直接見る」導線を作れる
（`http://localhost:3000/model/{id}/training-runs/{run_id}`等へのディープリンク）。

---

## 7. レポート出力

- 形式: Markdown（そのままArtifact化 or PDF変換）。
- 必須セクション: 仮説 → 試行した内容の一覧（データセット/モデル/ハイパラの表）→
  結果比較表・グラフ（AlgoForgeの「Data × Model Analysis」散布図の考え方を踏襲）→
  結論（成功基準を満たしたか）→ 推奨アクション（デプロイすべきか、さらなる研究が必要か）→
  使った予算の内訳 → 再現用リンク（AlgoForge run ID一覧）。
- 「結論に至らず予算切れ」の場合も必ずレポートを出す（何を試して何が分からなかったかを残す）。

### 7.1 既存研究へのAIレビュー（問題点・疑問点の指摘 + 重要度付け）

Agent Loop（§5、能動的に新しい訓練を実行して仮説を検証する）とは別に、**既存のreport
（`IMPORTED`か`COMPLETED`かを問わない）を読んで批評する、読み取り専用の軽量な操作**を
独立した機能として持つ。Agent Loopのようなbudget/Scheduler/write系MCP呼び出しを一切
必要としないため、実装コストが低く、フルのAgent Loopより先に単体で価値を出せる
（§10でPhase 0扱いにした理由）。

#### 目的関数を分ける — 「レビュー」は「研究」と別モード

`AgentRunner`（§5.1）は同じインターフェースを再利用するが、システムプロンプト（役割）が
根本的に違う: 研究モードは「仮説を検証するために次に何をすべきか」を考えるが、レビューモードは
**「この結論をどこまで信じてよいか、何が確認されていないか」だけを考える**。書き込み系
MCPツールは一切与えない（read-only: `list_*` / `get_*` / `compare_*` のみ）。

**もっとも重要な設計上の制約:** レビューは、対象reportが**自己申告していない**問題を
見つけたときに初めて価値を持つ。この「Five Axes of Scaling」のように、レポート自身が
Confirmed/Falsified/Partial/Exploratoryのbadgeで自己評価し、Evidence Ledger（§5）まで
用意している場合、それを機械的になぞって"再発見"しても付加価値はゼロ。プロンプトには
必ず「対象reportが既に開示・自己評価済みの限界は再指摘するな。開示されていない問題、
または開示の粒度が甘い問題だけを報告せよ」という指示を含める。

#### findingsのスキーマと重要度

```
reviews(id, target_type[report], target_id, reviewer_provider, reviewer_model,
        status[pending|completed|failed], created_at, completed_at)

review_findings(id, review_id, type[issue|question], importance[critical|high|medium|low],
                 target_ref, summary, detail, verified_via_mcp, verification_note,
                 suggested_action, promoted_question_id)
        -- target_ref: 対象reportの中の場所（例 "§12", "evidence_ledger:late-regime-instability"）
        -- type=issue: 結論の信頼性に影響する欠陥。type=question: 欠陥とは言えないが未確認・未解決の点
        -- verified_via_mcp: 主張をAlgoForgeの実データ(get_model_validations等)と突き合わせて
        --   確認を試みたか。試みて確認できなかった場合はimportanceを上げる材料になる
        -- promoted_question_id: 人間がこのfindingを新規research_questions(DRAFT)に昇格させた場合のFK
```

`importance`は4段階（`critical|high|medium|low`）。目安:
- **critical** — この欠陥が正しければ、報告書の中心的な結論（capstone等）が崩れる
- **high** — 特定の主要claimの信頼性を大きく損なう。再検証なしに次の意思決定の土台にすべきでない
- **medium** — 信頼性に一定の影響はあるが、claim全体を無効化するほどではない
- **low** — 些末、または研究そのものより運用上の注記（再現性メタデータの欠落など）

重要度をLLMの主観だけに任せると評価がブレる。既にこのワークスペースに実在する
**同一の物差し**を評価ルーブリックとして毎回システムプロンプトに含める:
[CLAUDE.mdの「Comparing training runs」規約](../CLAUDE.md#comparing-model-training-runs)
（seed discipline、matched optimizer steps等）と、対象report自身の「Shared methodology」
節（あれば）。この2つに対する逸脱を機械的にチェックする部分と、それ以外の自由な批評を
分けて評価する。

#### 「疑問点」から新しい研究課題への昇格

`type=question`のfindingは、そのまま[§3.1](#31-draftへの種の持ち込みseed_brief_json)の
`seed_brief_json`付き`DRAFT`研究課題の下書きになりうる。ただし**自動昇格はしない** — 
findingはあくまで一覧として人間に提示し、「これをresearch_questionsに昇格する」ボタンを
押した時だけ`promoted_question_id`が埋まり新しい`DRAFT`行が作られる（PENDING_APPROVAL
必須の原則、§3と同じく人間の意思決定を挟む）。`related_questions`には元のreportへの
参照を`relation_type: "raised_by_review"`として自動で入れる（§4のenum一覧に追加）。

#### 起動方法

- オンデマンド: 人間が特定のreportを指定して「レビューして」と依頼する（今回の主要ユースケース）。
- 任意で自動化: `system_settings.auto_review_on_import`をtrueにすると、IMPORTEDなreportが
  登録された瞬間に自動でレビューが1回走る。
- 将来的な拡張: REPORTING（§5の`conclude`後）で、Agent Loop自身が出したreportを公開前に
  セルフレビューする、という使い方も同じ仕組みで可能（Phase 3以降、§10）。ただしこの場合
  「自己申告していない問題を見つける」という前提が崩れやすい（同じLLM文脈の続きだと自分の
  結論を無批判に追認しがち）ので、レビュー呼び出しは会話履歴を共有しない独立したcontextで
  行う必要がある。

> **実例:** [research-review-five-axes-of-scaling.md](research-review-five-axes-of-scaling.md)に、
> 「Five Axes of Scaling」を対象にした実際のレビュー例（findings 6件、importance付き、
> うち2件をseed-q6/q7として新規DRAFT研究課題に昇格）を置いている。

---

## 8. 既存ロードマップ項目との関係

このサービスは`algoforge/docs/roadmap.md`に既にある以下の項目を前提/加速する。実装順序を
決めるときに参照:

| ロードマップ項目 | 現状 | この設計への影響 |
|---|---|---|
| MCP: Agent session persistence (Next) | 未着手 | Research Agent Service側で独自に持つので**必須依存ではない**。ただしAlgoForge側にも入れば二重化を避けられる |
| ML: AutoML mode（Later） | 未着手 | 本サービスの「train決定ロジック」がまさにこれ。AlgoForge本体に汎用AutoMLを後で足すなら、本サービスのstopping_policyロジックを吸い上げて共通化できる |
| Strategy: agentic condition handler（Later） | 未着手 | 直接の依存ではないが、将来「戦略の条件判定もこのエージェントに任せる」方向に自然につながる |
| MCP: `get_equity_curve(run_id)`（Next） | 未着手 | あると助かる（現状は`get_run_metrics`で代替可） — 必須ブロッカーではない |
| MCP: `algoforge://dashboard`が live counts を返す（Now, 未チェック） | 未着手 | Survey段階でdashboardリソースを使うなら、先にこれが直った方が良い |
| MCP: create_preprocessed_dataset 未実装 | 既知のギャップ | 前処理レシピを自律生成したい場合はREST直叩きが必要（§2参照） |

---

## 9. 技術スタック（案）

| 領域 | 案 | 備考 |
|---|---|---|
| Agent実行 | `AgentRunner`抽象 + Anthropic/OpenAI/Gemini各アダプタ | §5.1参照。Brief単位でプロバイダ/モデルを選択、Settingsにデフォルトを持つ |
| API | FastAPI（AlgoForgeと合わせて学習コストを下げる） | |
| DB | PostgreSQL（本番）/ SQLite（開発） | AlgoForgeとは別インスタンス/別スキーマ |
| 非同期実行 | Celeryまたは軽量なasyncioタスク | Research Loop自体はAlgoForgeほど重くない想定。まずはasyncio + 単一ワーカーで十分な可能性 |
| Webhook受信 | FastAPIエンドポイント + HMAC検証 | AlgoForgeの`webhooks/dispatcher.py`と対のクライアント実装 |
| フロントエンド | Next.js（研究課題投稿・Brief確認・進捗・レポート閲覧） | AlgoForgeのSWR/SSEパターンを踏襲 |
| 関連研究検索 | まずSQLのscope重なりフィルタ＋LLM判定（§5.3A） | pgvector等の埋め込み検索はPhase 3以降、過去研究が増えて1プロンプトに収まらなくなってから |

---

## 10. フェーズ別実装計画（案）

| フェーズ | 内容 |
|---|---|
| Phase 0 | **既存reportへのAIレビュー（§7.1、オンデマンドのみ）** — Agent Loop本体より先に単体で出荷できる最小機能。Research Brief生成（LLM 1発呼び出し + read-only MCP調査 + §5.3(A)のSQLフィルタ版prior-art search）、承認UIのみ。学習等の実行は手動トリガーのモック |
| Phase 1 | Agent Loopの最小実装（train/conclude の2択のみ）、Webhook受信、予算ガード、Scheduler（§5.2同時実行数 + §5.3(B)リソースclaim）、レポート生成（Markdown） |
| Phase 2 | backtest/collectを選択肢に追加、hparam search対応、セッション再開（プロセス再起動耐性） |
| Phase 3 | UI強化（進捗のライブ表示、レポートのPDF出力、複数研究課題の並行管理）、prior-art searchを埋め込みベクトル検索に切り替え |

---

## 11. オープンな論点（要検討）

1. **LLMベンダー** — ✅決定: Anthropic/OpenAI/Geminiの3社をサポートし、Brief単位で
   切り替え可能にする（§5.1）。残課題: 各社のtool-calling仕様差分（並列tool call可否、
   streaming時の挙動など）を`ToolSchemaTranslator`でどこまで吸収しきれるかは実装しながら
   検証が要る。
2. **同時実行数** — ✅決定: Web設定（`system_settings`）から変更可能にする（§5.2）。
   残課題: AlgoForge実際のキューconcurrencyは現状MCP経由で取得できないため手動入力に
   頼る。ズレると自己スロットリングが効かなくなるので、将来`get_queue_status()`的な
   MCPツールが欲しい。
3. **コスト見積もり** — ✅決定（LLM分）: `llm_pricing`テーブルで単価を持ち、トークン
   使用量から算出する以外に方法はない（§6）。ただし価格改定に追従する手動メンテが
   前提。AlgoForge側の計算コスト（GPU/CPU時間）はドル換算していない —
   MVPでは`budget`の回数・時間ベースの上限のみで代替し、統合的な「総額予算
   (`max_total_cost_usd`)」化は queueごとの時間単価設定が要るため後回しとする。
4. **複数研究課題の相互作用** — ✅決定: (A)重複研究はBRIEFING時のPrior-Art Search、
   (B)リソース書き込み競合は`resource_claims`によるSchedulerの拡張、で対応する（§5.3）。
   残課題: `relation_type`の閾値・判定基準（LLM判定のブレをどう抑えるか）、および
   `resource_claims`の対象を`deploy_model`以外（例えば`update_strategy`）にも広げるべきかは
   実装しながら判断する。根本対策はAlgoForge側のR-6（`ml_models`の楽観ロック）— 依存として
   優先度を上げた。
5. **人間の介入ポイント** — PAUSED中にユーザーがヒントを与える（chatのような）機能は
   MVP後の追加でよいか、Phase 1から要るか。
