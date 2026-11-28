# algoforge の移動と研究リポジトリの分割: 移行計画

- 版: **v0.1 草案。未承認・未実行。**(2026-11-28)
- この文書の作成で行ったのは読み取りの調査と、使い捨てフォルダでのジャンクションの試験だけである。何も移動・作成していない。
- パスは公開リポジトリに残るため、`$env:USERPROFILE`(ユーザーフォルダ)で書く。

## 0. 判断してほしい点

| # | 判断 | この案 |
|---|---|---|
| 1 | algoforge の移し方 | `C:\AIWS\algoforge` へ**実体を移し、元の場所にジャンクションを残す** |
| 2 | 研究リポジトリ | `C:\AIWS\algoforge-research`、**非公開**、研究ごとのフォルダ |
| 3 | パスの混入防止 | 研究リポジトリのコミット前チェックで、ユーザーフォルダのパスを拒否する |
| 4 | 分担 | 研究リポジトリの準備と移動後の確認は Claude、**移動の操作は利用者**(§4) |

## 1. 目的と、移しても残るもの

目的は、研究の報告や結果にユーザーフォルダのパスが出にくくすること。次は**この移行では消えない**。

| 残るもの | 理由 |
|---|---|
| Python の実行ファイル、一時フォルダ、Claude のデータのパス | ユーザーフォルダの下にある |
| Study Manager の添付の保存先のパス | Study Manager は移さない |
| ほかのプロジェクト(`finance` の下)のパス | 移さない |
| 公開済みの履歴(ファイル内のパス、コミットの作者名とメール) | 履歴の書き換えが必要(対象外) |

したがって、**「書き込む内容に入れない」仕組み(相対パス、コミット前チェック)が本体の対策**で、移動は補助である。

`C:\AIWS\finance` のジャンクション(2026-11-28 に作成されたもの)を使うだけでは解決しない。試すと、`git rev-parse --show-toplevel`、Python の `Path.resolve()`、`os.getcwd()` は**元のユーザーフォルダのパス**を返した(`os.path.abspath` はそのまま)。

## 2. 調べた依存

| 対象 | 状況 | 移動の影響 |
|---|---|---|
| Python | システムの Python 3.13 を使用。algoforge 内に仮想環境はない | なし(作り直し不要) |
| 編集可能インストール | `finance_client` など 5 件。algoforge は含まれない | なし(それらは移さない) |
| データベース | `*path*` 列と全テーブルに旧パスの記録なし | なし |
| Docker | PostgreSQL と Redis は名前付きボリューム。compose のプロジェクト名はフォルダ名 `infra` のまま | データへの影響なし |
| Docker のイメージ作成・開発用 compose | ビルドの起点が `../../` で、隣の `finance_client` を読む | **そのままでは失敗する。**さらに起点が `C:\AIWS` 全体になる。直すまで実行しない |
| `artifacts`、`backend/artifacts` | `D:\Data\algoforge\artifacts` へのジャンクション | フォルダと一緒に移る(実体は D のまま) |
| 稼働中のプロセス | API(8000)、Celery の worker と beat、Web の開発サーバー | **移動前に停止する** |
| 他プロジェクトからの参照 | `algoforge_pipeline` の 5 ファイル、Study Manager 側の 1 ファイルが algoforge のパスを持つ | 元の場所のジャンクションで動き続ける |
| 定期タスク | 2 件はほかのプロジェクトのもの | なし |
| A1 の凍結コード | リポジトリ内の相対パスと `L:` を使う | 移動後に起動前の確認を実行する |
| `web/.next` | 以前の場所の情報を含む作業用の出力 | 削除して作り直す |
| git のワークツリー | 以前のセッションの一時フォルダに 1 件 | ジャンクションで参照は保たれる |
| Claude のメモリ | フォルダのパスから作る名前の場所にある | コピーが必要 |

調べたのは文字列の検索とコマンドの確認で、すべての依存を保証するものではない。**エディタや別のセッションが開いているフォルダは移動に失敗することがある。**

## 3. 移行後の構成

```
C:\AIWS\
├ algoforge\             本体(今のリポジトリ。公開)
├ algoforge-research\    研究(新しいリポジトリ。非公開)
│  ├ CLAUDE.md
│  ├ .gitignore
│  ├ .githooks\pre-commit
│  ├ tools\check_algoforge.py
│  └ <study名>\          plan.md, report.md, code, results, requirements.txt, algoforge_commit.txt
└ finance\               (既存のジャンクション。変えない)
```

- 研究コードは環境変数 `ALGOFORGE_DIR`(= `C:\AIWS\algoforge`)で本体を読み込む。HTTP API だけで足りる研究は読み込まない。
- `check_algoforge.py` は、記録したコミットと本体の HEAD が一致し、未コミットの変更がないことを確認する。
- 大きなデータ(parquet、npz、モデル)、`.env`、秘密情報はコミットしない。
- コミット前チェックは `C:\Users\`、`C:/Users/`、`/c/Users/` を含む変更を拒否する。チェックのファイル自体にユーザー名は書かない。
- 作者は、このリポジトリの設定で GitHub の noreply アドレスにする。
- **既存の研究(DP1〜DP3、F1、A0・A1・A1R など)は移さない。**進行中の A1 は今の場所で続ける。

## 4. 手順

### 段階 1: 研究リポジトリの準備(Claude。移動の前でもできる。何も壊さない)

1. `C:\AIWS\algoforge-research` を作り、`git init`、上の構成のひな形を作る。
2. コミット前チェックが、パスを含むファイルを拒否することを試す。
3. GitHub の非公開リポジトリを作る。**作成と最初の push は、実行前に改めて確認する。**

### 段階 2: 移動前の記録(Claude または利用者)

```powershell
git -C $env:USERPROFILE\workspace\finance\algoforge status --short
```
```powershell
git -C $env:USERPROFILE\workspace\finance\algoforge rev-parse HEAD
```
```powershell
cmd /c "$env:USERPROFILE\workspace\finance\algoforge\scripts\test-all.bat unit"
```
```powershell
Push-Location $env:USERPROFILE\workspace\finance\algoforge\backend\research\anomaly_a1; python -c "import a1_s1; print(a1_s1.code_state()['head'])"; Pop-Location
```

テストの結果は、移動前から失敗している項目も含めて控える。未コミットの変更があれば先に片付ける。

### 段階 3: 移動(利用者。algoforge を開いている Claude のセッション、エディタ、端末をすべて閉じてから)

1. API、worker、beat、Web を停止する。残っていないことを確認する(何も表示されなければよい)。

```powershell
Get-CimInstance Win32_Process | Where-Object { $_.Name -match 'python|node' -and $_.CommandLine -match 'celery_worker|--port 8000|algoforge[\\/]+web' } | Select-Object ProcessId, Name
```

2. 移動する。同じドライブ内の名前の変更で、失敗した場合は元のままになる。**`Move-Item` は使わない**(失敗時にコピーと削除に切り替わると、`artifacts` のリンク先に触れる恐れがある)。

```powershell
[System.IO.Directory]::Move("$env:USERPROFILE\workspace\finance\algoforge", "C:\AIWS\algoforge")
```

3. 元の場所にジャンクションを作る。

```powershell
New-Item -ItemType Junction -Path "$env:USERPROFILE\workspace\finance\algoforge" -Target "C:\AIWS\algoforge"
```

4. Web の作業用の出力を消す。

```powershell
Remove-Item -Recurse -Force C:\AIWS\algoforge\web\.next
```

5. `C:\AIWS\algoforge` で新しい Claude のセッションを開く。

### 段階 4: 確認(Claude、新しいセッション)

| 確認 | 合格の条件 |
|---|---|
| `git status`、HEAD | 移動前と同じ |
| `git rev-parse --show-toplevel` | `C:/AIWS/algoforge` |
| `artifacts`、`backend/artifacts` | ジャンクションで、`D:` のファイルが読める |
| 単体テスト | 移動前と同じ結果 |
| A1 の `code_state()` | 例外なく同じ HEAD を返す |
| 起動(`scripts\start-stack.bat`、Web は `web` で `npm run dev`) | `/api/v1/health` が正常、画面でデータセットを 1 件表示できる |
| 元の場所のパス | ジャンクション経由で読める |

注意: 起動スクリプトは `colab` キューを含まない。今の worker は含むので、必要なら同じキュー指定で起動する。

### 段階 5: 後始末(Claude。確認がすべて通ってから)

1. メモリを新しい場所へコピーする。新しいフォルダ名は、最初のセッションができた後に確認する。

```powershell
Get-ChildItem $env:USERPROFILE\.claude\projects -Directory | Where-Object Name -match 'finance-algoforge$|AIWS-algoforge$'
```

2. `CLAUDE.md` のユーザーフォルダのパスを、書かない形に直す(以後の版から)。
3. `backend/.claude/settings.local.json` などの許可設定に旧パスがあれば、利用者が確認する。

## 5. 戻し方

プロセスを停止し、セッションを閉じてから行う。

1. 元の場所がジャンクションであることを確認する(`Junction` と表示される)。

```powershell
(Get-Item "$env:USERPROFILE\workspace\finance\algoforge").LinkType
```

2. リンクだけを削除する。**`Remove-Item -Recurse` は使わない。**

```powershell
cmd /c rmdir "$env:USERPROFILE\workspace\finance\algoforge"
```

3. 実体を戻す。

```powershell
[System.IO.Directory]::Move("C:\AIWS\algoforge", "$env:USERPROFILE\workspace\finance\algoforge")
```

## 6. 対象外(それぞれ別の作業と承認)

- Docker のイメージ作成と開発用 compose の修正(直すまで、イメージのビルドと開発用 compose は使わない。起動中の PostgreSQL と Redis はそのまま使える)。
- 公開済みの履歴の書き換え。
- 既存の研究コードと結果ファイルの移動や書き換え(DP2・DP3 の `s0.json` のパスを含む)。
- `finance` のほかのプロジェクト、Study Manager の移動。
- `algoforge` 本体へのコミット前チェックの追加(今のファイルがパスを含むため、方針を決めてから)。
- 元の場所のジャンクションの削除(参照がなくなってから)。
