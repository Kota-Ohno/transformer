# 開発者向けガイド

このドキュメントは、AIエージェントや開発者がこのプロジェクトで作業する際のガイドラインです。

## Gitコミット時の注意事項

### コミットメッセージの文字化け対策

Windows環境でGitコミットメッセージに日本語を含める場合、文字化けが発生する可能性があります。以下の方法で回避できます：

#### 方法1: 一時ファイルを使用（推奨）

```bash
# コミットメッセージを一時ファイルに書き込む
echo "コミットメッセージ" > commit_msg.txt
echo "" >> commit_msg.txt
echo "- 変更内容1" >> commit_msg.txt
echo "- 変更内容2" >> commit_msg.txt

# 一時ファイルを使用してコミット
git commit -F commit_msg.txt

# または、既存のコミットを修正する場合
git commit --amend -F commit_msg.txt

# 一時ファイルを削除
rm commit_msg.txt
```

#### 方法2: Gitエディタを使用

```bash
# Gitエディタを開いてコミットメッセージを記入
git commit

# または、既存のコミットを修正する場合
git commit --amend
```

#### 方法3: 環境変数の設定

PowerShellで以下の環境変数を設定することで、UTF-8エンコーディングを強制できます：

```powershell
$env:LANG = "ja_JP.UTF-8"
$env:LC_ALL = "ja_JP.UTF-8"
```

### コミットメッセージの確認

コミット後に文字化けしていないか確認：

```bash
git log -1 --pretty=format:"%s%n%b"
```

文字化けが発生している場合は、`git commit --amend`で修正してください。

## プロジェクト構造

- `docker/app/`: アプリケーションのメインコード
  - `core/`: エントリーポイント（main.py, train.py, predict.py）
  - `data/`: データ処理とトークナイザー
  - `models/`: Transformerモデルの実装
  - `utils/`: ユーティリティ関数と設定

## コーディング規約

- PythonコードはPEP 8に準拠
- 型ヒントの使用を推奨
- ドキュメント文字列（docstring）の記述を推奨

## テスト

```bash
# コンテナ内でテストを実行
pytest
```

## その他の注意事項

- Dockerコンテナ内で実行することを前提としています
- GPU環境が必要です（NVIDIA GPU + NVIDIA Container Toolkit）
- Weights & Biases（WandB）のAPIキーが必要な場合があります
