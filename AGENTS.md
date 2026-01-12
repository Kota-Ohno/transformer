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

#### 方法3: Git設定を使用（Windows推奨）

Windowsでは、環境変数`LANG`や`LC_ALL`の設定が期待通りに動作しない場合があります。代わりに、Gitの設定を直接変更することを推奨します：

```bash
# GitのコミットエンコーディングをUTF-8に設定
git config --global i18n.commitencoding utf-8
```

**注意**: PowerShellで`$env:LANG`や`$env:LC_ALL`を設定する方法は、Windowsのロケールシステムが異なるため、期待通りに動作しない場合があります。Windows環境では、方法1（一時ファイルを使用）または方法2（Gitエディタを使用）を優先的に使用することを推奨します。

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

### 基本原則

- Python 3.10以上が必須（mypy設定（`python_version = 3.10`）と一致させるため）
- PythonコードはPEP 8に準拠
- 型ヒントは必須（関数の引数、戻り値、クラスの属性など）
- ドキュメント文字列（docstring）は必須

### 型ヒント

すべての関数、メソッド、クラス属性には型ヒントを記述してください。型チェックには`mypy`を使用します。

#### mypyの実行

```bash
# プロジェクト全体をチェック
mypy docker/app

# 特定のファイルをチェック
mypy docker/app/models/model.py

# 厳密モードでチェック
mypy --strict docker/app
```

#### mypy設定（mypy.ini）

プロジェクトルートに`mypy.ini`を作成し、以下の設定を推奨します：

```ini
[mypy]
python_version = 3.10
warn_return_any = True
warn_unused_configs = True
disallow_untyped_defs = True
disallow_incomplete_defs = True
check_untyped_defs = True
disallow_untyped_decorators = True
no_implicit_optional = True
warn_redundant_casts = True
warn_unused_ignores = True
warn_no_return = True
warn_unreachable = True
strict_equality = True

# 無視するモジュール（必要に応じて調整）
[mypy-torch.*]
ignore_missing_imports = True

[mypy-numpy.*]
ignore_missing_imports = True
```

### Docstring形式

Googleスタイルのdocstringを使用してください。以下のセクションを含めることを推奨します：

- **Args**: 引数の説明（型と説明）
- **Returns**: 戻り値の説明（型と説明）
- **Raises**: 発生する可能性のある例外
- **Examples**: 使用例（オプションだが推奨）

### 具体例

以下は、PEP 8に準拠し、型ヒントとGoogleスタイルのdocstringを含む関数の例です：

```python
from typing import List, Optional
import torch
from torch import Tensor


def calculate_loss(
    predictions: Tensor,
    targets: Tensor,
    reduction: str = "mean",
    ignore_index: Optional[int] = None,
) -> Tensor:
    """損失を計算する関数。

    Args:
        predictions: モデルの予測値。形状は (batch_size, seq_len, vocab_size)。
        targets: 正解ラベル。形状は (batch_size, seq_len)。
        reduction: 損失の縮約方法。'mean'、'sum'、'none'のいずれか。デフォルトは'mean'。
        ignore_index: 無視するインデックス。Noneの場合はすべてのインデックスを考慮。

    Returns:
        計算された損失値。reductionが'mean'または'sum'の場合はスカラー、
        'none'の場合は各サンプルの損失を含むテンソル。

    Raises:
        ValueError: reductionが'mean'、'sum'、'none'のいずれでもない場合。
        RuntimeError: predictionsとtargetsの形状が一致しない場合。

    Examples:
        >>> pred = torch.randn(32, 100, 5000)
        >>> tgt = torch.randint(0, 5000, (32, 100))
        >>> loss = calculate_loss(pred, tgt)
        >>> print(loss.item())
    """
    if reduction not in ["mean", "sum", "none"]:
        raise ValueError(f"Invalid reduction: {reduction}")

    # 形状の検証
    if predictions.dim() != 3 or targets.dim() != 2:
        raise RuntimeError(
            f"Invalid tensor shapes: predictions must be 3D (batch_size, seq_len, vocab_size), "
            f"targets must be 2D (batch_size, seq_len). "
            f"Got predictions.shape={predictions.shape}, targets.shape={targets.shape}"
        )
    if predictions.shape[:2] != targets.shape:
        raise RuntimeError(
            f"Shape mismatch: predictions.shape[:2]={predictions.shape[:2]} "
            f"does not match targets.shape={targets.shape}"
        )

    # テンソルをフラット化: (batch_size, seq_len, vocab_size) -> (N, vocab_size)
    # および (batch_size, seq_len) -> (N,)
    batch_size, seq_len, vocab_size = predictions.shape
    predictions_flat = predictions.view(-1, vocab_size)
    targets_flat = targets.view(-1)

    # クロスエントロピー損失を計算
    import torch.nn.functional as F
    loss = F.cross_entropy(
        predictions_flat,
        targets_flat,
        ignore_index=ignore_index,
        reduction=reduction
    )

    return loss
```

### リンティングとフォーマット

コードの品質を保つため、以下のツールを使用してください：

- **flake8**: コードスタイルとエラーのチェック
- **black**: コードフォーマッター（PEP 8準拠）

#### 実行方法

```bash
# flake8でチェック
flake8 docker/app

# blackでフォーマット（変更を適用）
black docker/app

# blackでフォーマット（変更をプレビューのみ）
black --check docker/app
```

### CI/CDでの実行

CIパイプラインでは、以下のコマンドを順に実行してください：

```bash
# 1. コードフォーマットのチェック
black --check docker/app

# 2. リンティング
flake8 docker/app

# 3. 型チェック
mypy docker/app

# 4. テストの実行
pytest
```

## テスト

### pytest設定

プロジェクトでは`pytest.ini`を使用してpytestの設定を行っています。主な設定は以下の通りです：

- **テストファイルパターン**: `tests.py`, `test_*.py`, `*_test.py`
- **Pythonパス**: `.`（カレントディレクトリ）

以下は`pytest.ini`の設定例です：

```ini
[pytest]
python_files = tests.py test_*.py *_test.py
pythonpath = .
addopts = -q --maxfail=1
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    integration: marks tests as integration tests
```

### テストの実行方法

#### ローカル環境での実行

```bash
# プロジェクトルートから実行
pytest

# 特定のテストファイルを実行（Pattern A: モジュールと同一ディレクトリ）
pytest docker/app/models/test_encoder.py

# 特定のテストファイルを実行（Pattern B: 専用のtestsディレクトリ）
pytest docker/app/tests/test_example.py

# カバレッジ付きで実行（オプション、Pattern Aの例）
pytest --cov=docker/app --cov-report=html docker/app/models/test_encoder.py
```

#### Dockerコンテナ内での実行

```bash
# docker-composeを使用する場合
docker-compose exec app pytest

# docker runを使用する場合
docker run --rm \
  --gpus all \
  -v $(pwd)/docker/app:/src \
  -w /src \
  transformer-app pytest
```

#### テストの組織化

テストファイルの配置には以下の2つのパターンがあります：

**Pattern A: モジュールと同一ディレクトリに配置（推奨）**
- テストファイルを対象モジュールと同じディレクトリに配置します
- 例: `docker/app/models/test_*.py`、`docker/app/data/test_*.py`、`docker/app/models/*_test.py`、`docker/app/data/*_test.py`

**Pattern B: 専用のtestsディレクトリに配置**
- テストファイルを`docker/app/tests/`ディレクトリに集約します
- 例: `docker/app/tests/test_*.py`、`docker/app/tests/*_test.py`

**推奨**: Pattern A（モジュールと同一ディレクトリ）を推奨します。テストと実装コードが近くに配置されるため、保守性が向上します。

- テストファイル名は`test_*.py`または`*_test.py`の形式にしてください
- フィクスチャは`conftest.py`に定義（pytestが自動的に検出）
- モックライブラリとして`unittest.mock`を使用（標準ライブラリ）

### カバレッジ

カバレッジはオプションです。実行する場合は：

```bash
pytest --cov=docker/app --cov-report=term-missing
```

## その他の注意事項

- Dockerコンテナ内で実行することを前提としています
- GPU環境が必要です（NVIDIA GPU + NVIDIA Container Toolkit）
  - Docker実行時は`--gpus all`フラグが必要
  - NVIDIA Container Toolkitのインストールが必要
- Weights & Biases（WandB）のAPIキーが必要な場合があります
  - 環境変数`WANDB_API_KEY`で設定（例: `export WANDB_API_KEY=your_api_key_here`）
  - WandBを使用する機能（ロギング、実験管理など）を利用する場合に必要
- Docker統合の詳細
  - 環境変数: `TRANSFORMER_*`形式の環境変数で設定をオーバーライド可能
  - ボリュームマウント: `docker/app`ディレクトリを`/src`にマウント
  - GPUフラグ: `--gpus all`でGPUアクセスを有効化
