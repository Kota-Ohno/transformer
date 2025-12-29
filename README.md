# Transformer 翻訳モデル

Transformerアーキテクチャを使用した日英・英日翻訳システムの実装です。

## 概要

このプロジェクトは、"Attention Is All You Need"で提案されたTransformerモデルを使用して、日本語と英語間の機械翻訳システムを構築します。主な特徴は以下の通りです:

- SentencePieceによるサブワードトークナイザー
- モジュール化されたTransformerアーキテクチャ（エンコーダー、デコーダー、マルチヘッドアテンション）
- 効率的なトレーニング管理のための`Trainer`クラス
- 混合精度トレーニングでのメモリ効率化
- データ拡張機能（トークンマスキング、削除、置換、順序入れ替え）
- SacreBLEUによる標準化された評価
- チェックポイント機能とトレーニング再開機能
- Dockerコンテナによる簡単なセットアップと実行
- GPUメモリに基づくモデルサイズの自動調整

## 必要条件

- Docker と Docker Compose
- NVIDIA GPU と NVIDIA Container Toolkit
- Weights & Biases アカウント (オプション、トレーニング追跡用)

## セットアップと実行

### 1. 環境設定

`.env`ファイルにWandBのAPIキーを設定します:

```
WANDB_API_KEY=your_wandb_api_key_here
```

### 2. ビルドと起動

```bash
docker compose build --no-cache
docker compose up -d
docker exec -it transformer /bin/sh
```

### 3. データの準備とトレーニング

#### トレーニングデータの配置

トレーニングデータは、コンテナ内の `/src` ディレクトリ（プロジェクトの `docker/app` ディレクトリにマウント）に配置してください。

**データ配置パス:**
- コンテナ内: `/src/` または `/src/data/`
- ホスト側: `docker/app/` または `docker/app/data/`

**データ形式:**

このプロジェクトは、JSONL形式（JSON Lines）のデータを想定しています。各ファイルは1行に1つの翻訳ペアを含み、以下の形式で記述してください:

```json
{"japanese": "こんにちは", "english": "Hello"}
{"japanese": "ありがとう", "english": "Thank you"}
```

**必須フィールド:**
- `japanese`: 日本語の文（ソース言語）
- `english`: 英語の文（ターゲット言語）

**ファイル名の例:**
- `train.jsonl` - トレーニングデータ
- `val.jsonl` または `dev.jsonl` - 検証データ

**データの準備方法:**

1. **既存データセットを使用する場合:**
   - データセットをダウンロードし、上記のJSONL形式に変換してください
   - ファイルを `docker/app/` または `docker/app/data/` ディレクトリに配置

2. **データをコンテナにコピーする場合:**
   ```bash
   # ホスト側から実行
   docker cp /path/to/your/train.jsonl transformer:/src/train.jsonl
   docker cp /path/to/your/val.jsonl transformer:/src/val.jsonl
   ```

3. **ボリュームマウントを使用する場合:**
   `docker-compose.yml` の `volumes` セクションに以下を追加することで、データディレクトリをマウントできます:
   ```yaml
   volumes:
     - ./app:/src/
     - ./data:/src/data  # データディレクトリをマウント
   ```

**ディスク容量の目安:**
- **サンプルデータセット（開発用）**: 約100MB〜500MB（数千〜数万文対）
- **中規模データセット**: 約1GB〜5GB（10万〜50万文対）
- **大規模データセット**: 約10GB以上（100万文対以上）

**注意:** `text_tokenizer.py` を実行すると、元のJSONLファイルからトークナイズ済みデータ（`tokenized_train_data.pth`、`tokenized_val_data.pth`）が生成されます。これらのファイルは元のデータよりも大きくなる場合があります（通常は1.5〜2倍程度）。

#### トレーニングの実行

コンテナ内で以下のコマンドを実行します:

#### 基本的な使用方法

```bash
# データのトークナイズと前処理
# SentencePieceモデルのトレーニングとトークン化を実行します。
# --augment オプションでデータ拡張を有効にできます。
python text_tokenizer.py

# モデルのトレーニングを開始
# main.py は、GPU環境の自動検出や高速モードなどの最適化設定を適用します。
# より詳細な設定は train.py を直接使用してください。
python main.py

# 高速に効率的にトレーニングする場合 (推奨)
python main.py --fast

# 翻訳の推論
python predict.py
```

#### データ拡張の使用

データ拡張は `text_tokenizer.py` でデータを準備する際に適用されます。

```bash
# データ拡張を有効にしてトークナイズ (デフォルトの拡張率 0.3)
python text_tokenizer.py --augment

# データ拡張の割合を指定 (例: 元データの40%を拡張)
python text_tokenizer.py --augment --augment-factor 0.4

# サンプルサイズを指定 (開発用、本番環境では0で全データ使用)
python text_tokenizer.py --sample-size 5000
```

#### トレーニングオプション

`main.py` または `train.py` を実行する際に利用可能な主なオプション:

```bash
# main.py を使用した高速トレーニングモード
python main.py --fast

# main.py を使用した小さいモデルでのトレーニング
python main.py --small-model

# main.py を使用したサンプル数制限付きトレーニング (開発用)
python main.py --limit-samples 1000

# train.py を使用したデータ拡張を有効にしたトレーニング
python train.py --augment --augment-factor 0.3

# 最新のチェックポイントからトレーニングを再開
python train.py --resume

# 特定のチェックポイントからトレーニングを再開
python train.py --checkpoint models/checkpoints/checkpoint_epoch_10_20230401.pth

# エポック数とバッチサイズを指定
python train.py --epochs 20 --batch-size 32

# Weights & Biases のログを無効にする
python train.py --no-wandb

# 勾配蓄積ステップ数を指定
python train.py --grad-accum-steps 4

# PyTorch JIT コンパイルを有効にする
python train.py --jit
```

### 4. 対話型ヘルプの使用

各スクリプトのオプションや使用方法についての詳細情報を得るには、対話型ヘルプスクリプトを使用できます：

```bash
# 対話型メニューを表示
python help.py

# 特定のトピックのヘルプを直接表示
python help.py tokenizer  # text_tokenizer.pyのヘルプ
python help.py train      # train.pyのヘルプ
python help.py predict    # predict.pyのヘルプ
python help.py augment    # データ拡張機能のヘルプ
python help.py all        # すべてのヘルプを表示
```

## モデル構成

モデルのハイパーパラメータは `utils/config.py` の `ModelHyperparameters` クラスで定義されています。デフォルト設定は以下の通りです:

- 隠れ層次元 (`hidden_size`): 512
- アテンションヘッド数 (`num_heads`): 8
- エンコーダー/デコーダー層数 (`num_layers`): 6
- Feed-forward次元 (`d_ff`): 2048
- ドロップアウト率 (`dropout_rate`): 0.1
- 最大シーケンス長 (`max_seq_length`): 512

### GPUメモリに基づく自動調整

GPUメモリの利用可能性に応じて、モデルの層数、隠れ層次元、およびヘッド数が自動的に調整されます。これにより、様々なGPU環境で最適なモデルサイズが選択されます。

- **16GB以上**: 6レイヤー (hidden=512, heads=8)
- **8GB以上16GB未満**: 4レイヤー (hidden=512, heads=8)
- **4GB以上8GB未満**: 4レイヤー (hidden=384, heads=6)
- **4GB未満**: 3レイヤー (hidden=256, heads=4)

これらの設定は `utils/config.py` の `ModelConfig.from_gpu_memory()` メソッドによって決定されます。

### 重要な注意事項

GPUメモリに基づく自動調整機能により、学習時にモデルの構成（層数など）が変わる場合があります。モデルのロードエラーが発生した場合は、保存されたモデルのチェックポイントファイルに含まれる設定 (`model_config`) を確認してください。

## モデルアーキテクチャ

標準モデルは原論文「Attention Is All You Need」に基づいて実装されており、絶対位置エンコーディング (`models/layers.py:PositionalEncoding`) と標準的なフィードフォワードネットワーク (`models/layers.py:FeedForward`) を使用します。

## データ拡張技術

このプロジェクトでは、モデルの汎化性能を向上させるために以下のデータ拡張技術を実装しています (`data/data_augmentation.py`)：

1.  **トークンマスキング**: ランダムに選択したトークンを`<unk>`トークンに置き換えます
2.  **トークン削除**: ランダムにトークンを削除します
3.  **トークン置換**: ランダムにトークンを別のトークンに置き換えます
4.  **トークン順序入れ替え**: 局所的な窓内でトークンの順序をランダムに入れ替えます

これらの拡張は `text_tokenizer.py` でデータを準備する際に `--augment` オプションを有効にすることで適用されます。拡張の度合いは `--augment-factor` オプションで調整できます（デフォルトは元データの30%）。

## チェックポイントと再開機能

トレーニング中、以下のタイミングでチェックポイントが自動的に `models/checkpoints/` ディレクトリに保存されます：

1.  各エポックの終了時 (`checkpoint_epoch_*.pth`)
2.  検証損失またはBLEUスコアが改善した時 (`best_model_*.pth`)

トレーニングを中断した場合は、`train.py` 実行時に `--resume` オプションを使用して最新のチェックポイントから再開できます。特定のチェックポイントから再開する場合は、`--checkpoint PATH/TO/CHECKPOINT.pth` オプションを使用します。

## 分散トレーニング

複数のGPUを活用するには、`torch.distributed.launch` を使用して `train.py` を実行します:

```bash
# 例: 4GPUでの分散トレーニング
WORLD_SIZE=4 python -m torch.distributed.launch --nproc_per_node=4 train.py [その他のオプション]
```

## エラー解決

-   **NVIDIA Container Toolkit の問題:** NVIDIAの公式ドキュメントを参照してください: [https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
-   **モデル読み込みエラー:** 学習時と推論時でモデル設定が一致していることを確認してください。GPUメモリによる自動調整で層数が変わっている可能性もあります。
-   **その他のエラー:** 詳細なエラーログがコンソールに出力されます。データの読み込みに関する一部のエラーは自動的にリトライされ、バッチ処理中のエラーはスキップされて処理が続行される場合があります。

## パフォーマンス向上のヒント

1.  **データ量:** より多くの翻訳データを使用する (数十万文対以上を推奨)。
2.  **データ拡張:** `--augment` オプションを活用する。
3.  **バッチサイズ:** GPUメモリが許す限り大きくする。
4.  **トレーニング時間:** より多くのエポック数で学習する (デフォルトの早期停止条件は比較的緩やか)。
5.  **モデルサイズ:** `utils/config.py` で `hidden_size`, `num_layers`, `d_ff`, `num_heads` を調整する（GPUメモリに注意）。

## トークン化戦略

このプロジェクトでは、SentencePiece (`sentencepiece` ライブラリ) を使用してサブワードトークン化を行います。日本語と英語には別々のモデルが作成され (`models/sp_src.model`, `models/sp_tgt.model`)、語彙情報は `models/vocab_input.pth`, `models/vocab_output.pth` にも保存されます。トークン化のロジックは `data/tokenizer_utils.py` に集約されています。

## `main.py` と `train.py` の役割

-   **`main.py`**: このスクリプトは、トレーニングプロセスを開始するための高レベルなエントリポイントです。GPU環境の自動検出、モデルサイズの自動調整、高速モードなどの最適化設定を自動的に適用します。新規ユーザーや効率的なトレーニングを迅速に開始したい場合に推奨されます。内部的には `train.py` を呼び出します。
-   **`train.py`**: このスクリプトは、Transformerモデルのトレーニングを実行するコアロジックを含んでいます。`main.py` から呼び出されるか、より詳細なパラメータチューニングやカスタム設定が必要な上級ユーザーによって直接実行されます。トレーニングループ、評価、チェックポイント管理は `utils/trainer.py` の `Trainer` クラスによってカプセル化されています。

## 推奨トレーニング設定

最適な学習結果を得るための推奨コマンド設定例:

```bash
# 1. 前処理：データ拡張を使用してトークナイズ (必要に応じてサンプルサイズも指定)
python text_tokenizer.py --augment --augment-factor 0.4 [--sample-size N]

# 2. 高速に効率的に学習する場合 (推奨)
python main.py --fast

# 3. より詳細に設定して学習する場合
python train.py --batch-size 64 --epochs 30 --warmup-steps 4000 --augment --augment-factor 0.4

# 4. 長期トレーニング中断時の再開
python train.py --batch-size 64 --resume

# 5. 最終モデルでの推論
python predict.py
```

上記の設定は一般的な環境での推奨値です。お使いのGPUメモリに合わせてバッチサイズを調整してください。より大きなバッチサイズはより安定した学習につながります。

# Transformerモデルトレーニング高速化ガイド

このプロジェクトは日本語から英語への翻訳を行うTransformerモデルの実装です。以下に学習実行時間を短縮するための設定と改善点を説明します。

## 主な改善点

学習の実行時間短縮のために以下の最適化が適用されました：

1.  **データ処理の効率化**
    - データローダーのバッファリング強化
    - バッチ処理の最適化
    - シーケンス長制限の緩和

2.  **メモリ使用量の最適化**
    - 勾配蓄積ステップ数の増加
    - GPU PINメモリの事前割り当て
    - キャッシュクリア頻度の最適化

3.  **モデルサイズの調整**
    - よりコンパクトなモデル設定の追加
    - GPUメモリに基づく自動的なモデルサイズ調整

4.  **プログレス監視の効率化**
    - ログ出力頻度の削減
    - 進捗表示の更新頻度の最適化

## 使用方法

以下のコマンドで高速実行モードでトレーニングを開始できます：

```bash
python main.py --fast
```

さらに小さいモデルを使用する場合：

```bash
python main.py --fast --small-model
```

開発用にサンプル数を制限してテスト実行する場合：

```bash
python main.py --fast --limit-samples 1000
```

> **注意**: このプロジェクトはDockerコンテナ内で実行することを前提としています。
> コンテナ内では単に `python main.py --fast` というように実行できます。

## 主なコマンドラインオプション

-   `--fast`: 高速トレーニングモードを有効化（エポック数減少、勾配蓄積増加）
-   `--small-model`: 小さなモデルアーキテクチャを使用（hidden_size=256, heads=4, layers=4）
-   `--limit-samples N`: トレーニングデータをN個のサンプルに制限
-   `--no-nltk-download`: NLTKリソースのダウンロードをスキップ（既にダウンロード済みの場合）

## 構成設定

モデルサイズをさらにカスタマイズする場合は、`utils/config.py` を参照してください。環境変数によるオーバーライドも可能です。

```bash
export TRANSFORMER_MODEL_HIDDEN_SIZE=256
export TRANSFORMER_MODEL_NUM_HEADS=4
export TRANSFORMER_MODEL_NUM_LAYERS=4
export TRANSFORMER_TRAINING_GRADIENT_ACCUMULATION_STEPS=8
```
