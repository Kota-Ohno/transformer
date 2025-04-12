# Transformer 翻訳モデル

Transformerアーキテクチャを使用した日英・英日翻訳システムの実装です。

## 概要

このプロジェクトは、"Attention Is All You Need"で提案されたTransformerモデルを使用して、日本語と英語間の機械翻訳システムを構築します。主な特徴は以下の通りです:

- SentencePieceによるサブワードトークナイザー
- マルチヘッドアテンションとエンコーダー・デコーダーアーキテクチャ
- **強化版モデルオプション:**
    - 相対位置エンコーディング (Relative Multi-Head Attention内で実装)
    - Gated Linear Units (GLU) を使用したフィードフォワードネットワーク
- 混合精度トレーニングでのメモリ効率化
- データ拡張機能（トークンマスキング、削除、置換、順序入れ替え）
- BLEUおよびSacreBLEUによる評価
- チェックポイント機能とトレーニング再開機能
- Dockerコンテナによる簡単なセットアップと実行

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

コンテナ内で以下のコマンドを実行します:

#### 基本的な使用方法

```bash
# データのトークナイズと前処理
python text_tokenizer.py

# モデルのトレーニング
python train.py

# 翻訳の推論 (ビームサーチ使用)
python predict.py
```

#### データ拡張の使用

```bash
# データ拡張を有効にしてトークナイズ
python text_tokenizer.py --augment --augment-factor 0.3

# サンプルサイズを指定（開発用）
python text_tokenizer.py --sample-size 5000 --augment
```

#### トレーニングオプション

```bash
# データ拡張を有効にしてトレーニング
python train.py --augment --augment-factor 0.3

# 強化版モデル（相対位置エンコーディング、GLU）を使用
python train.py --enhanced

# 相対位置エンコーディングの最大距離を指定 (強化版モデルのみ)
python train.py --enhanced --rel-pos-max-dist 128

# チェックポイントから再開
python train.py --resume

# 特定のチェックポイントから再開
python train.py --checkpoint models/checkpoints/checkpoint_epoch_10_20230401.pth

# エポック数とバッチサイズを指定
python train.py --epochs 20 --batch-size 32

# WandBログを無効にする
python train.py --no-wandb
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
python help.py enhanced   # 強化版モデルのヘルプ
python help.py all        # すべてのヘルプを表示
```

## モデル構成

デフォルト設定では以下のようなモデル構成になっています:

- 隠れ層次元: 512
- アテンションヘッド数: 8
- エンコーダー/デコーダー層数: 6 (GPUメモリに応じて自動調整、下記参照)
- Feed-forward次元: 2048
- ドロップアウト率: 0.1
- 相対位置エンコーディングの最大距離 (強化版モデル): 64 (config.pyで変更可能、train.py実行時にも指定可能)

これらのパラメータは`config.py`で変更可能です。

### GPUメモリに基づく自動調整

バッチサイズだけでなく、GPUメモリ制約に基づいてモデルのエンコーダー/デコーダー層数と、場合によっては隠れ層次元やヘッド数も自動的に調整されます。トレーニング開始時に利用可能なGPUメモリに応じて以下の設定が適用されます：

- 16GB以上: 6レイヤー (hidden=512, heads=8)
- 8GB以上16GB未満: 4レイヤー (hidden=512, heads=8)
- 4GB以上8GB未満: 4レイヤー (hidden=384, heads=6)
- 4GB未満: 3レイヤー (hidden=256, heads=4)

### 重要な注意事項

学習時と推論時のモデル設定（特に `--enhanced` オプションの使用有無、`--rel-pos-max-dist` の値）の一貫性が重要です。設定が一致しない場合、モデルの読み込みに失敗することがあります。
GPUメモリに基づく自動調整機能により、学習時に層数などが変わる場合があるため、モデルロードエラーが発生した場合は、保存されたモデルのチェックポイントファイルに含まれる設定 (`model_config`) を確認してください。

## モデルアーキテクチャ

### 標準モデル

標準モデルは原論文「Attention Is All You Need」に基づいて実装されており、絶対位置エンコーディング (`layers.py:PositionalEncoding`) と標準的なフィードフォワードネットワーク (`layers.py:FeedForwardNetwork`) を使用します。

### 強化版モデル (`--enhanced` オプション)

最新の研究結果に基づいて以下の改良を加えたモデルを使用できます。`train.py` または `predict.py` 実行時に `--enhanced` オプションを指定してください。

1.  **相対位置エンコーディング (Relative Positional Encoding, RPE):**
    *   絶対的な位置ではなく、トークン間の相対的な距離に基づいて位置情報をアテンション計算に組み込みます。
    *   `advanced_layers.py` の `RelativeMultiHeadAttention` クラス内で実装されています。これは Shaw et al. (2018) や Raffel et al. (2019) の研究に触発された実装です。
    *   長いシーケンスや可変長のシーケンスに対して、より頑健な位置表現を提供することが期待されます。
    *   考慮する最大相対距離は `--rel-pos-max-dist` オプション (デフォルト: 64) で指定できます。

2.  **Gated Linear Units (GLU):**
    *   標準的なフィードフォワードネットワーク (FFN) の代わりに、ゲート機構を持つ `GatedLinearUnit` (`advanced_layers.py`) を使用します。
    *   これは Dauphin et al. (2017) の研究に基づくもので、ネットワーク内の情報の流れをより効果的に制御し、重要な特徴を選択的に伝播させることを目的としています。
    *   `EnhancedFeedForward` クラス (`advanced_layers.py`) 内で利用されています。

## データ拡張技術

このプロジェクトでは、モデルの汎化性能を向上させるために以下のデータ拡張技術を実装しています (`data_augmentation.py`)：

1.  **トークンマスキング**: ランダムに選択したトークンを`<unk>`トークンに置き換えます
2.  **トークン削除**: ランダムにトークンを削除します
3.  **トークン置換**: ランダムにトークンを別のトークンに置き換えます
4.  **トークン順序入れ替え**: 局所的な窓内でトークンの順序をランダムに入れ替えます

これらの拡張は `text_tokenizer.py` または `train.py` で `--augment` オプションを有効にすることで適用されます。拡張の度合いは `--augment-factor` オプションで調整できます（デフォルトは元データの30%）。

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
-   **モデル読み込みエラー:** 学習時と推論時でモデル設定（`--enhanced`, `--rel-pos-max-dist` など）が一致しているか確認してください。GPUメモリによる自動調整で層数が変わっている可能性もあります。
-   **その他のエラー:** 詳細なエラーログがコンソールに出力されます。データの読み込みに関する一部のエラーは自動的にリトライされ、バッチ処理中のエラーはスキップされて処理が続行される場合があります。

## パフォーマンス向上のヒント

1.  **データ量:** より多くの翻訳データを使用する (数十万文対以上を推奨)。
2.  **データ拡張:** `--augment` オプションを活用する。
3.  **強化版モデル:** `--enhanced` オプションを使用する。
4.  **バッチサイズ:** GPUメモリが許す限り大きくする。
5.  **トレーニング時間:** より多くのエポック数で学習する (デフォルトの早期停止条件は比較的緩やか)。
6.  **モデルサイズ:** `config.py` で `HIDDEN_SIZE`, `NUM_LAYERS`, `D_FF`, `NUM_HEADS` を調整する（GPUメモリに注意）。

## トークン化戦略

このプロジェクトでは、SentencePiece (`sentencepiece` ライブラリ) を使用してサブワードトークン化を行います。日本語と英語には別々のモデルが作成され (`models/sp_src.model`, `models/sp_tgt.model`)、それぞれ約8,000の語彙サイズが割り当てられます。語彙情報は `models/vocab_input.pth`, `models/vocab_output.pth` にも保存されます。

## 参考文献

- Vaswani et al. (2017). "Attention Is All You Need"
- Shaw et al. (2018). "Self-Attention with Relative Position Representations"
- Raffel et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- Dauphin et al. (2017). "Language Modeling with Gated Convolutional Networks"
- Kudo, T., & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
- Wei, J., & Zou, K. (2019). "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"

## 推奨トレーニング設定

最適な学習結果を得るための推奨コマンド設定例:

```bash
# 1. 前処理：データ拡張を使用してトークナイズ (必要に応じてサンプルサイズも指定)
# python text_tokenizer.py --augment --augment-factor 0.4 [--sample-size N]

# 2. 学習：強化版モデルと最適なパラメータを使用 (バッチサイズはGPUメモリに応じて調整)
python train.py --enhanced --rel-pos-max-dist 128 --batch-size 64 --epochs 30 --warmup-steps 4000 --augment --augment-factor 0.4

# 3. 長期トレーニング中断時の再開
# python train.py --enhanced --rel-pos-max-dist 128 --batch-size 64 --resume

# 4. 最終モデルでの推論
python predict.py --enhanced
```

上記の設定は一般的な環境での推奨値です。お使いのGPUメモリに合わせてバッチサイズを調整してください。より大きなバッチサイズはより安定した学習につながります。
