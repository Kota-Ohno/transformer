# Transformer 翻訳モデル

Transformerアーキテクチャを使用した日英・英日翻訳システムの実装です。

## 概要

このプロジェクトは、"Attention Is All You Need"で提案されたTransformerモデルを使用して、日本語と英語間の機械翻訳システムを構築します。主な特徴は以下の通りです:

- SentencePieceによるサブワードトークナイザー
- マルチヘッドアテンションとエンコーダー・デコーダーアーキテクチャ
- 混合精度トレーニングでのメモリ効率化
- データ拡張機能（トークンマスキング、削除、置換、順序入れ替え）
- 相対位置エンコーディング（RPE）とGated Linear Units（GLU）
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

# 翻訳の推論
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

# 相対位置エンコーディングの最大距離を指定
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
python help.py all        # すべてのヘルプを表示
```

## モデル構成

デフォルト設定では以下のようなモデル構成になっています:

- 隠れ層次元: 512
- アテンションヘッド数: 8
- エンコーダー/デコーダー層数: 6
- Feed-forward次元: 2048
- ドロップアウト率: 0.1

これらのパラメータは`config.py`で変更可能です。

## モデルアーキテクチャ

### 標準モデル

標準モデルは原論文「Attention Is All You Need」に基づいて実装されており、絶対位置エンコーディングとFFNを使用します。

### 強化版モデル（--enhanced オプション）

最新の研究結果に基づいて以下の改良を加えたモデルを使用できます：

1. **相対位置エンコーディング（RPE）**:
   - 絶対位置ではなく、トークン間の相対的な位置関係をモデル化
   - 長いシーケンスや未知の長さのシーケンスにより効果的
   - Shaw et al. (2018)とRaffel et al. (2019)の手法に基づく実装

2. **Gated Linear Units (GLU)**:
   - フィードフォワードネットワークの代わりにゲート機構を導入
   - 情報の流れをより効果的に制御し、重要な特徴を強調
   - Dauphin et al. (2017)の論文に基づく実装

強化版モデルを使用するには、`--enhanced`オプションを指定してください。

```bash
python train.py --enhanced
```

## データ拡張技術

このプロジェクトでは、モデルの汎化性能を向上させるために以下のデータ拡張技術を実装しています：

1. **トークンマスキング**: ランダムに選択したトークンを`<unk>`トークンに置き換えます
2. **トークン削除**: ランダムにトークンを削除します
3. **トークン置換**: ランダムにトークンを別のトークンに置き換えます
4. **トークン順序入れ替え**: 局所的な窓内でトークンの順序をランダムに入れ替えます

デフォルト設定では、元のデータセットの30%に対してこれらの拡張技術を適用します。この比率は`--augment-factor`オプションで調整できます。

## チェックポイントと再開機能

トレーニング中、以下のタイミングでチェックポイントが自動的に保存されます：

1. 各エポックの終了時
2. 検証損失またはBLEUスコアが改善した時（最良モデルとして保存）

トレーニングを中断した場合は、`--resume`オプションを使用して最新のチェックポイントから再開できます。特定のチェックポイントから再開する場合は、`--checkpoint`オプションを使用します。

## 分散トレーニング

複数のGPUを活用するには、以下のように環境変数を設定します:

```bash
# 4GPUでの分散トレーニングの例
WORLD_SIZE=4 python -m torch.distributed.launch --nproc_per_node=4 train.py
```

## エラー解決

NVIDIA Container Toolkitの問題が発生した場合は、以下のドキュメントを参照してください:
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html

コードの実行中にエラーが発生した場合、詳細なエラーログが表示されます。データの読み込みに関するエラーは自動的にリトライされ、バッチ処理中のエラーはスキップされて処理が続行されます。

## パフォーマンス向上のヒント

1. データ量を増やす (少なくとも数十万文対を推奨)
2. データ拡張を活用する (`--augment`オプションを使用)
3. 強化版モデルを使用する (`--enhanced`オプションを使用)
4. バッチサイズを大きくする (GPUメモリに合わせて調整)
5. より長いトレーニング時間 (初期設定の早期停止パラメータは低め)
6. モデルサイズの調整 (より大きなハイパーパラメータを試す)

## トークン化戦略

このプロジェクトでは、SentencePieceを使用してサブワードトークン化を行います。日本語と英語には別々のモデルが作成され、それぞれ約8,000のボキャブラリサイズが割り当てられます。

## 参考文献

- Vaswani et al. (2017). "Attention Is All You Need"
- Shaw et al. (2018). "Self-Attention with Relative Position Representations"
- Raffel et al. (2019). "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer"
- Dauphin et al. (2017). "Language Modeling with Gated Convolutional Networks"
- Kudo, T., & Richardson, J. (2018). "SentencePiece: A simple and language independent subword tokenizer and detokenizer for Neural Text Processing"
- Wei, J., & Zou, K. (2019). "EDA: Easy Data Augmentation Techniques for Boosting Performance on Text Classification Tasks"

## 推奨トレーニング設定

最適な学習結果を得るための推奨コマンド設定:

```bash
# 1. 前処理：データ拡張を使用してトークナイズ
python text_tokenizer.py --augment --augment-factor 0.4

# 2. 学習：強化版モデルと最適なパラメータを使用
python train.py --enhanced --rel-pos-max-dist 128 --batch-size 64 --epochs 30 --warmup-steps 4000 --augment --augment-factor 0.4

# 3. 長期トレーニング中断時の再開
python train.py --enhanced --rel-pos-max-dist 128 --batch-size 64 --resume

# 4. 最終モデルでの推論
python predict.py --model models/best_model.pth --enhanced
```

上記の設定は一般的な環境での推奨値です。お使いのGPUメモリに合わせてバッチサイズを調整してください。より大きなバッチサイズはより安定した学習につながります。

このコマンド構成には以下の最適化が含まれています:
- データ拡張による学習データの多様化 (augment-factor 0.4)
- 強化版モデル（相対位置エンコーディング、GLU）の使用
- 適切な数のウォームアップステップ
- 十分なエポック数（早期停止機能付き）
- 相対位置エンコーディングの最大距離を最適化（128）
