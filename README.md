# Transformer MLM（Masked Language Modeling）

Transformerアーキテクチャを使用したマスクド言語モデリング（MLM）の実装です。BERTと同じアーキテクチャで、WikiText-2データセットを使用して学習します。

## 概要

このプロジェクトは、"Attention Is All You Need"で提案されたTransformer Encoderを使用したマスクド言語モデリング（MLM）システムです。主な特徴は以下の通りです:

- **BERTベースアーキテクチャ**: Transformer Encoder + MLM Head
- **事前学習済みTokenizer**: Hugging Face `bert-base-cased` トークナイザー
- **動的マスキング**: BERT標準のマスキング戦略（15%マスク、80%[MASK]/10%ランダム/10%保持）
- **ラベルスムージング**: 過学習防止のためε=0.1のラベルスムージングを適用
- **TensorBoard統合**: 学習曲線のリアルタイム可視化
- **複数評価指標**: Loss、Perplexity、Masked Token Accuracy
- **Dockerコンテナ**: 簡単なセットアップと実行環境の再現性
- **M1/M2/M4 Mac対応**: Apple Siliconネイティブサポート

## 必要条件

- Docker と Docker Compose
- Mac: Apple Silicon (M1/M2/M4) または Intel Mac
- Linux: x86_64 アーキテクチャ
- Windows: WSL2推奨

**注意**: GPUはオプションです。CPUのみでも学習可能です。

## プロジェクト構造

```
docker/app/
├── core/
│   ├── train_mlm.py      # MLM学習スクリプト
│   └── predict_mlm.py    # MLM推論スクリプト
├── models/
│   ├── mlm_model.py      # MLMモデル実装
│   ├── encoder.py        # Transformer Encoder
│   ├── attention.py      # Multi-Head Attention
│   ├── layers.py         # Positional Encoding, FeedForward
│   └── loss.py           # 損失関数（Label Smoothing対応）
├── data/
│   ├── mlm_dataset.py    # WikiText-2データセット処理
│   └── tokenizer_utils.py # テキスト前処理ユーティリティ
├── utils/
│   ├── masking.py        # BERT標準マスキング実装
│   └── config.py         # 設定管理
└── tests/
    └── test_mlm.py       # MLMコンポーネントのテスト
```

## セットアップと実行

### 1. 環境設定

```bash
# リポジトリをクローン
cd transformer/docker

# Dockerイメージをビルド（初回のみ）
docker compose build

# コンテナを起動
docker compose up -d

# コンテナに入る
docker exec -it transformer bash
```

### 2. 動作確認

```bash
cd /src
python test_imports.py
```

全てのチェックマーク（✓）が表示されれば成功です。

### 3. MLM学習

#### 基本的な学習

```bash
python core/train_mlm.py --model-size small --epochs 10 --batch-size 32
```

#### 学習オプション

```bash
# モデルサイズの指定
python core/train_mlm.py --model-size base    # 768dim, 12layers（推奨）
python core/train_mlm.py --model-size small   # 256dim, 4layers（高速）
python core/train_mlm.py --model-size large   # 1024dim, 24layers（高精度）

# 学習率とエポック数の調整
python core/train_mlm.py --lr 3e-5 --epochs 20 --batch-size 16

# ラベルスムージングの調整
python core/train_mlm.py --label-smoothing 0.1

# TensorBoardログの保存場所を変更
python core/train_mlm.py --tensorboard-dir runs/my_experiment
```

#### 学習パラメータ一覧

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `--model-size` | base | モデルサイズ（small/base/large） |
| `--epochs` | 10 | 学習エポック数 |
| `--batch-size` | 32 | バッチサイズ |
| `--lr` | 5e-5 | 学習率 |
| `--weight-decay` | 0.01 | 重み減衰 |
| `--label-smoothing` | 0.1 | ラベルスムージング係数 |
| `--max-seq-length` | 512 | 最大シーケンス長 |
| `--patience` | 5 | 早期終了の忍耐エポック数 |

### 4. TensorBoardで学習状況を確認

```bash
# コンテナ内でTensorBoardを起動
tensorboard --logdir=runs/mlm_training --bind_all

# 別ターミナルからポートフォワード
docker port transformer 6006
# ブラウザで http://localhost:6006 を開く
```

### 5. MLM推論

#### 対話モード

```bash
python core/predict_mlm.py --interactive

# 入力例:
# The cat sat on the [MASK] and looked at the birds.
# I love to eat [MASK] for breakfast.
```

#### 単一テキスト

```bash
python core/predict_mlm.py --text "The cat sat on the [MASK] and looked at the birds."
```

#### ファイルからバッチ処理

```bash
# 予測したいテキストを1行ごとに記載
echo "The cat sat on the [MASK]." > input.txt
echo "I love [MASK] pizza." >> input.txt

# 実行
python core/predict_mlm.py --file input.txt --output results.json
```

## モデルアーキテクチャ

### BERT Base（デフォルト）

```
隠れ層次元: 768
アテンションヘッド数: 12
エンコーダー層数: 12
Feed-forward次元: 3072
最大シーケンス長: 512
ドロップアウト率: 0.1
パラメータ数: ~110M
```

### マスキング戦略

BERT標準のマスキング戦略を採用:

1. **15%のトークンを選択**
2. **選択されたトークンの80%を[MASK]に置き換え**
3. **10%をランダムなトークンに置き換え**
4. **10%はそのまま保持**

例:
```
入力: The cat sat on the mat and looked at the birds.
マスク: The cat sat on the [MASK] and looked at the birds.
ラベル: mat
```

## 評価指標

学習中に以下の指標が記録されます:

- **Loss**: クロスエントロピー損失
- **Perplexity**: 次のトークン予測の不確実性（低いほど良い）
- **Masked Token Accuracy**: マスクされたトークンの予測精度

## 推奨学習設定

### M4 Mac（16GB RAM）

```bash
python core/train_mlm.py \
    --model-size base \
    --epochs 10 \
    --batch-size 16 \
    --lr 5e-5 \
    --label-smoothing 0.1
```

学習時間: 約6〜12時間

### 高性能GPU環境

```bash
python core/train_mlm.py \
    --model-size large \
    --epochs 20 \
    --batch-size 64 \
    --lr 3e-5
```

## トラブルシューティング

### インポートエラー

```bash
# __init__.pyが正しく配置されているか確認
ls -la /src/models/__init__.py
ls -la /src/utils/__init__.py
ls -la /src/data/__init__.py
```

### メモリ不足エラー

```bash
# バッチサイズを減らす
python core/train_mlm.py --batch-size 8

# または小さいモデルを使用
python core/train_mlm.py --model-size small
```

### Dockerコンテナが起動しない

```bash
# ログを確認
docker logs transformer

# コンテナを再起動
docker compose down && docker compose up -d
```

## 開発者向け情報

### テストの実行

```bash
cd /src
python -m pytest tests/test_mlm.py -v
```

### コードスタイル

- PEP 8準拠
- 型ヒント必須
- Googleスタイルのdocstring

### 依存パッケージ

主要な依存パッケージ:
- PyTorch >= 2.0.0
- Transformers >= 4.30.0
- Datasets >= 2.14.5
- TensorBoard >= 2.20.0

## ライセンス

MIT License

## 参考文献

1. Vaswani et al. "Attention Is All You Need" (NeurIPS 2017)
2. Devlin et al. "BERT: Pre-training of Deep Bidirectional Transformers" (NAACL 2019)
3. Merity et al. "Pointer Sentinel Mixture Models" (ICLR 2017) - WikiText-2

## コントリビューション

バグ報告や機能追加のPRを歓迎します。

---

**注意**: このプロジェクトは学習目的の実装です。本番環境での使用には追加の最適化が必要です。
