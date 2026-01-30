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

#### 具体的な学習例

**試運転（1エポックだけ実行）:**
```bash
python core/train_mlm.py \
    --model-size small \
    --epochs 1 \
    --batch-size 16 \
    --save-dir models/test_run \
    --tensorboard-dir runs/test
```

**M4 Mac推奨設定（16GB RAM）:**
```bash
python core/train_mlm.py \
    --model-size base \
    --epochs 10 \
    --batch-size 8 \
    --lr 5e-5 \
    --label-smoothing 0.1 \
    --save-dir models/mlm_base \
    --tensorboard-dir runs/mlm_base \
    --patience 5
```

**高性能GPU環境:**
```bash
python core/train_mlm.py \
    --model-size large \
    --epochs 20 \
    --batch-size 64 \
    --lr 3e-5 \
    --label-smoothing 0.1 \
    --save-dir models/mlm_large \
    --tensorboard-dir runs/mlm_large
```

**学習再開（チェックポイントから）:**
```bash
python core/train_mlm.py \
    --model-size base \
    --resume \
    --save-dir models/mlm_base
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

#### 方法1: コンテナ内で起動（推奨）

```bash
# コンテナ内でTensorBoardを起動（バックグラウンドで実行可能）
tensorboard --logdir=runs --bind_all --port 6006

# 別ターミナル（ホスト側）でポートを確認
docker port transformer 6006

# ブラウザでアクセス
# http://localhost:6006
```

#### 方法2: SSHトンネル（リモートサーバーの場合）

```bash
# ローカルマシンからSSHトンネルを作成
ssh -L 6006:localhost:6006 user@remote-server

# ブラウザで http://localhost:6006 を開く
```

#### TensorBoardの主な機能

- **Scalars**: Loss、Perplexity、Accuracyの推移を確認
- **Graphs**: モデルの計算グラフを可視化
- **Histograms**: パラメータ分布の変化を追跡
- **HParams**: 異なるハイパーパラメータの比較

### 5. MLM推論

学習済みモデルを使用してマスクされたトークンを予測します。

#### 対話モード（インタラクティブ）

最も簡単な使用方法です。対話的に文章を入力して予測結果を確認できます。

```bash
python core/predict_mlm.py --interactive
```

**入力例:**
```
入力: The cat sat on the [MASK] and looked at the birds.

予測結果:
  1. [MASK] → 'mat'
     候補: mat(45.2%), bed(12.3%), floor(8.7%), ...

入力: I love to eat [MASK] for breakfast.

予測結果:
  1. [MASK] → 'pancakes'
     候補: pancakes(32.1%), eggs(18.5%), cereal(15.2%), ...
```

**終了方法:** `quit` または `exit` と入力、または Ctrl+C
# I love to eat [MASK] for breakfast.
```

#### 単一テキスト

特定の文章だけを予測したい場合に使用します。

```bash
# 基本的な使用法
python core/predict_mlm.py \
    --text "The cat sat on the [MASK] and looked at the birds."

# 学習済みモデルを指定する場合
python core/predict_mlm.py \
    --text "The cat sat on the [MASK]." \
    --model-path models/mlm_base \
    --top-k 10

# 出力例:
# 入力: The cat sat on the [MASK] and looked at the birds.
# 
# 予測 1:
#   最確: 'mat'
#   候補: mat(45.2%), bed(12.3%), floor(8.7%), chair(6.5%), sofa(5.2%)
```

#### ファイルからバッチ処理

複数の文章を一括で処理したい場合に使用します。

```bash
# 1. 予測したいテキストを1行ごとに記載したファイルを作成
cat > input.txt << 'EOF'
The cat sat on the [MASK] and looked at the birds.
I love [MASK] pizza.
The [MASK] is shining brightly today.
She works as a [MASK] at the hospital.
EOF

# 2. バッチ処理を実行
python core/predict_mlm.py \
    --file input.txt \
    --output results.json \
    --model-path models/mlm_base

# 3. 結果を確認
cat results.json
```

**入力ファイルの形式:**
- 1行に1つの文章
- 必ず1つ以上の `[MASK]` を含める
- 空行は無視されます

**出力ファイル (JSON形式):**
```json
[
  {
    "text": "The cat sat on the [MASK] and looked at the birds.",
    "predictions": [
      {
        "position": 5,
        "predicted": "mat",
        "candidates": [
          {"token": "mat", "probability": 0.452},
          {"token": "bed", "probability": 0.123}
        ]
      }
    ]
  }
]
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
# 1. ログを確認
docker logs transformer

# 2. コンテナの状態を確認
docker ps -a | grep transformer

# 3. クリーンアップして再起動
docker compose down -v
docker rm -f transformer 2>/dev/null
docker compose up -d

# 4. それでも起動しない場合は、イメージを再ビルド
docker compose down
docker compose build --no-cache
docker compose up -d
```

### 学習が遅い / GPUが使われていない

**M1/M2/M4 Macの場合:**
- Docker DesktopでRosettaは無効にしてください
- ネイティブARM64イメージを使用（Dockerfile.arm64）

**確認コマンド:**
```bash
# PyTorchのデバイス確認
python -c "import torch; print(f'Device: {torch.device}')"

# CPUのみの場合は、これが表示されます
# Device: cpu
```

**注意:** M1/M2/M4 MacではGPU（Metal）をPyTorchで使用するのは現在難しいため、CPU学習が推奨されます。M4 Macの高性能CPUで十分速く学習できます。

### データセットのダウンロードが失敗する

```bash
# キャッシュをクリアして再試行
rm -rf ~/.cache/huggingface/datasets
python core/train_mlm.py --cache-dir /tmp/datasets
```

### 予測結果がおかしい / 精度が低い

1. **学習が十分か確認**
   - 最低5エポック以上の学習が必要
   - Perplexityが10以下になるまで学習

2. **モデルパスを確認**
   ```bash
   # 正しいモデルパスを指定
   python core/predict_mlm.py --model-path models/mlm_base
   ```

3. **入力に[MASK]が含まれているか確認**
   ```bash
   # 正しい例
   echo "The cat sat on the [MASK]."
   
   # 誤った例（MASKがない）
   echo "The cat sat on the mat."
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
