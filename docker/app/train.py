import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math
from sklearn.model_selection import train_test_split
from datasets import load_dataset
from datetime import datetime

from data import set_data, build_vocabulary, tokenize, tokens_to_ids, create_data_loader
from config import TRANSLATION_SOURCE, TRANSLATION_DESTINATION, TRANSLATION_SOURCE2, TRANSLATION_DESTINATION2, WARMUP_STEPS, PATIENCE, BATCH_SIZE, LEARNING_RATE
from utils import validate, TranslationModel

# データとボキャブラリの読み込み
train_data = load_dataset("Amani27/massive_translation_dataset", split="train")
val_data = load_dataset("Amani27/massive_translation_dataset", split="validation")

train_dataset = set_data(train_data[TRANSLATION_SOURCE], train_data[TRANSLATION_DESTINATION])
val_dataset = set_data(val_data[TRANSLATION_SOURCE], val_data[TRANSLATION_DESTINATION])

train_dataset = train_dataset[:]
val_dataset = val_dataset[:]

# ソースとターゲットをペアにする
paired_train_dataset = list(zip(train_dataset[0], train_dataset[1]))
paired_val_dataset = list(zip(val_dataset[0], val_dataset[1]))

# 2つ目のデータセットを読み込む
train_data2 = load_dataset("Verah/JParaCrawl-Filtered-English-Japanese-Parallel-Corpus", split="train")

# model2_acceptedカラムが1のデータのみを抽出
filtered_train_data2 = train_data2.filter(lambda filter: filter['model2_accepted'] == 1)

# データセットを設定
train_dataset2 = set_data(filtered_train_data2[TRANSLATION_SOURCE2], filtered_train_data2[TRANSLATION_DESTINATION2])

# train_data2を学習用データとテストデータに分割
train_dataset2, val_dataset2 = train_test_split(train_dataset2, test_size=0.1, random_state=42)

# データセットを合体
paired_train_dataset += train_dataset2
paired_val_dataset += val_dataset2

# トークナイズ
print("======= tokenizing now =======")
tokenized_train_src = [tokenize(src, TRANSLATION_SOURCE) for src, _ in paired_train_dataset]
tokenized_val_src = [tokenize(src, TRANSLATION_SOURCE) for src, _ in paired_val_dataset]
print("======= 50% =======")
tokenized_train_tgt = [tokenize(tgt, TRANSLATION_DESTINATION) for _, tgt in paired_train_dataset]
tokenized_val_tgt = [tokenize(tgt, TRANSLATION_DESTINATION) for _, tgt in paired_val_dataset]
print("======= tokenizing finished ======")

# ボキャブラリの作成
print("======= build vocabulary now ======")
base_vocabulary = build_vocabulary(tokenized_train_src)
destination_vocabulary = build_vocabulary(tokenized_train_tgt)
print("======= build vocabulary finished ======")

# ボキャブラリの保存
print("======= save vocabulary now ======")
vocab_input_path = os.path.join("models", "vocab_input.pth")
torch.save(base_vocabulary, vocab_input_path)
vocab_output_path = os.path.join("models", "vocab_output.pth")
torch.save(destination_vocabulary, vocab_output_path)
print("======= save vocabulary finished ======")


# 単語IDに変換
print("======= translate tokens to ids ======")
train_dataset = [(tokens_to_ids(src, base_vocabulary), tokens_to_ids(tgt, destination_vocabulary)) for src, tgt in zip(tokenized_train_src, tokenized_train_tgt)]
val_dataset = [(tokens_to_ids(src, base_vocabulary), tokens_to_ids(tgt, destination_vocabulary)) for src, tgt in zip(tokenized_val_src, tokenized_val_tgt)]
print("======= translate finished ======")

# ハイパーパラメータの設定
from config import HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, DEVICE, NUM_EPOCHS
input_dim = len(base_vocabulary)
output_dim = len(destination_vocabulary)

# エンコーダーとデコーダーのインスタンスを作成
from encoder import Encoder
from decoder import Decoder
encoder = Encoder(input_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, DEVICE).to(DEVICE)
decoder = Decoder(output_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, output_dim, DROPOUT_RATE, DEVICE).to(DEVICE)

# モデルのインスタンスを作成
model = TranslationModel(encoder, decoder).to(DEVICE)

# 損失関数を定義
criterion = nn.CrossEntropyLoss(ignore_index=0)

# オプティマイザとスケジューラを設定
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=LEARNING_RATE)
total_steps = math.ceil(len(train_dataset) / BATCH_SIZE)
from utils import WarmupScheduler
scheduler = WarmupScheduler(optimizer, d_model=HIDDEN_SIZE, warmup_steps=WARMUP_STEPS, total_steps=total_steps)

# データローダーを作成
from data import create_data_loader
train_loader = create_data_loader(train_dataset, batch_size=BATCH_SIZE)
val_loader = create_data_loader(val_dataset, batch_size=BATCH_SIZE)

# Early Stoppingのパラメータ
best_val_loss = float('inf')
patience_counter = 0

# モデルのインスタンスを作成
model = TranslationModel(encoder, decoder).to(DEVICE)

# トレーニング ループ
for epoch in range(NUM_EPOCHS):
    start_time = time.time()
    total_loss = 0
    for i, (X_batch, y_batch) in enumerate(train_loader):
        # データを GPU に送る
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # 勾配をゼロで初期化
        optimizer.zero_grad()

        # 順伝播
        decoder_output = model(X_batch, y_batch)

        # 損失を計算
        loss = criterion(decoder_output.view(-1, output_dim), y_batch.view(-1))

        # 勾配を計算
        loss.backward()

        # パラメータを更新
        optimizer.step()
        scheduler.step()

        # 損失を累積
        total_loss += loss.item()

        # 進捗状況を出力
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")

    # 検証部分
    val_loss = validate(model, val_loader, criterion, DEVICE, output_dim)
    print(f"Validation Loss: {val_loss:.4f}")

    # Early Stoppingのチェック
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # 現在の日付とエポック数を取得
        current_date = datetime.now().strftime("%Y%m%d")
        model_filename = f"translation_model_{current_date}.pth"
        model_path = os.path.join("models", model_filename)
        # models ディレクトリが存在しない場合は作成
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # モデルを保存
        torch.save(model.state_dict(), model_path)
    else:
        patience_counter += 1
        if patience_counter >= PATIENCE:
            print("Early stopping triggered")
            break

    # エポックごとの要約を出力
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
