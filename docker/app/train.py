import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import math

from datetime import datetime

from data import create_data_loader
from config import WARMUP_STEPS, PATIENCE, BATCH_SIZE, LEARNING_RATE, INPUT_VOCAB_PATH, OUTPUT_VOCAB_PATH
from utils import validate, TranslationModel

from text_tokenizer import load_tokenized_data

# トークナイズ済みデータの読み込み
train_data_path = "tokenized_train_data.pth"
train_token_ids = load_tokenized_data(train_data_path)
if train_token_ids is None:
    print("先にtext_tokenizer.pyを実行してください")

val_data_path = "tokenized_val_data.pth"
val_token_ids = load_tokenized_data(val_data_path)
if val_token_ids is None:
    print("先にtext_tokenizer.pyを実行してください")

input_vocab = torch.load(INPUT_VOCAB_PATH)
output_vocab = torch.load(OUTPUT_VOCAB_PATH)

# ハイパーパラメータの設定
from config import HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, DEVICE, NUM_EPOCHS
input_dim = len(input_vocab)
output_dim = len(output_vocab)

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
total_steps = math.ceil(len(train_token_ids) / BATCH_SIZE)
from utils import WarmupScheduler
scheduler = WarmupScheduler(optimizer, d_model=HIDDEN_SIZE, warmup_steps=WARMUP_STEPS, total_steps=total_steps)

# データローダーを作成
from data import create_data_loader
train_loader = create_data_loader(train_token_ids, batch_size=BATCH_SIZE)
val_loader = create_data_loader(val_token_ids, batch_size=BATCH_SIZE)

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
