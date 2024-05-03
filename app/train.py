import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import time

from data import load_data, split_data, get_vocab

# データとボキャブラリの読み込み
data_path = 'path/to/your/dataset' # TODO:
vocab = get_vocab(data_path)
train_dataset, test_dataset = split_data(load_data(data_path))

# ハイパーパラメータの設定
from config import HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, DEVICE, MODEL_PATH, NUM_EPOCHS
input_dim = len(vocab)
output_dim = len(vocab)

# エンコーダーとデコーダーのインスタンスを作成
from encoder import Encoder
from decoder import Decoder
encoder = Encoder(input_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, HIDDEN_SIZE, DROPOUT_RATE, DEVICE).to(DEVICE)
decoder = Decoder(input_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, output_dim, DROPOUT_RATE, DEVICE).to(DEVICE)

# 損失関数を定義
criterion = nn.CrossEntropyLoss(ignore_index=0)

# オプティマイザとスケジューラを設定
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
total_steps = len(train_dataset) * NUM_EPOCHS
warmup_steps = 4000
from utils import WarmupScheduler
scheduler = WarmupScheduler(optimizer, d_model=HIDDEN_SIZE, warmup_steps=warmup_steps, total_steps=total_steps)

# データローダーを作成
from data import create_data_loader
train_loader = create_data_loader(train_dataset, batch_size=64)

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
        encoder_output = encoder(X_batch)
        decoder_output = decoder(y_batch, encoder_output)

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

    # エポックごとの要約を出力
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
    # モデルを保存
    torch.save({
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
    }, MODEL_PATH)