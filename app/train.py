import torch
import torch.nn as nn
import torch.optim as optim
import time
from datasets import load_dataset

from data import set_data, build_vocab, tokenize_japanese, tokenize_english, tokens_to_ids, create_data_loader
from config import TRANSLATION_SOURCE, TRANSLATION_DESTINATION, WARMUP_STEPS, PATIENCE
from utils import validate, TranslationModel

# データとボキャブラリの読み込み
train_data = load_dataset("Amani27/massive_translation_dataset", split="train")
val_data = load_dataset("Amani27/massive_translation_dataset", split="validation")

train_dataset = set_data(train_data[TRANSLATION_SOURCE], train_data[TRANSLATION_DESTINATION])
val_dataset = set_data(val_data[TRANSLATION_SOURCE], val_data[TRANSLATION_DESTINATION])

# TODO: テスト用
train_dataset = train_dataset[:50]
val_dataset = val_dataset[:10]

# ソースとターゲットをペアにする
paired_train_dataset = list(zip(train_dataset[0], train_dataset[1]))
paired_val_dataset = list(zip(val_dataset[0], val_dataset[1]))

# トークナイズ
print("======= tokenizing now =======")
tokenized_train_src = [tokenize_english(src) for src, _ in paired_train_dataset]
tokenized_val_src = [tokenize_english(src) for src, _ in paired_val_dataset]
print("50%...")
tokenized_train_tgt = [tokenize_japanese(tgt) for _, tgt in paired_train_dataset]
tokenized_val_tgt = [tokenize_japanese(tgt) for _, tgt in paired_val_dataset]
print("======= tokenizing finished ======")

# ボキャブラリの作成
print("======= build vocabulary now ======")
base_vocab = build_vocab(tokenized_train_src)
destination_vocab = build_vocab(tokenized_train_tgt)
print("======= build vocabulary finished ======")

# 単語IDに変換
train_dataset = [(tokens_to_ids(src, base_vocab), tokens_to_ids(tgt, destination_vocab)) for src, tgt in zip(tokenized_train_src, tokenized_train_tgt)]
val_dataset = [(tokens_to_ids(src, base_vocab), tokens_to_ids(tgt, destination_vocab)) for src, tgt in zip(tokenized_val_src, tokenized_val_tgt)]

# ハイパーパラメータの設定
from config import HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, DEVICE, MODEL_PATH, NUM_EPOCHS
input_dim = len(base_vocab)
output_dim = len(destination_vocab)

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
optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=0.001)
total_steps = len(train_dataset) * NUM_EPOCHS
from utils import WarmupScheduler
scheduler = WarmupScheduler(optimizer, d_model=HIDDEN_SIZE, warmup_steps=WARMUP_STEPS, total_steps=total_steps)

# データローダーを作成
from data import create_data_loader
train_loader = create_data_loader(train_dataset, batch_size=64)
val_loader = create_data_loader(val_dataset, batch_size=64)

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
            # ベストモデルを保存
            torch.save(model.state_dict(), MODEL_PATH)
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print("Early stopping triggered")
                break

    # エポックごとの要約を出力
    epoch_time = time.time() - start_time
    print(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
