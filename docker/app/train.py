import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
from datetime import datetime
from data import create_data_loader
from config import (
    WARMUP_STEPS, PATIENCE, BATCH_SIZE, LEARNING_RATE, INPUT_VOCAB_PATH, 
    OUTPUT_VOCAB_PATH, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, 
    DROPOUT_RATE, DEVICE, NUM_EPOCHS
)
from utils import validate, TranslationModel, WarmupScheduler
from text_tokenizer import load_tokenized_data
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_data(train_data_path, val_data_path):
    """トークナイズ済みデータを読み込む関数"""
    train_token_ids = load_tokenized_data(train_data_path)
    if train_token_ids is None:
        logging.error("先にtext_tokenizer.pyを実行してください")
        raise ValueError("トレーニングデータがロードできません")

    val_token_ids = load_tokenized_data(val_data_path)
    if val_token_ids is None:
        logging.error("先にtext_tokenizer.pyを実行してください")
        raise ValueError("検証データがロードできません")

    return train_token_ids, val_token_ids

def main():
    # トークナイズ済みデータのパス
    train_data_path = "tokenized_train_data.pth"
    val_data_path = "tokenized_val_data.pth"

    # データの読み込み
    train_token_ids, val_token_ids = load_data(train_data_path, val_data_path)
    
    # ボキャブラリの読み込み
    input_vocab = torch.load(INPUT_VOCAB_PATH)
    output_vocab = torch.load(OUTPUT_VOCAB_PATH)
    
    # 入力と出力の次元を設定
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
    criterion = nn.CrossEntropyLoss(ignore_index=input_vocab['<pad>'])
    
    # オプティマイザとスケジューラを定義
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = WarmupScheduler(optimizer, HIDDEN_SIZE, WARMUP_STEPS, NUM_EPOCHS)
    
    # データローダーを作成
    train_loader = create_data_loader(train_token_ids, BATCH_SIZE)
    val_loader = create_data_loader(val_token_ids, BATCH_SIZE)
    
    best_val_loss = float('inf')
    patience_counter = 0
    total_steps = len(train_loader)
    
    # トレーニングループ
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
            logging.info(f"Epoch [{epoch+1}/{NUM_EPOCHS}] Step [{i+1}/{total_steps}], Loss: {loss.item():.4f}")
        
        # 検証部分
        val_loss = validate(model, val_loader, criterion, DEVICE, output_dim)
        logging.info(f"Validation Loss: {val_loss:.4f}")
        
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
                logging.info("Early stopping triggered")
                break
        
        # エポックごとの要約を出力
        epoch_time = time.time() - start_time
        logging.info(f"Epoch {epoch+1} took {epoch_time:.2f} seconds.")
        
if __name__ == "__main__":
    main()
