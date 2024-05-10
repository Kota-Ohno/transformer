import torch
from data import load_data, create_data_loader, get_vocab
from encoder import Encoder
from decoder import Decoder
from train import criterion
from config import MODEL_PATH, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, DEVICE

# データの読み込みと前処理
test_dataset = load_data(test_data_path)
data_path = 'path/to/your/dataset' # TODO:

# データとボキャブラリの読み込み
vocab = get_vocab(data_path)
input_dim = len(vocab)
output_dim = len(vocab)

# ハイパーパラメータの設定
encoder = Encoder(input_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, HIDDEN_SIZE, DROPOUT_RATE, DEVICE).to(DEVICE)
decoder = Decoder(input_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, output_dim, DROPOUT_RATE, DEVICE).to(DEVICE)
# モデルをロード
checkpoint = torch.load(MODEL_PATH)
encoder.load_state_dict(checkpoint["encoder_state_dict"])
decoder.load_state_dict(checkpoint["decoder_state_dict"])

# 評価モードに設定
encoder.eval()
decoder.eval()

# データローダーを作成
test_loader = create_data_loader(test_dataset, batch_size=64)

# 評価ループ
total_loss = 0
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        # データを GPU に送る
        X_batch = X_batch.to(DEVICE)
        y_batch = y_batch.to(DEVICE)

        # 順伝播
        encoder_output = encoder(X_batch)
        decoder_output = decoder(y_batch, encoder_output)

        # 損失を計算
        loss = criterion(decoder_output.view(-1, output_dim), y_batch.view(-1))

        # 損失を累積
        total_loss += loss.item()

# 平均損失を出力
average_loss = total_loss / len(test_dataset)
print(f"Average Loss: {average_loss:.4f}")