import torch
# ハイパーパラメータの設定
MAX_SEQ_LENGTH = 512
HIDDEN_SIZE = 512
NUM_HEADS = 8
NUM_LAYERS = 6
D_FF = 2048  # フィードフォワードネットワークの内部層の次元
DROPOUT_RATE = 0.1
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EPOCHS = 10000  # トレーニングを実行するための設定
PATIENCE = 5
WARMUP_STEPS = 4000
BATCH_SIZE = 64

# 定数の定義
BOS_TOKEN = "<s>"
EOS_TOKEN = "</s>"
PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"

MODEL_PATH = "model.save"

TRANSLATION_SOURCE = "en_US"
TRANSLATION_DESTINATION = "ja_JP"