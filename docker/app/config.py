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
BATCH_SIZE = 16
LEARNING_RATE = 0.001

TRANSLATION_SOURCE = "ja_JP"
TRANSLATION_DESTINATION = "en_US"
TRANSLATION_SOURCE2 = "japanese"
TRANSLATION_DESTINATION2 = "english"

INPUT_VOCAB_PATH = "models/vocab_input.pth"
OUTPUT_VOCAB_PATH = "models/vocab_output.pth"

TOKENIZE_BATCH_SIZE = 128
