import torch
import sys
import os
import glob
from encoder import Encoder
from decoder import Decoder
from utils import TranslationModel
from config import DEVICE, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, TRANSLATION_SOURCE, TRANSLATION_DESTINATION, INPUT_VOCAB_PATH, OUTPUT_VOCAB_PATH
from data import tokenize, tokens_to_ids, ids_to_tokens

def load_model(input_vocab, output_vocab):
    input_dim = len(input_vocab)
    output_dim = len(output_vocab)
    encoder = Encoder(input_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, DROPOUT_RATE, DEVICE).to(DEVICE)
    decoder = Decoder(output_dim, HIDDEN_SIZE, NUM_HEADS, NUM_LAYERS, D_FF, output_dim, DROPOUT_RATE, DEVICE).to(DEVICE)
    model = TranslationModel(encoder, decoder).to(DEVICE)
    
    # modelsディレクトリから最新のモデルファイル名を取得
    model_files = glob.glob('models/*.pth')
    model_files = [f for f in model_files if not f.endswith('vocab_input.pth') and not f.endswith('vocab_output.pth')]
    model_filename = max(model_files, key=os.path.getctime)  # 最新のファイルを選択

    # ここでパスを結合する際に、"models" ディレクトリを重複させないように
    model_path = os.path.join("models", os.path.basename(model_filename))
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))  # DEVICEにマップ
    model.eval()
    return model

def load_vocab(vocab_path):
    return torch.load(vocab_path)

def preprocess_input(sentence, input_vocab):
    tokens = tokenize(sentence, TRANSLATION_SOURCE)
    # トークンからインデックスへのマッピングを一度取得
    input_stoi = input_vocab.get_stoi()
    token_ids = tokens_to_ids(tokens, input_stoi, input_vocab['<unk>'])  # vocabは事前に定義されたボキャブラリ
    return torch.tensor([token_ids], dtype=torch.long).to(DEVICE)

def predict(model, input_tensor):
    with torch.no_grad():
        output = model(input_tensor, input_tensor)
    return output

def main():
    input_vocab = load_vocab(INPUT_VOCAB_PATH)
    output_vocab = load_vocab(OUTPUT_VOCAB_PATH)
    model = load_model(input_vocab, output_vocab)

    spacer = ""
    if TRANSLATION_DESTINATION == 'en_US':
        spacer = " "

    print("モデルをロードしました。入力を待っています...")
    print("exitと入力すると終了します...")
    for line in sys.stdin:
        stripped_line = line.strip()
        if stripped_line.lower() == "exit":
            break
        input_tensor = preprocess_input(line.strip(), input_vocab)
        output_ids = predict(model, input_tensor).argmax(-1).squeeze().tolist()  # モデルの出力から最も可能性の高い単語のIDを取得
        output_tokens = ids_to_tokens(output_ids, output_vocab)  # IDをトークンに変換
        print("Output:", spacer.join(output_tokens))  # トークンをスペースで結合して出力
    print("終了しました")

if __name__ == "__main__":
    main()
