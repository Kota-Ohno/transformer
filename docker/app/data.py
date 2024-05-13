import torch
import torch.utils.data
from torch.utils.data import DataLoader
from config import TRANSLATION_SOURCE, TRANSLATION_DESTINATION

import spacy

# データセットクラス
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, Y = self.X[idx], self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, Y

# データの読み込みと前処理
def preprocess_data(data):
    # トークン化やエンコーディングの具体的な処理をここに実装
    # 例: data = tokenize_and_encode(data)
    return data

def load_data(data_path):
    with open(data_path, "r") as f:
        data = f.read().splitlines()
    data = preprocess_data(data)
    return data

# トレーニングセットとテストセットに分割
def set_data(X, y):
    dataset = MyDataset(X, y)
    return dataset

def pad_inner_seq(seq, pad_token, max_length):
    # 既存のシーケンスの長さが max_length 未満の場合、不足分を pad_token で埋める
    padded_seq = seq + [pad_token] * (max_length - len(seq))
    return padded_seq


def collate_fn(batch):
    X, Y = zip(*batch)
    
    # データセット全体で最長のシーケンス長を取得
    max_length_X = max(len(x) for x in X)
    max_length_Y = max(len(y) for y in Y)
    max_length = max(max_length_X, max_length_Y)
    
    # 各シーケンス内のトークンリストをパディング
    X_padded = [pad_inner_seq(x, 0, max_length) for x in X]
    Y_padded = [pad_inner_seq(y, 0, max_length) for y in Y]
    
    # テンソルに変換
    X_tensor = torch.tensor(X_padded, dtype=torch.long)
    Y_tensor = torch.tensor(Y_padded, dtype=torch.long)

    return X_tensor, Y_tensor

# データローダーを作成
def create_data_loader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# spacyのモデルをロード
nlp_ja = spacy.load("ja_core_news_md")
nlp_en = spacy.load("en_core_web_md")

def tokenize(sentence, lang):
    if lang == "ja_JP":
        doc = nlp_ja(sentence)
    elif lang == "en_US":
        doc = nlp_en(sentence)
    else:
        print("not yet implemented")

    tokens = [token.text for token in doc]
    print(".", end="")
    return tokens

# ボキャブラリの作成
def build_vocab(tokenized_data, special_tokens=None):
    # 特殊トークンを初期化
    if special_tokens is None:
        special_tokens = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}

    # すべてのトークンをフラットなリストにする
    all_tokens = [token for sentence in tokenized_data for token in sentence]
    # 重複を削除してボキャブラリを作成
    unique_tokens = set(all_tokens)
    # 特殊トークンを追加
    vocab = {token: idx + len(special_tokens) for idx, token in enumerate(unique_tokens)}
    # 特殊トークンのインデックスを更新
    vocab.update(special_tokens)
    return vocab

def tokens_to_ids(tokens, vocab):
    return [vocab.get(token, vocab['<unk>']) for token in tokens]

