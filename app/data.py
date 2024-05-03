import torch
import torch.utils.data
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from config import MAX_SEQ_LENGTH, BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, UNK_TOKEN
import collections
import MeCab

# データセットクラス
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, y, transform=None):
        self.X = X
        self.y = y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

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
def split_data(data, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y, test_size=test_size, random_state=random_state)
    train_dataset = MyDataset(X_train, y_train)
    test_dataset = MyDataset(X_test, y_test)
    return train_dataset, test_dataset

# バッチを作成するためのコラテーション関数
def collate_fn(batch):
    X, y = zip(*batch)
    X_padded = [pad_seq(x, PAD_TOKEN) for x in X]
    y_padded = [pad_seq(y, PAD_TOKEN) for y in y]
    return torch.tensor(X_padded, dtype=torch.long), torch.tensor(y_padded, dtype=torch.long)

# パディング関数
def pad_seq(seq, pad_token):
    # シーケンス長をパディングトークンで埋める
    seq_length = MAX_SEQ_LENGTH - len(seq)
    return seq + [pad_token] * seq_length

# データローダーを作成
def create_data_loader(dataset, batch_size):
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

# トークン化関数
def tokenize(data):
    # MeCabを使用したトークン化
    mecab = MeCab.Tagger("-Owakati")
    return mecab.parse(data).strip().split()

# ボキャブラリの作成
def build_vocab(data):
    # トークン化されたデータからボキャブラリを作成
    tokens = [token for line in data for token in tokenize(line)]
    vocab = collections.Counter(tokens)
    vocab = {word: i for i, word in enumerate(vocab)}
    return vocab

# ボキャブラリを返す関数
def get_vocab(data_path):
    data = load_data(data_path)
    vocab = build_vocab(data)
    return vocab