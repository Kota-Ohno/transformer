import torch
import torch.utils.data
from torch.utils.data import DataLoader
from collections import Counter
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

class Vocabulary:
    def __init__(self, special_tokens=None):
        # 特殊トークンの初期化
        if special_tokens is None:
            special_tokens = {'<pad>': 0, '<unk>': 1, '<s>': 2}
        
        self.token2id = special_tokens
        self.id2token = {v: k for k, v in special_tokens.items()}
        self.next_id = len(special_tokens)

    def add_token(self, token):
        if token not in self.token2id:
            self.token2id[token] = self.next_id
            self.id2token[self.next_id] = token
            self.next_id += 1

    def build_vocab(self, counter, min_freq=1):
        # カウンターの頻度でソート
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
        
        # 頻度がmin_freq以上のトークンを追加
        for token, freq in sorted_tokens:
            if freq >= min_freq:
                self.add_token(token)

    def __len__(self):
        return len(self.token2id)

    # __getitem__メソッドを追加
    def __getitem__(self, token):
        return self.token2id.get(token, self.token2id['<unk>'])

# 使用例
def build_vocabulary(tokenized_data, special_tokens=None):
    # トークンのカウント
    counter = Counter(token for sentence in tokenized_data for token in sentence)
    
    # Vocabularyオブジェクトの作成と構築
    vocabulary = Vocabulary(special_tokens)
    vocabulary.build_vocab(counter)
    
    return vocabulary

def tokens_to_ids(tokens, vocabulary):
    return [vocabulary[token] for token in tokens]

def ids_to_tokens(ids, vocabulary):
    return [vocabulary.id2token[id] for id in ids]
