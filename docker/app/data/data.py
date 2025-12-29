import torch
import torch.utils.data
from torch.utils.data import DataLoader
from collections import Counter
import spacy
import os
import sentencepiece as spm
import re

# データセットクラス
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, transform=None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x, y = self.X[idx], self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# データの読み込みと前処理
def preprocess_data(data):
    # トークン化やエンコーディングの具体的な処理をここに実装
    # 例: data = tokenize_and_encode(data)
    return data

def set_data(X, y):
    dataset = MyDataset(X, y)
    return dataset

def pad_inner_seq(seq, pad_token, max_length):
    """シーケンスをパディングする関数（一次元配列向け）"""
    # タプルの場合はリストに変換
    if isinstance(seq, tuple):
        seq = list(seq)

    # max_length以下なら埋める、超えていれば切り詰める
    if len(seq) < max_length:
        return seq + [pad_token] * (max_length - len(seq))
    else:
        return seq[:max_length]

def flatten_and_convert(sequence):
    """
    あらゆる形式のネストされた配列を平坦化し、すべての要素を整数に変換する関数
    無効な値は0に変換される
    """
    result = []

    def _flatten(item):
        if isinstance(item, (list, tuple)):
            for subitem in item:
                _flatten(subitem)
        else:
            try:
                result.append(int(item))
            except (ValueError, TypeError):
                result.append(0)

    _flatten(sequence)
    return result

def collate_fn(batch):
    """
    バッチデータをテンソルに変換するための関数
    どんなバッチデータでも安全に処理できるよう設計
    """
    # バッチからXとYのペアを取り出す
    X, Y = zip(*batch)

    # データを平坦化して整数のリストにする
    X_flat = [flatten_and_convert(x) for x in X]
    Y_flat = [flatten_and_convert(y) for y in Y]

    # 最大長を計算
    max_length_X = max([len(x) for x in X_flat], default=1)
    max_length_Y = max([len(y) for y in Y_flat], default=1)

    # パディングを適用
    X_padded = [pad_inner_seq(x, 0, max_length_X) for x in X_flat]
    Y_padded = [pad_inner_seq(y, 0, max_length_Y) for y in Y_flat]

    try:
        # テンソルに変換
        X_tensor = torch.tensor(X_padded, dtype=torch.long)
        Y_tensor = torch.tensor(Y_padded, dtype=torch.long)

        # ターゲット入力と出力の作成
        # 長さ1の場合の特別処理
        if Y_tensor.size(1) <= 1:
            # ダミーデータを追加（最小でも2の長さが必要）
            Y_tensor = torch.cat([Y_tensor, torch.zeros_like(Y_tensor)], dim=1)

        tgt_input = Y_tensor[:, :-1]  # 最後のトークンを除外
        tgt_output = Y_tensor[:, 1:]  # 最初のトークンを除外

        return X_tensor, tgt_input, tgt_output

    except Exception as e:
        print(f"Error in collate_fn: {e}")
        print(f"Emergency fallback activated")

        # 最後の手段: 完全に均一化されたダミーデータを返す
        batch_size = len(X)
        dummy_x = torch.zeros((batch_size, 2), dtype=torch.long)
        dummy_y = torch.zeros((batch_size, 1), dtype=torch.long)

        return dummy_x, dummy_y, dummy_y

# データローダーを作成
def create_data_loader(dataset_or_data, batch_size):
    """
    データセットまたはトークンIDのリストからデータローダーを作成

    Args:
        dataset_or_data: データセットインスタンスまたはトークンIDのリスト
        batch_size: バッチサイズ

    Returns:
        DataLoader: バッチ処理を行うデータローダー
    """
    # トークンIDのリストを受け取った場合、データセットに変換
    if isinstance(dataset_or_data, list):
        # 入力と出力を同じにする（オートエンコーダーのようなアプローチ）
        # 実際のアプリケーションでは、入力と出力を適切に分ける必要がある
        dataset = set_data(dataset_or_data, dataset_or_data)
    else:
        # すでにデータセットインスタンスの場合はそのまま使用
        dataset = dataset_or_data

    # データローダー作成
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
    # vocabularyがVocabularyクラスのインスタンスである場合
    if hasattr(vocabulary, 'id2token'):
        return [vocabulary.id2token[id] for id in ids]
    # vocabularyが辞書の場合（語彙ファイルからロードした場合などに発生）
    else:
        # トークンとIDのマッピングを反転して使用
        id_to_token = {v: k for k, v in vocabulary.items()}
        return [id_to_token.get(id, '<unk>') for id in ids]
