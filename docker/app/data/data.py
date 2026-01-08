import torch
import torch.utils.data
from torch.utils.data import DataLoader
from collections import Counter
import spacy
import os
import re
import logging
import traceback
import weakref
import threading
from typing import List, Tuple, Optional, Callable, Any, Dict

# データセットクラス
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X: List, Y: List, transform: Optional[Callable] = None):
        self.X = X
        self.Y = Y
        self.transform = transform

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> Tuple[Any, Any]:
        x, y = self.X[idx], self.Y[idx]
        if self.transform:
            x = self.transform(x)
        return x, y

# データの読み込みと前処理
def preprocess_data(data: Any) -> Any:
    # トークン化やエンコーディングの具体的な処理をここに実装
    # 例: data = tokenize_and_encode(data)
    return data

def set_data(X: List, y: List) -> MyDataset:
    dataset = MyDataset(X, y)
    return dataset

def pad_inner_seq(seq: List[int], pad_token: int, max_length: int) -> List[int]:
    """シーケンスをパディングする関数（一次元配列向け）"""
    # タプルの場合はリストに変換
    if isinstance(seq, tuple):
        seq = list(seq)

    # max_length以下なら埋める、超えていれば切り詰める
    if len(seq) < max_length:
        return seq + [pad_token] * (max_length - len(seq))
    else:
        return seq[:max_length]

def flatten_and_convert(sequence: Any) -> List[int]:
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

def collate_fn(batch: List[Tuple[Any, Any]], pad_token_id: int = 0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    バッチデータをテンソルに変換するための関数
    どんなバッチデータでも安全に処理できるよう設計

    Args:
        batch: バッチデータのリスト
        pad_token_id: パディングに使用するトークンID（デフォルト: 0）
    """
    # バッチからXとYのペアを取り出す
    X, Y = zip(*batch)

    # データを平坦化して整数のリストにする
    X_flat = [flatten_and_convert(x) for x in X]
    Y_flat = [flatten_and_convert(y) for y in Y]

    # 長さ1以下のシーケンスをフィルタリング（訓練に有用でないため除外）
    valid_indices = [i for i, y_seq in enumerate(Y_flat) if len(y_seq) > 1]

    if len(valid_indices) == 0:
        # バッチ内のすべてのサンプルが無効な場合
        raise ValueError(
            "Batch contains only sequences with length <= 1. "
            "Please ensure your dataset contains sequences with length > 1."
        )

    # 有効なサンプルのみを保持
    X_flat = [X_flat[i] for i in valid_indices]
    Y_flat = [Y_flat[i] for i in valid_indices]

    # 最大長を計算
    max_length_X = max([len(x) for x in X_flat], default=1)
    max_length_Y = max([len(y) for y in Y_flat], default=1)

    # パディングを適用
    X_padded = [pad_inner_seq(x, pad_token_id, max_length_X) for x in X_flat]
    Y_padded = [pad_inner_seq(y, pad_token_id, max_length_Y) for y in Y_flat]

    try:
        # テンソルに変換
        X_tensor = torch.tensor(X_padded, dtype=torch.long)
        Y_tensor = torch.tensor(Y_padded, dtype=torch.long)

        # ターゲット入力と出力の作成
        # フィルタリングにより、Y_tensor.size(1) > 1が保証される
        tgt_input = Y_tensor[:, :-1]  # 最後のトークンを除外
        tgt_output = Y_tensor[:, 1:]  # 最初のトークンを除外

        return X_tensor, tgt_input, tgt_output

    except Exception as e:
        # 完全な例外情報とトレースバックをログに記録
        logging.exception(f"Error in collate_fn: {e}")
        logging.error(f"Traceback:\n{traceback.format_exc()}")

        # エラーを再発生させて処理を停止し、問題を可視化
        raise RuntimeError(f"Failed to process batch in collate_fn: {e}") from e

# データローダーを作成
def create_data_loader(dataset_or_data: Any, batch_size: int, pad_token_id: Optional[int] = None) -> DataLoader:
    """
    データセットまたはトークンIDのリストからデータローダーを作成

    Args:
        dataset_or_data: データセットインスタンスまたはトークンIDのリスト
        batch_size: バッチサイズ
        pad_token_id: パディングに使用するトークンID（Noneの場合はデフォルトの0を使用）

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

    # pad_token_idが指定されている場合はlambdaでラップして渡す
    if pad_token_id is not None:
        collate_fn_with_pad = lambda batch: collate_fn(batch, pad_token_id=pad_token_id)
    else:
        collate_fn_with_pad = collate_fn

    # データローダー作成
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_with_pad)

# spacyのモデルを遅延ロード（グローバルロードを削除）
_nlp_ja = None
_nlp_en = None
_nlp_ja_lock = threading.Lock()
_nlp_en_lock = threading.Lock()

def get_nlp_ja() -> Optional[Any]:
    """
    日本語spacyモデルを遅延ロードするアクセサ関数（スレッドセーフ）
    モデルが存在しない場合はNoneを返し、エラーをログに記録
    """
    global _nlp_ja
    # ダブルチェックロッキングパターンを使用
    if _nlp_ja is None:
        with _nlp_ja_lock:
            # ロック取得後に再度チェック（他のスレッドが既にロードした可能性があるため）
            if _nlp_ja is None:
                try:
                    _nlp_ja = spacy.load("ja_core_news_md")
                    logging.info("Successfully loaded Japanese spacy model: ja_core_news_md")
                except OSError as e:
                    logging.error(f"Failed to load Japanese spacy model 'ja_core_news_md': {e}")
                    logging.error("Please install the model with: python -m spacy download ja_core_news_md")
                    _nlp_ja = None
                except Exception as e:
                    logging.error(f"Unexpected error loading Japanese spacy model: {e}")
                    logging.error(f"Traceback:\n{traceback.format_exc()}")
                    _nlp_ja = None
    return _nlp_ja

def get_nlp_en() -> Optional[Any]:
    """
    英語spacyモデルを遅延ロードするアクセサ関数（スレッドセーフ）
    モデルが存在しない場合はNoneを返し、エラーをログに記録
    """
    global _nlp_en
    # ダブルチェックロッキングパターンを使用
    if _nlp_en is None:
        with _nlp_en_lock:
            # ロック取得後に再度チェック（他のスレッドが既にロードした可能性があるため）
            if _nlp_en is None:
                try:
                    _nlp_en = spacy.load("en_core_web_md")
                    logging.info("Successfully loaded English spacy model: en_core_web_md")
                except OSError as e:
                    logging.error(f"Failed to load English spacy model 'en_core_web_md': {e}")
                    logging.error("Please install the model with: python -m spacy download en_core_web_md")
                    _nlp_en = None
                except Exception as e:
                    logging.error(f"Unexpected error loading English spacy model: {e}")
                    logging.error(f"Traceback:\n{traceback.format_exc()}")
                    _nlp_en = None
    return _nlp_en

def tokenize(sentence: str, lang: str) -> List[str]:
    # Validate language upfront
    if lang not in ("ja_JP", "en_US"):
        raise ValueError(f"Unsupported language: {lang}. Supported languages are 'ja_JP' and 'en_US'.")

    # Process sentence based on language with lazy-loaded models
    if lang == "ja_JP":
        nlp = get_nlp_ja()
        if nlp is None:
            raise RuntimeError(
                "Japanese spacy model 'ja_core_news_md' is not available. "
                "Please install it with: python -m spacy download ja_core_news_md"
            )
        doc = nlp(sentence)
    else:  # lang == "en_US"
        nlp = get_nlp_en()
        if nlp is None:
            raise RuntimeError(
                "English spacy model 'en_core_web_md' is not available. "
                "Please install it with: python -m spacy download en_core_web_md"
            )
        doc = nlp(sentence)

    tokens = [token.text for token in doc]
    return tokens

class Vocabulary:
    def __init__(self, special_tokens: Optional[Dict[str, int]] = None):
        # 特殊トークンの初期化
        if special_tokens is None:
            special_tokens = {'<pad>': 0, '<unk>': 1, '<s>': 2}

        self.token2id = special_tokens.copy()
        # '<unk>'が存在しない場合は追加（安全なフォールバック用）
        if '<unk>' not in self.token2id:
            # 既存のIDの最大値を取得し、+1した値を割り当て
            max_id = max(self.token2id.values()) if self.token2id else -1
            self.token2id['<unk>'] = max_id + 1

        self.id2token = {v: k for k, v in self.token2id.items()}
        # Compute next_id based on maximum existing ID to avoid collisions with non-contiguous IDs
        if self.token2id:
            self.next_id = max(self.token2id.values()) + 1
        else:
            self.next_id = 0

    def add_token(self, token: str) -> None:
        if token not in self.token2id:
            self.token2id[token] = self.next_id
            self.id2token[self.next_id] = token
            self.next_id += 1

    def build_vocab(self, counter: Counter, min_freq: int = 1) -> None:
        # カウンターの頻度でソート
        sorted_tokens = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

        # 頻度がmin_freq以上のトークンを追加
        for token, freq in sorted_tokens:
            if freq >= min_freq:
                self.add_token(token)

    def __len__(self) -> int:
        return len(self.token2id)

    # __getitem__メソッドを追加
    def __getitem__(self, token: str) -> int:
        # 安全なルックアップ：tokenが見つからない場合、'<unk>'を試み、それも存在しない場合は0を返す
        unk_id = self.token2id.get('<unk>', 0)
        return self.token2id.get(token, unk_id)

# 使用例
def build_vocabulary(tokenized_data: List[List[str]], special_tokens: Optional[Dict[str, int]] = None) -> Vocabulary:
    # トークンのカウント
    counter = Counter(token for sentence in tokenized_data for token in sentence)

    # Vocabularyオブジェクトの作成と構築
    vocabulary = Vocabulary(special_tokens)
    vocabulary.build_vocab(counter)

    return vocabulary

def tokens_to_ids(tokens: List[str], vocabulary: Vocabulary) -> List[int]:
    return [vocabulary[token] for token in tokens]

# モジュールレベルのキャッシュ
# WeakKeyDictionary: 弱参照可能なVocabularyオブジェクト用（GC時に自動削除）
_id_to_token_cache_weak = weakref.WeakKeyDictionary()
# WeakKeyDictionary: 組み込みdict語彙用（vocabularyオブジェクト自体をキーとして使用、GC時に自動削除）
_dict_vocab_cache = weakref.WeakKeyDictionary()
_dict_vocab_cache_lock = threading.Lock()

def clear_dict_vocab_cache() -> None:
    """組み込みdict語彙キャッシュをクリアする（メモリ管理用）"""
    global _dict_vocab_cache
    with _dict_vocab_cache_lock:
        _dict_vocab_cache.clear()

def ids_to_tokens(ids: List[int], vocabulary: Any) -> List[str]:
    # vocabularyがVocabularyクラスのインスタンスである場合（弱参照可能）
    if hasattr(vocabulary, 'id2token'):
        # 安全なルックアップを使用（存在しないIDの場合は'<unk>'を返す）
        return [vocabulary.id2token.get(id, '<unk>') for id in ids]
    # vocabularyが辞書の場合（語彙ファイルからロードした場合などに発生）
    else:
        # vocabularyオブジェクト自体をキーとしてキャッシュを管理（WeakKeyDictionary使用）
        with _dict_vocab_cache_lock:
            # キャッシュに存在しない場合のみ逆マッピングを構築
            if vocabulary not in _dict_vocab_cache:
                _dict_vocab_cache[vocabulary] = {v: k for k, v in vocabulary.items()}
            id_to_token = _dict_vocab_cache[vocabulary]
        return [id_to_token.get(id, '<unk>') for id in ids]
