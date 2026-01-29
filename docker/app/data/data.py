import torch
import torch.utils.data
from torch.utils.data import DataLoader
from collections import Counter
import spacy
import os
import re
import logging
import traceback
import threading
import functools
from typing import List, Tuple, Optional, Callable, Any, Dict

# データセットクラス
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, X: List, Y: List, transform: Optional[Callable] = None):
        # XとYがシーケンスまたは__len__を持つことを確認
        if not hasattr(X, '__len__'):
            raise TypeError(f"X must be a sequence or have __len__, got {type(X)}")
        if not hasattr(Y, '__len__'):
            raise TypeError(f"Y must be a sequence or have __len__, got {type(Y)}")

        # XとYの長さが一致することを確認
        len_x = len(X)
        len_y = len(Y)
        if len_x != len_y:
            raise ValueError(
                f"X and Y must have the same length, but got len(X)={len_x} and len(Y)={len_y}"
            )

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
    logger = logging.getLogger(__name__)

    def _flatten(item, path: str = ""):
        if isinstance(item, (list, tuple)):
            for idx, subitem in enumerate(item):
                new_path = f"{path}[{idx}]" if path else f"[{idx}]"
                _flatten(subitem, new_path)
        else:
            try:
                result.append(int(item))
            except (ValueError, TypeError) as e:
                # 警告ログを出力（ログレベルが有効な場合のみ）
                if logger.isEnabledFor(logging.WARNING):
                    logger.warning(
                        f"flatten_and_convert: 無効な値を検出しました。"
                        f"項目: {item!r}, 型: {type(item).__name__}, "
                        f"パス: {path if path else 'root'}, エラー: {e}. "
                        f"0に変換します。"
                    )
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

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            (X_tensor, tgt_input, tgt_output)のタプル

    Raises:
        ValueError: バッチが空の場合、またはすべてのサンプルが無効な場合
    """
    # 空のバッチに対する防御的チェック
    if not batch:
        raise ValueError(
            "Cannot process empty batch. "
            "Please ensure the dataset contains at least one sample and batch_size > 0."
        )

    # バッチからXとYのペアを取り出す
    X, Y = zip(*batch)

    # データを平坦化して整数のリストにする
    X_flat = [flatten_and_convert(x) for x in X]
    Y_flat = [flatten_and_convert(y) for y in Y]

    # 長さ1以下のYシーケンスと長さ0のXシーケンスをフィルタリング（訓練に有用でないため除外）
    original_len = len(X_flat)
    valid_indices = [
        i for i, (x_seq, y_seq) in enumerate(zip(X_flat, Y_flat))
        if len(y_seq) > 1 and len(x_seq) > 0
    ]

    # 無効なサンプルがフィルタされた場合に警告をログに記録
    dropped = original_len - len(valid_indices)
    if dropped > 0:
        logging.warning(
            f"無効なサンプルがフィルタされました: X_flatとY_flatから{dropped}個のサンプルを削除しました "
            f"(元のバッチサイズ: {original_len}, 残りのバッチサイズ: {len(valid_indices)}). "
            f"valid_indicesには{len(valid_indices)}個の有効なインデックスが含まれています。"
        )

    if len(valid_indices) == 0:
        # バッチ内のすべてのサンプルが無効な場合
        raise ValueError(
            "Batch contains only invalid sequences (Y length <= 1 or X length == 0). "
            "Please ensure your dataset contains sequences with Y length > 1 and X length > 0."
        )

    # 有効なサンプルのみを保持
    X_flat = [X_flat[i] for i in valid_indices]
    Y_flat = [Y_flat[i] for i in valid_indices]

    # 最大長を計算（X_flatが空でないことは既に確認済み）
    max_length_X = max([len(x) for x in X_flat])
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

        # エラーを再発生させて処理を停止し、問題を可視化
        raise RuntimeError(f"Failed to process batch in collate_fn: {e}") from e

# データローダーを作成
def create_data_loader(dataset_or_data: Any, batch_size: int, pad_token_id: Optional[int] = None, shuffle: bool = True) -> DataLoader:
    """
    データセットまたはトークンIDのリストからデータローダーを作成

    Args:
        dataset_or_data: データセットインスタンスまたはトークンIDのリスト。
            リストが渡された場合、X=Y（オートエンコーダー）として扱われます。
            入力と出力を分けたい場合は、MyDatasetインスタンスを直接渡すか、
            (X, Y)タプルのリストを渡してください。
        batch_size: バッチサイズ
        pad_token_id: パディングに使用するトークンID（Noneの場合はデフォルトの0を使用）
        shuffle: データをシャッフルするかどうか（デフォルト: True）

    Returns:
        DataLoader: バッチ処理を行うデータローダー

    Warning:
        プレーンなリスト（list）が渡された場合、set_data(dataset_or_data, dataset_or_data)
        が呼び出され、X=Yとして扱われます（オートエンコーダー動作）。
        入力と出力を分けたい場合は、MyDatasetインスタンスを直接作成して渡してください。
    """
    # トークンIDのリストを受け取った場合、データセットに変換
    if isinstance(dataset_or_data, list):
        # 2タプルのリストかどうかをチェック
        if len(dataset_or_data) > 0 and all(
            isinstance(item, (list, tuple)) and len(item) == 2
            for item in dataset_or_data
        ):
            # (X, Y)タプルのリストの場合、XとYに分割
            x_list, y_list = zip(*dataset_or_data)
            dataset = set_data(list(x_list), list(y_list))
        else:
            # プレーンなリストの場合、X=Yとして扱う（オートエンコーダー）
            dataset = set_data(dataset_or_data, dataset_or_data)
    else:
        # すでにデータセットインスタンスの場合はそのまま使用
        dataset = dataset_or_data

    # pad_token_idが指定されている場合はfunctools.partialでラップして渡す（pickle-safe）
    if pad_token_id is not None:
        collate_fn_with_pad = functools.partial(collate_fn, pad_token_id=pad_token_id)
    else:
        collate_fn_with_pad = collate_fn

    # データローダー作成
    return DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn_with_pad)

# spacyのモデルを遅延ロード（グローバルロードを削除）
_nlp_ja = None
_nlp_en = None
_nlp_ja_lock = threading.Lock()
_nlp_en_lock = threading.Lock()
_nlp_ja_load_failed = False
_nlp_en_load_failed = False

def get_nlp_ja() -> Optional[Any]:
    """
    日本語spacyモデルを遅延ロードするアクセサ関数（スレッドセーフ）
    モデルが存在しない場合はNoneを返し、エラーをログに記録
    """
    global _nlp_ja, _nlp_ja_load_failed
    # 以前のロード失敗をチェック（リトライを防ぐ）
    if _nlp_ja_load_failed:
        return None
    # ダブルチェックロッキングパターンを使用
    if _nlp_ja is None:
        with _nlp_ja_lock:
            # ロック取得後に再度チェック（他のスレッドが既にロードした可能性があるため）
            if _nlp_ja is None and not _nlp_ja_load_failed:
                try:
                    _nlp_ja = spacy.load("ja_core_news_md")
                    logging.info("Successfully loaded Japanese spacy model: ja_core_news_md")
                except OSError as e:
                    logging.error(f"Failed to load Japanese spacy model 'ja_core_news_md': {e}")
                    logging.error("Please install the model with: python -m spacy download ja_core_news_md")
                    _nlp_ja_load_failed = True
                    _nlp_ja = None
                except Exception as e:
                    logging.error(f"Unexpected error loading Japanese spacy model: {e}")
                    logging.error(f"Traceback:\n{traceback.format_exc()}")
                    _nlp_ja_load_failed = True
                    _nlp_ja = None
    return _nlp_ja

def get_nlp_en() -> Optional[Any]:
    """
    英語spacyモデルを遅延ロードするアクセサ関数（スレッドセーフ）
    モデルが存在しない場合はNoneを返し、エラーをログに記録
    """
    global _nlp_en, _nlp_en_load_failed
    # 以前のロード失敗をチェック（リトライを防ぐ）
    if _nlp_en_load_failed:
        return None
    # ダブルチェックロッキングパターンを使用
    if _nlp_en is None:
        with _nlp_en_lock:
            # ロック取得後に再度チェック（他のスレッドが既にロードした可能性があるため）
            if _nlp_en is None and not _nlp_en_load_failed:
                try:
                    _nlp_en = spacy.load("en_core_web_md")
                    logging.info("Successfully loaded English spacy model: en_core_web_md")
                except OSError as e:
                    logging.error(f"Failed to load English spacy model 'en_core_web_md': {e}")
                    logging.error("Please install the model with: python -m spacy download en_core_web_md")
                    _nlp_en_load_failed = True
                    _nlp_en = None
                except Exception as e:
                    logging.error(f"Unexpected error loading English spacy model: {e}")
                    logging.error(f"Traceback:\n{traceback.format_exc()}")
                    _nlp_en_load_failed = True
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

def ids_to_tokens(ids: List[int], vocabulary: Any) -> List[str]:
    """IDのリストをトークンのリストに変換する。

    Args:
        ids: トークンIDのリスト。
        vocabulary: Vocabularyクラスのインスタンスまたは辞書（token -> idのマッピング）。

    Returns:
        トークンのリスト。存在しないIDの場合は'<unk>'を返す。
    """
    # vocabularyがVocabularyクラスのインスタンスである場合（弱参照可能）
    if hasattr(vocabulary, 'id2token'):
        # 安全なルックアップを使用（存在しないIDの場合は'<unk>'を返す）
        return [vocabulary.id2token.get(id, '<unk>') for id in ids]
    # vocabularyが辞書の場合（語彙ファイルからロードした場合などに発生）
    else:
        # 辞書の場合は毎回逆マッピングを構築（キャッシュは呼び出し側で管理）
        id_to_token = {v: k for k, v in vocabulary.items()}
        return [id_to_token.get(id, '<unk>') for id in ids]
