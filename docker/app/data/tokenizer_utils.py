import os
import logging
import sentencepiece as spm
import re
from data.data import train_sentencepiece

# モジュールスコープのロガーを作成
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def train_and_load_sp_models(train_texts_src, train_texts_tgt):
    """
    ソースとターゲットのSentencePieceモデルをトレーニングして読み込みます。

    Args:
        train_texts_src (list): ソース言語のトレーニングテキスト
        train_texts_tgt (list): ターゲット言語のトレーニングテキスト

    Returns:
        tuple: (ソースモデル, ターゲットモデル)
    """
    # モデルパス
    src_model_prefix = os.path.join("models", "sp_src")
    tgt_model_prefix = os.path.join("models", "sp_tgt")

    # models ディレクトリがない場合は作成
    os.makedirs("models", exist_ok=True)

    # モデルをトレーニング
    logger.info("ソース言語のSentencePieceモデルをトレーニング中...")
    try:
        train_sentencepiece(train_texts_src, src_model_prefix)
    except Exception as err:
        logger.error(f"ソース言語のSentencePieceモデルのトレーニング中にエラーが発生しました (モデルパス: {src_model_prefix}): {err}")
        raise

    logger.info("ターゲット言語のSentencePieceモデルをトレーニング中...")
    try:
        train_sentencepiece(train_texts_tgt, tgt_model_prefix)
    except Exception as err:
        logger.error(f"ターゲット言語のSentencePieceモデルのトレーニング中にエラーが発生しました (モデルパス: {tgt_model_prefix}): {err}")
        raise

    # モデルをロード
    sp_src = spm.SentencePieceProcessor()
    sp_tgt = spm.SentencePieceProcessor()

    try:
        sp_src.load(f"{src_model_prefix}.model")
        sp_tgt.load(f"{tgt_model_prefix}.model")
    except Exception as e:
        logger.error(f"SentencePieceモデルのロード中にエラーが発生しました: {e}")
        raise

    return sp_src, sp_tgt

# テキスト正規化
def normalize_text(text, lang, normalize_numeric='<NUM>'):
    """
    基本的なテキスト正規化を行います

    Args:
        text (str): 正規化するテキスト
        lang (str): 言語コード（'en_US' または 'ja_JP'）
        normalize_numeric (str|None|False): 数字の正規化方法
            - 文字列（デフォルト: '<NUM>'）: 数字列をその文字列に置き換え
            - None または False: 数字を置き換えない

    Returns:
        str: 正規化されたテキスト
    """
    # 小文字化（英語のみ）
    if lang == "en_US":
        text = text.lower()

    # 空白の正規化
    text = re.sub(r'\s+', ' ', text)

    # 数字の正規化
    if normalize_numeric is not None and normalize_numeric is not False:
        # 数字列を指定されたトークンに置き換え（デフォルト: '<NUM>'）
        text = re.sub(r'\d+', str(normalize_numeric), text)
    # normalize_numeric が None または False の場合は数字を置き換えない

    # 句読点の周囲に空白を追加（英語のみ）
    if lang == "en_US":
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # 再度空白を正規化

    return text.strip()

# sentencepieceモデルを使ったトークナイズ
def tokenize_with_sentencepiece(text, sp_model, lang=None, normalize_numeric='<NUM>'):
    """
    sentencepieceモデルを使ってテキストをトークナイズします。

    Args:
        text (str): トークナイズするテキスト
        sp_model: sentencepieceモデル
        lang (str, optional): 言語（正規化に使用）
        normalize_numeric (str|None|False, optional): 数字の正規化方法
            - 文字列（デフォルト: '<NUM>'）: 数字列をその文字列に置き換え
            - None または False: 数字を置き換えない

    Returns:
        list[int]: トークンIDのリスト
    """
    if lang:
        text = normalize_text(text, lang, normalize_numeric=normalize_numeric)
    return sp_model.encode_as_ids(text)
