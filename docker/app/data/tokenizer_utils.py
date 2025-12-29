import os
import logging
import sentencepiece as spm
import re
from data.data import train_sentencepiece

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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
    print("ソース言語のSentencePieceモデルをトレーニング中...")
    train_sentencepiece(train_texts_src, src_model_prefix)

    print("ターゲット言語のSentencePieceモデルをトレーニング中...")
    train_sentencepiece(train_texts_tgt, tgt_model_prefix)

    # モデルをロード
    sp_src = spm.SentencePieceProcessor()
    sp_tgt = spm.SentencePieceProcessor()

    try:
        sp_src.load(f"{src_model_prefix}.model")
        sp_tgt.load(f"{tgt_model_prefix}.model")
    except Exception as e:
        logging.error(f"SentencePieceモデルのロード中にエラーが発生しました: {e}")
        raise

    return sp_src, sp_tgt

# テキスト正規化
def normalize_text(text, lang):
    """基本的なテキスト正規化を行います"""
    # 小文字化（英語のみ）
    if lang == "en_US":
        text = text.lower()

    # 空白の正規化
    text = re.sub(r'\s+', ' ', text)

    # 数字の正規化
    text = re.sub(r'\d+', '0', text)

    # 句読点の周囲に空白を追加（英語のみ）
    if lang == "en_US":
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # 再度空白を正規化

    return text.strip()

# sentencepieceモデルを使ったトークナイズ
def tokenize_with_sentencepiece(text, sp_model, lang=None):
    """
    sentencepieceモデルを使ってテキストをトークナイズします。

    Args:
        text (str): トークナイズするテキスト
        sp_model: sentencepieceモデル
        lang (str, optional): 言語（正規化に使用）

    Returns:
        list: トークンのリスト
    """
    if lang:
        text = normalize_text(text, lang)
    return sp_model.encode_as_ids(text)