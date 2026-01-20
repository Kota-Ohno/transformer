import os
import logging
import glob
import sentencepiece as spm
import re
from typing import List, Union, Tuple, Optional, Sequence
from data.data import train_sentencepiece

# モジュールスコープのロガーを作成
logger = logging.getLogger(__name__)


def _validate_and_filter_texts(
    texts: Union[str, List[str]], param_name: str
) -> List[str]:
    """
    テキストリストを検証してフィルタリングします。

    Args:
        texts: 検証するテキスト（文字列またはリスト）
        param_name: パラメータ名（エラーメッセージ用）

    Returns:
        list[str]: フィルタリングされた文字列のリスト

    Raises:
        ValueError: 入力が無効な場合
    """
    if texts is None:
        raise ValueError(
            f"{param_name} は None であってはなりません。"
            f"文字列のリストを渡してください。例: ['text1', 'text2', ...]"
        )

    # 単一の文字列をリストに変換
    if isinstance(texts, str):
        texts = [texts]

    # イテラブルでない場合はエラー
    if not hasattr(texts, "__iter__"):
        raise ValueError(
            f"{param_name} はイテラブル（リストなど）である必要があります。"
            f"現在の型: {type(texts).__name__}。"
            f"文字列のリストを渡してください。例: ['text1', 'text2', ...]"
        )

    # リストに変換して要素を確認
    texts = list(texts)

    # 各要素が文字列であることを確認し、空文字列や空白のみの文字列をフィルタリング
    filtered = []
    for i, text in enumerate(texts):
        if not isinstance(text, str):
            raise ValueError(
                f"{param_name} の要素はすべて文字列である必要があります。"
                f"インデックス {i} の要素の型: {type(text).__name__}。"
                f"文字列のリストを渡してください。例: ['text1', 'text2', ...]"
            )
        if text.strip():
            filtered.append(text)

    # フィルタリング後も空でないことを確認
    if not filtered:
        raise ValueError(
            f"{param_name} は空であってはなりません。"
            f"また、空白のみの文字列は無視されます。"
            f"少なくとも1つ以上の非空白文字を含む文字列が必要です。"
        )

    return filtered


def train_and_load_sp_models(
    train_texts_src: Union[str, List[str]],
    train_texts_tgt: Union[str, List[str]],
    save_path_src: Optional[str] = None,
    save_path_tgt: Optional[str] = None,
) -> Tuple[spm.SentencePieceProcessor, spm.SentencePieceProcessor]:
    """
    ソースとターゲットのSentencePieceモデルをトレーニングして読み込みます。

    Args:
        train_texts_src: ソース言語のトレーニングテキスト（文字列またはリスト）
        train_texts_tgt: ターゲット言語のトレーニングテキスト（文字列またはリスト）
        save_path_src: ソースモデルの保存パス（.model拡張子なし）。Noneの場合はデフォルトパスを使用。
        save_path_tgt: ターゲットモデルの保存パス（.model拡張子なし）。Noneの場合はデフォルトパスを使用。

    Returns:
        tuple: (ソースモデル, ターゲットモデル)

    Raises:
        ValueError: 入力パラメータが無効な場合（None、空、または文字列のリストでない場合）
        TypeError: save_path_srcまたはsave_path_tgtがNoneでもstrでもない場合
    """
    # パスパラメータの型チェック
    if save_path_src is not None and not isinstance(save_path_src, str):
        raise TypeError(f"save_path_src must be a str or None, got {type(save_path_src).__name__}")
    if save_path_tgt is not None and not isinstance(save_path_tgt, str):
        raise TypeError(f"save_path_tgt must be a str or None, got {type(save_path_tgt).__name__}")

    # 入力検証
    train_texts_src = _validate_and_filter_texts(train_texts_src, "train_texts_src")
    train_texts_tgt = _validate_and_filter_texts(train_texts_tgt, "train_texts_tgt")

    # モデルパス（保存パスが指定されている場合はそれを使用、そうでない場合はデフォルト）
    if save_path_src:
        # .model拡張子を削除（train_sentencepieceが追加するため）
        if save_path_src.endswith(".model"):
            src_model_prefix = save_path_src[:-len(".model")]
        else:
            src_model_prefix = save_path_src
    else:
        src_model_prefix = os.path.join("models", "sp_src")

    if save_path_tgt:
        # .model拡張子を削除（train_sentencepieceが追加するため）
        if save_path_tgt.endswith(".model"):
            tgt_model_prefix = save_path_tgt[:-len(".model")]
        else:
            tgt_model_prefix = save_path_tgt
    else:
        tgt_model_prefix = os.path.join("models", "sp_tgt")

    # 各プレフィックスのディレクトリを取得し、存在しない場合は作成
    src_dir = os.path.dirname(src_model_prefix) or "."
    tgt_dir = os.path.dirname(tgt_model_prefix) or "."
    if src_dir != ".":
        os.makedirs(src_dir, exist_ok=True)
    if tgt_dir != ".":
        os.makedirs(tgt_dir, exist_ok=True)

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
        # ソースモデルが既に作成されている場合、部分的なアーティファクトを削除
        src_files = glob.glob(f"{glob.escape(src_model_prefix)}*")
        if src_files:
            logger.info(f"部分的なソースモデルのアーティファクトを削除中: {src_files}")
            for file_path in src_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"削除しました: {file_path}")
                except OSError as e:
                    logger.warning(f"ファイルの削除に失敗しました ({file_path}): {e}")
        # ターゲットモデル側の部分的なアーティファクトも削除
        tgt_files = glob.glob(f"{glob.escape(tgt_model_prefix)}*")
        if tgt_files:
            logger.info(f"部分的なターゲットモデルのアーティファクトを削除中: {tgt_files}")
            for file_path in tgt_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"削除しました: {file_path}")
                except OSError as e:
                    logger.warning(f"ファイルの削除に失敗しました ({file_path}): {e}")
        raise

    # モデルをロード
    sp_src = spm.SentencePieceProcessor()
    sp_tgt = spm.SentencePieceProcessor()

    try:
        sp_src.load(f"{src_model_prefix}.model")
        sp_tgt.load(f"{tgt_model_prefix}.model")
    except Exception as e:
        logger.error(f"SentencePieceモデルのロード中にエラーが発生しました: {e}")
        # 生成されたアーティファクトを削除（トレーニング失敗時と同様のクリーンアップ）
        src_files = glob.glob(f"{glob.escape(src_model_prefix)}*")
        tgt_files = glob.glob(f"{glob.escape(tgt_model_prefix)}*")
        all_files = src_files + tgt_files
        if all_files:
            logger.info(f"部分的なモデルのアーティファクトを削除中: {all_files}")
            for file_path in all_files:
                try:
                    os.remove(file_path)
                    logger.debug(f"削除しました: {file_path}")
                except OSError as os_err:
                    logger.warning(f"ファイルの削除に失敗しました ({file_path}): {os_err}")
        raise

    return sp_src, sp_tgt

# テキスト正規化
def normalize_text(text: str, lang: str, normalize_numeric: Optional[Union[str, bool]] = '<NUM>') -> str:
    """
    基本的なテキスト正規化を行います

    Args:
        text (str): 正規化するテキスト
        lang (str): 言語コード（'en_US' または 'ja_JP'）
        normalize_numeric: 数字の正規化方法。以下のいずれかを指定可能:
            - True: 数字列を文字列'<NUM>'に置き換える（normalize_numericパラメータがTrueの場合）
            - 文字列: 数字列をその文字列に置き換える（デフォルト: '<NUM>'）
            - None または False: 数字を置き換えない（数値正規化を無効化）
            - 空文字列 ('') : 数字を置き換えない（数値正規化を無効化）

    Returns:
        str: 正規化されたテキスト

    Raises:
        TypeError: textがNoneまたはstr型でない場合
    """
    # langパラメータの検証
    if lang not in ('en_US', 'ja_JP'):
        raise ValueError(
            f"langパラメータは'en_US'または'ja_JP'である必要があります。"
            f"現在の値: {lang}"
        )

    # textの検証
    if text is None:
        raise TypeError(
            "text は None であってはなりません。"
            "文字列を渡してください。"
        )

    if not isinstance(text, str):
        raise TypeError(
            f"text は str 型である必要があります。"
            f"現在の型: {type(text).__name__}。"
            f"文字列を渡してください。"
        )

    # 小文字化（英語のみ）
    if lang == "en_US":
        text = text.lower()

    # 空白の正規化
    text = re.sub(r'\s+', ' ', text)

    # 数字の正規化
    # replacement_tokenを決定: Trueの場合は'<NUM>'、文字列の場合はその文字列を使用
    if normalize_numeric is True:
        replacement_token = '<NUM>'
    elif normalize_numeric is not None and normalize_numeric is not False and normalize_numeric != '':
        replacement_token = str(normalize_numeric)
    else:
        # normalize_numeric が None または False または空文字列の場合は数字を置き換えない
        replacement_token = None

    if replacement_token is not None:
        text = re.sub(r'\d+', replacement_token, text)

    # 句読点の周囲に空白を追加（英語のみ）
    if lang == "en_US":
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # 再度空白を正規化

    return text.strip()

# sentencepieceモデルを使ったトークナイズ
def tokenize_with_sentencepiece(
    text: str,
    sp_model: spm.SentencePieceProcessor,
    lang: Optional[str] = None,
    normalize_numeric: Optional[Union[str, bool]] = '<NUM>'
) -> List[int]:
    """
    sentencepieceモデルを使ってテキストをトークナイズします。

    Args:
        text (str): トークナイズするテキスト
        sp_model: sentencepieceモデル
        lang (str, optional): 言語（正規化に使用）
        normalize_numeric: 数字の正規化方法。以下のいずれかを指定可能:
            - True: 数字列を文字列'<NUM>'に置き換える（normalize_numericパラメータがTrueの場合）
            - 文字列: 数字列をその文字列に置き換える（デフォルト: '<NUM>'）
            - None または False: 数字を置き換えない（数値正規化を無効化）
            - 空文字列 ('') : 数字を置き換えない（数値正規化を無効化）

    Returns:
        list[int]: トークンIDのリスト

    Raises:
        TypeError: sp_modelがNoneまたはencode_as_idsメソッドを持たない場合、
                   またはtextがNoneまたはstr型でない場合
        ValueError: langが指定されている場合、normalize_textに無効または
                   サポートされていないlangが渡された場合
    """
    # sp_modelの検証
    if sp_model is None:
        raise TypeError(
            "sp_model は None であってはなりません。"
            "SentencePieceProcessorインスタンスを渡してください。"
        )

    if not hasattr(sp_model, "encode_as_ids"):
        raise TypeError(
            f"sp_model は encode_as_ids メソッドを実装している必要があります。"
            f"現在の型: {type(sp_model).__name__}。"
            f"SentencePieceProcessorインスタンスを渡してください。"
        )

    # textの検証
    if text is None:
        raise TypeError(
            "text は None であってはなりません。"
            "文字列を渡してください。"
        )

    if not isinstance(text, str):
        raise TypeError(
            f"text は str 型である必要があります。"
            f"現在の型: {type(text).__name__}。"
            f"文字列を渡してください。"
        )

    if lang:
        text = normalize_text(text, lang, normalize_numeric=normalize_numeric)
    return sp_model.encode_as_ids(text)
