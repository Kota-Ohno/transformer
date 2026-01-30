import re
from typing import List, Union, Optional
import logging

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

    # イテラブルでない場合はエラー（文字列は特別に扱うため、このチェックの後で変換）
    if not hasattr(texts, "__iter__"):
        raise ValueError(
            f"{param_name} はイテラブル（リストなど）である必要があります。"
            f"現在の型: {type(texts).__name__}。"
            f"文字列のリストを渡してください。例: ['text1', 'text2', ...]"
        )

    # 単一の文字列をリストに変換
    if isinstance(texts, str):
        texts = [texts]

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
    # replacement_token: 真偽値 True のみ '<NUM>'、非空の str のみカスタムトークンとして使用
    if normalize_numeric is True:
        replacement_token = '<NUM>'
    elif isinstance(normalize_numeric, str) and normalize_numeric != '':
        replacement_token = normalize_numeric
    elif normalize_numeric is None or normalize_numeric is False:
        replacement_token = None
    else:
        # 数値0など未対応の型はカスタムトークンとして扱わない
        replacement_token = None

    if replacement_token is not None:
        text = re.sub(r'\d+', replacement_token, text)

    # 句読点の周囲に空白を追加（英語のみ）
    if lang == "en_US":
        text = re.sub(r'([.,!?;:])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # 再度空白を正規化

    return text.strip()
