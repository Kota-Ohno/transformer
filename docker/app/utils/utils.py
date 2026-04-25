import torch
import nltk
import os
import logging
import traceback
from typing import Dict, Any, Set, Optional, Union, List

# constantsモジュールの安全なインポート
_constants_available = False
_constants_import_error: Optional[ImportError] = None
_constants_warning_logged = False

try:
    from utils import constants
    _constants_available = True
except ImportError as e:
    _constants_available = False
    _constants_import_error = e

def _maybe_log_constants_import_failure() -> None:
    """constantsモジュールのインポート失敗をロギング（初回のみ、ロギング設定後に呼び出す）"""
    global _constants_warning_logged
    if not _constants_available and not _constants_warning_logged and _constants_import_error is not None:
        logging.warning(
            f"constantsモジュールのインポートに失敗しました。デフォルト値を使用します。"
            f"エラー詳細: {_constants_import_error}"
        )
        _constants_warning_logged = True

# 評価指標用のダウンロード
def download_nltk_resources(
    all_resources: Optional[Set[str]] = None,
    critical_resources: Optional[Set[str]] = None,
) -> None:
    """必要なnltkリソースをダウンロードします。環境変数でスキップ可能。

    Args:
        all_resources: ダウンロードするすべてのリソースのセット。Noneの場合は
            critical_resourcesを使用します。デフォルトはNone。
        critical_resources: 重要なリソースのセット。これらのダウンロードに失敗した場合は
            例外を再発生させます。デフォルトは{'punkt'}。

    Raises:
        Exception: 重要なリソースのダウンロードに失敗した場合。
    """
    # 環境変数によるスキップ
    if os.environ.get('SKIP_NLTK_DOWNLOAD') == '1':
        logging.info("環境変数の設定によりNLTKリソースのダウンロードをスキップします")
        return

    if critical_resources is None:
        critical_resources = {'punkt'}

    if all_resources is None:
        all_resources = critical_resources

    # エラーログの状態を追跡する変数（tryブロックの外で初期化）
    download_error_logged = False

    try:
        # 直接ダウンロード（すでに存在するかどうかはdownload関数が内部でチェックする）
        # カスタムダウンロードディレクトリを設定（Docker環境に依存しない場所）
        # 環境変数NLTK_DATA_DIRから読み取る、フォールバックとしてutils.pyのディレクトリ + "nltk_data"
        nltk_data_dir = os.environ.get(
            'NLTK_DATA_DIR',
            os.path.join(os.path.dirname(__file__), "..", "nltk_data")
        )
        # 相対パスの場合は正規化
        nltk_data_dir = os.path.abspath(nltk_data_dir)
        os.makedirs(nltk_data_dir, exist_ok=True)

        # nltk.data.pathの先頭にカスタムディレクトリを追加（idempotent）
        if nltk_data_dir not in nltk.data.path:
            nltk.data.path.insert(0, nltk_data_dir)
        for resource in all_resources:
            try:
                logging.info(f"nltk resource {resource} をダウンロードしています...")
                download_result = nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                if not download_result:
                    error_msg = f"nltk resource '{resource}' のダウンロードに失敗しました"
                    logging.error(error_msg)
                    if resource in critical_resources:
                        raise RuntimeError(error_msg)
                    else:
                        logging.warning(f"非クリティカルリソース '{resource}' のダウンロード失敗を無視します")
                else:
                    logging.info(f"nltk resource {resource} のダウンロードが完了しました")
            except Exception as e:
                exception_traceback = traceback.format_exc()
                if resource in critical_resources:
                    # クリティカルリソースの場合はエラーレベルでログに記録
                    logging.error(
                        f"クリティカルなnltk resource '{resource}' のダウンロードに失敗しました: {e}\n"
                        f"完全な例外情報:\n{exception_traceback}"
                    )
                    download_error_logged = True
                    raise
                else:
                    # 非クリティカルリソースの場合は警告レベルでログに記録
                    logging.warning(
                        f"nltk resource '{resource}' のダウンロードに失敗しました: {e}\n"
                        f"完全な例外情報:\n{exception_traceback}"
                    )
    except Exception as e:
        # 既にログに記録されている場合は重複ログを避ける
        if not download_error_logged:
            exception_traceback = traceback.format_exc()
            logging.error(
                f"NLTKリソースのダウンロード処理中に予期しないエラーが発生しました: {e}\n"
                f"完全な例外情報:\n{exception_traceback}"
            )
        raise

# --- マスク生成関数 ---
def create_padding_mask(seq: torch.Tensor, pad_idx: int) -> torch.Tensor:
    """
    パディングトークンを無視するためのマスクを作成

    Args:
        seq: 入力シーケンス [batch_size, seq_len]
        pad_idx: パディングトークンのインデックス

    Returns:
        パディングマスク [batch_size, 1, 1, seq_len]
        値が1の箇所が注意を向ける場所、0の箇所は無視される
    """
    # パディングではない場所を1、パディングの場所を0にする
    mask = (seq != pad_idx).float().unsqueeze(1).unsqueeze(2)
    return mask

def create_subsequent_mask(seq: torch.Tensor) -> torch.Tensor:
    """
    デコーダーの自己注意用の因果的マスクを作成

    Args:
        seq: 入力シーケンス [batch_size, seq_len]

    Returns:
        後続マスク [batch_size, 1, seq_len, seq_len]
        下三角行列で、値が1の箇所が注意を向ける場所、0の箇所は無視される
    """
    seq_len = seq.size(1)

    # 下三角行列を作成 (対角成分を含む)
    mask = torch.tril(torch.ones((seq_len, seq_len), device=seq.device))

    # バッチ次元を追加 [batch_size, 1, seq_len, seq_len]
    mask = mask.unsqueeze(0).unsqueeze(1).expand(seq.size(0), 1, seq_len, seq_len)

    return mask

def create_src_mask(src: torch.Tensor, src_pad_idx: int) -> torch.Tensor:
    """
    ソースシーケンス用のマスクを作成（エンコーダー用）

    Args:
        src: ソースシーケンス [batch_size, src_len]
        src_pad_idx: ソースパディングトークンのインデックス

    Returns:
        ソースマスク [batch_size, 1, 1, src_len]
    """
    return create_padding_mask(src, src_pad_idx)

def create_tgt_mask(tgt: torch.Tensor, tgt_pad_idx: int) -> torch.Tensor:
    """
    ターゲットシーケンス用のマスクを作成（デコーダー用）
    パディングマスクと因果的マスクを組み合わせたもの

    Args:
        tgt: ターゲットシーケンス [batch_size, tgt_len]
        tgt_pad_idx: ターゲットパディングトークンのインデックス

    Returns:
        ターゲットマスク [batch_size, 1, tgt_len, tgt_len]
    """
    # パディングマスクを作成
    tgt_pad_mask = create_padding_mask(tgt, tgt_pad_idx)

    # 因果的マスクを作成
    tgt_len = tgt.shape[1]
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device, dtype=tgt_pad_mask.dtype))

    # 両方のマスクを結合（パディングマスクの形状を調整）
    # tgt_pad_mask: [batch_size, 1, 1, tgt_len] -> [batch_size, 1, tgt_len, tgt_len]
    # dim=2 を tgt_len に拡張
    tgt_pad_mask_expanded = tgt_pad_mask.expand(-1, -1, tgt_len, -1)
    tgt_mask = tgt_pad_mask_expanded * tgt_sub_mask.unsqueeze(0).unsqueeze(0)

    return tgt_mask
# --- ここまでマスク生成関数 ---


def convert_ids_to_text(ids: Any, id2word: Dict[int, str], skip_special: bool = False) -> Union[str, List[str]]:
    """
    トークンIDを文字列に変換します

    Args:
        ids: 変換するIDのリスト（torch.Tensorまたはlist）
        id2word: ID→単語の辞書
        skip_special: 特殊トークンをスキップするかどうか

    Returns:
        変換されたテキスト。1次元の場合は文字列、2次元（バッチ）の場合は文字列のリスト
    """
    _maybe_log_constants_import_failure()
    try:
        # テンソルの場合はリストに変換
        is_batch = False
        if isinstance(ids, torch.Tensor):
            if ids.ndim == 1:
                # 1次元テンソル: そのままリストに変換
                ids = ids.cpu().tolist()
            elif ids.ndim == 2:
                # 2次元テンソル: バッチ構造を保持
                ids_2d = ids.cpu().tolist()
                ids = ids_2d  # リストのリストとして保持
                is_batch = True
            else:
                # 3次元以上のテンソル: 予期しない次元数
                raise ValueError(
                    f"予期しないテンソルの次元数: {ids.ndim}。"
                    f"1次元（単一シーケンス）または2次元（バッチ）のみサポートされています。"
                )
        elif isinstance(ids, list) and len(ids) > 0 and isinstance(ids[0], list):
            # リストのリスト（2D構造）の場合: バッチ構造を保持
            is_batch = True

        # バッチ処理の場合
        if is_batch:
            # 各シーケンスを個別に処理
            result_strings = []
            for sequence_ids in ids:
                # IDから単語に変換
                words = []
                for idx in sequence_ids:
                    # 辞書に存在するかチェックしてからアクセス（KeyErrorを回避）
                    try:
                        word = id2word[idx]
                    except KeyError:
                        # 辞書にない場合はスキップ
                        continue
                    # 特殊トークンをスキップする場合
                    if skip_special:
                        # constantsモジュールが利用可能でSPECIAL_TOKENS属性が存在する場合のみチェック
                        if _constants_available and hasattr(constants, 'SPECIAL_TOKENS'):
                            if word in constants.SPECIAL_TOKENS:
                                continue
                        else:
                            # フォールバック: 一般的な特殊トークンをチェック
                            if word in {'<pad>', '<unk>', '<s>', '</s>'}:
                                continue
                    words.append(word)
                # 単語を連結して文字列に変換
                result_strings.append(' '.join(words))
            return result_strings

        # 単一シーケンス処理の場合
        # IDから単語に変換
        words = []
        for idx in ids:
            # 辞書に存在するかチェックしてからアクセス（KeyErrorを回避）
            try:
                word = id2word[idx]
            except KeyError:
                # 辞書にない場合はスキップ
                continue
            # 特殊トークンをスキップする場合
            if skip_special:
                # constantsモジュールが利用可能でSPECIAL_TOKENS属性が存在する場合のみチェック
                if _constants_available and hasattr(constants, 'SPECIAL_TOKENS'):
                    if word in constants.SPECIAL_TOKENS:
                        continue
                else:
                    # フォールバック: 一般的な特殊トークンをチェック
                    if word in {'<pad>', '<unk>', '<s>', '</s>'}:
                        continue
            words.append(word)

        # 単語を連結して文字列にして返す
        return ' '.join(words)
    except (TypeError, AttributeError) as e:
        # プログラミングエラーは再発生させる
        logging.exception("テキスト変換エラー（プログラミングエラー）")
        raise
    except ValueError as e:
        # ValueError（無効なテンソル次元など）は再発生させる
        logging.exception("テキスト変換エラー（値エラー）")
        raise
    except Exception as e:
        # その他の予期しないエラーはトレースバックをログに記録して再発生
        logging.exception("テキスト変換エラー")
        raise
