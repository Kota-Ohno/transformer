import torch
import nltk
import os
import logging
from typing import Dict, Any

# 評価指標用のダウンロード
def download_nltk_resources():
    """必要なnltkリソースをダウンロードします。環境変数でスキップ可能。"""
    # osモジュールが関数内でローカル変数として宣言される前にアクセスされているので、
    # このインポートが必要です
    import os

    # 環境変数によるスキップ
    if os.environ.get('SKIP_NLTK_DOWNLOAD') == '1':
        logging.info("環境変数の設定によりNLTKリソースのダウンロードをスキップします")
        return

    try:
        import nltk

        # カスタムダウンロードディレクトリを設定（Docker環境に依存しない場所）
        nltk_data_dir = os.path.join(os.getcwd(), "nltk_data")
        os.makedirs(nltk_data_dir, exist_ok=True)

        # nltk.data.pathの先頭にカスタムディレクトリを追加
        nltk.data.path.insert(0, nltk_data_dir)

        # 必須のリソース
        resources = ['punkt']

        # 直接ダウンロード（すでに存在するかどうかはdownload関数が内部でチェックする）
        for resource in resources:
            try:
                logging.info(f"nltk resource {resource} をダウンロードしています...")
                nltk.download(resource, download_dir=nltk_data_dir, quiet=True)
                logging.info(f"nltk resource {resource} のダウンロードが完了しました")
            except Exception as e:
                logging.error(f"{resource} のダウンロード中にエラー: {e}")
                # ダウンロードに失敗しても続行を試みる

    except Exception as e:
        logging.error(f"nltkリソースのダウンロード中にエラーが発生しました: {e}")

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
    tgt_sub_mask = torch.tril(torch.ones((tgt_len, tgt_len), device=tgt.device)).bool()

    # 両方のマスクを結合（パディングマスクの形状を調整）
    # tgt_pad_mask: [batch_size, 1, 1, tgt_len] -> [batch_size, 1, tgt_len, tgt_len]
    tgt_pad_mask_expanded = tgt_pad_mask.squeeze(2).expand(-1, -1, tgt_len, -1)
    tgt_mask = tgt_pad_mask_expanded & tgt_sub_mask.unsqueeze(0).unsqueeze(0)

    return tgt_mask
# --- ここまでマスク生成関数 ---


def convert_ids_to_text(ids: Any, id2word: Dict[int, str], skip_special: bool = False) -> str:
    """
    トークンIDを文字列に変換します

    Args:
        ids: 変換するIDのリスト（torch.Tensorまたはlist）
        id2word: ID→単語の辞書
        skip_special: 特殊トークンをスキップするかどうか

    Returns:
        変換されたテキスト
    """
    try:
        # テンソルの場合はリストに変換
        if isinstance(ids, torch.Tensor):
            ids = ids.cpu().tolist()

        # IDから単語に変換
        special_tokens = {'<pad>', '<unk>', '<s>', '</s>', '<bos>', '<eos>'}
        words = []
        for idx in ids:
            # 辞書にない場合はスキップ
            if idx not in id2word:
                continue
            word = id2word[idx]
            # 特殊トークンをスキップする場合
            if skip_special and word in special_tokens:
                continue
            words.append(word)

        # 単語を連結して文字列にして返す
        return ' '.join(words)
    except Exception as e:
        logging.error(f"テキスト変換エラー: {e}")
        return ""
