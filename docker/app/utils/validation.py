"""
トークンID検証用のユーティリティ関数
"""
import torch
from typing import Tuple
from utils.config import CONFIG


def validate_token_ids(
    x: torch.Tensor,
    vocab_size: int,
    tensor_name: str = "input"
) -> None:
    """
    トークンIDテンソルの型と範囲を検証します。

    Args:
        x: 検証するトークンIDテンソル [batch_size, seq_len]
        vocab_size: 語彙サイズ（有効なID範囲は[0, vocab_size-1]）
        tensor_name: エラーメッセージで使用するテンソル名

    Raises:
        TypeError: テンソルが整数型でない場合
        ValueError: トークンIDが有効範囲外の場合
    """
    # 整数型チェック
    integer_dtypes = {torch.int8, torch.int16, torch.int32, torch.int64, torch.long}
    if x.dtype not in integer_dtypes:
        raise TypeError(
            f"Expected integer dtype for {tensor_name} token indices, but got {x.dtype}. "
            f"Input shape: {x.shape}"
        )

    # 範囲チェック（GPU-CPU同期を避けるため、デバッグモードでのみ詳細チェック）
    if CONFIG.training_config.debug_mode:
        # デバッグモード: 詳細な範囲チェック（GPU-CPU同期あり）
        x_min = x.min().item()
        x_max = x.max().item()
        valid_min = 0
        valid_max = vocab_size - 1

        if x_min < valid_min or x_max > valid_max:
            raise ValueError(
                f"Token indices in {tensor_name} out of valid range [0, {valid_max}]. "
                f"Found range: [{x_min}, {x_max}]. "
                f"Input shape: {x.shape}, vocab_size: {vocab_size}"
            )
    else:
        # 本番モード: GPU側のブールチェック
        # 意図: 本番環境でのGPU→CPU同期を最小化するため、単一のis_valid.item()チェックのみを実行
        # 注意: is_valid.item()は毎回GPU→CPU同期を強制する
        is_valid = torch.all((x >= 0) & (x < vocab_size))
        if not is_valid.item():
            # エラー時のみ詳細情報を取得
            # 注意: 以下のtensor.item()/cpu()呼び出しは、エラーブランチが実行された場合のみ追加のGPU→CPU同期を発生させる
            x_min = x.min().item()
            x_max = x.max().item()

            # 無効な値の詳細情報を取得（デバッグ用）
            # 注意: 以下の操作はエラー時のみ実行され、追加のGPU→CPU同期を発生させる
            invalid_mask = (x < 0) | (x >= vocab_size)
            invalid_indices = torch.nonzero(invalid_mask, as_tuple=False)
            invalid_values = x[invalid_mask]
            num_invalid = invalid_mask.sum().item()

            # 最初の数個の無効なインデックスと値を取得（デバッグ用）
            sample_indices = invalid_indices[:10].cpu().tolist()
            sample_values = invalid_values[:10].cpu().tolist()

            raise ValueError(
                f"Invalid token IDs detected in {tensor_name}: {num_invalid} out-of-range values found. "
                f"Valid range is [0, {vocab_size - 1}], "
                f"but found values in range [{x_min}, {x_max}]. "
                f"Input shape: {x.shape}, vocab_size: {vocab_size}. "
                f"Sample invalid positions (batch_idx, seq_idx): {sample_indices}, "
                f"with values: {sample_values}"
            )
