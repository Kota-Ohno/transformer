"""
MLM用のマスキング戦略実装

BERT標準のマスキング戦略：
- 全トークンの15%を選択
- 選択されたトークンの80%を[MASK]に置き換え
- 10%をランダムなトークンに置き換え
- 10%はそのまま保持
"""
import torch
import random
from typing import Tuple, Optional


def create_mlm_mask(
    input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    mask_token_ratio: float = 0.8,
    random_token_ratio: float = 0.1,
    keep_token_ratio: float = 0.1,
    pad_token_id: int = 0,
    special_token_ids: Optional[list] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    MLM用のマスクを作成
    
    Args:
        input_ids: 入力トークンID [batch_size, seq_len]
        mask_token_id: [MASK]トークンのID
        vocab_size: 語彙サイズ
        mask_prob: マスク確率（デフォルト: 0.15）
        mask_token_ratio: [MASK]トークンに置き換える割合（デフォルト: 0.8）
        random_token_ratio: ランダムトークンに置き換える割合（デフォルト: 0.1）
        keep_token_ratio: そのまま保持する割合（デフォルト: 0.1）
        pad_token_id: パディングトークンのID（マスク対象外）
        special_token_ids: 特殊トークンのIDリスト（マスク対象外）
        
    Returns:
        masked_input_ids: マスク適用後の入力 [batch_size, seq_len]
        labels: 正解ラベル（マスクされていない位置は-100） [batch_size, seq_len]
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    # 特殊トークンのデフォルト設定
    if special_token_ids is None:
        special_token_ids = [pad_token_id]  # 最低限パディングは除外
    else:
        special_token_ids = list(special_token_ids) + [pad_token_id]
    
    # マスク対象かどうかのマスクを作成
    # 特殊トークンは除外
    special_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    for special_id in special_token_ids:
        special_mask |= (input_ids == special_id)
    
    # マスク確率に基づいてランダムに選択
    rand = torch.rand(batch_size, seq_len, device=device)
    mask_candidates = (rand < mask_prob) & ~special_mask
    
    # ラベルを作成（-100は損失計算時に無視される）
    labels = torch.where(
        mask_candidates,
        input_ids,
        torch.tensor(-100, device=device)
    )
    
    # マスク適用後の入力を作成
    masked_input_ids = input_ids.clone()
    
    # マスク候補の中で、さらに細分化
    # 80%を[MASK]に、10%をランダムに、10%はそのまま
    num_masks = mask_candidates.sum()
    
    if num_masks > 0:
        # マスク位置のインデックスを取得
        mask_indices = mask_candidates.nonzero(as_tuple=True)
        
        # ランダムな分割
        rand_for_split = torch.rand(num_masks, device=device)
        
        # [MASK]に置き換える位置（80%）
        mask_token_positions = rand_for_split < mask_token_ratio
        
        # ランダムトークンに置き換える位置（10%）
        random_token_positions = (rand_for_split >= mask_token_ratio) & \
                                 (rand_for_split < mask_token_ratio + random_token_ratio)
        
        # [MASK]トークンを適用
        if mask_token_positions.any():
            mask_positions = tuple(idx[mask_token_positions] for idx in mask_indices)
            masked_input_ids[mask_positions] = mask_token_id
        
        # ランダムトークンを適用
        if random_token_positions.any():
            random_positions = tuple(idx[random_token_positions] for idx in mask_indices)
            random_tokens = torch.randint(
                0, vocab_size, (random_token_positions.sum(),), device=device
            )
            masked_input_ids[random_positions] = random_tokens
        
        # 10%はそのまま保持（何もしない）
    
    return masked_input_ids, labels


def create_mlm_mask_numpy(
    input_ids: list,
    mask_token_id: int,
    vocab_size: int,
    mask_prob: float = 0.15,
    pad_token_id: int = 0,
    special_token_ids: Optional[list] = None
) -> Tuple[list, list]:
    """
    NumPy/Pythonリスト版のマスキング関数（非バッチ処理用）
    
    Args:
        input_ids: 入力トークンIDのリスト
        mask_token_id: [MASK]トークンのID
        vocab_size: 語彙サイズ
        mask_prob: マスク確率
        pad_token_id: パディングトークンのID
        special_token_ids: 特殊トークンのIDリスト
        
    Returns:
        masked_input_ids: マスク適用後の入力
        labels: 正解ラベル（マスクされていない位置は-100）
    """
    if special_token_ids is None:
        special_token_ids = {pad_token_id}
    else:
        special_token_ids = set(special_token_ids) | {pad_token_id}
    
    masked_input_ids = []
    labels = []
    
    for token_id in input_ids:
        if token_id in special_token_ids:
            # 特殊トークンはマスクしない
            masked_input_ids.append(token_id)
            labels.append(-100)
        elif random.random() < mask_prob:
            # 15%の確率でマスク対象
            labels.append(token_id)
            
            rand = random.random()
            if rand < 0.8:
                # 80%: [MASK]に置き換え
                masked_input_ids.append(mask_token_id)
            elif rand < 0.9:
                # 10%: ランダムなトークンに置き換え
                masked_input_ids.append(random.randint(0, vocab_size - 1))
            else:
                # 10%: そのまま保持
                masked_input_ids.append(token_id)
        else:
            # マスクしない
            masked_input_ids.append(token_id)
            labels.append(-100)
    
    return masked_input_ids, labels


def apply_mlm_mask_batch(
    batch_input_ids: torch.Tensor,
    mask_token_id: int,
    vocab_size: int,
    pad_token_id: int = 0,
    cls_token_id: Optional[int] = None,
    sep_token_id: Optional[int] = None,
    mask_prob: float = 0.15
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    バッチ処理用のMLMマスキング
    
    BERTの設定に従って、CLSとSEPトークンもマスク対象外にする
    
    Args:
        batch_input_ids: バッチ入力 [batch_size, seq_len]
        mask_token_id: [MASK]トークンのID
        vocab_size: 語彙サイズ
        pad_token_id: パディングトークンのID
        cls_token_id: [CLS]トークンのID（オプション）
        sep_token_id: [SEP]トークンのID（オプション）
        mask_prob: マスク確率
        
    Returns:
        masked_inputs: マスク適用後の入力
        labels: 正解ラベル
    """
    # 特殊トークンを収集
    special_ids = [pad_token_id]
    if cls_token_id is not None:
        special_ids.append(cls_token_id)
    if sep_token_id is not None:
        special_ids.append(sep_token_id)
    
    return create_mlm_mask(
        batch_input_ids,
        mask_token_id,
        vocab_size,
        mask_prob=mask_prob,
        pad_token_id=pad_token_id,
        special_token_ids=special_ids
    )
