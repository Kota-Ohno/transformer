"""
損失関数の実装
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class LabelSmoothingCrossEntropyLoss(nn.Module):
    """Label Smoothing付きCrossEntropyLoss
    
    "Attention Is All You Need"論文で提案されたラベルスムージングを実装。
    過学習を防ぎ、モデルの汎化性能を向上させる。
    
    Args:
        smoothing: ラベルスムージングの係数（0.0 = スムージングなし、1.0 = 最大スムージング）
        ignore_index: 無視するインデックス（パディングトークンなど）
        reduction: 損失の縮約方法（'mean', 'sum', 'none'）
    """
    
    def __init__(
        self,
        smoothing: float = 0.1,
        ignore_index: int = -100,
        reduction: str = "mean"
    ):
        super().__init__()
        self.smoothing = smoothing
        self.ignore_index = ignore_index
        self.reduction = reduction
        
        if not 0.0 <= smoothing <= 1.0:
            raise ValueError(f"smoothing must be between 0.0 and 1.0, got {smoothing}")
        
        if reduction not in ["mean", "sum", "none"]:
            raise ValueError(f"reduction must be 'mean', 'sum', or 'none', got {reduction}")
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: モデルの出力 [batch_size, vocab_size] または [batch_size, seq_len, vocab_size]
            targets: 正解ラベル [batch_size] または [batch_size, seq_len]
            
        Returns:
            損失値（スカラーまたはバッチごとの損失）
        """
        # 入力の形状を保存
        original_shape = logits.shape
        
        # 3次元テンソルの場合は2次元にフラット化
        if logits.dim() == 3:
            batch_size, seq_len, vocab_size = logits.shape
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1)
        else:
            vocab_size = logits.shape[-1]
        
        # ignore_indexを持つ要素をマスク
        mask = targets != self.ignore_index
        
        if not mask.any():
            # 全てignore_indexの場合は0を返す
            return torch.tensor(0.0, device=logits.device, requires_grad=True)
        
        # マスクを適用
        logits = logits[mask]
        targets = targets[mask]
        
        # スムージング後の確率分布を計算
        # 正解ラベル: 1 - smoothing + smoothing / vocab_size
        # その他のラベル: smoothing / vocab_size
        n_classes = vocab_size
        log_probs = F.log_softmax(logits, dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (n_classes - 1))
            true_dist.scatter_(1, targets.unsqueeze(1), 1.0 - self.smoothing)
        
        # KLダイバージェンスを計算
        loss = F.kl_div(log_probs, true_dist, reduction='none').sum(dim=-1)
        
        # 縮約
        if self.reduction == "mean":
            return loss.mean()
        elif self.reduction == "sum":
            return loss.sum()
        else:  # "none"
            return loss


class MLMLoss(nn.Module):
    """MLM用の損失関数
    
    マスクされたトークンのみを対象とした損失計算
    """
    
    def __init__(
        self,
        vocab_size: int,
        ignore_index: int = -100,
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.ignore_index = ignore_index
        
        if label_smoothing > 0:
            self.criterion = LabelSmoothingCrossEntropyLoss(
                smoothing=label_smoothing,
                ignore_index=ignore_index,
                reduction="mean"
            )
        else:
            self.criterion = nn.CrossEntropyLoss(
                ignore_index=ignore_index,
                reduction="mean"
            )
    
    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            logits: モデルの出力 [batch_size, seq_len, vocab_size]
            labels: 正解ラベル [batch_size, seq_len]（-100の位置は無視される）
            
        Returns:
            損失値（スカラー）
        """
        # CrossEntropyLossは[batch_size * seq_len, vocab_size]と[batch_size * seq_len]を期待
        batch_size, seq_len, vocab_size = logits.shape
        logits_flat = logits.view(-1, vocab_size)
        labels_flat = labels.view(-1)
        
        return self.criterion(logits_flat, labels_flat)


def create_loss_function(
    loss_type: str = "cross_entropy",
    vocab_size: Optional[int] = None,
    label_smoothing: float = 0.0,
    ignore_index: int = -100
) -> nn.Module:
    """損失関数を作成するファクトリ関数
    
    Args:
        loss_type: 損失関数のタイプ ("cross_entropy", "label_smoothing", "mlm")
        vocab_size: 語彙サイズ（MLM損失の場合に必要）
        label_smoothing: ラベルスムージング係数
        ignore_index: 無視するインデックス
        
    Returns:
        損失関数モジュール
    """
    if loss_type == "cross_entropy":
        return nn.CrossEntropyLoss(ignore_index=ignore_index)
    
    elif loss_type == "label_smoothing":
        return LabelSmoothingCrossEntropyLoss(
            smoothing=label_smoothing,
            ignore_index=ignore_index
        )
    
    elif loss_type == "mlm":
        if vocab_size is None:
            raise ValueError("vocab_size is required for MLM loss")
        return MLMLoss(
            vocab_size=vocab_size,
            ignore_index=ignore_index,
            label_smoothing=label_smoothing
        )
    
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
