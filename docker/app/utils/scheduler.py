"""
学習率のスケジューラを定義します。
"""
import math
import torch

# PyTorch 2.0以降では LRScheduler がパブリックAPIとして利用可能
# 後方互換性のために _LRScheduler も確認
try:
    # PyTorch 2.0以降のパブリックAPIを試す
    _BaseScheduler = torch.optim.lr_scheduler.LRScheduler
except AttributeError:
    # 古いバージョンの場合はプライベートAPIを使用
    _BaseScheduler = torch.optim.lr_scheduler._LRScheduler

class WarmupScheduler(_BaseScheduler):
    """
    ウォームアップ付きの学習率スケジューラ
    """
    def __init__(self, optimizer, d_model, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        # Input validation
        if warmup_steps <= 0:
            raise ValueError(f"warmup_steps must be greater than 0, got {warmup_steps}")
        if total_steps <= 0:
            raise ValueError(f"total_steps must be greater than 0, got {total_steps}")
        if warmup_steps >= total_steps:
            raise ValueError(f"warmup_steps must be < total_steps, got {warmup_steps} and {total_steps}")
        if d_model <= 0:
            raise ValueError(f"d_model must be greater than 0, got {d_model}")
        if min_lr < 0:
            raise ValueError(f"min_lr must be non-negative, got {min_lr}")
        if last_epoch < -1:
            raise ValueError(f"last_epoch must be >= -1, got {last_epoch}")

        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = max(self.last_epoch + 1, 1)
        if step <= self.warmup_steps:
            # Noamスタイルのウォームアップ: d_modelが学習率に影響する
            # scale = d_model**-0.5 * min(step**-0.5, step * warmup_steps**-1.5)
            scale = (self.d_model ** -0.5) * min(
                step ** -0.5,
                step * (self.warmup_steps ** -1.5)
            )
            # 各グループのLRを計算: lr_i = base_lr * scale
            lrs = []
            for base_lr in self.base_lrs:
                lr_i = base_lr * scale
                lrs.append(lr_i)
            return lrs
        else:
            # warmup終了時のLRを計算（不連続性を避けるため）
            # step == warmup_steps のときのscale: (d_model * warmup_steps) ** -0.5
            warmup_end_scale = (self.d_model * self.warmup_steps) ** -0.5

            # warmup_steps < total_steps は既に検証済みなので、常にこの計算を実行
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            # コサイン減衰を適用（各パラメータグループごとに計算）
            # warmup終了時のLRを開始値として使用
            lrs = []
            for base_lr in self.base_lrs:
                warmup_end_lr = base_lr * warmup_end_scale
                cos_value = math.cos(progress * math.pi)
                lr = self.min_lr + 0.5 * (warmup_end_lr - self.min_lr) * (1 + cos_value)
                lrs.append(max(self.min_lr, lr))
            return lrs
