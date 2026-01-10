"""
学習率のスケジューラを定義します。
"""
import math
import torch

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
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
        if step < self.warmup_steps:
            # 共有warmup乗数を計算
            common_lr = (self.d_model ** -0.5) * (step * self.warmup_steps ** -1.5)
            # warmup終了時の共通値を計算
            warmup_end = (self.d_model ** -0.5) * (self.warmup_steps ** -0.5)
            # 各グループのLRをスケールして、warmup終了時にbase_lrと一致させる
            lrs = []
            for base_lr in self.base_lrs:
                lr_i = max(self.min_lr, common_lr * (base_lr / warmup_end))
                lrs.append(lr_i)
            return lrs
        else:
            # warmup_steps < total_steps は既に検証済みなので、常にこの計算を実行
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            progress = min(max(progress, 0.0), 1.0)
            # コサイン減衰を適用（各パラメータグループごとに計算）
            lrs = []
            for base_lr in self.base_lrs:
                cos_value = math.cos(progress * math.pi)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + cos_value)
                lrs.append(max(self.min_lr, lr))
            return lrs
