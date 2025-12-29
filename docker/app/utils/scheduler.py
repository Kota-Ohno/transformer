"""
学習率のスケジューラを定義します。
"""
import torch

class WarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    """
    ウォームアップ付きの学習率スケジューラ
    """
    def __init__(self, optimizer, d_model, warmup_steps, total_steps, min_lr=1e-6, last_epoch=-1):
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        super(WarmupScheduler, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step = self.last_epoch + 1
        if step < self.warmup_steps:
            lr = (self.d_model ** -0.5) * (step * self.warmup_steps ** -1.5)
        else:
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            # コサイン減衰を適用
            lr = self.min_lr + 0.5 * (self.base_lrs[0] - self.min_lr) * (1 + torch.cos(torch.tensor(progress * 3.14159)))

        return [max(self.min_lr, lr) for _ in self.base_lrs]
