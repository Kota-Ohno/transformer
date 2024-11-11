import torch
import torch.nn as nn

class WarmupScheduler:
    def __init__(self, optimizer, d_model, warmup_steps, total_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = (self.d_model ** -0.5) * min(self.current_step ** -0.5, self.current_step * self.warmup_steps ** -1.5)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

def validate(model, val_loader, criterion, device, output_dim):
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            if torch.cuda.is_available():  # デバッグ用のメモリ使用量ログ
                print(f'GPU memory: {torch.cuda.memory_allocated() / 1024**2:.2f}MB')
            decoder_output = model(X_batch, y_batch)
            loss = criterion(decoder_output.view(-1, output_dim), y_batch.view(-1))
            total_val_loss += loss.item()
    return total_val_loss / len(val_loader)

class TranslationModel(nn.Module):
    def __init__(self, encoder, decoder):
        super(TranslationModel, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        encoder_output = self.encoder(src)
        output = self.decoder(tgt, encoder_output)
        return output
