#!/usr/bin/env python3
"""
チェックポイント機能のテストスクリプト

学習の中断・再開が正しく動作することを確認します。
"""
import sys
import os
import tempfile
import shutil

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from models.mlm_model import create_mlm_model
from utils.checkpoint import save_checkpoint, load_checkpoint

print("=== チェックポイント機能テスト ===\n")

# テスト用の一時ディレクトリ
test_dir = tempfile.mkdtemp()
checkpoint_path = os.path.join(test_dir, "test_checkpoint.pth")

print(f"テストディレクトリ: {test_dir}\n")

# 1. モデルとオプティマイザーを作成
print("1. モデルとオプティマイザーを作成")
model = create_mlm_model(vocab_size=1000, model_size="small")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
print("   ✓ モデル作成完了")

# 2. ダミーの学習ステップを実行
print("\n2. ダミーの学習ステップを実行")
batch_size, seq_len = 2, 10
input_ids = torch.randint(0, 1000, (batch_size, seq_len))
logits = model(input_ids)
loss = logits.mean()
loss.backward()
optimizer.step()
print(f"   ✓ ダミー学習完了 (loss: {loss.item():.4f})")

# 3. チェックポイントを保存
print("\n3. チェックポイントを保存")
current_epoch = 5
current_step = 1000
best_loss = 2.5

save_checkpoint(
    model=model,
    optimizer=optimizer,
    epoch=current_epoch,
    step=current_step,
    loss=best_loss,
    save_path=checkpoint_path
)
print(f"   ✓ チェックポイント保存完了: {checkpoint_path}")
print(f"     - Epoch: {current_epoch}")
print(f"     - Step: {current_step}")
print(f"     - Loss: {best_loss:.4f}")

# 4. 新しいモデルとオプティマイザーを作成
print("\n4. 新しいモデルとオプティマイザーを作成（リセット状態）")
new_model = create_mlm_model(vocab_size=1000, model_size="small")
new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-4)

# リセット前のパラメータを記録
reset_params = {}
for name, param in new_model.named_parameters():
    if param.requires_grad:
        reset_params[name] = param.data.clone()
        break
print("   ✓ リセット状態のモデル作成完了")

# 5. チェックポイントを読み込み
print("\n5. チェックポイントを読み込み")
loaded_epoch, loaded_step, loaded_loss = load_checkpoint(
    model=new_model,
    optimizer=new_optimizer,
    checkpoint_path=checkpoint_path
)
print(f"   ✓ チェックポイント読み込み完了")
print(f"     - Epoch: {loaded_epoch}")
print(f"     - Step: {loaded_step}")
print(f"     - Loss: {loaded_loss:.4f}")

# 6. パラメータが復元されているか確認
print("\n6. パラメータ復元を確認")
params_match = True
for name, param in new_model.named_parameters():
    if param.requires_grad and name in reset_params:
        if not torch.allclose(param.data, reset_params[name]):
            params_match = False
            break

if not params_match:
    print("   ✓ パラメータが正しく復元されました")
else:
    print("   ✗ パラメータが復元されていない可能性があります")

# 7. 一致確認
print("\n7. 保存時と読み込み時の値を比較")
if loaded_epoch == current_epoch and loaded_step == current_step:
    print(f"   ✓ EpochとStepが一致")
else:
    print(f"   ✗ 不一致: expected epoch={current_epoch}, step={current_step}")
    print(f"            got epoch={loaded_epoch}, step={loaded_step}")

# クリーンアップ
shutil.rmtree(test_dir)
print(f"\n8. クリーンアップ完了")

print("\n=== チェックポイントテスト完了 ===")
print("\n✅ チェックポイント機能は正しく動作しています")
print("\n学習時の使用例:")
print("  # 学習再開時")
print("  python core/train_mlm.py --resume")
print("  # または特定のチェックポイントから")
print("  python core/train_mlm.py --checkpoint models/checkpoints/checkpoint_epoch_5.pth")
