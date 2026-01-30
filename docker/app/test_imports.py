#!/usr/bin/env python3
"""
簡易動作確認スクリプト
"""
import sys
import os

print("=== MLM実装 動作確認 ===\n")

# 1. 基本モジュールのインポートテスト
print("1. モジュールインポートテスト...")
try:
    import torch
    print(f"   ✓ PyTorch {torch.__version__}")
except ImportError as e:
    print(f"   ✗ PyTorch エラー: {e}")
    sys.exit(1)

try:
    from transformers import AutoTokenizer
    print("   ✓ Transformers")
except ImportError as e:
    print(f"   ✗ Transformers エラー: {e}")
    sys.exit(1)

try:
    from datasets import load_dataset
    print("   ✓ Datasets")
except ImportError as e:
    print(f"   ✗ Datasets エラー: {e}")
    sys.exit(1)

# 2. 自作モジュールのインポートテスト
print("\n2. 自作モジュールインポートテスト...")
try:
    from models.mlm_model import MLMModel, create_mlm_model
    print("   ✓ models.mlm_model")
except Exception as e:
    print(f"   ✗ models.mlm_model エラー: {e}")

try:
    from models.loss import MLMLoss
    print("   ✓ models.loss")
except Exception as e:
    print(f"   ✗ models.loss エラー: {e}")

try:
    from utils.masking import create_mlm_mask
    print("   ✓ utils.masking")
except Exception as e:
    print(f"   ✗ utils.masking エラー: {e}")

try:
    from data.mlm_dataset import WikiText2MLMDataset
    print("   ✓ data.mlm_dataset")
except Exception as e:
    print(f"   ✗ data.mlm_dataset エラー: {e}")

# 3. 簡易機能テスト
print("\n3. 機能テスト...")

try:
    # MLMモデルの作成テスト
    model = create_mlm_model(vocab_size=1000, model_size="small")
    print("   ✓ MLMモデル作成")
    
    # フォワードパステスト
    batch_size, seq_len = 2, 20
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    logits = model(input_ids)
    assert logits.shape == (batch_size, seq_len, 1000)
    print("   ✓ フォワードパス")
    
    # 損失計算テスト
    criterion = MLMLoss(vocab_size=1000)
    labels = torch.randint(0, 1000, (batch_size, seq_len))
    loss = criterion(logits, labels)
    assert loss.item() > 0
    print(f"   ✓ 損失計算 (loss={loss.item():.4f})")
    
    # マスキングテスト
    masked_ids, mask_labels = create_mlm_mask(
        input_ids, mask_token_id=103, vocab_size=1000
    )
    print("   ✓ マスキング処理")
    
except Exception as e:
    print(f"   ✗ 機能テスト エラー: {e}")
    import traceback
    traceback.print_exc()

print("\n=== 動作確認完了 ===")
print("\n次のステップ:")
print("1. Dockerコンテナをビルド: docker compose build")
print("2. コンテナを起動: docker compose up -d")
print("3. コンテナ内で実行: docker exec -it <container_name> python core/train_mlm.py --help")
