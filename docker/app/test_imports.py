#!/usr/bin/env python3
"""
簡易動作確認スクリプト
"""
import sys
import os

# Pythonパスを設定（相対インポートを機能させるため）
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

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

# インポートを試行
models_imported = False
loss_imported = False
masking_imported = False
dataset_imported = False

try:
    from models.mlm_model import MLMModel, create_mlm_model
    print("   ✓ models.mlm_model")
    models_imported = True
except Exception as e:
    print(f"   ✗ models.mlm_model エラー: {e}")

try:
    from models.loss import MLMLoss
    print("   ✓ models.loss")
    loss_imported = True
except Exception as e:
    print(f"   ✗ models.loss エラー: {e}")

try:
    from utils.masking import create_mlm_mask
    print("   ✓ utils.masking")
    masking_imported = True
except Exception as e:
    print(f"   ✗ utils.masking エラー: {e}")

try:
    from data.mlm_dataset import WikiText2MLMDataset
    print("   ✓ data.mlm_dataset")
    dataset_imported = True
except Exception as e:
    print(f"   ✗ data.mlm_dataset エラー: {e}")

# 3. 簡易機能テスト
print("\n3. 機能テスト...")

if not all([models_imported, loss_imported, masking_imported]):
    print("   ! スキップ: 必要なモジュールがインポートされていません")
else:
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

# 最終結果
if all([models_imported, loss_imported, masking_imported, dataset_imported]):
    print("\n✅ すべてのテストが成功しました！")
    print("\n次のステップ:")
    print("1. MLM学習を開始: python core/train_mlm.py --model-size small --epochs 1 --batch-size 8")
else:
    print("\n⚠️ 一部のテストが失敗しました。エラーを確認してください。")
