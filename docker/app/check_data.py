#!/usr/bin/env python3
"""
データの前処理を確認するスクリプト
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
from transformers import AutoTokenizer

print("=== データ前処理の確認 ===\n")

# 1. トークナイザーを確認
print("1. トークナイザーの確認")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
print(f"   語彙サイズ: {tokenizer.vocab_size}")
print(f"   特殊トークン:")
print(f"     - CLS: {tokenizer.cls_token} (ID: {tokenizer.cls_token_id})")
print(f"     - SEP: {tokenizer.sep_token} (ID: {tokenizer.sep_token_id})")
print(f"     - MASK: {tokenizer.mask_token} (ID: {tokenizer.mask_token_id})")
print(f"     - PAD: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")

# 2. サンプルテキストをトークナイズ
print("\n2. サンプルテキストのトークナイズ")
sample_text = "The cat sat on the mat and looked at the birds."
tokens = tokenizer.tokenize(sample_text)
token_ids = tokenizer.encode(sample_text)

print(f"   テキスト: {sample_text}")
print(f"   トークン: {tokens}")
print(f"   トークンID: {token_ids}")
print(f"   トークン数: {len(tokens)}")

# 3. MLMマスキングのテスト
print("\n3. MLMマスキングのテスト")
from utils.masking import create_mlm_mask

input_ids = torch.tensor([token_ids])
print(f"   元の入力: {input_ids[0][:10].tolist()}...")

# special_token_idsとしてCLSとSEPを渡す
special_tokens = [tokenizer.cls_token_id, tokenizer.sep_token_id]

masked_ids, labels = create_mlm_mask(
    input_ids,
    mask_token_id=tokenizer.mask_token_id,
    vocab_size=tokenizer.vocab_size,
    mask_prob=0.15,
    pad_token_id=tokenizer.pad_token_id,
    special_token_ids=special_tokens
)

print(f"   マスク後: {masked_ids[0][:10].tolist()}...")
print(f"   ラベル:   {labels[0][:10].tolist()}...")

# マスクされた位置を確認
mask_positions = (labels[0] != -100).nonzero(as_tuple=True)[0]
print(f"   マスク位置数: {len(mask_positions)}")

if len(mask_positions) > 0:
    for pos in mask_positions[:3]:
        original_token = tokenizer.decode([input_ids[0, pos].item()])
        masked_token = tokenizer.decode([masked_ids[0, pos].item()])
        label_token = tokenizer.decode([labels[0, pos].item()])
        print(f"     位置 {pos}: '{original_token}' -> '{masked_token}' (正解: '{label_token}')")

# 4. WikiText-2データセットのサンプルを確認
print("\n4. WikiText-2データセットのサンプル確認")
from data.mlm_dataset import WikiText2MLMDataset

print("   データセットを読み込み中（最初の数件のみ）...")
dataset = WikiText2MLMDataset(
    tokenizer=tokenizer,
    max_seq_length=128,
    split="train",
    mask_prob=0.15
)

print(f"   データセットサイズ: {len(dataset)}")

# 最初のサンプルを確認
sample = dataset[0]
print(f"\n   サンプル0:")
print(f"     input_ids shape: {sample['input_ids'].shape}")
print(f"     labels shape: {sample['labels'].shape}")

# 実際のトークンを表示
input_tokens = tokenizer.convert_ids_to_tokens(sample['input_ids'][:20])
print(f"     入力トークン（最初の20個）: {input_tokens}")

# マスク位置を確認
mask_count = (sample['labels'] != -100).sum().item()
print(f"     マスクされたトークン数: {mask_count}")

# 5. マスキングの分布を確認
print("\n5. マスキングの分布確認（最初の10サンプル）")
total_masks = 0
total_tokens = 0
for i in range(min(10, len(dataset))):
    sample = dataset[i]
    mask_count = (sample['labels'] != -100).sum().item()
    token_count = (sample['input_ids'] != tokenizer.pad_token_id).sum().item()
    total_masks += mask_count
    total_tokens += token_count
    print(f"   サンプル {i}: {mask_count} マスク / {token_count} トークン ({mask_count/token_count*100:.1f}%)")

print(f"\n   平均マスク率: {total_masks/total_tokens*100:.1f}%")

# 6. 問題がないか確認
print("\n=== 確認完了 ===")
issues = []

if total_masks == 0:
    issues.append("❌ マスクが作成されていません")
else:
    print("✅ マスキングは正常に動作")

if total_masks / total_tokens < 0.10:
    issues.append(f"⚠️  マスク率が低いです: {total_masks/total_tokens*100:.1f}%")
elif total_masks / total_tokens > 0.20:
    issues.append(f"⚠️  マスク率が高いです: {total_masks/total_tokens*100:.1f}%")
else:
    print(f"✅ マスク率は適切です: {total_masks/total_tokens*100:.1f}%")

if len(issues) > 0:
    print("\n発見された問題:")
    for issue in issues:
        print(f"  {issue}")
else:
    print("\n✅ データの前処理に問題は見つかりませんでした")
