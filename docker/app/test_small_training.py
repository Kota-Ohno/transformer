#!/usr/bin/env python3
"""
小さなデータセットで学習をテスト
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from models.mlm_model import create_mlm_model
from models.loss import MLMLoss
from utils.masking import create_mlm_mask

print("=== 小規模テスト ===\n")

# 1. トークナイザー
print("1. トークナイザーを読み込み")
tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")
vocab_size = tokenizer.vocab_size
print(f"   語彙サイズ: {vocab_size}")

# 2. 小さなデータセットを作成
print("\n2. テストデータを作成")
texts = [
    "The cat sat on the mat.",
    "The dog ran in the park.",
    "A bird flew over the tree.",
    "The sun is shining bright.",
    "I love to read books.",
] * 20  # 100サンプル

# トークナイズ
encodings = tokenizer(
    texts,
    max_length=32,
    padding='max_length',
    truncation=True,
    return_tensors='pt'
)

print(f"   データ数: {len(texts)}")
print(f"   シーケンス長: {encodings['input_ids'].shape[1]}")

# 3. 簡易Dataset
class SimpleDataset(Dataset):
    def __init__(self, input_ids, attention_mask, tokenizer):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.input_ids)
    
    def __getitem__(self, idx):
        input_ids = self.input_ids[idx].clone()
        
        # マスキング
        masked_ids, labels = create_mlm_mask(
            input_ids.unsqueeze(0),
            mask_token_id=self.tokenizer.mask_token_id,
            vocab_size=self.tokenizer.vocab_size,
            mask_prob=0.15,
            pad_token_id=self.tokenizer.pad_token_id,
            special_token_ids=[self.tokenizer.cls_token_id, self.tokenizer.sep_token_id]
        )
        
        return {
            'input_ids': masked_ids.squeeze(0),
            'attention_mask': self.attention_mask[idx],
            'labels': labels.squeeze(0)
        }

dataset = SimpleDataset(
    encodings['input_ids'],
    encodings['attention_mask'],
    tokenizer
)
dataloader = DataLoader(dataset, batch_size=10, shuffle=True)

# 4. モデルを作成
print("\n3. モデルを作成")
model = create_mlm_model(vocab_size=vocab_size, model_size="small")
print(f"   パラメータ数: {sum(p.numel() for p in model.parameters()):,}")

# 5. 学習設定
print("\n4. 学習設定")
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = MLMLoss(vocab_size=vocab_size, ignore_index=-100)

# 6. 学習ループ
print("\n5. 学習開始（10エポック）")
model.train()

for epoch in range(10):
    total_loss = 0
    total_correct = 0
    total_masked = 0
    
    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask']
        labels = batch['labels']
        
        # フォワード
        logits = model(input_ids, attention_mask)
        
        # 損失
        loss = criterion(logits, labels)
        
        # バックワード
        loss.backward()
        optimizer.step()
        
        # 統計
        total_loss += loss.item()
        
        # Accuracy計算
        predictions = logits.argmax(dim=-1)
        mask = labels != -100
        correct = (predictions == labels) & mask
        total_correct += correct.sum().item()
        total_masked += mask.sum().item()
    
    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / total_masked if total_masked > 0 else 0
    perplexity = torch.exp(torch.tensor(avg_loss)).item()
    
    print(f"   Epoch {epoch+1}: Loss={avg_loss:.4f}, PPL={perplexity:.2f}, Acc={accuracy:.2%}")

# 7. 推論テスト
print("\n6. 推論テスト")
model.eval()

test_text = "The cat sat on the 化和済みinstance_token_ids."
inputs = tokenizer(test_text, return_tensors='pt')

with torch.no_grad():
    logits = model(inputs['input_ids'], inputs['attention_mask'])
    
# マスク位置を特定
mask_pos = (inputs['input_ids'] == tokenizer.mask_token_id).nonzero(as_tuple=True)[1]

if len(mask_pos) > 0:
    pos = mask_pos[0].item()
    probs = torch.softmax(logits[0, pos], dim=-1)
    top_k = torch.topk(probs, k=5)
    
    print(f"   入力: {test_text}")
    print(f"   予測:")
    for i, (prob, idx) in enumerate(zip(top_k.values, top_k.indices)):
        token = tokenizer.decode([idx.item()])
        print(f"     {i+1}. '{token}' ({prob.item():.2%})")

print("\n=== テスト完了 ===")
