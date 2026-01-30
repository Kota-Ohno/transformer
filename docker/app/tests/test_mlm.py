"""
MLMコンポーネントのテスト
"""
import pytest
import torch
import sys
import os

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.mlm_model import MLMModel, create_mlm_model, MLMHead
from models.loss import MLMLoss, LabelSmoothingCrossEntropyLoss
from utils.masking import create_mlm_mask, create_mlm_mask_numpy


class TestMLMHead:
    """MLM Headのテスト"""
    
    def test_mlm_head_output_shape(self):
        """出力形状が正しいことを確認"""
        batch_size = 2
        seq_len = 10
        hidden_size = 768
        vocab_size = 30000
        
        head = MLMHead(hidden_size, vocab_size)
        hidden_states = torch.randn(batch_size, seq_len, hidden_size)
        
        logits = head(hidden_states)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_mlm_head_gradient_flow(self):
        """勾配が流れることを確認"""
        head = MLMHead(768, 30000)
        hidden_states = torch.randn(2, 10, 768, requires_grad=True)
        
        logits = head(hidden_states)
        loss = logits.sum()
        loss.backward()
        
        assert hidden_states.grad is not None


class TestMLMModel:
    """MLMモデルのテスト"""
    
    def test_model_creation(self):
        """モデルが正しく作成されることを確認"""
        model = create_mlm_model(
            vocab_size=30000,
            model_size="small",
            max_seq_length=128
        )
        
        assert isinstance(model, MLMModel)
        assert model.vocab_size == 30000
    
    def test_model_forward(self):
        """フォワードパスが正しく動作することを確認"""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000
        
        model = create_mlm_model(
            vocab_size=vocab_size,
            model_size="small",
            max_seq_length=seq_len
        )
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        attention_mask = torch.ones(batch_size, seq_len)
        
        logits = model(input_ids, attention_mask)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)
    
    def test_model_without_attention_mask(self):
        """アテンションマスクなしでも動作することを確認"""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000
        
        model = create_mlm_model(
            vocab_size=vocab_size,
            model_size="small",
            max_seq_length=seq_len,
            pad_idx=0
        )
        
        input_ids = torch.randint(1, vocab_size, (batch_size, seq_len))  # 0はパディング
        
        logits = model(input_ids)
        
        assert logits.shape == (batch_size, seq_len, vocab_size)


class TestMLMLoss:
    """MLM損失関数のテスト"""
    
    def test_mlm_loss_calculation(self):
        """損失が正しく計算されることを確認"""
        batch_size = 2
        seq_len = 10
        vocab_size = 1000
        
        criterion = MLMLoss(vocab_size=vocab_size)
        
        # ランダムなロジットとラベル
        logits = torch.randn(batch_size, seq_len, vocab_size)
        labels = torch.randint(0, vocab_size, (batch_size, seq_len))
        # 一部を-100に設定（無視されるべき）
        labels[0, :3] = -100
        
        loss = criterion(logits, labels)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_mlm_loss_with_label_smoothing(self):
        """ラベルスムージング付き損失が動作することを確認"""
        criterion = MLMLoss(vocab_size=1000, label_smoothing=0.1)
        
        logits = torch.randn(2, 10, 1000)
        labels = torch.randint(0, 1000, (2, 10))
        
        loss = criterion(logits, labels)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)


class TestLabelSmoothingLoss:
    """ラベルスムージング損失のテスト"""
    
    def test_label_smoothing_loss(self):
        """ラベルスムージング損失が正しく計算されることを確認"""
        criterion = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
        
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        loss = criterion(logits, targets)
        
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_label_smoothing_vs_standard(self):
        """ラベルスムージングありとなしの損失値を比較"""
        logits = torch.randn(4, 100)
        targets = torch.randint(0, 100, (4,))
        
        criterion_standard = LabelSmoothingCrossEntropyLoss(smoothing=0.0)
        criterion_smoothing = LabelSmoothingCrossEntropyLoss(smoothing=0.1)
        
        loss_standard = criterion_standard(logits, targets)
        loss_smoothing = criterion_smoothing(logits, targets)
        
        # ラベルスムージングありの方が損失値が大きいはず
        assert loss_smoothing > loss_standard


class TestMasking:
    """マスキング関数のテスト"""
    
    def test_create_mlm_mask_shape(self):
        """マスク出力の形状が正しいことを確認"""
        batch_size = 2
        seq_len = 20
        vocab_size = 1000
        
        input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        masked_ids, labels = create_mlm_mask(
            input_ids,
            mask_token_id=103,
            vocab_size=vocab_size
        )
        
        assert masked_ids.shape == (batch_size, seq_len)
        assert labels.shape == (batch_size, seq_len)
    
    def test_create_mlm_mask_mask_rate(self):
        """適切な割合のトークンがマスクされることを確認"""
        seq_len = 1000
        vocab_size = 1000
        mask_prob = 0.15
        
        input_ids = torch.randint(10, vocab_size, (1, seq_len))  # 特殊トークンを避ける
        
        masked_ids, labels = create_mlm_mask(
            input_ids,
            mask_token_id=103,
            vocab_size=vocab_size,
            mask_prob=mask_prob,
            pad_token_id=0
        )
        
        # ラベルが-100でない（マスク対象）の割合を計算
        num_masked = (labels != -100).sum().item()
        actual_mask_rate = num_masked / seq_len
        
        # 15%前後であることを確認（確率なので厳密には一致しない）
        assert 0.10 < actual_mask_rate < 0.20
    
    def test_create_mlm_mask_numpy(self):
        """NumPy版マスキングが動作することを確認"""
        input_ids = list(range(20, 40))  # 20-39
        vocab_size = 100
        
        masked_ids, labels = create_mlm_mask_numpy(
            input_ids,
            mask_token_id=103,
            vocab_size=vocab_size,
            mask_prob=0.15,
            pad_token_id=0
        )
        
        assert len(masked_ids) == len(input_ids)
        assert len(labels) == len(input_ids)
        
        # マスクされていない位置は-100
        assert all(l == -100 or l in input_ids for l in labels)
    
    def test_create_mlm_mask_preserves_special_tokens(self):
        """特殊トークンがマスクされないことを確認"""
        batch_size = 1
        seq_len = 10
        pad_token_id = 0
        cls_token_id = 101
        sep_token_id = 102
        
        # 特殊トークンを含む入力
        input_ids = torch.tensor([
            [cls_token_id, 5, 6, 7, sep_token_id, 8, 9, pad_token_id, pad_token_id, pad_token_id]
        ])
        
        masked_ids, labels = create_mlm_mask(
            input_ids,
            mask_token_id=103,
            vocab_size=1000,
            mask_prob=1.0,  # 100%マスク（テスト用）
            pad_token_id=pad_token_id,
            special_token_ids=[cls_token_id, sep_token_id]
        )
        
        # 特殊トークンの位置を確認
        assert labels[0, 0] == -100  # CLSは無視
        assert labels[0, 4] == -100  # SEPは無視
        assert labels[0, 7] == -100  # PADは無視
        assert labels[0, 8] == -100  # PADは無視
        assert labels[0, 9] == -100  # PADは無視


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
