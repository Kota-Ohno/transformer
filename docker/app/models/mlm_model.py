"""
マスクド言語モデリング（MLM）用のモデル実装
"""
import torch
import torch.nn as nn
import math
from typing import Optional, Tuple

from .encoder import Encoder
from .layers import PositionalEncoding
from ..utils.config import CONFIG


class MLMHead(nn.Module):
    """MLM用の予測ヘッド
    
    BERTと同じ構造：
    - 線形変換（hidden_size -> hidden_size）
    - GELU活性化関数
    - Layer Normalization
    - 出力層（hidden_size -> vocab_size）
    """
    
    def __init__(self, hidden_size: int, vocab_size: int):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.activation = nn.GELU()
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.decoder = nn.Linear(hidden_size, vocab_size)
        
        # バイアス項の初期化
        nn.init.normal_(self.decoder.weight, std=0.02)
        nn.init.zeros_(self.decoder.bias)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: エンコーダーの出力 [batch_size, seq_len, hidden_size]
            
        Returns:
            予測ロジット [batch_size, seq_len, vocab_size]
        """
        hidden_states = self.dense(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.layer_norm(hidden_states)
        logits = self.decoder(hidden_states)
        return logits


class MLMModel(nn.Module):
    """マスクド言語モデリング（MLM）モデル
    
    Transformer Encoder + MLM Head の構成
    BERTと同じアーキテクチャを採用
    """
    
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        d_ff: int = 3072,
        max_seq_length: int = 512,
        dropout_rate: float = 0.1,
        pad_idx: int = 0,
    ):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.pad_idx = pad_idx
        
        # トークン埋め込み
        self.token_embedding = nn.Embedding(vocab_size, hidden_size, padding_idx=pad_idx)
        
        # 位置エンコーディング
        self.positional_encoding = PositionalEncoding(hidden_size, max_seq_length, dropout_rate)
        
        # Transformer Encoder
        self.encoder = Encoder(hidden_size, num_heads, d_ff, num_layers, dropout_rate)
        
        # MLM予測ヘッド
        self.mlm_head = MLMHead(hidden_size, vocab_size)
        
        # 重みの初期化
        self._init_weights()
        
    def _init_weights(self):
        """重みを初期化"""
        # 埋め込み層の初期化
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        if self.token_embedding.padding_idx is not None:
            with torch.no_grad():
                self.token_embedding.weight[self.token_embedding.padding_idx].fill_(0)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        フォワードパス
        
        Args:
            input_ids: 入力トークンID [batch_size, seq_len]
            attention_mask: アテンションマスク [batch_size, seq_len]
                           1=有効なトークン、0=パディング
            
        Returns:
            予測ロジット [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        
        # アテンションマスクの作成
        if attention_mask is None:
            # パディングトークンをマスク
            attention_mask = (input_ids != self.pad_idx).unsqueeze(1).unsqueeze(2)
        else:
            # [batch_size, seq_len] -> [batch_size, 1, 1, seq_len]
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        
        # トークン埋め込み
        embedded = self.token_embedding(input_ids) * math.sqrt(self.hidden_size)
        
        # 位置エンコーディングを追加
        encoded = self.positional_encoding(embedded)
        
        # Transformer Encoderを通過
        encoder_output = self.encoder(encoded, attention_mask)
        
        # MLMヘッドで予測
        logits = self.mlm_head(encoder_output)
        
        return logits
    
    def save_pretrained(self, save_directory: str):
        """モデルを保存"""
        import os
        os.makedirs(save_directory, exist_ok=True)
        
        # モデルの状態を保存
        model_path = os.path.join(save_directory, "pytorch_model.bin")
        torch.save(self.state_dict(), model_path)
        
        # 設定を保存
        config_dict = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "pad_idx": self.pad_idx,
        }
        config_path = os.path.join(save_directory, "config.json")
        import json
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False)
            
    @classmethod
    def from_pretrained(cls, model_directory: str):
        """保存されたモデルを読み込み"""
        import os
        import json
        
        # 設定を読み込み
        config_path = os.path.join(model_directory, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # モデルを作成
        model = cls(**config_dict)
        
        # 重みを読み込み
        model_path = os.path.join(model_directory, "pytorch_model.bin")
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        
        return model


def create_mlm_model(
    vocab_size: int,
    model_size: str = "base",
    **kwargs
) -> MLMModel:
    """MLMモデルを作成するファクトリ関数
    
    Args:
        vocab_size: 語彙サイズ
        model_size: モデルサイズ ("small", "base", "large")
        **kwargs: 追加の設定
        
    Returns:
        MLMModelインスタンス
    """
    configs = {
        "small": {
            "hidden_size": 256,
            "num_layers": 4,
            "num_heads": 4,
            "d_ff": 1024,
        },
        "base": {
            "hidden_size": 768,
            "num_layers": 12,
            "num_heads": 12,
            "d_ff": 3072,
        },
        "large": {
            "hidden_size": 1024,
            "num_layers": 24,
            "num_heads": 16,
            "d_ff": 4096,
        },
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}. Choose from {list(configs.keys())}")
    
    config = configs[model_size].copy()
    config.update(kwargs)
    config["vocab_size"] = vocab_size
    
    return MLMModel(**config)
