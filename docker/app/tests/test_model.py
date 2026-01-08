"""
モデル構築のテスト
"""
import torch

from models.model import create_transformer_model

def test_create_transformer_model():
    """create_transformer_modelが正しくモデルを構築できるかテスト"""
    input_vocab_size = 100
    output_vocab_size = 100
    hidden_size = 128
    num_heads = 4
    num_layers = 2
    d_ff = 256
    dropout = 0.1
    max_seq_length = 50

    model = create_transformer_model(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        hidden_size=hidden_size,
        num_heads=num_heads,
        num_layers=num_layers,
        d_ff=d_ff,
        dropout=dropout,
        max_seq_length=max_seq_length
    )

    assert model is not None
    assert isinstance(model.encoder.embedding, torch.nn.Embedding)
    assert model.encoder.embedding.num_embeddings == input_vocab_size
    assert model.decoder.embedding.num_embeddings == output_vocab_size
    assert model.encoder.d_model == hidden_size
    assert len(model.encoder.layers) == num_layers
    assert model.encoder.layers[0].self_attn.num_heads == num_heads
    assert model.decoder.d_model == hidden_size
    assert len(model.decoder.layers) == num_layers
    assert model.decoder.layers[0].self_attn.num_heads == num_heads

    # デコーダーのクロスアテンション（エンコーダ-デコーダアテンション）の確認
    assert hasattr(model.decoder.layers[0], 'encoder_attn'), "Decoder layer should have encoder_attn (cross-attention)"
    assert model.decoder.layers[0].encoder_attn.num_heads == num_heads, "Cross-attention num_heads should match"

    # 埋め込み次元がhidden_sizeと一致することを確認
    assert model.encoder.embedding.embedding_dim == hidden_size, "Encoder embedding_dim should match hidden_size"
    assert model.decoder.embedding.embedding_dim == hidden_size, "Decoder embedding_dim should match hidden_size"

    # フィードフォワード層のd_ffが正しく設定されていることを確認
    assert model.encoder.layers[0].feed_forward.linear1.out_features == d_ff, "Encoder feed-forward d_ff should match"
    assert model.decoder.layers[0].feed_forward.linear1.out_features == d_ff, "Decoder feed-forward d_ff should match"
