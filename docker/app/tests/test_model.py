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

    # 動作ベースのテスト: forward passを実行して出力形状とdtypeを確認
    batch_size = 4
    src_seq_len = 20
    tgt_seq_len = 15

    src = torch.randint(0, input_vocab_size, (batch_size, src_seq_len))
    tgt = torch.randint(0, output_vocab_size, (batch_size, tgt_seq_len))

    output, _ = model(src, tgt)

    # 出力形状の確認
    assert output.shape == (batch_size, tgt_seq_len, output_vocab_size), \
        f"Expected output shape ({batch_size}, {tgt_seq_len}, {output_vocab_size}), got {output.shape}"

    # 出力dtypeの確認
    assert output.dtype == torch.float32, \
        f"Expected output dtype torch.float32, got {output.dtype}"

    # 勾配計算の確認（オプション）
    loss = output.sum()
    loss.backward()
    # 勾配が計算されていることを確認（少なくとも1つのパラメータに勾配がある）
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_grad, "Gradients should be computed during backward pass"
