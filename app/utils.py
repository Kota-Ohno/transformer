import torch
def add_padding(x, max_length, pad_token=0):
    """
    シーケンスにパディングを追加して指定された最大長に合わせます。
    
    Args:
        x (torch.Tensor): パディングを追加するシーケンス。
        max_length (int): シーケンスの最大長。
        pad_token (int): パディングに使用するトークンの値。
    
    Returns:
        torch.Tensor: パディングが追加されたシーケンス。
    """
    padding_size = max_length - x.size(1)
    if padding_size > 0:
        padding = torch.full((x.size(0), padding_size), pad_token, dtype=x.dtype, device=x.device)
        x = torch.cat([x, padding], dim=1)
    return x