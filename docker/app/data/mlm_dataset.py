"""
MLM用のデータセット実装

WikiText-2データセットを使用し、Hugging Face Tokenizerでトークナイズ
"""
import torch
from torch.utils.data import Dataset
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)


class WikiText2MLMDataset(Dataset):
    """WikiText-2 MLMデータセット
    
    Hugging Face datasetsライブラリを使用してWikiText-2をダウンロードし、
    MLM用に前処理を行う。
    """
    
    def __init__(
        self,
        tokenizer,
        max_seq_length: int = 512,
        split: str = "train",
        mask_prob: float = 0.15,
        cache_dir: Optional[str] = None
    ):
        """
        Args:
            tokenizer: Hugging Faceトークナイザー（bert-base-casedなど）
            max_seq_length: 最大シーケンス長
            split: データセットのスプリット（"train", "validation", "test"）
            mask_prob: MLMのマスク確率
            cache_dir: データセットのキャッシュディレクトリ
        """
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.split = split
        self.mask_prob = mask_prob
        self.mask_token_id = tokenizer.mask_token_id
        self.vocab_size = tokenizer.vocab_size
        self.pad_token_id = tokenizer.pad_token_id
        
        # 特殊トークンID
        self.cls_token_id = tokenizer.cls_token_id
        self.sep_token_id = tokenizer.sep_token_id
        
        # データセットをダウンロード
        logger.info(f"WikiText-2 {split}データセットを読み込み中...")
        try:
            from datasets import load_dataset
            self.raw_dataset = load_dataset(
                "wikitext",
                "wikitext-2-raw-v1",
                split=split,
                cache_dir=cache_dir
            )
            logger.info(f"データセット読み込み完了: {len(self.raw_dataset)}件")
        except Exception as e:
            logger.error(f"データセットの読み込みに失敗しました: {e}")
            raise
        
        # テキストをトークナイズ
        logger.info("テキストをトークナイズ中...")
        self.examples = self._tokenize_texts()
        logger.info(f"トークナイズ完了: {len(self.examples)}件のシーケンス")
    
    def _tokenize_texts(self) -> List[List[int]]:
        """テキストをトークナイズしてシーケンスを作成"""
        examples = []
        
        # 全テキストを結合して一つの長いシーケンスにする
        full_text = " ".join([text for text in self.raw_dataset["text"] if text.strip()])
        
        # トークナイズ
        tokenized = self.tokenizer(
            full_text,
            add_special_tokens=False,
            return_tensors=None,
            truncation=False
        )["input_ids"]
        
        # 長いシーケンスをmax_seq_lengthのチャンクに分割
        # CLSとSEPトークン用に2つ分のスペースを確保
        effective_length = self.max_seq_length - 2
        
        for i in range(0, len(tokenized), effective_length):
            chunk = tokenized[i:i + effective_length]
            
            # CLSとSEPトークンを追加
            chunk = [self.cls_token_id] + chunk + [self.sep_token_id]
            
            # パディング
            if len(chunk) < self.max_seq_length:
                chunk = chunk + [self.pad_token_id] * (self.max_seq_length - len(chunk))
            
            examples.append(chunk)
        
        return examples
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Returns:
            辞書形式のデータ:
                - input_ids: 入力トークンID [max_seq_length]
                - attention_mask: アテンションマスク [max_seq_length]
                - labels: MLM用ラベル（-100は無視） [max_seq_length]
        """
        input_ids = self.examples[idx]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        
        # アテンションマスク（パディングトークンは0）
        attention_mask = (input_ids != self.pad_token_id).long()
        
        # MLMマスキングを適用
        from utils.masking import apply_mlm_mask_batch
        masked_input_ids, labels = apply_mlm_mask_batch(
            input_ids.unsqueeze(0),  # バッチ次元を追加
            mask_token_id=self.mask_token_id,
            vocab_size=self.vocab_size,
            pad_token_id=self.pad_token_id,
            cls_token_id=self.cls_token_id,
            sep_token_id=self.sep_token_id,
            mask_prob=self.mask_prob
        )
        
        return {
            "input_ids": masked_input_ids.squeeze(0),  # バッチ次元を除去
            "attention_mask": attention_mask,
            "labels": labels.squeeze(0)  # バッチ次元を除去
        }


def create_mlm_dataloaders(
    tokenizer,
    max_seq_length: int = 512,
    batch_size: int = 32,
    num_workers: int = 0,
    mask_prob: float = 0.15,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """MLM用のDataLoaderを作成
    
    Args:
        tokenizer: Hugging Faceトークナイザー
        max_seq_length: 最大シーケンス長
        batch_size: バッチサイズ
        num_workers: データローディング用ワーカー数
        mask_prob: MLMマスク確率
        cache_dir: データセットのキャッシュディレクトリ
        
    Returns:
        辞書: {"train": train_loader, "validation": val_loader, "test": test_loader}
    """
    from torch.utils.data import DataLoader
    
    datasets = {}
    for split in ["train", "validation", "test"]:
        dataset = WikiText2MLMDataset(
            tokenizer=tokenizer,
            max_seq_length=max_seq_length,
            split=split,
            mask_prob=mask_prob,
            cache_dir=cache_dir
        )
        
        # 訓練データのみシャッフル
        shuffle = (split == "train")
        
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True
        )
        
        datasets[split] = loader
        logger.info(f"{split} DataLoader作成完了: {len(dataset)}件")
    
    return datasets


def load_wikitext2_for_mlm(
    tokenizer_name: str = "bert-base-cased",
    max_seq_length: int = 512,
    batch_size: int = 32,
    mask_prob: float = 0.15,
    cache_dir: Optional[str] = None
) -> Dict[str, Any]:
    """WikiText-2データセットを読み込み、MLM用に準備
    
    便利関数：トークナイザーの作成からDataLoaderの作成まで一括で行う
    
    Args:
        tokenizer_name: 使用するトークナイザー名
        max_seq_length: 最大シーケンス長
        batch_size: バッチサイズ
        mask_prob: MLMマスク確率
        cache_dir: データセットのキャッシュディレクトリ
        
    Returns:
        辞書: {
            "tokenizers": tokenizer,
            "dataloaders": {"train": ..., "validation": ..., "test": ...},
            "vocab_size": vocab_size
        }
    """
    from transformers import AutoTokenizer
    
    logger.info(f"トークナイザーを読み込み中: {tokenizer_name}")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    
    logger.info(f"語彙サイズ: {tokenizer.vocab_size}")
    logger.info(f"特殊トークン - CLS: {tokenizer.cls_token}, SEP: {tokenizer.sep_token}, MASK: {tokenizer.mask_token}")
    
    dataloaders = create_mlm_dataloaders(
        tokenizer=tokenizer,
        max_seq_length=max_seq_length,
        batch_size=batch_size,
        mask_prob=mask_prob,
        cache_dir=cache_dir
    )
    
    return {
        "tokenizer": tokenizer,
        "dataloaders": dataloaders,
        "vocab_size": tokenizer.vocab_size
    }
