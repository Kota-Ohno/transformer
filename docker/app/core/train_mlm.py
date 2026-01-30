"""
MLM（Masked Language Modeling）の学習スクリプト

TensorBoardによる可視化と複数評価指標に対応
"""
import os
import sys
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# プロジェクトルートをパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from models.mlm_model import create_mlm_model
from models.loss import MLMLoss
from data.mlm_dataset import load_wikitext2_for_mlm
from utils.checkpoint import save_checkpoint, load_checkpoint


# ロギング設定
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def calculate_perplexity(loss: float) -> float:
    """損失からperplexityを計算"""
    return torch.exp(torch.tensor(loss)).item()


def calculate_accuracy(logits: torch.Tensor, labels: torch.Tensor, ignore_index: int = -100) -> float:
    """マスクされたトークンの予測精度を計算"""
    predictions = logits.argmax(dim=-1)
    mask = labels != ignore_index
    
    if mask.sum() == 0:
        return 0.0
    
    correct = (predictions == labels) & mask
    accuracy = correct.sum().float() / mask.sum().float()
    return accuracy.item()


def train_epoch(
    model: nn.Module,
    train_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 100
) -> Dict[str, float]:
    """1エポックの学習
    
    Returns:
        辞書: {"loss": 平均損失, "perplexity": 平均困惑度, "accuracy": 平均精度}
    """
    model.train()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        
        # 勾配をリセット
        optimizer.zero_grad()
        
        # フォワードパス
        logits = model(input_ids, attention_mask)
        
        # 損失計算
        loss = criterion(logits, labels)
        
        # バックワードパス
        loss.backward()
        
        # 勾配クリッピング
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        # パラメータ更新
        optimizer.step()
        
        # 統計情報を記録
        batch_loss = loss.item()
        batch_accuracy = calculate_accuracy(logits, labels)
        
        total_loss += batch_loss
        total_accuracy += batch_accuracy
        num_batches += 1
        
        # プログレスバー更新
        pbar.set_postfix({
            "loss": f"{batch_loss:.4f}",
            "acc": f"{batch_accuracy:.4f}"
        })
        
        # TensorBoardに記録（ログ間隔ごと）
        if writer is not None and batch_idx % log_interval == 0:
            global_step = epoch * len(train_loader) + batch_idx
            writer.add_scalar("Train/Loss", batch_loss, global_step)
            writer.add_scalar("Train/Accuracy", batch_accuracy, global_step)
            writer.add_scalar("Train/Perplexity", calculate_perplexity(batch_loss), global_step)
    
    # 平均を計算
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": avg_accuracy
    }


def evaluate(
    model: nn.Module,
    eval_loader,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
    writer: Optional[SummaryWriter] = None
) -> Dict[str, float]:
    """検証データでの評価
    
    Returns:
        辞書: {"loss": 平均損失, "perplexity": 平均困惑度, "accuracy": 平均精度}
    """
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    num_batches = 0
    
    with torch.no_grad():
        pbar = tqdm(eval_loader, desc=f"Epoch {epoch} [Eval]")
        
        for batch in pbar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            # フォワードパス
            logits = model(input_ids, attention_mask)
            
            # 損失計算
            loss = criterion(logits, labels)
            
            # 統計情報を記録
            batch_loss = loss.item()
            batch_accuracy = calculate_accuracy(logits, labels)
            
            total_loss += batch_loss
            total_accuracy += batch_accuracy
            num_batches += 1
            
            # プログレスバー更新
            pbar.set_postfix({
                "loss": f"{batch_loss:.4f}",
                "acc": f"{batch_accuracy:.4f}"
            })
    
    # 平均を計算
    avg_loss = total_loss / num_batches
    avg_accuracy = total_accuracy / num_batches
    perplexity = calculate_perplexity(avg_loss)
    
    # TensorBoardに記録
    if writer is not None:
        writer.add_scalar("Eval/Loss", avg_loss, epoch)
        writer.add_scalar("Eval/Perplexity", perplexity, epoch)
        writer.add_scalar("Eval/Accuracy", avg_accuracy, epoch)
    
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "accuracy": avg_accuracy
    }


def train_mlm(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    num_epochs: int,
    save_dir: str,
    writer: Optional[SummaryWriter] = None,
    log_interval: int = 100,
    patience: int = 5
) -> None:
    """MLMモデルの学習
    
    Args:
        model: MLMモデル
        train_loader: 訓練データローダー
        val_loader: 検証データローダー
        optimizer: オプティマイザー
        criterion: 損失関数
        device: デバイス
        num_epochs: エポック数
        save_dir: モデル保存ディレクトリ
        writer: TensorBoard writer
        log_interval: ログ出力間隔（バッチ数）
        patience: 早期終了の忍耐エポック数
    """
    os.makedirs(save_dir, exist_ok=True)
    
    best_val_loss = float('inf')
    patience_counter = 0
    
    logger.info(f"学習開始: デバイス={device}, エポック数={num_epochs}")
    
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        
        # 学習
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, writer, log_interval
        )
        
        # 検証
        val_metrics = evaluate(model, val_loader, criterion, device, epoch, writer)
        
        epoch_time = time.time() - epoch_start_time
        
        # ログ出力
        logger.info(
            f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s) - "
            f"Train Loss: {train_metrics['loss']:.4f}, PPL: {train_metrics['perplexity']:.2f}, Acc: {train_metrics['accuracy']:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f}, PPL: {val_metrics['perplexity']:.2f}, Acc: {val_metrics['accuracy']:.4f}"
        )
        
        # ベストモデルの保存
        if val_metrics['loss'] < best_val_loss:
            best_val_loss = val_metrics['loss']
            patience_counter = 0
            
            # モデルを保存
            model.save_pretrained(save_dir)
            logger.info(f"  -> ベストモデルを保存しました（Val Loss: {best_val_loss:.4f}）")
        else:
            patience_counter += 1
            logger.info(f"  -> Val Lossが改善されませんでした（patience: {patience_counter}/{patience}）")
        
        # 早期終了
        if patience_counter >= patience:
            logger.info(f"早期終了: {epoch}エポックで学習を停止します")
            break
    
    logger.info("学習完了！")


def main():
    parser = argparse.ArgumentParser(description="MLM（Masked Language Modeling）の学習")
    
    # モデル設定
    parser.add_argument("--model-size", type=str, default="base", choices=["small", "base", "large"],
                        help="モデルサイズ")
    parser.add_argument("--max-seq-length", type=int, default=512, help="最大シーケンス長")
    
    # 学習設定
    parser.add_argument("--epochs", type=int, default=10, help="学習エポック数")
    parser.add_argument("--batch-size", type=int, default=32, help="バッチサイズ")
    parser.add_argument("--lr", type=float, default=5e-5, help="学習率")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="重み減衰")
    parser.add_argument("--warmup-steps", type=int, default=1000, help="ウォームアップステップ数")
    parser.add_argument("--patience", type=int, default=5, help="早期終了の忍耐エポック数")
    
    # 損失関数設定
    parser.add_argument("--label-smoothing", type=float, default=0.1, help="ラベルスムージング係数")
    
    # データ設定
    parser.add_argument("--tokenizer", type=str, default="bert-base-cased",
                        help="使用するトークナイザー")
    parser.add_argument("--mask-prob", type=float, default=0.15, help="MLMマスク確率")
    parser.add_argument("--cache-dir", type=str, default=None, help="データセットキャッシュディレクトリ")
    
    # 出力設定
    parser.add_argument("--save-dir", type=str, default="models/mlm_model", help="モデル保存ディレクトリ")
    parser.add_argument("--tensorboard-dir", type=str, default="runs/mlm_training",
                        help="TensorBoardログディレクトリ")
    parser.add_argument("--log-interval", type=int, default=100, help="ログ出力間隔（バッチ数）")
    
    # その他
    parser.add_argument("--seed", type=int, default=42, help="乱数シード")
    parser.add_argument("--num-workers", type=int, default=0, help="データローダーのワーカー数")
    parser.add_argument("--no-cuda", action="store_true", help="CUDAを使用しない")
    
    args = parser.parse_args()
    
    # 乱数シードを設定
    torch.manual_seed(args.seed)
    
    # デバイスを設定
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    logger.info(f"使用デバイス: {device}")
    
    # TensorBoard writerを作成
    writer = SummaryWriter(args.tensorboard_dir)
    logger.info(f"TensorBoardログ: {args.tensorboard_dir}")
    
    # データセットを読み込み
    logger.info("データセットを読み込み中...")
    data = load_wikitext2_for_mlm(
        tokenizer_name=args.tokenizer,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        mask_prob=args.mask_prob,
        cache_dir=args.cache_dir
    )
    
    tokenizer = data["tokenizer"]
    train_loader = data["dataloaders"]["train"]
    val_loader = data["dataloaders"]["validation"]
    vocab_size = data["vocab_size"]
    
    logger.info(f"語彙サイズ: {vocab_size}")
    logger.info(f"訓練データ: {len(train_loader.dataset)}件")
    logger.info(f"検証データ: {len(val_loader.dataset)}件")
    
    # モデルを作成
    logger.info(f"MLMモデルを作成（サイズ: {args.model_size}）...")
    model = create_mlm_model(
        vocab_size=vocab_size,
        model_size=args.model_size,
        max_seq_length=args.max_seq_length,
        pad_idx=tokenizer.pad_token_id
    )
    model = model.to(device)
    
    # パラメータ数を表示
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"パラメータ数: {num_params:,}（{num_params/1e6:.2f}M）")
    
    # 損失関数を作成
    criterion = MLMLoss(
        vocab_size=vocab_size,
        ignore_index=-100,
        label_smoothing=args.label_smoothing
    )
    
    # オプティマイザーを作成
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )
    
    # 学習を実行
    train_mlm(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        num_epochs=args.epochs,
        save_dir=args.save_dir,
        writer=writer,
        log_interval=args.log_interval,
        patience=args.patience
    )
    
    # TensorBoard writerを閉じる
    writer.close()
    
    logger.info(f"\nモデルを保存しました: {args.save_dir}")
    logger.info(f"TensorBoardで確認: tensorboard --logdir={args.tensorboard_dir}")


if __name__ == "__main__":
    main()
