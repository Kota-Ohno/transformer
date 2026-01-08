"""
モデルのチェックポイント処理（保存・読み込み）を担当します。
"""
import os
import glob
import torch
import logging
import traceback
from datetime import datetime
from typing import Optional, Dict, Any
import torch.nn as nn
import torch.optim as optim
from utils.config import CONFIG

def setup_checkpointing_directory() -> str:
    """チェックポイント保存用のディレクトリを設定します"""
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    epoch: int,
    val_loss: float,
    bleu_score: float,
    is_best: bool = False,
    model_hidden_size: Optional[int] = None,
    model_num_heads: Optional[int] = None,
    model_num_layers: Optional[int] = None
) -> None:
    """
    モデルのチェックポイントを保存します

    Args:
        model: モデル
        optimizer: オプティマイザ
        scheduler: スケジューラ
        epoch: 現在のエポック
        val_loss: 検証損失
        bleu_score: BLEUスコア
        is_best: 最良モデルかどうか
        model_hidden_size: 実際に使用した隠れ層のサイズ
        model_num_heads: 実際に使用したアテンションヘッドの数
        model_num_layers: 実際に使用したレイヤー数
    """
    checkpoint_dir = setup_checkpointing_directory()

    # 現在の日付を取得
    current_date = datetime.now().strftime("%Y%m%d")

    # モデル設定情報を取得
    if hasattr(model, 'encoder') and hasattr(model.encoder, 'layers'):
        num_layers = len(model.encoder.layers)
    else:
        num_layers = CONFIG.model_hyperparameters.num_layers

    # 引数で渡された値があれば優先して使用
    hidden_size = model_hidden_size or CONFIG.model_hyperparameters.hidden_size
    num_heads = model_num_heads or CONFIG.model_hyperparameters.num_heads
    num_layers = model_num_layers or num_layers

    # モデル設定を辞書に保存
    model_config = {
        'HIDDEN_SIZE': hidden_size,
        'NUM_HEADS': num_heads,
        'NUM_LAYERS': num_layers,
        'D_FF': CONFIG.model_hyperparameters.d_ff,
        'DROPOUT_RATE': CONFIG.model_hyperparameters.dropout_rate,
        'MAX_SEQ_LENGTH': CONFIG.model_hyperparameters.max_seq_length,
        'REL_POS_MAX_DISTANCE': CONFIG.model_hyperparameters.rel_pos_max_distance
    }

    # チェックポイント情報を準備
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'val_loss': val_loss,
        'bleu_score': bleu_score,
        'date': current_date,
        'model_config': model_config  # モデル設定情報を追加
    }

    # 定期的なチェックポイントを保存
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_{current_date}.pth")
    try:
        torch.save(checkpoint, checkpoint_path)
        logging.info(f"チェックポイントを保存しました: {checkpoint_path}")
    except Exception as e:
        logging.error(f"チェックポイントの保存に失敗しました: {checkpoint_path}")
        logging.error(f"エラー詳細: {e}")
        logging.error(traceback.format_exc())
        raise

    # 最良モデルの場合は別名で保存（同じcheckpointsディレクトリに保存）
    if is_best:
        best_model_path = os.path.join(checkpoint_dir, "best_model.pth")
        temp_best_model_path = os.path.join(checkpoint_dir, "best_model.pth.tmp")

        try:
            # 一時ファイルに保存（アトミックな保存）
            torch.save(checkpoint, temp_best_model_path)

            # 一時ファイルを正式なファイル名にリネーム（アトミック操作）
            os.replace(temp_best_model_path, best_model_path)

            # リネームが成功したことを確認
            if not os.path.exists(best_model_path):
                raise RuntimeError(f"アトミック保存後のファイル確認に失敗しました: {best_model_path}")

            logging.info(f"最良モデルを保存しました: {best_model_path}")

            # アトミック保存が成功した後、古いbest_model_*.pthファイルを削除
            try:
                old_best_models = glob.glob(os.path.join(checkpoint_dir, "best_model_*.pth"))
                for old_file in old_best_models:
                    # 新しく書き込んだファイルはスキップ
                    if os.path.abspath(old_file) == os.path.abspath(best_model_path):
                        continue
                    try:
                        os.remove(old_file)
                        logging.info(f"古い最良モデルファイルを削除しました: {old_file}")
                    except Exception as e:
                        logging.warning(f"古い最良モデルファイルの削除に失敗しました: {old_file}, エラー: {e}")
            except Exception as cleanup_error:
                logging.warning(f"古い最良モデルファイルの列挙・削除処理中にエラーが発生しました: {cleanup_error}")

        except Exception as e:
            # 一時ファイルが残っている場合は削除を試みる
            if os.path.exists(temp_best_model_path):
                try:
                    os.remove(temp_best_model_path)
                except Exception as cleanup_error:
                    logging.warning(f"一時ファイルの削除に失敗しました: {temp_best_model_path}, エラー: {cleanup_error}")
            logging.error(f"最良モデルの保存に失敗しました: {best_model_path}")
            logging.error(f"エラー詳細: {e}")
            logging.error(traceback.format_exc())
            raise

def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    scheduler: Optional[Any] = None
) -> Dict[str, Any]:
    """
    チェックポイントからモデルを読み込みます

    Args:
        checkpoint_path: チェックポイントファイルのパス
        model: モデル
        optimizer: オプティマイザ（オプション）
        scheduler: スケジューラ（オプション）

    Returns:
        dict: チェックポイントの情報を含む辞書
    """
    try:
        logging.info(f"チェックポイントを読み込んでいます: {checkpoint_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        checkpoint = torch.load(checkpoint_path, map_location=device)

        model.load_state_dict(checkpoint['model_state_dict'])

        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        bleu_score = checkpoint.get('bleu_score', 0.0)

        # モデル設定情報を取得
        model_config = checkpoint.get('model_config', {})

        logging.info(f"チェックポイントを読み込みました (エポック {epoch}, 検証損失 {val_loss:.4f}, BLEU {bleu_score:.4f})")

        return {
            'epoch': epoch,
            'best_valid_loss': val_loss,
            'best_bleu': bleu_score,
            'last_valid_loss': val_loss,  # 最新の検証損失（そのエポックの値）
            'last_bleu': bleu_score,  # 最新のBLEUスコア（そのエポックの値）
            'model_config': model_config
        }

    except Exception as e:
        logging.error(f"チェックポイントの読み込みに失敗しました: {e}")
        logging.error(traceback.format_exc())
        raise

def find_latest_checkpoint() -> Optional[str]:
    """
    最新のチェックポイントを探します

    Returns:
        最新のチェックポイントパス、存在しない場合はNone
    """
    checkpoint_dir: str = os.path.join("models", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints: list[str] = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.startswith("checkpoint")]
    if not checkpoints:
        return None

    # 最新のファイルを見つける
    latest_checkpoint: str = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint
