"""
モデルのチェックポイント処理（保存・読み込み）を担当します。
"""
import os
import glob
import torch
import logging
import traceback
from datetime import datetime
from typing import Optional
from utils.config import MODEL_CONFIG

def setup_checkpointing_directory():
    """チェックポイント保存用のディレクトリを設定します"""
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def save_checkpoint(model, optimizer, scheduler, epoch, val_loss, bleu_score, is_best=False,
                  model_hidden_size=None, model_num_heads=None, model_num_layers=None):
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
        num_layers = MODEL_CONFIG.num_layers

    # 引数で渡された値があれば優先して使用
    hidden_size = model_hidden_size or MODEL_CONFIG.hidden_size
    num_heads = model_num_heads or MODEL_CONFIG.num_heads
    num_layers = model_num_layers or num_layers

    # モデル設定を辞書に保存
    model_config = {
        'HIDDEN_SIZE': hidden_size,
        'NUM_HEADS': num_heads,
        'NUM_LAYERS': num_layers,
        'D_FF': MODEL_CONFIG.d_ff,
        'DROPOUT_RATE': MODEL_CONFIG.dropout,
        'MAX_SEQ_LENGTH': MODEL_CONFIG.max_seq_length,
        'REL_POS_MAX_DISTANCE': MODEL_CONFIG.rel_pos_max_distance
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
            # 古いbest_model_*.pthファイルを削除
            old_best_models = glob.glob(os.path.join(checkpoint_dir, "best_model_*.pth"))
            for old_file in old_best_models:
                try:
                    os.remove(old_file)
                    logging.info(f"古い最良モデルファイルを削除しました: {old_file}")
                except Exception as e:
                    logging.warning(f"古い最良モデルファイルの削除に失敗しました: {old_file}, エラー: {e}")

            # 一時ファイルに保存（アトミックな保存）
            torch.save(checkpoint, temp_best_model_path)

            # 一時ファイルを正式なファイル名にリネーム（アトミック操作）
            os.replace(temp_best_model_path, best_model_path)

            logging.info(f"最良モデルを保存しました: {best_model_path}")
        except Exception as e:
            # 一時ファイルが残っている場合は削除を試みる
            if os.path.exists(temp_best_model_path):
                try:
                    os.remove(temp_best_model_path)
                except:
                    pass
            logging.error(f"最良モデルの保存に失敗しました: {best_model_path}")
            logging.error(f"エラー詳細: {e}")
            logging.error(traceback.format_exc())
            raise

def load_checkpoint(checkpoint_path, model, optimizer=None, scheduler=None):
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

        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
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
