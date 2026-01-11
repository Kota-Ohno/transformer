"""
モデルのチェックポイント処理（保存・読み込み）を担当します。
"""
import os
import glob
import torch
import logging
import traceback
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import torch.nn as nn
import torch.optim as optim
from utils.config import CONFIG

def setup_checkpointing_directory() -> str:
    """チェックポイント保存用のディレクトリを設定します"""
    checkpoint_dir = os.path.join("models", "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    return checkpoint_dir

def is_trusted_checkpoint_path(checkpoint_path: str) -> bool:
    """
    チェックポイントパスが信頼できるソースからのものかどうかを判定します。

    信頼できるソース:
    - models/checkpoints/ ディレクトリ内のファイル（自分たちが保存したチェックポイント）
    - models/ ディレクトリ内の .pth ファイル（自分たちが保存したモデル）

    Args:
        checkpoint_path: チェックポイントファイルのパス

    Returns:
        信頼できるパスの場合はTrue、そうでない場合はFalse
    """
    try:
        # シンボリックリンクを解決して正規化
        resolved = Path(checkpoint_path).resolve()

        # ファイルの存在と通常ファイルであることを確認
        if not resolved.exists() or not resolved.is_file():
            return False

        # 信頼できるディレクトリのリスト（シンボリックリンクを解決）
        trusted_dirs = [
            Path("models/checkpoints").resolve(),
            Path("models").resolve(),
        ]

        # チェックポイントファイルが信頼できるディレクトリ内にあるか確認
        for trusted_dir_obj in trusted_dirs:
            try:
                # 相対パスで判定（シンボリックリンク解決済み）
                resolved.relative_to(trusted_dir_obj)
                return True
            except ValueError:
                # 相対パスでない場合は次のディレクトリをチェック
                continue

        return False
    except Exception as e:
        logging.warning(f"チェックポイントパスの信頼性チェック中にエラーが発生しました: {e}")
        return False

def calculate_file_hash(file_path: str) -> Optional[str]:
    """
    ファイルのSHA256ハッシュを計算します。

    Args:
        file_path: ファイルのパス

    Returns:
        SHA256ハッシュ値（16進数文字列）、エラー時はNone
    """
    try:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # 大きなファイルでもメモリ効率的に処理
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception as e:
        logging.warning(f"ファイルハッシュの計算中にエラーが発生しました: {e}")
        return None

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
    model_num_layers: Optional[int] = None,
    best_valid_loss: Optional[float] = None,
    best_bleu: Optional[float] = None,
    last_valid_loss: Optional[float] = None,
    last_bleu: Optional[float] = None
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

    # 入力検証: best_valid_lossとbest_bleuがNoneの場合の処理
    # 真の履歴ベストメトリクスを上書きしないよう、警告をログに記録してフォールバックを明示
    if best_valid_loss is None:
        logging.warning(
            f"best_valid_lossがNoneです。現在のval_loss ({val_loss:.4f}) をフォールバックとして使用します。"
            f"エポック {epoch} のチェックポイント保存時に、履歴ベストメトリクスが正しく保存されない可能性があります。"
        )
        best_valid_loss = val_loss

    if best_bleu is None:
        logging.warning(
            f"best_bleuがNoneです。現在のbleu_score ({bleu_score:.4f}) をフォールバックとして使用します。"
            f"エポック {epoch} のチェックポイント保存時に、履歴ベストメトリクスが正しく保存されない可能性があります。"
        )
        best_bleu = bleu_score

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
        'val_loss': val_loss,  # 後方互換性のため保持
        'bleu_score': bleu_score,  # 後方互換性のため保持
        'date': current_date,
        'model_config': model_config,  # モデル設定情報を追加
        # bestとlastを分離して保存
        # 注意: best_valid_lossとbest_bleuは関数の開始部分で検証済み（Noneの場合は警告ログとフォールバック処理済み）
        'best_valid_loss': best_valid_loss,
        'best_bleu': best_bleu,
        'last_valid_loss': last_valid_loss if last_valid_loss is not None else val_loss,
        'last_bleu': last_bleu if last_bleu is not None else bleu_score
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
            # 現在のファイル名は"best_model.pth"だが、レガシーパターン"best_model_*.pth"もサポート
            try:
                # 現在のファイル名とレガシーパターンの両方をチェック
                best_model_path_current = os.path.join(checkpoint_dir, "best_model.pth")
                old_best_models = glob.glob(os.path.join(checkpoint_dir, "best_model_*.pth"))

                # 現在のファイル以外の古いファイルを削除
                for old_file in old_best_models:
                    # 現在のファイルと同じ場合はスキップ
                    if os.path.abspath(old_file) == os.path.abspath(best_model_path_current):
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
    scheduler: Optional[Any] = None,
    trusted_paths: Optional[list[str]] = None
) -> Dict[str, Any]:
    """
    チェックポイントからモデルを読み込みます。

    セキュリティ対策として、信頼できないソースからのチェックポイントは
    weights_only=Trueで読み込みます（pickleによる任意コード実行を防止）。

    Args:
        checkpoint_path: チェックポイントファイルのパス
        model: モデル
        optimizer: オプティマイザ（オプション）
        scheduler: スケジューラ（オプション）
        trusted_paths: 信頼できるパスのリスト（オプション、指定時はこのリストを優先）

    Returns:
        dict: チェックポイントの情報を含む辞書

    Raises:
        FileNotFoundError: チェックポイントファイルが見つからない場合
        RuntimeError: チェックポイントの読み込みに失敗した場合
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"チェックポイントファイルが見つかりません: {checkpoint_path}")

    try:
        logging.info(f"チェックポイントを読み込んでいます: {checkpoint_path}")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 信頼性チェック
        is_trusted = False
        if trusted_paths is not None:
            # 明示的に信頼できるパスが指定されている場合
            # シンボリックリンクを解決して比較（is_trusted_checkpoint_pathと一貫性を保つ）
            resolved_checkpoint_path = Path(checkpoint_path).resolve()
            # 各trusted_pathを解決し、チェックポイントパスがそのパスと等しいか、その中に含まれるかチェック
            for trusted_path in trusted_paths:
                resolved_trusted_path = Path(trusted_path).resolve(strict=False)
                # ファイルまたはディレクトリのいずれかとしてチェック
                if resolved_trusted_path.is_file():
                    # ファイルの場合は完全一致をチェック
                    if resolved_checkpoint_path == resolved_trusted_path:
                        is_trusted = True
                        break
                elif resolved_trusted_path.is_dir():
                    # ディレクトリの場合は、チェックポイントパスがそのディレクトリ内にあるかチェック
                    try:
                        resolved_checkpoint_path.relative_to(resolved_trusted_path)
                        is_trusted = True
                        break
                    except ValueError:
                        # 相対パスでない場合は次のパスをチェック
                        continue
        else:
            # デフォルトの信頼性チェック（models/checkpoints/ または models/ 内かどうか）
            is_trusted = is_trusted_checkpoint_path(checkpoint_path)

        # 信頼できないソースからの読み込みの場合は weights_only=True を使用
        if not is_trusted:
            logging.warning(
                f"信頼できないソースからのチェックポイント読み込みを検出しました: {checkpoint_path}\n"
                f"weights_only=True で読み込みます（model_config などのカスタムメタデータは読み込めません）。"
            )
            try:
                checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
            except Exception as weights_only_error:
                logging.error(
                    f"weights_only=True での読み込みに失敗しました: {weights_only_error}\n"
                    f"このチェックポイントは信頼できないソースからのものである可能性があります。"
                )
                raise RuntimeError(
                    f"信頼できないチェックポイントの読み込みに失敗しました: {weights_only_error}"
                ) from weights_only_error
        else:
            # 信頼できるソースからの読み込み（後方互換性のため通常の読み込み）
            logging.info(f"信頼できるソースからのチェックポイントを読み込みます: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)

        # チェックポイントの構造を確認
        if not isinstance(checkpoint, dict):
            raise ValueError(f"チェックポイントは辞書形式である必要があります。実際の型: {type(checkpoint)}")

        if 'model_state_dict' not in checkpoint:
            raise ValueError("チェックポイントに 'model_state_dict' が含まれていません。")

        # モデルの状態辞書を読み込み
        model.load_state_dict(checkpoint['model_state_dict'])

        # オプティマイザの状態辞書を読み込み
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

        # スケジューラの状態辞書を読み込み
        if scheduler is not None and checkpoint.get('scheduler_state_dict') is not None:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

        # メタデータを取得（信頼できないソースの場合は欠落している可能性がある）
        epoch = checkpoint.get('epoch', 0)
        val_loss = checkpoint.get('val_loss', float('inf'))
        bleu_score = checkpoint.get('bleu_score', 0.0)
        model_config = checkpoint.get('model_config', {})

        if not is_trusted and not model_config:
            logging.warning(
                "信頼できないソースからのチェックポイントのため、model_config は読み込めませんでした。"
                "デフォルト設定が使用されます。"
            )

        logging.info(f"チェックポイントを読み込みました (エポック {epoch}, 検証損失 {val_loss:.4f}, BLEU {bleu_score:.4f})")

        return {
            'epoch': epoch,
            'val_loss': val_loss,
            'bleu_score': bleu_score,
            'model_config': model_config
        }

    except FileNotFoundError:
        raise
    except Exception as e:
        logging.error(f"チェックポイントの読み込みに失敗しました: {e}")
        logging.error(traceback.format_exc())
        raise RuntimeError(f"チェックポイントの読み込みに失敗しました: {e}") from e

def find_latest_checkpoint() -> Optional[str]:
    """
    最新のチェックポイントを探します

    Returns:
        最新のチェックポイントパス、存在しない場合はNone
    """
    checkpoint_dir: str = os.path.join("models", "checkpoints")
    if not os.path.exists(checkpoint_dir):
        return None

    # 通常ファイルで、checkpointで始まり、.pth拡張子を持つもののみをフィルタリング
    checkpoints: list[str] = [
        os.path.join(checkpoint_dir, f)
        for f in os.listdir(checkpoint_dir)
        if f.startswith("checkpoint") and f.endswith(".pth") and os.path.isfile(os.path.join(checkpoint_dir, f))
    ]
    if not checkpoints:
        return None

    # 最新のファイルを見つける
    latest_checkpoint: str = max(checkpoints, key=os.path.getmtime)
    return latest_checkpoint
