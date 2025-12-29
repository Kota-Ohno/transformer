import time
import logging
import torch
import numpy as np
import random
from tqdm import tqdm
import os
import traceback
from datetime import datetime

from typing import Optional, Dict, Any, Tuple
from utils.config import CONFIG
from utils.evaluator import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint, setup_checkpointing_directory
from utils.scheduler import WarmupScheduler
from utils.constants import (
    CHECKPOINT_FREQUENCY_SHORT, CHECKPOINT_FREQUENCY_MEDIUM, CHECKPOINT_FREQUENCY_LONG,
    MAX_EPOCH_RETRIES
)
from utils.logging_config import setup_logging

# ロギング設定
setup_logging()

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, scaler, device, args, input_vocab, output_vocab):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.scaler = scaler
        self.device = device
        self.args = args
        self.input_vocab = input_vocab
        self.output_vocab = output_vocab

        self.best_valid_loss = float('inf')
        self.best_bleu = 0.0
        self.patience_counter = 0
        self.wandb_available = False
        self.max_epoch_retries = MAX_EPOCH_RETRIES

        # Weights & Biasesのセットアップ
        if not self.args.no_wandb:
            self._setup_wandb()

    def _setup_wandb(self):
        try:
            import wandb

            # batch_sizeを安全に解決
            batch_size = getattr(self.train_loader, "batch_size", None)

            if batch_size is None:
                try:
                    # 1バッチを取得してbatch_sizeを推論
                    batch = next(iter(self.train_loader))
                    if isinstance(batch, (list, tuple)) and len(batch) > 0:
                        batch_size = len(batch[0])
                    elif isinstance(batch, torch.Tensor):
                        batch_size = len(batch)
                    else:
                        raise ValueError(f"予期しないバッチ形式: {type(batch)}")
                except StopIteration:
                    logging.warning("train_loaderが空です。batch_sizeを推論できません。デフォルト値1を使用します")
                    batch_size = 1
                except Exception as e:
                    logging.warning(f"batch_sizeの推論中にエラーが発生しました: {e}。デフォルト値1を使用します")
                    batch_size = 1

            # batch_sizeがintであることを確認
            if not isinstance(batch_size, int):
                try:
                    batch_size = int(batch_size)
                except (ValueError, TypeError):
                    logging.warning(f"batch_sizeをintに変換できませんでした: {batch_size}。デフォルト値1を使用します")
                    batch_size = 1

            wandb_config = {
                "hidden_size": CONFIG.model_hyperparameters.hidden_size,
                "num_heads": CONFIG.model_hyperparameters.num_heads,
                "num_layers": CONFIG.model_hyperparameters.num_layers,
                "learning_rate": CONFIG.training_config.learning_rate,
                "batch_size": batch_size,
                "effective_batch_size": batch_size * CONFIG.training_config.gradient_accumulation_steps,
                "warmup_steps": self.args.warmup_steps,
                "model_type": "standard",
                "use_data_augmentation": self.args.augment,
                "augmentation_factor": self.args.augment_factor if self.args.augment else 0,
                "jit_compile": CONFIG.training_config.use_jit_compile,
                "grad_accum_steps": CONFIG.training_config.gradient_accumulation_steps
            }
            project_name = "transformer_training"
            wandb.init(project=project_name, config=wandb_config)
            model_name = "standard"
            wandb.run.name = f"{model_name}_h{CONFIG.model_hyperparameters.hidden_size}_l{CONFIG.model_hyperparameters.num_layers}_b{batch_size}"
            wandb.run.summary["model_architecture"] = f"{model_name}_h{CONFIG.model_hyperparameters.hidden_size}_l{CONFIG.model_hyperparameters.num_layers}"
            wandb.run.summary["input_vocab_size"] = len(self.input_vocab)
            wandb.run.summary["output_vocab_size"] = len(self.output_vocab)
            self.wandb_available = True
            logging.info("Weights & Biasesのログ記録を開始しました")
        except ImportError:
            logging.warning("wandbがインストールされていないため、W&Bのログ記録は無効になります")
            self.args.no_wandb = True
            self.wandb_available = False
        except Exception as e:
            logging.warning(f"W&Bの初期化エラー: {e}. ログ記録は無効になります")
            self.args.no_wandb = True
            self.wandb_available = False

    def _log_metrics(self, epoch, train_loss, val_loss, bleu_score):
        if not self.args.no_wandb and self.wandb_available:
            try:
                import wandb
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "bleu_score": bleu_score,
                    "learning_rate": self.scheduler.get_lr()[0] if self.scheduler and hasattr(self.scheduler, 'get_lr') else 0
                }
                wandb.log(metrics)
            except (ImportError, AttributeError):
                self.wandb_available = False
                self.args.no_wandb = True

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

    def _is_recoverable_error(self, exception):
        """回復可能なエラー（OOMなど）かどうかを判定"""
        error_str = str(exception).lower()
        # CUDA OOMエラーを検出
        if isinstance(exception, RuntimeError):
            if "out of memory" in error_str or "cuda" in error_str and "memory" in error_str:
                return True
        return False

    def _cleanup_after_error(self):
        """エラー後のクリーンアップ処理"""
        if self.device.type == 'cuda':
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        # オプティマイザーの状態をリセット
        self.optimizer.zero_grad(set_to_none=True)

    def _train_epoch(self):
        self.model.train()
        epoch_loss = 0
        total_batches = len(self.train_loader)
        accumulation_steps = CONFIG.training_config.gradient_accumulation_steps

        pbar = tqdm(enumerate(self.train_loader), total=total_batches, desc="Training")

        for i, batch in pbar:
            src, tgt_input, tgt_output = batch

            src = src.to(self.device, non_blocking=True)
            tgt_input = tgt_input.to(self.device, non_blocking=True)
            tgt_output = tgt_output.to(self.device, non_blocking=True)

            with torch.cuda.amp.autocast():
                output, _ = self.model(src, tgt_input)

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt_output = tgt_output.contiguous().view(-1)

                loss = self.criterion(output, tgt_output)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

            self.scaler.scale(loss).backward()

            if (i + 1) % accumulation_steps == 0 or (i + 1) == total_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.training_config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                if hasattr(self, "scheduler") and self.scheduler is not None:
                    self.scheduler.step()
                self.optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accumulation_steps

            current_lr = self.scheduler.get_lr()[0] if self.scheduler and hasattr(self.scheduler, 'get_lr') else 0.0
            pbar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "lr": f"{current_lr:.6f}"
            })

        return epoch_loss / total_batches

    def _load_checkpoint_if_needed(self) -> int:
        """
        必要に応じてチェックポイントを読み込みます。

        Returns:
            開始エポック番号
        """
        start_epoch = 0
        checkpoint_path = None

        if self.args.checkpoint:
            checkpoint_path = self.args.checkpoint
        elif self.args.resume:
            checkpoint_path = find_latest_checkpoint()

        if checkpoint_path and os.path.exists(checkpoint_path):
            try:
                logging.info(f"チェックポイント {checkpoint_path} からモデルを復元しています...")
                checkpoint_data = load_checkpoint(checkpoint_path, self.model, self.optimizer, self.scheduler)
                start_epoch = checkpoint_data.get('epoch', 0) + 1
                self.best_valid_loss = checkpoint_data.get('best_valid_loss', float('inf'))
                self.best_bleu = checkpoint_data.get('best_bleu', 0.0)
                logging.info(f"チェックポイントから復元完了: エポック {start_epoch}、最良検証損失 {self.best_valid_loss:.4f}")
            except Exception as e:
                logging.error(f"チェックポイントからの復元に失敗しました: {e}")

        return start_epoch

    def _determine_checkpoint_frequency(self) -> int:
        """
        チェックポイント保存頻度を決定します。

        Returns:
            チェックポイント保存頻度（エポック数）
        """
        if self.args.epochs <= 5:
            return CHECKPOINT_FREQUENCY_SHORT
        elif self.args.epochs <= 20:
            return CHECKPOINT_FREQUENCY_MEDIUM
        else:
            return CHECKPOINT_FREQUENCY_LONG

    def _process_epoch(self, epoch: int) -> Tuple[float, float]:
        """
        1エポックの処理を実行します。

        Args:
            epoch: エポック番号

        Returns:
            (valid_loss, bleu_score)のタプル
        """
        epoch_start_time = time.time()
        train_loss = self._train_epoch()
        valid_loss, bleu_score = evaluate(
            self.model, self.val_loader, self.criterion, self.device,
            CONFIG.training_config, self.output_vocab
        )
        epoch_minutes, epoch_seconds = self._epoch_time(epoch_start_time, time.time())

        logging.info(f"エポック: {epoch+1:02} | 所要時間: {epoch_minutes}m {epoch_seconds}s")
        logging.info(f"トレーニング損失: {train_loss:.4f} | 検証損失: {valid_loss:.4f} | BLEUスコア: {bleu_score:.4f}")

        self._log_metrics(epoch+1, train_loss, valid_loss, bleu_score)

        return valid_loss, bleu_score

    def _should_save_checkpoint(
        self,
        epoch: int,
        is_best: bool,
        save_checkpoint_frequency: int
    ) -> bool:
        """
        チェックポイントを保存すべきかどうかを判定します。

        Args:
            epoch: エポック番号
            is_best: 最良モデルかどうか
            save_checkpoint_frequency: チェックポイント保存頻度

        Returns:
            保存すべきかどうか
        """
        return (
            is_best or
            epoch == self.args.epochs - 1 or
            (epoch + 1) % save_checkpoint_frequency == 0
        )

    def _check_early_stopping(self) -> bool:
        """
        早期停止すべきかどうかをチェックします。

        Returns:
            早期停止すべきかどうか
        """
        if self.patience_counter >= CONFIG.training_config.patience:
            logging.info(f"{CONFIG.training_config.patience}エポックの間改善が見られないため、トレーニングを早期停止します")
            return True
        return False

    def train_model(self):
        """
        トレーニングループを実行します。
        """
        setup_checkpointing_directory()

        start_epoch = self._load_checkpoint_if_needed()
        logging.info(f"エポック {start_epoch + 1}/{self.args.epochs} からトレーニングを開始します...")

        save_checkpoint_frequency = self._determine_checkpoint_frequency()
        logging.info(f"チェックポイント保存頻度: {save_checkpoint_frequency}エポックごと")

        should_stop_training = False
        last_epoch = start_epoch

        for epoch in range(start_epoch, self.args.epochs):
            if should_stop_training:
                break

            retry_attempt = 0
            epoch_completed = False

            # エポック処理をリトライ可能なループで囲む
            while retry_attempt <= self.max_epoch_retries and not epoch_completed:
                try:
                    last_epoch = epoch + 1
                    valid_loss, bleu_score = self._process_epoch(epoch)

                    # 最良モデルの更新
                    is_best = False
                    if valid_loss < self.best_valid_loss:
                        self.best_valid_loss = valid_loss
                        self.patience_counter = 0
                        is_best = True
                        logging.info(f"最良の検証損失を更新: {self.best_valid_loss:.4f}")
                    else:
                        self.patience_counter += 1
                        logging.info(f"検証損失が改善されていません。忍耐カウンター: {self.patience_counter}/{CONFIG.training_config.patience}")

                    if bleu_score > self.best_bleu:
                        self.best_bleu = bleu_score
                        logging.info(f"最良のBLEUスコアを更新: {self.best_bleu:.4f}")

                    # チェックポイント保存
                    if self._should_save_checkpoint(epoch, is_best, save_checkpoint_frequency):
                        save_checkpoint(
                            model=self.model,
                            optimizer=self.optimizer,
                            scheduler=self.scheduler,
                            epoch=epoch,
                            val_loss=valid_loss,
                            bleu_score=bleu_score,
                            is_best=is_best,
                            model_hidden_size=CONFIG.model_hyperparameters.hidden_size,
                            model_num_heads=CONFIG.model_hyperparameters.num_heads,
                            model_num_layers=CONFIG.model_hyperparameters.num_layers
                        )
                    else:
                        logging.info(f"チェックポイント保存をスキップしました (頻度: {save_checkpoint_frequency}エポックごと)")

                    # 早期停止チェック
                    if self._check_early_stopping():
                        epoch_completed = True
                        should_stop_training = True
                        break

                    # 成功したらループを抜ける
                    epoch_completed = True

                    if self.device.type == 'cuda':
                        torch.cuda.empty_cache()

                except KeyboardInterrupt:
                    last_epoch = epoch + 1
                    logging.info("ユーザーによって中断されました。最終チェックポイントを保存します...")
                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        val_loss=valid_loss,
                        bleu_score=bleu_score,
                        is_best=False,
                        model_hidden_size=CONFIG.model_hyperparameters.hidden_size,
                        model_num_heads=CONFIG.model_hyperparameters.num_heads,
                        model_num_layers=CONFIG.model_hyperparameters.num_layers
                    )
                    epoch_completed = True
                    should_stop_training = True
                    break

                except Exception as e:
                    # 完全なトレースバックとコンテキストをログに記録
                    logging.error(f"エポック {epoch+1} の処理中にエラーが発生しました (リトライ試行: {retry_attempt}/{self.max_epoch_retries})")
                    logging.error(f"エラータイプ: {type(e).__name__}")
                    logging.error(f"エラーメッセージ: {str(e)}")
                    logging.error("完全なトレースバック:")
                    logging.error(traceback.format_exc())

                    # デバイス情報などのコンテキストを追加
                    if self.device.type == 'cuda':
                        logging.error(f"CUDAメモリ使用状況: 割り当て済み={torch.cuda.memory_allocated(self.device) / 1024**3:.2f}GB, "
                                     f"キャッシュ済み={torch.cuda.memory_reserved(self.device) / 1024**3:.2f}GB")

                    # 回復可能なエラーかどうかを判定
                    if self._is_recoverable_error(e):
                        if retry_attempt < self.max_epoch_retries:
                            logging.warning(f"回復可能なエラーを検出しました。クリーンアップ後にリトライします...")
                            self._cleanup_after_error()
                            retry_attempt += 1
                            # whileループが継続してリトライ
                        else:
                            logging.error(f"エポック {epoch+1} で最大リトライ回数 ({self.max_epoch_retries}) に達しました。"
                                         f"このエポックをスキップして次のエポックに進みます。")
                            epoch_completed = True  # リトライを諦めて次のエポックへ
                            break
                    else:
                        # 予期しない致命的なエラー: ログに記録して再発生
                        logging.error(f"致命的な予期しないエラーが発生しました。トレーニングを停止します。")
                        logging.error(f"エポック: {epoch+1}, リトライ試行: {retry_attempt}")
                        raise

        logging.info("トレーニングが完了しました")
        logging.info(f"最良の検証損失: {self.best_valid_loss:.4f}, 最良のBLEUスコア: {self.best_bleu:.4f}")

        if not self.args.no_wandb and self.wandb_available:
            try:
                import wandb
                wandb.run.summary["best_val_loss"] = self.best_valid_loss
                wandb.run.summary["best_bleu_score"] = self.best_bleu
                wandb.finish()
            except (ImportError, AttributeError):
                pass

        final_model_dir = "models"
        os.makedirs(final_model_dir, exist_ok=True)
        final_model_path = os.path.join(final_model_dir, f"final_model_epoch_{last_epoch}.pt")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'input_vocab': self.input_vocab,
            'output_vocab': self.output_vocab,
            'config': {
                'hidden_size': CONFIG.model_hyperparameters.hidden_size,
                'num_heads': CONFIG.model_hyperparameters.num_heads,
                'num_layers': CONFIG.model_hyperparameters.num_layers
            }
        }, final_model_path)
        logging.info(f"最終モデルを保存しました: {final_model_path}")
