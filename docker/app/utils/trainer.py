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

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, criterion, scheduler, scaler, device, args, input_vocab, output_vocab):
        # ロギング設定を最初に実行（_setup_wandb()でログ出力が行われるため）
        setup_logging()

        self.model = model
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
        # 最新の検証済みメトリクスを保存（KeyboardInterrupt時のチェックポイント保存用）
        self.last_valid_loss = None
        self.last_bleu = None

        # 致命的エラー: 空のtrain_loaderを検証（リトライループの外で検証）
        # これは回復不可能なエラーなので、リトライループで捕捉されないようにする
        if train_loader is None:
            error_msg = (
                "train_loaderがNoneです。データが読み込まれていないか、"
                "DataLoaderの設定に問題があります。データセットのパスやフィルタリング条件を確認してください。"
            )
            logging.error(error_msg)
            raise ValueError(error_msg)

        # IterableDatasetベースのDataLoaderに対応した空チェック
        try:
            if hasattr(train_loader, "__len__"):
                # __len__メソッドが存在する場合（通常のDataset）
                if len(train_loader) == 0:
                    error_msg = (
                        "train_loaderが空です。データが読み込まれていないか、"
                        "DataLoaderの設定に問題があります。データセットのパスやフィルタリング条件を確認してください。"
                    )
                    logging.error(error_msg)
                    raise ValueError(error_msg)
                # 通常のDatasetの場合、そのまま使用
                self.train_loader = train_loader
            else:
                # __len__メソッドが存在しない場合（IterableDataset）
                # イテレータを作成して空かどうかをチェック
                # 空チェックのみを行い、元のtrain_loaderを保持する
                # （各エポックで新しいイテレータを作成できるようにする）
                import itertools
                iterator = iter(train_loader)
                try:
                    # 最初のバッチを取得して空でないことを確認
                    first_item = next(iterator)
                    # 最初のアイテムを失わないように、最初のアイテムと残りのイテレータを結合
                    # 各エポックで新しいイテレータを作成できるようにラッパーを作成
                    class IterableWrapper:
                        def __init__(self, loader, first_item, remaining_iterator):
                            self.loader = loader
                            self.first_item = first_item
                            self.remaining_iterator = remaining_iterator
                            self._first_iteration = True

                        def __iter__(self):
                            # 最初のイテレーションでは、保存した最初のアイテムと残りのイテレータを使用
                            if self._first_iteration:
                                self._first_iteration = False
                                return itertools.chain([self.first_item], self.remaining_iterator)
                            else:
                                # 2回目以降は、元のloaderから新しいイテレータを作成
                                return iter(self.loader)

                    self.train_loader = IterableWrapper(train_loader, first_item, iterator)
                except StopIteration:
                    error_msg = (
                        "train_loaderが空です。データが読み込まれていないか、"
                        "DataLoaderの設定に問題があります。データセットのパスやフィルタリング条件を確認してください。"
                    )
                    logging.error(error_msg)
                    raise ValueError(error_msg)
        except TypeError as e:
            # TypeErrorが発生した場合も同じエラーメッセージで処理
            error_msg = (
                "train_loaderが空です。データが読み込まれていないか、"
                "DataLoaderの設定に問題があります。データセットのパスやフィルタリング条件を確認してください。"
            )
            logging.error(error_msg)
            raise ValueError(error_msg) from e

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
                    # DataLoaderのdataset属性からbatch_sizeを推論
                    if hasattr(self.train_loader, 'dataset') and hasattr(self.train_loader.dataset, '__len__'):
                        dataset_len = len(self.train_loader.dataset)
                        # IterableDatasetの場合は__len__が存在しない可能性があるため、明示的にチェック
                        if hasattr(self.train_loader, '__len__'):
                            num_batches = len(self.train_loader)
                            if num_batches > 0:
                                batch_size = max(1, dataset_len // num_batches)
                            else:
                                batch_size = 1
                        else:
                            # __len__が存在しない場合（IterableDataset）は、CONFIGから取得
                            batch_size = CONFIG.training_config.batch_size if hasattr(CONFIG.training_config, 'batch_size') else 1
                    else:
                        batch_size = CONFIG.training_config.batch_size if hasattr(CONFIG.training_config, 'batch_size') else 1
                except Exception as e:
                    logging.warning(f"batch_sizeの推論中にエラーが発生しました: {e}。CONFIGから取得します")
                    batch_size = CONFIG.training_config.batch_size if hasattr(CONFIG.training_config, 'batch_size') else 1

            # batch_sizeがintであることを確認
            if not isinstance(batch_size, int):
                try:
                    batch_size = int(batch_size)
                except (ValueError, TypeError):
                    logging.warning(f"batch_sizeをintに変換できませんでした: {batch_size}。デフォルト値1を使用します")
                    batch_size = 1

            # 学習率はCLI引数を優先し、Noneの場合はCONFIGから取得
            learning_rate = self.args.learning_rate if self.args.learning_rate is not None else CONFIG.training_config.learning_rate
            wandb_config = {
                "hidden_size": CONFIG.model_hyperparameters.hidden_size,
                "num_heads": CONFIG.model_hyperparameters.num_heads,
                "num_layers": CONFIG.model_hyperparameters.num_layers,
                "learning_rate": learning_rate,
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
            # wandb.runがNoneでないことを確認してからアクセス
            if wandb.run is not None:
                model_name = "standard"
                wandb.run.name = f"{model_name}_h{CONFIG.model_hyperparameters.hidden_size}_l{CONFIG.model_hyperparameters.num_layers}_b{batch_size}"
                wandb.run.summary["model_architecture"] = f"{model_name}_h{CONFIG.model_hyperparameters.hidden_size}_l{CONFIG.model_hyperparameters.num_layers}"
                wandb.run.summary["input_vocab_size"] = len(self.input_vocab)
                wandb.run.summary["output_vocab_size"] = len(self.output_vocab)
                self.wandb_available = True
                logging.info("Weights & Biasesのログ記録を開始しました")
            else:
                logging.warning("wandb.init()が成功しましたが、wandb.runがNoneです。W&Bのログ記録は無効になります")
                self.wandb_available = False
                self.args.no_wandb = True
        except ImportError:
            logging.warning("wandbがインストールされていないため、W&Bのログ記録は無効になります")
            self.args.no_wandb = True
            self.wandb_available = False
        except Exception as e:
            logging.warning(f"W&Bの初期化エラー: {e}. ログ記録は無効になります")
            self.args.no_wandb = True
            self.wandb_available = False

    def _get_current_lr(self) -> float:
        """現在の学習率を安全に取得するヘルパーメソッド。

        Returns:
            現在の学習率（float）。schedulerが存在しないか、get_lr()が空のリストを返す場合は0.0。
        """
        if not self.scheduler or not hasattr(self.scheduler, 'get_lr'):
            return 0.0

        try:
            lr_list = self.scheduler.get_lr()
            if lr_list:
                return float(lr_list[0])
            return 0.0
        except (IndexError, TypeError, AttributeError):
            return 0.0

    def _log_metrics(self, epoch, train_loss, val_loss, bleu_score):
        if not self.args.no_wandb and self.wandb_available:
            try:
                import wandb
                metrics = {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "bleu_score": bleu_score,
                    "learning_rate": self._get_current_lr()
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
        if not isinstance(exception, RuntimeError):
            return False
        error_str = str(exception).lower()
        # CUDA OOMエラーのみを検出（明示的なフレーズをチェック）
        if "out of memory" in error_str or "cuda out of memory" in error_str:
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

        # IterableDatasetの場合、len()がTypeErrorを発生させる可能性がある
        try:
            total_batches = len(self.train_loader)
        except (TypeError, AttributeError):
            # __len__が存在しない場合（IterableDataset）
            total_batches = None

        # 注意: 空のtrain_loaderのチェックは__init__で既に実行済み
        # ここではtotal_batchesが0になることはないはずだが、念のため確認
        if total_batches is not None and total_batches == 0:
            # これは通常発生しないはず（__init__で検証済み）
            # しかし、実行時にデータが削除された場合などに備えてエラーを発生
            raise RuntimeError(
                "train_loaderが空です。これは予期しない状態です。"
                "データが実行中に削除された可能性があります。"
            )

        accumulation_steps = CONFIG.training_config.gradient_accumulation_steps

        pbar = tqdm(enumerate(self.train_loader), total=total_batches, desc="Training")

        # IterableDatasetの場合、バッチ数をカウントする
        batch_count = 0

        for i, batch in pbar:
            batch_count += 1
            src, tgt_input, tgt_output = batch

            src = src.to(self.device, non_blocking=True)
            tgt_input = tgt_input.to(self.device, non_blocking=True)
            tgt_output = tgt_output.to(self.device, non_blocking=True)

            device = self.device
            with torch.amp.autocast(device_type=device.type):
                output, _ = self.model(src, tgt_input)

                output_dim = output.shape[-1]
                output = output.contiguous().view(-1, output_dim)
                tgt_output = tgt_output.contiguous().view(-1)

                loss = self.criterion(output, tgt_output)
                if accumulation_steps > 1:
                    loss = loss / accumulation_steps

            # 勾配スケーラーを使用する場合としない場合で処理を分岐
            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                # total_batchesがNoneの場合、最後のバッチかどうかを判定できないため、
                # 累積ステップの条件のみで判定する
                if (i + 1) % accumulation_steps == 0 or (total_batches is not None and (i + 1) == total_batches):
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.training_config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    if hasattr(self, "scheduler") and self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)
            else:
                # CPU環境またはGradScalerが利用できない場合
                loss.backward()

                # total_batchesがNoneの場合、最後のバッチかどうかを判定できないため、
                # 累積ステップの条件のみで判定する
                if (i + 1) % accumulation_steps == 0 or (total_batches is not None and (i + 1) == total_batches):
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.training_config.grad_clip_norm)
                    self.optimizer.step()
                    if hasattr(self, "scheduler") and self.scheduler is not None:
                        self.scheduler.step()
                    self.optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accumulation_steps

            current_lr = self._get_current_lr()
            pbar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "lr": f"{current_lr:.6f}"
            })

        # total_batchesがNone（IterableDataset）の場合、残りの勾配を適用
        if total_batches is None and batch_count % accumulation_steps != 0:
            # 残りの勾配を適用
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.training_config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), CONFIG.training_config.grad_clip_norm)
                self.optimizer.step()
            # スケジューラーのステップ
            if hasattr(self, "scheduler") and self.scheduler is not None:
                self.scheduler.step()
            self.optimizer.zero_grad(set_to_none=True)

        # total_batchesがNoneの場合、ループ内でカウントしたバッチ数を使用
        actual_batch_count = total_batches if total_batches is not None else batch_count
        if actual_batch_count > 0:
            return epoch_loss / actual_batch_count
        else:
            # バッチ数が0の場合（通常は発生しない）
            return 0.0

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
                # bestとlastを分離して復元
                # 新しいキーが存在する場合は優先、存在しない場合は古いキーにフォールバック
                if 'best_valid_loss' in checkpoint_data:
                    self.best_valid_loss = checkpoint_data['best_valid_loss']
                else:
                    # 後方互換性のため、古いキー名（val_loss）を使用
                    self.best_valid_loss = checkpoint_data.get('val_loss', float('inf'))
                if 'best_bleu' in checkpoint_data:
                    self.best_bleu = checkpoint_data['best_bleu']
                else:
                    # 後方互換性のため、古いキー名（bleu_score）を使用
                    self.best_bleu = checkpoint_data.get('bleu_score', 0.0)
                # last値の復元（存在しない場合はbest値を使用）
                self.last_valid_loss = checkpoint_data.get('last_valid_loss', self.best_valid_loss)
                self.last_bleu = checkpoint_data.get('last_bleu', self.best_bleu)
                logging.info(f"チェックポイントから復元完了: エポック {start_epoch}、最良検証損失 {self.best_valid_loss:.4f}、最新検証損失 {self.last_valid_loss:.4f}")
            except Exception as e:
                logging.exception(f"チェックポイントからの復元に失敗しました: {e}")
                raise

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

        # 最新の検証済みメトリクスを保存（KeyboardInterrupt時のチェックポイント保存用）
        self.last_valid_loss = valid_loss
        self.last_bleu = bleu_score

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
        # ロギング設定は __init__ で既に実行済み
        setup_checkpointing_directory()

        start_epoch = self._load_checkpoint_if_needed()
        logging.info(f"エポック {start_epoch + 1}/{self.args.epochs} からトレーニングを開始します...")

        save_checkpoint_frequency = self._determine_checkpoint_frequency()
        logging.info(f"チェックポイント保存頻度: {save_checkpoint_frequency}エポックごと")

        should_stop_training = False
        last_epoch = start_epoch
        # 訓練ループ開始前にデフォルト値を初期化（KeyboardInterrupt時のNameErrorを防ぐ）
        valid_loss = float('inf')
        bleu_score = 0.0

        for epoch in range(start_epoch, self.args.epochs):
            if should_stop_training:
                break

            retry_attempt = 0
            epoch_completed = False

            # エポック処理をリトライ可能なループで囲む
            # max_epoch_retriesは最大試行回数（例: 3なら最大3回試行）
            # retry_attemptは0から始まり、各リトライでインクリメントされる
            while retry_attempt < self.max_epoch_retries and not epoch_completed:
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
                            model_num_layers=CONFIG.model_hyperparameters.num_layers,
                            best_valid_loss=self.best_valid_loss,
                            best_bleu=self.best_bleu,
                            last_valid_loss=self.last_valid_loss,
                            last_bleu=self.last_bleu
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
                        try:
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except (RuntimeError, AssertionError) as e:
                            logging.warning(f"CUDAキャッシュの解放に失敗しました: {e}")

                except KeyboardInterrupt:
                    last_epoch = epoch + 1
                    logging.info("ユーザーによって中断されました。最終チェックポイントを保存します...")
                    # 最新の検証済みメトリクスを使用（存在しない場合はデフォルト値にフォールバック）
                    interrupt_valid_loss = getattr(self, 'last_valid_loss', None)
                    interrupt_bleu_score = getattr(self, 'last_bleu', None)

                    if interrupt_valid_loss is None:
                        interrupt_valid_loss = float('inf')
                        logging.warning("検証済みメトリクスが存在しないため、デフォルト値を使用します")
                    if interrupt_bleu_score is None:
                        interrupt_bleu_score = 0.0
                        logging.warning("検証済みメトリクスが存在しないため、デフォルト値を使用します")

                    save_checkpoint(
                        model=self.model,
                        optimizer=self.optimizer,
                        scheduler=self.scheduler,
                        epoch=epoch,
                        val_loss=interrupt_valid_loss,
                        bleu_score=interrupt_bleu_score,
                        is_best=False,
                        model_hidden_size=CONFIG.model_hyperparameters.hidden_size,
                        model_num_heads=CONFIG.model_hyperparameters.num_heads,
                        model_num_layers=CONFIG.model_hyperparameters.num_layers,
                        best_valid_loss=self.best_valid_loss,
                        best_bleu=self.best_bleu,
                        last_valid_loss=self.last_valid_loss,
                        last_bleu=self.last_bleu
                    )
                    epoch_completed = True
                    should_stop_training = True
                    break

                except Exception as e:
                    # 完全なトレースバックとコンテキストをログに記録
                    logging.error(f"エポック {epoch+1} の処理中にエラーが発生しました (リトライ試行: {retry_attempt + 1}/{self.max_epoch_retries})")
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
                        # リトライ試行回数を即座にインクリメント
                        retry_attempt += 1
                        if retry_attempt >= self.max_epoch_retries:
                            # リトライ回数が尽きた場合: チェックポイントを保存してから停止
                            logging.error(f"エポック {epoch+1} で最大リトライ回数 ({self.max_epoch_retries}) に達しました。"
                                         f"チェックポイントを保存してトレーニングを停止します。")
                            try:
                                save_checkpoint(
                                    model=self.model,
                                    optimizer=self.optimizer,
                                    scheduler=self.scheduler,
                                    epoch=epoch,
                                    val_loss=self.last_valid_loss if self.last_valid_loss is not None else float('inf'),
                                    bleu_score=self.last_bleu if self.last_bleu is not None else 0.0,
                                    is_best=False,
                                    model_hidden_size=CONFIG.model_hyperparameters.hidden_size,
                                    model_num_heads=CONFIG.model_hyperparameters.num_heads,
                                    model_num_layers=CONFIG.model_hyperparameters.num_layers,
                                    best_valid_loss=self.best_valid_loss,
                                    best_bleu=self.best_bleu,
                                    last_valid_loss=self.last_valid_loss,
                                    last_bleu=self.last_bleu
                                )
                                logging.info(f"OOMエラー後のチェックポイントを保存しました（エポック {epoch+1}）")
                            except Exception as checkpoint_error:
                                logging.error(f"チェックポイントの保存に失敗しました: {checkpoint_error}")
                            # 元の例外を再発生させてトレーニングを停止
                            raise
                        else:
                            # リトライ可能な場合: クリーンアップしてループを継続
                            logging.warning(f"回復可能なエラーを検出しました（リトライ試行: {retry_attempt}/{self.max_epoch_retries}）。クリーンアップ後にリトライします...")
                            self._cleanup_after_error()
                            # continueでループを継続してリトライ
                            continue
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
                if hasattr(wandb, "run") and wandb.run is not None:
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
