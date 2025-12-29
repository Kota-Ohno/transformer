import time
import logging
import torch
import numpy as np
import random
from tqdm import tqdm
import os
import wandb
import traceback
from datetime import datetime

from utils.config import CONFIG
from utils.evaluator import evaluate
from utils.checkpoint import save_checkpoint, load_checkpoint, find_latest_checkpoint, setup_checkpointing_directory
from utils.scheduler import WarmupScheduler

# ロギング設定
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

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

        # Weights & Biasesのセットアップ
        if not self.args.no_wandb:
            self._setup_wandb()

    def _setup_wandb(self):
        try:
            import wandb
            wandb_config = {
                "hidden_size": CONFIG.model_hyperparameters.hidden_size,
                "num_heads": CONFIG.model_hyperparameters.num_heads,
                "num_layers": CONFIG.model_hyperparameters.num_layers,
                "learning_rate": CONFIG.training_config.learning_rate,
                "batch_size": self.train_loader.batch_size,
                "effective_batch_size": self.train_loader.batch_size * CONFIG.training_config.gradient_accumulation_steps,
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
            wandb.run.name = f"{model_name}_h{CONFIG.model_hyperparameters.hidden_size}_l{CONFIG.model_hyperparameters.num_layers}_b{self.train_loader.batch_size}"
            wandb.run.summary["model_architecture"] = f"{model_name}_h{CONFIG.model_hyperparameters.hidden_size}_l{CONFIG.model_hyperparameters.num_layers}"
            wandb.run.summary["input_vocab_size"] = len(self.input_vocab)
            wandb.run.summary["output_vocab_size"] = len(self.output_vocab)
            logging.info("Weights & Biasesのログ記録を開始しました")
        except ImportError:
            logging.warning("wandbがインストールされていないため、W&Bのログ記録は無効になります")
            self.args.no_wandb = True
        except Exception as e:
            logging.warning(f"W&Bの初期化エラー: {e}. ログ記録は無効になります")
            self.args.no_wandb = True

    def _log_metrics(self, epoch, train_loss, val_loss, bleu_score):
        if not self.args.no_wandb:
            metrics = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "bleu_score": bleu_score,
                "learning_rate": self.scheduler.get_lr()[0]
            }
            wandb.log(metrics)

    def _epoch_time(self, start_time, end_time):
        elapsed_time = end_time - start_time
        elapsed_mins = int(elapsed_time / 60)
        elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
        return elapsed_mins, elapsed_secs

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
                self.optimizer.zero_grad(set_to_none=True)

            epoch_loss += loss.item() * accumulation_steps

            pbar.set_postfix({
                "loss": f"{loss.item() * accumulation_steps:.4f}",
                "lr": f"{self.scheduler.get_lr()[0]:.6f}"
            })

        return epoch_loss / total_batches

    def train_model(self):
        setup_checkpointing_directory()

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

        logging.info(f"エポック {start_epoch + 1}/{self.args.epochs} からトレーニングを開始します...")

        if self.args.epochs <= 5:
            save_checkpoint_frequency = 1
        elif self.args.epochs <= 20:
            save_checkpoint_frequency = 2
        else:
            save_checkpoint_frequency = 5
        logging.info(f"チェックポイント保存頻度: {save_checkpoint_frequency}エポックごと")

        for epoch in range(start_epoch, self.args.epochs):
            try:
                epoch_start_time = time.time()

                train_loss = self._train_epoch()
                valid_loss, bleu_score = evaluate(self.model, self.val_loader, self.criterion, self.device, CONFIG.training_config, self.output_vocab)

                epoch_minutes, epoch_seconds = self._epoch_time(epoch_start_time, time.time())

                logging.info(f"エポック: {epoch+1:02} | 所要時間: {epoch_minutes}m {epoch_seconds}s")
                logging.info(f"トレーニング損失: {train_loss:.4f} | 検証損失: {valid_loss:.4f} | BLEUスコア: {bleu_score:.4f}")

                self._log_metrics(epoch+1, train_loss, valid_loss, bleu_score)

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

                should_save_checkpoint = (
                    is_best or
                    epoch == self.args.epochs - 1 or
                    (epoch + 1) % save_checkpoint_frequency == 0
                )

                if should_save_checkpoint:
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

                if self.patience_counter >= CONFIG.training_config.patience:
                    logging.info(f"{CONFIG.training_config.patience}エポックの間改善が見られないため、トレーニングを早期停止します")
                    break

                if self.device.type == 'cuda':
                    torch.cuda.empty_cache()

            except KeyboardInterrupt:
                logging.info("ユーザーによって中断されました。最終チェックポイントを保存します...")
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    scheduler=self.scheduler,
                    epoch=epoch,
                    val_loss=valid_loss if 'valid_loss' in locals() else float('inf'),
                    bleu_score=bleu_score if 'bleu_score' in locals() else 0.0,
                    is_best=False,
                    model_hidden_size=CONFIG.model_hyperparameters.hidden_size,
                    model_num_heads=CONFIG.model_hyperparameters.num_heads,
                    model_num_layers=CONFIG.model_hyperparameters.num_layers
                )
                break

            except Exception as e:
                logging.error(f"エポック {epoch+1} の処理中にエラーが発生しました: {e}")
                traceback.print_exc()
                continue

        logging.info("トレーニングが完了しました")
        logging.info(f"最良の検証損失: {self.best_valid_loss:.4f}, 最良のBLEUスコア: {self.best_bleu:.4f}")

        if not self.args.no_wandb:
            try:
                wandb.run.summary["best_val_loss"] = self.best_valid_loss
                wandb.run.summary["best_bleu_score"] = self.best_bleu
                wandb.finish()
            except:
                pass

        final_model_path = os.path.join("models", f"final_model_epoch_{self.args.epochs}.pt")
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