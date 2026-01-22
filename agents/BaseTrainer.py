import pytorch_lightning as pl
import torch
import wandb
from pytorch_lightning.callbacks import (
    LearningRateMonitor,
    RichProgressBar,
    EarlyStopping,
)

from models.BaseModule import BaseModule
from utils.agent_utils import get_artifact, get_datamodule
from utils.callbacks import (
    AutoSaveModelCheckpoint,
    LogMetricsCallback,
    LogAudioPrediction,
    ConditionalTransformerUnfreezing
)
from utils.dataset_utils import create_tinyvox_vocabulary
from utils.logger import init_logger
from pathlib import Path


class BaseTrainer:
    def __init__(self, config, run_name, wb_run=None) -> None:
        self.config = config.hparams
        self.wb_run = wb_run
        self.run_name = run_name
        self.network_param = config.network_param

        self.logger = init_logger("BaseTrainer", "INFO")

        self.logger.info(f"Create vocabulary language for TinyVox ...")

        (
            config.network_param.vocab_file,
            config.network_param.len_vocab,
        ) = create_tinyvox_vocabulary(
            config.data_param.inventory_path,
            eos_token=config.network_param.eos_token,
            bos_token=config.network_param.bos_token,
            unk_token=config.network_param.unk_token,
            pad_token=config.network_param.pad_token,
            word_delimiter_token=config.network_param.word_delimiter_token
        )
        self.logger.info(f"Vocabulary file : {config.network_param.vocab_file}")

        self.logger.info("Loading Data module...")
        self.datamodule = get_datamodule(config.data_param)

        self.logger.info("Loading Model module...")
        self.pl_model = BaseModule(config.network_param, config.optim_param)



    def run(self):
        if self.config.tune_lr:
            tune_lr_trainer = pl.Trainer(
                logger=self.wb_run,
                devices=self.config.gpu,
                auto_lr_find=True,
                accelerator="auto",
                default_root_dir=self.wb_run.save_dir,
            )
            tune_lr_trainer.logger = self.wb_run

        if not self.config.debug_pytorch:
            torch.autograd.set_detect_anomaly(False)
            torch.autograd.profiler.profile(False)
            torch.autograd.profiler.emit_nvtx(False)
            torch.backends.cudnn.benchmark = True

        # Check for existing checkpoint
        checkpoint_dir = f"{self.config.weights_path}/{self.run_name}"
        latest_checkpoint = self._find_latest_checkpoint(checkpoint_dir)

        trainer = pl.Trainer(
            logger=self.wb_run,  # W&B integration
            callbacks=self.get_callbacks(),
            accelerator='gpu',
            devices=self.config.gpu,
            max_epochs=self.config.max_epochs,  # number of epochs
            log_every_n_steps=1,
            fast_dev_run=self.config.dev_run,
            precision=self.config.precision,
            enable_progress_bar=self.config.enable_progress_bar,
            val_check_interval=self.config.val_check_interval,
            limit_train_batches=self.config.limit_train_batches,
            limit_val_batches=self.config.limit_val_batches,
            accumulate_grad_batches=self.config.accumulate_grad_batches,
        )

        trainer.logger = self.wb_run

        self.datamodule.set_processor(self.pl_model.processor)

        if self.config.tune_lr:
            tune_lr_trainer.tune(self.pl_model, datamodule=self.datamodule)

        trainer.fit(self.pl_model, datamodule=self.datamodule, ckpt_path=latest_checkpoint)

    def get_callbacks(self):
        callbacks = [
            LearningRateMonitor(),
            LogMetricsCallback(),
            LogAudioPrediction(self.config.log_freq_audio, self.config.log_nb_audio),
        ]

        if self.config.enable_progress_bar:
            callbacks += [RichProgressBar()]

        if self.config.early_stopping:
            callbacks += [EarlyStopping(**self.config.early_stopping_params)]

        if self.network_param.conditional_transformer_unfreezing:
            callbacks += [ConditionalTransformerUnfreezing(
                unfreeze_step=self.network_param.transformer_unfreeze_step
            )]
        save_top_k = 1
        every_n_epochs = 1

        callbacks += [
            AutoSaveModelCheckpoint(  # ModelCheckpoint
                config=(self.network_param).__dict__,
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                monitor="val/per",
                mode="min",
                filename="epoch-{epoch:02d}-val_per={val/per:.2f}",
                verbose=True,
                dirpath=self.config.weights_path + f"/{self.run_name}",
                save_top_k=save_top_k,
                save_last=True,
                every_n_epochs=every_n_epochs,
                auto_insert_metric_name=False,
                every_n_train_steps=None,
                train_time_interval=None,
                save_on_train_epoch_end=True
            )
        ]  # our model checkpoint callback

        return callbacks

    def _find_latest_checkpoint(self, checkpoint_dir):
        """
        Find the last checkpoint for resuming training
        """
        checkpoint_path = Path(checkpoint_dir)

        if not checkpoint_path.exists():
            self.logger.info(f"Checkpoint directory does not exist: {checkpoint_path}")
            return None

        # Only look for "last.ckpt" - no fallback
        last_ckpt = checkpoint_path / "last.ckpt"
        if last_ckpt.exists() and last_ckpt.stat().st_size > 0:
            self.logger.info(f"Found last checkpoint: {last_ckpt.name}")
            return str(last_ckpt)

        self.logger.info(f"No last.ckpt found in: {checkpoint_path}")
        return None
