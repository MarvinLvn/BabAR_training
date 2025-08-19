import os
from pickle import FALSE
import random
from dataclasses import dataclass
from os import path as osp
from typing import Any, ClassVar, Dict, List, Optional
from simple_parsing.helpers import Serializable, choice, dict_field, list_field

import pytorch_lightning as pl
import simple_parsing
import torch
import torch.optim

################################## Global parameters ##################################


@dataclass
class Hparams:
    """Hyperparameters of for the run"""

    # wandb
    wandb_entity: str = "phorec"
    wandb_project: str = "dev_run" # name of the project/experiment
    debug_pytorch: bool = False  # if activated, allow pytorch debugging features (slower training)

    root_dir: str = os.getcwd()  # root_dir

    # basic params
    seed_everything: Optional[int] = None  # seed for the 5e whole run
    gpu: int = 1  # number or gpu
    max_epochs: int = 40  # maximum number of epochs
    weights_path: str = osp.join(os.getcwd(), "weights")


    # modes
    tune_lr: bool = False  # tune the model on first run
    dev_run: bool = False
    train: bool = True

    best_model: str = ""

    log_freq_audio: int = 3             # log audio examples every N epochs
    log_nb_audio: int = 8

    # trainer params
    val_check_interval: float = 1.0     # How often within one training epoch to check the validation set
                                        # (e.g., if set to .25 will validate 4 times during a training epoch)
    limit_train_batches: float = 1.0    # Run through, say 25% of the training set each epoch
    limit_val_batches: float = 1.0      # Run through, say 25% of the validation set each epoch
    enable_progress_bar: bool = True

    # testing params
    best_model_run: str = "WavLM"

    # The early stopping callback runs at the end of every validation epoch by default
    # Consequently, it is affected by check_val_every_n_epoch and val_check_interval
    early_stopping: bool = True
    early_stopping_params: Dict[str, Any] = dict_field(
        dict(monitor="val/per", patience=10, mode="min", verbose=True)
    )


@dataclass
class NetworkParams:
    network_name: str = "WavLM"  # Hubert, Wav2Vec2, WavLM
    pretrained_name: Optional[str] = ""

    freeze: bool = True
    freeze_transformer: bool = True

    # Dynamic unfreezing
    conditional_transformer_unfreezing: bool = False  # Enable dynamic unfreezing
    transformer_unfreeze_step: int = 10000  # Step to unfreeze at

    # Phoneme Tokenizer
    eos_token: str = "<blank>"
    bos_token: str = "<blank>"
    unk_token: str = "<unk>"
    pad_token: str = "<pad>"
    word_delimiter_token: str = "<blank>" # blank token


@dataclass
class DatasetParams:
    """Dataset Parameters
    ! The batch_size and number of crops should be defined here
    """

    # TinyVox Dataset Parameters
    dataset_path: str = "/scratch2/mlavechin/tinyvox/TinyVox"
    inventory_path: str= "/scratch2/mlavechin/tinyvox/TinyVox/unique_phonemes.json"
    use_vad: bool = False  # Use audio_with_vad folder instead of audio
    custom_dataset: bool = True # Flag to use TinyVox instead of HuggingFace
    debug_dataset: bool = False # If activated, will only load 1000 training samples
    create_dataset: "bool" = True # If activated, will recreate the dataset even if it already exists
    cache_dir: str = osp.join(os.getcwd(), "assets") # Where dataset files will be stored
    create_dataset: bool = False # Whether to recreate datasets even if they already exists

    # Dataloader parameters
    num_workers: int = 8  # number of workers for dataloaders
    batch_size: int = 128

    # Dataset processing parameters
    num_proc: int = 4


@dataclass
class OptimizerParams:
    """Optimization parameters"""

    optimizer: str = "AdamW"
    lr: float = 1e-4
    weight_decay: float = 1e-2
    accumulate_grad_batches: int = 8

    # Scheduler parameters (all step-based except ReduceLROnPlateau)
    scheduler: Optional[str] = None

    # Cosine scheduler (step-based)
    # Phase1: linear warmup from <warm_start_lr> to <lr> over <warmup_steps>
    # Phase2: cosine decay from base <lr> to <eta_min> over remaining epochs
    #    /-------\
    #   /         \
    #  /           \____
    # /                 \___
    max_steps: int = 260000
    warmup_steps: int = 10000
    warmup_start_lr: float = 0.0
    eta_min: float = 0.0

    # StepLR scheduler (step-based)
    # Multiplies <lr> by <gamma> every <step_size_steps>
    # __
    #   __
    #     __
    #        __
    step_size_steps: int = 50000
    gamma: float = 0.1

    # MultiStepLR scheduler (step-based)
    # Reduces <lr> by <gamma> at <milestone_steps>
    # ____
    #     ____
    #         ____
    milestones_steps: List[Any] = list_field(50000, 100000, 150000)

    # ReduceLROnPlateau scheduler (epoch-based)
    min_lr: float = 5e-9
    patience: int = 10 # in number of epochs

    # Tri-stage scheduler parameters
    #      / -------- \
    #     /            \
    #    /              \
    #   /                \____
    total_training_steps: int = 100000
    tri_stage_warmup_ratio: float = 0.1  # 10% warmup
    tri_stage_constant_ratio: float = 0.4 # 40% constant lr
    # Decay for the remaining steps (calculated automatically)
    # /!\ Careful to plan max_epochs accordingly (otherwise you'll be training with lr = 0)

@dataclass
class Parameters:
    """base options."""

    hparams: Hparams = Hparams()
    data_param: DatasetParams = DatasetParams()
    network_param: NetworkParams = NetworkParams()
    optim_param: OptimizerParams = OptimizerParams()

    def __post_init__(self):
        """Post-initialization code"""
        if self.hparams.seed_everything is None:
            self.hparams.seed_everything = random.randint(1, 10000)



        random.seed(self.hparams.seed_everything)
        torch.manual_seed(self.hparams.seed_everything)
        pl.seed_everything(self.hparams.seed_everything)

        if self.network_param.pretrained_name == "":
            if self.network_param.network_name == "Wav2Vec2":
                self.network_param.pretrained_name = "facebook/wav2vec2-base-960h"
            elif self.network_param.network_name == "WavLM":
                self.network_param.pretrained_name = "microsoft/wavlm-base"
            elif self.network_param.network_name == "Hubert":
                self.network_param.pretrained_name = "facebook/hubert-base-ls960"
            elif self.network_param.network_name == "WavLMplus":
                self.network_param.network_name = "WavLM"
                self.network_param.pretrained_name = "microsoft/wavlm-base-plus"
            else:
                raise NotImplementedError(
                    "Only Wav2Vec2, WavLM and Hubert are available."
                )
        print(f"Pretrained model: {self.network_param.pretrained_name}")

        self.data_param.wandb_project = self.hparams.wandb_project
        self.hparams.accumulate_grad_batches = self.optim_param.accumulate_grad_batches

    @classmethod
    def parse(cls):
        parser = simple_parsing.ArgumentParser()
        parser.add_arguments(cls, dest="parameters")
        args = parser.parse_args()
        instance: Parameters = args.parameters
        return instance
