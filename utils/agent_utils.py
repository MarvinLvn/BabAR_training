import errno
import importlib
import os
from pathlib import Path

import hashlib
import json
import torch
import wandb
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from transformers import Wav2Vec2Config

from datamodules.tinyvox_datamodule import TinyVoxDataModule
from config.hparams import Parameters
from models.models import CustomWav2Vec2ForCTC, CustomWav2Vec2Processor, CustomWav2Vec2Tokenizer


def get_net(network_name, network_param):
    """
    Get Network Architecture based on arguments provided
    """

    mod = importlib.import_module(f"models.{network_name}")
    net = getattr(mod, network_name)
    return net(network_param)


def get_artifact(name: str, type: str) -> str:
    """Artifact utilities
    Extracts the artifact from the name by downloading it locally>
    Return : str = path to the artifact
    """
    if name != "" and name is not None:
        artifact = wandb.run.use_artifact(name, type=type)
        artifact_dir = artifact.download()
        file_path = os.path.join(artifact_dir, os.listdir(artifact_dir)[0])
        return file_path
    else:
        return None


def get_datamodule(data_param):
    """
    Fetch Datamodule Function Pointer
    """
    return TinyVoxDataModule(data_param)

def get_model(model_name, params):
    """
    get features extractors
    """
    try:
        mod = importlib.import_module(f"models.models")
        net = getattr(mod, model_name)
        return net(params)
    except NotImplementedError:
        raise NotImplementedError(f"Not implemented only Wav2vec, WavLM and Hubert")


def parse_params(parameters: Parameters) -> dict:
    wdb_config = {}
    for k, v in vars(parameters).items():
        for key, value in vars(v).items():
            wdb_config[f"{k}-{key}"] = value
    return wdb_config


def get_progress_bar():
    return Progress(
        "[progress.description]{task.description}",
        SpinnerColumn(),
        BarColumn(),
        "[progress.percentage]{task.percentage:>3.0f}%",
        TextColumn("[bold blue]{task.fields[info]}", justify="right"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        "\n",
    )


def create_directory(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError as e:
        print(e)
        if e.errno != errno.EEXIST:
            raise

def load_custom_wav2vec2_model(model_path):
    """
    Load custom Jialu Li's children's ASR model
    Returns model and processor like HuggingFace
    """

    # Load checkpoints
    model_path = Path(model_path)
    wav2vec_checkpoint = torch.load(model_path / "wav2vec2.ckpt", map_location='cpu') # contains the encoder
    model_checkpoint = torch.load(model_path / "model.ckpt", map_location='cpu') # contains the CTC head

    # Parse label encoder
    label_map = {}
    id_to_label = {}

    with open(model_path / "label_encoder.txt", 'r') as f:
        content = f.read()
        lines = content.strip().split('\n')
        for line in lines:
            line = line.strip()
            if '=>' in line and not line.startswith("'starting_index"):
                parts = line.split(' => ')
                if len(parts) == 2:
                    phoneme = parts[0].strip().strip("'")
                    try:
                        idx = int(parts[1].strip())
                        label_map[phoneme] = idx
                        id_to_label[idx] = phoneme
                    except ValueError:
                        continue

    vocab_size = len(label_map)
    print(f"Vocabulary size: {vocab_size}")

    # Create configuration
    config = Wav2Vec2Config(
        vocab_size=vocab_size,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        conv_dim=[512, 512, 512, 512, 512, 512, 512],
        conv_stride=[5, 2, 2, 2, 2, 2, 2],
        conv_kernel=[10, 3, 3, 3, 3, 2, 2],
        conv_bias=False,
        num_conv_pos_embeddings=128,
        num_conv_pos_embedding_groups=16,
        do_stable_layer_norm=False,
        apply_spec_augment=False,
        mask_time_prob=0.0,
        final_dropout=0.0,
        pad_token_id=label_map.get('<blank>', 0),
        bos_token_id=label_map.get('<bos>', 1),
        eos_token_id=label_map.get('<eos>', 2),
    )

    # Create custom model
    model = CustomWav2Vec2ForCTC(config)

    # Load wav2vec2 encoder weights
    model_dict = model.state_dict()

    # Load wav2vec2 weights
    for key in wav2vec_checkpoint.keys():
        new_key = key.replace('model.', 'wav2vec2.')
        if new_key in model_dict:
            model_dict[new_key] = wav2vec_checkpoint[key]

    # Load the 2-layer CTC head weights
    if '0.linear.w.weight' in model_checkpoint and '1.w.weight' in model_checkpoint:
        model_dict['lm_head.0.weight'] = model_checkpoint['0.linear.w.weight']
        model_dict['lm_head.0.bias'] = model_checkpoint['0.linear.w.bias']
        model_dict['lm_head.1.weight'] = model_checkpoint['1.w.weight']
        model_dict['lm_head.1.bias'] = model_checkpoint['1.w.bias']
        print("Successfully loaded 2-layer CTC head weights")

    model.load_state_dict(model_dict, strict=False)
    model.eval()

    # Create processor
    tokenizer = CustomWav2Vec2Tokenizer(label_map, id_to_label)
    processor = CustomWav2Vec2Processor(tokenizer)

    return model, processor

def get_run_name(parameters):
    # Parse "general" arguments
    config_dict = {
        "wandb_project": parameters.hparams.wandb_project,
        "precision": parameters.hparams.precision,
        "max_epochs": parameters.hparams.max_epochs,
        "tune_lr": parameters.hparams.tune_lr,
        "val_check_interval": parameters.hparams.val_check_interval,
        "limit_train_batches": parameters.hparams.limit_train_batches,
        "limit_val_batches": parameters.hparams.limit_val_batches,
        "early_stopping": parameters.hparams.early_stopping,
        "network_name": parameters.network_param.network_name,
        "pretrained_name": parameters.network_param.pretrained_name,
        "freeze": parameters.network_param.freeze,
        "freeze_transformer": parameters.network_param.freeze_transformer,
        "conditional_transformer_unfreezing": parameters.network_param.conditional_transformer_unfreezing,
        "transformer_unfreeze_step": parameters.network_param.transformer_unfreeze_step,
        "use_vad": parameters.data_param.use_vad,
        "batch_size": parameters.data_param.batch_size,
        "optimizer": parameters.optim_param.optimizer,
        "learning_rate": parameters.optim_param.lr,
        "weight_decay": parameters.optim_param.weight_decay,
        "accumulate_grad_batches": parameters.optim_param.accumulate_grad_batches,
        "scheduler": parameters.optim_param.scheduler,
    }
    # Parse scheduler-specific arguments
    if config_dict['scheduler'] == 'Cosine':
        scheduler_dict = {
            'cosine.max_epochs': parameters.optim_param.max_epochs,
            'cosine.warmup_epochs': parameters.optim_param.warmup_epochs,
            'cosine.warmup_start_lr': parameters.optim_param.warmup_start_lr,
            'cosine.eta_min': parameters.optim_param.eta_min,
        }
    elif config_dict['scheduler'] == 'StepLR':
        scheduler_dict = {
            'step.step_size': parameters.optim_param.step_size,
            'step.gamma': parameters.optim_param.gamma
        }
    elif config_dict['scheduler'] == 'MultiStepLR':
        scheduler_dict = {
            'multi.milestones': parameters.optim_param.milestones,
        }
    elif config_dict['scheduler'] == 'ReduceLROnPlateau':
        scheduler_dict = {
            'reduce.min_lr': parameters.optim_param.min_lr,
            'reduce.patience': parameters.optim_param.patience,
        }
    elif config_dict['scheduler'] == 'TriStage':
        scheduler_dict = {
            'tristage.total_training_steps': parameters.optim_param.total_training_steps,
            'tristage.warmup_ratio': parameters.optim_param.tri_stage_warmup_ratio,
            'tristage.constant_ratio': parameters.optim_param.tri_stage_constant_ratio,
        }
    else:
        scheduler_dict = {}
    config_dict = {**config_dict, **scheduler_dict}
    config_str = json.dumps(config_dict, sort_keys=True)
    config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8] # 16^8 possibilities, probability of collision is very low
    network_name = config_dict['network_name']
    if config_dict['pretrained_name'] == 'microsoft/wavlm-base-plus':
        network_name = 'WavLMplus'
    elif config_dict['pretrained_name'] == 'facebook/wav2vec2-large-xlsr-53':
        network_name = 'Wav2Vec2XLSR'

    return (f"{parameters.hparams.wandb_project}_"
            f"{network_name}_"
            f"{'_CNN_not_freezed'*(not parameters.network_param.freeze)}"
            f"{f'_{parameters.hparams.limit_train_batches}_train'*(parameters.hparams.limit_train_batches!=1.0)}"
            f"{'_tf_freezed'*(parameters.network_param.freeze_transformer)}_"
            f"{config_hash}")
