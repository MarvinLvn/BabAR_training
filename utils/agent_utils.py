import errno
import importlib
import os
from pathlib import Path

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

from Datasets.tinyvox_datamodule import TinyVoxDataModule
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

