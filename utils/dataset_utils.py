import json
import os
import os.path as osp
import tarfile

import pandas as pd
import torch
import wget
from torch.nn.utils.rnn import pad_sequence

import json
from utils.logger import init_logger


def coll_fn(batch, processor):
    audio_arrays = [b["audio"] for b in batch]

    processed = processor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt"
    )

    return {
        "array": processed["input_values"],
        "path": [b["path"] for b in batch],
        "phonemes": [b["phonemes"] for b in batch],
        "sentence": [b["sentence"] for b in batch]
    }

def create_tinyvox_vocabulary(path_inventory, eos_token, bos_token, unk_token, pad_token, word_delimiter_token):
    logger = init_logger("create_tinyvox_vocabulary", "INFO")

    phonemes = sorted(json.load(open(path_inventory, 'r')))
    phoneme_vocab = {phonemes[i]: i for i in range(len(phonemes))}
    phoneme_vocab[eos_token] = len(phoneme_vocab)
    phoneme_vocab[bos_token] = len(phoneme_vocab)
    phoneme_vocab[unk_token] = len(phoneme_vocab)
    phoneme_vocab[pad_token] = len(phoneme_vocab)
    phoneme_vocab[word_delimiter_token] = len(phoneme_vocab)

    logger.info(f"Length vocabulary : {len(phoneme_vocab)}")

    vocab_path = osp.join(os.getcwd(), "assets", "vocab_phoneme")
    file_dict = os.path.join(vocab_path, f"vocab-phoneme-tinyvox.json")

    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)

    with open(file_dict, "w") as vocab_file:
        json.dump(phoneme_vocab, vocab_file)

    return file_dict, len(phoneme_vocab)

def create_vocabulary(
    ISO6393, path_csv, eos_token, bos_token, unk_token, pad_token, word_delimiter_token
):

    logger = init_logger("create_vocabulary", "INFO")

    df = pd.read_csv(osp.join(path_csv, "phoible.csv"))

    df_phoneme_target_lang = df[df["ISO6393"] == ISO6393]["Phoneme"]
    df_phoneme_target_lang.drop_duplicates(keep="first", inplace=True)
    df_phoneme_target_lang.reset_index(drop=True, inplace=True)

    phoneme_vocab = dict(df_phoneme_target_lang)
    phoneme_vocab = {v: k for k, v in phoneme_vocab.items()}

    phoneme_vocab[eos_token] = len(phoneme_vocab)
    phoneme_vocab[bos_token] = len(phoneme_vocab)
    phoneme_vocab[unk_token] = len(phoneme_vocab)
    phoneme_vocab[pad_token] = len(phoneme_vocab)
    phoneme_vocab[word_delimiter_token] = len(phoneme_vocab)

    logger.info(f"Length vocabulary : {len(phoneme_vocab)}")

    vocab_path = osp.join(os.getcwd(), "assets", "vocab_phoneme")
    file_dict = os.path.join(vocab_path, f"vocab-phoneme-{ISO6393}.json")

    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)

    with open(file_dict, "w") as vocab_file:
        json.dump(phoneme_vocab, vocab_file)

    return file_dict, len(phoneme_vocab)


def create_vocabulary2(
    language, path, eos_token, bos_token, unk_token, pad_token, word_delimiter_token
):

    logger = init_logger("create_vocabulary", "INFO")

    if not osp.exists(path):
        assets_path = "/".join(path.split("/")[:-1])
        url = "https://dl.fbaipublicfiles.com/cpc_audio/common_voices_splits.tar.gz"
        wget.download(url, assets_path)

        tar = tarfile.open(osp.join(assets_path, "common_voices_splits.tar.gz"), "r:gz")
        tar.extractall(assets_path)
        tar.close()

    json_file = osp.join(path, language, "phonesMatches_reduced.json")

    with open(json_file) as file:
        phoneme_vocab = json.load(file)

    phoneme_vocab[eos_token] = len(phoneme_vocab)
    phoneme_vocab[bos_token] = len(phoneme_vocab)
    phoneme_vocab[unk_token] = len(phoneme_vocab)
    phoneme_vocab[pad_token] = len(phoneme_vocab)
    phoneme_vocab[word_delimiter_token] = len(phoneme_vocab)

    logger.info(f"Length vocabulary : {len(phoneme_vocab)}")

    vocab_path = osp.join(os.getcwd(), "assets", "vocab_phoneme")
    file_dict = os.path.join(vocab_path, f"vocab-phoneme-{language}.json")

    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)

    with open(file_dict, "w") as vocab_file:
        json.dump(phoneme_vocab, vocab_file)

    return file_dict, len(phoneme_vocab)
