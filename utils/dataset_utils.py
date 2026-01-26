import json
import os
import os.path as osp

from utils.logger import init_logger

def coll_fn(batch, processor):
    audio_arrays = [b["audio"] for b in batch]

    processed = processor(
        audio_arrays,
        sampling_rate=16000,
        padding=True,
        return_tensors="pt",
        return_attention_mask=True,
    )

    return {
        "array": processed["input_values"],
        "attention_mask": processed["attention_mask"],
        "path": [b["path"] for b in batch],
        "phonemes": [b["phonemes"] for b in batch],
        "sentence": [b["sentence"] for b in batch]
    }

def create_tinyvox_vocabulary(path_inventory, eos_token, bos_token, unk_token, pad_token, word_delimiter_token):
    logger = init_logger("create_tinyvox_vocabulary", "INFO")

    vocab_path = osp.join(os.getcwd(), "assets", "vocab_phoneme")
    file_dict = os.path.join(vocab_path, f"vocab-phoneme-tinyvox.json")

    # If file already exists, just use it
    if os.path.exists(file_dict):
        with open(file_dict, 'r') as f:
            phoneme_vocab = json.load(f)
        logger.info(f"Using existing vocabulary: {len(phoneme_vocab)} entries")
        return file_dict, len(phoneme_vocab)

    # Otherwise, create it
    phonemes = sorted(json.load(open(path_inventory, 'r')))
    phoneme_vocab = {phonemes[i]: i for i in range(len(phonemes))}

    special_tokens = list(dict.fromkeys([eos_token, bos_token, unk_token, pad_token, word_delimiter_token]))

    for special_token in special_tokens:
        phoneme_vocab[special_token] = len(phoneme_vocab)

    logger.info(f"Length vocabulary : {len(phoneme_vocab)}")

    vocab_path = osp.join(os.getcwd(), "assets", "vocab_phoneme")
    file_dict = os.path.join(vocab_path, f"vocab-phoneme-tinyvox.json")

    if not os.path.exists(vocab_path):
        os.makedirs(vocab_path)

    with open(file_dict, "w") as vocab_file:
        json.dump(phoneme_vocab, vocab_file)

    return file_dict, len(phoneme_vocab)
