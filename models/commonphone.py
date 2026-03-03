"""
CommonPhone model wrapper for evaluation on TinyVox.

Loads the pklumpp/Wav2Vec2_CommonPhone model from HuggingFace and wraps it
in an interface compatible with evaluate_pretrained.py.

The model architecture is:
    Wav2Vec2Model (facebook/wav2vec2-large-xlsr-53) + nn.Linear(1024, 102)
    102 = 101 IPA phones + 1 CTC blank (index 0)

References:
    - HuggingFace: https://huggingface.co/pklumpp/Wav2Vec2_CommonPhone
    - GitHub: https://github.com/PKlumpp/phd_model
    - Klumpp (2024), "Phonetic Transfer Learning from Healthy References
      for the Analysis of Pathological Speech", PhD thesis, FAU Erlangen-Nürnberg
"""

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from transformers import (
    Wav2Vec2Model,
    Wav2Vec2Config,
    Wav2Vec2FeatureExtractor,
    PreTrainedModel,
    PretrainedConfig,
    PreTrainedTokenizer,
)


# ---------------------------------------------------------------------------
# Phone inventory (from PKlumpp/phd_model phonetics/ipa.py)
# Index 0 is CTC blank; indices 1-101 are IPA phones.
# ---------------------------------------------------------------------------
COMMONPHONE_SYMBOLS = {
    "<blank>": 0,
    "r": 1, "ʝ": 2, "ã": 3, "gː": 4, "t": 5, "n": 6, "w": 7, "u": 8,
    "l": 9, "yː": 10, "ʎ": 11, "bʲ": 12, "ə": 13, "ʃʲ": 14, "sː": 15,
    "zʲ": 16, "kː": 17, "y": 18, "ɒ": 19, "fʲ": 20, "ɑ": 21, "ʏ": 22,
    "ɣ": 23, "s": 24, "m": 25, "tː": 26, "xʲ": 27, "vː": 28, "ø": 29,
    "h": 30, "ɨ": 31, "dʲ": 32, "dː": 33, "bː": 34, "ɲː": 35, "ɑː": 36,
    "ɪ": 37, "ɛ": 38, "i": 39, "ʔ": 40, "g": 41, "ʃ": 42, "ɜː": 43,
    "mː": 44, "øː": 45, "fː": 46, "p": 47, "iː": 48, "(...)": 49,
    "v": 50, "ʌ": 51, "b": 52, "k": 53, "x": 54, "ɲ": 55, "ʒ": 56,
    "rː": 57, "eː": 58, "ç": 59, "ŋ": 60, "ɔː": 61, "œ": 62, "ẽ": 63,
    "θ": 64, "a": 65, "rʲ": 66, "vʲ": 67, "ʃː": 68, "æ": 69, "ɶ̃": 70,
    "pː": 71, "nː": 72, "lʲ": 73, "õ": 74, "pʲ": 75, "ɱ": 76, "ð": 77,
    "f": 78, "j": 79, "o": 80, "nʲ": 81, "sʲ": 82, "lː": 83, "e": 84,
    "d": 85, "ʊ": 86, "gʲ": 87, "z": 88, "ɛː": 89, "tʲ": 90, "β": 91,
    "mʲ": 92, "uː": 93, "ɥ": 94, "ʀ": 95, "aː": 96, "ɐ": 97, "ɔ": 98,
    "oː": 99, "ʎː": 100, "kʲ": 101,
}

COMMONPHONE_ID_TO_SYMBOL = {v: k for k, v in COMMONPHONE_SYMBOLS.items()}


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
class Wav2Vec2CommonPhoneConfig(PretrainedConfig):
    """Config for CommonPhone model on HuggingFace."""
    model_type = "wav2vec2"

    def __init__(self, n_classes: int = 102, **kwargs):
        self.n_classes = n_classes
        super().__init__(**kwargs)


class Wav2Vec2CommonPhone(PreTrainedModel):
    """
    Exact replica of the Wav2Vec2 class from PKlumpp/phd_model.
    Needed so that ``from_pretrained("pklumpp/Wav2Vec2_CommonPhone")`` works.
    """
    config_class = Wav2Vec2CommonPhoneConfig

    def __init__(self, config):
        super().__init__(config)
        self.wav2vec = Wav2Vec2Model(
            Wav2Vec2Config.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        )
        self.linear = nn.Linear(in_features=1024, out_features=config.n_classes)

    def forward(self, x):
        x = self.wav2vec(x)
        y = self.linear(x.last_hidden_state)
        return y, x.last_hidden_state, x.extract_features


class CommonPhoneModelWrapper(nn.Module):
    """
    Thin wrapper that gives the CommonPhone model a HuggingFace-like
    interface so ``model(input_values).logits`` works.
    """

    def __init__(self, commonphone_model: Wav2Vec2CommonPhone):
        super().__init__()
        self.model = commonphone_model

    def forward(self, input_values, **kwargs):
        logits, _, _ = self.model(input_values)
        return type("CTC_Output", (), {"logits": logits})()


# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------
class CommonPhoneTokenizer(PreTrainedTokenizer):
    """
    Minimal CTC tokenizer for the CommonPhone phone inventory.

    Provides the interface expected by ``evaluate_pretrained.compute_mapping``
    and ``processor.batch_decode``.
    """

    def __init__(self, **kwargs):
        self.label_map = dict(COMMONPHONE_SYMBOLS)
        self.id_to_label = dict(COMMONPHONE_ID_TO_SYMBOL)

        self._pad_token = "<blank>"
        self._bos_token = "<blank>"
        self._eos_token = "<blank>"
        self._unk_token = "<blank>"

        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs,
        )

    @property
    def vocab_size(self):
        return len(self.label_map)

    def get_vocab(self):
        return self.label_map.copy()

    def _tokenize(self, text):
        return text.strip().split()

    def _convert_token_to_id(self, token):
        return self.label_map.get(token, self.label_map.get(self._unk_token, 0))

    def _convert_id_to_token(self, index):
        return self.id_to_label.get(index, self._unk_token)

    def convert_tokens_to_string(self, tokens):
        return " ".join(tokens)

    # -- CTC decode --------------------------------------------------------
    def decode(self, token_ids, skip_special_tokens=True, **kwargs):
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze()
            token_ids = token_ids.tolist()

        # CTC: collapse consecutive duplicates, remove blank
        filtered = []
        prev = None
        for idx in token_ids:
            if idx != prev:
                token = self._convert_id_to_token(idx)
                if skip_special_tokens and token == self._pad_token:
                    pass
                elif token == "(...)":
                    pass  # silence token — skip
                else:
                    filtered.append(token)
            prev = idx

        return self.convert_tokens_to_string(filtered)

    def batch_decode(self, sequences, **kwargs):
        return [self.decode(seq, **kwargs) for seq in sequences]


# ---------------------------------------------------------------------------
# Processor
# ---------------------------------------------------------------------------
class CommonPhoneProcessor:
    """
    Audio processor + tokenizer, matching the interface of Wav2Vec2Processor.

    Audio normalisation follows the CommonPhone convention: per-utterance
    zero-mean / unit-variance normalisation (see PKlumpp/phd_model example.py).
    """

    def __init__(self, tokenizer: CommonPhoneTokenizer, sampling_rate: int = 16_000):
        self.tokenizer = tokenizer
        self.sampling_rate = sampling_rate

    # -- audio processing --------------------------------------------------
    def __call__(
        self,
        raw_speech,
        sampling_rate=None,
        padding=True,
        return_tensors="pt",
        **kwargs,
    ):
        if isinstance(raw_speech, list):
            return self._process_batch(raw_speech, sampling_rate, return_tensors, padding)

        if isinstance(raw_speech, np.ndarray):
            raw_speech = torch.tensor(raw_speech, dtype=torch.float32)
        elif not isinstance(raw_speech, torch.Tensor):
            raw_speech = torch.tensor(raw_speech, dtype=torch.float32)

        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
            if raw_speech.dim() == 1:
                raw_speech = resampler(raw_speech.unsqueeze(0)).squeeze(0)
            else:
                raw_speech = resampler(raw_speech)

        # Per-utterance z-normalisation
        raw_speech = self._normalize(raw_speech)

        if return_tensors == "pt":
            return {"input_values": raw_speech.unsqueeze(0)}
        return {"input_values": raw_speech.numpy()[np.newaxis]}

    def _process_batch(self, raw_speech_list, sampling_rate, return_tensors, padding):
        processed = []
        lengths = []

        for raw_speech in raw_speech_list:
            if isinstance(raw_speech, np.ndarray):
                raw_speech = torch.tensor(raw_speech, dtype=torch.float32)
            elif not isinstance(raw_speech, torch.Tensor):
                raw_speech = torch.tensor(raw_speech, dtype=torch.float32)

            if sampling_rate is not None and sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                if raw_speech.dim() == 1:
                    raw_speech = resampler(raw_speech.unsqueeze(0)).squeeze(0)
                else:
                    raw_speech = resampler(raw_speech)

            raw_speech = self._normalize(raw_speech)
            processed.append(raw_speech)
            lengths.append(raw_speech.shape[-1])

        if padding:
            max_len = max(lengths)
            padded = []
            for audio in processed:
                if audio.shape[-1] < max_len:
                    audio = torch.nn.functional.pad(audio, (0, max_len - audio.shape[-1]))
                padded.append(audio)
            batch = torch.stack(padded)
        else:
            batch = torch.stack(processed)

        if return_tensors == "pt":
            return {"input_values": batch}
        return {"input_values": batch.numpy()}

    @staticmethod
    def _normalize(audio: torch.Tensor) -> torch.Tensor:
        """Per-utterance zero-mean / unit-variance normalisation."""
        mean = audio.mean()
        std = audio.std()
        return (audio - mean) / (std + 1e-9)

    # -- delegate to tokenizer ---------------------------------------------
    def decode(self, *args, **kwargs):
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        return self.tokenizer.batch_decode(*args, **kwargs)


# ---------------------------------------------------------------------------
# Public loader
# ---------------------------------------------------------------------------
def load_commonphone_model():
    """
    Download and return (model, processor) for CommonPhone,
    compatible with the evaluate_pretrained.py pipeline.

    Returns
    -------
    model : CommonPhoneModelWrapper
        Wrapped model whose ``forward(input_values).logits`` returns
        CTC logits of shape ``[B, T, 102]``.
    processor : CommonPhoneProcessor
        Audio preprocessor + CTC tokenizer.
    """
    print("Loading CommonPhone model from pklumpp/Wav2Vec2_CommonPhone ...")
    raw_model = Wav2Vec2CommonPhone.from_pretrained("pklumpp/Wav2Vec2_CommonPhone")
    model = CommonPhoneModelWrapper(raw_model)

    tokenizer = CommonPhoneTokenizer()
    processor = CommonPhoneProcessor(tokenizer)

    print(f"  Phone vocabulary: {tokenizer.vocab_size} symbols "
          f"(101 IPA phones + 1 CTC blank)")
    return model, processor