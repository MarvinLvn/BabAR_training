import torch
import torchaudio
import torch.nn as nn
from transformers import HubertForCTC, Wav2Vec2ForCTC, WavLMForCTC, Wav2Vec2Config, Wav2Vec2Model
from transformers import PreTrainedTokenizer
import numpy as np
from pathlib import Path

class BaseModel(nn.Module):
    """
    BaseFeaturesExtractor class that will extract features according to the type of model
    https://huggingface.co/blog/fine-tune-wav2vec2-english
    """

    def __init__(self, params):
        super().__init__()
        self.params = params

    def forward(self, x):
        outputs = self.model(x)
        return outputs


class Wav2Vec2(BaseModel):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/wav2vec2#transformers.Wav2Vec2ForCTC
    """

    def __init__(self, params):
        super().__init__(params)

        self.model = Wav2Vec2ForCTC.from_pretrained(params.pretrained_name)
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(
            in_features=in_features, out_features=self.params.vocab_size
        )


class WavLM(BaseModel):
    """
    https://huggingface.co/docs/transformers/model_doc/wavlm#transformers.WavLMForCTC
    """

    def __init__(self, params):
        super().__init__(params)
        self.model = WavLMForCTC.from_pretrained(params.pretrained_name)
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(
            in_features=in_features, out_features=self.params.vocab_size
        )


class Hubert(BaseModel):
    """
    https://huggingface.co/docs/transformers/v4.16.2/en/model_doc/hubert#transformers.HubertForCTC
    """

    def __init__(self, params):
        super().__init__(params)
        self.model = HubertForCTC.from_pretrained(params.pretrained_name)
        in_features = self.model.lm_head.in_features
        self.model.lm_head = nn.Linear(
            in_features=in_features, out_features=self.params.vocab_size
        )


class BabyHubert(BaseModel):
    def __init__(self, params):
        super().__init__(params)

        from torchaudio.models import hubert_pretrain_base

        # Auto-download and load BabyHubert
        checkpoint_path = self._get_checkpoint_path(params.pretrained_name)

        # Load the model
        full_model = hubert_pretrain_base(num_classes=500)
        state_dict = torch.load(checkpoint_path, map_location='cpu')
        state_dict = {
            k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()
        }
        full_model.load_state_dict(state_dict)

        # Extract encoder and add CTC head
        self.encoder = full_model.wav2vec2
        self.lm_head = nn.Linear(768, params.vocab_size)

        # Create wrapper for compatibility
        self.model = self

    def forward(self, input_values, attention_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        # Get encoder outputs
        hidden_states, _ = self.encoder(input_values)

        # Apply CTC head
        logits = self.lm_head(hidden_states)

        return type('CTC_Output', (), {'logits': logits})()

    def _get_checkpoint_path(self, pretrained_name):
        import subprocess

        model_dir = Path(pretrained_name)
        checkpoint_path = model_dir / "model" / "babyhubert2-epoch=44-step=400000.ckpt"

        if not model_dir.exists():
            print(f"Downloading BabyHubert from GitHub...")
            model_dir.parent.mkdir(parents=True, exist_ok=True)
            subprocess.run([
                "git", "clone",
                "git@github.com:arxaqapi/babyhubert-temp.git",
                str(model_dir)
            ], check=True)

        return checkpoint_path

class CustomWav2Vec2ForCTC(nn.Module):
    """
    Custom Wav2Vec2 model that matches Jialu Li's architecture
    See https://huggingface.co/lijialudew/wav2vec_children_ASR
    """

    def __init__(self, config):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(config)
        self.dropout = nn.Dropout(config.final_dropout)

        # 2-layer CTC head to match the original model
        self.lm_head = nn.Sequential(
            nn.Linear(config.hidden_size, 384),  # 768 -> 384
            nn.Linear(384, config.vocab_size)  # 384 -> 53
        )

    def forward(self, input_values, attention_mask=None, output_attentions=None,
                output_hidden_states=None, return_dict=None):
        attention_mask=None
        outputs = self.wav2vec2(
            input_values,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        hidden_states = self.dropout(hidden_states)

        logits = self.lm_head(hidden_states)

        return type('CTC_Output', (), {
            'logits': logits,
            'hidden_states': outputs.hidden_states if output_hidden_states else None,
            'attentions': outputs.attentions if output_attentions else None,
        })()


class CustomWav2Vec2Tokenizer(PreTrainedTokenizer):
    """
    Custom tokenizer for the children's ASR model
    """

    def __init__(self, label_map, id_to_label, **kwargs):
        self.label_map = label_map
        self.id_to_label = id_to_label

        # Set special tokens
        self._pad_token = '<blank>'
        self._bos_token = '<bos>'
        self._eos_token = '<eos>'
        self._unk_token = '<UNK>'

        super().__init__(
            pad_token=self._pad_token,
            bos_token=self._bos_token,
            eos_token=self._eos_token,
            unk_token=self._unk_token,
            **kwargs
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
        return ' '.join(tokens)

    def decode(self, token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True):
        """Decode with CTC-specific logic"""
        if isinstance(token_ids, torch.Tensor):
            if token_ids.dim() > 1:
                token_ids = token_ids.squeeze()
            token_ids = token_ids.tolist()

        # CTC decoding - remove consecutive duplicates
        filtered_tokens = []
        previous_id = None

        for current_id in token_ids:
            if current_id != previous_id:
                token = self._convert_id_to_token(current_id)
                if skip_special_tokens and token in [self._pad_token, self._bos_token, self._eos_token]:
                    pass
                else:
                    filtered_tokens.append(token)
            previous_id = current_id

        result = self.convert_tokens_to_string(filtered_tokens)

        if clean_up_tokenization_spaces:
            result = ' '.join(result.split())

        return result

    def batch_decode(self, sequences, **kwargs):
        return [self.decode(seq, **kwargs) for seq in sequences]


class CustomWav2Vec2Processor:
    """
    Processor containing both audio processing and tokenizer (like HuggingFace)
    """

    def __init__(self, tokenizer, sampling_rate=16000):
        self.tokenizer = tokenizer  # This is the key - tokenizer as attribute
        self.sampling_rate = sampling_rate

    def __call__(self, raw_speech, sampling_rate=None, return_tensors="pt", padding=True):
        """Process audio input - supports both single and batch processing"""

        # Handle batch processing (list of audio arrays)
        if isinstance(raw_speech, list):
            return self._process_batch(raw_speech, sampling_rate, return_tensors, padding)

        # Handle single audio processing
        if isinstance(raw_speech, (list, np.ndarray)):
            raw_speech = torch.tensor(raw_speech, dtype=torch.float32)
        elif not isinstance(raw_speech, torch.Tensor):
            raw_speech = torch.tensor(raw_speech, dtype=torch.float32)

        # Resample if needed
        if sampling_rate is not None and sampling_rate != self.sampling_rate:
            resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
            if raw_speech.dim() == 1:
                raw_speech = resampler(raw_speech.unsqueeze(0)).squeeze()

        # Normalize
        if raw_speech.abs().max() > 0:
            raw_speech = raw_speech / raw_speech.abs().max()

        if return_tensors == "pt":
            return {
                "input_values": raw_speech.unsqueeze(0),
                "attention_mask": torch.ones(1, raw_speech.shape[-1])  # ADD THIS
            }
        else:
            return {
                "input_values": raw_speech.numpy(),
                "attention_mask": np.ones((1, raw_speech.shape[-1]))  # ADD THIS
            }

    def _process_batch(self, raw_speech_list, sampling_rate, return_tensors, padding):
        """Process a batch of audio arrays with attention masks"""
        processed_batch = []
        original_lengths = []

        for raw_speech in raw_speech_list:
            # Convert to tensor
            if isinstance(raw_speech, np.ndarray):
                raw_speech = torch.tensor(raw_speech, dtype=torch.float32)
            elif not isinstance(raw_speech, torch.Tensor):
                raw_speech = torch.tensor(raw_speech, dtype=torch.float32)

            # Resample if needed
            if sampling_rate is not None and sampling_rate != self.sampling_rate:
                resampler = torchaudio.transforms.Resample(sampling_rate, self.sampling_rate)
                if raw_speech.dim() == 1:
                    raw_speech = resampler(raw_speech.unsqueeze(0)).squeeze()
                else:
                    raw_speech = resampler(raw_speech)

            # Normalize
            if raw_speech.abs().max() > 0:
                raw_speech = raw_speech / raw_speech.abs().max()

            processed_batch.append(raw_speech)
            original_lengths.append(raw_speech.shape[-1])

        # Handle padding and create attention masks
        if padding:
            max_length = max(original_lengths)

            padded_batch = []
            attention_masks = []

            for i, audio in enumerate(processed_batch):
                original_length = original_lengths[i]

                if original_length < max_length:
                    # Pad audio with zeros
                    pad_length = max_length - original_length
                    padded_audio = torch.nn.functional.pad(audio, (0, pad_length), value=0.0)

                    # Create attention mask: 1 for real audio, 0 for padding
                    attention_mask = torch.cat([
                        torch.ones(original_length),
                        torch.zeros(pad_length)
                    ])
                else:
                    padded_audio = audio
                    attention_mask = torch.ones(original_length)

                padded_batch.append(padded_audio)
                attention_masks.append(attention_mask)

            # Stack into batch tensors
            batch_tensor = torch.stack(padded_batch)
            attention_mask_tensor = torch.stack(attention_masks)
        else:
            # No padding - just stack (will fail if different lengths)
            batch_tensor = torch.stack(processed_batch)
            attention_mask_tensor = torch.ones_like(batch_tensor)

        if return_tensors == "pt":
            return {
                "input_values": batch_tensor,
                "attention_mask": attention_mask_tensor
            }
        else:
            return {
                "input_values": batch_tensor.numpy(),
                "attention_mask": attention_mask_tensor.numpy()
            }

    # Delegate tokenizer methods to the tokenizer attribute
    def decode(self, *args, **kwargs):
        """Delegate to tokenizer"""
        return self.tokenizer.decode(*args, **kwargs)

    def batch_decode(self, *args, **kwargs):
        """Delegate to tokenizer"""
        return self.tokenizer.batch_decode(*args, **kwargs)