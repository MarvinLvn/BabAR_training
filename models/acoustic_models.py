import subprocess
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import hf_hub_download
from torchaudio.models import hubert_pretrain_base
from transformers import PreTrainedTokenizer
from transformers import Wav2Vec2Model, WavLMModel, HubertModel, HubertConfig, Wav2Vec2Config


def _make_mlp_head(input_size, output_size, hidden_ratio=0.5, dropout=0.1):
    """
    Create a 2-layer MLP head with ReLU and dropout

    Args:
        input_size: Input dimension
        output_size: Output dimension (vocab size)
        hidden_ratio: Ratio of hidden size to input size
        dropout: Dropout probability
    """
    hidden_size = int(input_size * hidden_ratio)
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_size, output_size)
    )

class AcousticModel(nn.Module):
    """Single acoustic model with multiple prediction heads"""

    def __init__(self, encoder, vocab_size):
        super().__init__()

        # Core components
        self.encoder = encoder
        self.config = encoder.config

        hidden_size = encoder.config.hidden_size

        self.phoneme_head = _make_mlp_head(
            hidden_size,
            vocab_size,
            hidden_ratio=0.5,
            dropout=0.1
        )

    def forward(self, input_values, attention_mask=None, output_hidden_states=False):
        """Forward pass through encoder only"""
        encoder_outputs = self.encoder(
            input_values,
            attention_mask=attention_mask,
            output_hidden_states=output_hidden_states
        )

        return type('ModelOutput', (), {
            'logits': None,
            'hidden_states': encoder_outputs.hidden_states if output_hidden_states else None,
            'last_hidden_state': encoder_outputs[0],
        })()

    def freeze_feature_encoder(self):
        """Freeze the feature extraction (convolutional) layers"""
        for name, param in self.encoder.named_parameters():
            if 'feature_extractor' in name or 'feature_projection' in name:
                param.requires_grad = False

    def freeze_encoder(self):
        """Freeze the entire encoder (feature extractor + transformer)"""
        for param in self.encoder.parameters():
            param.requires_grad = False

    def get_output_lengths(self, input_lengths):
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.config.conv_kernel, self.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths


def Wav2Vec2(params):
    """Load Wav2Vec2 encoder directly"""
    return Wav2Vec2Model.from_pretrained(params.pretrained_name)


def WavLM(params):
    """Load WavLM encoder directly"""
    return WavLMModel.from_pretrained(params.pretrained_name)


def Hubert(params):
    """Load HuBERT encoder directly"""
    return HubertModel.from_pretrained(params.pretrained_name)

def BabyHubert(params):
    """Load BabyHubert encoder"""
    # Load pretrained weights
    checkpoint_path = _get_babyhubert_checkpoint(params.pretrained_name)
    full_model = hubert_pretrain_base(num_classes=500)
    state_dict = torch.load(checkpoint_path, map_location='cpu')
    state_dict = {k.replace("model.", ""): v for k, v in state_dict["state_dict"].items()}
    full_model.load_state_dict(state_dict)

    # Create HuggingFace-style encoder
    config = HubertConfig(
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
        final_dropout=0.0
    )

    encoder = HubertModel(config)
    _transfer_babyhubert_weights(full_model.wav2vec2, encoder)
    return encoder

def W2VLB(params):
    """ Load Wav2Vec2 LittleBeats LENA 4300 encoder"""
    checkpoint_path = _get_w2vLB_checkpoint(params.pretrained_name)
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

    # Load the pretrained weights
    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=False)
    print("yes queen!")
    exit()

    return model



def _get_babyhubert_checkpoint(pretrained_name):
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

def _get_w2vLB_checkpoint(pretrained_name):
    model_dir = Path(pretrained_name)
    checkpoint_path = model_dir / "model" / "w2vLB_checkpoint.pt"
    if not model_dir.exists():
        print("Downloading W2V-LB checkpoint from hugging face")
        model_dir.mkdir(parents=True, exist_ok=True)

        # Download from HuggingFace Hub
        hf_hub_download(
            repo_id="lijialudew/wav2vec_LittleBeats_LENA",
            filename="LL_4300/checkpoint_best.pt",
            cache_dir=str(model_dir.parent),
            local_dir=str(model_dir),
            local_dir_use_symlinks=False
        )

    return checkpoint_path

def _transfer_babyhubert_weights(torchaudio_model, hf_encoder):
    """Transfer weights from torchaudio BabyHubert to HuggingFace encoder"""
    torchaudio_state = torchaudio_model.state_dict()
    hf_state = hf_encoder.state_dict()

    for ta_key, ta_tensor in torchaudio_state.items():
        hf_key = ta_key
        if ta_key.startswith('encoder.feature_projection.'):
            hf_key = ta_key.replace('encoder.feature_projection.', 'feature_projection.')
        elif ta_key.startswith('encoder.transformer.'):
            hf_key = ta_key.replace('encoder.transformer.', 'encoder.')
        hf_state[hf_key] = ta_tensor.clone()

    hf_encoder.load_state_dict(hf_state)
    print(f"Successfully transferred BabyHubert weights")


# ML: to modify
def _get_wav2vec2_checkpoint(pretrained_name):
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