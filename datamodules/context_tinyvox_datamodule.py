import re
from pathlib import Path
import pandas as pd
import numpy as np
import soundfile as sf
from datasets import Dataset
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from utils.constant import CHARS_TO_REMOVE_REGEX
from utils.logger import init_logger


class ContextualTinyVoxDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()
        self.config = dataset_param
        self.logger = init_logger('ContextualTinyVoxDataModule', 'INFO')

        # Context parameters
        self.context_duration = dataset_param.context_duration
        self.context_duration_ms = self.context_duration * 1000

        self.sampling_rate = 16000
        self.n_debug = 100
        self.processor = None
        self.config.dataset_path = Path(self.config.dataset_path)
        self.dataset_name = self.config.dataset_path.stem.lower()

        self.logger.info(f'Loading Contextual Dataset from: {self.config.dataset_path}')
        self.logger.info(f'Context duration: {self.context_duration}s')
        self.logger.info(f'Using VAD timing: {self.config.use_vad}')

    def _load_split(self, split):
        """Load and create contextual metadata (no caching)"""
        # Load CSV metadata
        csv_path = self.config.dataset_path / f'{split}.csv'
        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")

        df = pd.read_csv(csv_path)
        if self.config.debug_dataset:
            df = df.iloc[:min(len(df), self.n_debug)]

        # Retrieve original filename
        df['original_filename'] = df['audio_filename'].map(lambda x: '_'.join(x.split('_')[:-2]) + '.wav')

        self.logger.info(f"Loaded {len(df)} utterances for {split} split")

        # Filter out rows with missing data
        na_phones = df['phones'].isna()
        self.logger.info(f"Removed {na_phones.sum()} samples with NA phones.")
        df = df[~na_phones]

        # Create contextual samples metadata
        contextual_samples = self._create_contextual_metadata(df)

        self.logger.info(f"Created {len(contextual_samples)} contextual samples from CSV")

        # Create dataset from metadata only
        dataset = Dataset.from_list(contextual_samples)
        return dataset

    def _create_contextual_metadata(self, df):
        """Create contextual training sample metadata from utterance metadata"""
        contextual_samples = []

        # Group by original audio file
        grouped = df.groupby('original_filename')

        for original_filename, group in grouped:
            original_audio_path = self.config.dataset_path / 'original' / original_filename

            # Sort utterances by onset time
            group = group.sort_values('onset')

            # Create contextual sample for each utterance
            for idx, row in group.iterrows():
                sample = self._create_context_metadata_for_utterance(row, str(original_audio_path))
                if sample:
                    contextual_samples.append(sample)

        return contextual_samples

    def _create_context_metadata_for_utterance(self, target_row, original_audio_path):
        """Create metadata for a contextual sample centered around a target utterance"""

        if self.config.use_vad and pd.notna(target_row['with_vad_onset']):
            target_onset = target_row['with_vad_onset']
            target_offset = target_row['with_vad_offset']
        else:
            target_onset = target_row['onset']
            target_offset = target_row['offset']

        if pd.isna(target_onset) or pd.isna(target_offset):
            return None

        # Calculate desired context window (centered on target utterance)
        target_center = (target_onset + target_offset) / 2
        desired_start = target_center - self.context_duration_ms / 2
        desired_end = target_center + self.context_duration_ms / 2

        # Ensure the context always includes the full target utterance
        # This may expand the context beyond the requested duration for long utterances
        context_start = max(0, min(desired_start, target_onset))
        context_end = max(desired_end, target_offset)

        # Calculate actual duration needed (may be > requested duration)
        context_duration_ms = context_end - context_start

        # Calculate target position within the (possibly expanded) context
        target_start_in_context = target_onset - context_start
        target_end_in_context = target_offset - context_start

        # Precompute frame boundaries (for CTC loss optimization)
        estimated_frame_rate = 50.0  # frames per second
        target_start_frame = round(target_start_in_context * estimated_frame_rate / 1000.0)
        target_end_frame = round(target_end_in_context * estimated_frame_rate / 1000.0)
        target_start_frame = max(0, target_start_frame)
        target_end_frame = max(target_start_frame + 1, target_end_frame)

        # Clean up the phoneme and sentence strings
        phonemes = target_row['phones'].rstrip('|').strip() if pd.notna(target_row['phones']) else ""
        sentence = target_row['sentence'] if pd.notna(target_row['sentence']) else ""
        cleaned_sentence = re.sub(CHARS_TO_REMOVE_REGEX, '', sentence).lower().strip()

        return {
            'original_audio_path': original_audio_path,
            'target_phonemes': phonemes,
            'target_sentence': cleaned_sentence,  # Pre-cleaned
            # these fields indicate where in the whole audio, one can extract the context window
            'context_start_ms': float(context_start),
            'context_duration_ms': float(context_duration_ms),
            # these fields indicate where the target utterance start and end within the context window
            'target_start_ms': float(target_start_in_context),
            'target_end_ms': float(target_end_in_context),
            'target_start_frame': target_start_frame,
            'target_end_frame': target_end_frame,
            'audio_filename': target_row['audio_filename'],
        }

    def set_processor(self, processor):
        self.processor = processor

    def setup(self, stage):
        """Load and setup datasets"""
        if self.processor is None:
            raise ValueError("Processor must be set before calling setup().")

        if stage == 'fit':
            self.train_dataset = self._load_split('train')
            self.val_dataset = self._load_split('val')
        elif stage == 'test':
            self.test_dataset = self._load_split('test')
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def _load_audio_segment(self, audio_path, offset_ms, duration_ms):
        """Load audio segment using soundfile"""
        offset_samples = int(offset_ms * self.sampling_rate / 1000.0)
        duration_samples = int(duration_ms * self.sampling_rate / 1000.0)

        # Load audio segment with soundfile
        audio, sr = sf.read(
            audio_path,
            start=offset_samples,
            stop=offset_samples + duration_samples,
            dtype='float32'
        )

        # Handle mono conversion if needed
        if audio.ndim > 1:
            audio = audio.mean(axis=1)  # Convert to mono

        # Verify sample rate (should already be 16kHz but check)
        if sr != self.sampling_rate:
            raise ValueError(f"Sample rate mismatch in {audio_path}: expected {self.sampling_rate}, got {sr}")

        return audio

    def collate_fn(self, batch):
        """Load audio on-demand and create batch"""
        context_audios = []
        valid_samples = []

        # First pass: determine the maximum duration needed across the batch
        max_duration_ms = max(sample['context_duration_ms'] for sample in batch)
        expected_length = int(self.sampling_rate * max_duration_ms / 1000.0)

        # Second pass: load and pad all audio to the same length
        for sample in batch:
            # Load the context audio segment - will raise exception if fails
            audio = self._load_audio_segment(
                sample['original_audio_path'],
                sample['context_start_ms'],
                sample['context_duration_ms']
            )

            # Pad all samples to the maximum length in this batch
            if len(audio) < expected_length:
                audio = np.pad(audio, (0, expected_length - len(audio)), mode='constant', constant_values=0.0)

            context_audios.append(audio)
            valid_samples.append(sample)

        if not context_audios:
            raise ValueError("No valid audio samples in batch")

        # Process without attention masks
        processed = self.processor(
            context_audios,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        # Extract precomputed frame boundaries
        target_frame_starts = [sample["target_start_frame"] for sample in valid_samples]
        target_frame_ends = [sample["target_end_frame"] for sample in valid_samples]

        # Use pre-cleaned sentences
        cleaned_sentences = [sample.get('target_sentence', '') for sample in valid_samples]

        return {
            "array": processed["input_values"],
            "path": [sample["original_audio_path"] for sample in valid_samples],
            "phonemes": [sample["target_phonemes"] for sample in valid_samples],
            "sentence": cleaned_sentences,
            "target_frame_start": target_frame_starts,
            "target_frame_end": target_frame_ends,
            "target_start_ms": [sample["target_start_ms"] for sample in valid_samples],
            "target_end_ms": [sample["target_end_ms"] for sample in valid_samples],
            "audio_filename": [sample["audio_filename"] for sample in valid_samples],
        }

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )