import numpy as np
import soundfile as sf
from pathlib import Path
from typing import List, Dict
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader
from datasets import Dataset

from utils.logger import init_logger


class ContextualVTCDataModule(LightningDataModule):
    """
    DataModule for inference on VTC (Voice Type Classifier) output
    Similar to ContextualTinyVoxDataModule but without phoneme labels
    """

    def __init__(self, audio_path, rttm_path, context_duration=15.0,
                 batch_size=32, num_workers=4, speaker_filter='KCHI', max_utt_dur=None):
        super().__init__()

        self.audio_path = Path(audio_path)
        self.rttm_path = Path(rttm_path)
        self.context_duration = context_duration
        self.context_duration_ms = self.context_duration * 1000
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.speaker_filter = speaker_filter
        self.max_utt_dur = max_utt_dur

        self.sampling_rate = 16000
        self.processor = None

        self.logger = init_logger('ContextualVTCDataModule', 'INFO')
        self.logger.info(f'Audio file: {self.audio_path}')
        self.logger.info(f'RTTM file: {self.rttm_path}')
        self.logger.info(f'Context duration: {self.context_duration}s')
        self.logger.info(f'Speaker filter: {self.speaker_filter}')
        if self.max_utt_dur is not None:
            self.logger.info(f'Max utterance duration: {self.max_utt_dur}s')

    def parse_rttm(self) -> List[Dict]:
        """Parse RTTM file and extract utterances for specified speaker type"""
        utterances = []
        filtered_count = 0

        with open(self.rttm_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or not line.startswith('SPEAKER'):
                    continue

                parts = line.split()
                if len(parts) < 8:
                    continue

                # RTTM format: SPEAKER <file> <channel> <onset> <duration> <NA> <NA> <speaker> <NA> <NA>
                onset = float(parts[3])  # seconds
                duration = float(parts[4])  # seconds
                speaker = parts[7]

                if speaker == self.speaker_filter:
                    # Filter by max utterance duration if specified
                    if self.max_utt_dur is not None and duration > self.max_utt_dur:
                        filtered_count += 1
                        continue

                    utterances.append({
                        'onset': onset * 1000,  # Convert to ms
                        'offset': (onset + duration) * 1000,  # Convert to ms
                        'duration': duration * 1000,  # Convert to ms
                        'speaker': speaker
                    })

        self.logger.info(f"Found {len(utterances)} utterances for speaker '{self.speaker_filter}'")
        if self.max_utt_dur is not None and filtered_count > 0:
            self.logger.info(f"Filtered out {filtered_count} utterances longer than {self.max_utt_dur}s")
        return utterances

    def _create_contextual_metadata(self, utterances: List[Dict]) -> List[Dict]:
        """Create contextual training sample metadata from utterance metadata"""
        contextual_samples = []

        # Get total audio duration
        info = sf.info(self.audio_path)
        total_duration_ms = info.duration * 1000

        for idx, utt in enumerate(utterances):
            sample = self._create_context_metadata_for_utterance(
                utt, total_duration_ms, idx
            )
            if sample:
                contextual_samples.append(sample)

        self.logger.info(f"Created {len(contextual_samples)} contextual samples")
        return contextual_samples

    def _create_context_metadata_for_utterance(self, target_utt: Dict,
                                               total_duration_ms: float,
                                               utterance_id: int) -> Dict:
        """Create metadata for a contextual sample centered around a target utterance"""
        target_onset = target_utt['onset']
        target_offset = target_utt['offset']

        # Calculate desired context window (centered on target utterance)
        target_center = (target_onset + target_offset) / 2
        desired_start = target_center - self.context_duration_ms / 2
        desired_end = target_center + self.context_duration_ms / 2

        # Ensure the context always includes the full target utterance
        context_start = max(0, min(desired_start, target_onset))
        context_end = max(desired_end, target_offset)

        # Clip to audio boundaries
        context_start = max(0, context_start)
        context_end = min(total_duration_ms, context_end)

        # Calculate actual duration needed
        context_duration_ms = context_end - context_start

        # Calculate target position within context
        target_start_in_context = target_onset - context_start
        target_end_in_context = target_offset - context_start

        # Precompute frame boundaries (for CTC loss optimization)
        estimated_frame_rate = 50.0  # frames per second
        target_start_frame = round(target_start_in_context * estimated_frame_rate / 1000.0)
        target_end_frame = round(target_end_in_context * estimated_frame_rate / 1000.0)
        target_start_frame = max(0, target_start_frame)
        target_end_frame = max(target_start_frame + 1, target_end_frame)

        return {
            'audio_path': str(self.audio_path),
            'utterance_id': utterance_id,
            'speaker': target_utt['speaker'],
            # Original utterance timing (for output)
            'utterance_onset_sec': target_onset / 1000.0,
            'utterance_duration_sec': target_utt['duration'] / 1000.0,
            # Context window (for loading audio)
            'context_start_ms': float(context_start),
            'context_duration_ms': float(context_duration_ms),
            # Target position within context (for frame extraction)
            'target_start_ms': float(target_start_in_context),
            'target_end_ms': float(target_end_in_context),
            'target_start_frame': target_start_frame,
            'target_end_frame': target_end_frame,
        }

    def set_processor(self, processor):
        """Set the audio processor (from model)"""
        self.processor = processor

    def setup(self, stage=None):
        """Load and setup dataset"""
        if self.processor is None:
            raise ValueError("Processor must be set before calling setup().")

        # Parse RTTM
        utterances = self.parse_rttm()

        if len(utterances) == 0:
            self.logger.warning(f"No utterances found for speaker '{self.speaker_filter}'")
            # Create empty dataset
            self.dataset = Dataset.from_list([])
            return

        # Create contextual samples
        contextual_samples = self._create_contextual_metadata(utterances)

        # Create dataset
        self.dataset = Dataset.from_list(contextual_samples)

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
            audio = audio.mean(axis=1)

        # Verify sample rate
        if sr != self.sampling_rate:
            raise ValueError(
                f"Sample rate mismatch in {audio_path}: "
                f"expected {self.sampling_rate}, got {sr}"
            )

        return audio

    def collate_fn(self, batch):
        """Load audio on-demand and create batch"""
        context_audios = []
        valid_samples = []

        # Determine the maximum duration needed
        max_duration_ms = max(sample['context_duration_ms'] for sample in batch)
        expected_length = int(self.sampling_rate * max_duration_ms / 1000.0)

        # Load and pad all audio to the same length
        for sample in batch:
            # Load the context audio segment
            audio = self._load_audio_segment(
                sample['audio_path'],
                sample['context_start_ms'],
                sample['context_duration_ms']
            )

            # Pad all samples to the maximum length in this batch
            if len(audio) < expected_length:
                audio = np.pad(
                    audio,
                    (0, expected_length - len(audio)),
                    mode='constant',
                    constant_values=0.0
                )

            context_audios.append(audio)
            valid_samples.append(sample)

        if not context_audios:
            raise ValueError("No valid audio samples in batch")

        # Process audio
        processed = self.processor(
            context_audios,
            sampling_rate=self.sampling_rate,
            padding=True,
            return_tensors="pt",
        )

        # Extract precomputed frame boundaries
        target_frame_starts = [sample["target_start_frame"] for sample in valid_samples]
        target_frame_ends = [sample["target_end_frame"] for sample in valid_samples]

        result = {
            "array": processed["input_values"],
            "path": [sample["audio_path"] for sample in valid_samples],
            "target_frame_start": target_frame_starts,
            "target_frame_end": target_frame_ends,
            "target_start_ms": [sample["target_start_ms"] for sample in valid_samples],
            "target_end_ms": [sample["target_end_ms"] for sample in valid_samples],
            "utterance_id": [sample["utterance_id"] for sample in valid_samples],
            "utterance_onset_sec": [sample["utterance_onset_sec"] for sample in valid_samples],
            "utterance_duration_sec": [sample["utterance_duration_sec"] for sample in valid_samples],
            "speaker": [sample["speaker"] for sample in valid_samples],
        }

        return result

    def dataloader(self):
        """Return dataloader for inference"""
        return DataLoader(
            self.dataset,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=self.collate_fn,
            pin_memory=True,
        )

    def predict_dataloader(self):
        """Alias for dataloader (for Lightning Trainer)"""
        return self.dataloader()