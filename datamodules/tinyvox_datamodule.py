import re
import os
import re
from pathlib import Path

import pandas as pd
from datasets import Dataset, Audio
from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader

from utils.constant import CHARS_TO_REMOVE_REGEX
from utils.dataset_utils import coll_fn
from utils.logger import init_logger


class TinyVoxDataModule(LightningDataModule):
    def __init__(self, dataset_param):
        super().__init__()
        self.config = dataset_param
        self.logger = init_logger('TinyVoxDataModule', 'INFO')
        self.audio_folder = 'audio_with_vad' if self.config.use_vad else 'audio'
        self.sampling_rate = 16000
        self.n_debug = 100
        self.processor = None # this set in setup
        self.config.dataset_path = Path(self.config.dataset_path)

        self.logger.info(f'Loading Dataset : {self.config.dataset_path / self.audio_folder}')

    def _load_split(self, split):
        """Load a dataset split from CSV and audio files"""
        save_path = Path('assets') / 'datasets' / f'tinyvox_{self.audio_folder}' / split
        save_dir = save_path / 'raw'

        # A. Load pickle file if it exists (if create_dataset = False)
        if save_dir.exists() and not self.config.create_dataset:
            self.logger.info(f"Loading cached {split} dataset from {save_dir}")
            return Dataset.load_from_disk(str(save_dir))

        csv_path = self.config.dataset_path / f'{split}.csv'
        audio_dir = self.config.dataset_path / self.audio_folder

        if not csv_path.is_file():
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        if not audio_dir.is_dir():
            raise FileNotFoundError(f"Audio directory not found: {audio_dir}")

        # B.1 Load CSV
        df = pd.read_csv(csv_path)
        if self.config.debug_dataset:
            df = df.iloc[:min(len(df), self.n_debug)]


        self.logger.info(f"Loaded {len(df)} samples for {split} split")
        na_phones = df['phones'].isna()
        self.logger.info(f"Removed {na_phones.sum()} samples with NA phones.")
        df = df[~na_phones]

        # B.2 Verify required columns
        required_cols = ['audio_filename', 'phones', 'sentence']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns in CSV: {missing_cols}")

        # B.3 Create full audio paths
        df['full_audio_path'] = df['audio_filename'].apply(lambda x: str(audio_dir / x))

        # B.4 Verify audio files exist
        missing_files = []
        for idx, row in df.iterrows():
            if not os.path.exists(row['full_audio_path']):
                missing_files.append(row['full_audio_path'])

        if missing_files:
            self.logger.warning(f"Found {len(missing_files)} missing audio files")
            # Remove missing files
            df = df[~df['full_audio_path'].isin(missing_files)]
            self.logger.info(f"After filtering: {len(df)} samples remaining")

        # B.5 Convert to HuggingFace Dataset format
        dataset_dict = {
            'audio': df['full_audio_path'].tolist(),
            'path': df['full_audio_path'].tolist(),
            'phonemes': df['phones'].tolist(),
            'sentence': df['sentence'].tolist()
        }

        dataset = Dataset.from_dict(dataset_dict)
        dataset = dataset.cast_column("audio", Audio(sampling_rate=self.sampling_rate))

        # C. Save dataset
        self.logger.info(f"Saving processed {split} dataset to {save_dir}")
        save_path.mkdir(exist_ok=True, parents=True)
        dataset.save_to_disk(str(save_dir))

        return dataset


    def _process_split(self, split, processor, batch_size=512):
        """Process dataset split - format phonemes"""
        dataset = getattr(self, f"{split}_dataset")

        save_path = Path('assets') / 'datasets' / f'tinyvox_{self.audio_folder}' / split
        save_dir = save_path / 'processed'

        # A. Load pickle file if it exists (if create_dataset = False)
        if save_dir.exists() and not self.config.create_dataset:
            self.logger.info(f"Loading cached {split} dataset from {save_dir}")
            dataset = Dataset.load_from_disk(str(save_dir), keep_in_memory=False)
        else:
            # B.1 Remove punctuation (for purely aesthetic purpose to log info in wandb)
            dataset = dataset.map(
                lambda x: {
                    'sentence': re.sub(
                        CHARS_TO_REMOVE_REGEX, '', x['sentence']
                    ).lower()
                },
                num_proc=self.config.num_proc,
                load_from_cache_file=False
            )

            # B.2 Apply processor
            dataset = dataset.map(
                lambda batch: {
                    'audio': processor(
                        [ad['array'] for ad in batch['audio']], sampling_rate=16000
                ).input_values
                },
                batched=True,
                batch_size=batch_size,
                num_proc=self.config.num_proc,
                cache_file_name=str(save_path / f'dataset_cache.arrow'),
                load_from_cache_file=False
            )

            # C. Save dataset
            self.logger.info(f"Saving processed {split} dataset to {save_dir}")
            dataset.save_to_disk(str(save_dir))

        setattr(self, f"{split}_dataset", dataset)
        self.logger.info(f"Processed {split} dataset: {len(dataset)} samples")

    def set_processor(self, processor):
        self.processor = processor

    def setup(self, stage):
        """ Load and setup datasets """
        if self.processor is None:
            raise ValueError("Processor must be set before calling setup().")

        if stage == 'fit':
            self.train_dataset = self._load_split('train')
            self.val_dataset = self._load_split('val')
            self._process_split('train', self.processor)
            self._process_split('val', self.processor)
        elif stage == 'test':
            self.test_dataset = self._load_split('test')
            self._process_split('test', self.processor)
        else:
            raise ValueError(f"Unknown stage: {stage}")

    def collate_fn(self, batch):
        return coll_fn(batch, self.processor)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            collate_fn=self.collate_fn,
        )
