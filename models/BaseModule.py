from itertools import chain

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from pytorch_lightning import LightningModule
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, MultiStepLR
from transformers import (
    Wav2Vec2PhonemeCTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
)

from utils.agent_utils import get_model
from utils.logger import init_logger
from utils.schedulers import TriStageLR
from decoders import CTCGreedyDecoder, CTCBeamSearchDecoder

class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param, vocab_phoneme_path=None):
        """
        method used to define our model parameters
        """
        super(BaseModule, self).__init__()
        self.save_hyperparameters()

        logger = init_logger('BaseModule', 'INFO')

        # Optimizer
        self.optim_param = optim_param
        self.lr = optim_param.lr

        logger.info(f'Optimizer : {optim_param.optimizer}, lr : {optim_param.lr}')

        # Tokenizer
        if vocab_phoneme_path is not None:
            # dirty hack to make the model usable across servers: shouldn't store paths in the checkpoint
            network_param.vocab_file = vocab_phoneme_path

        self.phonemes_tokenizer = Wav2Vec2PhonemeCTCTokenizer(
            vocab_file=network_param.vocab_file,
            eos_token=network_param.eos_token,
            bos_token=network_param.bos_token,
            unk_token=network_param.unk_token,
            pad_token=network_param.pad_token,
            word_delimiter_token=network_param.word_delimiter_token,
            do_phonemize=False,
        )

        network_param.vocab_size = self.phonemes_tokenizer.vocab_size

        # Loss function
        self.phoneme_blank_id = self.phonemes_tokenizer.encoder[network_param.word_delimiter_token]
        self.loss = nn.CTCLoss(
            blank=self.phoneme_blank_id
        )

        # Feature_extractor
        feature_extractor = Wav2Vec2FeatureExtractor(
            feature_size=1,
            sampling_rate=16000,
            padding_value=0.0,
            do_normalize=True,
            return_attention_mask=False,
        )

        logger.info(f'Features extractor : {network_param.network_name}')
        self.processor = Wav2Vec2Processor(
            feature_extractor=feature_extractor, tokenizer=self.phonemes_tokenizer
        )

        # Model
        self.model = get_model(network_param.network_name, network_param)

        logger.info(f'Model: {network_param.network_name}')
        self._configure_training_mode(network_param, logger)

        # Decoder
        self.decoder_type = network_param.decoder_type
        if self.decoder_type == 'greedy':
            self.decoder = CTCGreedyDecoder(self.phonemes_tokenizer)
        elif self.decoder_type == 'beam_search':
            self.decoder = CTCBeamSearchDecoder(
                tokenizer=self.phonemes_tokenizer,
                beam_size=network_param.beam_size,
                language_model_path=network_param.language_model_path,
                lm_weight=network_param.lm_weight,
                word_score=network_param.word_score,
                blank_token=network_param.word_delimiter_token
            )
        else:
            raise ValueError(f'Unknown decoder type: {self.decoder_type}')


    def _configure_training_mode(self, network_param, logger):
        """Configure which parts of the model to train"""

        # First, unfreeze everything
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        # Then selectively freeze based on config
        if network_param.freeze:
            logger.info("Freezing feature extractor")
            self.model.freeze_feature_encoder()

        if network_param.freeze_transformer:
            logger.info("Freezing transformer layers")
            self.model.freeze_encoder()

        # Set proper train/eval modes based on what's trainable
        for name, module in self.model.named_modules():
            if any(p.requires_grad for p in module.parameters()):
                module.train()
            elif list(module.parameters()):  # Has params but none trainable
                module.eval()

        # Log what's actually trainable
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Trainable parameters: {trainable_params:,} / {total_params:,} "
                    f"({100 * trainable_params / total_params:.1f}%)")

    def forward(self, x):
        output = self.model(x)
        return output

    def training_step(self, batch, batch_idx):
        """needs to return a loss from a single batch"""
        total_loss, phone_preds, phone_targets = self._get_outputs(batch, batch_idx)

        if total_loss != total_loss:
            print('loss is nan, model collapse, exiting')
            exit(1)

        self.log('train/total_loss', total_loss, batch_size=len(phone_preds))

        return {
            'loss': total_loss,
            'preds': phone_preds,
            'targets': phone_targets,
        }

    def validation_step(self, batch, batch_idx):
        total_loss, phone_preds, phone_targets = self._get_outputs(batch, batch_idx)

        self.log('val/total_loss', total_loss, batch_size=len(phone_preds))

        return {
            'loss': total_loss,
            'preds': phone_preds,
            'targets': phone_targets,
        }

    def test_step(self, batch, batch_idx):
        total_loss, phone_preds, phone_targets = self._get_outputs(batch, batch_idx)

        self.log('test/total_loss', total_loss, batch_size=len(phone_preds))

        return {
            'loss': total_loss,
            'preds': phone_preds,
            'targets': phone_targets,
        }

    def configure_optimizers(self):
        """defines model optimizer"""
        optimizer = getattr(torch.optim, self.optim_param.optimizer)
        optimizer = optimizer(
            self.parameters(),
            lr=self.lr,
            weight_decay=self.optim_param.weight_decay
        )

        if self.optim_param.scheduler is not None:
            if self.optim_param.scheduler == 'TriStage':
                scheduler = {
                    'scheduler': TriStageLR(
                        optimizer,
                        total_steps=self.optim_param.total_training_steps,
                        warmup_ratio=self.optim_param.tri_stage_warmup_ratio,
                        constant_ratio=self.optim_param.tri_stage_constant_ratio
                    ),
                    'interval': 'step',
                    'frequency': 1,
                    'name': 'tri_stage_lr'
                }
            elif self.optim_param.scheduler == 'Cosine':
                # Warmup from warmup_start_lr to lr
                warmup_scheduler = LinearLR(
                    optimizer,
                    start_factor=self.optim_param.warmup_start_lr / self.optim_param.lr,
                    total_iters=self.optim_param.warmup_steps
                )

                # Cosine decay from lr to eta_min
                cosine_scheduler = CosineAnnealingLR(
                    optimizer,
                    T_max=self.optim_param.max_steps - self.optim_param.warmup_steps,
                    eta_min=self.optim_param.eta_min
                )

                scheduler = {
                    'scheduler': SequentialLR(
                        optimizer,
                        [warmup_scheduler, cosine_scheduler],
                        [self.optim_param.warmup_steps]
                    ),
                    'interval': 'step',
                    'frequency': 1,
                    'name': 'cosine_lr'
                }

            elif self.optim_param.scheduler == 'StepLR':
                scheduler = {
                    'scheduler': StepLR(
                        optimizer,
                        step_size=self.optim_param.step_size_steps,
                        gamma=self.optim_param.gamma,
                    ),
                    'interval': 'step',
                    'frequency': 1,
                    'name': 'step_lr'
                }
            elif self.optim_param.scheduler == 'MultiStepLR':
                scheduler = {
                    'scheduler': MultiStepLR(
                        optimizer,
                        milestones=self.optim_param.milestones_steps,
                        gamma=self.optim_param.gamma,
                    ),
                    'interval': 'step',
                    'frequency': 1,
                    'name': 'multistep_lr'
                }
            elif self.optim_param.scheduler == 'ReduceLROnPlateau':
                scheduler = {
                    'scheduler': ReduceLROnPlateau(
                        optimizer,
                        mode='min',
                        patience=self.optim_param.patience,
                        min_lr=self.optim_param.min_lr,
                    ),
                    'monitor': 'val/loss',
                    'interval': 'epoch',  # This one stays epoch-based
                    'frequency': 1,
                    'name': 'plateau_lr'
                }
            else:
                raise ValueError(f"Unknown scheduler: {self.optim_param.scheduler}")

            return [optimizer], [scheduler]

        return optimizer

    def get_hidden_states(self, batch):
        """Get hidden states from encoder and extract target frames"""
        hidden_states = self.model(batch['array']).last_hidden_state

        # Contextual training: extract target frames
        target_frame_starts = torch.tensor(batch['target_frame_start'], dtype=torch.long)
        target_frame_ends = torch.tensor(batch['target_frame_end'], dtype=torch.long)

        frame_lengths = target_frame_ends - target_frame_starts
        max_target_frames = frame_lengths.max().item()

        # Create indices for extraction
        batch_indices = torch.arange(hidden_states.shape[0]).unsqueeze(1)
        frame_indices = torch.arange(max_target_frames).unsqueeze(0)
        absolute_indices = target_frame_starts.unsqueeze(1) + frame_indices
        absolute_indices = torch.clamp(absolute_indices, 0, hidden_states.shape[1] - 1)

        # Extract target frames
        hidden_states = hidden_states[batch_indices, absolute_indices]
        is_valid_mask = frame_indices < frame_lengths.unsqueeze(1)
        input_lengths = frame_lengths

        return hidden_states, input_lengths, is_valid_mask

    def get_logits(self, hidden_states):
        return self.model.phoneme_head(hidden_states)

    def mask_logits(self, logits, is_valid_mask):
        blank_id = self.phoneme_blank_id
        blank_logits = torch.full_like(logits[0, 0], float('-inf'))
        blank_logits[blank_id] = 10.0
        masked_logits = logits.clone()
        masked_logits[~is_valid_mask] = blank_logits
        return masked_logits

    def _compute_phoneme_loss(self, masked_logits, phoneme_sequences, input_lengths):
        log_probs = F.log_softmax(masked_logits, dim=-1).permute(1, 0, 2)

        labels = self.processor.tokenizer(phoneme_sequences).input_ids
        targets = torch.tensor(list(chain.from_iterable(labels)), dtype=torch.long)
        target_lengths = torch.tensor([len(targ) for targ in labels], dtype=torch.long)

        loss = self.loss(log_probs, targets, input_lengths, target_lengths)
        if torch.isinf(loss):
            print("input_lengths", input_lengths)
            print("target_lengths", target_lengths)
            print("total_loss", loss)

        return loss, labels

    def _get_outputs(self, batch, batch_idx):
        """convenience function since train/valid/test steps are similar"""
        hidden_states, input_lengths, is_valid_mask = self.get_hidden_states(batch)

        # Get phoneme logits from (possibly enriched) hidden states
        logits = self.get_logits(hidden_states)
        logits = self.mask_logits(logits, is_valid_mask)
        total_loss, batch['labels'] = self._compute_phoneme_loss(logits, batch['phonemes'], input_lengths)



        # Decode phoneme predictions
        with torch.no_grad():
            phone_preds = self._decode_predictions(logits.detach())
        phone_targets = self.processor.batch_decode(batch['labels'], group_tokens=False)

        return (
            total_loss,
            phone_preds,
            phone_targets,
        )

    def _decode_predictions(self, output):
        return self.processor.batch_decode(torch.argmax(output, dim=-1))