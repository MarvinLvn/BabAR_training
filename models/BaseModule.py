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

class BaseModule(LightningModule):
    def __init__(self, network_param, optim_param):
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
        # https://github.com/huggingface/transformers/blob/v4.16.2/src/transformers/models/wav2vec2_phoneme/tokenization_wav2vec2_phoneme.py
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
        self.loss = nn.CTCLoss(
            blank=self.phonemes_tokenizer.encoder[network_param.word_delimiter_token]
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

    def _configure_training_mode(self, network_param, logger):
        """Configure which parts of the model to train"""

        # First, unfreeze everything
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True

        # Then selectively freeze based on config
        if network_param.freeze:
            logger.info("Freezing feature extractor")
            self.model.model.freeze_feature_encoder()

        if network_param.freeze_transformer:
            logger.info("Freezing transformer layers")
            # Freeze transformer but keep head trainable
            for name, param in self.model.named_parameters():
                if 'lm_head' not in name:  # Keep lm_head trainable
                    param.requires_grad = False

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

        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)

        if loss != loss:
            print('loss is nan, model collapse, exiting')
            exit(1)

        # Log loss
        self.log('train/loss', loss, batch_size=len(preds))

        return {'loss': loss, 'logits': logits.detach(), 'preds': preds, 'targets': targets}

    def validation_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)

        # Log loss
        self.log('val/loss', loss, batch_size=len(preds))

        return {'loss': loss, 'logits': logits, 'preds': preds, 'targets': targets}

    def test_step(self, batch, batch_idx):
        """used for logging metrics"""
        loss, logits, preds, targets = self._get_outputs(batch, batch_idx)

        # Log loss
        self.log('test/loss', loss, batch_size=len(preds))

        return {'loss': loss, 'logits': logits, 'preds': preds, 'targets': targets}

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

    def _get_feat_extract_output_lengths(self, input_lengths):
        """
        Computes the output length of the convolutional layers
        """
        def _conv_out_length(input_length, kernel_size, stride):
            return (input_length - kernel_size) // stride + 1

        for kernel_size, stride in zip(self.model.model.config.conv_kernel, self.model.model.config.conv_stride):
            input_lengths = _conv_out_length(input_lengths, kernel_size, stride)

        return input_lengths

    def _get_outputs(self, batch, batch_idx):
        """convenience function since train/valid/test steps are similar"""
        x = batch

        output = self(x['array']).logits

        if 'target_frame_start' in x and 'target_frame_end' in x:
            # Contextual training: extract target frames
            target_frame_starts = torch.tensor(x['target_frame_start'])
            target_frame_ends = torch.tensor(x['target_frame_end'])

            frame_lengths = target_frame_ends - target_frame_starts
            max_target_frames = frame_lengths.max().item()

            # Create indices for extraction
            batch_indices = torch.arange(output.shape[0]).unsqueeze(1)
            frame_indices = torch.arange(max_target_frames).unsqueeze(0)
            absolute_indices = target_frame_starts.unsqueeze(1) + frame_indices
            absolute_indices = torch.clamp(absolute_indices, 0, output.shape[1] - 1)

            # Extract target frames
            output = output[batch_indices, absolute_indices]

            # Create mask
            is_valid_mask = torch.arange(max_target_frames, device=output.device).unsqueeze(0) < frame_lengths.to(
                output.device).unsqueeze(1)
            blank_token_id = self.loss.blank

            # Set invalid frames to blank token (simple masking)
            blank_logits = torch.full_like(output[0, 0], float('-inf'))
            blank_logits[blank_token_id] = 10.0
            output[~is_valid_mask] = blank_logits
            input_lengths = frame_lengths
        else:
            # Regular training: use full sequence
            audio_lengths = x['attention_mask'].sum(dim=1) if 'attention_mask' in x else torch.tensor(
                [output.shape[1]] * output.shape[0])
            input_lengths = self._get_feat_extract_output_lengths(audio_lengths)


        # process outputs
        log_probs = F.log_softmax(output, dim=-1)
        log_probs = log_probs.permute(1, 0, 2)
        # process targets
        # extract the indices from the dictionary
        x['labels'] = self.processor.tokenizer(x['phonemes']).input_ids
        target_lengths = torch.LongTensor([len(targ) for targ in x['labels']])
        targets = torch.Tensor(list(chain.from_iterable(x['labels']))).int()
        loss = self.loss(log_probs, targets, input_lengths, target_lengths)

        if torch.isinf(loss):
            print("paths", x['path'])
            print("input_lengths", input_lengths)
            print("target_lengths", target_lengths)
            print("loss", loss)
            
        # to compute metric and log samples
        phone_preds = self._decode_predictions(output)

        phone_targets = self.processor.batch_decode(x['labels'], group_tokens=False)

        return loss, output, phone_preds, phone_targets

    def _decode_predictions(self, output):
        return self.processor.batch_decode(torch.argmax(output, dim=-1))


