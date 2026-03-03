#!/usr/bin/env python3
"""
Inference script for applying pretrained phoneme recognizer to child utterances

Example usage:
python infer.py \
    --checkpoint_path weights/my_model/best.ckpt \
    --audio_path data/audio.wav \
    --rttm_path data/audio.rttm \
    --output_path results/predictions.csv \
    --speaker_filter KCHI \
    --context_duration 15 \
    --batch_size 32
"""

import argparse
from pathlib import Path
import torch
from tqdm import tqdm
import pandas as pd
import time
import soundfile as sf
import json
from models.BaseModule import BaseModule
from datamodules.contextual_vtc_datamodule import ContextualVTCDataModule
from utils.logger import init_logger


def save_batch(batch, predictions, output_folder, batch_idx):
    """
    Save all data from a batch to a specified folder

    Args:
        batch: Batch dict from dataloader
        predictions: List of predicted phoneme sequences
        output_folder: Path to folder where to save batch data
        batch_idx: Index of the batch (for naming files)
    """
    output_folder = Path(output_folder)
    batch_folder = output_folder / f"batch_{batch_idx:04d}"
    batch_folder.mkdir(parents=True, exist_ok=True)

    # Save audio arrays
    audio_folder = batch_folder / "audio"
    audio_folder.mkdir(exist_ok=True)

    batch_size = batch['array'].shape[0]
    sampling_rate = 16000

    for i in range(batch_size):
        # Get audio data
        context_audio = batch['array'][i].cpu().numpy()

        # Save full context window (15s)
        context_path = audio_folder / f"sample_{i:03d}_context.wav"
        sf.write(context_path, context_audio, sampling_rate)

        # Extract and save target segment
        # Convert frame indices to sample indices
        target_start_frame = batch['target_frame_start'][i]
        target_end_frame = batch['target_frame_end'][i]

        # Frame rate is 50 fps, so each frame corresponds to 20ms = 320 samples at 16kHz
        samples_per_frame = sampling_rate // 50  # 320 samples per frame

        target_start_sample = target_start_frame * samples_per_frame
        target_end_sample = target_end_frame * samples_per_frame

        # Clip to valid range
        target_start_sample = max(0, min(target_start_sample, len(context_audio)))
        target_end_sample = max(target_start_sample, min(target_end_sample, len(context_audio)))

        # Extract target segment
        target_audio = context_audio[target_start_sample:target_end_sample]

        # Save target segment
        target_path = audio_folder / f"sample_{i:03d}_target.wav"
        sf.write(target_path, target_audio, sampling_rate)

        # Save metadata for this sample
        metadata = {
            'path': batch['path'][i],
            'utterance_id': batch['utterance_id'][i],
            'utterance_onset_sec': batch['utterance_onset_sec'][i],
            'utterance_duration_sec': batch['utterance_duration_sec'][i],
            'speaker': batch['speaker'][i],
            'target_frame_start': batch['target_frame_start'][i],
            'target_frame_end': batch['target_frame_end'][i],
            'target_start_ms': batch['target_start_ms'][i],
            'target_end_ms': batch['target_end_ms'][i],
            'context_audio_file': f"sample_{i:03d}_context.wav",
            'target_audio_file': f"sample_{i:03d}_target.wav",
            'context_duration_sec': len(context_audio) / sampling_rate,
            'target_duration_sec': len(target_audio) / sampling_rate,
            'predicted_phonemes': predictions[i],
        }

        metadata_path = audio_folder / f"sample_{i:03d}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)

    # Save batch-level metadata
    batch_metadata = {
        'batch_idx': batch_idx,
        'batch_size': batch_size,
        'array_shape': list(batch['array'].shape),
        'sampling_rate': sampling_rate,
    }

    batch_metadata_path = batch_folder / "batch_metadata.json"
    with open(batch_metadata_path, 'w') as f:
        json.dump(batch_metadata, f, indent=2)

    return batch_folder


def load_model(checkpoint_path: Path, vocab_phoneme_path: Path = None):
    """Load model from checkpoint"""
    logger = init_logger("load_model", "INFO")
    logger.info(f"Loading model from: {checkpoint_path}")

    if vocab_phoneme_path is None:
        # Try to auto-detect
        vocab_phoneme_path = Path('assets/vocab_phoneme/vocab-phoneme-tinyvox.json')

    if not vocab_phoneme_path.exists():
        raise FileNotFoundError(f"Vocabulary file not found: {vocab_phoneme_path}")

    model = BaseModule.load_from_checkpoint(
        checkpoint_path,
        vocab_phoneme_path=vocab_phoneme_path,
    )
    model.eval()

    return model


def predict_batch(model, batch, device: str):
    """
    Predict phonemes for a batch of utterances

    Args:
        model: Loaded BaseModule
        batch: Batch dict from dataloader
        device: Device to use

    Returns:
        List of predicted phoneme sequences
    """
    with torch.no_grad():
        # Move to device
        batch['array'] = batch['array'].to(device)

        # Get hidden states and extract target frames
        hidden_states, input_lengths, is_valid_mask = model.get_hidden_states(batch)

        # Get phoneme logits
        logits = model.get_logits(hidden_states)

        # Mask logits
        logits = model.mask_logits(logits, is_valid_mask)

        # Decode predictions
        if model.decoder_type == 'greedy':
            predictions = model.decoder.decode(logits)
        elif model.decoder_type == 'beam_search':
            predictions, _ = model.decoder.decode(logits)
        else:
            predictions = model.processor.batch_decode(torch.argmax(logits, dim=-1))
    return predictions


def main():
    parser = argparse.ArgumentParser(
        description='Apply phoneme recognizer to child utterances from RTTM',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to model checkpoint (.ckpt)')
    parser.add_argument('--audio_path', required=True,
                        help='Path to audio file (.wav)')
    parser.add_argument('--rttm_path', required=True,
                        help='Path to RTTM file with utterance boundaries')

    # Optional arguments
    parser.add_argument('--output_folder', required=True,
                        help='Output folder where to stored the .csv file')
    parser.add_argument('--vocab_phoneme_path', default=None,
                        help='Path to vocabulary file (auto-detected if not provided)')
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                        help='Device to use')
    parser.add_argument('--context_duration', type=float, default=15.0,
                        help='Context window duration in seconds')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for inference')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--max_utt_dur', type=float, default=None,
                        help='Maximum utterance duration in seconds (filter out longer utterances)')

    args = parser.parse_args()

    # Start timing
    start_time = time.time()

    # Initialize logger
    logger = init_logger("infer", "INFO")

    # Convert paths
    checkpoint_path = Path(args.checkpoint_path)
    audio_path = Path(args.audio_path)
    rttm_path = Path(args.rttm_path)
    output_folder = Path(args.output_folder)
    output_path = output_folder / (audio_path.stem + '_phorec.csv')

    # Check if output already exists
    if output_path.exists():
        logger.info(f"Output file already exists: {output_path}. Skipping inference. Delete the file to rerun.")
        return

    # Validate inputs
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    if not rttm_path.exists():
        raise FileNotFoundError(f"RTTM file not found: {rttm_path}")

    # Set device
    device = 'cuda' if (args.device == 'cuda' and torch.cuda.is_available()) else 'cpu'
    logger.info(f"Using device: {device}")

    # Load model
    vocab_path = Path(args.vocab_phoneme_path) if args.vocab_phoneme_path else None
    model = load_model(checkpoint_path, vocab_path)
    model = model.to(device)

    logger.info(f"Model loaded successfully")
    logger.info(f"Decoder type: {model.decoder_type}")

    # Setup datamodule with initial batch size
    current_batch_size = args.batch_size
    min_batch_size = 16

    while current_batch_size >= min_batch_size:
        try:
            logger.info(f"Setting up datamodule with batch size: {current_batch_size}...")
            datamodule = ContextualVTCDataModule(
                audio_path=audio_path,
                rttm_path=rttm_path,
                context_duration=args.context_duration,
                batch_size=current_batch_size,
                num_workers=args.num_workers,
                speaker_filter='KCHI',
                max_utt_dur=args.max_utt_dur
            )

            # Set processor and setup
            datamodule.set_processor(model.processor)
            datamodule.setup()

            # Get dataloader
            dataloader = datamodule.dataloader()

            logger.info(f"Context duration: {args.context_duration}s")
            logger.info(f"Batch size: {current_batch_size}")
            logger.info(f"Total batches: {len(dataloader)}")

            # Process in batches
            logger.info("Processing utterances...")
            results = []

            for batch_idx, batch in enumerate(tqdm(dataloader, desc="Processing batches")):
                try:
                    # Get predictions
                    predictions = predict_batch(model, batch, device)

                    # Store results with specified columns
                    for i, pred in enumerate(predictions):
                        results.append({
                            'filename': audio_path.name,
                            'onset': batch['utterance_onset_sec'][i],
                            'offset': batch['utterance_onset_sec'][i] + batch['utterance_duration_sec'][i],
                            'speaker': batch['speaker'][i],
                            'phonemes': pred,
                        })

                except RuntimeError as e:
                    if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                        logger.warning(f"Out of memory error at batch {batch_idx}")
                        logger.warning(f"Current batch size: {current_batch_size}")

                        # Clear cache
                        if device == 'cuda':
                            torch.cuda.empty_cache()

                        # Reduce batch size and retry
                        current_batch_size = max(min_batch_size, current_batch_size // 2)
                        logger.info(f"Reducing batch size to {current_batch_size} and restarting...")

                        # Break out of batch loop to restart with new batch size
                        raise
                    else:
                        # Re-raise if not an OOM error
                        raise

            # If we got here, processing completed successfully
            break

        except RuntimeError as e:
            if "out of memory" in str(e).lower() or "oom" in str(e).lower():
                if current_batch_size <= min_batch_size:
                    logger.error(f"Out of memory even with minimum batch size ({min_batch_size})")
                    raise
                # Continue to next iteration with reduced batch size
                continue
            else:
                # Re-raise if not an OOM error
                raise

    # Save results as CSV
    df = pd.DataFrame(results, columns=['filename', 'onset', 'offset', 'speaker', 'phonemes'])
    output_folder.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    # Calculate runtime
    end_time = time.time()
    runtime = end_time - start_time

    logger.info(f"Results saved to: {output_path}")
    logger.info(f"Runtime: {runtime:.2f} seconds ({runtime / 60:.2f} minutes)")
    logger.info("Inference complete!")


if __name__ == "__main__":
    main()