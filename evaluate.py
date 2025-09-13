#!/usr/bin/env python3
"""
Script to evaluate PyTorch Lightning checkpoints on TinyVox

Example usage:
python evaluate.py --checkpoint_path weights/my_run/epoch-05-val_per=0.12.ckpt --split test
"""

import argparse
import json
import time
from pathlib import Path
import sys

import pandas as pd
import torch
from tqdm import tqdm

from models.BaseModule import BaseModule
from datamodules.tinyvox_datamodule import TinyVoxDataModule
from config.hparams import DatasetParams
from utils.per import DetailedPhonemeErrorRate
from utils.logger import init_logger


def load_model(checkpoint_path: Path):
    logger = init_logger("load_model", "INFO")
    logger.info(f"Loading model from: {checkpoint_path}")

    model = BaseModule.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model


def evaluate_model(model, dataloader, device, save_details=False):
    """Evaluate model on given dataloader"""
    detailed_per_metric = DetailedPhonemeErrorRate()
    detailed_results = []

    model = model.to(device)

    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            inputs = {"input_values": batch["array"].to(device)}

            # Get predictions
            outputs = model(inputs["input_values"])
            batch_predictions = model._decode_predictions(
                outputs.logits,
                batch.get("attention_mask")
            )

            batch_targets = batch['phonemes']

            # Update metrics
            detailed_per_metric.update(batch_predictions, batch_targets)

            # Store detailed results if requested
            if save_details:
                audio_filenames = [Path(path).name for path in batch['path']]

                for i in range(len(batch_predictions)):
                    from utils.per import _compute_single_detailed_per
                    sample_metrics = _compute_single_detailed_per(
                        batch_predictions[i], batch_targets[i]
                    )

                    detailed_results.append({
                        'audio_filename': audio_filenames[i],
                        'reference': batch_targets[i],
                        'hypothesis': batch_predictions[i],
                        'per': sample_metrics['per'],
                        'insertions': sample_metrics['insertions'],
                        'deletions': sample_metrics['deletions'],
                        'substitutions': sample_metrics['substitutions'],
                        'total_errors': sample_metrics['total_errors'],
                        'ref_length': sample_metrics['ref_length']
                    })

            # Memory cleanup
            if batch_idx % 25 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute final metrics
    final_metrics = detailed_per_metric.compute()

    results = {
        'per': final_metrics['per'].item(),
        'total_samples': final_metrics['num_samples'].item(),
        'total_insertions': final_metrics['insertions'].item(),
        'total_deletions': final_metrics['deletions'].item(),
        'total_substitutions': final_metrics['substitutions'].item(),
        'total_errors': final_metrics['total_errors'].item(),
        'total_ref_phonemes': final_metrics['total_ref_tokens'].item(),
        'avg_insertions_per_sample': final_metrics['avg_insertions_per_sample'].item(),
        'avg_deletions_per_sample': final_metrics['avg_deletions_per_sample'].item(),
        'avg_substitutions_per_sample': final_metrics['avg_substitutions_per_sample'].item(),
    }

    return results, detailed_results


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate PyTorch Lightning models on TinyVox',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Required arguments
    parser.add_argument('--checkpoint_path', required=True,
                        help='Path to PyTorch Lightning checkpoint (.ckpt file)')

    # Dataset arguments
    parser.add_argument('--dataset_path', default='/scratch2/mlavechin/tinyvox/TinyVox',
                        help='Path to TinyVox dataset')
    parser.add_argument('--inventory_path', default=None,
                        help='Path to unique_phonemes.json (auto-detected if not provided)')
    parser.add_argument('--use_vad', action='store_true',
                        help='Use audio_with_vad folder instead of audio')

    # Evaluation arguments
    parser.add_argument('--split', default='test', choices=['train', 'val', 'test'],
                        help='Dataset split to evaluate on')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='Number of workers for data loading')

    parser.add_argument('--save_details', action='store_true',
                        help='Save detailed per-sample results to CSV')

    # Technical arguments
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                        help='Device to use for evaluation')

    args = parser.parse_args()

    # Initialize logger
    logger = init_logger("evaluate", "INFO")

    # Convert paths
    checkpoint_path = Path(args.checkpoint_path)
    dataset_path = Path(args.dataset_path)

    # Validate inputs
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Auto-detect inventory path
    if args.inventory_path is None:
        inventory_path = dataset_path / 'unique_phonemes.json'
    else:
        inventory_path = Path(args.inventory_path)

    if not inventory_path.exists():
        raise FileNotFoundError(f"Inventory not found: {inventory_path}")

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    checkpoint_name = checkpoint_path.stem
    dataset_suffix = 'vad' if args.use_vad else 'raw'
    decode_suffix = 'greedy'
    output_dir = Path(f"evaluation_results/{checkpoint_name}_{dataset_suffix}_{args.split}_{decode_suffix}")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model(checkpoint_path)
    logger.info(f"Model loaded successfully")

    # Setup data
    logger.info("Setting up dataset...")
    data_params = DatasetParams()
    data_params.dataset_path = str(dataset_path)
    data_params.inventory_path = str(inventory_path)
    data_params.use_vad = args.use_vad
    data_params.custom_dataset = True
    data_params.batch_size = args.batch_size
    data_params.create_dataset = False
    data_params.num_workers = args.num_workers
    data_params.num_proc = 1

    # Initialize datamodule
    datamodule = TinyVoxDataModule(data_params)
    datamodule.set_processor(model.processor)

    # Setup the requested split
    if args.split == 'test':
        datamodule.setup('test')
        dataloader = datamodule.test_dataloader()
    elif args.split == 'val':
        datamodule.setup('fit')
        dataloader = datamodule.val_dataloader()
    elif args.split == 'train':
        datamodule.setup('fit')
        dataloader = datamodule.train_dataloader()

    logger.info(f"Dataset loaded: {args.split} split")

    # Run evaluation
    logger.info("Starting evaluation...")
    eval_start_time = time.time()

    results, detailed_results = evaluate_model(
        model, dataloader, device, args.save_details
    )

    eval_total_time = time.time() - eval_start_time

    # Print results
    print(f"Phoneme Error Rate (PER): {results['per']:.4f}")

    # Save results
    results_dict = {
        'checkpoint_path': str(checkpoint_path),
        'dataset_path': str(dataset_path),
        'split': args.split,
        'use_vad': args.use_vad,
        'eval_total_time': eval_total_time,
        'results': results
    }

    results_file = output_dir / 'results.json'
    with open(results_file, 'w') as f:
        json.dump(results_dict, f, indent=2)

    logger.info(f"Results saved to: {results_file.absolute()}")

    # Save detailed results if requested
    if args.save_details and detailed_results:
        results_df = pd.DataFrame(detailed_results)
        csv_file = output_dir / 'detailed_results.csv'
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Detailed results saved to: {csv_file}")

    logger.info("Evaluation complete!")


if __name__ == "__main__":
    main()