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
import torchaudio

import pandas as pd
import torch
from tqdm import tqdm

from models.BaseModule import BaseModule
from datamodules.contextual_tinyvox_datamodule import ContextualTinyVoxDataModule
from config.hparams import DatasetParams
from utils.per import PhonemeErrorRate, DetailedPhonemeErrorRate
from utils.logger import init_logger
from decoders import DecodingPipeline

def load_model(checkpoint_path: Path, vocab_phoneme_path: Path):
    logger = init_logger("load_model", "INFO")
    logger.info(f"Loading model from: {checkpoint_path}")

    # Load checkpoint and override the vocab file path
    model = BaseModule.load_from_checkpoint(
        checkpoint_path,
        vocab_phoneme_path=vocab_phoneme_path,
    )
    model.eval()
    return model


def get_metrics(model):
    phoneme_metric = DetailedPhonemeErrorRate()
    articulatory_detailed_metrics = {}

    if hasattr(model.model, 'articulatory_heads') and model.model.articulatory_heads is not None:
        feature_names = list(model.model.articulatory_vocabs.keys())
        for feature_name in feature_names:
            articulatory_detailed_metrics[feature_name] = DetailedPhonemeErrorRate()
        print(f"Model has articulatory heads for features: {feature_names}")

    return phoneme_metric, articulatory_detailed_metrics


def evaluate_model(model, decoding_pipeline, dataloader, device, save_details=False, postprocessing=False):
    """Evaluate model on given dataloader"""
    phoneme_metric, articulatory_detailed_metrics = get_metrics(model)
    detailed_results = []
    articulatory_detailed_results = {feature_name: [] for feature_name in articulatory_detailed_metrics.keys()}

    model = model.to(device)
    has_articulatory = hasattr(model.model, 'articulatory_heads') and model.model.articulatory_heads is not None

    if postprocessing:
        from utils.articulatory_features import ArticulatoryFeatureExtractor
        art_feature_extractor = ArticulatoryFeatureExtractor()
        articulatory_postproc_detailed_metrics = {
            feature_name: DetailedPhonemeErrorRate()
            for feature_name in art_feature_extractor.feature_names
        }
        articulatory_postproc_detailed_results = {
            feature_name: [] for feature_name in art_feature_extractor.feature_names
        }
        print(f"Will evaluate {len(art_feature_extractor.feature_names)} articulatory features via postprocessing")

    print("Running evaluation...")
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            # 1. Get phoneme predictions and update metrics
            batch['array'] = batch['array'].to(device)
            hidden_states, input_lengths, is_valid_mask = model.get_hidden_states(batch)
            phoneme_logits = model.get_logits(hidden_states, head='phoneme', is_valid_mask=is_valid_mask)
            batch_predictions = decoding_pipeline.decode(phoneme_logits)
            if decoding_pipeline.decoder_type == 'beam_search':
                batch_predictions = batch_predictions[0]
            batch_targets = batch['phonemes']
            phoneme_metric.update(batch_predictions, batch_targets)

            # 2. Get articulatory predictions and update metrics
            if has_articulatory:
                for feature_name, vocab in model.model.articulatory_vocabs.items():
                    feature_logits = model.get_logits(hidden_states, head=feature_name, is_valid_mask=is_valid_mask)
                    batch_feature_preds = model.decode_articulatory_predictions(feature_logits, vocab,
                                                                                model.hparams.network_param.word_delimiter_token)
                    batch_feature_targets = [' '.join(map(str, seq)) for seq in
                                             batch['articulatory_features'][feature_name]]

                    # Update detailed metrics only
                    articulatory_detailed_metrics[feature_name].update(batch_feature_preds, batch_feature_targets)

                    # Store detailed results if requested
                    if save_details:
                        audio_filenames = [Path(path).name for path in batch['path']]
                        from utils.per import _compute_single_detailed_per
                        for i in range(len(batch_feature_preds)):
                            sample_metrics = _compute_single_detailed_per(
                                batch_feature_preds[i], batch_feature_targets[i]
                            )
                            articulatory_detailed_results[feature_name].append({
                                'audio_filename': audio_filenames[i],
                                'reference': batch_feature_targets[i],
                                'hypothesis': batch_feature_preds[i],
                                'error_rate': sample_metrics['per'],
                                'insertions': sample_metrics['insertions'],
                                'deletions': sample_metrics['deletions'],
                                'substitutions': sample_metrics['substitutions'],
                                'total_errors': sample_metrics['total_errors'],
                                'ref_length': sample_metrics['ref_length']
                            })

            # 3. Postprocess phoneme predictions to get articulatory features
            if postprocessing:
                audio_filenames = [Path(path).name for path in batch['path']]
                for i in range(len(batch_predictions)):
                    pred_phonemes = batch_predictions[i]
                    target_phonemes = batch_targets[i]

                    # Extract articulatory features from both predicted and target phonemes
                    pred_features = art_feature_extractor.get_articulatory_features(pred_phonemes)
                    target_features = art_feature_extractor.get_articulatory_features(target_phonemes)

                    # Compare each articulatory feature
                    for feature_name in art_feature_extractor.feature_names:
                        pred_seq = ' '.join(map(str, pred_features[feature_name]))
                        target_seq = ' '.join(map(str, target_features[feature_name]))
                        articulatory_postproc_detailed_metrics[feature_name].update([pred_seq], [target_seq])

                        # Store detailed results if requested
                        if save_details:
                            from utils.per import _compute_single_detailed_per
                            sample_metrics = _compute_single_detailed_per(pred_seq, target_seq)
                            articulatory_postproc_detailed_results[feature_name].append({
                                'audio_filename': audio_filenames[i],
                                'reference': target_seq,
                                'hypothesis': pred_seq,
                                'error_rate': sample_metrics['per'],
                                'insertions': sample_metrics['insertions'],
                                'deletions': sample_metrics['deletions'],
                                'substitutions': sample_metrics['substitutions'],
                                'total_errors': sample_metrics['total_errors'],
                                'ref_length': sample_metrics['ref_length']
                            })

            # Store phoneme detailed results if requested
            if save_details:
                audio_filenames = [Path(path).name for path in batch['path']]
                from utils.per import _compute_single_detailed_per
                for i in range(len(batch_predictions)):
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

    # Compute final metrics
    final_metrics = phoneme_metric.compute()

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

    if save_details:
        results['phoneme_order'] = final_metrics['phoneme_order']
        results['inserted_phonemes'] = final_metrics['inserted_phonemes']
        results['deleted_phonemes'] = final_metrics['deleted_phonemes']
        results['substitution_matrix'] = final_metrics['substitution_matrix']

    # Add articulatory head results
    if has_articulatory:
        for feature_name, detailed_metric in articulatory_detailed_metrics.items():
            feature_detailed_metrics = detailed_metric.compute()
            # Get simple error rate from detailed metrics
            results[f'{feature_name}_er'] = feature_detailed_metrics['per'].item()

        # Add detailed articulatory metrics if save_details is True
        if save_details:
            results['articulatory_details'] = {}
            for feature_name, detailed_metric in articulatory_detailed_metrics.items():
                feature_detailed_metrics = detailed_metric.compute()
                results['articulatory_details'][feature_name] = {
                    'error_rate': feature_detailed_metrics['per'].item(),
                    'total_samples': feature_detailed_metrics['num_samples'].item(),
                    'total_insertions': feature_detailed_metrics['insertions'].item(),
                    'total_deletions': feature_detailed_metrics['deletions'].item(),
                    'total_substitutions': feature_detailed_metrics['substitutions'].item(),
                    'total_errors': feature_detailed_metrics['total_errors'].item(),
                    'total_ref_tokens': feature_detailed_metrics['total_ref_tokens'].item(),
                    'avg_insertions_per_sample': feature_detailed_metrics['avg_insertions_per_sample'].item(),
                    'avg_deletions_per_sample': feature_detailed_metrics['avg_deletions_per_sample'].item(),
                    'avg_substitutions_per_sample': feature_detailed_metrics['avg_substitutions_per_sample'].item(),
                    'feature_order': feature_detailed_metrics['phoneme_order'],
                    'inserted_features': feature_detailed_metrics['inserted_phonemes'],
                    'deleted_features': feature_detailed_metrics['deleted_phonemes'],
                    'substitution_matrix': feature_detailed_metrics['substitution_matrix'],
                }

    # Add postprocessing results
    if postprocessing:
        results['articulatory_postproc_details'] = {}
        for feature_name, detailed_metric in articulatory_postproc_detailed_metrics.items():
            feature_detailed_metrics = detailed_metric.compute()
            results[f'{feature_name}_er_postproc'] = feature_detailed_metrics['per'].item()

            if save_details:
                results['articulatory_postproc_details'][feature_name] = {
                    'error_rate': feature_detailed_metrics['per'].item(),
                    'total_samples': feature_detailed_metrics['num_samples'].item(),
                    'total_insertions': feature_detailed_metrics['insertions'].item(),
                    'total_deletions': feature_detailed_metrics['deletions'].item(),
                    'total_substitutions': feature_detailed_metrics['substitutions'].item(),
                    'total_errors': feature_detailed_metrics['total_errors'].item(),
                    'total_ref_tokens': feature_detailed_metrics['total_ref_tokens'].item(),
                    'avg_insertions_per_sample': feature_detailed_metrics['avg_insertions_per_sample'].item(),
                    'avg_deletions_per_sample': feature_detailed_metrics['avg_deletions_per_sample'].item(),
                    'avg_substitutions_per_sample': feature_detailed_metrics['avg_substitutions_per_sample'].item(),
                    'feature_order': feature_detailed_metrics['phoneme_order'],
                    'inserted_features': feature_detailed_metrics['inserted_phonemes'],
                    'deleted_features': feature_detailed_metrics['deleted_phonemes'],
                    'substitution_matrix': feature_detailed_metrics['substitution_matrix'],
                }

    # Return postproc results if they exist
    if postprocessing and save_details:
        return results, detailed_results, articulatory_detailed_results, articulatory_postproc_detailed_results
    else:
        return results, detailed_results, articulatory_detailed_results, {}

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
    parser.add_argument('--vocab_phoneme_path', default=None,
                        help='Path to vocab-phoneme-tinyvox (auto-detected if not provided)')
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

    # Data
    parser.add_argument('--context_duration', type=int, default=15,
                        help='Context size in seconds')

    # Decoding
    parser.add_argument('--decoder_type', default='greedy',
                        choices=['greedy', 'beam_search'])
    parser.add_argument('--beam_size', type=int, default=5)
    parser.add_argument('--language_model_path', type=str, default=None)
    parser.add_argument('--lm_weight', type=float, default=1.0)
    parser.add_argument('--word_score', type=float, default=0.0)
    parser.add_argument('--postprocessing', action='store_true',
                        help='Evaluate articulatory feature error rates by postprocessing predicted phoneme sequences')
    # Technical arguments
    parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'],
                        help='Device to use for evaluation')

    args = parser.parse_args()

    # Initialize logger
    logger = init_logger("evaluate", "INFO")

    # Convert paths
    checkpoint_path = Path(args.checkpoint_path)
    dataset_path = Path(args.dataset_path)
    language_model_path = Path(args.language_model_path) if args.language_model_path is not None else None

    # Validate inputs
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    # Auto-detect inventory path
    if args.vocab_phoneme_path is None:
        vocab_phoneme_path = Path('assets/vocab_phoneme/vocab-phoneme-tinyvox.json')
    else:
        vocab_phoneme_path = Path(args.vocab_phoneme_path)

    if not vocab_phoneme_path.exists():
        raise FileNotFoundError(f"vocab_phoneme_path not found: {vocab_phoneme_path}")

    # Set device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    logger.info(f"Using device: {device}")

    checkpoint_name = checkpoint_path.name
    model_name = checkpoint_path.parent.name
    dataset_suffix = 'vad' if args.use_vad else 'raw'
    if args.decoder_type == 'greedy':
        decode_suffix = 'greedy'
    else:
        decode_suffix = f'beam_search_lm_{language_model_path.stem}_beam_size_{args.beam_size}_lm_weight_{args.lm_weight}_word_score_{args.word_score}'
    output_dir = Path(f"evaluation_results/{model_name}/{checkpoint_name}_{dataset_suffix}/{args.split}_{decode_suffix}")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    acoustic_model = load_model(checkpoint_path, vocab_phoneme_path)

    # Decoding pipeline
    decoding_pipeline = DecodingPipeline(
        tokenizer=acoustic_model.phonemes_tokenizer,
        decoder_type=args.decoder_type,
        beam_size=args.beam_size,
        language_model_path=args.language_model_path,
        lm_weight=args.lm_weight,
        word_score=args.word_score
    )
    logger.info(f"Model loaded successfully")

    # Setup data
    logger.info("Setting up dataset...")
    data_params = DatasetParams()
    data_params.dataset_path = str(dataset_path)
    data_params.vocab_phoneme_path = str(vocab_phoneme_path)
    data_params.use_vad = args.use_vad
    data_params.custom_dataset = True
    data_params.batch_size = args.batch_size
    data_params.create_dataset = False
    data_params.num_workers = args.num_workers
    data_params.context_duration = args.context_duration
    data_params.num_proc = 1

    # Initialize datamodule
    datamodule = ContextualTinyVoxDataModule(data_params)
    datamodule.set_processor(acoustic_model.processor)
    if hasattr(acoustic_model.model, 'articulatory_heads') and acoustic_model.model.articulatory_heads is not None:
        datamodule.set_articulatory_feature_extractor()

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

    results, detailed_results, articulatory_detailed_results, articulatory_postproc_detailed_results = evaluate_model(
        acoustic_model, decoding_pipeline, dataloader, device, args.save_details, args.postprocessing
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
        json.dump(results_dict, f, indent=2, default=str)

    logger.info(f"Results saved to: {results_file.absolute()}")

    # Save detailed phoneme results if requested
    if args.save_details and detailed_results:
        results_df = pd.DataFrame(detailed_results)
        csv_file = output_dir / 'detailed_phoneme_results.csv'
        results_df.to_csv(csv_file, index=False)
        logger.info(f"Detailed phoneme results saved to: {csv_file}")

    # Save detailed articulatory results if requested
    if args.save_details and articulatory_detailed_results:
        for feature_name, feature_results in articulatory_detailed_results.items():
            if feature_results:
                results_df = pd.DataFrame(feature_results)
                csv_file = output_dir / f'detailed_{feature_name}_results.csv'
                results_df.to_csv(csv_file, index=False)
                logger.info(f"Detailed {feature_name} results saved to: {csv_file}")

    # Save detailed postprocessing articulatory results if requested
    if args.save_details and articulatory_postproc_detailed_results:
        for feature_name, feature_results in articulatory_postproc_detailed_results.items():
            if feature_results:
                results_df = pd.DataFrame(feature_results)
                csv_file = output_dir / f'detailed_{feature_name}_postproc_results.csv'
                results_df.to_csv(csv_file, index=False)
                logger.info(f"Detailed {feature_name} postprocessing results saved to: {csv_file}")

    logger.info("Evaluation complete!")

if __name__ == "__main__":
    main()