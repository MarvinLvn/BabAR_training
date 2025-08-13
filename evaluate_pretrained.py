#!/usr/bin/env python3
"""
Script to evaluate pretrained model on TinyVox

Example use:

python evaluate_pretrained.py --dataset_path /scratch2/mlavechin/tinyvox/TinyVox --network_name Wav2Vec2 --pretrained_name facebook/wav2vec2-lv-60-espeak-cv-ft --batch_size 64 --save_details --split test
"""
import argparse
import json
from pathlib import Path

import pandas as pd
import panphon.distance as pdist
import torch
from tqdm import tqdm
from transformers import (
    Wav2Vec2ForCTC, Wav2Vec2Processor,
    WavLMForCTC,
    HubertForCTC
)

from datamodules.tinyvox_datamodule import TinyVoxDataModule
from config.hparams import DatasetParams
from utils.per import DetailedPhonemeErrorRate
from utils.agent_utils import load_custom_wav2vec2_model

def load_model_and_processor(model_name, pretrained_name):
    """Load pretrained model and processor without modification"""
    print(f"Loading {model_name}: {pretrained_name}")
    if model_name == 'Wav2Vec2':
        if Path(pretrained_name).exists():
            model, processor = load_custom_wav2vec2_model(pretrained_name)
        else:
            model = Wav2Vec2ForCTC.from_pretrained(pretrained_name)
            processor = Wav2Vec2Processor.from_pretrained(pretrained_name)
    elif model_name == 'WavLM':
        model = WavLMForCTC.from_pretrained(pretrained_name)
        # WavLM doesn't have its own processor, use Wav2Vec2
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    elif model_name == 'Hubert':
        model = HubertForCTC.from_pretrained(pretrained_name)
        # HuBERT doesn't have its own processor, use Wav2Vec2
        processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    else:
        raise ValueError(f"Unknown model type: {model_name}")

    return model, processor

def simplify_sound(complex_sound, target_inventory):
    if complex_sound in target_inventory:
        return complex_sound

    distance_calculator = pdist.Distance()

    distances = {}
    for sound in target_inventory:
        distance = distance_calculator.feature_edit_distance(complex_sound, sound)
        distances[sound] = distance

    closest_sound = min(distances, key=distances.get)
    return closest_sound


def compute_mapping(processor, dataset_path, out_dir=None, recompute_mapping=False):
    '''
    For each sound of the input_inventor (model dependent), maps it to the target phonetic inventory.
    Time complexity is O(n*m) with n the size of the input inventory and m the size of the target inventory.

    :param input_inventory: Input inventory predicted by the model
    :param dataset_path: Path to the dataset folder with unique_phonemes.json containing the target inventory
    :param out_dir: Where to store the resulting mapping
    :return: The resulting mapping
    '''
    out_file = out_dir / 'inventory_mapping.json'
    if out_file.is_file() and not recompute_mapping:
        print(f"Loading precomputed mapping from {out_file}")
        with open(out_file, 'r') as f:
            return json.load(f)

    # Read target inventory
    print(f"{out_file} not found. Computing it.")
    with open(dataset_path / 'unique_phonemes.json') as fin:
        target_inventory = set(json.load(fin))

    # Compute mapping
    input_inventory = set(processor.tokenizer.get_vocab())

    # Initialize mapping
    mapping = {}
    if processor.tokenizer.pad_token:
        mapping[processor.tokenizer.pad_token] = ''
        input_inventory.discard(processor.tokenizer.pad_token)
    if processor.tokenizer.bos_token:
        mapping[processor.tokenizer.bos_token] = ''
        input_inventory.discard(processor.tokenizer.bos_token)
    if processor.tokenizer.eos_token:
        mapping[processor.tokenizer.eos_token] = ''
        input_inventory.discard(processor.tokenizer.eos_token)
    if processor.tokenizer.unk_token:
        mapping[processor.tokenizer.unk_token] = '<unk>'
        input_inventory.discard(processor.tokenizer.unk_token)
    if hasattr(processor.tokenizer, 'word_delimiter_token') and processor.tokenizer.word_delimiter_token:
        mapping[processor.tokenizer.word_delimiter_token] = ''
        input_inventory.discard(processor.tokenizer.word_delimiter_token)

    # For sounds second
    for input_sound in tqdm(input_inventory):
        mapping[input_sound] = simplify_sound(input_sound, target_inventory)

    # Find unused sounds from target inventory
    used_target_sounds = set(mapping.values()) - {'', '<unk>'}  # Exclude special mappings
    unused_sounds = target_inventory - used_target_sounds
    mapping['unused'] = sorted(unused_sounds)
    print(f"Mapped {len(input_inventory)} input sounds to {len(target_inventory)} target sounds")
    print(f"Found {len(unused_sounds)} sounds that cannot be predicted: {unused_sounds}.")

    if out_dir is not None:
        with open(out_file, 'w') as f:
            json.dump(mapping, f, indent=2, ensure_ascii=False)
    return mapping


def apply_phoneme_mapping(predictions, phoneme_mapping):
    """Apply phoneme mapping to convert model predictions to target inventory"""
    mapped_predictions = []

    for prediction in predictions:
        mapped_tokens = []
        for token in prediction.split():
            if token in phoneme_mapping:
                mapped_token = phoneme_mapping[token]
                if mapped_token and mapped_token != '':  # Skip removed tokens
                    mapped_tokens.append(mapped_token)
        mapped_predictions.append(' '.join(mapped_tokens))

    return mapped_predictions

def remove_word_boundaries(phoneme_sequence):
    """Remove word boundary tokens from phoneme sequence"""
    return ' '.join(token for token in phoneme_sequence.split() if token != '|')


def evaluate_pretrained(model_name, pretrained_name, dataset_path, use_vad=False, batch_size=32, out_dir=None,
                        recompute_mapping=False, split='test', save_details=False):
    """
    Evaluate a pre-trained model on TinyVox dataset
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 1. Load TinyVox data
    print("Setting up TinyVox data...")

    # Create dataset params
    data_params = DatasetParams()
    data_params.dataset_path = dataset_path
    data_params.use_vad = use_vad
    data_params.custom_dataset = True
    data_params.batch_size = batch_size
    data_params.create_dataset = False
    data_params.num_workers = 8
    data_params.num_proc = 8

    # 2. Load pretrained model as-is (no modification)
    model, processor = load_model_and_processor(model_name, pretrained_name)
    model = model.to(device)
    model.eval()

    # 3. Compute mapping between predicted and expected phonetic inventory
    phoneme_mapping = compute_mapping(processor, dataset_path, out_dir.parent.parent, recompute_mapping)
    print(f'Mapping {len(phoneme_mapping.keys())} input phonemes to target phonemes.')
    processor.tokenizer.get_vocab()

    # 4. Setup data
    print("Loading data...")
    datamodule = TinyVoxDataModule(data_params)
    if split == 'test':
        datamodule.setup(split, processor)
        dataloader = datamodule.test_dataloader
    elif split == 'val':
        datamodule.setup('fit', processor)
        dataloader = datamodule.val_dataloader
    elif split == 'train':
        datamodule.setup('fit', processor)
        dataloader = datamodule.train_dataloader

    # 5. Run evaluation
    print("Running evaluation...")
    detailed_per_metric = DetailedPhonemeErrorRate()
    detailed_results = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(dataloader())):

            inputs = {
                "input_values": batch["input_values"].to(device),
            }

            # Get predictions from model
            logits = model(**inputs).logits
            predicted_ids = torch.argmax(logits, dim=-1)

            # Move to CPU immediately and decode
            batch_predictions = processor.batch_decode(predicted_ids.cpu())

            # Map to target inventory
            batch_predictions = apply_phoneme_mapping(batch_predictions, phoneme_mapping)

            # Remove word boundaries in target
            batch_targets = [remove_word_boundaries(target) for target in batch['phonemes']]

            # Update metric directly (but reset periodically)
            detailed_per_metric.update(batch_predictions, batch_targets)

            # Process detailed results immediately if needed
            if save_details:
                audio_filenames = [Path(path).name for path in batch['path']]

                for i in range(len(batch_predictions)):
                    # Compute per-sample metrics without storing in main metric
                    from utils.per import _compute_single_detailed_per
                    sample_metrics = _compute_single_detailed_per(
                        batch_predictions[i],
                        batch_targets[i]
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

            # Clean up GPU memory and variables
            del logits, predicted_ids, inputs, batch_predictions, batch_targets

            # Aggressive memory cleanup every 25 batches
            if batch_idx % 25 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Compute final metrics
    print("Computing final metrics...")
    final_detailed_per = detailed_per_metric.compute()

    # Save detailed CSV if requested
    if save_details and detailed_results:
        results_df = pd.DataFrame(detailed_results)
        csv_file = out_dir / 'detailed_results.csv'
        results_df.to_csv(csv_file, index=False)
        print(f"Detailed per-sample results saved to: {csv_file}")

    # Prepare summary statistics for JSON output
    summary_stats = {
        'overall_per': final_detailed_per['per'].item(),
        'total_samples': final_detailed_per['num_samples'].item(),
        'total_insertions': final_detailed_per['insertions'].item(),
        'total_deletions': final_detailed_per['deletions'].item(),
        'total_substitutions': final_detailed_per['substitutions'].item(),
        'total_errors': final_detailed_per['total_errors'].item(),
        'total_ref_phonemes': final_detailed_per['total_ref_tokens'].item(),
        'avg_insertions_per_sample': final_detailed_per['avg_insertions_per_sample'].item(),
        'avg_deletions_per_sample': final_detailed_per['avg_deletions_per_sample'].item(),
        'avg_substitutions_per_sample': final_detailed_per['avg_substitutions_per_sample'].item(),
    }

    # Final memory cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        'per': final_detailed_per['per'].item(),
        'detailed_metrics': final_detailed_per,
        'summary_stats': summary_stats,
    }


def main():
    parser = argparse.ArgumentParser(description='Hybrid evaluation: TinyVox data + pretrained models as-is')
    parser.add_argument('--dataset_path', required=True, help='Path to TinyVox dataset')
    parser.add_argument('--network_name', required=False, choices=['Wav2Vec2', 'WavLM', 'Hubert'])
    parser.add_argument('--pretrained_name', required=False, help="HuggingFace model identifier or path to Jialu Li's model")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--use_vad', action='store_true')
    parser.add_argument('--recompute_mapping', action='store_true')
    parser.add_argument('--split', required=False, default='test', choices=['train', 'val', 'test'])
    parser.add_argument('--save_details', action='store_true', help='Save detailed per-sample results to CSV')

    args = parser.parse_args()
    args.dataset_path = Path(args.dataset_path)


    if args.pretrained_name not in ['facebook/wav2vec2-lv-60-espeak-cv-ft'] and not Path(args.pretrained_name).exists():
        raise ValueError(f"Unknown pretrained model: {args.pretrained_name}")

    model_short = args.pretrained_name.replace('/', '_')

    if args.use_vad:
        dataset_name = 'tinyvox_audio_with_vad'
    else:
        dataset_name = 'tinyvox_audio'
    out_dir = Path(f"results/{model_short}/{dataset_name}/{args.split}")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset_path}")
    print(f"Model: {args.network_name} - {args.pretrained_name}")

    results = evaluate_pretrained(
        args.network_name,
        args.pretrained_name,
        args.dataset_path,
        args.use_vad,
        args.batch_size,
        out_dir,
        args.recompute_mapping,
        args.split,
        args.save_details
    )

    print(f"Evaluation complete!")
    print(f"Phoneme Error Rate (PER): {results['per']:.4f}")

    detailed = results['detailed_metrics']
    print(f"Total samples: {detailed['num_samples'].item()}")
    print(f"Total errors: {detailed['total_errors'].item()}")
    print(f"  - Insertions: {detailed['insertions'].item()}")
    print(f"  - Deletions: {detailed['deletions'].item()}")
    print(f"  - Substitutions: {detailed['substitutions'].item()}")

    # Save results
    out_file = out_dir / 'tinyvox_per.json'
    out_dict = {
        'per': results['per'],
        'summary_stats': results['summary_stats']
    }

    with open(out_file, 'w') as f:
        json.dump(out_dict, f, indent=2, default=str)

    print(f"Results saved to: {out_file}")



if __name__ == "__main__":
    main()