"""Train N-gram language model on TinyVox phoneme sequences using KenLM command-line tools"""

import argparse
import subprocess
import os
from pathlib import Path
import pandas as pd


def prepare_training_text(dataset_path, output_file):
    """Extract phoneme sequences from TinyVox training set and save as text"""
    dataset_path = Path(dataset_path)
    train_csv = dataset_path / 'train.csv'

    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)
    phoneme_seqs = df['phones'].dropna().tolist()

    # Clean sequences and add sentence boundaries
    with open(output_file, 'w') as f:
        for seq in phoneme_seqs:
            cleaned = seq.strip().rstrip('|').strip()
            if cleaned:
                # Add sentence boundaries and write
                f.write(f"{cleaned}\n")

    print(f"Training data prepared: {len(phoneme_seqs)} sequences")
    return len(phoneme_seqs)


def train_lm_with_kenlm(text_file, output_path, order=3):
    """Train language model using KenLM command-line tools"""
    arpa_file = f"{output_path}.arpa"
    binary_file = f"{output_path}.klm"

    print(f"Training {order}-gram model...")

    # Step 1: Train ARPA model
    cmd = f"lmplz -o {order} --mem 10G --discount_fallback < {text_file} > {arpa_file}"
    subprocess.run(cmd, shell=True, check=True)

    # Step 2: Convert to binary
    cmd = f"build_binary {arpa_file} {binary_file}"
    subprocess.run(cmd, shell=True, check=True)

    # Step 3: Remove ARPA file
    os.remove(arpa_file)

    print(f"Model saved: {binary_file}")
    return binary_file


def main():
    parser = argparse.ArgumentParser(description='Train N-gram LM using KenLM command-line tools')
    parser.add_argument('--dataset_path', required=True, help='Path to TinyVox dataset')
    parser.add_argument('--output_path', required=True, help='Output directory for trained models')
    parser.add_argument('--orders', nargs='+', type=int, default=[3],
                        help='N-gram orders to train (default: 3)')

    args = parser.parse_args()

    if 1 in args.orders:
        raise ValueError('KenLM implementation assumes at least a bigram model.')
    # Create output directory
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare shared training data file
    training_file = output_dir / "training_data.txt"

    num_sequences = prepare_training_text(args.dataset_path, training_file)
    print(f"Training data saved to: {training_file}")
    print("-" * 50)

    # Train models for each order
    trained_models = []
    for order in sorted(args.orders):
        model_path = output_dir / f"{order}gram"
        final_path = train_lm_with_kenlm(str(training_file), str(model_path), order)
        trained_models.append(final_path)

    print("-" * 50)
    print(f"Training complete! Processed {num_sequences} sequences")
    print("Trained models:")
    for model in trained_models:
        print(f"  {model}")
    print(f"Training data kept at: {training_file}")


if __name__ == "__main__":
    main()