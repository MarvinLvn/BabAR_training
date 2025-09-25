"""Compare language models - simple table output"""

import argparse
import math
from pathlib import Path
import pandas as pd
from models.language_models import KenLMWrapper


def load_sequences(dataset_path, split):
    csv_path = Path(dataset_path) / f'{split}.csv'
    df = pd.read_csv(csv_path)
    sequences = []
    for seq in df['phones'].dropna():
        cleaned = seq.strip().rstrip('|').strip()
        if cleaned:
            sequences.append(cleaned)
    return sequences


def compute_perplexity(lm, sequences):
    total_log_prob = 0.0
    total_tokens = 0

    for seq in sequences:
        log_prob = lm.score_sequence(seq)
        tokens = seq.split()
        total_log_prob += log_prob
        total_tokens += len(tokens)

    avg_log_prob = total_log_prob / total_tokens
    perplexity = math.pow(10, -avg_log_prob)
    return perplexity


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--lm_dir', required=True)
    parser.add_argument('--orders', nargs='+', type=int, default=[1, 2, 3, 4, 5])
    args = parser.parse_args()

    train_sequences = load_sequences(args.dataset_path, 'train')
    val_sequences = load_sequences(args.dataset_path, 'val')

    print(f"Order  Train    Val      Ratio")
    print("-" * 30)

    lm_dir = Path(args.lm_dir)
    for order in sorted(args.orders):
        model_file = lm_dir / f"{order}gram.klm"

        if not model_file.exists():
            print(f"{order:<6} N/A      N/A      N/A")
            continue

        lm = KenLMWrapper(str(model_file))
        train_ppl = compute_perplexity(lm, train_sequences)
        val_ppl = compute_perplexity(lm, val_sequences)
        ratio = val_ppl / train_ppl

        print(f"{order:<6} {train_ppl:<8.2f} {val_ppl:<8.2f} {ratio:.2f}")


if __name__ == "__main__":
    main()