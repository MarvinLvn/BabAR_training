# models/language_models.py
import kenlm
import math
import subprocess
from pathlib import Path
import pandas as pd
from collections import defaultdict, Counter


def train_kenlm_model(dataset_path, output_path, order=3):
    """Train N-gram model and save as binary"""
    print(f"Training {order}-gram model...")

    # Load training data
    dataset_path = Path(dataset_path)
    train_csv = dataset_path / 'train.csv'
    if not train_csv.exists():
        raise FileNotFoundError(f"Training CSV not found: {train_csv}")

    df = pd.read_csv(train_csv)
    phoneme_seqs = df['phones'].dropna().tolist()

    # Clean sequences
    sequences = []
    for seq in phoneme_seqs:
        cleaned = seq.strip().rstrip('|').strip()
        if cleaned:
            sequences.append(cleaned)

    print(f"Training on {len(sequences)} phoneme sequences")

    # Collect n-gram statistics
    ngram_counts = defaultdict(Counter)
    vocab = set()

    for seq in sequences:
        tokens = ['<s>'] + seq.split() + ['</s>']
        vocab.update(tokens)

        max_n = min(order, len(tokens))
        for n in range(1, max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n])
                ngram_counts[n][ngram] += 1

    # Add <unk> token to vocabulary
    vocab.add('<unk>')

    print(f"Vocabulary size: {len(vocab)}")

    # Create ARPA file then convert to binary
    arpa_file = f"{output_path}.arpa"
    binary_file = f"{output_path}.klm"

    _write_arpa_file(ngram_counts, vocab, arpa_file, order)

    # Convert to binary
    cmd = f"build_binary {arpa_file} {binary_file}"
    subprocess.run(cmd, shell=True, check=True)

    # Remove ARPA file, keep only binary
    Path(arpa_file).unlink()

    print(f"Binary model saved to: {binary_file}")
    return binary_file


def _write_arpa_file(ngram_counts, vocab, arpa_file, order):
    """Write ARPA format language model"""
    with open(arpa_file, 'w') as f:
        # Header
        f.write("\\data\\\n")
        for n in range(1, order + 1):
            if n in ngram_counts and len(ngram_counts[n]) > 0:
                f.write(f"ngram {n}={len(ngram_counts[n])}\n")
        f.write("\n")

        # Add <unk> to unigrams with small probability
        if 1 in ngram_counts:
            total_unigrams = sum(ngram_counts[1].values())
            unk_count = 1  # Small count for <unk>
            ngram_counts[1][('<unk>',)] = unk_count

        # N-gram sections
        for n in range(1, order + 1):
            if n in ngram_counts and len(ngram_counts[n]) > 0:
                f.write(f"\\{n}-grams:\n")

                for ngram, count in sorted(ngram_counts[n].items()):
                    if n == 1:
                        total_unigrams = sum(ngram_counts[1].values())
                        prob = count / total_unigrams
                    else:
                        context = ngram[:-1]
                        context_count = ngram_counts[n - 1].get(context, 0)
                        prob = count / context_count if context_count > 0 else 1e-10

                    log_prob = math.log10(prob) if prob > 0 else -99.0
                    ngram_str = ' '.join(ngram)
                    f.write(f"{log_prob:.6f}\t{ngram_str}\n")

                f.write("\n")

        f.write("\\end\\\n")


class KenLMWrapper:
    """Simple wrapper for KenLM models"""

    def __init__(self, model_path):
        self.model = kenlm.Model(str(model_path))

    def score_sequence(self, sequence):
        """Score a complete phoneme sequence"""
        return self.model.score(sequence, bos=True, eos=True)

    def score_next_phoneme(self, context, phoneme):
        """Score next phoneme given context"""
        if context:
            context_str = ' '.join(context) if isinstance(context, list) else context
            full_sequence = f"{context_str} {phoneme}"
            context_score = self.model.score(context_str, bos=True, eos=False)
            full_score = self.model.score(full_sequence, bos=True, eos=False)
            return full_score - context_score
        else:
            return self.model.score(phoneme, bos=True, eos=False)