from torchmetrics import Metric
import torch
from torch import Tensor, tensor
from typing import Any, Dict, List, Optional, Union, Tuple


class PhonemeErrorRate(Metric):
    """
    https://github.com/PyTorchLightning/metrics/blob/master/torchmetrics/text/wer.py#L23-L93
    """

    def __init__(self, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)
        self.add_state("errors", tensor(0, dtype=torch.float), dist_reduce_fx="sum")
        self.add_state("total", tensor(0, dtype=torch.float), dist_reduce_fx="sum")

    def update(self, preds, targets):
        """
        preds : list of sentence phoneme
        targets : list of sentence phoneme
        """
        errors, total = _per_update(preds, targets)

        self.errors += errors
        self.total += total

    def compute(self):
        return _per_compute(self.errors, self.total)

def _per_update(
    preds: Union[str, List[str]],
    target: Union[str, List[str]],
) -> Tuple[Tensor, Tensor]:
    """Update the wer score with the current set of references and predictions.
    Args:
        preds: Transcription(s) to score as a string or list of strings
        target: Reference(s) for each speech input as a string or list of strings
    Returns:
        Number of edit operations to get from the reference to the prediction, summed over all samples
        Number of words overall references
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(target, str):
        target = [target]
    errors = tensor(0, dtype=torch.float)
    total = tensor(0, dtype=torch.float)
    for pred, tgt in zip(preds, target):
        pred_tokens = pred.split()
        tgt_tokens = tgt.split()
        errors += _edit_distance(pred_tokens, tgt_tokens)
        total += len(tgt_tokens)
    return errors, total


def _per_compute(errors: Tensor, total: Tensor) -> Tensor:
    """Compute the word error rate.
    Args:
        errors: Number of edit operations to get from the reference to the prediction, summed over all samples
        total: Number of words overall references
    Returns:
        Word error rate score
    """
    return errors / total


def _edit_distance(prediction_tokens: List[str], reference_tokens: List[str]) -> int:
    """Standard dynamic programming algorithm to compute the edit distance.
    Args:
        prediction_tokens: A tokenized predicted sentence
        reference_tokens: A tokenized reference sentence
    Returns:
        Edit distance between the predicted sentence and the reference sentence
    """
    dp = [[0] * (len(reference_tokens) + 1) for _ in range(len(prediction_tokens) + 1)]
    for i in range(len(prediction_tokens) + 1):
        dp[i][0] = i
    for j in range(len(reference_tokens) + 1):
        dp[0][j] = j
    for i in range(1, len(prediction_tokens) + 1):
        for j in range(1, len(reference_tokens) + 1):
            if prediction_tokens[i - 1] == reference_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1]) + 1
    return dp[-1][-1]


class DetailedPhonemeErrorRate(Metric):
    """
    Enhanced PER metric that tracks insertions, deletions, and substitutions separately
    """

    def __init__(self, compute_on_step=False):
        super().__init__(compute_on_step=compute_on_step)
        self.add_state("insertions", tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("deletions", tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("substitutions", tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("total_ref_tokens", tensor(0, dtype=torch.int), dist_reduce_fx="sum")
        self.add_state("num_samples", tensor(0, dtype=torch.int), dist_reduce_fx="sum")

    def update(self, preds, targets):
        """
        preds : list of sentence phonemes
        targets : list of sentence phonemes
        """
        insertions, deletions, substitutions, total_ref = _detailed_per_update(preds, targets)

        self.insertions += insertions
        self.deletions += deletions
        self.substitutions += substitutions
        self.total_ref_tokens += total_ref
        self.num_samples += len(preds)

    def compute(self):
        """
        Returns a dictionary with detailed metrics
        """
        total_errors = self.insertions + self.deletions + self.substitutions
        per = total_errors / self.total_ref_tokens if self.total_ref_tokens > 0 else tensor(0.0)

        return {
            'per': per,
            'insertions': self.insertions,
            'deletions': self.deletions,
            'substitutions': self.substitutions,
            'total_errors': total_errors,
            'total_ref_tokens': self.total_ref_tokens,
            'num_samples': self.num_samples,
            'avg_insertions_per_sample': self.insertions / self.num_samples if self.num_samples > 0 else tensor(0.0),
            'avg_deletions_per_sample': self.deletions / self.num_samples if self.num_samples > 0 else tensor(0.0),
            'avg_substitutions_per_sample': self.substitutions / self.num_samples if self.num_samples > 0 else tensor(
                0.0),
        }

    def compute_sample_metrics(self, pred, target):
        """
        Compute detailed metrics for a single prediction-target pair
        Returns dict with per-sample metrics
        """
        return _compute_single_detailed_per(pred, target)


def _detailed_per_update(
        preds: Union[str, List[str]],
        targets: Union[str, List[str]],
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Update the detailed PER metrics with current predictions and targets

    Returns:
        insertions: Total insertions across all samples
        deletions: Total deletions across all samples
        substitutions: Total substitutions across all samples
        total_ref_tokens: Total reference tokens across all samples
    """
    if isinstance(preds, str):
        preds = [preds]
    if isinstance(targets, str):
        targets = [targets]

    total_insertions = tensor(0, dtype=torch.int)
    total_deletions = tensor(0, dtype=torch.int)
    total_substitutions = tensor(0, dtype=torch.int)
    total_ref_tokens = tensor(0, dtype=torch.int)

    for pred, target in zip(preds, targets):
        metrics = _compute_single_detailed_per(pred, target)
        total_insertions += metrics['insertions']
        total_deletions += metrics['deletions']
        total_substitutions += metrics['substitutions']
        total_ref_tokens += metrics['ref_length']

    return total_insertions, total_deletions, total_substitutions, total_ref_tokens


def _compute_single_detailed_per(pred: str, target: str) -> Dict[str, Union[float, int]]:
    """
    Compute detailed PER metrics for a single prediction-target pair

    Returns:
        Dictionary with insertions, deletions, substitutions, etc.
    """
    pred_tokens = pred.split()
    target_tokens = target.split()

    m, n = len(pred_tokens), len(target_tokens)

    # DP table for edit distance with operation tracking
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    ops = [[None] * (n + 1) for _ in range(m + 1)]

    # Initialize base cases
    for i in range(m + 1):
        dp[i][0] = i
        ops[i][0] = 'I' if i > 0 else None
    for j in range(n + 1):
        dp[0][j] = j
        ops[0][j] = 'D' if j > 0 else None

    # Fill DP table
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if pred_tokens[i - 1] == target_tokens[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                ops[i][j] = 'M'  # Match
            else:
                # Find minimum cost operation
                costs = [
                    (dp[i - 1][j] + 1, 'I'),  # Insertion
                    (dp[i][j - 1] + 1, 'D'),  # Deletion
                    (dp[i - 1][j - 1] + 1, 'S')  # Substitution
                ]
                min_cost, min_op = min(costs)
                dp[i][j] = min_cost
                ops[i][j] = min_op

    # Backtrack to count operations
    i, j = m, n
    insertions = deletions = substitutions = 0

    while i > 0 or j > 0:
        if i > 0 and j > 0 and ops[i][j] == 'M':
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and ops[i][j] == 'S':
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and ops[i][j] == 'I':
            insertions += 1
            i -= 1
        elif j > 0 and ops[i][j] == 'D':
            deletions += 1
            j -= 1
        else:
            break

    total_errors = insertions + deletions + substitutions
    per = total_errors / len(target_tokens) if len(target_tokens) > 0 else 0.0

    return {
        'per': per,
        'insertions': insertions,
        'deletions': deletions,
        'substitutions': substitutions,
        'total_errors': total_errors,
        'ref_length': len(target_tokens),
        'pred_length': len(pred_tokens)
    }