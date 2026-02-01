"""
Soft sequential probability function for interleaved datasets.

Provides a "soft" version of sequential dataset consumption where earlier
datasets in the sequence get higher probability, but the transition is gradual
rather than hard-switching.
"""

from typing import List


def soft_sequential(
    step: int,
    datasets: List,
    examples_per_dataset: List[int],
    exhausted: List[bool],
) -> List[float]:
    """
    Dynamic probability function that softly sequences through datasets.

    Similar to consuming datasets sequentially (A, then B, then C), but with
    a gradual transition. The probability of drawing from dataset i is proportional
    to its remaining examples, multiplied by the probability "left over" from all
    previous datasets.

    Algorithm:
        For dataset sequence [A, B, C]:
        - Dataset A gets probability proportional to its remaining fraction
        - The leftover probability is distributed among B, C using the same rule
        - This continues recursively

    Example:
        Given datasets [A, B, C] with 10 examples each:
        - A consumed 3/10 (7 remaining): A gets 70% probability
        - Remaining 30% split between B and C
        - B consumed 2/10 (8 remaining): B gets 30% * 80% = 24% probability
        - C gets remaining: 30% * 20% = 6% probability

    Args:
        step: Current iteration step (unused, but part of signature)
        datasets: List of child datasets (in order)
        examples_per_dataset: Number of examples already yielded from each dataset
        exhausted: Boolean list indicating which datasets are exhausted

    Returns:
        List of weights (one per dataset) for probabilistic sampling

    Example:
        >>> from forgather.ml.datasets import interleave_datasets
        >>> from forgather.ml.datasets.soft_sequential import soft_sequential
        >>> interleaved = interleave_datasets(
        ...     [ds1, ds2, ds3],
        ...     probabilities=soft_sequential,
        ...     seed=42
        ... )
    """
    weights = []
    remaining_prob = 1.0

    for i, (dataset, count, is_exhausted) in enumerate(
        zip(datasets, examples_per_dataset, exhausted)
    ):
        if is_exhausted:
            # Exhausted dataset gets zero weight
            weights.append(0.0)
            continue

        # Compute proportion of remaining examples for this dataset
        if hasattr(dataset, "__len__"):
            try:
                total_length = len(dataset)
                remaining = max(0, total_length - count)

                if total_length > 0:
                    proportion = remaining / total_length
                else:
                    # Empty dataset, give it equal weight
                    proportion = 1.0 / max(1, len(datasets) - sum(exhausted))
            except (TypeError, AttributeError):
                # len() failed, use equal proportion
                num_remaining = sum(1 for ex in exhausted if not ex)
                proportion = 1.0 / max(1, num_remaining)
        else:
            # No length info, use equal proportion among remaining datasets
            num_remaining = sum(1 for ex in exhausted if not ex)
            proportion = 1.0 / max(1, num_remaining)

        # This dataset gets 'proportion' of the remaining probability
        weight = remaining_prob * proportion
        weights.append(weight)

        # Reduce remaining probability for subsequent datasets
        remaining_prob *= (1.0 - proportion)

    # Handle edge case where all weights are 0
    if sum(weights) == 0:
        weights = [1.0] * len(weights)

    return weights
