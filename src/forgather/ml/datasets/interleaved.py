"""
Interleaved Dataset for combining multiple datasets.

Protocol-based implementation that works with any iterable dataset,
not just HuggingFace datasets. Preserves efficient checkpoint protocol.
"""

from typing import Any, Callable, Dict, List, Optional, Union

from torch.utils.data import IterableDataset as TorchIterableDataset


class InterleavedDataset(TorchIterableDataset):
    """
    Interleaved dataset combining multiple datasets.

    Works with any iterable dataset supporting the stateful protocol
    (state_dict/load_state_dict). Enables efficient multi-dataset
    pre-training with checkpoint support.

    Args:
        datasets: List of datasets to interleave
        probabilities: Sampling probabilities for each dataset. Can be:
            - None: Round-robin (equal probability)
            - List[float]: Static probabilities (normalized automatically)
            - Callable: Dynamic probabilities function called each iteration
                        Signature: (step, datasets, examples_per_dataset, exhausted) -> List[float]
        seed: Random seed for reproducible sampling
        stopping_strategy: 'first_exhausted' or 'all_exhausted'
    """

    def __init__(
        self,
        datasets: List,
        probabilities: Optional[Union[List[float], Callable]] = None,
        seed: Optional[int] = None,
        stopping_strategy: str = "first_exhausted",
    ):
        if not datasets:
            raise ValueError("Cannot interleave empty list of datasets")

        self.datasets = datasets
        self.seed = seed
        self.stopping_strategy = stopping_strategy

        # Handle probabilities (static list or dynamic callable)
        self._probabilities_callable = callable(probabilities)
        if self._probabilities_callable:
            # Store the callable function
            self._probabilities_fn = probabilities
            self.probabilities = None  # Will be computed dynamically
        else:
            self._probabilities_fn = None
            self.probabilities = probabilities

            # Validate static probabilities
            if probabilities is not None:
                if len(probabilities) != len(datasets):
                    raise ValueError(
                        f"Probabilities length ({len(probabilities)}) must match datasets length ({len(datasets)})"
                    )
                if not all(p >= 0 for p in probabilities):
                    raise ValueError("All probabilities must be non-negative")
                prob_sum = sum(probabilities)
                if prob_sum == 0:
                    raise ValueError("At least one probability must be > 0")
                # Normalize probabilities
                self.probabilities = [p / prob_sum for p in probabilities]

        # Validate stopping strategy
        if stopping_strategy not in ["first_exhausted", "all_exhausted"]:
            raise ValueError(
                f"Unsupported stopping_strategy: {stopping_strategy}. "
                f"Use 'first_exhausted' or 'all_exhausted'"
            )

        # Checkpoint state - which dataset and position within it
        self._current_dataset_index = 0
        self._current_example_count = 0  # Total examples yielded
        self._datasets_exhausted = [False] * len(datasets)

    def __repr__(self):
        s = "InterleavedDataset(\n"
        for ds in self.datasets:
            s += "  " + repr(ds) + ",\n"
        s += "  probabilities=" + repr(self.probabilities) + ",\n"
        s += "  seed=" + repr(self.seed) + ",\n"
        s += "  stopping_strategy=" + repr(self.stopping_strategy) + ",\n"
        s += "  _probabilities_fn=" + repr(self._probabilities_fn) + "\n)\n"
        return s

    def __iter__(self):
        """
        Interleave examples from child datasets.

        Uses probabilities for sampling if provided, otherwise round-robin.
        Respects stopping_strategy for when to stop iteration.
        """
        import random

        # Create iterators for all datasets
        iterators = [iter(dataset) for dataset in self.datasets]
        exhausted = [False] * len(self.datasets)

        # Track examples per dataset (for dynamic probabilities and checkpointing)
        examples_per_dataset = [0] * len(self.datasets)

        # Setup RNG if using probabilities (static or dynamic)
        use_probabilities = (
            self.probabilities is not None or self._probabilities_callable
        )
        rng = random.Random(self.seed) if use_probabilities else None

        # Track how many examples we've yielded (for checkpoint restoration)
        examples_yielded = 0

        # For round-robin, track current index
        current_idx = 0

        # Track iteration step for dynamic probabilities
        step = 0

        while True:
            # Check stopping condition (only for first_exhausted here)
            if self.stopping_strategy == "first_exhausted":
                if any(exhausted):
                    break

            # Choose which dataset to sample from
            if use_probabilities:
                # Probabilistic sampling from non-exhausted datasets
                available_indices = [i for i, ex in enumerate(exhausted) if not ex]
                if not available_indices:
                    break

                # Get current probabilities (static or dynamic)
                if self._probabilities_callable:
                    # Call dynamic probability function
                    current_probs = self._probabilities_fn(
                        step, self.datasets, examples_per_dataset, exhausted
                    )
                    # Validate returned probabilities
                    if len(current_probs) != len(self.datasets):
                        raise ValueError(
                            f"Probability function returned {len(current_probs)} values, "
                            f"expected {len(self.datasets)}"
                        )
                else:
                    # Use static probabilities
                    current_probs = self.probabilities

                # Compute probabilities for available datasets only
                available_probs = [current_probs[i] for i in available_indices]
                prob_sum = sum(available_probs)
                if prob_sum == 0:
                    break
                normalized_probs = [p / prob_sum for p in available_probs]

                # Sample from available datasets
                chosen_idx = rng.choices(available_indices, weights=normalized_probs)[0]
            else:
                # Round-robin through non-exhausted datasets
                attempts = 0
                while exhausted[current_idx] and attempts < len(self.datasets):
                    current_idx = (current_idx + 1) % len(self.datasets)
                    attempts += 1

                if exhausted[current_idx]:
                    # All exhausted
                    break

                chosen_idx = current_idx
                current_idx = (current_idx + 1) % len(self.datasets)

            # Try to get next example from chosen dataset
            try:
                example = next(iterators[chosen_idx])
                examples_yielded += 1
                examples_per_dataset[chosen_idx] += 1
                step += 1

                # Update checkpoint position
                self._current_dataset_index = chosen_idx
                self._current_example_count = examples_yielded

                yield example

            except StopIteration:
                # Mark as exhausted
                exhausted[chosen_idx] = True
                self._datasets_exhausted[chosen_idx] = True

                # For all_exhausted, continue with remaining datasets
                if self.stopping_strategy == "all_exhausted":
                    # Check if all are now exhausted
                    if all(exhausted):
                        break
                # For first_exhausted, we break at top of loop (already checked)

    def __len__(self) -> int:
        """
        Compute total length based on stopping strategy.

        Returns:
            Total number of examples that will be yielded
        """
        dataset_lengths = [len(ds) for ds in self.datasets]

        if self.stopping_strategy == "first_exhausted":
            if self.probabilities is None:
                # Round-robin: min_length * num_datasets
                return min(dataset_lengths) * len(self.datasets)
            else:
                # With probabilities: complex calculation
                # Approximate as maximum samples, with balanced datasets
                # TODO: Improve on this!
                min_length = sum(dataset_lengths)
                return min_length
        else:  # all_exhausted
            if self.probabilities is None:
                # Round-robin: each dataset visited once fully
                return sum(dataset_lengths)
            else:
                # With probabilities: max samples needed to exhaust all
                # This is an approximation
                return sum(dataset_lengths)

    @property
    def column_names(self) -> List[str]:
        """Get column names from first dataset."""
        if not self.datasets:
            return []
        if hasattr(self.datasets[0], "column_names"):
            return self.datasets[0].column_names
        return []

    @property
    def features(self):
        """Get features from first dataset."""
        if not self.datasets:
            return None
        if hasattr(self.datasets[0], "features"):
            return self.datasets[0].features
        return None

    @property
    def n_shards(self) -> int:
        """Total number of shards across all datasets."""
        total = 0
        for ds in self.datasets:
            if hasattr(ds, "n_shards"):
                total += ds.n_shards
            else:
                total += 1  # Assume 1 shard if not specified
        return total

    def state_dict(self) -> Dict[str, Any]:
        """
        Get checkpoint state for all child datasets.

        Returns:
            Dictionary with state for each child dataset plus interleaving state
        """
        state = {
            "current_dataset_index": self._current_dataset_index,
            "current_example_count": self._current_example_count,
            "datasets_exhausted": self._datasets_exhausted.copy(),
            "probabilities": self.probabilities,
            "seed": self.seed,
            "stopping_strategy": self.stopping_strategy,
            "child_states": [],
        }

        # Save state for each child dataset
        for i, dataset in enumerate(self.datasets):
            if hasattr(dataset, "state_dict"):
                state["child_states"].append(dataset.state_dict())
            else:
                # Dataset doesn't support state_dict, save None
                state["child_states"].append(None)

        return state

    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Restore checkpoint state for all child datasets.

        Args:
            state_dict: Dictionary from previous state_dict() call
        """
        self._current_dataset_index = state_dict["current_dataset_index"]
        self._current_example_count = state_dict["current_example_count"]
        self._datasets_exhausted = state_dict.get(
            "datasets_exhausted", [False] * len(self.datasets)
        )

        # Restore state for each child dataset
        child_states = state_dict.get("child_states", [])
        for i, (dataset, child_state) in enumerate(zip(self.datasets, child_states)):
            if child_state is not None and hasattr(dataset, "load_state_dict"):
                dataset.load_state_dict(child_state)


def balance_remaining_examples(
    step: int,
    datasets: List,
    examples_per_dataset: List[int],
    exhausted: List[bool],
) -> List[float]:
    """
    Dynamic probability function that weights datasets by estimated remaining examples.

    This encourages all datasets to finish at approximately the same time by
    giving higher weight to datasets with more remaining examples. Useful for
    balanced multi-dataset training where you want to consume all data sources
    proportionally.

    Args:
        step: Current iteration step (unused, but part of signature)
        datasets: List of child datasets
        examples_per_dataset: Number of examples already yielded from each dataset
        exhausted: Boolean list indicating which datasets are exhausted

    Returns:
        List of weights (one per dataset) for probabilistic sampling

    Example:
        >>> interleaved = interleave_datasets(
        ...     [ds1, ds2, ds3],
        ...     probabilities=balance_remaining_examples,
        ...     seed=42
        ... )
    """
    weights = []
    for i, (dataset, count, is_exhausted) in enumerate(
        zip(datasets, examples_per_dataset, exhausted)
    ):
        if is_exhausted:
            # Exhausted dataset gets zero weight
            weights.append(0.0)
        else:
            # Estimate remaining examples
            if hasattr(dataset, "__len__"):
                try:
                    total_length = len(dataset)
                    remaining = max(0, total_length - count)
                    weights.append(float(remaining))
                except (TypeError, AttributeError):
                    # len() failed, use equal weight
                    weights.append(1.0)
            else:
                # No length info, use equal weight
                weights.append(1.0)

    # Handle case where all weights are 0
    if sum(weights) == 0:
        weights = [1.0] * len(weights)

    return weights


def interleave_datasets(
    datasets: List,
    probabilities: Optional[Union[List[float], Callable]] = None,
    seed: Optional[int] = None,
    stopping_strategy: str = "first_exhausted",
):
    """
    Interleave multiple datasets into a single dataset.

    Protocol-based implementation that works with any iterable dataset,
    not just HuggingFace datasets. Preserves efficient checkpoint protocol.

    Args:
        datasets: List of datasets to interleave (any iterable with optional state_dict/load_state_dict)
        probabilities: Sampling probabilities for each dataset. Can be:
            - None: Round-robin (equal probability)
            - List[float]: Static probabilities (normalized automatically)
            - Callable: Dynamic probabilities function called each iteration
                        Signature: (step, datasets, examples_per_dataset, exhausted) -> List[float]
                        See balance_remaining_examples() for example implementation
        seed: Random seed for reproducible sampling
        stopping_strategy: 'first_exhausted' or 'all_exhausted'

    Returns:
        InterleavedDataset combining all input datasets

    Examples:
        # Round-robin interleaving
        >>> ds1 = fast_load_iterable_dataset("dataset1", split="train")
        >>> ds2 = fast_load_iterable_dataset("dataset2", split="train")
        >>> combined = interleave_datasets([ds1, ds2])

        # Probabilistic sampling (70% ds1, 30% ds2)
        >>> combined = interleave_datasets([ds1, ds2], probabilities=[0.7, 0.3], seed=42)

        # All exhausted (oversample smaller dataset)
        >>> combined = interleave_datasets([ds1, ds2], stopping_strategy="all_exhausted")

        # Works with StatefulDataLoader for checkpointing
        >>> dataloader = StatefulDataLoader(combined, batch_size=32)
        >>> # Training loop with checkpoint save/restore
    """
    return InterleavedDataset(
        datasets=datasets,
        probabilities=probabilities,
        seed=seed,
        stopping_strategy=stopping_strategy,
    )
