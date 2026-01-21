from torch.utils.data import IterableDataset


class IterableDatasetWithLength(IterableDataset):
    """
    A wrapper for iterable datasets that preserves length information.

    When converting map-style datasets to iterable datasets, the __len__ method
    is lost. This wrapper preserves the original dataset length to enable
    proper epoch step calculation in trainers.

    Automatically forwards all unknown methods and attributes to the wrapped
    dataset, including state_dict() and load_state_dict() for checkpointing.
    """

    def __init__(self, iterable_dataset, length: int):
        self._dataset = iterable_dataset
        self._length = length

    def __len__(self) -> int:
        return self._length

    def __iter__(self):
        return iter(self._dataset)

    def __repr__(self):
        return f"IterableDatasetWithLength({repr(self._dataset)}, length={repr(self._length)})"

    def map(self, *args, **kwargs):
        """Override map to preserve length information without double-wrapping."""
        mapped_dataset = self._dataset.map(*args, **kwargs)
        return IterableDatasetWithLength(mapped_dataset, self._length)

    def shuffle(self, *args, **kwargs):
        """Override shuffle to preserve length information without double-wrapping."""
        shuffled_dataset = self._dataset.shuffle(*args, **kwargs)
        return IterableDatasetWithLength(shuffled_dataset, self._length)

    def filter(self, *args, **kwargs):
        """Override filter - note that this may change the length."""
        filtered_dataset = self._dataset.filter(*args, **kwargs)
        # We can't know the new length after filtering, so we lose it
        return filtered_dataset

    def __getattr__(self, name):
        """Forward all unknown attributes/methods to the wrapped dataset."""
        return getattr(self._dataset, name)


def to_iterable_dataset_with_length(dataset, **kwargs):
    """
    Convert a map-style dataset to an iterable dataset while preserving length.

    Args:
        dataset: The map-style dataset to convert
        **kwargs: Additional arguments passed to to_iterable_dataset()

    Returns:
        IterableDatasetWithLength: Wrapped iterable dataset with preserved length
    """
    # If already wrapped, don't double-wrap
    if isinstance(dataset, IterableDatasetWithLength):
        return dataset

    if not hasattr(dataset, "__len__"):
        raise ValueError("Dataset must have __len__ method to preserve length")

    original_length = len(dataset)
    iterable_dataset = dataset.to_iterable_dataset(**kwargs)
    return IterableDatasetWithLength(iterable_dataset, original_length)
