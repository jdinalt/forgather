from torch.utils.data import IterableDataset


class IterableDatasetWithLength(IterableDataset):
    """
    A thin wrapper that adds a known length to an iterable dataset.

    PyTorch's ``IterableDataset`` does not require ``__len__``, but trainers
    and data-loader utilities often query it to calculate epoch step counts.
    When a map-style ``Dataset`` is converted to an iterable form with
    ``to_iterable_dataset()``, the length information is lost. This wrapper
    re-attaches it.

    All attribute and method accesses that are not handled by this class are
    forwarded transparently to the wrapped dataset via ``__getattr__``,
    including ``state_dict`` / ``load_state_dict`` for checkpointing, and
    HuggingFace Dataset attributes such as ``column_names`` and ``features``.

    Parameters
    ----------
    iterable_dataset : IterableDataset
        The dataset to wrap. Any iterable dataset is accepted.
    length : int
        The length to report from ``__len__``. This value is not validated
        against the actual number of examples; the caller is responsible for
        supplying a consistent value.

    Notes
    -----
    The `map` and `shuffle` methods are overridden to return a new
    ``IterableDatasetWithLength`` with the same reported length, so that
    the length is preserved through chained transformations.

    `filter` is *not* overridden: the filtered dataset is returned as-is
    because the new length cannot be determined without iterating.

    Examples
    --------
    >>> from torch.utils.data import IterableDataset
    >>> ds = some_map_style_dataset.to_iterable_dataset()
    >>> ds_with_len = IterableDatasetWithLength(ds, length=len(some_map_style_dataset))
    >>> len(ds_with_len)
    1000
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
        """
        Apply a map transformation while preserving the reported length.

        Delegates to the wrapped dataset's ``map`` method and re-wraps the
        result in a new ``IterableDatasetWithLength`` with the same length.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the wrapped dataset's ``map``.
        **kwargs
            Keyword arguments forwarded to the wrapped dataset's ``map``.

        Returns
        -------
        IterableDatasetWithLength
            Mapped dataset with the same reported length as this instance.
        """
        mapped_dataset = self._dataset.map(*args, **kwargs)
        return IterableDatasetWithLength(mapped_dataset, self._length)

    def shuffle(self, *args, **kwargs):
        """
        Shuffle the dataset while preserving the reported length.

        Delegates to the wrapped dataset's ``shuffle`` method and re-wraps the
        result in a new ``IterableDatasetWithLength`` with the same length.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the wrapped dataset's ``shuffle``.
        **kwargs
            Keyword arguments forwarded to the wrapped dataset's ``shuffle``.

        Returns
        -------
        IterableDatasetWithLength
            Shuffled dataset with the same reported length as this instance.
        """
        shuffled_dataset = self._dataset.shuffle(*args, **kwargs)
        return IterableDatasetWithLength(shuffled_dataset, self._length)

    def filter(self, *args, **kwargs):
        """
        Filter the dataset.

        Delegates to the wrapped dataset's ``filter`` method. The length
        information is *not* preserved because the post-filter count cannot
        be determined without iterating.

        Parameters
        ----------
        *args
            Positional arguments forwarded to the wrapped dataset's ``filter``.
        **kwargs
            Keyword arguments forwarded to the wrapped dataset's ``filter``.

        Returns
        -------
        IterableDataset
            Filtered dataset without a ``__len__`` method.
        """
        filtered_dataset = self._dataset.filter(*args, **kwargs)
        # We can't know the new length after filtering, so we lose it
        return filtered_dataset

    def __getattr__(self, name):
        """Forward all unknown attribute and method accesses to the wrapped dataset."""
        return getattr(self._dataset, name)


def to_iterable_dataset_with_length(dataset, **kwargs):
    """
    Convert a map-style dataset to an iterable dataset while preserving its length.

    If ``dataset`` is already an ``IterableDatasetWithLength`` it is returned
    unchanged to avoid double-wrapping.

    Parameters
    ----------
    dataset : Dataset
        A map-style HuggingFace ``Dataset`` (or any object with ``__len__``
        and ``to_iterable_dataset``).
    **kwargs
        Additional keyword arguments forwarded to ``dataset.to_iterable_dataset()``.

    Returns
    -------
    IterableDatasetWithLength
        Iterable dataset that reports the original map-style length via
        ``__len__``.

    Raises
    ------
    ValueError
        If ``dataset`` does not have a ``__len__`` method.
    """
    # If already wrapped, don't double-wrap
    if isinstance(dataset, IterableDatasetWithLength):
        return dataset

    if not hasattr(dataset, "__len__"):
        raise ValueError("Dataset must have __len__ method to preserve length")

    original_length = len(dataset)
    iterable_dataset = dataset.to_iterable_dataset(**kwargs)
    return IterableDatasetWithLength(iterable_dataset, original_length)
