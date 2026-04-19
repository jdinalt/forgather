import logging

import torch


def plot_token_length_histogram(
    dataset,
    tokenizer,
    output_file=None,
    sample_size=1000,
    feature="text",
    min=None,
    max=None,
):
    """
    Plot a histogram of token lengths in the dataset.
    If output_file is provided, save the histogram to that file.
    Otherwise, display the histogram.
    Args:
        dataset: The dataset to analyze.
        tokenizer: The tokenizer to use for tokenization.
        output_file: Optional; if provided, save the histogram to this file.
        sample_size: Number of samples to use for the histogram.
        feature: The feature in the dataset to analyze (default is 'text').
        min: Minimum length for the histogram (optional).
        max: Maximum length for the histogram (optional).
    """
    from itertools import islice

    import matplotlib.pyplot as plt
    import numpy as np

    # Suppress matplotlib warnings about missing fonts
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    if tokenizer:
        samples = [sample[feature] for sample in islice(dataset.shuffle(), sample_size)]
        outputs = tokenizer(
            samples,
            return_length=True,
        )
        lengths = torch.tensor(outputs["length"])
    else:
        lengths = torch.tensor(
            [
                len(sample["input_ids"])
                for sample in islice(dataset.shuffle(), sample_size)
            ]
        )
    print(f"sample size: {len(lengths)}")
    print(f"min: {lengths.min()}")
    print(f"max: {lengths.max()}")
    print(f"mean: {lengths.float().mean()}")
    print(f"median: {lengths.float().median()}")
    print(f"std: {lengths.float().std()}")
    counts, bins = np.histogram(lengths.numpy(), bins=100, density=True)
    fig, axs = plt.subplots(1, 1, figsize=(20, 5))
    axs.stairs(counts, bins)

    if output_file:
        plt.savefig(output_file, format="svg")
    else:
        plt.show()
