# SmolLM-Corpus

https://huggingface.co/datasets/HuggingFaceTB/smollm-corpus

> This dataset is a curated collection of high-quality educational and synthetic data designed for training small language models.

## Configurations

- [smollm-corpus/cosmopedia-v2.yaml](./templatelib/configs/smollm-corpus/cosmopedia-v2.yaml) Fast Cosmopedia v2  
Cosmopedia v2 is an enhanced version of Cosmopedia, the largest synthetic dataset for pre-training, consisting of over 39 million textbooks, blog posts, and stories

- [smollm-corpus/fineweb-edu-dedup.yaml](./templatelib/configs/smollm-corpus/fineweb-edu-dedup.yaml) : Fast FineWeb-Edu-Dedup  
FineWeb-Edu-Dedup is a deduplicated subset of the FineWeb-Edu dataset, containing 220 billion tokens of educational web pages.

- [smollm-corpus/cosmopedia-v2-packed.yaml](./templatelib/configs/smollm-corpus/cosmopedia-v2-packed.yaml) : Fast Cosmopedia v2 packed  
Fast Cosmopedia v2 Packed Sequences

- [smollm-corpus/fineweb-edu-packed.yaml](./templatelib/configs/smollm-corpus/fineweb-edu-packed.yaml) Fast FineWeb-Edu-Dedup Packed  
Fast FineWeb-Edu-Dedup Packed Sequences

- [smollm-corpus/interleaved.yaml](./templatelib/configs/smollm-corpus/interleaved.yaml) Small LM Interleaved  
Interleave all Small-LM datasets

- [smollm-corpus/interleaved-packed.yaml](./templatelib/configs/smollm-corpus/interleaved-packed.yaml) Small LM Interleaved Packed  
Interleave and Pack all Small-LM datasets

## Supporting Templates

- [datasets/dataset_type.yaml](../../../templatelib/base/datasets/dataset_type.yaml)
The base template for all dataset projects.

- [datasets/tokenized_dataset.yaml](../../../templatelib/base/datasets/tokenized_dataset.yaml)
A sub-class of "dataset_type.yaml," which provides some common definitions.

- [small-lm-base.yaml](./templatelib/small-lm-base.yaml)  
This defines the base template shared between the datasets. As these datasets only have a "train" split, we create the other splits by slicing "train."

- [packed.yaml](./templatelib/packed.yaml)
This defines the common parameters for the packed configurations.

## Fast Dataset Loading with `fast_load_iterable_dataset`

This project demonstrates `fast_load_iterable_dataset`, a high-performance dataset loader designed for large-scale training with efficient checkpoint resumption.

### The Problem with Standard HuggingFace Loading

Traditional HuggingFace dataset loading faces two critical issues for large datasets:

1. **Slow Initial Load**: Every time you load a dataset, HuggingFace re-downloads, re-processes, and rebuilds it
2. **Inefficient Checkpoint Resumption**: When resuming training mid-epoch, the standard approach iterates through N examples to reach the checkpoint position, which can take hours for large datasets

### How `fast_load_iterable_dataset` Solves This

**First Load** (Slow - One Time Only):
- Downloads dataset from HuggingFace
- Indexes the Arrow files in HuggingFace's cache
- Saves index with file paths and per-file example counts
- Typical time: 10-20 minutes for large datasets

**Subsequent Loads** (Instant):
- Reads index file (< 1 second)
- Memory-maps Arrow files directly
- No download, no processing, no waiting
- Typical time: < 1 second

**Checkpoint Resumption** (Position-Based):
- Instead of iterating N steps, jumps directly to file and example position
- Resumption time: < 1 second (vs hours with standard approach)
- Preserves exact training state across restarts

### Split Notation Support

The loader supports HuggingFace's split notation for creating virtual splits without copying data:

```yaml
# From small-lm-base.yaml
train_dataset_split: !singleton:forgather.ml.datasets:fast_load_iterable_dataset
    path: HuggingFaceTB/smollm-corpus
    name: cosmopedia-v2
    split: "train[10000:]"  # Everything after first 10k examples

validation_dataset_split: !singleton:forgather.ml.datasets:fast_load_iterable_dataset
    path: HuggingFaceTB/smollm-corpus
    name: cosmopedia-v2
    split: "train[0:1000]"  # First 1k examples for validation

test_dataset_split: !singleton:forgather.ml.datasets:fast_load_iterable_dataset
    path: HuggingFaceTB/smollm-corpus
    name: cosmopedia-v2
    split: "train[1000:10000]"  # Examples 1k-10k for testing
```

The split is applied virtually (no data copying), and all splits share the same index cache.

### Key Features

- **Instant Loading**: < 1 second after initial indexing
- **Efficient Checkpointing**: Position-based resumption (file_idx, example_idx)
- **Natural Sharding**: Each Arrow file = 1 shard for distributed training
- **Split Notation**: Virtual splits without data duplication
- **Memory Efficient**: Memory-maps Arrow files (no RAM overhead)
- **HuggingFace Compatible**: Works with existing HF datasets

## Batched Map Operations for Efficient Processing

The packed configurations (`*-packed.yaml`) demonstrate batched map operations, critical for efficient tokenization and sequence packing.

### The Problem with Single-Example Processing

Standard map operations process one example at a time:
```python
# Inefficient: tokenizes one document at a time
def tokenize_one(example):
    return tokenizer(example["text"])
```

This is slow for two reasons:
1. **Tokenizer Overhead**: Modern tokenizers are optimized for batches
2. **No Cross-Document Packing**: Can't pack multiple documents into fixed-length sequences

### Batched Processing Solution

With `batched=True`, the map function receives batches in dictionary format:

```python
# From packed.yaml - efficient batch processing
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 512
    overflow: True
    packed: True          # Pack multiple docs into sequences
    shuffle_output: True  # Randomize packed sequence order
    stride: 0
    packing_strategy: "best_fit"  # Optimize space utilization
    min_len: 16
    add_bos: True
    add_eos: True

.define: &map_kwargs !dict
    batch_size: 4096  # Process 4096 examples at once
```

**How It Works:**
1. Loader collects 4096 examples into a batch
2. Batch is converted to dict format: `{"text": ["doc1", "doc2", ...]}`
3. `block_tokenize_fn` tokenizes all documents in batch
4. Packing algorithm combines documents into optimal fixed-length sequences
5. Result can have different number of examples (N→M mapping)

### Sequence Packing Strategies

The `block_tokenize_fn` supports multiple packing strategies:

**Greedy** (Default):
- Sequential processing, fills sequences in order
- Fast and simple
- Already achieves 95%+ utilization with overflow=True

**Best-Fit**:
- Sorts documents by length, packs optimally
- Can reduce output blocks by 50% with overflow=False
- Trade-off: Non-random sequence order (use shuffle_output=True)

**First-Fit**:
- Middle ground between greedy and best-fit
- Good utilization with better performance than best-fit

### N→M Mapping

Batched operations can return different numbers of examples:

**Input Batch** (N=1000 documents):
```python
{"text": ["doc1", "doc2", ..., "doc1000"]}
```

**Output Batch** (M=750 packed sequences):
```python
{
    "input_ids": [[tokens...], [tokens...], ..., [tokens...]],  # 750 sequences
    "document_starts": [[0, 512, 1024], [0, 768], ...]  # Document boundaries
}
```

The packing eliminates wasted space, resulting in fewer but fuller sequences.

## Interleaving Multiple Datasets with `interleave_datasets`

Pre-training small language models typically requires combining multiple diverse datasets. The `interleave_datasets` function enables efficient multi-dataset training with checkpoint support.

### The Problem with Standard Interleaving

HuggingFace's `datasets.interleave_datasets()` has critical limitations:

1. **Type Checking**: Only works with HuggingFace Dataset/IterableDataset types
2. **Lost Checkpoint Protocol**: Converts to standard HF iterable, losing efficient position-based resumption
3. **No Custom Datasets**: Can't interleave custom dataset implementations

### Protocol-Based Interleaving Solution

Forgather's `interleave_datasets` uses duck typing instead of type checking, working with any iterable dataset:

```yaml
# From interleaved.yaml
train_dataset: !singleton:forgather.ml.datasets:interleave_datasets
    probabilities: [ 1, 1 ]  # Equal sampling from each dataset
    seed: 42                 # Reproducible sampling
    stopping_strategy: "first_exhausted"
    datasets:
        - !call:getitem [ *fineweb, 'train_dataset' ]
        - !call:getitem [ *cosmopedia, 'train_dataset' ]
```

### Interleaving Strategies

**Round-Robin** (probabilities=None):
- Cycles through datasets sequentially
- Dataset 1 → Dataset 2 → Dataset 1 → Dataset 2 → ...
- Predictable and simple

**Probabilistic Sampling** (probabilities=[w1, w2, ...]):
- Samples from datasets according to weights
- Weights don't need to sum to 1 (they're normalized)
- Example: `[1, 1]` = 50/50, `[7, 3]` = 70/30
- Uses seed for reproducibility

**Dynamic Probabilities** (probabilities=callable):
- Accepts callable function for dynamic weight computation
- Function called each iteration with current state
- Enables advanced patterns like curriculum learning and balanced exhaustion
- Signature: `(step, datasets, examples_per_dataset, exhausted) -> List[float]`

### Dynamic Probability Functions

The `probabilities` parameter can accept a callable function for computing weights dynamically based on training progress.

**Balanced Exhaustion with `balance_remaining_examples`:**

The built-in `balance_remaining_examples` function weights datasets by their estimated remaining examples, encouraging all datasets to finish at approximately the same time:

```python
from forgather.ml.datasets import interleave_datasets, balance_remaining_examples

# Datasets will be sampled proportionally to remaining examples
interleaved = interleave_datasets(
    [ds1, ds2, ds3],
    probabilities=balance_remaining_examples,
    seed=42,
    stopping_strategy="all_exhausted"
)
```

**How it works:**
- Computes remaining examples: `total_length - examples_consumed`
- Assigns weight proportional to remaining count
- Dataset with more remaining gets sampled more frequently
- All datasets finish at approximately the same time

**Curriculum Learning with Custom Functions:**

Create custom probability functions for curriculum learning, where the data distribution changes over training:

```python
def curriculum_probabilities(step, datasets, examples_per_dataset, exhausted):
    """Gradually transition from easy (ds0) to hard (ds1) examples."""
    if step < 10000:
        # First 10k steps: 80% easy, 20% hard
        return [0.8, 0.2]
    elif step < 50000:
        # Transition period: gradually shift weights
        progress = (step - 10000) / 40000.0  # 0 to 1
        easy_weight = 0.8 - 0.6 * progress   # 0.8 → 0.2
        hard_weight = 0.2 + 0.6 * progress   # 0.2 → 0.8
        return [easy_weight, hard_weight]
    else:
        # After 50k steps: 20% easy, 80% hard
        return [0.2, 0.8]

interleaved = interleave_datasets(
    [easy_dataset, hard_dataset],
    probabilities=curriculum_probabilities,
    seed=42
)
```

**Function Parameters:**
- `step` (int): Current iteration count (starts at 0)
- `datasets` (List): List of child datasets (for checking lengths, etc.)
- `examples_per_dataset` (List[int]): Number of examples consumed from each dataset
- `exhausted` (List[bool]): Whether each dataset is exhausted

**Function Returns:**
- `List[float]`: Weights for each dataset (will be normalized automatically)

### Stopping Strategies

**first_exhausted** (Default - Undersampling):
- Stops when first dataset runs out
- Total examples: `min(lengths) × num_datasets` (for round-robin)
- Use for balanced sampling from unbalanced datasets

**all_exhausted** (Oversampling):
- Continues until all datasets consumed
- Total examples: `sum(lengths)`
- Longer datasets contribute more examples
- Use when you want all data from all sources

### Nested Checkpoint State

The interleaved dataset preserves checkpoint state for all child datasets:

```python
# Save state mid-training
state = interleaved_dataset.state_dict()
# Returns:
# {
#     "current_dataset_index": 0,
#     "current_example_count": 12500,
#     "datasets_exhausted": [False, False],
#     "child_states": [
#         {"current_file_index": 5, "current_example_index": 234, ...},
#         {"current_file_index": 3, "current_example_index": 567, ...}
#     ]
# }

# Restore state instantly (< 1 second)
interleaved_dataset.load_state_dict(state)
```

Each child dataset maintains its own position-based checkpoint, enabling instant resumption of the entire interleaved pipeline.

### Combining Interleaving with Packing

The `interleaved-packed.yaml` configuration demonstrates the most powerful pattern: combining multiple large datasets with efficient packing:

```yaml
# From interleaved-packed.yaml
[fineweb]
    == super()
    config_template: "smollm-corpus/fineweb-edu-packed.yaml"  # Already packed

[cosmopedia]
    == super()
    config_template: "smollm-corpus/cosmopedia-v2-packed.yaml"  # Already packed

# Interleave the pre-packed datasets
train_dataset: !singleton:forgather.ml.datasets:interleave_datasets
    probabilities: [ 1, 1 ]
    datasets:
        - !call:getitem [ *fineweb, 'train_dataset' ]
        - !call:getitem [ *cosmopedia, 'train_dataset' ]
```

**Benefits:**
1. Each dataset is packed independently (optimal packing per source)
2. Interleaving mixes the packed sequences
3. All datasets load instantly (< 1 second)
4. Full checkpoint protocol preserved across entire pipeline
5. Distributed training friendly (natural sharding)

**Performance:**
- Initial load: 10-20 min (one-time indexing)
- Subsequent loads: < 1 second
- Checkpoint resumption: < 1 second (vs hours with standard approach)
- GPU utilization: 95%+ (efficient packing eliminates wasted padding)

## Testing

Note: It is assumed that the tokenizers as the specified paths have been built. If not, build them from `examples/tokenizers` or use a path to another tokenizer.

The first time you load the dataset, the load time will be much longer. It must be downloaded, built, and indexed. After that, loading is nearly instantaneous.

**Test Examples**

```bash
# Load dataset and dump first three examples from target split, without tokenizing
# Other splits include: eval_dataset_split and test_dataset_split
forgather -t smollm-corpus/fineweb-edu-dedup.yaml dataset --target train_dataset_split -n 3
forgather -t smollm-corpus/cosmopedia-v2.yaml dataset --target train_dataset_split -n 3

# Load and tokenizer first three examples from train split
forgather -t smollm-corpus/fineweb-edu-dedup.yaml dataset --target train_dataset -n 3 -s -T ../../../tokenizers/wikitext_32k/
forgather -t smollm-corpus/cosmopedia-v2.yaml dataset --target train_dataset -n 3 -s -T ../../../tokenizers/wikitext_32k/

# Load packed dataset with token block size of 2048 and show first packed example
forgather -t smollm-corpus/fineweb-edu-packed.yaml dataset --target train_dataset  -n 1 -s -T ../../../tokenizers/wikitext_32k/ --max-length 2048

# Randomly interleave all Small LM datasets
forgather -t smollm-corpus/interleaved.yaml dataset --target train_dataset  -n 8 -s -T ../../../tokenizers/wikitext_32k/

# Randomly interleave all packed Small LM datasets
forgather -t smollm-corpus/interleaved-packed.yaml dataset --target train_dataset  -n 4 -s -T ../../../tokenizers/wikitext_32k/ --max-length 2048
```

## Usage

To include the "interleaved-packed" dataset in a training project...

```yaml
[datasets_preprocessor_args]
# Overrides for forgather.ml.datasets:block_tokenize_fn
.define: &datasets_preprocessor_args !dict
    # This sets the packed token-block size
    max_length: 4096

[datasets_definition]
.define: &dataset_dict !call:forgather:from_project
    project_dir: "{{ abspath(joinpath(ns.forgather_dir, 'examples/datasets/HuggingFaceTB/')) }}"
    config_template: "smollm-corpus/interleaved-packed.yaml"
    targets: [ "train_dataset", "eval_dataset" ]
    preprocess_args: *datasets_preprocessor_args
    tokenizer: *tokenizer

train_dataset: &train_dataset !call:getitem [ *dataset_dict, 'train_dataset' ]
eval_dataset: &eval_dataset !call:getitem [ *dataset_dict, 'eval_dataset' ]

[datacollator]
data_collator: &data_collator !singleton:forgather.ml.data_collator:DataCollatorForCausalLM@DataCollatorForCausalLM
    tokenizer: *tokenizer
    return_tensors: pt
```

The following Python code loads the same dataset as above, then uses it to generate batches with a torchdata StatefulDataLoader.

```python
from transformers import AutoTokenizer
from torchdata.stateful_dataloader import StatefulDataLoader
from forgather import from_project
from forgather.ml.data_collator import DataCollatorForCausalLM

tokenizer = AutoTokenizer.from_pretrained("./tokenizers/wikitext_32k/")

dataset_dict = from_project(
    project_dir="examples/datasets/HuggingFaceTB/",
    config_template="smollm-corpus/interleaved-packed.yaml",
    targets=[ "train_dataset", "eval_dataset" ],
    preprocess_args=dict(
        max_length=4096,
    ),
    tokenizer=tokenizer,
)

dataloader = StatefulDataLoader(
    dataset_dict["train_dataset"],
    batch_size=2,
    collate_fn=DataCollatorForCausalLM(
        tokenizer=tokenizer,
        return_tensors="pt",
    ),
    drop_last=True,
    num_workers=1,
    pin_memory=True,
)

for i, batch in zip(range(3), dataloader):
    decoded = tokenizer.batch_decode(batch["input_ids"])
    print(f"{i:-^20}")
    print(decoded)
```