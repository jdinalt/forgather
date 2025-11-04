# Sequence Packing Guide

## Overview

Sequence packing is a technique for combining multiple training examples into single sequences to maximize GPU utilization. Instead of padding short sequences to `max_length`, multiple documents are concatenated together, significantly improving training efficiency.

The `block_tokenize_fn` in Forgather supports three packing strategies with different trade-offs between performance, efficiency, and complexity.

## Why Sequence Packing?

### Without Packing
```
Sequence 1: [doc1_tokens..................] (30% utilized)
Sequence 2: [doc2_tokens..........] (20% utilized)
Sequence 3: [doc3_tokens.....] (15% utilized)
```
Most of the sequence is padding, wasting GPU compute.

### With Packing
```
Sequence 1: [doc1_tokens|doc2_tokens|doc3_tokens...] (95% utilized)
```
Multiple documents combined into one sequence, maximizing GPU usage.

### Benefits
- **Reduced training time**: 2-3x speedup possible with efficient packing
- **Better GPU utilization**: Less wasted compute on padding tokens
- **Cost savings**: Faster training = lower cloud compute costs

## Operational Theory

### How Packing Works

1. **Tokenization**: Documents are tokenized with BOS/EOS tokens marking boundaries
2. **Packing**: Multiple tokenized documents are combined into sequences up to `max_length`
3. **Position IDs**: The data collator generates position IDs, resetting at each BOS token
4. **Attention Masking**: Flex attention prevents tokens from attending across document boundaries

### Document Boundaries

Each document in a packed sequence has:
- **BOS token** (Beginning of Sequence) at the start
- **EOS token** (End of Sequence) at the end

Example packed sequence:
```
[BOS, doc1_token1, doc1_token2, ..., EOS, BOS, doc2_token1, doc2_token2, ..., EOS, ...]
```

The attention mask ensures tokens in `doc1` cannot attend to tokens in `doc2`, maintaining proper causal masking.

### Batch Processing

**Important**: HuggingFace `dataset.map()` processes data in batches, and packing optimization occurs **within each batch only**. State cannot be maintained across batches.

**Implication**: Use large batch sizes (e.g., `batch_size=1000`) to minimize data loss at batch boundaries.

```python
dataset = dataset.map(
    block_tokenize_fn,
    batched=True,
    batch_size=1000,  # Large batch size critical for efficiency
    remove_columns=["text"],
)
```

## Packing Strategies

### Greedy (Default)

**Algorithm**: Sequential processing - documents are packed in the order they appear.

**How it works**:
1. Take first document, add to current sequence
2. If document fits, add it; if not, start new sequence
3. Continue with next document

**Characteristics**:
- Simple and fast
- Preserves document order
- Achieves 95%+ utilization in most cases
- No sorting overhead

**Best for**:
- `overflow=True` with long documents
- When document order matters

**Example**:
```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 4096
    overflow: True
    packed: True
    packing_strategy: "greedy"  # Default
    min_len: 16
    add_bos: True
    add_eos: True
```

### Best Fit

**Algorithm**: Best-Fit Decreasing bin packing - documents sorted by length, each placed in the fullest bin that fits.

**How it works**:
1. Sort documents by length (longest first)
2. For each document, find the bin (output sequence) with the **least remaining space** that can still fit it
3. If no bin fits, create a new bin

**Characteristics**:
- Optimal space utilization
- Sorts documents by length (breaks original order)
- Can reduce output blocks by 50% with `overflow=False`
- Slightly slower due to sorting and bin selection

**Best for**:
- `overflow=False` (truncate mode) - **best use case**
- Maximizing space efficiency
- Datasets with high length variability

**Trade-off**: Produces non-random sequence ordering (largest documents packed first). **Solution**: Use `shuffle_output=True` to randomize.

**Example**:
```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 4096
    overflow: False  # Truncate mode - best_fit shines here
    packed: True
    packing_strategy: "best_fit"
    shuffle_output: True  # Randomize to prevent bias
    seed: 42  # Optional: reproducibility during development
    min_len: 16
    add_bos: True
    add_eos: True
```

### First Fit

**Algorithm**: First-Fit Decreasing bin packing - documents sorted by length, each placed in the first bin with space.

**How it works**:
1. Sort documents by length (longest first)
2. For each document, find the **first bin** that has space
3. If no bin fits, create a new bin

**Characteristics**:
- Good space utilization (close to best_fit)
- Faster than best_fit (stops at first fit, doesn't search all bins)
- Middle ground between greedy and best_fit
- Also breaks document order

**Best for**:
- When you want better packing than greedy but faster than best_fit
- Large datasets where best_fit's search overhead matters

**Example**:
```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 4096
    overflow: True
    packed: True
    packing_strategy: "first_fit"
    shuffle_output: True
    min_len: 16
    add_bos: True
    add_eos: True
```

## Strategy Comparison

| Strategy   | Speed    | Utilization | Order Preserved | Best Use Case                    |
|------------|----------|-------------|-----------------|----------------------------------|
| greedy     | Fastest  | 95%+        | Yes             | overflow=True, general use       |
| first_fit  | Fast     | 96-98%      | No (sorted)     | Balance speed/efficiency         |
| best_fit   | Moderate | 98%+        | No (sorted)     | overflow=False (50% improvement) |

## Overflow Modes

### overflow=True (Split Long Documents)

Documents longer than `max_length` are split into multiple chunks with optional stride (overlap).

**Use when**: Documents may exceed `max_length` and you want to preserve all content.

**Greedy behavior**: Splits documents sequentially, may fragment across sequences
**Optimized behavior**: Splits documents intelligently, tries to fit chunks in existing bins

**Example**:
```yaml
overflow: True
stride: 128  # 128-token overlap between chunks
```

### overflow=False (Truncate Long Documents)

Documents longer than `max_length` are truncated to fit.

**Use when**: You can afford to lose content from very long documents (e.g., pretraining on web text).

**Greedy behavior**: Truncates documents, often leaves sequences with low utilization
**Optimized behavior**: Packs truncated documents efficiently, **50% fewer output blocks**

**Example**:
```yaml
overflow: False  # Truncate
packing_strategy: "best_fit"  # Get maximum benefit
```

## Shuffle Output

**Problem**: Optimized strategies (best_fit, first_fit) sort documents by length, causing output sequences to be ordered by "fullness". This can bias training, with the model seeing many full sequences first, then partial sequences.

**Solution**: `shuffle_output=True` randomizes the order of output sequences while preserving packing efficiency.

**Recommendations**:
- Always use `shuffle_output=True` with optimized strategies during training
- Use fixed `seed` during development for reproducibility
- Remove seed (or set to `None`) in production for randomness

**Example**:
```yaml
packing_strategy: "best_fit"
shuffle_output: True
seed: 42  # Optional: reproducibility
```

## Use Case Strategies

### Pretraining on Web Text

**Scenario**: Large corpus with highly variable document lengths (100-10000 tokens).

**Recommendation**:
```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 4096
    overflow: False  # Truncate very long documents
    packed: True
    packing_strategy: "best_fit"  # Maximum efficiency
    shuffle_output: True  # Prevent training bias
    min_len: 64
    add_bos: True
    add_eos: True
```

**Why**:
- `overflow=False`: Web text often has very long documents (articles, papers) - truncation acceptable
- `best_fit`: Reduces output blocks by 50%, saving compute
- `shuffle_output=True`: Prevents bias from length-sorted sequences
- Large `max_length`: Maximize context window usage

### Finetuning on Structured Data

**Scenario**: Dataset with consistent document lengths (e.g., chat conversations, Q&A pairs).

**Recommendation**:
```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 2048
    overflow: True  # Don't lose any content
    packed: True
    packing_strategy: "greedy"  # Already efficient with consistent lengths
    min_len: 32
    add_bos: True
    add_eos: True
```

**Why**:
- `overflow=True`: Preserve all content (important for structured data)
- `greedy`: Fast and efficient when lengths are consistent
- No shuffle needed (greedy preserves order, no bias)

### Mixed-Length Instruction Following

**Scenario**: Instruction-following dataset with variable lengths (short queries + long responses).

**Recommendation**:
```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 4096
    overflow: True
    packed: True
    packing_strategy: "first_fit"  # Good balance
    shuffle_output: True
    stride: 64  # Small overlap for long responses
    min_len: 32
    add_bos: True
    add_eos: True
```

**Why**:
- `overflow=True`: Keep full instructions and responses
- `first_fit`: Better packing than greedy, faster than best_fit
- `shuffle_output=True`: Randomize length distribution
- `stride=64`: Maintain context when splitting long responses

### Continued Pretraining with Long Context

**Scenario**: Extending pretrained model to longer context (32k-128k tokens).

**Recommendation**:
```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 32768  # Very long sequences
    overflow: True
    packed: True
    packing_strategy: "greedy"  # Sufficient at this scale
    min_len: 8192
    add_bos: True
    add_eos: True
```

**Why**:
- `max_length` very large: Most documents fit without splitting
- `greedy`: At this scale, documents naturally pack well
- High `min_len`: Ensure sequences are substantial
- No optimization needed (already near 100% utilization)

## Configuration Reference

### Full Dataset Pipeline

```python
from datasets import load_dataset
from transformers import AutoTokenizer
from forgather.ml.datasets import block_tokenize_fn
from forgather.ml.data_collator import DataCollatorForCausalLM

# Load dataset
dataset = load_dataset("your_dataset")

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("your_model")

# Apply packing
packed_dataset = dataset.map(
    lambda features: block_tokenize_fn(
        features,
        tokenizer=tokenizer,
        feature="text",
        max_length=4096,
        overflow=False,
        packed=True,
        packing_strategy="best_fit",
        shuffle_output=True,
        seed=42,  # Remove for production randomness
        min_len=64,
        add_bos=True,
        add_eos=True,
    ),
    batched=True,
    batch_size=1000,  # Large batch size critical!
    remove_columns=["text"],
    num_proc=4,  # Parallel processing
)

# Data collator for packed sequences
data_collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    packed_sequences=True,  # Enable position ID generation
    padding="longest",
    max_length=4096,
)

# Use in DataLoader
from torch.utils.data import DataLoader

dataloader = DataLoader(
    packed_dataset,
    batch_size=8,  # Training batch size
    collate_fn=data_collator,
)
```

### Forgather YAML Configuration

```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    tokenizer: *tokenizer
    feature: "text"
    max_length: 4096
    overflow: False
    packed: True
    packing_strategy: "best_fit"
    shuffle_output: True
    seed: !var { name: "seed", default: 42 }
    stride: 0
    min_len: 64
    max_len: null
    add_bos: True
    add_eos: True

[train_dataset]
    == super()
    map_fn: *map_function
    batched: True
    batch_size: 1000
    num_proc: 4

[data_collator]
.define: &data_collator !singleton:forgather.ml.data_collator:DataCollatorForCausalLM
    tokenizer: *tokenizer
    packed_sequences: True
    padding: "longest"
    max_length: 4096
```

## Performance Tips

### 1. Use Large Batch Sizes

```python
dataset.map(fn, batched=True, batch_size=1000)  # Good
dataset.map(fn, batched=True, batch_size=10)    # Bad (loses data at boundaries)
```

### 2. Match Strategy to Use Case

- `overflow=False` → use `best_fit` (50% improvement)
- `overflow=True` with short docs → use `greedy` (already efficient)
- `overflow=True` with mixed lengths → use `first_fit` (balanced)

### 3. Always Shuffle with Optimized Strategies

```yaml
packing_strategy: "best_fit"
shuffle_output: True  # Prevent training bias
```

### 4. Tune min_len

Too low: Wastes compute on tiny sequences
Too high: Discards useful data

Recommended: `min_len = max_length / 64` to `max_length / 16`

```yaml
max_length: 4096
min_len: 64  # 4096/64 = reasonable minimum
```

### 5. Monitor Packing Efficiency

After packing, check utilization:

```python
lengths = [len(seq) for seq in dataset["input_ids"]]
avg_length = sum(lengths) / len(lengths)
utilization = avg_length / max_length * 100
print(f"Utilization: {utilization:.1f}%")
```

Target: >90% utilization for efficient training.

## Troubleshooting

### Low Utilization (<80%)

**Causes**:
- Small batch size (data lost at boundaries)
- Wrong strategy for use case
- Documents much shorter than max_length

**Solutions**:
- Increase `batch_size` to 1000+
- Try `best_fit` strategy
- Reduce `max_length` to match document lengths better

### Training Bias (Model learns sequence length patterns)

**Cause**: Using `best_fit` or `first_fit` without shuffling

**Solution**: Set `shuffle_output=True`

### Out of Memory

**Cause**: Very long sequences with packing

**Solutions**:
- Reduce training batch size (not map batch_size)
- Use gradient accumulation
- Enable gradient checkpointing
- Reduce `max_length`

### Slow Packing

**Cause**: `best_fit` on very large batches

**Solutions**:
- Try `first_fit` (faster with similar results)
- Use `greedy` if utilization already >90%
- Reduce `num_proc` in map (reduces overhead)

## Advanced Topics

### Custom Packing Logic

If you need custom packing behavior beyond the three strategies, you can extend the bin-packing system:

```python
from forgather.ml.datasets.block_tokenizer import (
    Document, Bin, pack_sequences_optimized
)

# Create your own packing function
def my_custom_packer(documents, max_length, ...):
    # Your custom logic here
    pass
```

### Integration with Flex Attention

Packed sequences require special attention masking. Forgather's `DataCollatorForCausalLM` automatically generates:

1. **Position IDs**: Reset at each EOS token
2. **Sequence IDs**: Identify document boundaries
3. **Attention masks**: Prevent cross-document attention

Example with custom attention:

```python
from forgather.ml.data_collator import get_pos_ids_for_packed_sequence

# In your collator
if self.packed_sequences:
    position_ids = get_pos_ids_for_packed_sequence(
        input_ids,
        self.tokenizer.eos_token_id
    )
```

### Monitoring Training

Track these metrics to ensure packing is helping:

- **Tokens per second**: Should increase with packing
- **GPU utilization**: Should be higher with packing
- **Loss convergence**: Should be similar to unpacked training

If packing hurts performance, check for:
- Incorrect attention masking
- Position ID errors
- Batch size too small

## References

- [Hugging Face Blog: LLM Sequence Packing](https://huggingface.co/blog/sirluk/llm-sequence-packing)
- Forgather source: `src/forgather/ml/datasets/block_tokenizer.py`
- Data collator: `src/forgather/ml/data_collator.py`
- Tests: `tests/test_packing_comparison.py`
