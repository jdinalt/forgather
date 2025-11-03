# Sequence Packing Quick Reference

## TL;DR - Which Strategy?

| Your Scenario | Use This |
|--------------|----------|
| Truncating long documents (`overflow=False`) | `packing_strategy="best_fit"` + `shuffle_output=True` |
| Variable-length documents, need all content | `packing_strategy="greedy"` or `"first_fit"` |
| Consistent document lengths | `packing_strategy="greedy"` |
| Very long sequences (>16k tokens) | `packing_strategy="greedy"` |
| Want speed > efficiency | `packing_strategy="greedy"` |
| Want efficiency > speed | `packing_strategy="best_fit"` + `shuffle_output=True` |

## Quick Start

### Pretraining (Most Common)

```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 4096
    overflow: False
    packed: True
    packing_strategy: "best_fit"
    shuffle_output: True
    min_len: 64
    add_bos: True
    add_eos: True
```

### Finetuning

```yaml
[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    max_length: 2048
    overflow: True
    packed: True
    packing_strategy: "greedy"
    min_len: 32
    add_bos: True
    add_eos: True
```

## Parameter Cheat Sheet

| Parameter | Values | Default | When to Change |
|-----------|--------|---------|----------------|
| `max_length` | 512-131072 | 512 | Match your model's context window |
| `overflow` | True/False | True | Set False to truncate long docs (pretraining) |
| `packed` | True/False | False | Set True to enable packing |
| `packing_strategy` | "greedy"/"best_fit"/"first_fit" | "greedy" | Use "best_fit" with overflow=False |
| `shuffle_output` | True/False | False | Set True with best_fit/first_fit |
| `seed` | int or None | None | Set during dev for reproducibility |
| `min_len` | 1-max_length | 1 | Set to ~max_length/64 to filter short seqs |
| `add_bos` | True/False | True | Usually keep True |
| `add_eos` | True/False | True | Usually keep True |
| `stride` | 0-512 | 0 | Set 64-256 for document overlap |

## Performance Impact

### Best Fit vs Greedy (overflow=False, 200 docs, max_length=128)

| Metric | Greedy | Best Fit | Improvement |
|--------|--------|----------|-------------|
| Output blocks | 200 | 100 | **50% fewer** |
| Avg utilization | 49.8% | 98.4% | **+48.6 pp** |
| Speed | Fastest | Fast | Minimal difference |

### Best Fit vs Greedy (overflow=True, 500 docs, max_length=4096)

| Metric | Greedy | Best Fit | Improvement |
|--------|--------|----------|-------------|
| Output blocks | 12 | 12 | None |
| Avg utilization | 95.4% | 95.4% | None |
| Speed | Fastest | Fast | Minimal difference |

**Takeaway**: Use best_fit with `overflow=False` for maximum benefit.

## Common Mistakes

### ❌ Don't Do This

```yaml
# Small batch size - loses data at boundaries
map_fn: *tokenizer
batched: True
batch_size: 10  # BAD!

# Using best_fit without shuffle - causes training bias
packing_strategy: "best_fit"
shuffle_output: False  # BAD!

# Optimized strategy without packing enabled
packed: False
packing_strategy: "best_fit"  # Ignored! Needs packed=True
```

### ✅ Do This

```yaml
# Large batch size - minimal data loss
map_fn: *tokenizer
batched: True
batch_size: 1000  # GOOD!

# Shuffling with optimized strategies
packing_strategy: "best_fit"
shuffle_output: True  # GOOD!

# Enable packing with strategy
packed: True
packing_strategy: "best_fit"  # Works!
```

## Decision Tree

```
Start: Do you need sequence packing?
│
├─ No  → Use standard tokenization
│
└─ Yes → Are you truncating long documents (overflow=False)?
    │
    ├─ Yes → Use packing_strategy="best_fit", shuffle_output=True
    │        (50% fewer blocks, 98% utilization)
    │
    └─ No (overflow=True) → Are documents highly variable in length?
        │
        ├─ Yes → Use packing_strategy="first_fit", shuffle_output=True
        │        (Good balance of speed and efficiency)
        │
        └─ No → Use packing_strategy="greedy"
                 (Fast, simple, already efficient)
```

## Testing Your Config

After configuring packing, verify it's working:

```python
# Check utilization
lengths = [len(seq) for seq in dataset["input_ids"]]
avg = sum(lengths) / len(lengths)
utilization = (avg / max_length) * 100
print(f"Utilization: {utilization:.1f}%")

# Target: >90% for good efficiency

# Check sequence count
print(f"Output sequences: {len(dataset)}")
print(f"Input documents: {original_dataset_size}")
print(f"Packing ratio: {original_dataset_size / len(dataset):.1f}x")

# Good packing ratio: 2-10x depending on document lengths
```

## Complete Example

```yaml
# Pretraining configuration with optimized packing
[tokenizer]
.define: &tokenizer !singleton:transformers:AutoTokenizer.from_pretrained
    pretrained_model_name_or_path: "meta-llama/Llama-3.2-1B"

[map_function]
.define: &map_function !partial:forgather.ml.datasets:block_tokenize_fn
    tokenizer: *tokenizer
    feature: "text"
    max_length: 4096
    overflow: False           # Truncate long docs
    packed: True              # Enable packing
    packing_strategy: "best_fit"  # Optimal for overflow=False
    shuffle_output: True      # Randomize output order
    seed: null                # Random (use 42 for reproducible dev)
    stride: 0                 # No overlap
    min_len: 64               # Filter very short sequences
    add_bos: True             # Add BOS tokens
    add_eos: True             # Add EOS tokens

[train_dataset]
    == super()
    map_fn: *map_function
    batched: True
    batch_size: 1000          # Large batch critical!
    num_proc: 4               # Parallel processing
    remove_columns: ["text"]

[data_collator]
.define: &data_collator !singleton:forgather.ml.data_collator:DataCollatorForCausalLM
    tokenizer: *tokenizer
    packed_sequences: True    # Generate position IDs
    padding: "longest"
    max_length: 4096
```

## One-Liners

```bash
# Test packing efficiency
python -c "from datasets import load_dataset; ds=load_dataset('your_dataset'); print(f'Docs: {len(ds)}, Avg length: {sum(len(x) for x in ds[\"text\"])/len(ds)}')"

# Quick benchmark
python tests/test_packing_comparison.py

# Check packed dataset
python -c "from datasets import load_dataset; ds=load_dataset('path/to/packed'); lengths=[len(x) for x in ds['input_ids']]; print(f'Utilization: {sum(lengths)/len(lengths)/4096*100:.1f}%')"
```

## Further Reading

- Full guide: `docs/datasets/sequence-packing.md`
- Source code: `src/forgather/ml/datasets/block_tokenizer.py`
- Tests: `tests/test_packing_comparison.py`
- HF Blog: https://huggingface.co/blog/sirluk/llm-sequence-packing
