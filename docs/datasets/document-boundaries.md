# Explicit Document Boundary Tracking

## Overview

Forgather's document packing system now supports **explicit document boundary tracking**, which allows packed sequences to work with any tokenizer configuration, including models that lack BOS/EOS tokens (e.g., Qwen3).

## The Problem

### Traditional Approach (Token-Based)

Previously, the system used special tokens (BOS/EOS) to mark document boundaries:

```python
# Document packing with BOS/EOS markers
[BOS, doc1_tok1, doc1_tok2, EOS, BOS, doc2_tok1, doc2_tok2, EOS, ...]
```

The data collator would detect these special tokens and reset position IDs at each boundary, ensuring correct attention masking.

### Issues with Token-Based Approach

1. **Missing Tokens**: Models like Qwen3 don't define BOS tokens
2. **Token Aliasing**: Some tokenizers alias PAD==EOS (e.g., Qwen3's `<|endoftext|>`), causing every padding token to be treated as a document boundary
3. **Semantic Mismatch**: Tokens like `<|im_end|>` mark message boundaries, not document boundaries
4. **Broken Conversations**: In chat datasets, each message would be treated as a separate document, preventing cross-message attention

## The Solution

### Explicit Boundary Metadata

The new system stores document start positions as metadata alongside input IDs:

```python
{
    "input_ids": [tok1, tok2, tok3, tok4, tok5, tok6, tok7, tok8],
    "document_starts": [0, 4]  # Doc1 starts at 0, Doc2 starts at 4
}
```

This approach:
- Works with ANY tokenizer (no special tokens required)
- Is space-efficient (stores 2-5 integers vs full position_ids array)
- Has clear semantics (explicit boundaries, not inferred)
- Maintains backward compatibility (falls back to token-based detection)

## Usage

### Basic Example

```python
from transformers import AutoTokenizer
from forgather.ml.datasets.block_tokenizer import block_tokenize_fn
from forgather.ml.data_collator import DataCollatorForCausalLM

# Load Qwen3 tokenizer (no BOS token)
tokenizer = AutoTokenizer.from_pretrained("qwen3-1.7b-base")

# Pack documents
features = {
    "text": [
        "Document 1 text...",
        "Document 2 text...",
        "Document 3 text...",
    ]
}

result = block_tokenize_fn(
    features=features,
    tokenizer=tokenizer,
    feature="text",
    max_length=512,
    packed=True,
    packing_strategy="greedy",  # or "best_fit", "first_fit"
    add_eos=True,
)

# Result contains both input_ids and document_starts
print(result["input_ids"])        # Packed token sequences
print(result["document_starts"])  # Document boundary positions
```

### Training with Packed Sequences

```python
from torch.utils.data import DataLoader
from forgather.ml.data_collator import DataCollatorForCausalLM

# Create collator - packed sequences are auto-detected from document_starts!
collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    # packed_sequences parameter is optional - auto-detects from document_starts field
    padding="longest",
    return_tensors="pt",
)

# Create DataLoader
dataloader = DataLoader(
    dataset,
    batch_size=4,
    collate_fn=collator,
)

# Training loop
for batch in dataloader:
    # batch contains:
    # - input_ids: Token IDs
    # - labels: Labels for causal LM
    # - position_ids: Position IDs that reset at document boundaries

    outputs = model(
        input_ids=batch["input_ids"],
        position_ids=batch["position_ids"],  # Critical for proper RoPE embeddings
        labels=batch["labels"],
    )
```

## Auto-Detection

The `DataCollatorForCausalLM` automatically detects when to generate position IDs based on the presence of the `document_starts` field:

**Behavior:**
- `packed_sequences=None` (default): Auto-detects from `document_starts` field
- `packed_sequences=True`: Always generates position IDs (uses token-based detection if `document_starts` missing)
- `packed_sequences=False`: Never generates position IDs (explicit disable)

**Examples:**

```python
# Auto-detection (recommended)
collator = DataCollatorForCausalLM(tokenizer=tokenizer)
# → Generates position_ids if document_starts present, otherwise doesn't

# Explicit enable
collator = DataCollatorForCausalLM(tokenizer=tokenizer, packed_sequences=True)
# → Always generates position_ids (falls back to token-based if needed)

# Explicit disable
collator = DataCollatorForCausalLM(tokenizer=tokenizer, packed_sequences=False)
# → Never generates position_ids, even if document_starts present
```

This means **you don't need to configure anything** - the system automatically works correctly whether you're using packed sequences or not!

## How It Works

### 1. Tokenization Phase

The `block_tokenize_fn` tracks where each document starts as it packs them:

```python
# Internal tracking during packing
bin.documents = [
    (doc_idx=0, tokens_added=50),   # First doc: 50 tokens
    (doc_idx=1, tokens_added=30),   # Second doc: 30 tokens
    (doc_idx=2, tokens_added=20),   # Third doc: 20 tokens
]

# Generates document_starts
document_starts = [0, 50, 80]  # Start positions
```

### 2. Collation Phase

The `DataCollatorForCausalLM` converts boundaries to position IDs:

```python
# Input
input_ids = [tok0, ..., tok49, tok50, ..., tok79, tok80, ..., tok99]
document_starts = [0, 50, 80]

# Generated position IDs
position_ids = [0, 1, ..., 49, 0, 1, ..., 29, 0, 1, ..., 19]
                 ↑              ↑              ↑
            Reset at each document boundary
```

### 3. Model Forward Pass

Position IDs ensure correct RoPE embeddings and attention masking:

```python
# RoPE uses position IDs to encode positional information
q_rotated = apply_rope(queries, position_ids)
k_rotated = apply_rope(keys, position_ids)

# Attention mask prevents cross-document attention
# (tokens at position 0 in different documents can't attend to each other)
attention_scores = q_rotated @ k_rotated.T  # With causal mask based on position_ids
```

## Configuration Examples

### Forgather Project Configuration

```yaml
-- extends "types/training_script/causal_lm/causal_lm.yaml"

-- block datasets_preprocessor_args
datasets_preprocessor_args: !dict
    max_length: 4096
    packed: True                    # Enable packing
    packing_strategy: "best_fit"   # Optimized bin packing
    overflow: False                 # Truncate long documents
    add_eos: True                   # Add EOS tokens (for compatibility)
    # Note: add_bos will be auto-disabled if tokenizer has no BOS
-- endblock datasets_preprocessor_args

-- block datacollator
datacollator: !partial:forgather.ml.data_collator:DataCollatorForCausalLM
    # packed_sequences: auto-detects from document_starts (no configuration needed!)
    max_length: 4096
    return_tensors: "pt"
-- endblock datacollator
```

### Dataset Inspection

```bash
# View packed dataset with document counts
forgather -t config.yaml dataset --tokenized --examples 5

# Output:
# ------------- 0 Tokens: 512, Documents: 3, Features: dict_keys(['input_ids', 'document_starts']) -------------
# Document 1 text... Document 2 text... Document 3 text...
```

## Backward Compatibility

The system automatically maintains backward compatibility:

1. **New Datasets**: Automatically generate `document_starts` when using `block_tokenize_fn`
2. **Old Datasets**: Fall back to token-based boundary detection if `document_starts` is missing
3. **Legacy Code**: Existing configurations continue to work without changes

```python
# Collator automatically detects available method
if "document_starts" in features:
    # Use explicit boundaries (preferred)
    position_ids = generate_from_boundaries(input_ids, document_starts)
else:
    # Fall back to token-based detection (legacy)
    position_ids = generate_from_tokens(input_ids, eos_token_id)
```

## Performance Considerations

### Storage Overhead

- **Document Starts**: ~2-5 integers per sequence (minimal)
- **Full Position IDs**: ~512-4096 integers per sequence (50-100x larger)

### Computation

- **Boundary-Based**: O(num_docs × seq_length) - faster for few documents
- **Token-Based**: O(seq_length) - faster for many documents per sequence

In practice, the boundary-based approach is faster for typical packing ratios (2-5 documents per sequence).

## Testing

Comprehensive tests are available in:
- `tests/test_document_boundaries.py` - Unit tests for all components
- `tests/test_qwen3_packing.py` - Integration tests with real Qwen3 tokenizer

Run tests:
```bash
pytest tests/test_document_boundaries.py -v
pytest tests/test_qwen3_packing.py -v
```

## Troubleshooting

### Issue: Position IDs not resetting

**Symptom**: Model fails to train, attention spans across documents

**Solution**: Ensure `packed_sequences=True` in `DataCollatorForCausalLM`

```python
collator = DataCollatorForCausalLM(
    tokenizer=tokenizer,
    packed_sequences=True,  # Must be True!
    return_tensors="pt",
)
```

### Issue: Document starts field missing

**Symptom**: `KeyError: 'document_starts'` when using old dataset

**Solution**: Re-tokenize dataset with updated `block_tokenize_fn`, or use token-based fallback

```python
# Option 1: Re-tokenize with new code (recommended)
dataset = dataset.map(
    lambda features: block_tokenize_fn(...),
    batched=True,
)

# Option 2: Collator will automatically fall back to token-based detection
```

### Issue: PAD tokens treated as boundaries

**Symptom**: Position IDs reset at every padding token

**Solution**: This is a sign of PAD==EOS aliasing (like Qwen3's `<|endoftext|>`). The new system solves this by using explicit boundaries instead of token-based detection.

## References

- Original sequence packing blog: https://huggingface.co/blog/sirluk/llm-sequence-packing
- Forgather block tokenizer: `src/forgather/ml/datasets/block_tokenizer.py`
- Forgather data collator: `src/forgather/ml/data_collator.py`
