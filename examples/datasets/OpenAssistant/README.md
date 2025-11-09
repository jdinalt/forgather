# OpenAssistant Dataset

[OpenAssistant](https://github.com/LAION-AI/Open-Assistant/) is a conversational AI dataset created by the LAION AI community. This implementation provides a high-performance, configurable dataset generator for training chat-based language models with the Forgather framework.

We take the conversation trees from the dataset and dynamically generate examples on-the-fly by 
performing a random tree-walk, where the branch probabilities are determined by the quality metrics.

## Overview

The OpenAssistant dataset consists of conversation trees - multi-turn dialogues between users and assistants where each message can have multiple reply branches. Our implementation:

- **Randomly samples conversation threads** from trees using quality-weighted branching
- **Applies chat templates** to format conversations for model training
- **Supports sequence packing** to maximize GPU utilization
- **Provides extensive filtering** by language, quality, and message attributes
- **Guarantees deterministic output** with configurable random seeds
- **Optimized for performance** with lazy dataset creation and custom fingerprinting

## Dataset Structure

**Source Format**: JSONL files containing conversation trees
- Each tree has a root prompt with nested reply branches
- Messages include quality scores, language tags, and metadata
- Trees are split deterministically into train/validation/test sets

**Output Format**:
- **Basic mode**: One conversation per example
- **Packed mode**: Multiple conversations packed into fixed-length sequences

## Key Features

### Quality-Weighted Sampling

When traversing conversation trees, replies are sampled using quality scores with temperature-controlled softmax:
- Higher quality messages have higher probability of selection
- `branch_temperature` parameter controls randomness (1.0 = proportional, > 0 = more uniform probabilities, < 0 = more deterministic)
- Missing quality scores are imputed with median values

### Filtering Options

**Language filtering**: Select specific languages (e.g., `['en', 'es', 'de']`)
**Quality threshold**: Minimum quality score for messages
**Thread length**: Min/max conversation turns (default: 2-7)
**Content filtering**: Exclude deleted or synthetic messages

### Deterministic Generation

All randomness uses configurable seeds for reproducibility:
- Tree selection and thread generation use deterministic hashing (CRC32)
- Identical configurations always produce identical datasets
- Each split (train/val/test) gets a unique derived seed

## Configuration Examples

### Basic Configuration

The `[openassistant.yaml](./templatelib/configs/openassistant.yaml)` configuration provides single conversations per example:

**Usage**:
```bash
# Dump first five examples from "train" split
forgather -t openassistant.yaml dataset -T path/to/tokenizer \
    --target train_dataset_split -n 5

# Generate dataset statistics and sequence length (tokens)
forgather -t openassistant.yaml dataset \
-T path/to/tokenizer --target train_dataset_split -H

# With Mistral 7B tokenizer...
sample size: 1000
min: 16
max: 2140
mean: 297.5119934082031
median: 234.0
std: 251.50820922851562
```

### Packed Configuration

The `[openassistant_packed.yaml](./templatelib/configs/openassistant_packed.yaml)` extends the base config to pack multiple conversations into each example:

**Usage**:
```bash
# Dump first three packed examples
forgather -t openassistant_packed.yaml dataset -T /path/to/tokenizer \
    --target train_dataset --max-length 2048 -s -n 3
```

## Direct Python Usage

### Basic Usage

```python
# Add src directory to path for imports
# This example assumes we are in the "examples" directory
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from openassistant import OpenAssistantDatasetDict, OpenAssistantConfig
from transformers import AutoTokenizer

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("/path/to/tokenizer")

# Create configuration
config = OpenAssistantConfig(
    languages=['en'],
    min_quality=0.5,
    min_thread_length=2,
    max_thread_length=7,
    exclude_deleted=True,
    exclude_synthetic=True,
    branch_temperature=1.0,
    seed=42,
    dataset_length=10000,
    val_split=10,
    test_split=10
)

# Create dataset dict
dataset_dict = OpenAssistantDatasetDict(
    tokenizer=tokenizer,
    chat_template="",  # Use tokenizer's chat template
    **config.__dict__
)

# Access splits
train_dataset = dataset_dict['train']
val_dataset = dataset_dict['validation']
test_dataset = dataset_dict['test']

# Iterate through examples
for example in train_dataset:
    print(example['text'])
    break
```

### Advanced: Custom Chat Template

```python
from pathlib import Path

# Define custom template
custom_template = """{% for message in messages %}
<|{{ message['role'] }}|>
{{ message['content'] }}
<|end|>
{% endfor %}"""

# Save template to file
template_path = Path("custom_template.jinja")
template_path.write_text(custom_template)

# Create dataset with custom template
dataset_dict = OpenAssistantDatasetDict(
    tokenizer=tokenizer,
    chat_template=str(template_path),
    languages=['en', 'es'],  # Multiple languages
    min_quality=0.6,         # Higher quality threshold
    branch_temperature=0.5,  # More deterministic branching
    dataset_length=50000,    # Larger dataset
    seed=123
)
```

### Advanced: Parsing Conversations

To parse the ChatML formatted text back into structured messages:

```python
import re

def parse_conversation(text):
    """Parse ChatML formatted text into messages."""
    pattern = r'<\|im_start\|>(user|assistant)\n(.*?)<\|im_end\|>'
    matches = re.findall(pattern, text, re.DOTALL)

    messages = []
    for role, content in matches:
        messages.append({
            'role': role,
            'content': content.strip()
        })
    return messages

# Create dataset
dataset_dict = OpenAssistantDatasetDict(
    tokenizer=None,
    chat_template="",  # Uses default ChatML template
    languages=['en'],
    dataset_length=100
)

# Parse examples
for example in dataset_dict['train']:
    messages = parse_conversation(example['text'])
    for msg in messages:
        print(f"{msg['role']}: {msg['content'][:100]}...")
    break
```

See `examples/conversation_parsing.py` for a complete working example.

## Implementation Details

### Performance Optimization

The implementation uses several optimizations for high performance:

1. **Custom fingerprinting**: Avoids expensive dill pickling by hashing configuration parameters
2. **Lazy split creation**: Datasets are created only when accessed
3. **Pre-indexed trees**: Tree lookups use O(1) indexing instead of linear search
4. **Streaming datasets**: Uses `IterableDataset` for memory-efficient iteration

**Performance metrics**:
- Dataset initialization: ~4 seconds (loads 13,854 trees)
- Split creation: ~0.002 seconds (17,670x faster than naive implementation)
- Example generation: <0.001 seconds per example

### Tree Database

The `TreeDatabase` class provides efficient access to conversation trees:

```python
class TreeDatabase:
    """Fast, indexed access to conversation trees with filtering."""

    def __init__(self, trees: List[Dict], config: Dict):
        # Index trees by ID for O(1) lookup
        self.trees_by_id = {tree['message_tree_id']: tree for tree in trees}
        self.tree_ids = list(self.trees_by_id.keys())
        self.config = config

    def get_random_tree(self, rng: random.Random) -> Dict:
        """Select a random tree using provided RNG."""
        tree_id = rng.choice(self.tree_ids)
        return self.trees_by_id[tree_id]
```

### Thread Generator

The `ThreadGenerator` class implements deterministic conversation sampling:

```python
class ThreadGenerator:
    """Generates conversation threads from trees with quality-weighted branching."""

    def __init__(self, tree_db, length, config, chat_template, template_args, seed):
        self.tree_db = tree_db
        self.length = length
        self.config = config
        self.chat_template = chat_template
        self.template_args = template_args
        self.rng = random.Random(seed)
        self.count = 0

    def __iter__(self):
        while self.count < self.length:
            tree = self.tree_db.get_random_tree(self.rng)
            thread = extract_random_thread(tree, self.config, self.rng)

            if thread:
                self.count += 1

                # Apply chat template or return raw messages
                if self.chat_template:
                    text = self.chat_template.render(messages=thread, **self.template_args)
                    yield {'text': text}
                else:
                    yield {'messages': thread}
```

### Deterministic Hashing

Python's built-in `hash()` is randomized for security. We use CRC32 for fast, deterministic hashing:

```python
import zlib

def deterministic_hash(s: str) -> int:
    """Create deterministic hash using CRC32."""
    return zlib.crc32(s.encode()) & 0xFFFFFFFF
```

## Configuration Parameters

### OpenAssistantConfig

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_file_path` | str | Auto-detected | Path to trees JSONL file |
| `cache_dir` | str | None | Cache directory for downloads |
| `languages` | List[str] | `['en']` | Languages to include |
| `min_quality` | float | None | Minimum quality threshold |
| `min_thread_length` | int | 2 | Minimum conversation turns |
| `max_thread_length` | int | 7 | Maximum conversation turns |
| `exclude_deleted` | bool | True | Exclude deleted messages |
| `exclude_synthetic` | bool | True | Exclude synthetic messages |
| `branch_temperature` | float | 1.0 | Branching randomness (higher = more random) |
| `seed` | int | 42 | Random seed for reproducibility |
| `dataset_length` | int | 10000 | Examples per epoch (-1 for infinite) |
| `val_split` | int | 10 | Validation split percentage |
| `test_split` | int | 10 | Test split percentage |

### OpenAssistantDatasetDict

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `tokenizer` | PreTrainedTokenizer | None | HuggingFace tokenizer (for template args) |
| `chat_template` | str | "" | Path to template file or empty for tokenizer's template |
| `**config_params` | - | - | All OpenAssistantConfig parameters |

## Dataset Statistics

**Source**: OpenAssistant 2023-11-05 release
**Total trees**: 13,854 conversation trees
**Languages**: 35+ languages (English is most common)
**Quality scores**: Community-labeled quality ratings
**Average tree depth**: ~3-5 turns
**Max tree depth**: 15+ turns in some cases

**Default split distribution** (with 10% val, 10% test):
- Train: ~11,083 trees (80%)
- Validation: ~1,385 trees (10%)
- Test: ~1,386 trees (10%)

## Common Use Cases

### Multi-language Training

```bash
# Train on English, Spanish, and German
forgather -t openassistant.yaml dataset \
    --languages en,es,de \
    --dataset-length 100000
```

### High-Quality Subset

```bash
# Use only high-quality conversations
forgather -t openassistant.yaml dataset \
    --min-quality 0.7 \
    --min-thread-length 3
```

### Experimentation with Small Datasets

```bash
# Quick testing with small dataset
forgather -t openassistant.yaml dataset \
    --dataset-length 1000 \
    --seed 42
```

## References

- **OpenAssistant Project**: https://github.com/LAION-AI/Open-Assistant
- **Dataset**: https://huggingface.co/datasets/OpenAssistant/oasst2
- **Paper**: "[OpenAssistant Conversations - Democratizing Large Language Model Alignment (2023)](https://arxiv.org/abs/2304.07327)" 

## License

The OpenAssistant dataset is released under Apache 2.0 license. This implementation is part of the Forgather framework.
