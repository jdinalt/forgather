# OpenAssistant Dataset Examples

This directory contains working Python examples demonstrating various features of the OpenAssistant dataset implementation.

## Prerequisites

- Python 3.8+
- transformers library
- A tokenizer (examples use `~/ai_assets/models/fg_mistral/`)

## Running the Examples

All examples can be run directly:

```bash
# Make scripts executable
chmod +x *.py

# Run any example
./basic_usage.py
./custom_chat_template.py
./conversation_parsing.py
./quality_filtering.py
```

Or with python:

```bash
python basic_usage.py
python custom_chat_template.py
python conversation_parsing.py
python quality_filtering.py
```

## Examples Overview

### 1. basic_usage.py

**Demonstrates**: Core functionality and dataset creation

**Features**:
- Loading tokenizer
- Creating OpenAssistantConfig
- Creating OpenAssistantDatasetDict
- Accessing train/validation/test splits
- Iterating through examples

**Output**: Shows 3 example conversations with default configuration

### 2. custom_chat_template.py

**Demonstrates**: Using custom Jinja2 chat templates

**Features**:
- Creating custom chat template format
- Saving template to file
- Using template with dataset
- Configuring multiple languages
- Adjusting quality and temperature parameters

**Output**: Shows 2 examples formatted with custom template

### 3. conversation_parsing.py

**Demonstrates**: Parsing formatted conversation text

**Features**:
- Creating dataset with default ChatML formatting
- Parsing ChatML formatted text into structured messages
- Inspecting conversation turns
- Working with role and content fields

**Output**: Shows 3 conversations parsed into turn-by-turn messages

### 4. quality_filtering.py

**Demonstrates**: Advanced filtering and configuration

**Features**:
- Comparing datasets with/without quality filters
- Testing different quality thresholds
- Experimenting with branch temperature
- Understanding the impact of filtering parameters

**Output**: Shows examples from different configurations with quality analysis

## Example Output

### Basic Usage

```
============================================================
OpenAssistant Dataset - Basic Usage Example
============================================================

1. Loading tokenizer...
   Loaded tokenizer from /home/user/ai_assets/models/fg_mistral/

2. Creating configuration...
   Dataset length: 100
   Languages: ['en']
   Min quality: 0.5

3. Creating dataset dict...
   Available splits: ['train', 'validation', 'test']

4. Accessing splits...
   ✓ Train split created
   ✓ Validation split created
   ✓ Test split created

5. Generating examples from train split...
------------------------------------------------------------

Example 1:
<|im_start|>user
What is the best way to learn Python?<|im_end|>
<|im_start|>assistant
...
```

## Customization

All examples can be easily customized by modifying parameters:

```python
# Adjust dataset size
dataset_length=1000  # Default: varies by example

# Change languages
languages=['en', 'es', 'de']  # Default: ['en']

# Filter by quality
min_quality=0.7  # Default: None (no filter)

# Adjust conversation length
min_thread_length=3  # Default: 2
max_thread_length=10  # Default: 7

# Control branching randomness
branch_temperature=0.5  # Default: 1.0
```

## Notes

- Examples use small dataset sizes for quick execution
- Modify `tokenizer_path` if your tokenizer is in a different location
- The `custom_chat_template.py` example creates and cleans up a temporary template file
- All examples use deterministic seeds for reproducible output

## Troubleshooting

**Import errors**: Ensure you're running from the examples directory or that `src/` is in your Python path

**Tokenizer not found**: Update the `tokenizer_path` variable to point to your tokenizer location

**Out of memory**: Reduce `dataset_length` parameter in the examples
