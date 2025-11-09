# Examples Index

Quick reference for finding the right example for your use case.

## By Use Case

### Getting Started
- **basic_usage.py** - Start here for your first OpenAssistant dataset

### Custom Formatting
- **custom_chat_template.py** - Using custom Jinja2 templates for conversation formatting

### Data Analysis
- **conversation_parsing.py** - Parsing ChatML formatted conversations into structured data
- **quality_filtering.py** - Filtering and analyzing dataset quality

## By Feature

### Chat Templates
- custom_chat_template.py - Creating and using custom templates
- conversation_parsing.py - Parsing template output

### Dataset Configuration
- basic_usage.py - Basic configuration options
- quality_filtering.py - Advanced filtering (quality, temperature, length)

### Multi-language Support
- custom_chat_template.py - Using multiple languages

### Data Inspection
- conversation_parsing.py - Analyzing conversation structure
- quality_filtering.py - Comparing different configurations

## Quick Command Reference

```bash
# Run all examples
for script in *.py; do python "$script"; done

# Test all examples work
for script in *.py; do
    python "$script" > /dev/null 2>&1 && echo "✓ $script" || echo "✗ $script"
done
```

## File Sizes

- basic_usage.py: ~2.6KB
- conversation_parsing.py: ~2.6KB
- custom_chat_template.py: ~2.4KB
- quality_filtering.py: ~3.8KB

## Execution Time

All examples run in under 10 seconds with small dataset sizes configured for quick testing.

## Requirements

All examples require:
- Python 3.8+
- transformers library
- Access to a tokenizer at `~/ai_assets/models/fg_mistral/` (configurable)

See README.md for detailed documentation of each example.
