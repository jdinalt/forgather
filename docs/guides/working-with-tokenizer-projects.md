# Working with Tokenizer Projects

Tokenizer projects define custom tokenizer configurations — for example, a BPE tokenizer trained on a specific corpus. They follow the standard Forgather project layout and use the same CLI commands as any other project.

```bash
# List available tokenizer configurations
forgather ls

# Show the preprocessed tokenizer configuration
forgather -t <config_name> pp

# Build the tokenizer (if it has not yet been built)
forgather -t <config_name> construct
```

For example tokenizer projects, see [`examples/tokenizers/`](../examples/tokenizers/README.md).
