# Dataset Project CLI Reference

Forgather dataset configurations wrap datasets in a uniform abstraction layer that hides the details of loading, preprocessing, and tokenizing data.

## Targets

Every dataset project is expected to export the following targets:

**Mandatory:**
- `train_dataset` — The primary training split with preprocessing and tokenization applied.
- `eval_dataset` — A hold-out dataset derived from the validation or test split, depending on availability and configuration.

**Optional:**
- `train_dataset_split` — The unprocessed train split.
- `validation_dataset_split` — The unprocessed validation split.
- `test_dataset_split` — The unprocessed test split.

**Terminology:** The "train" split is the only one the model should ever be trained on. The "validation" split is used for hyperparameter tuning and checking for over/under-fit, while the "test" split is held out from both the model and the experimenter until all models are fully trained. "eval" refers to either the validation or test split depending on the use-case.

## General CLI

```bash
# List available configurations in the current project
forgather ls

# List all dataset configurations recursively across subdirectories
forgather ls -r

# Show project metadata in markdown format
forgather meta

# Show preprocessed dataset configuration
forgather -t tinystories.yaml pp

# Show preprocessed configuration with template debugging enabled
forgather -t tinystories.yaml pp -d

# Parse preprocessed configuration and compile into a Forgather graph
# (if the error is not at the preprocessing level, check here next)
forgather -t tinystories.yaml graph

# As above, but with more detailed template debugging enabled
forgather -t tinystories.yaml graph -d

# Show template hierarchy in markdown format
forgather -t tinystories.yaml trefs --format md
```

Note that the template name in \[square-brackets\] in `forgather ls` output is the default when no `-t` flag is specified.

## Inspecting Targets

List and materialize individual targets for inspection and testing:

```bash
# List all exported targets
forgather -t tinystories.yaml targets
tokenizer
train_dataset_split
validation_dataset_split
test_dataset_split
train_dataset
eval_dataset
meta
main

# Construct a target and print it
forgather -t tinystories.yaml construct --target train_dataset_split
Dataset({
    features: ['text'],
    num_rows: 2119719
})

# Show the equivalent Python code to construct a target
forgather -t tinystories.yaml code --target train_dataset_split
```

Note: preprocessed dataset targets require arguments (tokenizer, preprocess args) that `construct` does not inject automatically. Use the `dataset` command below to inspect tokenized targets.

## Dataset CLI

The `forgather dataset` subcommand is specifically for inspecting and testing dataset configurations:

```
forgather dataset [-h] [-T TOKENIZER_PATH] [--pp] [-H] [--target TARGET]
                  [--histogram-samples HISTOGRAM_SAMPLES] [-c CHAT_TEMPLATE]
                  [-n EXAMPLES] [-s]

options:
  -T, --tokenizer-path  Path to tokenizer to test
  --pp                  Show preprocessed configuration
  -H, --histogram       Generate dataset token length histogram and statistics
  --target TARGET       The dataset to sample from; see "forgather targets"
  --histogram-samples N Number of samples to use for histogram
  -c, --chat-template   Path to chat template
  -n, --examples N      Number of examples to print
  -s, --tokenized       The split is already tokenized (decode input_ids instead of reading text)
```

### Examples

```bash
# Print first three examples from the raw train split
forgather -t tinystories.yaml dataset --target train_dataset_split -n 3

# Print first three examples with preprocessing, tokenization, and decoding
forgather -t tinystories.yaml dataset --target train_dataset -n 3 -T path/to/tokenizer

# Generate a token length histogram (saves an SVG to the project directory)
forgather -t tinystories.yaml dataset --target train_dataset_split \
    -H -T ~/models/llama-2-7b-fg --histogram-samples 100000

# Show preprocessed examples with a chat template applied, tokenized, and decoded
forgather -t samantha.yaml dataset --target train_dataset \
    --chat-template ../../../chat_templates/chatml.jinja \
    -T ~/models/llama-2-7b-fg -n 3

# Inspect a packed/tokenized split (use -s to decode input_ids)
forgather -t openorca-packed.yaml dataset --target train_dataset \
    -T ~/models/fg_llama_1b --max-length 2048 -s -n 2
```
