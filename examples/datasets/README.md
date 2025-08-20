# Forgather Dataset Definitions

A forgather dataset configuraitons wrap datasets in a uniform abstraction layer. This abstraction is responsible for hiding the details of how to load the dataset, how to preprocess the dataset, and how to tokenize the dataset, if applicable.

**Mandatory Targets**

- train_dataset: The primary training dataset split, with preprocessing and tokenization applied.
- eva_dataset: This is a hold-out dataset, which can either be derived from the "validation" or "test" split, depending upon availability and configuration.

**Optional Targets**

- train_dataset_split: The unprocessed train split
- validation_dataset_split: The unprocessed validation split
- test_dataset: The unprocessed test split

 **Terminology**

 The "train" split is the only one the model should ever be trained on. The "validation" split is typically used for hyper-parameter tuning and checking for over/under-fit, while the "test" is intended to be held out from both the model and the experimenter, until all models have been fully trained and it's time to compare the end results.

 We use the term "eval" to refer to either the "validation" or the "test" split, depending upon the use-case.

## General Command Line Interface

```bash
usage: forgather [-h] [-p PROJECT_DIR] [-t CONFIG_TEMPLATE] {index,ls,meta,targets,tlist,graph,trefs,pp,tb,code,construct,train,dataset} ...

Forgather CLI

positional arguments:
  {index,ls,meta,targets,tlist,graph,trefs,pp,tb,code,construct,train,dataset}
                        subcommand help
    index               Show project index
    ls                  List available configurations
    meta                Show meta configuration
    targets             Show output targets
    tlist               List available templates.
    graph               Preprocess and parse into node graph
    trefs               List referenced templates
    pp                  Preprocess configuration
    tb                  Start Tensorboard for project
    code                Output configuration as Python code
    construct           Materialize and print a target
    train               Run configuration with train script
    dataset             Dataset preprocessing and testing

options:
  -h, --help            show this help message and exit
  -p PROJECT_DIR, --project-dir PROJECT_DIR
  -t CONFIG_TEMPLATE, --config-template CONFIG_TEMPLATE
                        Configuration Template Name
```

### Examples

 ```bash
# List all datasets in all subdirectories
forgather ls -r
...

# List datasets in current project directroy
forgather ls
Generic Datasets Meta : Some datasets
    tinystories.yaml               Tiny Stories : Dataset containing synthetically generated (by GPT-3.5 and GPT-4) short stories that only use a small vocabulary.
    [tinystories-abridged.yaml]    Tiny Stories Abridged : Dataset containing synthetically generated (by GPT-3.5 and GPT-4) short stories that only use a small vocabulary.

# Show project meta-data in markdown format
forgather meta
```

Note that the template name in \[square-brackets\] is the default, if otherwise unspecified.

When running the above commands, the pretty name and description show an error, it indicates that there was either a preprocessing or parsing error, which needs to be debugged.

```bash
# Show preprocessed dataset configuration
forgather -t tinystories.yaml pp

# Show preprocessed dataset configuration with template debugging enabled
forgather -t tinystories.yaml pp -d

# Parse preprocessed configuraiton and compile into a Forgather graph
# If there is an error, and it's not at the preprocessing level, this is the next place to look.
forgather -t tinystories.yaml graph

# As above, but with more detailed template debugging enabled.
forgather -t tinystories.yaml graph -d

# Show template hierarchy in markdown format
forgather -t tinystories.yaml trefs --format md
```
Each configuration is expected to export one or more targets, with the primary ones documented above.

```bash
# List targets
forgather -t tinystories.yaml targets
tokenizer
train_dataset_split
validation_dataset_split
test_dataset_split
train_dataset
eval_dataset
meta
main
```
Each target can be constructed individually for inspection and testing.

```bash
forgather -t tinystories.yaml construct --target train_dataset_split
Dataset({
    features: ['text'],
    num_rows: 2119719
})
```

Each target can also be represented by the equivalent Python code to construct it.

```python
forgather -t tinystories.yaml code --target train_dataset_split
from datasets import load_dataset

def construct(
):
    train_dataset_split = load_dataset(
        'roneneldan/TinyStories',
        split='train',
    )
    
    return train_dataset_split

# Note that the preprocessed dataset require arguments to be passed in; this will prevent "construct" from working, as it does not know
# which args are required. Use the "dataset" command to inject the correct arguments.
forgather -t tinystories.yaml code --target train_dataset      
from datasets import load_dataset
from forgather.ml.datasets import preprocess_dataset

def construct(
    preprocess_args,
    tokenizer,
):
    train_dataset_split = load_dataset(
        'roneneldan/TinyStories',
        split='train',
    )

    train_dataset = preprocess_dataset(
        dataset=train_dataset_split,
        tokenizer=tokenizer,
        desc='Tokenizing train',
        fn_kwargs=preprocess_args,
    )
    
    return train_dataset
```

## Dataset CLI

```bash
# Dataset Specific
usage: forgather dataset [-h] [-T TOKENIZER_PATH] [--pp] [-H] [--target TARGET] [--histogram-samples HISTOGRAM_SAMPLES] [-c CHAT_TEMPLATE] [-n EXAMPLES] [-s]

options:
  -h, --help            show this help message and exit
  -T TOKENIZER_PATH, --tokenizer-path TOKENIZER_PATH
                        Path to tokenizer to test
  --pp                  Show preprocessed configuration
  -H, --histogram       Generate dataset token length historgram and statistics
  --target TARGET       The dataset to sample from; see "forgather targets"
  --histogram-samples HISTOGRAM_SAMPLES
                        Number of samples to use for histogram
  -c CHAT_TEMPLATE, --chat-template CHAT_TEMPLATE
                        Path to chat template
  -n EXAMPLES, --examples EXAMPLES
                        Number of examples to print
  -s, --tokenized       The split is already tokenized
```

### Examples

```bash
# Print first three examples in raw train split
forgather -t tinystories.yaml dataset --target train_dataset_split -n 3
Printing 3 examples from the train dataset:
Tokenizer path not provided, skipping tokenization.
----------------------------------------
One day, a little girl named Lily found a needle in her room. She knew it was difficult to play with it because it was sharp. Lily wanted to share the needle with her mom, so she could sew a button on her shirt.

Lily went to her mom and said, "Mom, I found this needle. Can you share it with me and sew my shirt?" Her mom smiled and said, "Yes, Lily, we can share the needle and fix your shirt."
...
```

```bash
# Print first three examples, with preprocessing, tokenization, and decoding the resuult
forgather -t tinystories.yaml dataset --target train_dataset -n 3 -T path/to/tokenizer

# Generate token length historgram
forgather -t tinystories.yaml dataset --target train_dataset_split -H -T ~/ai_assets/models/llama-2-7b-fg --histogram-samples 100000
Generating token-length histogram: /home/dinalt/ai_assets/forgather/examples/datasets/roneneldan/tinystories.svg
sample size: 100000
min: 1
max: 1229
mean: 243.0797576904297
median: 209.0
std: 116.73296356201172
```
Example Histogram: [tinystories.svg](roneneldan/tinystories.svg)

Some dataseet need a chat template for their preprocessor. Some tokenizer have one embedded in them, if not you can manualy specify one..

```bash
# Show first three preprocessed examples, with the chat template applied, tokenized, and decoded
forgather -t samantha.yaml dataset --target train_dataset --chat-template ../../../chat_templates/chatml.jinja -T ~/ai_assets/models/llama-2-7b-fg -n 3
Printing 3 examples from the train dataset:                                                                                                                                                                                                                                                                                  
----------------------------------------                                                                                                                                                                                                                                                                                     
<s> <|im_start|>user
Hey Samantha, I've run into a bit of a tricky situation at work, and I'm not sure how to handle it. Do you have any advice?<|im_end|>
<|im_start|>assistant
I'd be happy to help if I can. Can you give me some details about the situation you're facing?<|im_end|>
...
```