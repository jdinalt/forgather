# H.P. Lovecraft Project Tutorial

Finetune a model on the complete works of H.P. Lovecraft to summon the Elder Gods

## What You'll Learn

This tutorial teaches you how to:
- ✓ Create a Forgather workspace
- ✓ Create a new dataset project from raw text files
- ✓ Create a finetuning project for your new dataset
- ✓ Train a 7B parameter model, with a 4K context length, on a single GPU
- ✓ Run the resulting model on an inference server to generate new stories with context of 8K or longer.

**Time required**: ~2 hours
**Hardware requirements**: 1 GPU with 24 GB of VRAM (RTX 3090, RTX 4090, etc.)

## Setup

This is a complete project from scratch. The only things provided are instructions and a collections of text files to train on. The first order of business is to extract the text files from the archive:

```bash
# Extract text files
tar -xzf hp_lovecraft.tgz

# This will produce the directory "hp_lovecraft"
# If you like, you can take a look at the stories
less hp_lovecraft/the_call_of_cthulhu.txt
```

The tutorial assumes that everything is created within the tutorial directory, but feel free to work outside of the "Forgather" tree, if you like. This will only require that you adjust the paths from the examples accordingly.

You will need a 7B Llama-flavored model to train. If you have already gone through the "Samantha" tutorial, you can reuse the existing Mistral 7B model from that project. In theory, any model with <= 7B Parameters should work.

### Download Example Model
```bash
# Download the model
MODELS_DIR="~/models" # Change this to where you store your models...
SRC_MODEL="${MODELS_DIR}/mistral_7b"
mkdir -p "${MODELS_DIR}"
huggingface-cli download mistralai/Mistral-7B-v0.1 --local-dir "${SRC_MODEL}" \
--exclude "*.safetensors" "model.safetensors.index.json"
```

### Convert Model to Fg Format

One of our memory saving strategies, CPU activation offloading, was not working with the Mistral model when last checked. One workaround is to convert the model to the Forgather format, which does work.

```bash
# From the Forgather root directory
# Set name for converted model
FG_MODEL="${MODELS_DIR}/fg_mistral_7b"

# Convert model to Forgather Llama/Mistral implementation
scripts/convert_llama.py --model-type mistral --dtype bfloat16 --max-length 4096 \
 "${SRC_MODEL}" "${FG_MODEL}"
```

### Convert Model Back to HF Format

```bash
scripts/convert_llama.py --reverse --model-type mistral --dtype bfloat16 \
--max-length 32000 "${FG_MODEL}" OUTPUT_MODEL_PATH
```

### Syntax Highlighting

We will be editing Forgather configuration files. If you have not already installed the syntax-highlighting plugins for vim / VS Code, follow the instructions in "syntax_highlighting/" This will make the config files much more readable.

Otherwise, "yaml" syntax tends to be the closest option.

### Setup VS Code

You don't have to use VS Code, but it makes editing files much easier, as we have an integration which allows the CLI tool to open files directly in the editor. If you are running from a VS Code terminal, everything should "just work." If you are using an external terminal, you can still use it to open files for editing, like this:
From a VS code terminal, run:

```bash
# Get VSCODE_IPC_HOOK_CLI environment variable
env | grep VSCODE_IPC_HOOK_CLI
VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-ff6b36d6-cb18-4f6c-8d7d-2b354c82a7ea.sock
```

Once you have the value, copy and paste is into whatever other terminal you are using.

```bash
# Export the IPC HOOK
export VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-ff6b36d6-cb18-4f6c-8d7d-2b354c82a7ea.sock
```

## Create a Forgather Workspace

A forgather workspace defines a common set of configurations for a collection projects. While it is possible to manually construct one, it's much easier to use the CLI.

When creating a new workspace, you must specify the location of the Forgather directory, as this is needed to find the template libraries. The path may be relative, as in this example, or absolute.

The Forgather template libraries to include can be specified with "-l LIB_NAME." In this case, we are including the base-templates and the finetune libraries.

```bash
# Create a new Forgather workspace
forgather ws create --name "H.P. Lovecraft Workspace" --description "H.P. Lovecraft tutorial workspace" \
--forgather-dir ../../../ -l base -l finetune
```

Enter the new workspace and take a look at some of the generated files.

```bash
# Enter new workspace directory
cd hp_lovecraft_workspace/
cat forgather_workspace/base_directories.yaml
cat forgather_workspace/meta_defaults.yaml
```

The common workspace files can be found in the "forgather_workspace" directory. Any templates located here are available to all projects in the workspace.

 The "base_directories.yaml" file contains path definitions which should be available to both the meta-configuration and all projects. The CLI tool automatically generates a definition for finding the "forgather" directory, but you could add more paths there. For example, the paths to where you store your models and datasets.

The primary role of "meta_defaults.yaml" is to define the default search paths for forgather configuration files. The file is "extended" by each project's "meta.yaml" file, which can override the defaults.

## Create a Forgather Project

From inside of the workspace directory, we can create projects, which belong to the workspace. First. let's create a new project for our dataset. When creating a new project we can (optionally) copy the default configuration from an existing config.

We are creating a local dataset with relatively large examples (complete stories). A good starting point will be to use the "local_dataset" example's "sliding_window" configuration, which can split large examples into multiple, overlapping, blocks.

Note: If the original file were located within our search paths, we could instead "extend" it, which would reduce code duplication. For the tutorial, it will be easier to understand if we just duplicate and modify it -- we can always refactor it later.

```bash
# Create a new Forgather dataset project
forgather project create --name "Lovecraft Dataset" --description "The complete works of H.P. Lovecraft" \
--default-config lovecraft.yaml ../../../datasets/local_dataset/templatelib/configs/sliding_window.yaml

# Enter the new project directory
cd lovecraft_dataset/
```

At this point, I would recommend switching to interactive mode, as it makes the workflow much easier. You can still continue from your shell, if your prefer. If running in interactive mode, you can drop the "forgather" prefix from the commands.

```bash
# Start Forgather shell
forgather -i
```

Take a quick look at the new project. Note how our configuration is just a direct copy of the original "local_dataset" configuration.
```bash
# Show project info
forgather project show

# List configurations
forgather ls
```

### Customize the Configuration

Next, we will customize the copied configuration. Open the configuration in an editor.

```bash
# From the interactive shell, you can use the "edit" command.
forgather:lovecraft_dataset> edit
# Enter the number corresponding to "lovecraft.yaml"
Select template(s) (0-35): 23

# From the shell, the equivalent command would be something like this...
vim templates/configs/lovecraft.yaml
```

The base file is pretty close to what we need. We will need to update the meta-data and the dataset definition, as shown below:

```yaml
# Update the metadata
[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft"
    -- set ns.config_description = "The complete works of H.P. Lovecraft"
...

# Modify the datast_dict definition
[dataset_dict]
dataset_dict: &dataset_dict !singleton:datasets:load_dataset
    arg0: "text"
    data_dir: {{ dataset_path | default("/path/to/dataset") }}
    sample_by: "document"
    data_files:
        train: "*.txt" # Train on all files
        validation: "the_call_of_cthulhu.txt" # Validate only on this one file
...

[train_dataset]
    ...
    # Change this to false. It's a small dataset and this keeps things simple
    to_iterable: False
```

To keep things simple, we validate on a single story, which is also part of the "train" split. Ideally, you should exclude it from "train," but I don't want to over-complicate this example.

### Test the Configuration

```bash
# Show the preprocessed configuration
forgather pp

# Construct the base dataset and display it
# Note that the configuration does not know the path to the text files, so we have to provide it.
# This should show the number of examples in the dataset splits
forgather construct --target dataset_dict --dataset-path ../../hp_lovecraft


# Dump the first example (story) from the train split
# This is a fairly large file, so you may want to pipe it through "head" or "less"
# If using the interactive interface, the "less" pager will be automatically used.
forgather dataset --target train_dataset_split --dataset-path ../../hp_lovecraft -n 1
```

Next, we will test the sliding-window pre-processor. This requires a tokenizer, as specified by the "-T TOKENIZER" argument. If you have already built any of the Forgather tokenizers, these will work. You can also just point it at the model directory of the model you will be using.

- --window-size: This is how may tokens are visible in each window. We will be using 4096 for the real configuration, but to demo what it does, set it to 64.
- --stride: This controls how many tokens overlap between windows. In this case, eight.
- -n : This is how may examples to show

```bash
forgather dataset --target train_dataset --dataset-path ../../hp_lovecraft -T ../../../../../tokenizers/wikitext_32k/ \
--window-size 64 --stride 8 -n 3
```

### Create a New Configuration

Next, let's create a new configuration for 4K tokens. While we are at it, we will hard-code the path to the raw data as well.

```bash
# Create a new configuration, named "4K.yaml"
# We will copy the existing configuration as a starting point
forgather project new_config 4k.yaml templates/configs/lovecraft.yaml
```

Open the new configuration in an editor.

```bash
# Interactive mode
forgather:lovecraft_dataset> edit
...
Project Configs:
  23. templates/configs/4k.yaml
  24. templates/configs/lovecraft.yaml.
...
Select template(s) (0-36): 23

# Shell
vim templates/configs/4k.yaml
```

We can delete most of the template, as we will inherit from the original. The final template should look like this, sans-comments:

```yaml
# Inherit from our previous configuration
-- extends 'configs/lovecraft.yaml'

# Update the meta-data
[config_metadata]
    == super()
    -- set ns.config_name = "Lovecraft 4K"
    -- set ns.config_description = "Lovecraft with 4K Blocks"

# Hard-code the map-function parameters
[map_function]
.define: &map_function !partial:forgather.ml.dataset.block_tokenizer:block_tokenize_fn
    block_size: 4096
    stride: 512

# Hard-code the location of the raw-data. If you have moved it outside of the Forgather directory, you will
# need to modify the example for wherever the data is at.
[dataset_dict]
    == super()
    data_dir: "{{ joinpath(ns.forgather_dir, "examples/tutorials/hp_lovecraft_project/hp_lovecraft/") }}"
```

#### Directory Paths

A brief digression is in order. When specifying paths, you should try to either make them absolute or relative to a well-defined symbolic location. If you make them completely relative, say "../../hp_lovecraft," then your configuration MUST be executed from the project directory, otherwise, the relative path will be incorrect. To help address this issue, we define a number of symbolic directories, which you can use to anchor relative paths. For example, the above example uses "ns.forgather" as an anchor point, which was imported from "base_directories.yaml."

- ns.forgather : The location of the forgather directory
- project_dir : The absolute path to the project directory
- workspace_root : The absolute path to the workspace directory
- user_home_dir() : Returns the absolute path to the user's home directory
- getcwd() : The current-working-directory
- forgather_config_dir() : Get the platform-specific config directory for Forgather

The following additional locations, as defined by https://pypi.org/project/platformdirs/
- user_data_dir()
- user_cache_dir()
- user_config_dir()
- site_data_dir()
- site_config_dir()

The Jinja2 environment also exports a number of directory manipulation functions:
- joinpath(*names) : Join a list of file-path segments via os.path.join()
- basename(path) : Get the file name part of a path; os.path.basename()
- dirname(path) : Get the directory par of the path; os.path.dirname()
- splitext(path) : Split the extension from a path; os.path.splitext()
- normpath(path) : Normalize a file path; os.path.normpath()
- abspath(path) : Convert path to absolute path; os.path.abspath()
- relpath(path) : Convert a path to a relative path; os.path.relpath()

### Test the "4k" Configuration

Make sure that it parses. An easy way to do this is with the "ls" command, as it both preprocesses and parses each configuration. If something fails, add "-d" to the command to debug the failure.

```bash
# Make sure the configuration parses
forgather ls
Lovecraft Dataset : The complete works of H.P. Lovecraft
    4k.yaml                        Lovecraft 4K : Lovecraft with 4K Blocks
    [lovecraft.yaml]               Lovecraft : The complete works of H.P. Lovecraft
```

Given that we have hard-codes the arguments, let's make sure that they work.

```bash
# If using interactive mode, change the selected configuration
forgather:lovecraft_dataset> config 4k.yaml

# And test it...
forgather:lovecraft_dataset [4k.yaml]> dataset --target train_dataset --dataset-path ../../hp_lovecraft \
-T ../../../../../tokenizers/wikitext_32k/ -n 2

# Otherwise, from the shell...
forgather -t 4k.yaml dataset --target train_dataset --dataset-path ../../hp_lovecraft \
-T ../../../../../tokenizers/wikitext_32k/ -n 2 | head # or "less"
```

### Check the Sequence Length Histogram

Given that we are splitting the dataset into 4K blocks, we should expect that most of the samples should be about 4K tokens in length. We can verify this with the "--histogram" argument.

```bash
frogather -t 4k.yaml dataset --target train_dataset --dataset-path ../../hp_lovecraft -s \
-T ../../../../../tokenizers/wikitext_32k/ --histogram
sample size: 195
min: 490
max: 4096
mean: 3491.60009765625
median: 4096.0
std: 1085.0946044921875
```
Note that this also outputs a ".svg" file, with the histogram.

## Create a new "finetune" Project

First, move back to the workspace directory.

```bash
cd ..
```

Now, we will create a second project. This one will be for the actual training. The closest existing configuration to what we will be doing is the "Samantha long context" configuration, so we will copy that as our default configuration.

```bash
# Create a new fine-tune project
forgather project create --name "Finetune Lovecraft" --description "Finetune a model on the complete works of H.P. Lovecraft" \
--default-config 1gpu/default.yaml ../../../finetune/samantha/templates/configs/1gpu_llama_7b/long_context.yaml

# Enter the project directory
cd finetune_lovecraft/
```

### Copy Project Level Configuration

As the copied configuration depends upon a project-level, "samantha.yaml," configuration, we will also copy that as well. We can control where the copied configuration goes with the "--type CONFIG_TYPE" argument, like this:

```bash
forgather project new_config --type project project.yaml ../../../../finetune/samantha/templates/samantha.yaml
```

### Customize the Configuration

The new files can be found at:
- Project: "templates/project.yaml"
- Default Config: "templates/configs/1gpu/default.yaml"

These files are pretty close to what we need. We just need to make a few small changes...

project.yaml:
```yaml
# Update the meta-data for our project
[config_metadata]
    == super()
    ## We find the dataset project relative to the "workspace_root"
    -- set ns.default_dataset_proj = joinpath(workspace_root, 'lovecraft_dataset')
    ## Use our config with static settings
    -- set ns.default_dataset_config = "4k.yaml"
...
```

default.yaml:
```yaml
# Change the parent template to match
-- extends 'project.yaml'

# Update meta-data with appropriate descriptions
[config_metadata]
    == super()
    -- set ns.config_name = "Finetune Lovecraft Default"
    -- set ns.config_description = "Train with 4096 token context on single GPU, 24 GBs"
    -- set ns.log_name = "1gpu_4096"
...
```

### Verify the Finetune Configuration

```bash
# Verify that the config parses
forgather ls
Finetune Lovecraft : Finetune a model on the complete works of H.P. Lovecraft
    [1gpu/default.yaml]            Finetune Lovecraft Default : Train with 4096 token context on single GPU, 24 GBs\

# Take a look at the final, preprocessed, configuration. Verify that this looks like what you want.
forgather pp
```

## Train

First, check what dynamic arguments the training configuration accepts:

```bash
forgather:finetune_lovecraft> train --help
```

At a minimum, we will need to specify the path to our model. Let's also train for 3 epochs.
On an RTX 4090, this takes about 30 minutes -- it's a small dataset, with only 153 examples (overlapping blocks).

First, check that everything is working with a short test run:
```bash
forgather train --max-steps 10 --save-strategy no --model-id-or-path ~/ai_assets/models/fg_mistral
```

Assuming that it runs, go ahead a train on the complete dataset for 3 epochs.

```bash
forgather train --train-epochs 3 --model-id-or-path ~/ai_assets/models/fg_mistral
```

## Inference

You can test the resulting model using the provided Open-AI compatible inference server and client or with 3rd party tools. Keep in mind that if you have converted the model to "Forather" format, the model will not be optimized for inference (no KV cache), thus it may struggle (slow) with long context lengths. You can convert it back to HF format for faster inference.

We will use the model "as-is" to verify that it is working.

```bash
# Start inference server (from 'forgather' directory)
# Change the model path to match your output directory.
tools/inference_server/server.py -d "cuda:0" -T bfloat16 \
-c -m /home/dinalt/ai_assets/models/fg_mistral

# Note: -c : This will search for the latest checkpoint, rather than loading the weights from the root directory.
```

We did not teach the model a chat-format, so we should use "completion mode." We need "seed" text, which has been lifted from the start of 'The Call of Cthulu.' The model will take the seed and continue generation from there.

```bash
./tools/inference_server/client.py --stream --completion "Of such great powers or beings there may be conceivably a survival" --max-tokens 512
```

Of such great powers or beings there may be conceivably a survival in terms of legend and half-legend. Of this I cannot doubt. But it is not on such powers or beings that my reflections have mainly been concentrated. What I have chiefly sought to investigate is the nature and extent of the powers or abilities which are potentially, if unwittingly, within the reach of any human individual.
 
 The ancient magicians were no doubt nearer right than any others in believing that man is related to natural forces of a certain kind; and that he can, through certain rites and ceremonies, influence these forces in his favour. But they went astray in selecting the forces which they tried to influence; and in thinking that their influence could be exercised only in certain well-defined ways, and for certain well-defined ends.
 
 The truth is that man is related to the whole universe as intimately as the cells of his body are related to each other; and that, if he is sufficiently enlightened, he can deliberately call upon any force or forces in the cosmos for any purpose which is sufficiently vivid and pressing. The ancient magicians, in short, had the psychology of their age when they sought to influence the forces of Nature; but they lacked the cosmology of ours, when we know that there is no such thing as "Nature" apart from the human mind and brain.

...

## Long Inference

To speed up inference (or to share the model), you can convert the Fg model back to HF format.

```bash
scripts/convert_llama.py --reverse --model-type mistral --dtype bfloat16 --max-length 32000 \
/home/dinalt/rust/models/fg_mistral /home/dinalt/rust/models/hf_lovecraft_mistral
```

Start the inference server on the converted model:

```bash
tools/inference_server/server.py -d "cuda:0" -T bfloat16 \
-c -m /home/dinalt/ai_assets/models/hf_lovecraft_mistral
```

And test long-context inference, beyond what we trained at (8192 tokens).

```bash
./tools/inference_server/client.py --stream --completion \
"Of such great powers or beings there may be conceivably a survival" --max-tokens 8192 | tee lovecraftian.txt
```

This will both stream the output and save it to "lovecraftian.txt."