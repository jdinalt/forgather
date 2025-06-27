# fgcli.py -- Forgather Command Line Interface

```
usage: fgcli.py [-h] [-p PROJECT_DIR] [-t CONFIG_TEMPLATE] {index,ls,pp,trefs,tlist,meta,targets,code,construct,graph,tb,train} ...

Forgather CLI

positional arguments:
  {index,ls,pp,trefs,tlist,meta,targets,code,construct,graph,tb,train}
                        subcommand help
    index               Show project index
    ls                  List available configurations
    pp                  Preprocess configuration
    trefs               List referenced templates
    tlist               List available templates.
    meta                Show meta configuration
    targets             Show output targets
    code                Output configuration as Python code
    construct           Materialize and print a target
    graph               Preprocess and parse into node graph
    tb                  Start Tensorboard for project
    train               Run configuration with train script

options:
  -h, --help            show this help message and exit
  -p PROJECT_DIR, --project-dir PROJECT_DIR
                        The relative path to the project directory.
  -t CONFIG_TEMPLATE, --config-template CONFIG_TEMPLATE
                        Configuration Template Name
```


For examples, assume working directory is "forgather/examples/tiny_experiments/tiny_models"

---
### index : Show project index

```
fgcli.py index
## Tiny Models

A collection of tiny models to train on the Tiny Stories dataset with the tiny_stories_2k tokenizer.

This allows for direct comparison of model archetectures.
...
```
---
### ls : Project List

Lists projects and available configurations.

```
usage: fgcli.py ls [-h] [--recursive]

options:
  -h, --help       show this help message and exit
  --recursive, -r  Search for projects in all sub-directories and list them.
``` 

```
fgcli.py ls
Tiny Models : Train and compare performance of different small language model architectures
    [tiny_causal.yaml]             Tiny Causal : A tiny causal transformer.
    tiny_gpt2.yaml                 Tiny GPT2 : A tiny GPT2 model.
    tiny_llama.yaml                Tiny LLama : A tiny llama model.
```

Note that the default configuration is shown between square-brackets []

As this requires preprocessing and parsing (YAML) all of the configurations, this is also very useful for finding
configuration syntax errors. If such an error occurs, the configuration's name will be shown as "SYNTAX ERROR." This
should be run before checking in changes to catch such errors.

---
### pp : Preprocess a configuration

Dump the preprocessed configuration of the specified configuration to stdout.

```
fgcli.py -t tiny_llama.yaml pp
#---------------------------------------
#               Tiny LLama               
#---------------------------------------
# 2025-06-21T22:48:57
# Description: A tiny llama model.
...
```
---
### trefs : List templates used by the configuration.

This lists the template hierarchy, with the specified configuration at the root.

TEMPLATE_NAME : TEMPLATE_PATH
TEMPLATE_NAME : The name of the template, as viewed from the template namespace.
TEMPLATE_PATH : The path to the template in the filesystem.

```
fgcli.py -t tiny_llama.yaml trefs
 configs/tiny_llama.yaml : templates/configs/tiny_llama.yaml
     project.yaml : templates/project.yaml
         projects/tiny.yaml : ../tiny_templates/projects/tiny.yaml
             prompts/tiny_stories.yaml : ../tiny_templates/prompts/tiny_stories.yaml
             types/training_script/causal_lm/causal_lm.yaml : ../../../templatelib/base/types/training_script/causal_lm/causal_lm.yaml
                 trainers/trainer.yaml : ../../../templatelib/base/trainers/trainer.yaml
...
```
### tlist : List all available templates

This list all templates available in the search-path and shows the inheritance hierarchy.

```
usage: fgcli.py tlist [-h] [--format {md,files}]

options:
  -h, --help           show this help message and exit
  --format {md,files}  Output format.
```

md : Show as markdown
files : List file paths to templates.

```
fgcli.py tlist
templates/project.yaml
templates/configs/tiny_causal.yaml
templates/configs/tiny_gpt2.yaml
templates/configs/tiny_llama.yaml
../forgather_workspace/base_directories.yaml
../forgather_workspace/meta_defaults.yaml
...

# Search for the word "extends" in all templates.
fgcli.py tlist | xargs grep extends
templates/project.yaml:-- extends 'projects/tiny.yaml'
templates/project.yaml:-- extends 'tiny.trainer_config'
templates/configs/tiny_causal.yaml:-- extends 'project.yaml'
templates/configs/tiny_causal.yaml:-- extends 'models/tiny/tiny_causal.yaml'
templates/configs/tiny_gpt2.yaml:-- extends 'project.yaml'
...

fgcli.py tlist --format md
- [tiny.trainer_config](../forgather_workspace/projects/tiny.yaml)
    - [project.trainer_config](templates/project.yaml)
- [prompts/tiny_stories.yaml](../../../templatelib/examples/prompts/tiny_stories.yaml)
- [trainers/minimal_trainer.yaml](../../../templatelib/base/trainers/minimal_trainer.yaml)
    - [trainers/base_trainer.yaml](../../../templatelib/base/trainers/base_trainer.yaml)
        - [trainers/trainer.yaml](../../../templatelib/base/trainers/trainer.yaml)
            - [trainers/accel_trainer.yaml](../../../templatelib/base/trainers/accel_trainer.yaml)
            - [trainers/pipeline_trainer.yaml](../../../templatelib/base/trainers/pipeline_trainer.yaml)
            - [trainers/hf_trainer.yaml](../../../templatelib/base/trainers/hf_trainer.yaml)
...
```
---
### targets : Show available targets in the configuration

```
fgcli.py -t tiny_llama.yaml targets
distributed_env
testprompts
generation_config
model_constructor_args
tokenizer
model_config
model
...
```
---
### code : Generate Python code for the specified target

```
fgcli.py -t tiny_llama.yaml code --target distributed_env
from forgather.ml.distributed import DistributedEnvironment

def construct(
):
    distributed_env = DistributedEnvironment()
    
    return distributed_env
```
---
### construct : Materialize a specified target

This will attempt to construct the specified target, which can be helful when trying to debug a configuration.

```
fgcli.py -t tiny_llama.yaml construct --target distributed_env
DistributedEnvironment(rank=0, local_rank=0, world_size=1, local_world_size=1, master_addr=localhost, master_port=29501, backend=None)
```
---
### graph : Construct and display the node graph

```
usage: fgcli.py graph [-h] [--format {none,repr,yaml,fconfig,python}]

options:
  -h, --help            show this help message and exit
  --format {none,repr,yaml,fconfig,python}
                        Graph format
```

Format Options:
 - none: Just silently construct the graph. This just verifies that there are no syntax errors present.
 - repr: Show the graph using repr(). This is very verbose.
 - yaml: Convert the graph to yaml and dump it
 - fconfig : Format using internal fconfig() call.
 - python : Convert entire graph to Python code.

The 'yaml' target can be useful for more easily determining what the actual configuration is, when there are
many overrides.

---
### tb : Start Tensorboard for project output directory.

```
usage: fgcli.py tb [-h] [--all] [--dry-run] ...

positional arguments:
  remainder   All arguments after -- will be forwarded as Tensroboard arguments.

options:
  -h, --help  show this help message and exit
  --all       Configure TB to watch all model directories
  --dry-run   Just show the generated commandline, without actually executing it.
```

```
# To bind only to localhost
fgcli.py -t tiny_llama.yaml tb

# To bind to all interfaces
fgcli.py -t tiny_llama.yaml tb -- --bind_all

# To bind to all interfaces and show all models in models directory
fgcli.py -t tiny_llama.yaml tb --all -- --bind_all
```

---
### train : Use torchrun to run the project with the training script.

```
usage: fgcli.py train [-h] [-d DEVICES] [--dry-run] ...

positional arguments:
  remainder             All arguments after -- will be forwarded as torchrun arguments.

options:
  -h, --help            show this help message and exit
  -d DEVICES, --devices DEVICES
                        CUDA Visible Devices e.g. "0,1"
  --dry-run             Just show the generated commandline, without actually executing it.
```

```
# Run training on the default target.
fgcli.py train

# Only run on GPU 0 and just print the command, without executing it.
fgcli.py -t tiny_llama.yaml train -d 0 --dry-run
CUDA_VISIBLE_DEVICES="0" torchrun --standalone --nproc-per-node gpu /path/to/forgather/scripts/train_script.py -p . tiny_llama.yaml
```
