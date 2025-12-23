# Custom Llama Models

This directory defines models derived from the architecture defined in the base Llama project. As to allow us to extend the definitions from the
[base Llama project](../../../../models/llama/), we add that project to our own search path in the project's [meta](./meta.yaml) file. The result is
that all configurations defined in that project are visible to our project. We can then derive new configurations from the existing ones.

For each custom model architecture definition, there should be a configuration file. This project includes just a single example:

- [configs/post_layer_norm.yaml](./templates/configs/post_layer_norm.yaml) : A Llama using Post Layer Norm layers

While it is possible to directly inline a new model definition into a training configuration, this approach is much more modular and makes
it much easier to test the new model architecture.

Because this project is defined as a union with another project, we can see both our configurations and those from the base project:

```bash
forgather ls
llama : Llama models
    117M.yaml                      Llama 117M : A Llama with 117M parameters
    4M.yaml                        Tiny Llama 4M : A tiny Llama with 4M parameters
    default.yaml                   Default Llama : Meta Llama
    llama2_7b.yaml                 Llama2 7B : Meta Llama2 7B
    llama3.2_1b.yaml               Llama3.2 1B : Meta Llama3.2 1B
    [post_layer_norm.yaml]         Post Layer Norm Llama : A tiny llama using Post Layer Norm
```

Note that the default has been changed to our newly defined configuration, `post_layer_norm.yaml`.

```bash
# Show model definition
forgather -t CONFIG_NAME pp
```

You can construct a instance of the new model definition like this:

```bash
# Construct model instance on meta device
forgather model construct

# Inspect the generated model definition's files
ls output_models/post_layer_norm_llama/
```

Note that this does not save any weights, but as it is a HF model, a new instance, with initialized weights, could be constructed from the definition like this:

```python
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

config = AutoConfig.from_pretrained('./output_models/post_layer_norm_llama', trust_remote_code=True)
model = AutoModelForCausalLM.from_config(config)
tokenizer = AutoTokenizer.from_pretrained('./output_models/post_layer_norm_llama')
```

When defining a custom architecture, it's a good idea to test the model with a forward and backward pass to see if there are any issues, before training it.

```bash
# Give model a "kick" test to make sure it does not fall over when actually used.
forgather model --device cpu test
...
Setting learning-rate=0.01
batch.keys()=dict_keys(['input_ids', 'labels'])
step: 1, loss: 7.780828475952148, logits.shape: torch.Size([2, 512, 2000])
Test Completed
```

While this is no guarantee that it will work well, it should at least be possible to try to train it.

To see how to use this model in a project, refer to the tutorial [experimental_llama.yaml](../../templates/configs/experimental_llama.yaml)

To train this model, from the tutorial directory, run:

```bash
forgather -t experimental_llama.yaml train
```