# Forgather

Forgather is a configuration-driven ML framework, built on template inheritance and code generation, designed to simplify the creation and maintainance of complex ML experiment configurations.

## What problems does it solve?

Forgather was created primarily to address work-flow issues encountered with my own ML experiments.

- Configuration Explosion: Every new configuration starts as a complete copy of a previous. Before long, everythig is a copy-of-a-copy-of-a-copy. Propagating bug fixes and features accross a myriad of branches becomes an ever growing workload.
- "Types" as Hyper-parameters: If I want to change change the optimizer from AdamW to SomeCoolNewOptimizer, I can't do it in the configuration file -- at least if there is not allready a special-case for handling "SomeCoolNewOptimizer" in the configuration parser. I should be able to change this in my configuraiton file as easily as I can change the learning-rate, without having to special-case it.
- Dynamic Model Architecture: I would like to be able to specify (and change) the model architecture directly in the configuration file with only a few lines of code.
- Modular Components Library: I often find that I would like to perform an experiment requires either modifying something from an existing library or writing a new implementation from scratch. Often, the library implementation is dauntingly complex, and writing it from scratch is less trouble, but is still a fair amount of work. I would be nice to have a colleciton of sufficiently simple "hackable" modular parts, which fit together nicely and are compatibly with 3rd party libraries.
- Reproducibility: It can be very frustrating when you find that you can't reproduce the results of a prior experiment because "something" has chaged, but it is not obvious what that "something" is. This is easy to do when quickly iterating over hyper-parameters by directly modifying the configuration file.

## Solutions

### Configuration Explosion

Forgather's approach is to provide a configuration definition language which is composed of YAML and Jinja2. By using template inheritance, we can create new configurations by inheriting from existing ones and by specifying only how they differ from the originals. When new functionality is added to a base configuration or a bug is fixed, it is automatically propagated throughout the template hierarchy.

For example, if we wished to take an existing configuration and change a single hyper-parameter in the configuration, we could do so like this:

```yaml
-- extends 'base_configuration.yaml'

-- block optimizer
    == super()
    # Experiment overrides
    lr: 1.0e-3
-- endblock optimizer
```

### "Types" as Hyper-parameters

YAML already has support for user-defined type tags. Forgather uses this feature to define tags for dynamic type imports. For example, we can define a partial function for constructing an AdamW optimizer like this:

```yaml
.define: &optimizer !partial:torch:optim.AdamW
    lr: 1.0e-3
```

This definition can then be used as an argument to another dynamic type:

```yaml
trainer: !singleton:forgather.ml.trainer:Trainer
    ...
    optimizer_factory: *optimizer
```

### Dynamic Model Architecture

A model architecture definition can be specified as a a collection of modular building blocks. For example, if we had an existing model definition and wished to replace the positional encoder with a custom one:

```yaml
-- extends 'my_transformer_model.yaml'

-- block positional_encoder_factory
# Replace standard Sinosoidal PE with Walsh PE.
positional_encoder_factory: &positional_encoder_factory !partial:.walsh_pe:WalshPE
    d_model: !var "hidden_size"
    max_sequence_length: !var "max_sequence_length"
-- endblock positional_encoder_factory
```

This new definition can then be used to automatically generate the Python code for constructing the model. The generated code and the referenced source files  will then be copied into the output directory, resulting in stand-alone model, without any Forgather dependencies.

### Modular Components Library

#### Dynamic Transformer Modules

The model library contains a collection of interchangeable Transformer language modules which can be easily composed into different model architectures.

#### Trainers

Forgather offers a collection of "Trainer" classes which implement a subset of the API of the Huggingface Trainer class. This makes it possible to quickly substitute a custom Trainer in your configuration without needed to make substantial changes to the configuraiton. They are also compatible with the HF Trainer add-on API, which allows them to share extensions. The base implementation is much ligher than the HF Trainer, making them much easier to understand and modify.

This includes a fast and simple Trainer, a multi-GPU variant based upon the Accelerate library, and a Pipeline Parallel implementation, which makes use of the latest [PyTorch Pipleline Parallelism APIs](https://docs.pytorch.org/docs/stable/distributed.pipelining.html).

#### Optimizers

A collection of optimizer implementaitons for further customization and experimentation.

- AdamW
- SGD
- AdaFactor
- [GaLore](https://arxiv.org/abs/2403.03507)
- [Apollo](https://arxiv.org/pdf/2412.05270)

### Reproducibility

To aid with reproducibility, the library automatically saves the generated configuration file with the logs and a snapshot of the model's source code. This ensures that even if you are directly modifying the configuration on each trial, you will have a record of the configuraiton used.

By saving the model's implementation with the weights, it is assured that you will not break a trained model when the model library is updated.

## Getting Started

Clone the repo and run:

```
# From the installation directory.
pip install -e .
```

The see the "examples/tutorials" directory.
