# Tiny Titan

This my first shot at creating a native Torch Titain trainer, which uses dependency injection for most the majority of the training assets. This is opposed to the torchtitan approach of giving the passing a configuration into the trainer and having it construct all of the assets internally from the configuration.

**Update**

We now have basic templates for creating Forgather Titan trainer projects, with a couple of basic examples.

- default.yaml : This recreates the Tiny Llama tutorial using Torch Titain
- bigger.yaml : The switches to a 117M parameter Llama3 model, an 8K tokenizer, and trains on FineWeb-Edu-Dedup using DDP x 4 GPUs

## Why?

The dependency injection approach has a number of advantages. The biggest is that the coupling between components is much looser, in that nothing requires explicitly defined support, as long as the interfaces are compatible. For example, I can inject any optimizer I want, without ever having to add explcit support for it in torchtitan. There's no longer any need for pre-registering components to be able to use them. You just construct the object and inject it into the trainer. Being able to do this through configuration files was one of the major driving forces for writing Forgather in the first place.

This is also an advantage, even outside of the context of Forgather. If I am constructing objects for my trainer in native Python, why should I not be able to just pass in a constructor for a model or an optimizer? Why can't I just build the dataloader myself and pass it into the trainer? I don't need the trainer to do these things for me. It just gets in the way and make the task more difficult. Doubly so, when you want to construct an object which does not have support. Say for example, constructing an Adafactor optimizer.