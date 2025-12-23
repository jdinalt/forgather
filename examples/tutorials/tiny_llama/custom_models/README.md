# Custom Models

This is an example of how to use an existing model project as a base and add new, derived, configurations from it.

In this case, we are using "[examples/models/llama](../../../models/llama/)" as the base project. To make this work, we need to be able to find this project directory from
projects defined in this directory.

- [base_directories.yaml](./forgather_workspace/base_directories.yaml) : This just defines where the base of the Forgather directory is located. We will use this as an anchor-point for finding the parent project.
- [meta_defaults.yaml](./forgather_workspace/meta_defaults.yaml) : This should include the same search configuration search paths as our parent.

We then can create a directory for each sub-project. In this case, just "llama."

See the [README.md](./llama/README.md) in that directory for further details.