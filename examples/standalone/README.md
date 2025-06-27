# Standalone Examples

This directory contains self-contained Forgather examples that demonstrate specific patterns and use cases.

## Projects

### external_workspace_example/
Demonstrates how to create a completely separate Forgather workspace with its own project structure. This example shows how to set up an independent workspace that can be used for custom model development outside the main Forgather repository.

## Usage

Each project contains its own configuration and can be run independently:

```bash
cd external_workspace_example/alibi_glu_project
fgcli.py ls
fgcli.py -t train_alibi_glu.yaml train
```