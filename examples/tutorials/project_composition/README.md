# Project Composition

This example demostrates the use of [Jinja2 template inheritance](https://jinja.palletsprojects.com/en/3.1.x/templates/#template-inheritance).

---

In addition to YAML, there is a Jinja2 preprocessing stage which allows for things like macros and template inheritance. This can help eliminate unnecessary repition by factoring out the common elements in a set of configurtions.

In this example, we define a base-template ("list_base.yaml") for defining a list and extend the definition for the first configuration, "list.yaml." In the second configuration, "full_list.yaml," we extend the definition of "list.yaml."

We use a list to keep this example simple, but this pattern is used extensively in the main Forgather template library for much more complex use-cases.

## Project Setup

The project meta-config is much the same as the first example project, although we only specify the default config this time, as the other defaults will work.

## Configurations

Under "Available Configurations," there are two configs listed:
- list.yaml : A short list, derived from base_list.yaml
- full_list.yaml : Alonger list, derived from list.yaml

## Included Templates

Note the hierarchical template listing for the selected configuration. You can examine the referenced templates by clicking on the links in the index.

---

