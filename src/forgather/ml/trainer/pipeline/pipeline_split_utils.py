"""
Pipeline parallel model splitting utilities.

Based on TorchTitan's pipeline_module_split approach.
"""

import copy
from typing import List
from torch import nn


def generate_llm_fqn_per_model_part(
    num_stages: int,
    num_layers: int,
    input_weight: int = 1,
    output_weight: int = 1,
) -> List[List[str]]:
    """
    Programmatically generates module names per model part for LLM models.

    This function distributes the model components across pipeline stages,
    treating the input encoder and output decoder with configurable weights
    to account for their computational cost relative to transformer layers.

    Args:
        num_stages: Number of pipeline stages
        num_layers: Total number of transformer layers in the model
        input_weight: Weight for input modules (input_encoder) in layer calculation
        output_weight: Weight for output modules (layer_norm + output_decoder) in layer calculation

    Returns:
        List of lists containing module names for each model part.
        Module names use the attribute names from CasualLM:
        - "input_encoder" for token embeddings + positional encoding
        - "layer_stack.layers.0", "layer_stack.layers.1", ... for transformer layers
        - "layer_stack.layer_norm" for final normalization
        - "output_decoder" for the output projection layer

    Example:
        generate_llm_fqn_per_model_part(2, 4, input_weight=1, output_weight=1)
        might return:
        [
            ["input_encoder", "layer_stack.layers.0", "layer_stack.layers.1"],
            ["layer_stack.layers.2", "layer_stack.layers.3", "layer_stack.layer_norm", "output_decoder"]
        ]
    """
    if num_stages < 1:
        raise ValueError("Number of stages must be at least 1")

    if num_stages == 1:
        # Single stage gets everything
        layer_names = [f"layer_stack.layers.{i}" for i in range(num_layers)]
        return [
            ["input_encoder"]
            + layer_names
            + ["layer_stack.layer_norm", "output_decoder"]
        ]

    # Calculate effective layers including weights
    num_effective_layers = num_layers + input_weight + output_weight

    if num_stages > num_effective_layers:
        raise ValueError(
            f"Number of stages ({num_stages}) cannot be greater than effective layers ({num_effective_layers})"
        )

    # Calculate layers per stage (distribute evenly)
    layers_per_stage = num_effective_layers // num_stages
    extra_layers = num_effective_layers % num_stages

    # Feasibility check: Ensure at least 1 layer in each PP stage
    if layers_per_stage == 0:
        raise ValueError(
            f"Configuration would result in empty stages. "
            f"With {num_stages} stages and {num_effective_layers} effective layers "
            f"(num_layers={num_layers} + input_weight={input_weight} + output_weight={output_weight}), "
            f"each stage would get {layers_per_stage} layers on average. "
            f"Reduce num_stages or increase num_layers/weights."
        )

    module_names_per_stage = []
    current_layer = 0

    for stage_idx in range(num_stages):
        stage_modules = []

        # Calculate effective layers for this stage
        effective_layers_for_stage = layers_per_stage
        if stage_idx < extra_layers:
            effective_layers_for_stage += 1

        # First stage: handle input modules with weighting
        if stage_idx == 0:
            stage_modules.append("input_encoder")
            # Account for input weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - input_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layer_stack.layers.{current_layer}")
                    current_layer += 1

        # Last stage: handle output modules with weighting
        elif stage_idx == num_stages - 1:
            # Account for output weight in layer distribution
            remaining_layers_for_stage = effective_layers_for_stage - output_weight

            # Add transformer layers
            for _ in range(remaining_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layer_stack.layers.{current_layer}")
                    current_layer += 1

            # Add output modules
            stage_modules.extend(["layer_stack.layer_norm", "output_decoder"])

        # Middle stages: only transformer layers
        else:
            for _ in range(effective_layers_for_stage):
                if current_layer < num_layers:
                    stage_modules.append(f"layer_stack.layers.{current_layer}")
                    current_layer += 1

        module_names_per_stage.append(stage_modules)

    return module_names_per_stage


def split_model(model: nn.Module, module_names: List[str]) -> None:
    """
    Splits a CasualLM model by deleting modules not in the provided list.

    This function modifies the model in-place, deleting layers and setting
    top-level modules to None as appropriate for a pipeline stage.

    The function handles the CasualLM structure:
    - model.causal_lm.input_encoder
    - model.causal_lm.layer_stack (contains layers ModuleDict and layer_norm)
    - model.causal_lm.output_decoder

    Args:
        model: The model to split (typically DynamicCasualLM wrapping CasualLM)
        module_names: List of module names to keep in this stage.
                     Examples: "input_encoder", "layer_stack.layers.0", "output_decoder"

    Returns:
        None (modifies model in-place)

    Example:
        # Keep only layers 0-1 and input encoder
        split_model(model, ["input_encoder", "layer_stack.layers.0", "layer_stack.layers.1"])
    """
    # Access the actual CasualLM model
    causal_lm = model.model

    # Create a set of modules to keep for faster lookup
    modules_to_keep = set(module_names)

    # Handle input_encoder
    if "input_encoder" not in modules_to_keep:
        causal_lm.input_encoder = None

    # Handle output_decoder
    if "output_decoder" not in modules_to_keep:
        causal_lm.output_decoder = None

    # Handle layer_stack - this is more complex
    if causal_lm.layer_stack is not None:
        layer_stack = causal_lm.layer_stack

        # Check which layers to keep
        layers_to_keep = {
            name.split(".", 2)[2]  # "layer_stack.layers.0" -> "0"
            for name in modules_to_keep
            if name.startswith("layer_stack.layers.")
        }

        # Delete layers not in the keep set
        if layers_to_keep:
            for layer_name in list(layer_stack.layers.keys()):
                if layer_name not in layers_to_keep:
                    del layer_stack.layers[layer_name]
        else:
            # No layers to keep, empty the ModuleDict
            layer_stack.layers = nn.ModuleDict()

        # Handle layer_norm
        if "layer_stack.layer_norm" not in modules_to_keep:
            layer_stack.layer_norm = None

        # If the entire layer_stack is empty, set it to None
        if len(layer_stack.layers) == 0 and layer_stack.layer_norm is None:
            causal_lm.layer_stack = None
