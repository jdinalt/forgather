"""
vLLM Plan Validation Utility

This module provides utilities to validate that vLLM tensor parallel and pipeline parallel
plans match the actual model structure. This helps catch configuration errors early.
"""

import re
from typing import Dict, List, Tuple, Optional, Set
from torch import nn
import logging

logger = logging.getLogger(__name__)


def validate_tp_plan(
    model: nn.Module,
    tp_plan: Dict[str, str],
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that tensor parallel plan patterns match actual model parameters.

    Args:
        model: The PyTorch model to validate against
        tp_plan: Dictionary mapping FQN patterns to parallelism styles ("colwise" or "rowwise")
        strict: If True, raises detailed warnings for unmatched patterns

    Returns:
        Tuple of (is_valid, list of validation messages)

    Example:
        >>> tp_plan = {
        ...     "model.layer_stack.layers.*.attention.query_linear": "colwise",
        ...     "model.layer_stack.layers.*.attention.output_linear": "rowwise",
        ... }
        >>> is_valid, messages = validate_tp_plan(model, tp_plan)
        >>> if not is_valid:
        ...     for msg in messages:
        ...         print(msg)
    """
    messages = []
    is_valid = True

    # Get all parameter names from the model
    param_names = set(name for name, _ in model.named_parameters())

    # Track which patterns matched at least one parameter
    pattern_matches: Dict[str, Set[str]] = {pattern: set() for pattern in tp_plan.keys()}

    # Convert glob patterns to regex patterns
    for pattern, style in tp_plan.items():
        # Validate style
        if style not in ("colwise", "rowwise"):
            is_valid = False
            messages.append(f"Invalid parallelism style '{style}' for pattern '{pattern}'. Must be 'colwise' or 'rowwise'.")
            continue

        # Convert glob pattern to regex
        # Replace * with [^.]+ to match any characters except dots (layer boundaries)
        regex_pattern = pattern.replace(".", r"\.").replace("*", r"[^.]+")
        regex_pattern = f"^{regex_pattern}$"

        try:
            compiled_pattern = re.compile(regex_pattern)
        except re.error as e:
            is_valid = False
            messages.append(f"Invalid regex pattern '{pattern}': {e}")
            continue

        # Find matching parameters
        for param_name in param_names:
            if compiled_pattern.match(param_name):
                pattern_matches[pattern].add(param_name)

    # Check for patterns that didn't match anything
    unmatched_patterns = [p for p, matches in pattern_matches.items() if not matches]
    if unmatched_patterns:
        is_valid = False
        messages.append(f"The following TP plan patterns did not match any model parameters:")
        for pattern in unmatched_patterns:
            messages.append(f"  - {pattern}")
            # Try to provide helpful suggestions
            similar_params = _find_similar_params(pattern, param_names)
            if similar_params:
                messages.append(f"    Did you mean one of these?")
                for param in similar_params[:3]:  # Show top 3 suggestions
                    messages.append(f"      - {param}")

    # In strict mode, log all matches for verification
    if strict and messages:
        messages.append("\nMatched parameters:")
        for pattern, matches in pattern_matches.items():
            if matches:
                messages.append(f"  Pattern '{pattern}' ({tp_plan[pattern]}):")
                for match in sorted(matches)[:5]:  # Show first 5 matches
                    messages.append(f"    - {match}")
                if len(matches) > 5:
                    messages.append(f"    ... and {len(matches) - 5} more")

    return is_valid, messages


def validate_pp_plan(
    model: nn.Module,
    pp_plan: Dict[str, Tuple[List[str], List[str]]],
    strict: bool = True
) -> Tuple[bool, List[str]]:
    """
    Validate that pipeline parallel plan module names exist in the model.

    Args:
        model: The PyTorch model to validate against
        pp_plan: Dictionary mapping module names to (input_names, output_names) tuples
        strict: If True, raises detailed warnings for issues

    Returns:
        Tuple of (is_valid, list of validation messages)

    Example:
        >>> pp_plan = {
        ...     "model.input_encoder": (["input_ids"], ["hidden_states"]),
        ...     "model.layer_stack": (["hidden_states", "attention_mask"], ["hidden_states"]),
        ... }
        >>> is_valid, messages = validate_pp_plan(model, pp_plan)
    """
    messages = []
    is_valid = True

    # Get all module names from the model
    module_names = set(name for name, _ in model.named_modules())

    # Check each module in the plan
    for module_name, io_spec in pp_plan.items():
        if module_name not in module_names:
            is_valid = False
            messages.append(f"PP plan module '{module_name}' not found in model")

            # Try to provide helpful suggestions
            similar_modules = _find_similar_params(module_name, module_names)
            if similar_modules:
                messages.append(f"  Did you mean one of these?")
                for mod in similar_modules[:3]:
                    messages.append(f"    - {mod}")

        # Validate I/O specification format
        if not isinstance(io_spec, (list, tuple)) or len(io_spec) != 2:
            is_valid = False
            messages.append(f"PP plan for '{module_name}' must be a tuple of (input_names, output_names)")
            continue

        inputs, outputs = io_spec
        if not isinstance(inputs, list) or not isinstance(outputs, list):
            is_valid = False
            messages.append(f"PP plan I/O for '{module_name}' must be lists of strings")

    # In strict mode, show all validated modules
    if strict:
        messages.append("\nValidated PP plan modules:")
        for module_name, (inputs, outputs) in pp_plan.items():
            if module_name in module_names:
                messages.append(f"  {module_name}")
                messages.append(f"    Inputs:  {inputs}")
                messages.append(f"    Outputs: {outputs}")

    return is_valid, messages


def validate_vllm_plans(
    model: nn.Module,
    tp_plan: Optional[Dict[str, str]] = None,
    pp_plan: Optional[Dict[str, Tuple[List[str], List[str]]]] = None,
    strict: bool = False
) -> bool:
    """
    Validate both tensor parallel and pipeline parallel plans.

    Args:
        model: The PyTorch model to validate against
        tp_plan: Tensor parallel plan (optional)
        pp_plan: Pipeline parallel plan (optional)
        strict: If True, logs detailed information even when valid

    Returns:
        True if all provided plans are valid

    Example:
        >>> # After model generation
        >>> model = DynamicCasualLM.from_pretrained("path/to/model")
        >>> if hasattr(model, '_tp_plan') and model._tp_plan:
        ...     validate_vllm_plans(model, tp_plan=model._tp_plan, pp_plan=model._pp_plan)
    """
    all_valid = True

    if tp_plan:
        logger.info("Validating vLLM tensor parallel plan...")
        tp_valid, tp_messages = validate_tp_plan(model, tp_plan, strict=strict)
        if not tp_valid or (strict and tp_messages):
            for msg in tp_messages:
                if "did not match" in msg.lower() or "invalid" in msg.lower():
                    logger.warning(msg)
                else:
                    logger.info(msg)
        if tp_valid:
            logger.info(f"✓ Tensor parallel plan validated ({len(tp_plan)} patterns)")
        all_valid = all_valid and tp_valid

    if pp_plan:
        logger.info("Validating vLLM pipeline parallel plan...")
        pp_valid, pp_messages = validate_pp_plan(model, pp_plan, strict=strict)
        if not pp_valid or (strict and pp_messages):
            for msg in pp_messages:
                if "not found" in msg.lower() or "invalid" in msg.lower():
                    logger.warning(msg)
                else:
                    logger.info(msg)
        if pp_valid:
            logger.info(f"✓ Pipeline parallel plan validated ({len(pp_plan)} modules)")
        all_valid = all_valid and pp_valid

    return all_valid


def _find_similar_params(pattern: str, param_names: Set[str], max_results: int = 5) -> List[str]:
    """
    Find parameter names similar to the given pattern.
    Uses simple substring matching and common prefixes.
    """
    # Extract key terms from pattern (words separated by dots)
    pattern_parts = pattern.replace("*", "").split(".")
    pattern_parts = [p for p in pattern_parts if p]  # Remove empty strings

    # Score each parameter name
    scored_params = []
    for param in param_names:
        param_parts = param.split(".")
        score = 0

        # Award points for matching parts
        for pattern_part in pattern_parts:
            if any(pattern_part in param_part for param_part in param_parts):
                score += 1

        # Award points for matching structure (number of dots)
        pattern_depth = pattern.count(".")
        param_depth = param.count(".")
        if abs(pattern_depth - param_depth) <= 1:
            score += 0.5

        if score > 0:
            scored_params.append((score, param))

    # Sort by score and return top results
    scored_params.sort(reverse=True, key=lambda x: x[0])
    return [param for _, param in scored_params[:max_results]]


def print_model_structure(model: nn.Module, max_depth: int = 3, show_params: bool = True):
    """
    Print model structure to help with creating vLLM plans.

    Args:
        model: The PyTorch model
        max_depth: Maximum depth to print (default 3)
        show_params: Whether to show parameter names (default True)

    Example:
        >>> model = DynamicCasualLM.from_pretrained("path/to/model")
        >>> print_model_structure(model, max_depth=4)
    """
    print("\n=== Model Structure ===")

    # Print module hierarchy
    print("\nModules:")
    for name, module in model.named_modules():
        depth = name.count(".")
        if depth <= max_depth:
            indent = "  " * depth
            module_type = type(module).__name__
            print(f"{indent}{name or 'root'} ({module_type})")

    # Print parameters if requested
    if show_params:
        print("\nParameters (for TP plan):")
        param_groups = {}
        for name, param in model.named_parameters():
            depth = name.count(".")
            if depth <= max_depth + 1:  # Show one level deeper for params
                # Group by parent module
                parts = name.rsplit(".", 1)
                parent = parts[0] if len(parts) > 1 else "root"
                if parent not in param_groups:
                    param_groups[parent] = []
                param_groups[parent].append(name)

        for parent in sorted(param_groups.keys()):
            depth = parent.count(".")
            if depth <= max_depth:
                indent = "  " * depth
                print(f"{indent}{parent}:")
                for param in param_groups[parent]:
                    print(f"{indent}  - {param}")
