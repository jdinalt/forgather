#!/usr/bin/env python3
"""
Checkpoint Parameter Modification Tool

Safely modify optimizer, scheduler, and other component parameters in Forgather checkpoints.
Supports atomic file operations with validation to prevent corruption.
"""

import argparse
import ast
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch


def discover_checkpoint_files(checkpoint_path: str, component: str) -> List[str]:
    """
    Discover all checkpoint files for the component.

    Handles Forgather file naming patterns:
    - component_state.pt (single file or GLOBAL/REPLICATED)
    - component_state_rank_0.pt, component_state_rank_1.pt (PER_RANK)
    - component_state_group_*_grank_*_rank_*.pt (PER_GROUP)
    - component_state_node_*_rank_*.pt (PER_NODE)

    Returns list of absolute paths to all files that need modification.
    """
    checkpoint_path = Path(checkpoint_path).resolve()

    # If path is a file, use it directly
    if checkpoint_path.is_file():
        return [str(checkpoint_path)]

    # If it's a directory, search for component files
    if not checkpoint_path.is_dir():
        raise FileNotFoundError(f"Checkpoint path not found: {checkpoint_path}")

    # Try to use manifest if available
    manifest_path = checkpoint_path / "checkpoint_manifest.json"
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                manifest = json.load(f)

            component_info = manifest.get("components", {}).get(component)
            if component_info and "ranks" in component_info:
                # Use manifest to determine files
                pattern = component_info.get("sharing_pattern", "unknown")
                files = []

                if pattern in ["global", "replicated"]:
                    # Single file
                    file_path = checkpoint_path / f"{component}_state.pt"
                    if file_path.exists():
                        files.append(str(file_path))
                elif pattern == "per_rank":
                    # Per-rank files
                    for rank in component_info["ranks"]:
                        file_path = checkpoint_path / f"{component}_state_rank_{rank}.pt"
                        if file_path.exists():
                            files.append(str(file_path))

                if files:
                    return files
        except (json.JSONDecodeError, KeyError):
            # Fall through to glob-based discovery
            pass

    # Fall back to glob-based discovery
    patterns = [
        f"{component}_state.pt",
        f"{component}_state_rank_*.pt",
        f"{component}_state_group_*_grank_*_rank_*.pt",
        f"{component}_state_node_*_rank_*.pt",
    ]

    files = []
    for pattern in patterns:
        files.extend(checkpoint_path.glob(pattern))

    if not files:
        raise FileNotFoundError(
            f"No {component} checkpoint files found in {checkpoint_path}\n"
            f"Looking for patterns: {', '.join(patterns)}"
        )

    return sorted(str(f) for f in files)


def list_modifiable_parameters(state_dict: dict, component: str) -> dict:
    """
    Discover non-tensor values that can be modified.

    Args:
        state_dict: Loaded checkpoint component state dict
        component: Component type ("optimizer", "scheduler", etc.)

    Returns:
        Dictionary of modifiable parameters with structure:
        {
            "param_groups": [
                {"lr": (float, 0.001), "weight_decay": (float, 0.0), ...},
                ...
            ],
            # For scheduler:
            "state": {"last_epoch": (int, 100), "_step_count": (int, 100)},
        }
    """
    modifiable = {}

    if component == "optimizer":
        # Extract param_groups
        param_groups = state_dict.get("param_groups", [])
        modifiable["param_groups"] = []

        for pg in param_groups:
            pg_params = {}
            for key, value in pg.items():
                # Skip 'params' key (list of parameter indices)
                if key == "params":
                    continue
                # Only include non-tensor values
                if not isinstance(value, torch.Tensor):
                    pg_params[key] = (type(value).__name__, value)
            modifiable["param_groups"].append(pg_params)

    elif component == "scheduler":
        # Extract scheduler state (non-tensor values)
        state = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor) and key != "param_groups":
                state[key] = (type(value).__name__, value)
        modifiable["state"] = state

    else:
        # Generic component - list all non-tensor values
        state = {}
        for key, value in state_dict.items():
            if not isinstance(value, torch.Tensor):
                state[key] = (type(value).__name__, value)
        modifiable["state"] = state

    return modifiable


def parse_value(value_str: str) -> Any:
    """
    Parse value string into appropriate Python type.

    Supports:
    - Numbers: "0.01" -> 0.01, "1e-4" -> 0.0001
    - Booleans: "True", "false"
    - Tuples: "(0.9,0.999)" -> (0.9, 0.999)
    - Lists: "[1,2,3]" -> [1, 2, 3]
    - Strings: "'adam'" -> "adam"

    Uses ast.literal_eval() for safe parsing.
    """
    try:
        return ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        # Try as string if literal_eval fails
        return value_str


def modify_state_dict(
    state_dict: dict,
    modifications: List[Tuple[str, str, Any]],
    component: str,
    param_group_idx: Optional[int] = None
) -> Tuple[dict, List[dict]]:
    """
    Apply modifications to checkpoint component state dict.

    Args:
        state_dict: Loaded state dict (optimizer, scheduler, etc.)
        modifications: List of (key, operation, value) tuples
                       operation is either "set" or "scale"
        component: Component type for proper handling
        param_group_idx: For optimizer, target specific param group (default: all)

    Returns:
        (modified_state_dict, list_of_changes)

    Note: Modifies state dict in-place (already a copy from torch.load).
    """
    changes = []

    if component == "optimizer":
        param_groups = state_dict.get("param_groups", [])

        if not param_groups:
            raise ValueError("No param_groups found in optimizer state")

        # Determine which param groups to modify
        if param_group_idx is not None:
            if param_group_idx >= len(param_groups):
                raise ValueError(
                    f"Param group index {param_group_idx} out of range "
                    f"(optimizer has {len(param_groups)} param groups)"
                )
            target_groups = [param_group_idx]
        else:
            target_groups = list(range(len(param_groups)))

        # Apply modifications to each target param group
        for pg_idx in target_groups:
            pg = param_groups[pg_idx]

            for key, operation, value in modifications:
                if key not in pg:
                    raise ValueError(
                        f"Parameter '{key}' not found in param_group {pg_idx}\n"
                        f"Available parameters: {', '.join(k for k in pg.keys() if k != 'params')}"
                    )

                old_value = pg[key]

                if operation == "set":
                    new_value = value
                elif operation == "scale":
                    if not isinstance(old_value, (int, float)):
                        raise ValueError(
                            f"Cannot scale non-numeric parameter '{key}' "
                            f"(type: {type(old_value).__name__})"
                        )
                    new_value = old_value * value
                else:
                    raise ValueError(f"Unknown operation: {operation}")

                pg[key] = new_value

                changes.append({
                    "param_group": pg_idx,
                    "parameter": key,
                    "old_value": old_value,
                    "new_value": new_value,
                    "operation": operation,
                })

    elif component == "scheduler":
        # Modify scheduler state directly
        for key, operation, value in modifications:
            if key not in state_dict:
                raise ValueError(
                    f"Parameter '{key}' not found in scheduler state\n"
                    f"Available parameters: {', '.join(state_dict.keys())}"
                )

            old_value = state_dict[key]

            if operation == "set":
                new_value = value
            elif operation == "scale":
                if not isinstance(old_value, (int, float)):
                    raise ValueError(
                        f"Cannot scale non-numeric parameter '{key}' "
                        f"(type: {type(old_value).__name__})"
                    )
                new_value = old_value * value
            else:
                raise ValueError(f"Unknown operation: {operation}")

            state_dict[key] = new_value

            changes.append({
                "parameter": key,
                "old_value": old_value,
                "new_value": new_value,
                "operation": operation,
            })

    else:
        # Generic component modification
        for key, operation, value in modifications:
            if key not in state_dict:
                raise ValueError(
                    f"Parameter '{key}' not found in {component} state\n"
                    f"Available parameters: {', '.join(state_dict.keys())}"
                )

            old_value = state_dict[key]

            if operation == "set":
                new_value = value
            elif operation == "scale":
                if not isinstance(old_value, (int, float)):
                    raise ValueError(
                        f"Cannot scale non-numeric parameter '{key}' "
                        f"(type: {type(old_value).__name__})"
                    )
                new_value = old_value * value
            else:
                raise ValueError(f"Unknown operation: {operation}")

            state_dict[key] = new_value

            changes.append({
                "parameter": key,
                "old_value": old_value,
                "new_value": new_value,
                "operation": operation,
            })

    return state_dict, changes


def save_checkpoint_atomically(
    file_path: str,
    state_dict: dict,
    backup: bool = True,
    verbose: bool = False
) -> bool:
    """
    Atomically save modified checkpoint with optional backup.

    Atomic save procedure (prevents corruption even on power loss):
    1. If backup requested:
       a. Copy original to backup file (file.pt -> file.pt.bak)
       b. os.fsync() on backup file
       c. os.fsync() on directory to persist metadata
    2. Write modified state to temporary file (file.pt.tmp)
    3. os.fsync() on temporary file (ensure data persisted)
    4. Validate temporary file can be loaded
    5. os.rename(file.pt.tmp, file.pt) - atomic replacement
    6. os.fsync() on directory to persist rename

    If validation fails:
    - Delete temporary file (don't rename it)
    - Original file remains untouched
    - If backup exists, it's still available

    Returns True if save succeeded and validation passed.
    """
    file_path = Path(file_path)
    dir_path = file_path.parent

    # Step 1: Create backup with fsync
    if backup:
        backup_path = Path(str(file_path) + '.bak')
        if verbose:
            print(f"  Creating backup: {backup_path}")

        shutil.copy2(file_path, backup_path)

        # Fsync backup file
        fd = os.open(backup_path, os.O_RDONLY)
        os.fsync(fd)
        os.close(fd)

        # Fsync directory to persist metadata
        dir_fd = os.open(dir_path, os.O_RDONLY)
        os.fsync(dir_fd)
        os.close(dir_fd)

    # Step 2: Write to temp file
    temp_path = Path(str(file_path) + '.tmp')
    if verbose:
        print(f"  Writing to temp file: {temp_path}")

    torch.save(state_dict, temp_path)

    # Fsync temp file
    fd = os.open(temp_path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

    # Step 3: Validate temp file
    if verbose:
        print(f"  Validating temp file...")

    try:
        loaded = torch.load(temp_path, map_location='cpu')

        # Verify structure matches
        if set(loaded.keys()) != set(state_dict.keys()):
            raise ValueError(
                f"Checkpoint structure mismatch after save\n"
                f"Expected keys: {set(state_dict.keys())}\n"
                f"Got keys: {set(loaded.keys())}"
            )

        if verbose:
            print(f"  ✓ Checkpoint loads successfully")
            print(f"  ✓ State structure intact")

    except Exception as e:
        # Validation failed - delete temp file, keep original
        if verbose:
            print(f"  ✗ Validation failed: {e}")
        temp_path.unlink()
        raise RuntimeError(
            f"Modified checkpoint failed validation: {e}\n"
            f"Deleted temp file, original unchanged.\n"
            f"Backup available at: {file_path}.bak" if backup else ""
        )

    # Step 4: Atomic replace
    if verbose:
        print(f"  Atomic rename...")

    os.rename(temp_path, file_path)

    # Fsync directory to persist rename
    dir_fd = os.open(dir_path, os.O_RDONLY)
    os.fsync(dir_fd)
    os.close(dir_fd)

    if verbose:
        print(f"  ✓ {file_path}")

    return True


def update_checkpoint_manifest(
    checkpoint_dir: str,
    component: str,
    modified_files: List[str],
    verbose: bool = False
) -> None:
    """
    Atomically update checkpoint_manifest.json after modifying component files.

    Atomic manifest update procedure:
    1. Create backup of manifest (checkpoint_manifest.json.bak)
    2. os.fsync() on backup
    3. Load current manifest
    4. Update manifest:
       - size_bytes: Recalculate from modified files
       - checksum: Clear (no longer valid after modification)
       - metadata: Add modification timestamp and tool name
    5. Write updated manifest to temp file (checkpoint_manifest.json.tmp)
    6. os.fsync() on temp file
    7. os.rename() to replace original
    8. os.fsync() on directory

    If manifest doesn't exist, log warning and skip (backward compatibility).
    """
    checkpoint_dir = Path(checkpoint_dir)
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"

    if not manifest_path.exists():
        if verbose:
            print("  No checkpoint_manifest.json found, skipping manifest update")
        return

    if verbose:
        print(f"\nUpdating checkpoint manifest (atomic)...")

    # Step 1: Create backup
    backup_path = Path(str(manifest_path) + '.bak')
    if verbose:
        print(f"  Backing up manifest: {backup_path}")

    shutil.copy2(manifest_path, backup_path)

    # Fsync backup
    fd = os.open(backup_path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

    # Fsync directory
    dir_fd = os.open(checkpoint_dir, os.O_RDONLY)
    os.fsync(dir_fd)
    os.close(dir_fd)

    # Step 2: Load and update manifest
    with open(manifest_path, 'r') as f:
        manifest = json.load(f)

    # Update component info
    if "components" in manifest and component in manifest["components"]:
        component_info = manifest["components"][component]

        # Recalculate size
        total_size = sum(os.path.getsize(f) for f in modified_files)
        component_info["size_bytes"] = total_size

        # Clear checksum (no longer valid)
        if "checksum" in component_info:
            del component_info["checksum"]

        # Add modification metadata
        if "metadata" not in component_info:
            component_info["metadata"] = {}

        from datetime import datetime
        component_info["metadata"]["modified_by"] = "forgather checkpoint modify"
        component_info["metadata"]["modified_at"] = datetime.now().isoformat()

    # Step 3: Write to temp file
    temp_path = Path(str(manifest_path) + '.tmp')
    if verbose:
        print(f"  Writing updated manifest: {temp_path}")

    with open(temp_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    # Fsync temp file
    fd = os.open(temp_path, os.O_RDONLY)
    os.fsync(fd)
    os.close(fd)

    # Step 4: Atomic replace
    os.rename(temp_path, manifest_path)

    # Fsync directory
    dir_fd = os.open(checkpoint_dir, os.O_RDONLY)
    os.fsync(dir_fd)
    os.close(dir_fd)

    if verbose:
        print(f"  ✓ {manifest_path}")


def format_value(value: Any) -> str:
    """Format value for display."""
    if isinstance(value, float):
        if abs(value) < 1e-3 or abs(value) > 1e4:
            return f"{value:.2e}"
        return f"{value:.6f}".rstrip('0').rstrip('.')
    return str(value)


def print_changes_table(changes: List[dict], component: str) -> None:
    """Print a formatted table of changes."""
    if not changes:
        print("No changes to apply.")
        return

    # Determine columns
    has_param_group = "param_group" in changes[0]

    # Calculate column widths
    if has_param_group:
        pg_width = max(len("Group"), max(len(str(c["param_group"])) for c in changes))
        param_width = max(len("Parameter"), max(len(c["parameter"]) for c in changes))
    else:
        pg_width = 0
        param_width = max(len("Parameter"), max(len(c["parameter"]) for c in changes))

    old_width = max(len("Old Value"), max(len(format_value(c["old_value"])) for c in changes))
    new_width = max(len("New Value"), max(len(format_value(c["new_value"])) for c in changes))

    # Print header
    print("\nChanges to apply:")
    if has_param_group:
        print(f"┌─{'─' * pg_width}─┬─{'─' * param_width}─┬─{'─' * old_width}─┬─{'─' * new_width}─┐")
        print(f"│ {'Group':<{pg_width}} │ {'Parameter':<{param_width}} │ {'Old Value':<{old_width}} │ {'New Value':<{new_width}} │")
        print(f"├─{'─' * pg_width}─┼─{'─' * param_width}─┼─{'─' * old_width}─┼─{'─' * new_width}─┤")
    else:
        print(f"┌─{'─' * param_width}─┬─{'─' * old_width}─┬─{'─' * new_width}─┐")
        print(f"│ {'Parameter':<{param_width}} │ {'Old Value':<{old_width}} │ {'New Value':<{new_width}} │")
        print(f"├─{'─' * param_width}─┼─{'─' * old_width}─┼─{'─' * new_width}─┤")

    # Print rows
    for change in changes:
        old_val = format_value(change["old_value"])
        new_val = format_value(change["new_value"])

        if has_param_group:
            pg = str(change["param_group"])
            param = change["parameter"]
            print(f"│ {pg:<{pg_width}} │ {param:<{param_width}} │ {old_val:<{old_width}} │ {new_val:<{new_width}} │")
        else:
            param = change["parameter"]
            print(f"│ {param:<{param_width}} │ {old_val:<{old_width}} │ {new_val:<{new_width}} │")

    # Print footer
    if has_param_group:
        print(f"└─{'─' * pg_width}─┴─{'─' * param_width}─┴─{'─' * old_width}─┴─{'─' * new_width}─┘")
    else:
        print(f"└─{'─' * param_width}─┴─{'─' * old_width}─┴─{'─' * new_width}─┘")


def cmd_list(args):
    """List modifiable parameters in a checkpoint component."""
    try:
        files = discover_checkpoint_files(args.checkpoint_path, args.component)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\nComponent: {args.component}")
    print(f"Files: {', '.join(Path(f).name for f in files)}")

    # Load first file to inspect structure
    state_dict = torch.load(files[0], map_location='cpu')
    modifiable = list_modifiable_parameters(state_dict, args.component)

    if args.component == "optimizer" and "param_groups" in modifiable:
        print()
        for i, pg_params in enumerate(modifiable["param_groups"]):
            print(f"Param group {i}:")
            for key, (type_name, value) in sorted(pg_params.items()):
                if args.verbose:
                    print(f"  {key}: {format_value(value)} ({type_name})")
                else:
                    print(f"  {key}: {format_value(value)}")
            print()

    elif "state" in modifiable:
        print("\nState:")
        for key, (type_name, value) in sorted(modifiable["state"].items()):
            if args.verbose:
                print(f"  {key}: {format_value(value)} ({type_name})")
            else:
                print(f"  {key}: {format_value(value)}")
        print()

    return 0


def cmd_modify(args):
    """Modify parameters in a checkpoint component."""
    # Parse modifications
    modifications = []

    for set_arg in args.set:
        if '=' not in set_arg:
            print(f"Error: Invalid --set format: {set_arg}", file=sys.stderr)
            print("Expected: KEY=VALUE", file=sys.stderr)
            return 1

        key, value_str = set_arg.split('=', 1)
        try:
            value = parse_value(value_str)
        except Exception as e:
            print(f"Error: Failed to parse value '{value_str}' for parameter '{key}': {e}", file=sys.stderr)
            return 1

        modifications.append((key, "set", value))

    for scale_arg in args.scale:
        if '=' not in scale_arg:
            print(f"Error: Invalid --scale format: {scale_arg}", file=sys.stderr)
            print("Expected: KEY=FACTOR", file=sys.stderr)
            return 1

        key, factor_str = scale_arg.split('=', 1)
        try:
            factor = parse_value(factor_str)
            if not isinstance(factor, (int, float)):
                raise ValueError(f"Scale factor must be numeric, got {type(factor).__name__}")
        except Exception as e:
            print(f"Error: Failed to parse scale factor '{factor_str}' for parameter '{key}': {e}", file=sys.stderr)
            return 1

        modifications.append((key, "scale", factor))

    if not modifications:
        print("Error: No modifications specified (use --set or --scale)", file=sys.stderr)
        return 1

    # Discover checkpoint files
    try:
        files = discover_checkpoint_files(args.checkpoint_path, args.component)
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"\nDiscovering checkpoint files...")
    if args.verbose:
        for f in files:
            print(f"  Found: {f}")
    else:
        print(f"Found: {', '.join(Path(f).name for f in files)}")

    # Load first file to preview changes
    state_dict = torch.load(files[0], map_location='cpu')

    print(f"\nComponent: {args.component}")
    if args.component == "optimizer":
        num_groups = len(state_dict.get("param_groups", []))
        print(f"Param groups: {num_groups}")

    # Apply modifications to preview
    try:
        _, changes = modify_state_dict(
            state_dict.copy() if args.component != "optimizer" else state_dict,
            modifications,
            args.component,
            args.param_group
        )
    except ValueError as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1

    # Print changes table
    print_changes_table(changes, args.component)

    print(f"\nFiles to modify: {len(files)}")
    for f in files:
        print(f"  - {f}")

    # Dry run - stop here
    if args.dry_run:
        print("\n[DRY RUN] No changes written.")
        return 0

    # Confirm with user
    if not args.force:
        print()
        if not args.no_backup:
            response = input("Create backup? [Y/n]: ").strip().lower()
            create_backup = response != 'n'
        else:
            create_backup = False

        response = input("Proceed with modification? [y/N]: ").strip().lower()
        if response != 'y':
            print("Aborted.")
            return 0
    else:
        create_backup = not args.no_backup

    # Apply modifications to all files
    print()
    if create_backup:
        print("Creating backup (with fsync)...")

    modified_files = []

    for file_path in files:
        try:
            # Load state
            state_dict = torch.load(file_path, map_location='cpu')

            # Apply modifications
            modified_state, _ = modify_state_dict(
                state_dict,
                modifications,
                args.component,
                args.param_group
            )

            # Save atomically
            if not args.quiet:
                print(f"\nModifying {file_path}...")

            save_checkpoint_atomically(
                file_path,
                modified_state,
                backup=create_backup,
                verbose=args.verbose
            )

            modified_files.append(file_path)

            if not args.quiet and not args.verbose:
                print(f"  ✓ {Path(file_path).name}")

        except Exception as e:
            print(f"\nError modifying {file_path}: {e}", file=sys.stderr)
            return 1

    # Update manifest
    checkpoint_dir = Path(args.checkpoint_path)
    if checkpoint_dir.is_file():
        checkpoint_dir = checkpoint_dir.parent

    try:
        update_checkpoint_manifest(
            str(checkpoint_dir),
            args.component,
            modified_files,
            verbose=args.verbose
        )
    except Exception as e:
        print(f"\nWarning: Failed to update checkpoint manifest: {e}", file=sys.stderr)

    print(f"\nDone! Modified {len(modified_files)} file(s).")

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Modify optimizer, scheduler, and other component parameters in Forgather checkpoints"
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # List command
    list_parser = subparsers.add_parser(
        'list',
        help='List modifiable parameters in a checkpoint component'
    )
    list_parser.add_argument(
        'checkpoint_path',
        help='Path to checkpoint directory or file'
    )
    list_parser.add_argument(
        '--component',
        default='optimizer',
        help='Component to inspect (default: optimizer)'
    )
    list_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Show value types and detailed information'
    )

    # Modify command
    modify_parser = subparsers.add_parser(
        'modify',
        help='Modify parameters in a checkpoint component'
    )
    modify_parser.add_argument(
        'checkpoint_path',
        help='Path to checkpoint directory or file'
    )
    modify_parser.add_argument(
        '--component',
        default='optimizer',
        help='Component to modify (default: optimizer)'
    )
    modify_parser.add_argument(
        '--set',
        action='append',
        default=[],
        metavar='KEY=VALUE',
        help='Set parameter to exact value (can be used multiple times)'
    )
    modify_parser.add_argument(
        '--scale',
        action='append',
        default=[],
        metavar='KEY=FACTOR',
        help='Multiply parameter by factor (can be used multiple times)'
    )
    modify_parser.add_argument(
        '--param-group',
        type=int,
        metavar='INDEX',
        help='For optimizer: target specific param group (default: all)'
    )
    modify_parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without saving'
    )
    modify_parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup creation (still uses atomic operations)'
    )
    modify_parser.add_argument(
        '--force',
        action='store_true',
        help='Skip confirmation prompts'
    )
    modify_parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Detailed logging'
    )
    modify_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='Minimal output'
    )

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return 1

    if args.command == 'list':
        return cmd_list(args)
    elif args.command == 'modify':
        return cmd_modify(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
