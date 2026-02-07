import json
import os
from pathlib import Path

from forgather import Project
from forgather.ml.sharded_checkpoint import create_pretrained_symlinks

from .dynamic_args import get_dynamic_args
from .utils import assert_project_class


def checkpoint_cmd(args):
    """checkpoint commands."""

    if hasattr(args, "cp_subcommand"):
        match args.cp_subcommand:
            case "link":
                link_command(args)
            case "inspect":
                inspect_command(args)


def link_command(args):
    assert_project_class(args, "type.training_script")
    if not args.output_path:
        config_name = args.config_template
        if args.config_template is None:
            args.config_template = ""

        project_args = get_dynamic_args(args)
        proj = Project(
            config_name=args.config_template,
            project_dir=args.project_dir,
            **project_args,
        )
        proj_meta = proj("meta")
        output_dir = proj_meta["output_dir"]
    else:
        output_dir = args.output_path

    print(f"Creating symlinks to newest checkpoint in {output_dir}")
    link_files = create_pretrained_symlinks(
        output_dir, force_overwrite=args.force, dry_run=args.dry_run
    )
    print(f"Created links: {link_files}")


def inspect_command(args):
    """
    Inspect a checkpoint and display its structure, manifest, and validation info.

    Shows:
    - Checkpoint manifest (if present)
    - File structure and sizes
    - Component patterns and ranks
    - Validation status
    - Missing or unexpected files
    """
    checkpoint_path = args.checkpoint_path

    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint path does not exist: {checkpoint_path}")
        return

    print(f"=" * 80)
    print(f"Checkpoint Inspection: {checkpoint_path}")
    print(f"=" * 80)
    print()

    # Check for manifest
    manifest_path = os.path.join(checkpoint_path, "checkpoint_manifest.json")
    has_manifest = os.path.exists(manifest_path)

    if has_manifest:
        print("✓ Checkpoint manifest found")
        inspect_with_manifest(checkpoint_path, manifest_path, args)
    else:
        print("⚠ No checkpoint manifest found (legacy checkpoint)")
        inspect_without_manifest(checkpoint_path, args)


def inspect_with_manifest(checkpoint_path: str, manifest_path: str, args):
    """Inspect checkpoint using manifest."""
    try:
        from forgather.ml.trainer.checkpoint_types import CheckpointManifest

        manifest = CheckpointManifest.load(manifest_path)

        print()
        print("Checkpoint Metadata:")
        print(f"  World Size: {manifest.world_size}")
        print(f"  Created: {manifest.timestamp}")
        if manifest.pytorch_version:
            print(f"  PyTorch Version: {manifest.pytorch_version}")
        if manifest.forgather_version:
            print(f"  Forgather Version: {manifest.forgather_version}")
        if manifest.training_args_hash:
            print(f"  Config Hash: {manifest.training_args_hash}")

        print()
        print(f"Components ({len(manifest.components)}):")
        print()

        total_size = 0
        for key, comp_manifest in sorted(manifest.components.items()):
            print(f"  [{key}]")
            print(f"    Pattern: {comp_manifest.sharing_pattern}")
            print(f"    Size: {_format_size(comp_manifest.size_bytes)}")
            print(f"    Saved by ranks: {comp_manifest.ranks}")

            if comp_manifest.replicated_across:
                print(f"    Replicated across: {comp_manifest.replicated_across}")
            if comp_manifest.group_name:
                print(f"    Process group: {comp_manifest.group_name}")
            if comp_manifest.checksum:
                print(f"    Checksum: {comp_manifest.checksum[:16]}...")

            total_size += comp_manifest.size_bytes

            # Verify files exist
            if args.verbose:
                files = _find_component_files(checkpoint_path, key, comp_manifest)
                if files:
                    print(f"    Files ({len(files)}):")
                    for f in files[:5]:  # Show first 5
                        print(f"      - {os.path.basename(f)}")
                    if len(files) > 5:
                        print(f"      ... and {len(files) - 5} more")
                else:
                    print(f"    ⚠ No files found!")

            print()

        print(f"Total checkpoint size: {_format_size(total_size)}")

        # Validation
        if args.validate:
            print()
            print("Validation:")
            validate_checkpoint_files(checkpoint_path, manifest, args)

    except Exception as e:
        print(f"Error reading manifest: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()


def inspect_without_manifest(checkpoint_path: str, args):
    """Inspect legacy checkpoint without manifest."""
    print()
    print("Scanning checkpoint directory...")
    print()

    # Find all state files
    state_files = []
    for root, dirs, files in os.walk(checkpoint_path):
        for file in files:
            if file.endswith(".pt") or file.endswith(".safetensors"):
                full_path = os.path.join(root, file)
                state_files.append(full_path)

    if not state_files:
        print("⚠ No checkpoint files found (.pt or .safetensors)")
        return

    print(f"Found {len(state_files)} checkpoint file(s):")
    print()

    total_size = 0
    for file_path in sorted(state_files):
        rel_path = os.path.relpath(file_path, checkpoint_path)
        size = os.path.getsize(file_path)
        total_size += size

        print(f"  {rel_path}")
        print(f"    Size: {_format_size(size)}")

        # Try to infer pattern from filename
        pattern = _infer_pattern_from_filename(os.path.basename(file_path))
        if pattern:
            print(f"    Inferred pattern: {pattern}")

        print()

    print(f"Total size: {_format_size(total_size)}")


def _find_component_files(checkpoint_path: str, key: str, comp_manifest) -> list:
    """Find all files for a component based on its manifest."""
    import glob

    pattern = comp_manifest.sharing_pattern
    files = []

    if pattern == "global" or pattern == "replicated":
        # Single file: {key}_state.pt
        path = os.path.join(checkpoint_path, f"{key}_state.pt")
        if os.path.exists(path):
            files.append(path)
    elif pattern == "per_rank":
        # Multiple files: {key}_state_rank_*.pt
        pattern_str = os.path.join(checkpoint_path, f"{key}_state_rank_*.pt")
        files = glob.glob(pattern_str)
    elif pattern == "per_group":
        # Multiple files: {key}_state_group_*_rank_*.pt
        pattern_str = os.path.join(checkpoint_path, f"{key}_state_group_*_rank_*.pt")
        files = glob.glob(pattern_str)
    elif pattern == "per_node":
        # Multiple files: {key}_state_node_*_rank_*.pt
        pattern_str = os.path.join(checkpoint_path, f"{key}_state_node_*_rank_*.pt")
        files = glob.glob(pattern_str)

    return files


def validate_checkpoint_files(checkpoint_path: str, manifest, args):
    """Validate that all expected files exist."""
    issues = []

    for key, comp_manifest in manifest.components.items():
        files = _find_component_files(checkpoint_path, key, comp_manifest)

        if not files:
            issues.append(f"  ✗ Component '{key}': No files found")
        else:
            expected_count = len(comp_manifest.ranks)
            actual_count = len(files)

            if (
                actual_count != expected_count
                and comp_manifest.sharing_pattern not in ("global", "replicated")
            ):
                issues.append(
                    f"  ⚠ Component '{key}': Expected {expected_count} files, found {actual_count}"
                )
            else:
                print(f"  ✓ Component '{key}': {len(files)} file(s)")

    if issues:
        print()
        print("Issues:")
        for issue in issues:
            print(issue)
    else:
        print("  All components validated ✓")


def _format_size(size_bytes: int) -> str:
    """Format size in bytes to human-readable string."""
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} PB"


def _infer_pattern_from_filename(filename: str) -> str:
    """Infer sharing pattern from filename."""
    if "_rank_" in filename:
        if "_group_" in filename:
            return "per_group"
        elif "_node_" in filename:
            return "per_node"
        else:
            return "per_rank"
    else:
        return "global or replicated"
