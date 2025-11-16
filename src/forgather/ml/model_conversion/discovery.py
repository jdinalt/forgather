"""Auto-discovery system for model converters.

This module provides functionality to automatically discover and import
model converter modules from various locations, eliminating the need for
hardcoded imports in the conversion scripts.
"""

import os
import sys
import logging
import importlib.util
from pathlib import Path
from typing import List, Optional

logger = logging.getLogger(__name__)


def discover_converters_in_directory(directory: str, recursive: bool = True) -> None:
    """Discover and import converter modules from a directory.

    Searches for Python files named 'converter.py' or '*_converter.py' in the
    specified directory. When found, imports them to trigger @register_converter
    decorator execution.

    Args:
        directory: Path to directory to search for converters
        recursive: If True, search subdirectories recursively (default: True)

    Note:
        Converters must use the @register_converter decorator to be registered.
        Import errors are logged as warnings but don't stop discovery.
    """
    directory = Path(directory).resolve()

    if not directory.exists():
        logger.warning(f"Converter directory does not exist: {directory}")
        return

    if not directory.is_dir():
        logger.warning(f"Converter path is not a directory: {directory}")
        return

    # Find all converter files
    pattern = "**/*converter.py" if recursive else "*converter.py"
    converter_files = list(directory.glob(pattern))

    logger.info(f"Found {len(converter_files)} converter file(s) in {directory}")

    for converter_file in converter_files:
        _import_converter_module(converter_file)


def discover_builtin_converters(forgather_root: Optional[str] = None) -> None:
    """Discover and import builtin Forgather model converters.

    Automatically finds and imports all converter modules in the standard
    Forgather location: examples/models/*/src/converter.py

    Args:
        forgather_root: Path to Forgather root directory. If None, attempts to
                       find it from the current file's location.
    """
    if forgather_root is None:
        # Try to find Forgather root from this module's location
        # This file is at: {forgather_root}/src/forgather/ml/model_conversion/discovery.py
        current_file = Path(__file__).resolve()
        forgather_root = current_file.parent.parent.parent.parent.parent

    forgather_root = Path(forgather_root).resolve()
    models_dir = forgather_root / "examples" / "models"

    if not models_dir.exists():
        logger.warning(f"Builtin models directory not found: {models_dir}")
        return

    logger.info(f"Discovering builtin converters in: {models_dir}")

    # Find all model subdirectories with src/converter.py
    for model_dir in models_dir.iterdir():
        if not model_dir.is_dir():
            continue

        converter_file = model_dir / "src" / "converter.py"
        if converter_file.exists():
            _import_converter_module(converter_file, is_builtin=True)


def discover_from_paths(paths: List[str], forgather_root: Optional[str] = None) -> None:
    """Discover converters from multiple paths.

    Combines builtin discovery with custom path discovery. Processes paths in order:
    1. Builtin converters (if forgather_root provided or detectable)
    2. Custom paths in the order specified

    Args:
        paths: List of directory paths to search for converters
        forgather_root: Optional path to Forgather root for builtin discovery
    """
    # First, discover builtin converters
    if forgather_root or _can_find_forgather_root():
        discover_builtin_converters(forgather_root)
    else:
        logger.info("Skipping builtin converter discovery (Forgather root not found)")

    # Then discover from custom paths
    for path in paths:
        logger.info(f"Searching for converters in: {path}")
        discover_converters_in_directory(path)


def _import_converter_module(module_path: Path, is_builtin: bool = False) -> None:
    """Import a converter module from file path.

    Uses importlib to dynamically import a module from a file path. When the module
    is imported, any @register_converter decorators will execute and register the
    converters.

    Args:
        module_path: Path to the Python module file
        is_builtin: Whether this is a builtin Forgather converter
    """
    module_path = Path(module_path).resolve()

    # Create a unique module name based on the file path
    # Use relative path for builtin, absolute for external
    if is_builtin:
        # For builtin, use examples.models.{model_name}.src.converter pattern
        parts = module_path.parts
        try:
            models_idx = parts.index("models")
            model_name = parts[models_idx + 1]
            module_name = f"examples.models.{model_name}.src.converter"
        except (ValueError, IndexError):
            module_name = f"converter_{module_path.stem}_{id(module_path)}"
    else:
        # For external, use a unique name based on path
        module_name = (
            f"external_converter_{module_path.stem}_{abs(hash(str(module_path)))}"
        )

    try:
        # For builtin converters, we need to ensure the parent package structure exists
        if is_builtin:
            # Add forgather root to sys.path if needed
            parts = module_path.parts
            models_idx = parts.index("models")
            forgather_root = str(Path(*parts[: models_idx - 1]))

            if forgather_root not in sys.path:
                sys.path.insert(0, forgather_root)

            # Import using standard import mechanism for builtin
            # This properly handles package structure
            import importlib

            module = importlib.import_module(module_name)
            logger.info(f"Imported builtin converter module: {module_name}")
        else:
            # For external converters, use file-based loading
            parent_dir = str(module_path.parent)
            if parent_dir not in sys.path:
                sys.path.insert(0, parent_dir)

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            if spec is None or spec.loader is None:
                logger.warning(f"Could not load module spec for: {module_path}")
                return

            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            logger.info(
                f"Imported external converter module: {module_name} from {module_path}"
            )

    except Exception as e:
        logger.warning(f"Failed to import converter from {module_path}: {e}")


def _can_find_forgather_root() -> bool:
    """Check if we can find the Forgather root directory."""
    try:
        current_file = Path(__file__).resolve()
        forgather_root = current_file.parent.parent.parent.parent.parent
        models_dir = forgather_root / "examples" / "models"
        return models_dir.exists()
    except Exception:
        return False
