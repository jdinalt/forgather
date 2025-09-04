import os
import tempfile
import subprocess
from typing import Optional, List


def add_output_arg(parser):
    """Add output file argument to parser"""
    parser.add_argument(
        "-o",
        "--output-file",
        type=os.path.expanduser,
        default=None,
        help="Write output to file",
    )


def add_editor_arg(parser):
    """Add editor output argument to parser"""
    parser.add_argument(
        "-e",
        "--edit",
        action="store_true",
        help="Write output to temporary file and open in editor",
    )


def write_output(args, data):
    """For commands with an '-o' argument, either write data to stdout or to file"""
    if args.output_file:
        try:
            with open(args.output_file, "w") as f:
                f.write(data)
            print(f"Wrote output to {args.output_file}")
        except Exception as e:
            print(e)
    else:
        print(data)


def write_output_or_edit(args, data, file_extension=".txt", command_name="output"):
    """Write data to file, temp file + editor, or stdout based on flags

    Args:
        args: Command line arguments
        data: Content to write
        file_extension: File extension for the temporary file
        command_name: Name of the command generating the output (for filename)
    """
    if hasattr(args, "edit") and args.edit:
        try:
            # Create workspace temp directory if it doesn't exist
            temp_dir = "./tmp"
            os.makedirs(temp_dir, exist_ok=True)

            # Generate predictable filename based on config and command
            config_prefix = ""
            if hasattr(args, "config_template") and args.config_template:
                # Remove .yaml extension and path components for clean prefix
                config_name = os.path.basename(args.config_template)
                if config_name.endswith(".yaml"):
                    config_name = config_name[:-5]
                config_prefix = f"{config_name}_"

            filename = f"{config_prefix}{command_name}{file_extension}"
            temp_path = os.path.join(temp_dir, filename)

            # Write content to the file
            with open(temp_path, "w") as f:
                f.write(data)

            print(f"Opening temporary file in editor: {temp_path}")
            _open_in_editor(
                temp_path
            )  # Use full path - editor will handle it correctly

        except Exception as e:
            print(f"Error creating/opening temporary file: {e}")
            # Fall back to stdout
            print(data)
    else:
        # Use existing write_output for -o flag or stdout
        write_output(args, data)


def should_use_absolute_paths(args):
    """Check if we should use absolute paths (when using -e flag)

    When using -e flag, we create files in ./tmp/ directory,
    so we need to adjust paths to be relative to ./tmp/
    """
    return hasattr(args, "edit") and args.edit


def adjust_path_for_tmp_dir(path):
    """Adjust a relative path to be relative to ./tmp/ directory instead of project root

    Args:
        path: Original path relative to project root

    Returns:
        Path relative to ./tmp/ directory
    """
    if os.path.isabs(path):
        return path
    # Convert path relative to project root to be relative to ./tmp/
    return os.path.join("..", path)


def _get_best_editor() -> str:
    """Determine the best editor to use, checking for VS Code remote CLI first.

    Returns:
        Path to the best available editor
    """
    # Check for VS Code remote CLI first
    vscode_ipc = os.environ.get("VSCODE_IPC_HOOK_CLI")
    if vscode_ipc:
        # Look for remote CLI in VS Code server directories
        vscode_server_dirs = []
        vscode_server_base = os.path.expanduser("~/.vscode-server/bin")

        if os.path.exists(vscode_server_base):
            try:
                # Sort by modification time (newest first)
                for version_dir in os.listdir(vscode_server_base):
                    version_path = os.path.join(vscode_server_base, version_dir)
                    if os.path.isdir(version_path):
                        vscode_server_dirs.append(version_path)
            except OSError:
                pass

            # Check for remote CLI in each server directory
            for server_dir in vscode_server_dirs:
                remote_cli = os.path.join(server_dir, "bin", "remote-cli", "code")
                if os.path.exists(remote_cli) and os.access(remote_cli, os.X_OK):
                    return remote_cli

    # Fall back to user's EDITOR or vim
    return os.environ.get("EDITOR", "vim")


def _get_vim_server_info(editor: str) -> tuple[str, Optional[str]]:
    """Get vim executable and server name if clientserver mode is requested.

    Args:
        editor: The editor path/command

    Returns:
        Tuple of (editor_command, server_name) where server_name is None if not using clientserver
    """
    # Check if this is vim or nvim
    editor_name = os.path.basename(editor)
    if editor_name not in ["vim", "nvim"]:
        return editor, None

    # Check if forgather_vim_server is set
    if not os.environ.get("FORGATHER_VIM_SERVER"):
        return editor, None

    # Check if vim was compiled with clientserver support
    try:
        result = subprocess.run(
            [editor, "--version"], capture_output=True, text=True, timeout=5
        )
        if "+clientserver" not in result.stdout:
            print(f"Warning: {editor_name} was not compiled with +clientserver support")
            print("Falling back to normal vim mode")
            return editor, None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return editor, None

    # Use forgather as server name
    server_name = "forgather"
    return editor, server_name


def _open_in_editor(file_path: str):
    """Open a single file in the configured editor.

    Args:
        file_path: Path to the file to open
    """
    editor = _get_best_editor()
    editor_name = os.path.basename(editor)

    # Check for vim clientserver mode
    vim_editor, vim_server = _get_vim_server_info(editor)

    try:
        if vim_server:
            print(f"Opening {file_path} in vim server '{vim_server}'...")
            cmd_args = [
                vim_editor,
                "--servername",
                vim_server,
                "--remote-tab",
                file_path,
            ]
        else:
            print(f"Opening {file_path} in {editor_name}...")
            # Choose command args based on editor type
            if "remote-cli/code" in editor:
                # VS Code remote CLI
                cmd_args = [editor, file_path]
            elif editor_name in ["code", "code.exe"]:
                # VS Code - reuse existing window
                cmd_args = [editor, "-r", file_path]
            else:
                # For other editors, just pass the file
                cmd_args = [editor, file_path]

        # Launch editor
        subprocess.run(cmd_args, check=True)

    except FileNotFoundError:
        print(
            f"Editor '{editor}' not found. Set EDITOR environment variable to your preferred editor."
        )
    except subprocess.CalledProcessError as e:
        print(f"Error launching editor: {e}")
