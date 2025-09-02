"""Interactive CLI shell with tab completion for forgather."""

import cmd
import shlex
import sys
import os
import subprocess
import shutil
from typing import List, Optional
import traceback
import signal
import atexit

from forgather.meta_config import MetaConfig
from .commands import get_env
from .main import get_subcommand_registry, parse_args, main as cli_main

try:
    import readline

    HAS_READLINE = True
except ImportError:
    HAS_READLINE = False


class ForgatherShell(cmd.Cmd):
    """Interactive shell for forgather with tab completion."""

    def __init__(self, project_dir: str = "."):
        super().__init__()
        # Use the same project discovery logic as the rest of the CLI
        try:
            discovered_dir = MetaConfig.find_project_dir(project_dir)
            self.project_dir = discovered_dir
            project_found = True
        except Exception:
            # Fallback to provided directory if project discovery fails
            self.project_dir = os.path.abspath(project_dir)
            project_found = False

        self.current_template = None
        self._cached_templates = None
        self._cached_commands = None

        # Setup command history
        self._setup_history()

        # Setup signal handlers
        self._setup_signal_handlers()

        # Create dynamic intro message
        if project_found:
            self.intro = (
                f"Welcome to the Forgather interactive shell.\n"
                f"Project found at: {self.project_dir}\n"
                f"Use tab completion for templates, commands, and directories.\n"
                f'Examples: "template <TAB>", "-t <TAB>", "tr<TAB>", "cd <TAB>"\n'
                f"Type help or ? to list commands.\n"
            )
        else:
            self.intro = (
                f"Welcome to the Forgather interactive shell.\n"
                f"Working in: {self.project_dir} (no project meta.yaml found)\n"
                f'Use "cd <TAB>" to navigate to a project directory.\n'
                f"Type help or ? to list commands.\n"
            )

        self._update_prompt()

    def _update_prompt(self):
        """Update the prompt to show current directory and template."""
        dir_name = os.path.basename(self.project_dir) or "root"
        template_part = f" [{self.current_template}]" if self.current_template else ""
        self.prompt = f"forgather:{dir_name}{template_part}> "

    def _setup_history(self):
        """Setup persistent command history."""
        if not HAS_READLINE:
            return

        # Find workspace root for history file using existing MetaConfig method
        try:
            workspace_root = MetaConfig.find_workspace_dir(self.project_dir)
        except Exception:
            # Fallback to project directory if workspace discovery fails
            workspace_root = self.project_dir

        self.history_file = os.path.join(workspace_root, ".forgather_history")

        # Load existing history
        try:
            if os.path.exists(self.history_file):
                readline.read_history_file(self.history_file)
        except (OSError, IOError):
            pass  # Ignore if we can't read the history file

        # Track startup history length for safer concurrent access
        self._startup_history_length = readline.get_current_history_length()

        # Set history size limit (like bash default)
        readline.set_history_length(1000)

        # Register cleanup function to save history on exit
        atexit.register(self._save_history)

    def _save_history(self):
        """Save command history to file with better concurrent access handling."""
        if not HAS_READLINE or not hasattr(self, "history_file"):
            return

        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(self.history_file), exist_ok=True)

            # Get current history length
            current_length = readline.get_current_history_length()

            # Only save new commands (those added since startup)
            if current_length > self._startup_history_length:
                # Append only new history entries to avoid overwriting other instances
                self._append_new_history_entries()

        except (OSError, IOError):
            pass  # Ignore if we can't save

    def _append_new_history_entries(self):
        """Append only new history entries to the history file."""
        try:
            import fcntl  # For file locking on Unix systems

            # Get new history entries
            new_entries = []
            current_length = readline.get_current_history_length()

            for i in range(self._startup_history_length, current_length):
                try:
                    entry = readline.get_history_item(
                        i + 1
                    )  # readline uses 1-based indexing
                    if entry:
                        new_entries.append(entry)
                except:
                    pass

            if new_entries:
                # Append new entries to file with file locking
                with open(self.history_file, "a", encoding="utf-8") as f:
                    try:
                        fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                        for entry in new_entries:
                            f.write(entry + "\n")
                    finally:
                        fcntl.flock(f.fileno(), fcntl.LOCK_UN)

        except ImportError:
            # Fallback for systems without fcntl (like Windows)
            self._append_new_history_entries_no_lock()
        except (OSError, IOError):
            pass

    def _append_new_history_entries_no_lock(self):
        """Fallback method for systems without file locking."""
        try:
            # Get new history entries
            new_entries = []
            current_length = readline.get_current_history_length()

            for i in range(self._startup_history_length, current_length):
                try:
                    entry = readline.get_history_item(i + 1)
                    if entry:
                        new_entries.append(entry)
                except:
                    pass

            if new_entries:
                # Simple append (less safe but works on all platforms)
                with open(self.history_file, "a", encoding="utf-8") as f:
                    for entry in new_entries:
                        f.write(entry + "\n")

        except (OSError, IOError):
            pass

    def _setup_signal_handlers(self):
        """Setup signal handlers for better user experience."""
        # Store original SIGINT handler for restoration if needed
        self._original_sigint = signal.signal(signal.SIGINT, signal.default_int_handler)

    def cmdloop(self, intro=None):
        """Enhanced cmdloop with better Ctrl+C handling."""
        while True:
            try:
                super().cmdloop(intro)
                break  # Normal exit
            except KeyboardInterrupt:
                # Ctrl+C pressed - clear line and continue
                print()  # Move to new line
                # Continue the loop to show prompt again

    def _get_available_templates(self) -> List[str]:
        """Get list of available template names."""
        if self._cached_templates is not None:
            return self._cached_templates

        try:
            meta = MetaConfig(self.project_dir)
            templates = []
            for config_name, path in meta.find_templates(meta.config_prefix):
                templates.append(config_name)
            self._cached_templates = templates
            return templates
        except Exception:
            return []

    def _get_available_commands(self) -> List[str]:
        """Get list of available subcommands."""
        if self._cached_commands is not None:
            return self._cached_commands

        registry = get_subcommand_registry()
        self._cached_commands = list(registry.keys())
        return self._cached_commands

    def _get_available_targets(self) -> List[str]:
        """Get list of available targets."""
        try:
            from forgather.meta_config import MetaConfig
            from .commands import get_env, get_config, set_default_template

            # Use similar logic to targets_cmd
            meta = MetaConfig(self.project_dir)
            args_mock = type(
                "Args",
                (),
                {
                    "config_template": self.current_template,
                    "project_dir": self.project_dir,
                },
            )()
            set_default_template(meta, args_mock)
            env = get_env(meta, self.project_dir)
            config, pp_config = get_config(meta, env, args_mock.config_template)

            return list(config.keys())
        except Exception:
            # Return empty list if we can't get targets (e.g., no template set)
            return []

    def _get_all_template_files(self) -> List[tuple]:
        """Get all available template files with their paths."""
        try:
            meta = MetaConfig(self.project_dir)
            templates = []
            for template_name, template_path in meta.find_templates():
                # Make path relative to project directory for cleaner display
                try:
                    rel_path = os.path.relpath(template_path, self.project_dir)
                except ValueError:
                    # Can't make relative (different drives on Windows), use absolute
                    rel_path = template_path
                templates.append((template_name, template_path, rel_path))
            return sorted(templates, key=lambda x: x[2])  # Sort by relative path
        except Exception:
            return []

    def _invalidate_cache(self):
        """Invalidate cached template and command lists."""
        self._cached_templates = None
        self._cached_commands = None
        # Note: targets are not cached as they depend on current template

    def _page_output(self, text: str) -> None:
        """Page output through a pager if it's long, otherwise print normally.

        Args:
            text: The text to display
        """
        lines = text.count("\n") + 1
        terminal_height = shutil.get_terminal_size().lines

        # Only use pager if output is longer than terminal height minus some buffer
        if lines > terminal_height - 3:
            pager_cmd = os.environ.get("PAGER", "less")
            try:
                # Try to use the system pager
                result = subprocess.run(
                    [pager_cmd], input=text, text=True, capture_output=False
                )
            except (FileNotFoundError, subprocess.SubprocessError):
                # Fallback to regular print if pager fails
                print(text)
        else:
            # Print normally for short output
            print(text)

    def _interactive_template_selector(self) -> Optional[List[str]]:
        """Interactive template selector with arrow key navigation.

        Returns:
            List of paths to selected template files, or None if cancelled
        """
        templates = self._get_all_template_files()
        if not templates:
            print("No templates found.")
            return None

        # For simplicity, use a numbered menu approach that works universally
        # (Arrow key navigation would require more complex terminal handling)
        print("\nAvailable templates:")
        print("=" * 50)

        # Group templates by category for better organization
        categories = {}
        for template_name, template_path, rel_path in templates:
            # Determine category from path
            if "experiments/" in rel_path or "configs/" in rel_path:
                category = "Project Configs"
            elif "templatelib/examples/" in rel_path:
                category = "Example Templates"
            elif "templatelib/base/" in rel_path:
                category = "Base Templates"
            elif "forgather_workspace/" in rel_path:
                category = "Workspace Templates"
            else:
                category = "Project Templates"

            if category not in categories:
                categories[category] = []
            categories[category].append((template_name, template_path, rel_path))

        # Display categorized templates with numbers
        template_list = []
        current_num = 1

        for category in sorted(categories.keys()):
            print(f"\n{category}:")
            for template_name, template_path, rel_path in categories[category]:
                print(f"  {current_num:2d}. {rel_path}")
                template_list.append((template_name, template_path, rel_path))
                current_num += 1

        print(f"\n   0. Cancel")
        print("=" * 50)
        print("You can select multiple files:")
        print("  - Single file: 5")
        print("  - Multiple files: 1,3,7")
        print("  - Ranges: 1-5,8,10-12")

        # Get user selection
        while True:
            try:
                choice = input(f"Select template(s) (0-{len(template_list)}): ").strip()
                if choice == "0" or choice.lower() in ["cancel", "quit", "exit"]:
                    return None

                # Parse selection string (supports single numbers, comma-separated, and ranges)
                selected_indices = self._parse_selection_string(
                    choice, len(template_list)
                )
                if selected_indices is None:
                    continue  # Error message already printed by _parse_selection_string

                # Convert indices to file paths
                selected_paths = []
                for idx in selected_indices:
                    selected = template_list[idx - 1]  # Convert to 0-based index
                    selected_paths.append(selected[1])  # Add full path

                return selected_paths

            except KeyboardInterrupt:
                print("\nCancelled")
                return None

    def _parse_selection_string(
        self, selection: str, max_number: int
    ) -> Optional[List[int]]:
        """Parse a selection string like '1,3,5-8' into a list of integers.

        Args:
            selection: Selection string (e.g., '1,3,5-8')
            max_number: Maximum valid number

        Returns:
            List of selected indices (1-based), or None if invalid
        """
        try:
            indices = []
            parts = selection.split(",")

            for part in parts:
                part = part.strip()
                if "-" in part and part.count("-") == 1:
                    # Range like "5-8"
                    start_str, end_str = part.split("-")
                    start_num = int(start_str.strip())
                    end_num = int(end_str.strip())

                    if start_num > end_num:
                        print(f"Invalid range: {part} (start > end)")
                        return None
                    if start_num < 1 or end_num > max_number:
                        print(f"Range {part} is outside valid range (1-{max_number})")
                        return None

                    indices.extend(range(start_num, end_num + 1))
                else:
                    # Single number
                    num = int(part)
                    if num < 1 or num > max_number:
                        print(f"Number {num} is outside valid range (1-{max_number})")
                        return None
                    indices.append(num)

            # Remove duplicates and sort
            indices = sorted(set(indices))
            return indices

        except ValueError:
            print(f"Invalid selection format: {selection}")
            print("Use format like: 1,3,5 or 1-5,8,10-12")
            return None

    def _get_best_editor(self) -> str:
        """Determine the best editor to use, checking for VS Code remote CLI first.

        Returns:
            Path to the best available editor
        """
        # Check if we're in a VS Code terminal with remote CLI available
        vscode_ipc = os.environ.get("VSCODE_IPC_HOOK_CLI")
        if vscode_ipc:
            # Look for VS Code remote CLI in common locations
            vscode_server_dirs = []
            vscode_server_base = os.path.expanduser("~/.vscode-server/bin")

            if os.path.exists(vscode_server_base):
                # Find all version directories
                try:
                    for version_dir in os.listdir(vscode_server_base):
                        version_path = os.path.join(vscode_server_base, version_dir)
                        if os.path.isdir(version_path):
                            vscode_server_dirs.append(version_path)
                except OSError:
                    pass

            # Check each version directory for the remote CLI
            for server_dir in vscode_server_dirs:
                remote_cli_path = os.path.join(server_dir, "bin/remote-cli/code")
                if os.path.exists(remote_cli_path) and os.access(
                    remote_cli_path, os.X_OK
                ):
                    return remote_cli_path

        # Fall back to user's EDITOR or vim
        return os.environ.get("EDITOR", "vim")

    def _get_vim_server_info(self, editor: str) -> tuple[str, Optional[str]]:
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

        # Check for server name in environment
        server_name = os.environ.get("VIM_SERVERNAME")
        if not server_name:
            return editor, None

        # Check if vim was compiled with clientserver support
        try:
            result = subprocess.run(
                [editor, "--version"], capture_output=True, text=True, timeout=5
            )
            if "+clientserver" not in result.stdout:
                print(
                    f"Warning: {editor_name} was not compiled with +clientserver support"
                )
                print("Falling back to normal vim mode")
                return editor, None
        except (subprocess.SubprocessError, subprocess.TimeoutExpired):
            # If we can't check, assume it works
            pass

        return editor, server_name

    def _open_vim_server_tabs(
        self, vim_editor: str, server_name: str, files: List[str]
    ) -> None:
        """Open multiple files as tabs in a vim server instance.

        Args:
            vim_editor: Path to vim executable
            server_name: Name of the vim server
            files: List of file paths to open
        """
        try:
            for file_path in files:
                result = subprocess.run(
                    [
                        vim_editor,
                        "--servername",
                        server_name,
                        "--remote-tab",
                        file_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=10,
                )

                if result.returncode != 0:
                    print(f"Warning: Failed to open {file_path} in vim server")
                    if result.stderr:
                        print(f"  Error: {result.stderr.strip()}")

            print("Files opened in vim server.")

        except subprocess.TimeoutExpired:
            print("Timeout: Vim server took too long to respond")
        except Exception as e:
            print(f"Error opening files in vim server: {e}")

    def _complete_path(self, text: str, only_dirs: bool = False) -> List[str]:
        """Complete file/directory paths like bash does.

        Args:
            text: The partial path to complete
            only_dirs: If True, only complete directories

        Returns:
            List of matching paths
        """
        # Save current working directory and change to shell's project directory
        original_cwd = os.getcwd()
        try:
            os.chdir(self.project_dir)

            # Handle home directory expansion
            if text.startswith("~"):
                text = os.path.expanduser(text)

            # If text is empty, start from current directory
            if not text:
                search_dir = "."
                prefix = ""
                filename_part = ""
            elif text.endswith("/"):
                # If text ends with '/', we're completing inside that directory
                search_dir = text
                prefix = text
                filename_part = ""
            else:
                # Split the path into directory and filename parts
                dir_part = os.path.dirname(text)
                filename_part = os.path.basename(text)

                # If there's no directory part, search current directory
                if not dir_part:
                    search_dir = "."
                    prefix = ""
                else:
                    search_dir = dir_part
                    prefix = dir_part + os.sep

            try:
                # Get all entries in the search directory
                entries = os.listdir(search_dir)

                matches = []

                # Add special directories if they match and we're at the start of filename
                if not filename_part or filename_part.startswith("."):
                    special_dirs = [".", ".."]
                    for special in special_dirs:
                        if special.startswith(filename_part):
                            completion = prefix + special + os.sep
                            matches.append(completion)

                for entry in entries:
                    # Skip hidden files unless explicitly requested
                    if entry.startswith(".") and not filename_part.startswith("."):
                        continue

                    # Skip special directories (already handled above)
                    if entry in [".", ".."]:
                        continue

                    # Check if entry matches the partial filename
                    if entry.startswith(filename_part):
                        full_path = os.path.join(search_dir, entry)

                        # Filter by type if requested
                        if only_dirs and not os.path.isdir(full_path):
                            continue

                        # Build the completion string
                        completion = prefix + entry

                        # Add trailing slash for directories
                        if os.path.isdir(full_path):
                            completion += os.sep

                        matches.append(completion)

                return sorted(matches)

            except (OSError, PermissionError):
                # Return empty list if we can't read the directory
                return []

        finally:
            # Always restore the original working directory
            os.chdir(original_cwd)

    def do_cd(self, arg):
        """Change project directory: cd <directory>

        Usage:
          cd <directory>        # Change to directory
          cd --help            # Show this help message

        Supports tab completion for directory paths.
        Examples: cd <TAB>, cd tu<TAB>, cd ~/pr<TAB>
        """
        # Handle help flags
        if arg and arg.strip() in ["--help", "-h", "help"]:
            print(self.do_cd.__doc__)
            return

        if not arg:
            print("Usage: cd <directory>")
            return

        new_dir = os.path.expanduser(arg)

        # If it's a relative path, resolve it relative to current shell directory
        if not os.path.isabs(new_dir):
            new_dir = os.path.join(self.project_dir, new_dir)

        # Normalize the path to resolve .. and . components
        new_dir = os.path.normpath(new_dir)

        if os.path.isdir(new_dir):
            try:
                # Use project discovery logic to find the actual project directory
                self.project_dir = MetaConfig.find_project_dir(new_dir)
                print(f"Found project at: {self.project_dir}")
            except Exception:
                # Fallback to provided directory if no project found
                self.project_dir = os.path.abspath(new_dir)
                print(f"Changed to: {self.project_dir} (no project meta.yaml found)")

            self.current_template = None  # Reset template when changing directory
            self._invalidate_cache()
            self._update_prompt()
        else:
            print(f"Directory not found: {new_dir}")

    def complete_cd(self, text, line, begidx, endidx):
        """Tab completion for cd command - complete directory paths."""
        # Extract the full path being completed from the command line
        # line looks like: "cd some/path/partial"
        # We need to get everything after "cd "
        cmd_line = line[3:].strip()  # Remove "cd " prefix

        # If there's text being completed, we need to complete that part
        if text:
            # User is typing something after the last /
            # We need to find the directory part and the filename part
            path_to_complete = cmd_line
        else:
            # User hit TAB right after a / or space
            # Complete based on the full path typed so far
            path_to_complete = cmd_line

        completions = self._complete_path(path_to_complete, only_dirs=True)

        # Filter completions to only those that start with the text being typed
        if text:
            filtered = [
                comp
                for comp in completions
                if os.path.basename(comp.rstrip("/")).startswith(text)
            ]
            # Return just the filename part that would replace 'text'
            return [
                os.path.basename(comp.rstrip("/")) + ("/" if comp.endswith("/") else "")
                for comp in filtered
            ]
        else:
            # Return just the filename parts, not full paths
            return [
                os.path.basename(comp.rstrip("/")) + ("/" if comp.endswith("/") else "")
                for comp in completions
            ]

    def do_pwd(self, arg):
        """Show current project directory"""
        print(os.path.abspath(self.project_dir))

    def do_configs(self, arg):
        """List available templates

        Usage:
          configs               # List all available templates
          configs --help        # Show this help message
        """
        # Handle help flags
        if arg and arg.strip() in ["--help", "-h", "help"]:
            print(self.do_configs.__doc__)
            return

        templates = self._get_available_templates()
        if templates:
            print("Available templates:")
            for template in sorted(templates):
                marker = " *" if template == self.current_template else ""
                print(f"  {template}{marker}")
        else:
            print("No templates found in current project directory")

    def do_config(self, arg):
        """Set current template: config <template_name>

        Usage:
          config                # Show current template
          config <template>     # Set current template
          config --help         # Show this help message
        """
        # Handle help flags
        if arg and arg.strip() in ["--help", "-h", "help"]:
            print(self.do_config.__doc__)
            return

        if not arg:
            if self.current_template:
                print(f"Current template: {self.current_template}")
            else:
                print("No template set")
            return

        templates = self._get_available_templates()
        if arg in templates:
            self.current_template = arg
            self._update_prompt()
            print(f"Set template to: {self.current_template}")
        else:
            print(f"Template not found: {arg}")
            if templates:
                print("Available templates:", ", ".join(sorted(templates)))

    def complete_config(self, text, line, begidx, endidx):
        """Tab completion for config command - supports nested template paths."""
        templates = self._get_available_templates()

        # Reconstruct the full template path being completed
        # For config command, the template starts after 'config '
        template_path = self._reconstruct_config_template_path(
            line, begidx, endidx, text
        )

        # Find matching templates
        matching_templates = [t for t in templates if t.startswith(template_path)]

        # Return the completion parts that should replace 'text'
        if template_path == text:
            # Simple case: text contains the full path
            return matching_templates
        else:
            # Complex case: text is only the part after the last '/'
            results = []
            for template in matching_templates:
                if "/" in template_path:
                    # Get the part after the last '/' in the template path
                    prefix = template_path.rsplit("/", 1)[0] + "/"
                    if template.startswith(prefix):
                        completion_part = template[len(prefix) :]
                        results.append(completion_part)
                else:
                    # No '/' in template_path, return the full template
                    results.append(template)
            return results

    def do_commands(self, arg):
        """List available commands

        Usage:
          commands              # List all available forgather commands
          commands --help       # Show this help message
        """
        # Handle help flags
        if arg and arg.strip() in ["--help", "-h", "help"]:
            print(self.do_commands.__doc__)
            return

        commands = self._get_available_commands()
        print("Available commands:")
        for command in sorted(commands):
            print(f"  {command}")

    def do_edit(self, arg):
        """Interactively select and edit template files

        Usage:
          edit                    # Interactive template selector (supports multiple files)
          edit <template_path>    # Edit specific template file
          edit --help             # Show this help message

        Environment Variables:
          VSCODE_IPC_HOOK_CLI    # VS Code IPC socket (auto-detected in VS Code terminals)
                                 # Copy from VS Code session to use elsewhere:
                                 # export VSCODE_IPC_HOOK_CLI=/tmp/vscode-ipc-*.sock
          VIM_SERVERNAME         # Vim server instance name for clientserver mode
          EDITOR                 # Preferred editor (default: vim, auto-detects VS Code)

        Editor Priority (highest first):
          1. VS Code Remote CLI (when VSCODE_IPC_HOOK_CLI is set)
          2. Vim Clientserver (when VIM_SERVERNAME is set)
          3. User's EDITOR environment variable
          4. Default vim

        Multiple File Selection:
          - Single file: 5
          - Multiple files: 1,3,7
          - Ranges: 1-5,8,10-12
        """
        # Handle help flags
        if arg and arg.strip() in ["--help", "-h", "help"]:
            print(self.do_edit.__doc__)
            return

        if arg:
            # Direct edit of specified file
            template_paths = [arg]
            template_path = arg
            if not os.path.isabs(template_path):
                # Try to resolve relative to project directory
                full_path = os.path.join(self.project_dir, template_path)
                if os.path.exists(full_path):
                    template_paths = [full_path]
                elif not os.path.exists(template_path):
                    print(f"Template file not found: {template_path}")
                    return
        else:
            # Interactive template selection (can return multiple files)
            template_paths = self._interactive_template_selector()
            if not template_paths:
                return

        # Determine editor to use - check for VS Code remote CLI first
        editor = self._get_best_editor()

        # Process each file - ensure they exist or create them if needed
        files_to_edit = []
        for template_path in template_paths:
            if not os.path.exists(template_path):
                # Ask if user wants to create a new file
                try:
                    create = (
                        input(
                            f"Template file doesn't exist. Create {template_path}? (y/N): "
                        )
                        .strip()
                        .lower()
                    )
                    if create not in ["y", "yes"]:
                        print(f"Skipping {template_path}")
                        continue

                    # Create directory if needed
                    os.makedirs(os.path.dirname(template_path), exist_ok=True)

                    # Create empty file
                    with open(template_path, "w") as f:
                        f.write("# New template file\n")
                except KeyboardInterrupt:
                    print("\nCancelled")
                    return

            files_to_edit.append(template_path)

        if not files_to_edit:
            print("No files to edit.")
            return

        # Launch editor with all files
        editor_name = os.path.basename(editor)

        # Check for vim clientserver mode
        vim_editor, vim_server = self._get_vim_server_info(editor)

        if len(files_to_edit) == 1:
            if vim_server:
                print(f"Opening {files_to_edit[0]} in vim server '{vim_server}'...")
                cmd_args = [
                    vim_editor,
                    "--servername",
                    vim_server,
                    "--remote-tab",
                    files_to_edit[0],
                ]
            else:
                print(f"Opening {files_to_edit[0]} in {editor_name}...")
                cmd_args = [editor, files_to_edit[0]]
        else:
            if vim_server:
                print(
                    f"Opening {len(files_to_edit)} files in vim server '{vim_server}' tabs..."
                )
                for f in files_to_edit:
                    rel_path = os.path.relpath(f, self.project_dir)
                    print(f"  - {rel_path}")
                # For vim clientserver, we need to open files one by one as tabs
                self._open_vim_server_tabs(vim_editor, vim_server, files_to_edit)
                return  # Skip the subprocess.run below since we handled it already
            else:
                print(f"Opening {len(files_to_edit)} files in {editor_name}...")
                for f in files_to_edit:
                    rel_path = os.path.relpath(f, self.project_dir)
                    print(f"  - {rel_path}")

                # Choose command args based on editor type
                if "remote-cli/code" in editor:
                    # VS Code remote CLI - just pass all files
                    cmd_args = [editor] + files_to_edit
                elif editor_name in ["vim", "nvim"]:
                    # Vim/Neovim - use -p flag for tabs
                    cmd_args = [editor, "-p"] + files_to_edit
                elif editor_name in ["code", "code.exe"]:
                    # VS Code standard - use -r flag to reuse window
                    cmd_args = [editor, "-r"] + files_to_edit
                else:
                    # For other editors, just pass all files
                    cmd_args = [editor] + files_to_edit

        try:
            subprocess.run(cmd_args)
            print("Editor closed.")

            # Invalidate template cache since files might have changed
            self._invalidate_cache()

        except FileNotFoundError:
            print(
                f"Editor '{editor}' not found. Set EDITOR environment variable to your preferred editor."
            )
        except Exception as e:
            print(f"Error launching editor: {e}")

    def complete_edit(self, text, line, begidx, endidx):
        """Tab completion for edit command - complete template file paths."""
        # Get all template files for completion
        templates = self._get_all_template_files()
        completions = []

        for template_name, template_path, rel_path in templates:
            # Offer both relative path and basename completion
            if rel_path.startswith(text):
                completions.append(rel_path)

            # Also offer completion by filename (basename of rel_path)
            basename = os.path.basename(rel_path)
            if basename.startswith(text):
                completions.append(rel_path)  # Return full relative path for clarity

        return sorted(set(completions))  # Remove duplicates and sort

    def do_debug(self, arg):
        """Toggle debug mode for tab completion"""
        if hasattr(self, "_debug_completion"):
            self._debug_completion = not self._debug_completion
        else:
            self._debug_completion = True

        status = "enabled" if self._debug_completion else "disabled"
        print(f"Debug mode {status}")

    def do_exit(self, arg):
        """Exit the interactive shell"""
        print("Goodbye!")
        return True

    def do_quit(self, arg):
        """Exit the interactive shell"""
        return self.do_exit(arg)

    def do_EOF(self, arg):
        """Exit on Ctrl+D"""
        print()
        return self.do_exit(arg)

    def default(self, line):
        """Handle forgather commands with optional -t template prefix."""
        if not line.strip():
            return

        # Parse the command line
        try:
            args = shlex.split(line)
        except ValueError as e:
            print(f"Error parsing command: {e}")
            return

        # Build the forgather command
        forgather_args = []

        # Add project directory if different from current
        if os.path.abspath(self.project_dir) != os.path.abspath("."):
            forgather_args.extend(["-p", self.project_dir])

        # Check if the command starts with -t (template specification)
        if args and args[0] == "-t":
            if len(args) < 2:
                print("Error: -t requires a template name")
                return
            forgather_args.extend(["-t", args[1]])
            command_args = args[2:]
        elif self.current_template:
            # Use current template if set
            forgather_args.extend(["-t", self.current_template])
            command_args = args
        else:
            # No template specified
            command_args = args

        # Add the actual command
        forgather_args.extend(command_args)

        if not command_args:
            print("Error: No command specified")
            return

        # Validate that the command exists
        commands = self._get_available_commands()
        if command_args[0] not in commands:
            print(f"Error: Unknown command '{command_args[0]}'")
            print("Available commands:", ", ".join(sorted(commands)))
            return

        # Execute the forgather command
        print(f"Executing: forgather {' '.join(forgather_args)}")

        # Check if this is a command that should use pager in interactive mode
        should_page = command_args[0] in ["pp", "graph", "code", "construct", "dataset"]

        if should_page:
            # Capture output for paging
            import io
            from contextlib import redirect_stdout

            captured_output = io.StringIO()
            try:
                # Save original sys.argv and replace with our args
                orig_argv = sys.argv[:]
                sys.argv = ["forgather"] + forgather_args

                # Capture stdout
                with redirect_stdout(captured_output):
                    cli_main()

                # Use pager for captured output
                output = captured_output.getvalue()
                if output.strip():  # Only page if there's actual output
                    self._page_output(output)

            except SystemExit as e:
                # Normal exit from CLI (e.g., help commands)
                # Show any captured output even if there was a SystemExit
                output = captured_output.getvalue()
                if output.strip():
                    self._page_output(output)
            except KeyboardInterrupt:
                print("\nCommand interrupted")
            except Exception as e:
                print(f"Error executing command: {e}")
                # Show any captured output even on error
                output = captured_output.getvalue()
                if output.strip():
                    self._page_output(output)
                traceback.print_exc()
            finally:
                # Restore original sys.argv
                sys.argv = orig_argv
        else:
            # Normal execution without paging
            try:
                # Save original sys.argv and replace with our args
                orig_argv = sys.argv[:]
                sys.argv = ["forgather"] + forgather_args

                # Call the main CLI function
                cli_main()

            except SystemExit as e:
                # Normal exit from CLI (e.g., help commands)
                pass
            except KeyboardInterrupt:
                print("\nCommand interrupted")
            except Exception as e:
                print(f"Error executing command: {e}")
                traceback.print_exc()
            finally:
                # Restore original sys.argv
                sys.argv = orig_argv

    def completenames(self, text, *ignored):
        """Tab completion for command names."""
        # First, try built-in commands
        builtins = [
            "cd",
            "pwd",
            "configs",
            "config",
            "commands",
            "edit",
            "debug",
            "exit",
            "quit",
        ]
        builtin_matches = [cmd for cmd in builtins if cmd.startswith(text)]

        # Then, try forgather subcommands
        commands = self._get_available_commands()
        command_matches = [cmd for cmd in commands if cmd.startswith(text)]

        # Also support -t prefix
        if text == "-t" or "-t".startswith(text):
            builtin_matches.append("-t")

        return builtin_matches + command_matches

    def _reconstruct_template_path(
        self, line: str, begidx: int, endidx: int, text: str
    ) -> str:
        """Reconstruct the full template path being completed.

        The cmd module treats '/' as word boundaries, so when completing
        'pipeline_llama_30b/1f', the text parameter might only contain '1f'.
        This method reconstructs the full path by looking at the command line.

        Args:
            line: Full command line
            begidx: Start position of text being completed
            endidx: End position of text being completed
            text: The text fragment being completed

        Returns:
            The full template path being completed
        """
        # Extract everything after '-t '
        t_pos = line.find("-t ")
        if t_pos == -1:
            return text

        # Get the template part starting after '-t '
        template_start = t_pos + 3  # After '-t '
        template_part = line[template_start:endidx]

        # Remove any trailing spaces
        template_part = template_part.strip()

        return template_part

    def _reconstruct_config_template_path(
        self, line: str, begidx: int, endidx: int, text: str
    ) -> str:
        """Reconstruct the full template path being completed for config command.

        Similar to _reconstruct_template_path but for 'config' command instead of '-t'.

        Args:
            line: Full command line
            begidx: Start position of text being completed
            endidx: End position of text being completed
            text: The text fragment being completed

        Returns:
            The full template path being completed
        """
        # Extract everything after 'config '
        config_pos = line.find("config ")
        if config_pos == -1:
            return text

        # Get the template part starting after 'config '
        template_start = config_pos + 7  # After 'config '
        template_part = line[template_start:endidx]

        # Remove any trailing spaces
        template_part = template_part.strip()

        return template_part

    def completedefault(self, text, line, begidx, endidx):
        """Tab completion for default commands (forgather subcommands)."""
        # Debug output to help diagnose completion issues
        if hasattr(self, "_debug_completion") and self._debug_completion:
            print(f"\n[DEBUG] completedefault called:")
            print(f"  text: '{text}'")
            print(f"  line: '{line}'")
            print(f"  begidx: {begidx}, endidx: {endidx}")
            print(f"  line[:begidx]: '{line[:begidx]}'")
            print(f"  line[begidx:endidx]: '{line[begidx:endidx]}'")

        args = shlex.split(line[:begidx])

        # Handle template completion after -t
        # Check if -t is anywhere in the args and we're completing a template
        if len(args) >= 2 and "-t" in args:
            # Find the position of -t
            t_index = None
            for i, arg in enumerate(args):
                if arg == "-t":
                    t_index = i
                    break

            if t_index is not None and t_index < len(args) - 1:
                # We're completing something after -t (either a new template or continuing one)
                templates = self._get_available_templates()

                # Reconstruct the full template path being completed
                template_path = self._reconstruct_template_path(
                    line, begidx, endidx, text
                )

                # Find matching templates
                matching_templates = [
                    t for t in templates if t.startswith(template_path)
                ]

                # Return the completion parts that should replace 'text'
                if template_path == text:
                    # Simple case: text contains the full path
                    return matching_templates
                else:
                    # Complex case: text is only the part after the last '/'
                    results = []
                    for template in matching_templates:
                        if "/" in template_path:
                            # Get the part after the last '/' in the template path
                            prefix = template_path.rsplit("/", 1)[0] + "/"
                            if template.startswith(prefix):
                                completion_part = template[len(prefix) :]
                                results.append(completion_part)
                        else:
                            # No '/' in template_path, return the full template
                            results.append(template)
                    return results
        elif len(args) >= 1 and args[-1] == "-t":
            # Simple case: completing right after -t with no partial template yet
            templates = self._get_available_templates()
            return [t for t in templates if t.startswith(text)]

        # Handle target completion after --target
        if len(args) >= 1 and args[-1] == "--target":
            targets = self._get_available_targets()
            return [t for t in targets if t.startswith(text)]

        # Handle command completion
        if not args or (len(args) == 1 and not line[:begidx].endswith(" ")):
            # Completing first word - show commands
            commands = self._get_available_commands()
            return [cmd for cmd in commands if cmd.startswith(text)]

        # Simple filesystem completion - use the same approach as cd command
        # Extract the full path being completed from the command line
        # Get everything after the command and flags by finding the actual path argument
        words = line.split()
        path_to_complete = text  # Default to the text being completed

        # Try to find a more complete path context by looking at the line
        if len(words) > 1:
            # Look for the last argument that looks like a path
            for word in reversed(words[1:]):
                if (
                    "/" in word
                    or word in [".", ".."]
                    or word.startswith("~")
                    or word.startswith("./")
                ):
                    # This looks like a path - use it as context
                    if word.endswith(text) and text:
                        # The text is the end of this path
                        path_to_complete = word
                    elif not text and word.endswith("/"):
                        # We're completing inside this directory
                        path_to_complete = word
                    elif not text:
                        # Empty text at end of line - complete this path
                        # This handles cases like "~/path/partial-name" where we want to complete "partial-name"
                        path_to_complete = word
                    break

        # Get completions using the same method as cd
        completions = self._complete_path(path_to_complete, only_dirs=False)

        # Handle expandable path completion - check if this is an expansion case
        # Also handle case where text is empty but path_to_complete is expandable
        is_expandable_path = (text and (text.startswith("~") or "$" in text)) or (
            not text
            and path_to_complete
            and (path_to_complete.startswith("~") or "$" in path_to_complete)
        )

        if is_expandable_path:
            # For expandable paths, we need different logic
            # The completions already have full expanded paths
            # We need to return what should replace the expandable text

            # Handle the case where text is empty but we have an expandable path_to_complete
            if not text and path_to_complete:
                # Use path_to_complete for expansion logic
                expanded_text = os.path.expanduser(path_to_complete)
                if "$" in path_to_complete:
                    expanded_text = os.path.expandvars(expanded_text)
                original_path = path_to_complete
            else:
                # Use text for expansion logic (original case)
                expanded_text = os.path.expanduser(text)
                if "$" in text:
                    expanded_text = os.path.expandvars(expanded_text)
                original_path = text

            text_basename = os.path.basename(expanded_text)

            # Filter completions based on the basename pattern
            filtered = []
            for comp in completions:
                comp_basename = os.path.basename(comp.rstrip("/"))
                if comp_basename.startswith(text_basename):
                    if not text and path_to_complete:
                        # Special case: text is empty, we're completing a partial path
                        # Return just the suffix that completes the partial name
                        # For ~/ai_assets/models/llama-2- completing to llama-2-30b-fg/
                        # We want to return just the suffix: 30b-fg/
                        if comp_basename.startswith(text_basename):
                            suffix = comp_basename[len(text_basename) :]
                            if comp.endswith("/"):
                                suffix += "/"
                            filtered.append(suffix)
                    else:
                        # Original logic for when text is not empty
                        if original_path.startswith("~"):
                            # Keep the tilde format
                            expanded_prefix = os.path.expanduser("~")
                            if comp.startswith(expanded_prefix):
                                # Replace the expanded home with ~
                                completed_path = "~" + comp[len(expanded_prefix) :]
                                filtered.append(completed_path)
                            else:
                                filtered.append(comp)
                        else:
                            # For $HOME or other env vars, return expanded path
                            filtered.append(comp)

            return filtered
        else:
            # Regular filesystem completion (non-expandable paths)
            if text:
                filtered = [
                    comp
                    for comp in completions
                    if os.path.basename(comp.rstrip("/")).startswith(text)
                ]
                # Return just the filename part that would replace 'text'
                return [
                    os.path.basename(comp.rstrip("/"))
                    + ("/" if comp.endswith("/") else "")
                    for comp in filtered
                ]
            else:
                # Return just the filename parts, not full paths
                return [
                    os.path.basename(comp.rstrip("/"))
                    + ("/" if comp.endswith("/") else "")
                    for comp in completions
                ]


def interactive_main(project_dir: str = "."):
    """Start the interactive forgather shell."""
    try:
        shell = ForgatherShell(project_dir)
        shell.cmdloop()
    except KeyboardInterrupt:
        print("\nExiting...")
        sys.exit(0)


if __name__ == "__main__":
    interactive_main()
