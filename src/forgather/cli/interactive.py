"""Interactive CLI shell with tab completion for forgather."""

import cmd
import shlex
import sys
import os
import subprocess
import shutil
from typing import List, Optional
import traceback

from forgather.meta_config import MetaConfig
from .commands import get_env
from .main import get_subcommand_registry, parse_args, main as cli_main


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

    def _interactive_template_selector(self) -> Optional[str]:
        """Interactive template selector with arrow key navigation.

        Returns:
            Path to selected template file, or None if cancelled
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

        # Get user selection
        while True:
            try:
                choice = input(f"Select template (0-{len(template_list)}): ").strip()
                if choice == "0" or choice.lower() in ["cancel", "quit", "exit"]:
                    return None

                choice_num = int(choice)
                if 1 <= choice_num <= len(template_list):
                    selected = template_list[choice_num - 1]
                    return selected[1]  # Return full path
                else:
                    print(f"Please enter a number between 0 and {len(template_list)}")
            except ValueError:
                print("Please enter a valid number")
            except KeyboardInterrupt:
                print("\nCancelled")
                return None

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

        Supports tab completion for directory paths.
        Examples: cd <TAB>, cd tu<TAB>, cd ~/pr<TAB>
        """
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
        """List available templates"""
        templates = self._get_available_templates()
        if templates:
            print("Available templates:")
            for template in sorted(templates):
                marker = " *" if template == self.current_template else ""
                print(f"  {template}{marker}")
        else:
            print("No templates found in current project directory")

    def do_config(self, arg):
        """Set current template: template <template_name>"""
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
        """Tab completion for template command."""
        templates = self._get_available_templates()
        return [t for t in templates if t.startswith(text)]

    def do_commands(self, arg):
        """List available commands"""
        commands = self._get_available_commands()
        print("Available commands:")
        for command in sorted(commands):
            print(f"  {command}")

    def do_edit(self, arg):
        """Interactively select and edit a template file

        Usage:
          edit                    # Interactive template selector
          edit <template_path>    # Edit specific template file
        """
        if arg:
            # Direct edit of specified file
            template_path = arg
            if not os.path.isabs(template_path):
                # Try to resolve relative to project directory
                full_path = os.path.join(self.project_dir, template_path)
                if os.path.exists(full_path):
                    template_path = full_path
                elif not os.path.exists(template_path):
                    print(f"Template file not found: {template_path}")
                    return
        else:
            # Interactive template selection
            template_path = self._interactive_template_selector()
            if not template_path:
                return

        # Determine editor to use
        editor = os.environ.get("EDITOR", "vim")

        # Ensure the file exists (create if it's a new template)
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
                    return

                # Create directory if needed
                os.makedirs(os.path.dirname(template_path), exist_ok=True)

                # Create empty file
                with open(template_path, "w") as f:
                    f.write("# New template file\n")
            except KeyboardInterrupt:
                print("\nCancelled")
                return

        # Launch editor
        print(f"Opening {template_path} in {editor}...")
        try:
            subprocess.run([editor, template_path])
            print("Editor closed.")

            # Invalidate template cache since file might have changed
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

    def completedefault(self, text, line, begidx, endidx):
        """Tab completion for default commands (forgather subcommands)."""
        args = shlex.split(line[:begidx])

        # Handle template completion after -t
        if len(args) >= 1 and args[-1] == "-t":
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
                    break

        # Get completions using the same method as cd
        completions = self._complete_path(path_to_complete, only_dirs=False)

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
