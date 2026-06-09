"""Main CLI controller and argument parsing."""

import argparse
import logging
import os
import sys
from argparse import RawTextHelpFormatter

from .commands import (
    code_cmd,
    construct_cmd,
    graph_cmd,
    index_cmd,
    ls_cmd,
    meta_cmd,
    pp_cmd,
    targets_cmd,
    tb_cmd,
    template_list,
)
from .dynamic_args import partition_args


def parse_global_args(args=None):
    """Parse global arguments and return global args + remaining args for subcommand."""
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Forgather CLI",
        epilog=(""),
        add_help=False,  # We'll handle help at the subcommand level
    )

    parser.add_argument(
        "-p",
        "--project-dir",
        type=str,
        default=".",
    )

    parser.add_argument(
        "-t",
        "--config-template",
        type=str,
        default=None,
        help="Configuration Template Name",
    )

    parser.add_argument(
        "--no-dyn",
        action="store_true",
        help="Disable processing of dynamic args defined in configuration templates",
    )

    parser.add_argument(
        "-i",
        "--interactive",
        action="store_true",
        help="Start interactive shell with tab completion",
    )

    # Parse known args to separate global from subcommand args
    global_args, remaining_args = parser.parse_known_args(args)

    return global_args, remaining_args


# Global options that take a value (the following token is the value, not the
# subcommand). Mirrors parse_global_args.
_GLOBAL_VALUE_FLAGS = ("-p", "--project-dir", "-t", "--config-template")


def _subcommand_token_index(tokens):
    """Index of the subcommand token (the first positional), skipping leading
    global options.

    Used to gate the per-command arg carve-outs so a *positional argument* that
    happens to equal a command name — e.g. the word "server" inside a
    ``forgather docs search …`` query — does not trigger another command's
    arg-rewriting. Returns ``None`` if there is no positional token.
    """
    i = 0
    while i < len(tokens):
        tok = tokens[i]
        if tok.startswith("--") and "=" in tok:
            i += 1  # --opt=value is a single token
        elif tok in _GLOBAL_VALUE_FLAGS:
            i += 2  # value-taking global flag consumes the next token
        elif tok.startswith("-"):
            i += 1  # boolean / unknown global flag
        else:
            return i
    return None


def get_subcommand_registry():
    """Registry of all available subcommands and their argument parsers."""
    from .agent_args import create_agent_parser
    from .checkpoint_args import create_checkpoint_parser
    from .cluster_args import create_cluster_parser
    from .commands_args import (
        create_code_parser,
        create_construct_parser,
        create_graph_parser,
        create_index_parser,
        create_ls_parser,
        create_meta_parser,
        create_pp_parser,
        create_targets_parser,
        create_tb_parser,
        create_tlist_parser,
    )
    from .dataset_args import create_dataset_parser
    from .dataset_server_args import create_dataset_server_parser
    from .diloco_args import create_diloco_parser
    from .docs_args import create_docs_parser, create_search_parser
    from .eval_args import create_eval_parser
    from .gpu_args import create_gpu_parser
    from .job_args import create_job_parser
    from .logs_args import create_logs_parser
    from .mkdocs_args import create_mkdocs_parser
    from .model_args import create_model_parser
    from .plot_args import create_plot_parser
    from .project_args import create_project_parser
    from .sched_args import create_sched_parser
    from .submit_args import create_submit_parser
    from .tls_args import create_tls_parser
    from .train_args import create_train_parser
    from .trefs_args import create_trefs_parser
    from .workspace_args import create_ws_parser
    from .wrappers_args import (
        create_convert_parser,
        create_finalize_parser,
        create_inf_parser,
        create_server_parser,
        create_update_parser,
    )

    return {
        "index": create_index_parser,
        "ls": create_ls_parser,
        "meta": create_meta_parser,
        "targets": create_targets_parser,
        "tlist": create_tlist_parser,
        "graph": create_graph_parser,
        "trefs": create_trefs_parser,
        "pp": create_pp_parser,
        "tb": create_tb_parser,
        "code": create_code_parser,
        "construct": create_construct_parser,
        "train": create_train_parser,
        "submit": create_submit_parser,
        "dataset": create_dataset_parser,
        "dataset-server": create_dataset_server_parser,
        "ws": create_ws_parser,
        "model": create_model_parser,
        "project": create_project_parser,
        "inf": create_inf_parser,
        "server": create_server_parser,
        "convert": create_convert_parser,
        "finalize": create_finalize_parser,
        "update": create_update_parser,
        "checkpoint": create_checkpoint_parser,
        "logs": create_logs_parser,
        "plot": create_plot_parser,
        "diloco": create_diloco_parser,
        "eval": create_eval_parser,
        "sched": create_sched_parser,
        "job": create_job_parser,
        "gpu": create_gpu_parser,
        "cluster": create_cluster_parser,
        "mkdocs": create_mkdocs_parser,
        "docs": create_docs_parser,
        "search": create_search_parser,  # alias for `docs search`
        "tls": create_tls_parser,
        "agent": create_agent_parser,
    }


def _summarize_description(description, max_len=62):
    """Reduce a parser ``description`` to a single-line summary.

    Several subcommands stuff full usage, subcommand lists, or workflow
    examples into their ``description=`` so it shows under their own
    ``--help``. The top-level listing only wants one line each, so take the
    first non-empty line and length-cap it. Each subcommand's own
    ``--help`` still renders the full text.
    """
    if not description:
        return ""
    first_line = next(
        (line.strip() for line in description.splitlines() if line.strip()), ""
    )
    if len(first_line) > max_len:
        first_line = first_line[: max_len - 1].rstrip() + "…"
    return first_line


def iter_command_summaries():
    """Yield ``(name, one-line summary)`` for every subcommand, sorted by name.

    Shared by the top-level ``--help`` and the interactive ``commands`` listing.
    Uses a ``no_dyn=True`` dummy so building each parser doesn't load a project
    config (and stays fast / quiet).
    """
    registry = get_subcommand_registry()
    dummy_global_args = argparse.Namespace(
        project_dir=".", config_template=None, no_dyn=True
    )
    for cmd_name in sorted(registry.keys()):
        try:
            summary = _summarize_description(
                registry[cmd_name](dummy_global_args).description
            )
        except Exception:
            summary = "[Error loading description]"
        yield cmd_name, summary


def show_main_help():
    """Show the main help message with available subcommands."""
    print("Forgather CLI")
    print()
    print("Usage: forgather [global options] <subcommand> [subcommand options]")
    print(
        "       forgather -i                      # Interactive mode with tab completion"
    )
    print()
    print("Global options:")
    print("  -p, --project-dir DIR    Project directory (default: current directory)")
    print("  -t, --config-template T  Configuration template name")
    print("  -i, --interactive        Start interactive shell with tab completion")
    print("  --no-dyn                 Disable dynamic help (from config meta-data)")
    print("  --help                   Show this help message")
    print()
    print("Available subcommands:")
    for cmd_name, summary in iter_command_summaries():
        print(f"  {cmd_name:<14} {summary}")
    print()
    print("Use 'forgather <subcommand> --help' for help on a specific subcommand.")


def parse_args(args=None):
    """Parse arguments with dynamic subcommand handling."""
    # Special handling for commands that have flag conflicts with global args
    # - 'inf' command: --interactive/-i conflicts with global interactive mode
    # - 'convert' command: -t conflicts with global --config-template
    args_list = args if args is not None else sys.argv[1:]

    # Track which flags to restore and workarounds needed
    inf_interactive_workaround = False
    convert_t_workaround = False
    convert_original_args = None  # Save original args for convert command
    finalize_t_workaround = False
    finalize_original_args = None  # Save original args for finalize command
    server_original_args = None  # Save original args for server command
    dataset_server_original_args = None  # Save original args for dataset-server
    removed_flags = []

    # The subcommand is the first positional token (after any leading global
    # options). Gate EVERY per-command carve-out below on it, so a positional
    # argument that happens to equal a command name — e.g. the word
    # "server"/"convert"/"finalize" inside a `docs search …` query — never
    # triggers another command's arg-rewriting. (It also means exactly one
    # carve-out can fire per invocation, instead of any whose name appears.)
    _sub_idx = _subcommand_token_index(args_list)
    _sub = args_list[_sub_idx] if _sub_idx is not None else None

    # Handle 'inf' command with --interactive/-i
    if _sub == "inf":
        inf_idx = _sub_idx
        remaining_after_inf = args_list[inf_idx + 1 :]
        if "--interactive" in remaining_after_inf or "-i" in remaining_after_inf:
            inf_interactive_workaround = True
            args_for_global = args_list.copy()
            if "--interactive" in remaining_after_inf:
                args_for_global.remove("--interactive")
                removed_flags.append(
                    (
                        "--interactive",
                        inf_idx + remaining_after_inf.index("--interactive") + 1,
                    )
                )
            if "-i" in remaining_after_inf:
                args_for_global.remove("-i")
                removed_flags.append(
                    ("-i", inf_idx + remaining_after_inf.index("-i") + 1)
                )
            args_list = args_for_global

    # Handle 'convert' command with -t
    # Save original args for convert, then remove -t to prevent global parser from consuming it
    if _sub == "convert":
        convert_idx = _sub_idx
        remaining_after_convert = args_list[convert_idx + 1 :]
        # Save the original args after 'convert' for later use
        convert_original_args = remaining_after_convert.copy()
        # Check if -t appears after 'convert' (not before, which would be global -t)
        if "-t" in remaining_after_convert:
            convert_t_workaround = True
            args_for_global = args_list.copy()
            # Find -t after convert and remove it along with its value for global parsing
            t_idx_in_remaining = remaining_after_convert.index("-t")
            t_idx_in_full = convert_idx + 1 + t_idx_in_remaining
            args_for_global.pop(t_idx_in_full)  # Remove -t
            # Also remove the value after -t if it exists and doesn't start with -
            if t_idx_in_full < len(args_for_global) and not args_for_global[
                t_idx_in_full
            ].startswith("-"):
                args_for_global.pop(t_idx_in_full)
            args_list = args_for_global

    # Handle 'server' command — forwards everything; `-p` means --port to the
    # server, not the global --project-dir. Strip anything after `server` from
    # what the global parser sees and preserve the original tokens verbatim.
    # Applies only when `server` is the actual subcommand — never when the word
    # "server" appears as a positional (e.g. inside a `docs search …` query, or
    # as `inf server` / `cluster server`, where the parent owns the subtree).
    if _sub == "server":
        server_original_args = args_list[_sub_idx + 1 :]
        args_list = args_list[: _sub_idx + 1]

    # 'dataset-server start' uses REMAINDER passthrough so the
    # underlying script's --port (-p) doesn't conflict with the global
    # -p / --project-dir; --help and other server flags should also
    # reach the script verbatim. The diagnostic actions (status, list,
    # cache, local) are parsed normally — they have no flag conflicts.
    if _sub == "dataset-server":
        ds_idx = _sub_idx
        after_ds = args_list[ds_idx + 1 :]
        if after_ds and after_ds[0] == "start":
            # Capture everything after 'start' as the remainder; keep
            # 'dataset-server start' visible to the global parser so
            # the subcommand is still routed correctly.
            dataset_server_original_args = after_ds[1:]
            args_list = args_list[: ds_idx + 2]

    # Handle 'finalize' command with -t (same conflict as convert; -t is the
    # finalize chat-template-path flag)
    if _sub == "finalize":
        fin_idx = _sub_idx
        remaining_after_fin = args_list[fin_idx + 1 :]
        finalize_original_args = remaining_after_fin.copy()
        if "-t" in remaining_after_fin:
            finalize_t_workaround = True
            args_for_global = args_list.copy()
            t_idx_in_remaining = remaining_after_fin.index("-t")
            t_idx_in_full = fin_idx + 1 + t_idx_in_remaining
            args_for_global.pop(t_idx_in_full)  # Remove -t
            if t_idx_in_full < len(args_for_global) and not args_for_global[
                t_idx_in_full
            ].startswith("-"):
                args_for_global.pop(t_idx_in_full)
            args_list = args_for_global

    # Parse global args with potentially modified args_list
    global_args, remaining_args = parse_global_args(args_list)

    # Restore removed flags to remaining_args
    if removed_flags:
        # Sort by position to restore in correct order
        removed_flags.sort(key=lambda x: x[1])
        # Find the subcommand in remaining_args to know where to insert
        subcommand = None
        if remaining_args:
            subcommand = remaining_args[0]

        if subcommand in ["inf", "convert"]:
            subcommand_idx = remaining_args.index(subcommand)
            # Insert flags after the subcommand (and after any positional arg for inf)
            insert_pos = (
                subcommand_idx + 2
                if subcommand == "inf" and len(remaining_args) > subcommand_idx + 1
                else subcommand_idx + 1
            )
            for flag, _ in removed_flags:
                remaining_args.insert(insert_pos, flag)
                insert_pos += 1

    # Handle interactive mode (but skip if it was the inf command workaround)
    if global_args.interactive and not inf_interactive_workaround:
        from .interactive import interactive_main

        interactive_main(global_args.project_dir)
        sys.exit(0)

    # Handle case where no subcommand is provided or --help is requested globally
    if not remaining_args or (remaining_args and remaining_args[0] in ["--help", "-h"]):
        show_main_help()
        sys.exit(0)

    # Extract subcommand name
    subcommand = remaining_args[0]
    subcommand_args = remaining_args[1:]

    # Get subcommand registry
    registry = get_subcommand_registry()

    # Check if subcommand exists
    if subcommand not in registry:
        if subcommand == "control":
            # 'control' was the local, server-less trainer-control CLI. It's
            # removed now that the forgather server is the default control
            # plane: server-managed jobs are controlled via 'forgather job'.
            print(
                "Error: 'forgather control' was removed. Use 'forgather job "
                "<save|stop|save-stop|abort|status|tail|logs>' against the "
                "forgather server instead (start one with 'forgather server').",
                file=sys.stderr,
            )
            sys.exit(1)
        print(f"Error: Unknown subcommand '{subcommand}'")
        print()
        show_main_help()
        sys.exit(1)

    # Create subcommand parser and parse its arguments
    subcommand_parser = registry[subcommand](global_args)

    try:
        # For convert, finalize, and server commands, pass all args as remainder
        # without parsing. Their flags conflict with global flags (e.g. server's
        # -p/--port vs global -p/--project-dir).
        # `dataset-server start` is REMAINDER-passthrough; other
        # `dataset-server` subactions parse normally.
        ds_is_start = (
            subcommand == "dataset-server"
            and subcommand_args
            and subcommand_args[0] == "start"
        )
        if subcommand in ["convert", "finalize", "server", "update"]:
            sub_args = argparse.Namespace()
            if subcommand == "convert":
                sub_args.remainder = (
                    convert_original_args
                    if convert_original_args is not None
                    else subcommand_args
                )
            elif subcommand == "finalize":
                sub_args.remainder = (
                    finalize_original_args
                    if finalize_original_args is not None
                    else subcommand_args
                )
            elif subcommand == "server":
                sub_args.remainder = (
                    server_original_args
                    if server_original_args is not None
                    else subcommand_args
                )
            else:  # update — forwards everything (incl. --help) to update.py
                sub_args.remainder = subcommand_args
            sub_args.dummy = ""
        elif ds_is_start:
            # 'dataset-server start' flags were carved out of the global parse
            # so the server's -p/-H don't collide with global -p/--project-dir.
            # Parse them through the dataset-server subparser now; parse_known
            # keeps any extra server flags (e.g. TLS) for the --local-only
            # foreground path to forward verbatim.
            start_tokens = (
                dataset_server_original_args
                if dataset_server_original_args is not None
                else subcommand_args[1:]
            )
            sub_args, ds_extra = subcommand_parser.parse_known_args(
                ["start"] + start_tokens
            )
            sub_args.extra = ds_extra
        else:
            sub_args = subcommand_parser.parse_args(subcommand_args)
    except SystemExit:
        # argparse calls sys.exit on help or error - let it through
        raise

    # Get dynamic argument names from the parser (if available)
    dynamic_arg_names = getattr(subcommand_parser, "_dynamic_arg_names", [])

    # Partition the subcommand arguments
    if dynamic_arg_names:
        built_in_sub_args, dynamic_sub_args = partition_args(
            sub_args, dynamic_arg_names
        )
    else:
        built_in_sub_args = vars(sub_args)
        dynamic_sub_args = {}

    # Combine global and built-in subcommand args into a single namespace
    combined_args = argparse.Namespace()

    # Add global args
    for key, value in vars(global_args).items():
        setattr(combined_args, key, value)

    # Add built-in subcommand args
    for key, value in built_in_sub_args.items():
        setattr(combined_args, key, value)

    # Add the command name
    combined_args.command = subcommand

    # Store dynamic args separately for easy access
    combined_args._dynamic_args = dynamic_sub_args

    return combined_args


def main():
    logging.basicConfig(level=logging.WARNING)
    """Main CLI entry point."""
    try:
        args = parse_args()
        match args.command:
            case "index":
                index_cmd(args)
            case "ls":
                ls_cmd(args)
            case "meta":
                meta_cmd(args)
            case "targets":
                targets_cmd(args)
            case "tlist":
                template_list(args)
            case "graph":
                graph_cmd(args)
            case "trefs":
                from .trefs import trefs_cmd

                trefs_cmd(args)
            case "pp":
                pp_cmd(args)
            case "checkpoint":
                from .checkpoint import checkpoint_cmd

                checkpoint_cmd(args)
            case "tb":
                tb_cmd(args)
            case "code":
                code_cmd(args)
            case "construct":
                construct_cmd(args)
            case "train":
                from .train import train_cmd

                train_cmd(args)
            case "submit":
                from .submit import submit_cmd

                rc = submit_cmd(args)
                if rc:
                    sys.exit(rc)
            case "dataset":
                from .dataset import dataset_cmd

                dataset_cmd(args)
            case "dataset-server":
                from .dataset_server import dataset_server_cmd

                rc = dataset_server_cmd(args)
                if rc is not None and rc != 0:
                    sys.exit(rc)
            case "ws":
                from .workspace import ws_cmd

                ws_cmd(args)

            case "project":
                from .project import project_cmd

                project_cmd(args)
            case "model":
                from .model import model_cmd

                model_cmd(args)
            case "inf":
                from .inference import inf_cmd

                inf_cmd(args)
            case "server":
                from .server import server_cmd

                server_cmd(args)
            case "convert":
                from .convert import convert_cmd

                convert_cmd(args)
            case "finalize":
                from .finalize import finalize_cmd

                finalize_cmd(args)
            case "update":
                from .update import update_cmd

                update_cmd(args)
            case "logs":
                from .logs import logs_cmd

                logs_cmd(args)
            case "plot":
                from .plot import plot_cmd

                plot_cmd(args)
            case "diloco":
                from .diloco import diloco_cmd

                # Propagate the subcommand's exit code so the new
                # read/diagnostic verbs (status / servers / logs) are
                # scriptable (non-zero on error).
                sys.exit(diloco_cmd(args) or 0)
            case "eval":
                from .eval import eval_cmd

                eval_cmd(args)
            case "sched":
                from .sched import sched_cmd

                sched_cmd(args)
            case "job":
                from .job import job_cmd

                job_cmd(args)
            case "agent":
                from .agent import agent_cmd

                sys.exit(agent_cmd(args) or 0)
            case "gpu":
                from .gpu import gpu_cmd

                gpu_cmd(args)
            case "cluster":
                from .cluster import cluster_cmd

                rc = cluster_cmd(args)
                if rc is not None and rc != 0:
                    sys.exit(rc)
            case "mkdocs":
                from .mkdocs import mkdocs_cmd

                mkdocs_cmd(args)
            case "docs":
                from .docs import docs_cmd

                rc = docs_cmd(args)
                if rc:
                    sys.exit(rc)
            case "search":
                # Top-level alias for `forgather docs search` (the parser sets
                # docs_action="search" so the namespace dispatches correctly).
                from .docs import docs_cmd

                rc = docs_cmd(args)
                if rc:
                    sys.exit(rc)
            case "tls":
                from .tls import tls_cmd

                rc = tls_cmd(args)
                if rc:
                    sys.exit(rc)
            case _:
                index_cmd(args)
    except SystemExit:
        # Let argparse's sys.exit calls through (for help, errors, etc.)
        raise
    except KeyboardInterrupt:
        sys.exit(1)
