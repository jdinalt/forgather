#!/usr/bin/env python3

import os
import argparse
from argparse import RawTextHelpFormatter
import sys
from pprint import pp
import subprocess
import glob

from forgather.project import Project
import forgather.nb.notebooks as nb
from forgather.meta_config import preprocessor_globals, MetaConfig
from forgather.config import ConfigEnvironment, fconfig
from forgather.codegen import generate_code
from forgather.yaml_encoder import to_yaml
from forgather.latent import Latent
from forgather.template_utils import (
    get_extends_graph,
    template_extends_iter,
    template_data_iter,
    extends_graph_iter,
)


def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Forgather CLI",
        epilog=(""),
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

    subparsers = parser.add_subparsers(dest="command", help="subcommand help")

    index_parser = subparsers.add_parser("index", help="Show project index")
    ls_parser = subparsers.add_parser("ls", help="List available configurations")
    pp_parser = subparsers.add_parser("pp", help="Preprocess configuration")
    referenced_templates_parser = subparsers.add_parser(
        "trefs", help="List referenced templates"
    )
    all_templates_parser = subparsers.add_parser(
        "tlist", help="List available templates."
    )
    meta_parser = subparsers.add_parser("meta", help="Show meta configuration")
    targets_parser = subparsers.add_parser("targets", help="Show output targets")
    code_parser = subparsers.add_parser(
        "code", help="Output configuration as Python code"
    )
    construct_parser = subparsers.add_parser(
        "construct", help="Materialize and print a target"
    )
    graph_parser = subparsers.add_parser(
        "graph", help="Preprocess and parse into node graph"
    )
    tb_parser = subparsers.add_parser("tb", help="Start Tensorboard for project")
    train_parser = subparsers.add_parser(
        "train", help="Run configuration with train script"
    )

    code_parser.add_argument(
        "--target",
        type=str,
        default="main",
        help="Output target name",
    )

    construct_parser.add_argument(
        "--target",
        type=str,
        default="main",
        help="Output target name",
    )


    construct_parser.add_argument(
        "--call",
        action="store_true",
        help="Call the materialized object",
    )


    graph_parser.add_argument(
        "--format",
        type=str,
        choices=["none", "repr", "yaml", "fconfig", "python"],
        default="yaml",
        help="Graph format",
    )

    all_templates_parser.add_argument(
        "--format",
        type=str,
        choices=["md", "files"],
        default="files",
        help="Output format.",
    )

    referenced_templates_parser.add_argument(
        "--format",
        type=str,
        choices=["md", "files"],
        default="files",
        help="Output format.",
    )

    tb_parser.add_argument(
        "--all",
        action="store_true",
        help="Configure TB to watch all model directories",
    )

    tb_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="All arguments after -- will be forwarded as Tensroboard arguments.",
    )

    tb_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the generated commandline, without actually executing it.",
    )

    ls_parser.add_argument(
        "--recursive",
        "-r",
        action="store_true",
        help="Search for project in all sub-directories and list them.",
    )

    train_parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default=None,
        help='CUDA Visible Devices e.g. "0,1"',
    )

    train_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just show the generated commandline, without actually executing it.",
    )

    train_parser.add_argument(
        "remainder",
        nargs=argparse.REMAINDER,
        help="All arguments after -- will be forwarded as torchrun arguments.",
    )

    args = parser.parse_args(args)

    return args

def set_default_template(meta, args):
    if not args.config_template:
        default_config = meta.default_config()
        args.config_template = default_config

def get_env(meta, project_dir):
    # Create new config environment and load configuration
    environment = ConfigEnvironment(
        searchpath=meta.searchpath,
        global_vars=preprocessor_globals(project_dir, meta.workspace_root),
    )
    return environment


def get_config(meta, env, config_template):
    return env.load(meta.config_path(config_template)).get()

def list_configurations(args):
    if args.recursive:
        for root, dirs, files in os.walk(args.project_dir):
            for file_name in files:
                if file_name == "meta.yaml":
                    print(f"\nProject Path: {root}\n")
                    try:
                        list_project(root)
                    except:
                        print(f"PARSE ERROR: {os.path.join(root, file_name)}")
    else:
        list_project(args.project_dir)

def list_project(project_dir):
    meta = MetaConfig(project_dir)
    meta_config = meta.config_dict
    project_name = meta_config.get("name", "Anonymous")
    project_description = meta_config.get("description", "No Description")
    print(f"{project_name} : {project_description}")
    env = get_env(meta, project_dir)
    for config_name, path in meta.find_templates(meta.config_prefix):
        try:
            config, pp_config = get_config(meta, env, config_name)
            config_meta = Latent.materialize(config.meta)
            config_long_name = config_meta.get("config_name", "Anonymous")
            config_description = config_meta.get("config_description", "No Description")
        except Exception as e:
            config_long_name = "PARSE ERROR"
            config_description = "An error occured while parsing the configuration."
        if config_name == meta.default_config():
            config_name = f"[{config_name}]"
        print(f"    {config_name:<30} {config_long_name} : {config_description}")


def list_targets(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    env = get_env(meta, args.project_dir)
    config, pp_config = get_config(meta, env, args.config_template)
    s = ""
    for target in config.keys():
        s += f"{target}\n"
    print(s)


def preprocess(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    env = get_env(meta, args.project_dir)
    pp_config = env.preprocess(meta.config_path(args.config_template))
    print(pp_config)


def as_code(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    env = get_env(meta, args.project_dir)
    config, pp_config = get_config(meta, env, args.config_template)
    code = generate_code(config[args.target])
    print(code)


def construct(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    proj = Project(args.config_template, args.project_dir)
    target = proj(args.target)
    if args.call:
        target = target()
    pp(target)


def show_meta(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    md = nb.render_meta(meta, "# Meta Config\n")
    print(md)


def list_referenced_templates(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    env = get_env(meta, args.project_dir)

    match args.format:
        case "md":
            print(
                nb.render_referenced_templates_tree(
                    env, meta.config_path(args.config_template)
                )
            )
        case "files":
            # Yields # tuple(level: int, name: str, path: str)
            for level, name, path in env.find_referenced_templates(
                meta.config_path(args.config_template)
            ):
                print(os.path.relpath(path))
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def construct_graph(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    env = get_env(meta, args.project_dir)
    config, pp_config = get_config(meta, env, args.config_template)
    match args.format:
        case "none":
            pass
        case "fconfig":
            print(fconfig(config))
        case "repr":
            print(repr(config))
        case "yaml":
            print(to_yaml(config))
        case "python":
            print(generate_code(config["main"]))
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def start_tensorboard(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    env = get_env(meta, args.project_dir)
    config, pp_config = get_config(meta, env, args.config_template)
    config_meta = Latent.materialize(config.meta)

    if args.all:
        output_dir = os.path.abspath(config_meta["models_dir"])
    else:
        output_dir = os.path.abspath(config_meta["output_dir"])

    cmd_args = [
        "tensorboard",
        "--logdir",
        output_dir,
    ]

    if len(args.remainder) > 1 and args.remainder[0] == "--":
        cmd_args.extend(args.remainder[1:])

    cmd_str = ""
    for arg in cmd_args:
        cmd_str += f"{arg} "

    print(f"{cmd_str}")

    # Run the command
    if not args.dry_run:
        subprocess.run(cmd_args)


def train_script(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    env = get_env(meta, args.project_dir)
    config, pp_config = get_config(meta, env, args.config_template)
    config_meta = Latent.materialize(config.meta)
    nproc_per_node = config_meta["nproc_per_node"]
    train_script_path = os.path.join(
        config_meta["forgather_dir"], "scripts", "train_script.py"
    )

    cmd_args = ["torchrun"]

    if len(args.remainder) > 1 and args.remainder[0] == "--":
        cmd_args.extend(args.remainder[1:])
    else:
        # Apply defaults, if not specified
        cmd_args.extend(
            [
                "--standalone",
                "--nproc-per-node",
                str(nproc_per_node),
            ]
        )

    # Apply path to script and project directory argument to script.
    cmd_args.extend(
        [
            os.path.normpath(train_script_path),
            "-p",
            os.path.normpath(args.project_dir),
        ]
    )

    # Optionally, apply system search path from meta.
    if meta.system_path is not None:
        cmd_args.extend(["-s", meta.system_path])

    # Add the config template name
    cmd_args.append(args.config_template)

    # Generate equivalent command string
    cmd_str = ""

    if args.devices:
        cmd_str += f'CUDA_VISIBLE_DEVICES="{args.devices}" '
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    for arg in cmd_args:
        cmd_str += f"{arg} "

    print(f"{cmd_str}")

    # Run the command
    if not args.dry_run:
        subprocess.run(cmd_args)


def template_list(args):
    meta = MetaConfig(args.project_dir)
    set_default_template(meta, args)
    match args.format:
        case "md":
            print(nb.render_extends_graph(meta))
        case "files":
            for template_name, template_path in meta.find_templates():
                print(template_path)
        case _:
            raise Exception(f"Unrecognized format {args.format}")


def show_index(args):
    md = nb.render_project_index(args.project_dir)
    print(md)


def main():
    args = parse_args()
    match args.command:
        case "index":
            show_index(args)
        case "ls":
            list_configurations(args)
        case "meta":
            show_meta(args)
        case "targets":
            list_targets(args)
        case "tlist":
            template_list(args)
        case "graph":
            construct_graph(args)
        case "trefs":
            list_referenced_templates(args)
        case "pp":
            preprocess(args)
        case "tb":
            start_tensorboard(args)
        case "code":
            as_code(args)
        case "construct":
            construct(args)
        case "train":
            train_script(args)
        case _:
            show_index(args)


if __name__ == "__main__":
    main()
