#!/usr/bin/env python

import os
import argparse
from argparse import RawTextHelpFormatter
import sys
from pprint import pp
import subprocess

from forgather.project import Project
import forgather.nb.notebooks as nb
from forgather.meta_config import preprocessor_globals, MetaConfig
from forgather.config import ConfigEnvironment, fconfig
from forgather.codegen import generate_code
from forgather.yaml_encoder import to_yaml
from forgather.latent import Latent

def parse_args(args=None):
    parser = argparse.ArgumentParser(
        formatter_class=RawTextHelpFormatter,
        description="Forgather CLI",
        epilog=(
            ""
        ),
    )
    parser.add_argument(
        "-p",
        "--project-dir",
        type=str,
        default=".",
        help="The relative path to the project directory.",
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
    templates_parser = subparsers.add_parser("templates", help="List referenced templates")
    meta_parser = subparsers.add_parser("meta", help="Show meta configuration")
    targets_parser = subparsers.add_parser("targets", help="Show output targets")
    code_parser = subparsers.add_parser("code", help="Output configuration as Python code")
    construct_parser = subparsers.add_parser("construct", help="Materialize and print a target")
    graph_parser = subparsers.add_parser("graph", help="Preprocess and parse into node graph")
    tb_parser = subparsers.add_parser("tb", help="Start Tensorboard for project")
    train_parser = subparsers.add_parser("train", help="Run configuration with train script")

    code_parser.add_argument(
        "--target",
        type=str,
        default="main",
        help="Output target name",
    )

    construct_parser.add_argument(
        "--target",
        type=str,
        default="",
        help="Output target name",
    )

    graph_parser.add_argument(
        "--format",
        type=str,
        choices=['none', 'repr', 'yaml', 'fconfig', 'python'],
        default="yaml",
        help="Graph format",
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

    train_parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default=None,
        help="CUDA Visible Devices e.g. \"0,1\"",
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

def get_meta(args):
    meta = MetaConfig(args.project_dir)
    if not args.config_template:
        default_config = meta.default_config()
        args.config_template = default_config
    return meta

def get_env(meta, args):
    # Create new config environment and load configuration
    environment = ConfigEnvironment(
        searchpath=meta.searchpath,
        global_vars=preprocessor_globals(args.project_dir, meta.workspace_root),
    )
    return environment

def get_config(meta, env, args):
    return env.load(meta.config_path(args.config_template)).get()

def list_configurations(args):
    meta = get_meta(args)
    for config, path in meta.find_templates(meta.config_prefix):
        print(config)

def list_targets(args):
    meta = get_meta(args)
    env = get_env(meta, args)
    config, pp_config = get_config(meta, env, args)
    s = ""
    for target in config.keys():
        s += f"{target}\n"
    print(s)

def preprocess(args):
    meta = get_meta(args)
    env = get_env(meta, args)
    pp_config = env.preprocess(meta.config_path(args.config_template))
    print(pp_config)

def as_code(args):
    meta = get_meta(args)
    env = get_env(meta, args)
    config, pp_config = get_config(meta, env, args)
    code = generate_code(config[args.target])
    print(code)

def construct(args):
    proj = Project(args.config_template, args.project_dir)
    target = proj(args.target)
    pp(target)

def show_meta(args):
    meta = get_meta(args)
    md = nb.render_meta(meta, "# Meta Config\n")
    print(md)

def list_referenced_templates(args):
    meta = get_meta(args)
    env = get_env(meta, args)
    # Yields # tuple(level: int, name: str, path: str)
    for level, name, path in env.find_referenced_templates(meta.config_path(args.config_template)):
        print(f"{' ' * 4 * level} {name} : {os.path.relpath(path)}")

def construct_graph(args):
    meta = get_meta(args)
    env = get_env(meta, args)
    config, pp_config = get_config(meta, env, args)
    match args.format:
        case 'none':
            pass
        case 'fconfig':
            print(fconfig(config))
        case 'repr':
            print(repr(config))
        case 'yaml':
            print(to_yaml(config))
        case 'python':
            print(generate_code(config["main"]))
        case _:
            raise Exception(f"Unrecognized format {args.format}")

def start_tensorboard(args):
    meta = get_meta(args)
    env = get_env(meta, args)
    config, pp_config = get_config(meta, env, args)
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

    if len(args.remainder) > 1 and args.remainder[0] == '--':
        cmd_args.extend(args.remainder[1:])

    cmd_str = ""
    for arg in cmd_args:
        cmd_str += f"{arg} "

    print(f"{cmd_str}")

    # Run the command
    if not args.dry_run:
        subprocess.run(cmd_args)

def train_script(args):
    meta = get_meta(args)
    env = get_env(meta, args)
    config, pp_config = get_config(meta, env, args)
    config_meta = Latent.materialize(config.meta)
    nproc_per_node = config_meta["nproc_per_node"]
    train_script_path = os.path.join(
        config_meta["forgather_dir"], "scripts", "train_script.py"
    )

    cmd_args = [ "torchrun" ]

    if len(args.remainder) > 1 and args.remainder[0] == '--':
        cmd_args.extend(args.remainder[1:])
    else:
        # Apply defaults, if not specified
        cmd_args.extend([
            "--standalone",
            "--nproc-per-node",
            str(nproc_per_node),
        ])

    # Apply path to script and project directory argument to script.
    cmd_args.extend([
        os.path.normpath(train_script_path),
        "-p",
        os.path.normpath(args.project_dir),
    ])

    # Optionally, apply system search path from meta.
    if meta.system_path is not None:
        cmd_args.extend(["-s", meta.system_path])

    # Add the config template name
    cmd_args.append(args.config_template)

    # Generate equivalent command string
    cmd_str = ""

    if args.devices:
        cmd_str += f"CUDA_VISIBLE_DEVICES=\"{args.devices}\" "
        os.environ["CUDA_VISIBLE_DEVICES"] = args.devices

    for arg in cmd_args:
        cmd_str += f"{arg} "

    print(f"{cmd_str}")

    # Run the command
    if not args.dry_run:
        subprocess.run(cmd_args)

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
        case "graph":
            construct_graph(args)
        case "templates":
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
