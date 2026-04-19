"""Dispatcher for `forgather eval {list, show, test}`."""

import os
import subprocess
import sys
from pathlib import Path

from forgather.eval_config import TestConfig
from forgather.latent import Latent
from forgather.meta_config import MetaConfig
from forgather.project import Project
from forgather.user_config import eval_search_paths


def _forgather_dir() -> str:
    """Locate the forgather repo root. Walk up from this file until we find
    a `templatelib` directory (sentinel used elsewhere in the codebase).
    """
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "templatelib").is_dir():
            return str(parent)
    # Fall back: three levels up (src/forgather/cli -> repo root)
    return str(here.parents[3])


def _discover_eval_projects(search_paths):
    """Yield (project_dir, MetaConfig) for each project under search_paths."""
    for root in search_paths:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            # Skip hidden dirs
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if "meta.yaml" not in filenames:
                continue
            try:
                meta = MetaConfig(dirpath)
            except Exception:
                continue
            yield dirpath, meta


def _iter_eval_configs(search_paths):
    """Yield (name, project_dir, config_template, TestConfig) for every config
    tagged with a ``type.evaluation`` config_class.

    The config's ``main`` dict is loaded into a ``TestConfig`` dataclass so
    optional fields (``default_batch_size`` / ``default_max_length`` /
    ``default_stride``) pick up their library-level defaults when the YAML
    does not set them.
    """
    for project_dir, meta in _discover_eval_projects(search_paths):
        for template_name, _template_path in meta.find_templates(meta.config_prefix):
            try:
                proj = Project(template_name, project_dir)
                cfg_class = Latent.materialize(proj.config.meta).get("config_class", "")
            except Exception:
                continue
            if not cfg_class.startswith("type.evaluation"):
                continue
            try:
                data = TestConfig(**proj())
            except Exception:
                continue
            # Use eval_name if present, otherwise strip extension from template.
            name = (
                data.eval_name or os.path.splitext(os.path.basename(template_name))[0]
            )
            yield name, project_dir, template_name, data


def _find_eval_config(name, search_paths):
    for entry in _iter_eval_configs(search_paths):
        cfg_name, project_dir, template, data = entry
        if cfg_name == name:
            return project_dir, template, data
    raise SystemExit(f"Error: no eval config named '{name}' found in search paths")


def _resolve_model_path(args):
    """Resolve --model, else fall back to the current project's output_dir.

    ``output_dir`` is exposed in the meta block by ``training_script``-type
    configs (see ``templatelib/base/training_script/training_script_type.yaml``).
    If the user is inside a non-training project (e.g. a raw model or dataset
    project), they must pass ``-M`` explicitly.
    """
    if args.model:
        return os.path.abspath(args.model)

    try:
        project_dir = MetaConfig.find_project_dir(args.project_dir)
    except ValueError:
        raise SystemExit(
            f"Error: no --model provided and no Forgather project was found at or "
            f"above {os.path.abspath(args.project_dir)}. Pass -M PATH to specify "
            "a model."
        )

    meta = MetaConfig(project_dir)
    proj = Project(args.config_template or meta.default_config(), project_dir)
    cfg_meta = Latent.materialize(proj.config.meta)
    output_dir = cfg_meta.get("output_dir")
    if not output_dir:
        raise SystemExit(
            f"Error: no --model provided and project '{project_dir}' does not "
            "expose `output_dir` in its meta block (only training_script-type "
            "configs do). Pass -M PATH to specify a model."
        )
    return os.path.abspath(output_dir)


def list_cmd(args):
    forgather_dir = _forgather_dir()
    paths = eval_search_paths(forgather_dir)
    any_found = False
    print(f"{'NAME':<24} {'DESCRIPTION':<60} PATH")
    for name, project_dir, template, data in _iter_eval_configs(paths):
        any_found = True
        loc = f"{project_dir}:{template}"
        print(f"{name:<24} {data.description:<60} {loc}")
    if not any_found:
        print("(no eval configs found)")
        print(f"Search paths: {paths}")


def show_cmd(args):
    forgather_dir = _forgather_dir()
    paths = eval_search_paths(forgather_dir)
    project_dir, template, data = _find_eval_config(args.name, paths)
    if args.pp:
        proj = Project(template, project_dir)
        print(proj.pp_config)
        return
    print(f"Name:             {data.eval_name or args.name}")
    print(f"Config:           {data.name}")
    print(f"Description:      {data.description}")
    print(f"Project:          {project_dir}")
    print(f"Template:         {template}")
    print(f"Dataset project:  {data.dataset_proj}")
    print(f"Dataset config:   {data.dataset_config}")
    print(f"Dataset target:   {data.dataset_target}")
    print(f"Default batch:    {data.default_batch_size}")
    print(f"Default max_len:  {data.default_max_length}")
    print(f"Default stride:   {data.default_stride}")


def test_cmd(args):
    forgather_dir = _forgather_dir()
    paths = eval_search_paths(forgather_dir)
    project_dir, template, _data = _find_eval_config(args.name, paths)
    model_path = _resolve_model_path(args)

    eval_script = os.path.join(forgather_dir, "scripts", "eval_script.py")

    env = os.environ.copy()
    if args.devices:
        env["CUDA_VISIBLE_DEVICES"] = args.devices

    if args.trainer == "simple":
        cmd = [sys.executable, eval_script]
    else:
        cmd = [
            "torchrun",
            "--standalone",
            "--nproc-per-node",
            "gpu",
            eval_script,
        ]

    cmd.extend(
        [
            "--eval-project",
            project_dir,
            "--eval-config",
            template,
            "--model",
            model_path,
            "--trainer",
            args.trainer,
            "--dtype",
            args.dtype,
            "--attn-implementation",
            args.attn_implementation,
            "--max-steps",
            str(args.max_steps),
        ]
    )
    if args.batch_size is not None:
        cmd.extend(["--batch-size", str(args.batch_size)])
    if args.max_length is not None:
        cmd.extend(["--max-length", str(args.max_length)])
    if args.checkpoint:
        cmd.extend(["--checkpoint", args.checkpoint])
    if args.no_checkpoint:
        cmd.append("--no-checkpoint")
    if args.compile:
        cmd.append("--compile")
    if args.output_dir:
        cmd.extend(["--output-dir", args.output_dir])

    print(" ".join(cmd))
    if args.dry_run:
        return
    result = subprocess.run(cmd, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def eval_cmd(args):
    sub = getattr(args, "eval_subcommand", None)
    if sub == "list":
        list_cmd(args)
    elif sub == "show":
        show_cmd(args)
    elif sub == "test":
        test_cmd(args)
    else:
        print("Usage: forgather eval {list, show, test} [...]")
        print("Use 'forgather eval --help' for details.")
        sys.exit(1)
