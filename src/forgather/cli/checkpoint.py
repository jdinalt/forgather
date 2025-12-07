from forgather.ml.sharded_checkpoint import create_pretrained_symlinks
from .dynamic_args import get_dynamic_args
from forgather import Project


def checkpoint_cmd(args):
    """checkpoint commands."""

    if hasattr(args, "cp_subcommand"):
        match args.cp_subcommand:
            case "link":
                link_command(args)


def link_command(args):
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
