from typing import Any, List
from dataclasses import dataclass, field
import os
from importlib.metadata import version
import platform

from forgather.config import load_config

def preprocessor_globals(project_directory):
    return dict(
        project_directory=project_directory,
        hostname=platform.node(),
        uname=platform.uname(),
        versions={"python": platform.python_version()}
        | {
            lib: version(lib)
            for lib in (
                "torch",
                "transformers",
                "accelerate",
            )
        },
    )

@dataclass()
class MetaConfig:
    project_dir: str
    meta_path: str
    searchpath: List[str]
    system_path: str
    train_script: str

    def __init__(self, project_dir, meta_name="meta.yaml"):
        self.meta_path = os.path.join(project_dir, meta_name)
        config = load_config(self.meta_path)
        self.project_dir = project_dir
        self.searchpath = []
        for path in config.searchdir:
            norm_path = self.norm_path(path)
            assert os.path.exists(norm_path), f"Search dir {norm_path} does not exist."
            assert os.path.isdir(
                norm_path
            ), f"Search dir {norm_path} is not a directory."
            self.searchpath.append(os.path.abspath(norm_path))
        self.config_prefix = config.config_prefix
        self.system_path = self.norm_path(config.system_path)
        self.train_script = self.norm_path(config.train_script)

    def norm_path(self, path):
        return os.path.normpath(os.path.join(self.project_dir, path))

    def find_templates(self, prefix="", suffix=".yaml"):
        """
        List all templates in the searchpath matching prefix and suffix

        ```
        # Find all templates under a 'models' directory in any searchpath
        for template_name, template_path in meta.find_templates('models'):
            ...
        ```
        """
        for templates_dir in self.searchpath:
            templates_dir = os.path.relpath(templates_dir)
            templates_dir = os.path.join(templates_dir, prefix)
            for dirpath, dirnames, filenames in os.walk(templates_dir):
                # Remove hidden
                for dirname in dirnames:
                    if dirname.startswith("."):
                        dirnames.remove(dirname)
                for filename in filenames:
                    if filename.endswith(suffix):
                        template_path = os.path.join(dirpath, filename)
                        # strip prefix
                        template_name = template_path[len(templates_dir) :]
                        if template_name.startswith("/"):
                            template_name = template_name[1:]
                        yield (template_name, template_path)