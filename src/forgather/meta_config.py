from typing import Any, List
from dataclasses import dataclass, field
import os
from importlib.metadata import version
import platform

from .config import ConfigEnvironment, ConfigDict
from .preprocess import forgather_config_dir


def preprocessor_globals(project_dir, workspace_root):
    return dict(
        project_dir=project_dir,
        workspace_root=workspace_root,
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


WORKSPACE_CONFIG_DIR_NAME = "forgather_workspace"


@dataclass()
class MetaConfig:
    project_dir: str
    name: str
    meta_path: str
    searchpath: List[str]
    system_path: str
    config_prefix: str
    default_cfg: str
    config_dict: dict
    workspace_root: str

    def __init__(self, project_dir=".", meta_name="meta.yaml"):
        self.name = meta_name
        self.meta_path = os.path.join(project_dir, meta_name)
        config = self._load_config(self.meta_path, project_dir=project_dir)
        self.config_dict = config
        self.project_dir = project_dir
        self.searchpath = config.get(
            "searchdir", [os.path.join(project_dir, "templates")]
        )
        self.searchpath = [os.path.abspath(path) for path in self.searchpath]
        self.config_prefix = config.get("config_prefix", "configs")
        self.default_cfg = config.get("default_config", None)
        self.system_path = config.get("system_path", None)
        if self.system_path is not None:
            self.system_path = self.norm_path(self.system_path)

    def norm_path(self, path):
        return os.path.normpath(os.path.join(self.project_dir, path))

    def default_config(self):
        if self.default_cfg is not None:
            return self.default_cfg
        else:
            # Pick the first in the list.
            return next(self.find_templates(self.config_prefix))[0]

    def config_path(self, config_template=None):
        if config_template is None or len(config_template) == 0:
            config_template = self.default_config()
        return os.path.join(self.config_prefix, config_template)

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

    def _load_config(self, config_path: str | os.PathLike, /, **kwargs) -> ConfigDict:
        project_directory, template_name = os.path.split(config_path)
        assert os.path.exists(
            project_directory
        ), f"The directory, '{project_directory}', does not exist."
        assert os.path.isdir(
            project_directory
        ), f"'{project_directory}' is not a directory."
        assert os.path.isfile(
            config_path
        ), f"'The template, '{template_name}', does not exist in '{project_directory}'"

        # Build searchpath for meta-config.
        # We include the project, the workspace config, and the user's Forgather config directory.
        searchpath = [project_directory]

        self.workspace_root = self._find_workspace_dir(project_directory)
        searchpath.append(os.path.join(self.workspace_root, WORKSPACE_CONFIG_DIR_NAME))
        kwargs["workspace_root"] = self.workspace_root

        user_templates_dir = os.path.join(forgather_config_dir(), "templates")
        if os.path.isdir(user_templates_dir):
            searchpath.append(user_templates_dir)

        self.environment = ConfigEnvironment(searchpath=searchpath)
        config = self.environment.load(template_name, **kwargs)
        return config.config

    def _find_workspace_dir(self, project_dir):
        """
        Recurisvely search parent directories for Forgather workspace config directory
        """
        workspace_root = os.path.abspath(project_dir)

        while True:
            workspace_config_dir = os.path.join(
                workspace_root, WORKSPACE_CONFIG_DIR_NAME
            )
            if os.path.isdir(workspace_config_dir):
                return workspace_root
            parent_dir, _ = os.path.split(workspace_root)
            if parent_dir == workspace_root:
                raise RuntimeError(
                    f"Workspace directory,'forgather_workspace', was not found under project directory {project_dir}"
                )
            workspace_root = parent_dir
