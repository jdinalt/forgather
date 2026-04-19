import os
import platform
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, List, Optional

from .config import ConfigDict, ConfigEnvironment
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
PROJECT_META_NAME = "meta.yaml"


@dataclass()
class MetaConfig:
    # The path of the project directory
    project_dir: str

    # The name of the meta file
    name: str

    # The name of the current project
    project_name: Optional[str]

    # The description of the current project
    description: Optional[str]

    # The path to the meta file
    meta_path: str

    # Paths to search for config templates in
    searchpath: List[str]

    # The value of the system_path from the meta-config
    system_path: Optional[str]

    # The name of the sub-directory in which leaf configurations are located
    config_prefix: str

    # The default configuration
    default_cfg: Optional[str]

    # The raw config dictionary
    config_dict: dict

    # The path to the workspace root
    workspace_root: str

    def __init__(self, project_dir=".", meta_name=PROJECT_META_NAME):
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
        self.project_name = config.get("name", None)
        self.description = config.get("description", None)
        if self.system_path is not None:
            self.system_path = self.norm_path(self.system_path)

    def __str__(self):
        s = ""
        s += f"Project Name: {self.project_name}\n"
        s += f"Description: {self.description}\n"
        s += f"Default Config: {self.default_cfg}\n"
        s += f"Project Directory: {self.project_dir}\n"
        s += f"Workspace Root: {self.workspace_root}\n"
        s += f"Config Prefix: {self.config_prefix}\n"
        s += f"Search Path: {self.searchpath}\n"

        return s

    def norm_path(self, path):
        return os.path.normpath(os.path.join(self.project_dir, path))

    def default_config(self):
        """Get the name of the default config"""
        if self.default_cfg is not None:
            return self.default_cfg
        else:
            # Pick the first in the list.
            return next(self.find_templates(self.config_prefix))[0]

    def config_path(self, config_template=None):
        """Given a config template name or None, return a path to the config (or default config)"""
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
        if not os.path.exists(project_directory):
            raise ValueError(f"The directory, '{project_directory}', does not exist.")
        elif not os.path.isdir(project_directory):
            raise ValueError(f"The directory, '{project_directory}', does not exist.")
        elif not os.path.isfile(config_path):
            raise ValueError(
                f"'The template, '{template_name}', does not exist in '{project_directory}'"
            )
        # Build searchpath for meta-config.
        # We include the project, the workspace config, and the user's Forgather config directory.
        searchpath = [project_directory]

        self.workspace_root = self.find_workspace_dir(project_directory)
        searchpath.append(os.path.join(self.workspace_root, WORKSPACE_CONFIG_DIR_NAME))
        kwargs["workspace_root"] = self.workspace_root

        user_templates_dir = os.path.join(forgather_config_dir(), "templates")
        if os.path.isdir(user_templates_dir):
            searchpath.append(user_templates_dir)

        self.environment = ConfigEnvironment(
            searchpath=searchpath,
            global_vars=preprocessor_globals(project_directory, self.workspace_root),
        )
        config = self.environment.load(template_name, **kwargs)
        return config.config

    @staticmethod
    def find_workspace_dir(project_dir):
        """
        Recursively search parent directories for Forgather workspace config directory
        """

        def is_workspace(root_dir):
            workspace_config_dir = os.path.join(root_dir, WORKSPACE_CONFIG_DIR_NAME)
            return os.path.isdir(workspace_config_dir)

        workspace_root = MetaConfig._find_dir(project_dir, is_workspace)
        if not workspace_root:
            raise ValueError(
                f"Workspace directory,'forgather_workspace', was not found under project directory {project_dir}"
            )
        return workspace_root

    @staticmethod
    def find_project_dir(project_dir):
        def is_project(root_dir):
            target_dir = os.path.join(root_dir, PROJECT_META_NAME)
            return os.path.isfile(target_dir)

        found_project_dir = MetaConfig._find_dir(project_dir, is_project)
        if not found_project_dir:
            raise ValueError(f"No projects where found at or below {project_dir}")
        return found_project_dir

    @staticmethod
    def _find_dir(root, match_regex):
        root = os.path.abspath(root)

        while True:
            if match_regex(root):
                return root
            parent_dir, _ = os.path.split(root)
            if parent_dir == root:
                return None
            root = parent_dir
