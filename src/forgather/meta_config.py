import os
import platform
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any, List, Optional

from .config import ConfigDict, ConfigEnvironment
from .preprocess import forgather_config_dir


def preprocessor_globals(project_dir, workspace_root):
    """Build the dictionary of global variables injected into every Jinja2 template.

    Parameters
    ----------
    project_dir : str
        Absolute path to the current project directory.  Exposed in templates
        as ``project_dir``.
    workspace_root : str
        Absolute path to the workspace root directory.  Exposed in templates
        as ``workspace_root``.

    Returns
    -------
    dict
        Mapping of variable names to values, including ``project_dir``,
        ``workspace_root``, ``hostname``, ``uname``, and ``versions`` (a dict
        of library version strings for Python, torch, transformers, and
        accelerate).
    """
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
    """Project metadata loaded from ``meta.yaml``.

    ``MetaConfig`` reads and parses the ``meta.yaml`` file that sits at the
    root of every Forgather project.  It resolves template search paths,
    locates the workspace root by walking up the directory tree, and exposes
    the configuration values needed by :class:`~forgather.project.Project` to
    set up its :class:`ConfigEnvironment`.

    Parameters
    ----------
    project_dir : str or os.PathLike, optional
        Path to the project directory containing ``meta.yaml``.  Defaults to
        the current working directory (``"."``)
    meta_name : str, optional
        Name of the metadata file to load.  Defaults to ``"meta.yaml"``.

    Attributes
    ----------
    project_dir : str
        Path to the project directory as supplied to ``__init__``.
    name : str
        Name of the meta file (e.g. ``"meta.yaml"``).
    project_name : str or None
        Human-readable project name declared in ``meta.yaml``.
    description : str or None
        Short project description declared in ``meta.yaml``.
    meta_path : str
        Absolute path to the meta file.
    searchpath : list of str
        Ordered list of absolute directory paths searched for config templates.
        Derived from the ``searchdir`` key in ``meta.yaml``, defaulting to
        ``[project_dir/templates]``.
    system_path : str or None
        Optional system-level template search path from ``meta.yaml``.
    config_prefix : str
        Sub-directory inside the search path where leaf configuration files
        live.  Defaults to ``"configs"``.
    default_cfg : str or None
        Name of the default configuration file as declared in ``meta.yaml``.
        When ``None``, :meth:`default_config` picks the first template found
        under *config_prefix*.
    config_dict : dict
        Raw dictionary parsed from ``meta.yaml``.
    workspace_root : str
        Absolute path to the workspace root directory (the directory that
        contains ``forgather_workspace/``), found by walking up from
        *project_dir*.

    Raises
    ------
    ValueError
        If the project directory does not exist, ``meta.yaml`` is not found, or
        no ``forgather_workspace/`` directory exists in the ancestor hierarchy.

    Examples
    --------
    >>> meta = MetaConfig("/path/to/my_project")
    >>> print(meta.project_name)
    My Project
    >>> print(meta.searchpath)
    ['/path/to/my_project/templates', '/path/to/workspace/forgather_workspace']
    """

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
        """Return the name of the default configuration template.

        Returns
        -------
        str
            The value of ``default_config`` from ``meta.yaml`` if set;
            otherwise the name of the first template discovered under
            *config_prefix* across all search paths.
        """
        if self.default_cfg is not None:
            return self.default_cfg
        else:
            # Pick the first in the list.
            return next(self.find_templates(self.config_prefix))[0]

    def config_path(self, config_template=None):
        """Return the template-relative path for the given configuration name.

        Parameters
        ----------
        config_template : str or None, optional
            Name of the configuration template (e.g. ``"train.yaml"``).
            When ``None`` or an empty string, the default configuration is used.

        Returns
        -------
        str
            Path of the form ``"{config_prefix}/{config_template}"`` suitable
            for passing to :meth:`ConfigEnvironment.load`.
        """
        if config_template is None or len(config_template) == 0:
            config_template = self.default_config()
        return os.path.join(self.config_prefix, config_template)

    def find_templates(self, prefix="", suffix=".yaml"):
        """Iterate over all templates in the search path matching a prefix and suffix.

        Walks every directory in :attr:`searchpath`, descending into the
        sub-directory given by *prefix*, and yields ``(name, path)`` pairs for
        every file whose name ends with *suffix*.  Hidden directories
        (names starting with ``"."``) are skipped.

        Parameters
        ----------
        prefix : str, optional
            Sub-directory to search within each search-path entry.  Defaults to
            ``""`` (search from the root of each search-path entry).
        suffix : str, optional
            File extension filter.  Defaults to ``".yaml"``.

        Yields
        ------
        template_name : str
            Template name relative to the prefixed search directory, suitable
            for use with :meth:`ConfigEnvironment.load`.
        template_path : str
            Filesystem path to the template file.

        Examples
        --------
        Find all templates under a ``models`` directory in any search-path entry:

        >>> for template_name, template_path in meta.find_templates("models"):
        ...     print(template_name, template_path)
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
        """Walk up the directory tree to find the Forgather workspace root.

        The workspace root is the nearest ancestor directory that contains a
        ``forgather_workspace/`` sub-directory.

        Parameters
        ----------
        project_dir : str
            Starting directory for the upward search.

        Returns
        -------
        str
            Absolute path to the workspace root directory.

        Raises
        ------
        ValueError
            If no ``forgather_workspace/`` directory is found in any ancestor.
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
        """Walk up the directory tree to find the nearest Forgather project directory.

        A project directory is one that directly contains a ``meta.yaml`` file.

        Parameters
        ----------
        project_dir : str
            Starting directory for the upward search.

        Returns
        -------
        str
            Absolute path to the nearest project directory that contains
            ``meta.yaml``.

        Raises
        ------
        ValueError
            If no project directory is found at or above *project_dir*.
        """

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
