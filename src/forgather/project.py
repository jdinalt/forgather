import os
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, List, Optional

from forgather.config import ConfigEnvironment
from forgather.dotdict import DotDict
from forgather.latent import Latent
from forgather.meta_config import MetaConfig, preprocessor_globals


@dataclass()
class Project:
    """Central user-facing abstraction for a Forgather ML experiment.

    A ``Project`` loads a YAML configuration file through a Jinja2 template
    inheritance chain, parses it into a node graph, and can materialise any
    named target from that graph into live Python objects.  It is the primary
    entry point for interactive experiment development and for training scripts.

    Parameters
    ----------
    config_name : str, optional
        Name of the configuration template to load (e.g. ``"train_tiny_llama.yaml"``).
        An empty string or ``None`` loads the project's default configuration as
        declared in ``meta.yaml``.
    project_dir : str or os.PathLike, optional
        Path to the project directory.  Must contain a ``meta.yaml`` file and a
        ``templates/`` sub-directory.  Defaults to the current working directory.
    **kwargs
        Additional keyword arguments forwarded to the Jinja2 preprocessor as
        template variables.

    Attributes
    ----------
    config_name : str
        Name of the selected configuration; automatically set to the project
        default when *config_name* is empty or ``None``.
    project_dir : str
        Absolute path to the project directory.
    meta : MetaConfig
        Parsed project metadata (search paths, default config, etc.).
    environment : ConfigEnvironment
        Jinja2 + YAML preprocessing environment used to load templates.
    config : Any
        The parsed node graph produced from the preprocessed YAML.  ``None``
        when no configuration has been loaded yet.
    pp_config : str
        The fully preprocessed YAML text (after Jinja2 rendering), useful for
        debugging template issues.

    Examples
    --------
    Load a project from the current directory using the default configuration:

    >>> proj = Project()
    >>> training_script = proj()

    Load a specific configuration and materialise individual targets:

    >>> proj = Project("train_tiny_llama.yaml", "examples/tutorials/tiny_llama")
    >>> model = proj("model")
    >>> model, tokenizer = proj("model", "tokenizer")

    Notes
    -----
    When debugging a configuration it is usually easier to construct the project
    incrementally for better diagnostic messages.  See ``project_config.ipynb``
    for a step-by-step notebook example.
    """

    config_name: str
    project_dir: str
    meta: MetaConfig
    environment: ConfigEnvironment
    config: Any
    pp_config: str

    def __init__(
        self,
        config_name: Optional[str] = "",
        project_dir: Optional[str | os.PathLike] = ".",
        **kwargs,
    ):
        assert os.path.exists(
            project_dir
        ), f"The directory, '{project_dir}', does not exist."
        assert os.path.isdir(project_dir), f"'{project_dir}' is not a directory."

        self.project_dir = os.path.abspath(project_dir)

        # Load project meta-data
        self.meta = MetaConfig(self.project_dir)

        # Get the default configuration
        default_config = self.meta.default_config()
        if config_name is None:
            config_name = ""
        self.config_name = config_name if len(config_name) else default_config

        # Construct a project environment
        self.environment = ConfigEnvironment(
            searchpath=self.meta.searchpath,
            global_vars=preprocessor_globals(project_dir, self.meta.workspace_root),
        )

        if config_name is not None:
            self.load_config(config_name, **kwargs)
        else:
            self.config = None
            self.pp_config = None

    def load_config(self, config_name: str, **kwargs):
        """Load and parse the named configuration template.

        Preprocesses the template through the Jinja2 environment, then parses
        the resulting YAML into a node graph.  The results are stored in
        ``self.config`` and ``self.pp_config``.

        Parameters
        ----------
        config_name : str
            Name of the configuration template to load, relative to the
            project's ``config_prefix`` directory (e.g. ``"train.yaml"``).
        **kwargs
            Additional keyword arguments forwarded to the Jinja2 preprocessor
            as template variables.
        """
        # Load the pre-processed config and the config graph
        self.config, self.pp_config = self.environment.load(
            self.meta.config_path(config_name), **kwargs
        ).get()

    def add_template(self, name, data):
        """Add an in-memory template definition to the Jinja2 loader.

        Parameters
        ----------
        name : str
            Template name used to reference this template from other templates
            (e.g. via ``-- extends`` or ``-- include``).
        data : str
            Raw template source text.
        """
        loader = self.environment.get_loader()
        loader.add_template(name, data)

    def __call__(self, *args, asdict=False, **kwargs):
        """Materialise one or more targets from the loaded configuration graph.

        Each call traverses the node graph and constructs fresh Python objects
        for the requested targets.  Calling this method multiple times will
        produce independent object instances; share a single call when you need
        objects that reference each other (e.g. model and optimizer sharing the
        same parameter tensors).

        Parameters
        ----------
        *args : str
            Names of the output targets to build.  When called with no
            arguments (or with a single empty string), the ``"main"`` target is
            built.  When multiple names are given, a generator that yields the
            corresponding objects in the same order is returned.
        asdict : bool, optional
            When ``True`` the return value is always a :class:`~forgather.dotdict.DotDict`
            mapping target names to their materialised objects, regardless of
            how many targets were requested.  Default is ``False``.
        **kwargs
            Additional context variables forwarded to the graph materialisation
            engine.

        Returns
        -------
        object
            The materialised ``"main"`` target when called with no arguments.
        object
            The single materialised target when exactly one name is given and
            *asdict* is ``False``.
        generator
            A generator yielding the materialised targets in order when multiple
            names are given and *asdict* is ``False``.
        DotDict
            A dot-accessible dictionary mapping every requested target name to
            its materialised object when *asdict* is ``True``.

        Raises
        ------
        RuntimeError
            If no configuration has been loaded (i.e. ``self.config`` is ``None``).

        Examples
        --------
        >>> proj = Project("train.yaml")

        Build the default ``main`` target:

        >>> training_script = proj()

        Build a single named target:

        >>> model = proj("model")

        Unpack multiple targets in one call (avoids duplicate construction):

        >>> model, tokenizer = proj("model", "tokenizer")

        Collect targets into a dot-accessible dict:

        >>> outputs = proj("model", "tokenizer", asdict=True)
        >>> outputs.model
        """

        if self.config is None:
            raise RuntimeError("The project does not have a loaded configuration")

        if len(args) == 0 or args[0] == "":
            mtargets = ("main",)
        elif isinstance(args[0], list):
            # Preserve legacy interface for now.
            asdict = True
            mtargets = args[0]
        else:
            mtargets = args

        kwargs |= dict(pp_config=self.pp_config)
        outputs = Latent.materialize(
            self.config, mtargets=mtargets, context_vars=kwargs
        )

        if asdict:
            return DotDict(outputs)
        if len(mtargets) == 1:
            return outputs[mtargets[0]]
        else:
            return (outputs[key] for key in mtargets)


def from_project(
    project_dir: str,
    config_template: str | None = None,
    targets: str | List[str] = "",
    pp_debug=False,
    pp_kwargs: dict | None = None,
    **config_kwargs,
):
    """Load and materialise targets from a separate Forgather project.

    Convenience helper that lets one configuration import live Python objects
    produced by another project.  It is typically called from within a YAML
    configuration via a ``!call`` or ``!singleton`` tag.

    Parameters
    ----------
    project_dir : str
        Path to the project directory to load.
    config_template : str or None, optional
        Name of the configuration template to use.  ``None`` loads the project
        default.
    targets : str or list of str, optional
        Target name or list of target names to materialise.  An empty string
        builds the ``"main"`` target.
    pp_debug : bool, optional
        When ``True``, print preprocessing diagnostics (template name, targets,
        and the preprocessed config text) before materialising.  Default is
        ``False``.
    pp_kwargs : dict or None, optional
        Keyword arguments forwarded to the Jinja2 preprocessor of the
        sub-project.  Defaults to an empty dict.
    **config_kwargs
        Additional keyword arguments forwarded to the graph materialisation
        engine of the sub-project.

    Returns
    -------
    object
        The materialised target(s) from the sub-project, identical in type to
        what ``Project.__call__`` would return for the same *targets* argument.
    """
    if pp_kwargs is None:
        pp_kwargs = {}

    proj = Project(config_template, project_dir, **pp_kwargs)
    if pp_debug:
        print(f"Loading sub-project: {project_dir}:{config_template}")
        print(f"{targets=}")
        print(f"{config_kwargs=}")
        print(f"{pp_kwargs=}")
        print(proj.pp_config)

    return proj(targets, **config_kwargs)
