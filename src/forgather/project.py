from typing import Callable, List, Any, Optional, Iterable
from dataclasses import dataclass, field
import os

from forgather.meta_config import MetaConfig, preprocessor_globals
from forgather.config import ConfigEnvironment
from forgather.latent import Latent
from forgather.dotdict import DotDict


@dataclass()
class Project:
    """
    A high-level, abstract project representation, which hides details of the underlying mechanics.

    project_dir: The location of the project directory
    config_name: The name of the configuration to loade; an empty string loads the default.

    dataclass attributes:

    config_name: The name of the selected configuration; automatically populated with
        the default, if unspecified.
    project_dir: The absolute path to the project directory.
    meta: The project's meta-config
    environment: The projects config envrionment
    config: The constructed node-graph, representing the configuration.
    pp_config: The pre-processed configuration.

    Hint: If you are debugging a configuration, it's usually easier to incrementally
        construct the project for better diagnostics. See: 'project_config.ipynb'
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
        """
        Loaded the specified configurtion

        config_name: The name of the configuration to load.
        """
        # Load the pre-processed config and the config graph
        self.config, self.pp_config = self.environment.load(
            self.meta.config_path(config_name), **kwargs
        ).get()

    def __call__(self, *args, asdict=False, **kwargs):
        """
        Construct and return an instance of the configuration

        make_targets: The output targets to make. By default, this is 'main'
            If a string, returns the specified target. If an Iterable of str, returns
            a dictionary of the specified targets. If Iterable, invalid targets will be removed
            from the returned dictionary.

            Note: Each call will construct a new set of objects, thus you could end up with duplicates
            if you call this seperately on different targets.
        kwargs: Additional keyword-args to pass to the graph constructor.

        ```python
        # Construct and return main target
        proj = Project()
        main_target = proj()

        # Construct only confg-meta
        meta = proj("meta")

        # Construct the model and tokenizer
        model, tokenizer = proj("model", "tokenizer")

        # Construct a dictionary of objects
        outputs = proj("model", "tokenizer", asdict=True)
        ```
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
    **config_kwargs,
):
    proj = Project(config_template, project_dir)
    return proj(targets, **config_kwargs)
