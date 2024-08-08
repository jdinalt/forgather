from typing import Callable, List, Any, Optional
from dataclasses import dataclass, field
import os

from forgather.meta_config import MetaConfig, preprocessor_globals
from forgather.config import ConfigEnvironment
from forgather.latent import Latent


@dataclass()
class Project:
    config_name: str
    project_dir: str
    meta: MetaConfig
    environment: ConfigEnvironment
    config: Any
    pp_config: str

    def __init__(
        self,
        project_dir: Optional[str | os.PathLike] = ".",
        config_name: Optional[str] = "",
    ):
        """
        An abstract project representation which hides some of the details
        of the underlying mechanics.

        project_dir: The location of the project directory
        cnfig_name: The name of the configuration to loade; an empty string loads the default.

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
        assert os.path.exists(
            project_dir
        ), f"The directory, '{project_dir}', does not exist."
        assert os.path.isdir(project_dir), f"'{project_dir}' is not a directory."

        self.project_dir = os.path.abspath(project_dir)

        # Load project meta-data
        self.meta = MetaConfig(self.project_dir)

        # Get the default configuration
        default_config = self.meta.default_config()
        self.config_name = config_name if len(config_name) else default_config

        # Get the config name, with the config-prefix prepended
        config_template_path = self.meta.config_path(config_name)

        # Construct a project environment
        self.environment = ConfigEnvironment(
            searchpath=self.meta.searchpath,
            global_vars=preprocessor_globals(project_dir),
        )

        # Load the pre-processed config and the config graph
        self.config, self.pp_config = self.environment.load(config_template_path).get()

    def __call__(self, **kwargs):
        """
        Construct and return an instance of the configuration

        kwargs: Additional keyword-args to pass to the graph constructor.

        Note: Finer grained construction can be obtained by selecting the desired targets within
        the configuration node graph. i.e.

        ```python
        Latent.materialize(proj.config['meta'])

        # Or, if the target is a node
        proj.config['main']['generated_code']()
        ```

        Note: Many project types expect the pre-processed config to be passed in to the
            constructor. This call adds this for you, but you may need to do this manually for
            certain objects.

        ```python
        proj.config['main'](pp_config=proj.pp_config)
        ```
        """
        kwargs |= dict(pp_config=self.pp_config)
        return Latent.materialize(self.config, **kwargs)
