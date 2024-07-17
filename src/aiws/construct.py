from typing import Callable, List, Any
import os

from .distributed import main_process_first


def register_for_auto_class(object, /, *args, **kwargs):
    """
    Register an object as a HF AutoClass

    PretrainedModel and PretrainedConfig both support this method. When applied,
    the source code for the respective objects will be automatically saved
    with the model weights.

    This is very useful for custom models, as it simplifies things if the code is
    stored with the model.

    The following macro demonstrates how it can be used in a configuration script.
    ```
    ## Custom model constructor
    ## Defines a constructor for a custom model and registers both the
    ## configuration and model class for AutoConfig/AutoModel construction.
    -- macro custom_model(model_path, model_cls, config_cls, model_config)
    !object:aiws.construct:register_for_auto_class
        - !object:{{model_path}}:{{model_cls}}
            - !object:aiws.construct:register_for_auto_class
                - !object:{{model_path}}:{{config_cls}}
                    kwargs: *{{model_config}}
    -- endmacro
    ```
    """
    object.register_for_auto_class(*args, **kwargs)
    return object


@main_process_first()
def build_rule(
    target: str | os.PathLike,
    recipe: Callable,
    prerequisites: List[str | os.PathLike],
    loader: Callable,
) -> Any:
    assert isinstance(recipe, Callable)
    assert isinstance(loader, Callable)

    if os.path.exists(target):
        build_target = False
        target_mtime = os.path.getmtime(target)
        for dependency in prerequisites:
            dep_mtime = os.path.getmtime(dependency)
            if target_mtime < dep_mtime:
                build_target = True
                break
    else:
        build_target = True

    if build_target:
        output = recipe()
        if output is not None:
            return output
    return loader()
