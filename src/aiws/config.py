import platform
from importlib.metadata import version


def base_preprocessor_globals():
    return dict(
        script_args="N/A",
        world_size=1,
        rank=0,
        local_rank=0,
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
