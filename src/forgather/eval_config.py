"""Shared dataclasses for the ``forgather eval`` subsystem.

Two schemas live here:

``TestConfig``
    The static evaluation-config schema. An eval YAML project's ``main`` target
    materializes to a dict that maps onto this dataclass. Consumed by
    ``forgather.cli.eval`` (to populate ``list``/``show``) and by
    ``scripts/eval_script.py`` (to drive the run). Optional fields carry
    library-level defaults so ``forgather eval show`` always displays the
    effective values the runtime will use.

``EvalResult``
    The record written to ``{model}/evals/<name>_<ts>/results.json`` after a
    run. Built once in rank-0 of ``scripts/eval_script.py`` via
    ``EvalResult.from_config(test_config, ...)``; outcome fields are filled in
    after ``trainer.evaluate()`` returns. Serialize with
    ``dataclasses.asdict`` for JSON output.

Adding a new identity field to ``TestConfig`` only requires mirroring it on
``EvalResult`` (with a pass-through in ``from_config``) — no template edits.
"""

import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, Tuple


@dataclass
class TestConfig:
    # Required — every eval config must set these.
    eval_name: str
    name: str
    description: str
    dataset_proj: str
    dataset_config: str
    dataset_target: str

    # Optional — defaults picked up when the YAML does not override.
    default_batch_size: int = 8
    default_max_length: int = 4096
    default_stride: int = 0


@dataclass
class EvalResult:
    # Identity — mirrors TestConfig fields 1:1.
    eval_name: str
    config_name: str
    description: str
    dataset_proj: str
    dataset_config: str
    dataset_target: str

    # Run parameters — resolved values actually used (not the defaults).
    model_path: str
    checkpoint_path: Optional[str]
    batch_size: int
    max_length: int
    stride: int
    dtype: str
    attn_implementation: str
    trainer: str
    world_size: int

    # Outcomes — filled in after ``trainer.evaluate()`` completes.
    eval_loss: Optional[float] = None
    perplexity: Optional[float] = None
    # Tokenizer-agnostic metrics. ``bpb`` (bits-per-byte) is the right number
    # to compare across models that use different tokenizers; ``perplexity``
    # is only comparable within a tokenizer family because the denominator
    # (tokens) varies with vocabulary.
    bpb: Optional[float] = None
    bpc: Optional[float] = None
    tokens_per_byte: Optional[float] = None
    total_bytes: Optional[int] = None
    total_chars: Optional[int] = None
    total_predicted_tokens: Optional[int] = None
    wall_time_s: Optional[float] = None
    timestamp: Optional[str] = None

    @classmethod
    def from_config(
        cls,
        config: TestConfig,
        *,
        model_path: str,
        checkpoint_path: Optional[str],
        batch_size: int,
        max_length: int,
        stride: int,
        dtype: str,
        attn_implementation: str,
        trainer: str,
        world_size: int,
    ) -> "EvalResult":
        """Build a pre-evaluation record from a ``TestConfig`` + runtime params.

        Outcome fields start as ``None``; set them directly on the instance
        after ``trainer.evaluate()`` returns, then ``asdict()`` for JSON.
        """
        return cls(
            eval_name=config.eval_name,
            config_name=config.name,
            description=config.description,
            dataset_proj=config.dataset_proj,
            dataset_config=config.dataset_config,
            dataset_target=config.dataset_target,
            model_path=model_path,
            checkpoint_path=checkpoint_path,
            batch_size=batch_size,
            max_length=max_length,
            stride=stride,
            dtype=dtype,
            attn_implementation=attn_implementation,
            trainer=trainer,
            world_size=world_size,
        )


def iter_eval_configs(
    search_paths: Iterable[str],
) -> Iterator[Tuple[str, str, str, TestConfig]]:
    """Walk ``search_paths`` and yield every ``type.evaluation`` config found.

    Each yielded tuple is ``(name, project_dir, template, test_config)``:

    - ``name`` prefers the config's ``eval_name`` field, falling back to the
      template file's basename without extension.
    - ``project_dir`` is the directory containing the project's ``meta.yaml``.
    - ``template`` is the template name relative to the project's
      ``config_prefix`` (ready to pass to :class:`~forgather.project.Project`).
    - ``test_config`` is the materialized :class:`TestConfig` dataclass so
      optional fields fall back to library-level defaults.

    Configs whose ``main`` target raises during materialization, or whose
    ``meta.config_class`` is missing / not under ``type.evaluation``, are
    silently skipped so one bad config doesn't break the listing. Callers
    that need a hard error should use :func:`find_eval_config` instead.
    """
    # Local imports: the library's core dataclasses (TestConfig/EvalResult)
    # are used by the eval script on every rank, but the discovery helpers
    # are only used by CLI / server dispatch. Importing Project/MetaConfig
    # at module-load would pull Jinja2 into hot startup paths.
    from .latent import Latent
    from .meta_config import MetaConfig
    from .project import Project

    for root in search_paths:
        if not os.path.isdir(root):
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            if "meta.yaml" not in filenames:
                continue
            try:
                meta = MetaConfig(dirpath)
            except Exception:
                continue
            for template_name, _template_path in meta.find_templates(
                meta.config_prefix
            ):
                try:
                    proj = Project(template_name, dirpath)
                    cfg_class = Latent.materialize(proj.config.meta).get(
                        "config_class", ""
                    )
                except Exception:
                    continue
                if not cfg_class.startswith("type.evaluation"):
                    continue
                try:
                    data = TestConfig(**proj())
                except Exception:
                    continue
                name = (
                    data.eval_name
                    or os.path.splitext(os.path.basename(template_name))[0]
                )
                yield name, dirpath, template_name, data


def find_eval_config(
    name: str,
    search_paths: Iterable[str],
) -> Tuple[str, str, TestConfig]:
    """Resolve ``name`` to a specific eval config under ``search_paths``.

    Returns ``(project_dir, template, test_config)``. Raises :class:`LookupError`
    when no config with that name exists. Callers in the CLI typically
    re-raise as :class:`SystemExit`; server callers translate to HTTP 404.
    """
    # list() is fine — the number of eval configs in a repo is small and
    # we'd walk them all anyway before deciding no match exists.
    paths_list: List[str] = list(search_paths)
    for entry in iter_eval_configs(paths_list):
        cfg_name, project_dir, template, data = entry
        if cfg_name == name:
            return project_dir, template, data
    raise LookupError(
        f"no eval config named {name!r} found in search paths: {paths_list}"
    )
