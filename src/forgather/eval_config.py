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

from dataclasses import dataclass
from typing import Optional


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
