"""Argument parser for the `eval` subcommand.

Args that are forwarded from ``forgather eval test`` to
``scripts/eval_script.py`` are declared once in ``_EVAL_SCRIPT_ARGS`` below
and consumed by three call sites:

* ``add_eval_script_args(parser)`` — registers them on the ``test`` subparser
  here, and on the script's own parser inside ``scripts/eval_script.py``.
* ``forward_eval_script_args(args)`` — builds the CLI tokens used by
  ``test_cmd`` when spawning the script via subprocess/torchrun.
* ``eval_script_args_to_job_params(args)`` — builds the ``job_params`` dict
  the server's queue stores for the ``--enqueue`` path.

Adding a new passthrough arg means appending one entry to
``_EVAL_SCRIPT_ARGS`` and using it in ``scripts/eval_script.py`` — nothing
else in this file or ``eval.py`` needs to change.
"""

import argparse
import os
from argparse import RawTextHelpFormatter
from typing import Any

path_type = lambda x: os.path.normpath(os.path.expanduser(x))


# --- Shared passthrough arg spec ----------------------------------------
#
# Each entry is a dict with:
#   flags:        tuple of flag strings, e.g. ("--batch-size",)
#   kwargs:       dict passed verbatim to ``parser.add_argument``
#   forward:      "value" (emit ``--flag VAL`` when value is not None)
#               | "flag"  (emit ``--flag`` when value is truthy; store_true args)
#               | "always" (emit ``--flag VAL`` unconditionally — for args
#                           with non-None defaults the script depends on)
#   enqueue_key:  key under which the value appears in ``job_params``.
#                 None to omit from enqueue.
#   enqueue_when: "always"   — always include in job_params (bool args
#                              normalized via bool() for JSON cleanliness)
#               | "truthy"   — include when truthy (store_true bools,
#                              and value args where empty/falsy is a "no")
#               | "not_none" — include when the value is not None.
#                              Use this for numeric value args so the
#                              local CLI path and the enqueue payload
#                              agree on whether 0 is a legitimate value.
_EVAL_SCRIPT_ARGS: list[dict[str, Any]] = [
    {
        "flags": ("--trainer",),
        "kwargs": dict(
            choices=["ddp", "simple", "pipeline"],
            default="ddp",
            help="Trainer backend (default: ddp)",
        ),
        "forward": "always",
        "enqueue_key": "trainer",
        "enqueue_when": "always",
    },
    {
        "flags": ("--checkpoint",),
        "kwargs": dict(type=path_type, default=None, help="Explicit checkpoint path"),
        "forward": "value",
        "enqueue_key": "checkpoint_path",
        "enqueue_when": "truthy",
    },
    {
        "flags": ("--no-checkpoint",),
        "kwargs": dict(
            action="store_true",
            help="Use from_pretrained on the model dir instead of resuming a checkpoint",
        ),
        "forward": "flag",
        "enqueue_key": "no_checkpoint",
        "enqueue_when": "truthy",
    },
    {
        "flags": ("--fused-loss",),
        "kwargs": dict(
            action="store_true",
            help=(
                "Enable fused output-linear-cross-entropy-loss; useful for "
                "reducing memory when models have a large vocabulary."
            ),
        ),
        "forward": "flag",
        "enqueue_key": "fused_loss",
        "enqueue_when": "truthy",
    },
    {
        "flags": ("--batch-size",),
        "kwargs": dict(type=int, default=None),
        "forward": "value",
        "enqueue_key": "batch_size",
        "enqueue_when": "not_none",
    },
    {
        "flags": ("--max-length",),
        "kwargs": dict(type=int, default=None),
        "forward": "value",
        "enqueue_key": "max_length",
        "enqueue_when": "not_none",
    },
    {
        "flags": ("--stride",),
        "kwargs": dict(type=int, default=None),
        "forward": "value",
        "enqueue_key": "stride",
        "enqueue_when": "not_none",
    },
    {
        "flags": ("--max-steps",),
        "kwargs": dict(type=int, default=-1),
        "forward": "always",
        "enqueue_key": "max_steps",
        "enqueue_when": "always",
    },
    {
        "flags": ("--dtype",),
        "kwargs": dict(default="bfloat16"),
        "forward": "always",
        "enqueue_key": "dtype",
        "enqueue_when": "always",
    },
    {
        "flags": ("--attn-implementation",),
        "kwargs": dict(default="sdpa"),
        "forward": "always",
        "enqueue_key": "attn_implementation",
        "enqueue_when": "always",
    },
    {
        "flags": ("--compile",),
        "kwargs": dict(action="store_true"),
        "forward": "flag",
        "enqueue_key": "compile",
        "enqueue_when": "always",
    },
    {
        "flags": ("--output-dir",),
        "kwargs": dict(
            type=path_type,
            default=None,
            help="Override where evals/ is written (default: model path)",
        ),
        "forward": "value",
        "enqueue_key": "output_dir",
        "enqueue_when": "truthy",
    },
]


def _dest_for(spec: dict) -> str:
    """Compute the argparse ``dest`` for a spec entry (matches argparse's default)."""
    explicit = spec["kwargs"].get("dest")
    if explicit:
        return explicit
    return spec["flags"][-1].lstrip("-").replace("-", "_")


def add_eval_script_args(parser) -> None:
    """Register the args forwarded from ``forgather eval test`` to ``eval_script.py``."""
    for spec in _EVAL_SCRIPT_ARGS:
        parser.add_argument(*spec["flags"], **spec["kwargs"])


def _forward_from_lookup(specs, get) -> list[str]:
    """Spec-driven forwarder. ``get(spec, default)`` reads the value."""
    tokens: list[str] = []
    for spec in specs:
        flag = spec["flags"][-1]
        default = spec["kwargs"].get("default")
        val = get(spec, default)
        mode = spec["forward"]
        if mode == "flag":
            if val:
                tokens.append(flag)
        elif mode == "value":
            if val is not None:
                tokens.extend([flag, str(val)])
        elif mode == "always":
            # Fall back to the spec default if the caller passed None — without
            # this, an "always" arg with an explicit ``None`` in a future
            # job_params payload would emit the literal string ``"None"``.
            if val is None:
                val = default
            tokens.extend([flag, str(val)])
        else:  # pragma: no cover
            raise ValueError(f"unknown forward mode {mode!r} for {flag}")
    return tokens


def forward_eval_script_args(args) -> list[str]:
    """Return CLI tokens that forward ``args`` to ``scripts/eval_script.py``.

    Callers must emit the script-only args (``--eval-project``,
    ``--eval-config``, ``--model``) separately; those are not passthrough.
    """
    return _forward_from_lookup(
        _EVAL_SCRIPT_ARGS,
        lambda spec, default: getattr(args, _dest_for(spec), default),
    )


def forward_eval_script_args_from_params(params: dict) -> list[str]:
    """Like :func:`forward_eval_script_args`, but reads from a ``job_params``
    dict (used by the server-side enqueue path in
    ``tools/forgather_server/eval_ops.py``).

    Spec entries with ``enqueue_key=None`` are skipped — they aren't exposed
    over the queue interface.
    """
    enqueue_specs = [s for s in _EVAL_SCRIPT_ARGS if s.get("enqueue_key") is not None]
    return _forward_from_lookup(
        enqueue_specs,
        lambda spec, default: params.get(spec["enqueue_key"], default),
    )


def eval_script_args_to_job_params(args) -> dict[str, Any]:
    """Translate passthrough args into the ``job_params`` dict for ``--enqueue``."""
    out: dict[str, Any] = {}
    for spec in _EVAL_SCRIPT_ARGS:
        key = spec.get("enqueue_key")
        if key is None:
            continue
        val = getattr(args, _dest_for(spec), None)
        when = spec.get("enqueue_when", "truthy")
        if when == "always":
            # Normalize store_true to bool for JSON cleanliness.
            if spec["kwargs"].get("action") == "store_true":
                out[key] = bool(val)
            else:
                out[key] = val
        elif when == "truthy":
            if val:
                out[key] = val
        elif when == "not_none":
            if val is not None:
                out[key] = val
        else:  # pragma: no cover
            raise ValueError(f"unknown enqueue_when {when!r} for {spec['flags'][-1]}")
    return out


_VALID_FORWARD_MODES = frozenset({"flag", "value", "always"})
_VALID_ENQUEUE_WHEN = frozenset({"always", "truthy", "not_none"})


def _validate_spec() -> None:
    """Validate ``_EVAL_SCRIPT_ARGS`` at import time.

    Catches typos in the enum-like ``forward`` / ``enqueue_when`` fields and
    missing required keys, so a malformed spec entry fails loudly at import
    instead of silently no-op'ing the first time the affected flag is used.
    This is the bug class the spec exists to prevent in the first place.
    """
    seen_keys: set[str] = set()
    for i, spec in enumerate(_EVAL_SCRIPT_ARGS):
        prefix = f"_EVAL_SCRIPT_ARGS[{i}]"
        for required in ("flags", "kwargs", "forward"):
            if required not in spec:
                raise RuntimeError(f"{prefix}: missing required key {required!r}")
        if not isinstance(spec["flags"], tuple) or not spec["flags"]:
            raise RuntimeError(f"{prefix}: 'flags' must be a non-empty tuple")
        if spec["forward"] not in _VALID_FORWARD_MODES:
            raise RuntimeError(
                f"{prefix} ({spec['flags'][-1]}): forward={spec['forward']!r} "
                f"not in {sorted(_VALID_FORWARD_MODES)}"
            )
        ek = spec.get("enqueue_key")
        if ek is not None:
            if ek in seen_keys:
                raise RuntimeError(
                    f"{prefix} ({spec['flags'][-1]}): duplicate enqueue_key {ek!r}"
                )
            seen_keys.add(ek)
            when = spec.get("enqueue_when")
            if when not in _VALID_ENQUEUE_WHEN:
                raise RuntimeError(
                    f"{prefix} ({spec['flags'][-1]}): enqueue_when={when!r} "
                    f"not in {sorted(_VALID_ENQUEUE_WHEN)}"
                )


_validate_spec()


def passthrough_enqueue_keys() -> frozenset[str]:
    """Return the set of ``job_params`` keys this spec recognizes.

    Useful for callers that want to filter a job_params dict down to keys
    that will actually be forwarded (avoids leaking webui- or scheduler-
    internal fields into the inner-script argv).
    """
    return frozenset(
        s["enqueue_key"] for s in _EVAL_SCRIPT_ARGS if s.get("enqueue_key")
    )


def create_eval_parser(_global_args):
    parser = argparse.ArgumentParser(
        prog="forgather eval",
        description="Evaluate a model on a named eval config",
        formatter_class=RawTextHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="eval_subcommand", help="Eval subcommands")

    # list
    subparsers.add_parser(
        "list",
        help="List available eval configs",
        formatter_class=RawTextHelpFormatter,
    )

    # show
    show = subparsers.add_parser(
        "show",
        help="Show details of an eval config",
        formatter_class=RawTextHelpFormatter,
    )
    show.add_argument("name", help="Eval config name (e.g., c4, tinystories)")
    show.add_argument(
        "--pp",
        action="store_true",
        help="Show the preprocessed YAML",
    )

    # test
    test = subparsers.add_parser(
        "test",
        help="Run an eval and write results to {model}/evals/",
        formatter_class=RawTextHelpFormatter,
    )
    test.add_argument("name", help="Eval config name")
    test.add_argument(
        "-M",
        "--model",
        type=path_type,
        default=None,
        help="Path to model directory. Defaults to current project's output_dir.",
    )
    test.add_argument(
        "-d",
        "--devices",
        type=str,
        default=None,
        help='CUDA_VISIBLE_DEVICES, e.g. "0,1"',
    )
    # Passthrough args (forwarded to scripts/eval_script.py).
    add_eval_script_args(test)
    test.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the torchrun command without executing it",
    )
    test.add_argument(
        "--enqueue",
        action="store_true",
        help="Submit to the forgather-server queue instead of running locally.",
    )
    test.add_argument(
        "--priority",
        type=int,
        default=0,
        help="Queue priority for --enqueue (default: 0).",
    )
    test.add_argument(
        "--server",
        type=str,
        default=None,
        metavar="URL",
        help="forgather-server URL for --enqueue (default: $FORGATHER_SERVER_URL or http://127.0.0.1:8765).",
    )

    return parser
