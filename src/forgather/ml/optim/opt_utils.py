from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable

from torch.nn import Parameter

from ..distributed import prefix_logger_rank

if TYPE_CHECKING:
    from ..trainer.trainer_types import OptimizerFactoryT, OptimizerT

# Mapping from group name to {"regex": <pattern>, "config": <overrides>,
# "factory": <optimizer factory>}. Parameters are assigned to the first group
# whose regex matches (dict insertion order), so more-specific groups should
# come first. Parameters that match no group fall through to an implicit
# default group with no overrides and the trainer's top-level optimizer factory.
#
# The dict-of-dicts form is deliberate: named entries let child templates
# override individual groups by key (via `== super()` + a new entry) rather
# than having to re-specify the whole list. A group whose spec is ``None``
# is treated as removed, so a child template can delete an inherited group
# cleanly by re-declaring its name with a null value.
#
# The optional ``factory`` key lets a group be optimized by a different
# optimizer than the top-level ``optimizer_factory`` (e.g. Muon for matrix
# parameters and AdamW for embeddings/norms/biases). Groups without a
# ``factory`` are bucketed onto the default factory.
OptimGroupMap = Mapping[str, Mapping[str, Any] | None]
ParamGroups = list[dict[str, Any]]
NamedParameters = Iterable[tuple[str, Parameter]]

# Reserved name used for the implicit fall-through group. Chosen to be
# unlikely to collide with any user-supplied group name.
_DEFAULT_GROUP = "__default__"

logger = logging.getLogger(__name__)
prefix_logger_rank(logger)


def build_parameter_groups(
    named_parameters: NamedParameters,
    optimizer_groups: OptimGroupMap,
    debug: bool = False,
) -> ParamGroups:
    """
    Assign model parameters to optimizer param groups by regex matching.

    Each parameter is assigned to the first group whose regex matches its
    fully-qualified name (dict insertion order). Parameters that match no
    user group fall through to an implicit default group with no overrides,
    so every parameter is guaranteed to end up in some group. Empty groups
    (including the default, if every parameter matched a user group) are
    omitted from the returned list.

    Parameters
    ----------
    named_parameters : NamedParameters
        Iterable of (name, parameter) pairs, e.g. from
        ``model.named_parameters()``.
    optimizer_groups : OptimGroupMap
        Mapping of group name to a spec dict with keys:
        ``regex`` (str) and ``config`` (dict of hyperparameter
        overrides). The ``config`` dict is merged into the returned
        param group and forwarded to the optimizer. ``config`` may be
        omitted or ``None`` for groups that only carve parameters out
        without changing hyperparameters. A spec value of ``None``
        marks the group as removed - useful for clearing an inherited
        group from a child template.
    debug : bool, optional
        When True, log each parameter -> group assignment at INFO.

    Returns
    -------
    ParamGroups
        A list of param group dicts ready to pass to a torch optimizer.
    """
    if _DEFAULT_GROUP in optimizer_groups:
        raise ValueError(
            f"Group name {_DEFAULT_GROUP!r} is reserved for the fall-through "
            "default group; rename your group."
        )

    normalised = _normalise_optimizer_groups(optimizer_groups)

    groups = _assign_groups(named_parameters, normalised, debug=debug)

    # Preserve user-declared ordering, with the default group appended last.
    ordered_names = [name for name in normalised if name in groups]
    if _DEFAULT_GROUP in groups:
        ordered_names.append(_DEFAULT_GROUP)

    param_groups: ParamGroups = []
    for group_name in ordered_names:
        params = groups[group_name]
        if not params:
            continue
        overrides = (
            {} if group_name == _DEFAULT_GROUP else normalised[group_name].overrides
        )
        param_groups.append({"params": params} | overrides)

    return param_groups


@dataclass
class _NormalisedSpec:
    """Validated form of a single ``optimizer_groups`` entry."""

    regex: str
    overrides: dict[str, Any]
    factory: Callable[..., Any] | None


def _normalise_optimizer_groups(
    optimizer_groups: OptimGroupMap,
) -> dict[str, _NormalisedSpec]:
    """Validate the user spec up front so bad configs fail with a clear
    message rather than an obscure error deeper in. A spec value of ``None``
    means "remove this group" and is filtered out here."""
    normalised: dict[str, _NormalisedSpec] = {}
    for group_name, spec in optimizer_groups.items():
        if spec is None:
            continue
        if not isinstance(spec, dict) or "regex" not in spec:
            raise ValueError(
                f"optimizer_groups[{group_name!r}] must be a dict with a "
                f"'regex' key (and optional 'config'/'factory'), or null to "
                f"remove the group; got {spec!r}"
            )
        regex = spec["regex"]
        overrides = spec.get("config") or {}
        if not isinstance(overrides, dict):
            raise ValueError(
                f"optimizer_groups[{group_name!r}]['config'] must be a dict "
                f"or null; got {overrides!r}"
            )
        factory = spec.get("factory")
        if factory is not None and not callable(factory):
            raise ValueError(
                f"optimizer_groups[{group_name!r}]['factory'] must be a "
                f"callable or null; got {factory!r}"
            )
        normalised[group_name] = _NormalisedSpec(
            regex=regex, overrides=dict(overrides), factory=factory
        )
    return normalised


def _assign_groups(
    named_parameters: NamedParameters,
    normalised: dict[str, _NormalisedSpec],
    debug: bool = False,
) -> dict[str, list[tuple[str, Parameter]]]:
    """Bucket each ``(name, param)`` into the first regex match (dict
    insertion order). Unmatched params land in the implicit default group."""
    groups: dict[str, list[tuple[str, Parameter]]] = defaultdict(list)
    for param_name, parameter in named_parameters:
        for group_name, spec in normalised.items():
            if re.search(spec.regex, param_name) is not None:
                groups[group_name].append((param_name, parameter))
                if debug:
                    logger.info(f"param group: {group_name} <- {param_name}")
                break
        else:
            groups[_DEFAULT_GROUP].append((param_name, parameter))
            if debug:
                logger.info(f"param group: {_DEFAULT_GROUP} <- {param_name}")
    return groups


def build_optimizer_buckets(
    named_parameters: NamedParameters,
    optimizer_groups: OptimGroupMap,
    default_factory: OptimizerFactoryT,
    debug: bool = False,
) -> list[tuple[OptimizerFactoryT, ParamGroups]]:
    """
    Bucket parameters by the optimizer factory that should own them.

    Each entry in ``optimizer_groups`` may carry an optional ``factory`` key.
    Groups that omit ``factory`` (and the implicit fall-through default group)
    resolve onto ``default_factory``. Buckets are keyed by factory identity
    (``id(factory)``), so the same factory object reused across multiple
    groups produces a single bucket containing all of those groups' param
    groups (preserving declaration order).

    Returns
    -------
    list[tuple[OptimizerFactoryT, ParamGroups]]
        One ``(factory, param_groups)`` pair per distinct factory, in the
        order each factory is first encountered. Empty buckets and empty
        param groups are filtered out.
    """
    if _DEFAULT_GROUP in optimizer_groups:
        raise ValueError(
            f"Group name {_DEFAULT_GROUP!r} is reserved for the fall-through "
            "default group; rename your group."
        )

    normalised = _normalise_optimizer_groups(optimizer_groups)
    groups = _assign_groups(named_parameters, normalised, debug=debug)

    ordered_names = [name for name in normalised if name in groups]
    if _DEFAULT_GROUP in groups:
        ordered_names.append(_DEFAULT_GROUP)

    # Bucket by factory identity, preserving first-encounter order.
    bucket_order: list[int] = []
    bucket_factory: dict[int, OptimizerFactoryT] = {}
    bucket_param_groups: dict[int, ParamGroups] = {}

    for group_name in ordered_names:
        params = groups[group_name]
        if not params:
            continue
        if group_name == _DEFAULT_GROUP:
            factory = default_factory
            overrides: dict[str, Any] = {}
        else:
            spec = normalised[group_name]
            factory = spec.factory if spec.factory is not None else default_factory
            overrides = spec.overrides

        key = id(factory)
        if key not in bucket_factory:
            bucket_order.append(key)
            bucket_factory[key] = factory
            bucket_param_groups[key] = []
        bucket_param_groups[key].append({"params": params} | overrides)

    return [(bucket_factory[k], bucket_param_groups[k]) for k in bucket_order]


def make_grouped_optimizer(
    named_parameters: NamedParameters,
    optimizer_groups: OptimGroupMap,
    optimizer_factory: OptimizerFactoryT,
    debug: bool = False,
) -> OptimizerT:
    """
    Build an optimizer with regex-assigned parameter groups.

    Equivalent to calling ``build_parameter_groups`` and passing the result
    to ``optimizer_factory``. Retained for callers that want a one-shot
    helper (e.g. template configs that bind an optimizer factory directly).
    For trainer-level use, prefer passing ``optimizer_groups`` to the
    Trainer constructor: it calls ``build_parameter_groups`` internally and
    also honours the ``debug_optimizer_groups`` training argument.
    """
    param_groups = build_parameter_groups(
        named_parameters, optimizer_groups, debug=debug
    )
    return optimizer_factory(param_groups)
