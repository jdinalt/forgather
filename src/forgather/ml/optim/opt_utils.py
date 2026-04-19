from __future__ import annotations

import logging
import re
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import TYPE_CHECKING, Any

from torch.nn import Parameter

from ..distributed import prefix_logger_rank

if TYPE_CHECKING:
    from ..trainer.trainer_types import OptimizerFactoryT, OptimizerT

# Mapping from group name to {"regex": <pattern>, "config": <overrides>}.
# Parameters are assigned to the first group whose regex matches (dict
# insertion order), so more-specific groups should come first. Parameters
# that match no group fall through to an implicit default group with no
# overrides.
#
# The dict-of-dicts form is deliberate: named entries let child templates
# override individual groups by key (via `== super()` + a new entry) rather
# than having to re-specify the whole list. A group whose spec is ``None``
# is treated as removed, so a child template can delete an inherited group
# cleanly by re-declaring its name with a null value.
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

    # Normalise and validate the user spec up front so bad configs fail
    # with a clear message instead of an obscure KeyError deeper in. A spec
    # value of None means "remove this group" and is filtered out here.
    normalised: dict[str, tuple[str, dict[str, Any]]] = {}
    for group_name, spec in optimizer_groups.items():
        if spec is None:
            continue
        if not isinstance(spec, dict) or "regex" not in spec:
            raise ValueError(
                f"optimizer_groups[{group_name!r}] must be a dict with a "
                f"'regex' key (and optional 'config'), or null to remove the "
                f"group; got {spec!r}"
            )
        regex = spec["regex"]
        overrides = spec.get("config") or {}
        if not isinstance(overrides, dict):
            raise ValueError(
                f"optimizer_groups[{group_name!r}]['config'] must be a dict "
                f"or null; got {overrides!r}"
            )
        normalised[group_name] = (regex, overrides)

    groups: dict[str, list[tuple[str, Parameter]]] = defaultdict(list)

    for param_name, parameter in named_parameters:
        for group_name, (regex, _overrides) in normalised.items():
            if re.search(regex, param_name) is not None:
                groups[group_name].append((param_name, parameter))
                if debug:
                    logger.info(f"param group: {group_name} <- {param_name}")
                break
        else:
            groups[_DEFAULT_GROUP].append((param_name, parameter))
            if debug:
                logger.info(f"param group: {_DEFAULT_GROUP} <- {param_name}")

    # Preserve user-declared ordering, with the default group appended last.
    ordered_names = [name for name in normalised if name in groups]
    if _DEFAULT_GROUP in groups:
        ordered_names.append(_DEFAULT_GROUP)

    param_groups: ParamGroups = []
    for group_name in ordered_names:
        params = groups[group_name]
        if not params:
            continue
        overrides = {} if group_name == _DEFAULT_GROUP else normalised[group_name][1]
        param_groups.append({"params": params} | overrides)

    return param_groups


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
