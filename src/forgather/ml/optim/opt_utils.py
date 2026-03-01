import re
from pprint import pp


def make_regex_optimizer_groups(named_parameters, group_map, group_config, debug=False):
    groups = {group_name: [] for regex, group_name in group_map}

    for param_name, param_value in named_parameters:
        for regex, group_name in group_map:
            m = re.search(regex, param_name)
            if m is not None:
                if debug:
                    print(f"param group: {group_name} <- {param_name}")
                groups[group_name].append((param_name, param_value))
                break
    param_groups = [
        {"params": params} | group_config[group_name]
        for group_name, params in groups.items()
    ]

    return param_groups


def make_grouped_optimizer(
    named_parameters,
    optimizer_factory,
    group_map,
    group_config,
    debug=False,
):
    param_groups = make_regex_optimizer_groups(
        named_parameters, group_map, group_config, debug=debug
    )

    return optimizer_factory(param_groups)
