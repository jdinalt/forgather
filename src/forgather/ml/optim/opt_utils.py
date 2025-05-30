import re
from pprint import pp

def make_regex_optimizer_groups(named_parameters, group_map, group_config, debug=False):
    groups = { group_name: [] for regex, group_name in group_map }
        
    for param_name, param_value in named_parameters:
        for regex, group_name in group_map:
            m = re.search(regex, param_name)
            if m is not None:
                groups[group_name].append((param_name, param_value))
                break
    param_groups = [ { "params": params } | group_config[group_name] for group_name, params in groups.items() ]
    if debug:
        pp(param_groups)
    return param_groups

def make_grouped_optimizer(named_parameters, opt_ctor, group_map, group_config, opt_args=None, opt_kwargs=None, debug=False):
    param_groups = make_regex_optimizer_groups(named_parameters, group_map, group_config, debug=debug)
    if opt_kwargs is None:
        opt_kwargs = {}
    if opt_args is None:
        opt_args = []
    
    return opt_ctor(param_groups, *opt_args, **opt_kwargs)