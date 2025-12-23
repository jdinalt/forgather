import re
from typing import Dict, Iterable, List, Tuple

from torch import Tensor

PSubList = Iterable["PSub"]  # Recursive list of regex parameter name subsitutions
PFqn = str  # Parameter Fully Qualified Name
PList = Iterable[PFqn]  # List of full qualified parameter names
PSubPattern = str  # A regex pattern to match
PSubRepl = str  # A string template for replacement. See re.sub()
PSub = Tuple[
    PSubPattern, PSubRepl, PSubList
]  # A recursive FQN name subsitution definition
PDict = Dict[PFqn, Tensor]  # A parameter dict -- model.state_dict()


def sub_param_name(s: str, psub_list: PSubList):
    """
    Recursively replace 's' with matches from psub_list
    """
    for pattern, repl, child_list in psub_list:
        match = re.match(pattern, s)
        if match:
            # print(match)
            # head = s[:match.end()]
            tail = s[match.end() :]
            s = match.expand(repl) + sub_param_name(tail, child_list)
    return s


def remap_parameter_fqns(plist: PList, sub_list: PSubList) -> List[Tuple[PFqn, PFqn]]:
    """
    Given an iterable of parameter FQNs and a substitution list, returns
    a list of tuples of parameter mappings from x -> y
    """
    mapping = []
    for input_name in plist:
        output_name = sub_param_name(input_name, sub_list)
        mapping.append((input_name, output_name))
    return mapping


def remap_state_dict(state_dict: PDict, psub_list: PSubList) -> PDict:
    """
    Given a state dictionary and a parameter substitution list, return a
    state dictionary with the substitued parameter names.
    """
    output_dict = {}
    for x, y in remap_parameter_fqns(state_dict.keys(), psub_list):
        output_dict[y] = state_dict[x]
    return output_dict
