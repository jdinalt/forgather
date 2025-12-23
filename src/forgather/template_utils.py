import re
from collections import defaultdict
from pprint import pp
from types import NoneType
from typing import Dict, Iterable, List, Tuple

from forgather.preprocess import preprocess, split_templates

TemplateName = str  # Template name in template namespace
TemplateFilePath = str  # Path in file-system to template file
TemplateData = str  # Actual template contents
TemplateFileIter = Iterable[Tuple[TemplateName, TemplateFilePath]]
TemplateDataIter = Iterable[Tuple[TemplateName, TemplateFilePath, TemplateData]]
# (template_name, template_extends_name)
TemplateExtendsIter = Iterable[Tuple[TemplateName, TemplateFilePath, TemplateName]]
ExtendsNode = NoneType | List[Tuple[TemplateName, TemplateFilePath, "ExtendsNode"]]
TemplateList = List[TemplateName]


def template_data_iter(template_iter: TemplateFileIter) -> TemplateDataIter:
    """
    Given a TemplateFileIter return a TemplateDataIter

    If the templates files have inline template splits, the file data will
    be split on split boundaries and the returned tuples will include the sub-templates.
    """
    for template_name, template_path in template_iter:
        with open(template_path, "r", encoding="utf-8") as f:
            raw_template = f.read()
        for template_name, template_data in split_templates(
            raw_template, template_name
        ):
            yield (template_name, template_path, template_data)


def template_extends_iter(template_iter: TemplateDataIter) -> TemplateExtendsIter:
    """
    Determine which templates are extended by other templates.
    Given a TemplateDataIter, returns an Iterable of tuples:
    (template_name, extends_name)
    """
    extends_re = re.compile(r"{% extends ('|\")(.*)('|\") %}")

    for template_name, template_path, template_data in template_iter:
        # Filter raw template through precrocess, which will convert
        # "extends" line-statements into regual Jinja2 syntax, then
        # find any extends tags in the template.
        extends_match = extends_re.search(preprocess(template_data))
        if extends_match:
            extends_match = extends_match[2]
        yield template_name, template_path, extends_match


def get_extends_graph(template_iter: TemplateExtendsIter) -> ExtendsNode:
    """
    Given a TemplateExtendsIter, build a template inheritance graph
    """
    extends_map = defaultdict(list)

    for template_name, template_path, template_extends in template_iter:
        extends_map[template_extends].append((template_name, template_path))
    extends_graph = _build_extends_graph(extends_map[None], extends_map)
    return extends_graph


def _build_extends_graph(
    templates: TemplateList, extends_map: Dict[TemplateName, TemplateList]
) -> ExtendsNode:
    extends_graph = []
    for template_name, template_path in templates:
        extended_by = extends_map.get(template_name, None)
        if extended_by:
            extended_by = _build_extends_graph(extended_by, extends_map)
        extends_graph.append((template_name, template_path, extended_by))
    return extends_graph


def extends_graph_iter(
    extends_graph: ExtendsNode, level: int = 0
) -> Iterable[Tuple[int, TemplateName, TemplateFilePath]]:
    """
    Given a template inheritance graph, returns an iterable of nodes in the graph.
    """
    for template_name, template_path, extended_by in extends_graph:
        yield level, template_name, template_path
        if extended_by:
            yield from extends_graph_iter(extended_by, level + 1)
