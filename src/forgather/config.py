import os
import sys
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from pprint import pformat
from typing import Any, Callable, Dict, Iterable, Iterator, List, Optional, Set, Tuple

from jinja2 import Environment, meta
from platformdirs import user_config_dir
from yaml import SafeLoader

from .latent import (
    CallableNode,
    FactoryNode,
    Latent,
    MetaNode,
    Node,
    PartialNode,
    SingletonNode,
    VarNode,
)
from .preprocess import PPEnvironment
from .utils import (
    AutoName,
    add_exception_notes,
    format_line_numbers,
    indent_block,
    track_depth,
)
from .yaml_utils import (
    CallableConstructor,
    dict_constructor,
    dlist_constructor,
    list_constructor,
    load_depth_first,
    tuple_constructor,
    var_constructor,
)


class ConfigText(str):
    """Preprocessed configuration text with optional line-number formatting.

    A thin ``str`` subclass returned by :meth:`ConfigEnvironment.preprocess`.
    Identical to a plain string in every other respect.
    """

    def with_line_numbers(self, show_line_numbers=True):
        """Return the text with prepended line numbers.

        Parameters
        ----------
        show_line_numbers : bool, optional
            When ``True`` (default) each line is prefixed with its 1-based line
            number.  When ``False`` the string is returned unchanged.

        Returns
        -------
        str
            The (optionally numbered) configuration text.
        """
        return format_line_numbers(self) if show_line_numbers else self


class Config:
    """Container pairing a parsed node graph with its preprocessed YAML source.

    Returned by :meth:`ConfigEnvironment.load` and related methods.  Keeping
    the preprocessed text alongside the parsed graph makes it easy to display
    debugging information when materialisation fails.

    Attributes
    ----------
    config : Any
        The parsed node graph (typically a :class:`ConfigDict` or a single
        :class:`~forgather.latent.Node`).
    pp_config : str
        The fully preprocessed YAML text produced by Jinja2 rendering.
    """

    def __init__(self, config, pp_config):
        self.config = config
        self.pp_config = pp_config

    def get(self):
        """Return ``(config, pp_config)`` as a two-element tuple.

        Returns
        -------
        tuple
            ``(self.config, self.pp_config)``
        """
        return self.config, self.pp_config

    def __repr__(self):
        return (
            f"{type(self).__name__}(config={self.config}, pp_config={self.pp_config})"
        )


def fconfig(obj, sort_items=True, indent_level=2, visited=None):
    """Recursively pretty-format a configuration object as a human-readable string.

    Handles the full range of types that can appear in a Forgather configuration:
    plain Python scalars, mappings, sequences, :class:`Node` instances, and
    :class:`Config` / :class:`ConfigText` containers.

    Parameters
    ----------
    obj : Any
        The configuration object to format.
    sort_items : bool, optional
        When ``True`` (default), mapping keys and string sequences are sorted
        alphabetically.
    indent_level : int, optional
        Number of spaces used for each level of indentation.  Default is ``2``.
    visited : set or None, optional
        Set of already-visited node identities used to prevent infinite recursion
        on cyclic graphs.  Pass ``None`` (default) to start a fresh set.

    Returns
    -------
    str
        Human-readable representation of the configuration.

    Notes
    -----
    TODO: Rewrite using reprlib.
    """
    if visited is None:
        visited = set()

    def indent_block(block):
        indent = " " * indent_level
        s = "".join(map(lambda s: indent + s + "\n", block.split("\n")))
        return s[:-1]

    if isinstance(obj, ConfigText):
        return obj.with_line_numbers()
    elif isinstance(obj, Config):
        return fconfig(
            dict(config=obj.config, pp_config=obj.pp_config),
            sort_items,
            indent_level,
            visited,
        )
    elif isinstance(obj, str):
        return f"'{obj}'"
    elif isinstance(obj, Set):
        return fconfig(tuple(obj), sort_items, indent_level, visited)
    elif isinstance(obj, Mapping):
        s = ""
        items = obj.items()
        if sort_items:
            items = dict(sorted(items)).items()
        for key, value in items:
            fmt_value = fconfig(value, sort_items, indent_level, visited)
            if "\n" in fmt_value or len(fmt_value) > 80:
                s += f"{key}:\n" + indent_block(fmt_value) + "\n"
            else:
                s += f"{key}: " + fmt_value + "\n"
        return s[:-1]
    elif isinstance(obj, Sequence):
        s = ""
        items = obj
        if sort_items:
            sortable = True
            for value in items:
                if not isinstance(value, str):
                    sortable = False
                    break
            if sortable:
                items = sorted(items)
        for value in items:
            s += "- " + fconfig(value, sort_items, indent_level, visited) + "\n"
        return s[:-1]
    elif isinstance(obj, Node):
        s = ""
        if isinstance(obj, VarNode):
            return f"var {obj.constructor}={obj.value}\n"
        elif isinstance(obj, SingletonNode):
            s += "singleton "
        elif isinstance(obj, PartialNode):
            s += "lambda "
        elif isinstance(obj, CallableNode):
            s += "callable "
        else:
            s += "node "
        s += f"{repr(obj.identity)} {obj.constructor}"
        if obj.identity in visited:
            s += "elided ..."
            return s

        visited.add(obj.identity)
        if isinstance(obj, CallableNode):
            if len(obj.submodule_searchpath):
                s += f" searchpath={obj.submodule_searchpath}"
            if len(obj.args):
                s += "\n" + indent_block(
                    fconfig(obj.args, sort_items, indent_level, visited)
                )
            if len(obj.kwargs):
                s += "\n" + indent_block(
                    fconfig(obj.kwargs, sort_items, indent_level, visited)
                )
        return s
    else:
        return pformat(obj)


def pconfig(obj, /, *args, **kwargs):
    """Print a configuration object to stdout.

    Convenience wrapper around :func:`fconfig`.

    Parameters
    ----------
    obj : Any
        The configuration object to print.
    *args
        Positional arguments forwarded to :func:`fconfig`.
    **kwargs
        Keyword arguments forwarded to :func:`fconfig`.
    """
    print(fconfig(obj, *args, **kwargs))


# We will be adding custom YAML tags to the loader; create a new class,
# as we don't want these applied to all instances of SafeLoader.
class ConfigLoader(SafeLoader):
    pass


ConfigLoader.add_multi_constructor("!factory", CallableConstructor(FactoryNode))
ConfigLoader.add_multi_constructor("!singleton", CallableConstructor(SingletonNode))
ConfigLoader.add_multi_constructor("!call", CallableConstructor(SingletonNode))

# Deprecated
ConfigLoader.add_multi_constructor("!lambda", CallableConstructor(PartialNode))


ConfigLoader.add_multi_constructor("!partial", CallableConstructor(PartialNode))
ConfigLoader.add_multi_constructor("!meta", CallableConstructor(MetaNode))
ConfigLoader.add_constructor("!var", var_constructor)
ConfigLoader.add_multi_constructor("!tuple", tuple_constructor)
ConfigLoader.add_multi_constructor("!list", list_constructor)
ConfigLoader.add_multi_constructor("!dlist", dlist_constructor)
ConfigLoader.add_multi_constructor("!dict", dict_constructor)


class ConfigDict(dict):
    """Dictionary wrapper for a parsed configuration that supports attribute access.

    Keys whose names start with ``"."`` are filtered out on construction (they
    are used internally as YAML section markers and are not part of the public
    configuration).  All remaining keys are accessible as attributes in addition
    to standard item access.

    Examples
    --------
    >>> cfg = ConfigDict({"model": "llama", ".internal": "ignored"})
    >>> cfg.model
    'llama'
    >>> ".internal" in cfg
    False
    """

    def __getattr__(self, name):
        return self[name]

    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def __init__(self, dct):
        iter = filter(lambda item: not item[0].startswith("."), dct.items())
        for key, value in iter:
            self[key] = value


class ConfigEnvironment:
    """Jinja2 preprocessing and YAML parsing environment for Forgather configurations.

    ``ConfigEnvironment`` wraps a :class:`~forgather.preprocess.PPEnvironment`
    (a customised Jinja2 environment) and a suite of custom YAML constructors
    that translate ``!call``, ``!singleton``, ``!factory``, ``!partial``, and
    ``!var`` tags into :class:`~forgather.latent.Node` objects.  The result of
    :meth:`load` is a :class:`Config` containing both the parsed node graph and
    the preprocessed YAML text.

    Parameters
    ----------
    searchpath : str, os.PathLike, or iterable of str/PathLike, optional
        Directories searched for templates, in priority order.  Non-existent
        directories are silently ignored.  Defaults to ``(".",)``.
    pp_environment : jinja2.Environment or None, optional
        A pre-configured Jinja2 environment to use instead of the default
        :class:`~forgather.preprocess.PPEnvironment`.  When ``None`` (default)
        a fresh ``PPEnvironment`` is created from *searchpath*.
    global_vars : dict or None, optional
        Variables injected into the Jinja2 global namespace and available in
        every template.  Merged with any variables already present in
        *pp_environment*.  Defaults to ``{}``.

    Examples
    --------
    >>> env = ConfigEnvironment(searchpath=["/path/to/templates"])
    >>> config = env.load("configs/train.yaml")
    >>> node_graph, pp_text = config.get()
    """

    def __init__(
        self,
        searchpath: Iterable[str | os.PathLike] | str | os.PathLike = tuple("."),
        pp_environment: Optional[Environment] = None,
        global_vars: Optional[Dict[str, Any]] = None,
    ):
        if global_vars is None:
            global_vars = {}
        # Convert search path to tuple, if str or os.PathLike
        if isinstance(searchpath, os.PathLike) or isinstance(searchpath, str):
            searchpath = [searchpath]
        assert isinstance(searchpath, Iterable), "searchpath must be Iterable"

        # Remove non-existent directories from searchpath
        searchpath = list(filter(lambda path: os.path.isdir(path), searchpath))

        if pp_environment is None:
            pp_environment = PPEnvironment(searchpath=searchpath)
        self.pp_environment = pp_environment
        self.pp_environment.globals |= global_vars

    def get_pp_environment(self):
        return self.pp_environment

    def get_loader(self):
        return self.pp_environment.loader

    def preprocess(
        self,
        config_path: os.PathLike | str,
        /,
        **kwargs,
    ) -> ConfigText:
        """Render a configuration template through Jinja2 and return the result.

        Locates *config_path* in the search path, renders it with the
        configured global variables plus any extra *kwargs*, and returns the
        resulting YAML text.

        Parameters
        ----------
        config_path : str or os.PathLike
            Template path relative to the search path (e.g. ``"configs/train.yaml"``).
        **kwargs
            Additional keyword arguments passed as Jinja2 template variables,
            overriding globals for this render.

        Returns
        -------
        ConfigText
            The rendered YAML text.  :class:`ConfigText` is a ``str`` subclass
            that additionally exposes :meth:`~ConfigText.with_line_numbers`.
        """
        template = self.pp_environment.get_template(str(config_path))
        return ConfigText(template.render(**kwargs))

    def preprocess_from_string(
        self,
        config: str,
        /,
        **kwargs,
    ) -> ConfigText:
        """Render a configuration template supplied as a string through Jinja2.

        Parameters
        ----------
        config : str
            Raw template source text (may contain Jinja2 directives).
        **kwargs
            Additional keyword arguments passed as Jinja2 template variables.

        Returns
        -------
        ConfigText
            The rendered YAML text.
        """
        template = self.pp_environment.from_string(config)
        return ConfigText(template.render(**kwargs))

    def load(
        self,
        config_path: os.PathLike | str,
        /,
        **kwargs,
    ) -> Config:
        """Preprocess and parse a configuration file into a node graph.

        Combines :meth:`preprocess` and :meth:`load_from_ppstring` into a
        single call.

        Parameters
        ----------
        config_path : str or os.PathLike
            Template path relative to the search path.
        **kwargs
            Additional Jinja2 template variables forwarded to :meth:`preprocess`.

        Returns
        -------
        Config
            Container holding the parsed node graph and the preprocessed YAML
            text.

        Raises
        ------
        Exception
            Any YAML or node-graph parse error, annotated with the numbered
            preprocessed source for easier debugging.
        """
        pp_config = self.preprocess(config_path, **kwargs)
        return self.load_from_ppstring(pp_config)

    def load_from_string(
        self,
        config: str,
        /,
        **kwargs,
    ) -> Config:
        """Preprocess and parse a configuration supplied as a string.

        Parameters
        ----------
        config : str
            Raw template source text.
        **kwargs
            Additional Jinja2 template variables forwarded to
            :meth:`preprocess_from_string`.

        Returns
        -------
        Config
            Container holding the parsed node graph and the preprocessed YAML
            text.
        """
        pp_config = self.preprocess_from_string(config, **kwargs)
        return self.load_from_ppstring(pp_config)

    def load_from_ppstring(self, pp_config: str) -> Config:
        """Parse an already-preprocessed YAML string into a node graph.

        Parses *pp_config* with the custom YAML constructors (``!call``,
        ``!singleton``, ``!factory``, ``!partial``, ``!var``, etc.) and
        validates the resulting graph with :meth:`~forgather.latent.Latent.check`.

        Parameters
        ----------
        pp_config : str
            Fully rendered YAML text (output of Jinja2 preprocessing).

        Returns
        -------
        Config
            Container holding the parsed node graph and *pp_config*.

        Raises
        ------
        Exception
            Any YAML parse error or node-graph validation error, annotated with
            line-numbered source text.
        """
        try:
            loaded_config = load_depth_first(pp_config, Loader=ConfigLoader)
            Latent.check(loaded_config)
        except Exception as error:
            raise add_exception_notes(error, format_line_numbers(pp_config))
        if isinstance(loaded_config, dict):
            loaded_config = ConfigDict(loaded_config)
        return Config(loaded_config, pp_config)

    def find_referenced_templates(
        self,
        template_name: os.PathLike | str,
        /,
        **kwargs,
    ) -> Iterator[Tuple[int, str, str]]:
        """Iterate over the full template inheritance hierarchy for a given template.

        Traces actual template loading at render time so that dynamic
        ``extends`` / ``include`` references (those whose targets are computed
        by Jinja2 expressions) are captured in addition to statically declared
        ones.

        Parameters
        ----------
        template_name : str or os.PathLike
            Name of the root template to analyse (relative to the search path).
        **kwargs
            Forwarded to the inner :meth:`load` call so dynamic-args
            conditional includes (e.g. ``include "trainers/" + trainer_type +
            ".yaml"``) resolve to the right files. Omit for the static-default
            view.

        Yields
        ------
        level : int
            Depth of this template in the hierarchy (0 = root).
        name : str
            Template name as it appears in the loader.
        filename : str
            Filesystem path to the template file.
        """
        # Use render-time tracing to get complete template hierarchy
        load_sequence, dependencies = self._trace_template_rendering(
            template_name, **kwargs
        )

        # Convert to the expected format with hierarchy levels
        template_levels = self._build_hierarchy_levels(load_sequence, dependencies)

        for template_name, level in template_levels:
            # Get filename for this template
            filename = next(
                (filename for name, filename in load_sequence if name == template_name),
                template_name,
            )
            yield (level, template_name, filename)

    def get_template_dependencies(self, template_name: os.PathLike | str, /, **kwargs):
        """Return raw dependency relationships for a template, suitable for graph generation.

        Parameters
        ----------
        template_name : str or os.PathLike
            Name of the root template to analyse.
        **kwargs
            Forwarded to the inner :meth:`load` so the trace reflects which
            templates are actually included given those Jinja variables.

        Returns
        -------
        load_sequence : list of tuple[str, str]
            Ordered list of ``(template_name, filename)`` pairs in the order
            templates were loaded during rendering.
        dependencies_dict : dict[str, set[str]]
            Mapping from each template name to the set of template names it
            directly references (via ``extends`` or ``include``).
        """
        return self._trace_template_rendering(template_name, **kwargs)

    def _trace_template_rendering(
        self, template_name: os.PathLike | str, /, **kwargs
    ) -> Tuple[List[Tuple[str, str]], Dict[str, Set[str]]]:
        """
        Trace all templates loaded during rendering using a tracing loader

        Returns: (load_sequence, dependencies)
        """
        from .preprocess import PPLoader

        # Create a tracing version of the loader
        class TracingPPLoader(PPLoader):
            def __init__(self, original_loader):
                # Copy configuration from original loader
                if hasattr(original_loader, "searchpath"):
                    super().__init__(original_loader.searchpath)
                else:
                    super().__init__([])

                # Copy existing templates
                if hasattr(original_loader, "templates"):
                    self.templates = original_loader.templates.copy()

                self.load_trace = []
                self.load_stack = []
                self.dependencies = {}
                self.is_tracing = False
                # Also track static relationships from original method
                self.static_dependencies = {}

            def get_source(self, environment, template):
                result = super().get_source(environment, template)

                if self.is_tracing:
                    source, filename, uptodate = result
                    self.load_trace.append((template, filename))

                    # Analyze the source to understand relationship types
                    static_refs = self._get_static_references(
                        source, environment, template, filename
                    )

                    # Store static relationships for this template (these are the real dependencies)
                    if static_refs:
                        self.static_dependencies[template] = set(static_refs)

                    self.load_stack.append(template)

                return result

            def _get_static_references(
                self, source, environment, template_name, filename
            ):
                """Get static template references using the original method logic"""
                try:
                    ast = environment.parse(
                        source, name=template_name, filename=filename
                    )
                    return sorted(
                        (
                            x
                            for x in meta.find_referenced_templates(ast)
                            if x is not None
                        ),
                        key=lambda a: 1 if a.endswith(".yaml") else -1,
                    )
                except:
                    return []

        # Replace loader temporarily
        original_loader = self.pp_environment.loader
        tracing_loader = TracingPPLoader(original_loader)

        try:
            self.pp_environment.loader = tracing_loader
            tracing_loader.is_tracing = True

            # Render the template to trace all dependencies. Forwarding
            # **kwargs is what lets dynamic-args-driven includes resolve
            # to the right templates (e.g. trainer_type → trainers/X.yaml).
            self.load(template_name, **kwargs)

            # Use static dependencies, but also try to infer dynamic relationships
            static_deps = tracing_loader.static_dependencies.copy()

            # Post-process to identify likely dynamic relationships
            dynamic_deps = self._identify_dynamic_relationships(
                tracing_loader.load_trace, static_deps, tracing_loader
            )

            # Merge dynamic relationships into static ones
            for parent, children in dynamic_deps.items():
                if parent not in static_deps:
                    static_deps[parent] = set()
                static_deps[parent].update(children)

            return tracing_loader.load_trace.copy(), static_deps

        finally:
            # Restore original loader
            self.pp_environment.loader = original_loader

    def _build_hierarchy_levels(
        self, load_sequence: List[Tuple[str, str]], dependencies: Dict[str, Set[str]]
    ) -> List[Tuple[str, int]]:
        """
        Build hierarchy levels preserving multiple inheritance hierarchies

        This tries to maintain the structure where templates that are included/extended
        appear at appropriate levels rather than forcing a single linear hierarchy.
        """
        if not load_sequence:
            return []

        # Build reverse dependency map (child -> parents)
        parents = {}
        for parent, children in dependencies.items():
            for child in children:
                if child not in parents:
                    parents[child] = set()
                parents[child].add(parent)

        # Find root templates (those with no parents or are the starting template)
        root_template = load_sequence[0][0]
        all_templates = {name for name, _ in load_sequence}
        roots = {root_template}

        # Also consider templates that aren't children of any other template as roots
        children_set = set()
        for children in dependencies.values():
            children_set.update(children)

        for template in all_templates:
            if template not in children_set and template != root_template:
                roots.add(template)

        # Use a more sophisticated approach that preserves hierarchy structure
        levels = []
        processed = set()

        def assign_level(template, level, visited_path=None):
            if visited_path is None:
                visited_path = set()

            if template in visited_path:  # Cycle detection
                return
            if template in processed:
                return

            processed.add(template)
            levels.append((template, level))
            visited_path.add(template)

            # Process children
            children = dependencies.get(template, set())
            for child in sorted(children):  # Sort for consistent output
                if child not in processed:
                    assign_level(child, level + 1, visited_path.copy())

            visited_path.remove(template)

        # Process templates in order they appear in load_sequence
        # This preserves the natural discovery order while building hierarchy
        remaining_templates = {name for name, _ in load_sequence}

        # Start with the main root
        if root_template in remaining_templates:
            assign_level(root_template, 0)
            remaining_templates.remove(root_template)

        # Process templates in load order, but only if they haven't been processed
        # This helps maintain the structure where includes appear at reasonable levels
        for template_name, _ in load_sequence:
            if template_name in remaining_templates:
                # Check if this template has any unprocessed parents
                template_parents = parents.get(template_name, set())
                unprocessed_parents = template_parents - processed

                if not unprocessed_parents:  # No unprocessed parents, can be a root
                    assign_level(template_name, 0)
                    remaining_templates.remove(template_name)

        # Finally, ensure all templates from load_sequence are included
        # (in case there are disconnected components)
        for template_name, _ in load_sequence:
            if template_name not in processed:
                # Find the minimum level based on parents
                min_level = 0
                if template_name in parents:
                    parent_levels = []
                    for parent in parents[template_name]:
                        parent_level = next(
                            (level for t, level in levels if t == parent), -1
                        )
                        if parent_level >= 0:
                            parent_levels.append(parent_level)
                    if parent_levels:
                        min_level = max(parent_levels) + 1

                levels.append((template_name, min_level))
                processed.add(template_name)

        return levels

    def _identify_dynamic_relationships(
        self, load_sequence, static_deps, tracing_loader
    ):
        """
        Identify dynamic template relationships using a simple heuristic

        Look for specific known patterns like 'tiny.trainer_config' followed by 'trainers/*'
        """
        dynamic_deps = {}

        # Simple heuristic: look for known dynamic patterns
        for i, (template_name, filename) in enumerate(load_sequence):
            # Look for tiny.trainer_config followed by trainers/ templates
            if template_name == "tiny.trainer_config":
                # Look at the next few templates
                for j in range(i + 1, min(i + 3, len(load_sequence))):
                    next_template, _ = load_sequence[j]
                    if next_template.startswith("trainers/"):
                        # This is likely the dynamic resolution
                        dynamic_deps[template_name] = {next_template}
                        break

        return dynamic_deps

    def _has_dynamic_extends(self, source):
        """Check if template has dynamic extends/include syntax"""
        import re

        dynamic_pattern = re.compile(
            r"--\s*(?:extends|include)\s+([a-zA-Z_][a-zA-Z0-9_]*(?:\.[a-zA-Z_][a-zA-Z0-9_]*)*)(?:\s|$)",
            re.MULTILINE,
        )
        return bool(dynamic_pattern.search(source))

    def _looks_like_dynamic_target(self, template_name):
        """Check if template name looks like a likely dynamic resolution target"""
        return (
            template_name.startswith("trainers/")
            or template_name.startswith("models/")
            or template_name.startswith("datasets/")
            or template_name.startswith("callbacks/")
            or "trainer" in template_name.lower()
            or "model" in template_name.lower()
        )
