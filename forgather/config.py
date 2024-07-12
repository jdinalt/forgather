from typing import (
    Any, Callable, Iterable, Optional, List, Type,
    Set, Dict, Container, Sequence, NamedTuple, Set
)
from collections.abc import Sequence, Mapping
from types import NoneType
from enum import Enum
import os
import sys
import yaml
import datetime
import time
import platform
import re
from importlib.metadata import version

from pprint import pformat
import jinja2
from jinja2.ext import Extension
from jinja2.sandbox import SandboxedEnvironment
from . import Latent
from .dynamic import normalize_import_spec
from .yaml_utils import (
    callable_constructor,
    load_depth_first,
    tuple_constructor
)
from .utils import DiagnosticEnum

class LoadMethod(DiagnosticEnum):
    """
    FROM_STRING: The 'config' is interpreted as as string.
    FROM_FILE: The 'config' is interpreted as an exact file path.
    FROM_FILE_SEARCH: The 'config' is interpreted as a file path which
        may be relative to any of the provided paths in 'search_paths'
    """
    FROM_STRING = "from_string"
    FROM_FILE = "from_file"
    FROM_FILE_SEARCH = "from_file_search"

class InvalidLoadMethod(Exception):
    pass

class WhitelistError(Exception):
    pass

DEFAULT_LOAD_METHOD = LoadMethod.FROM_FILE_SEARCH
TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

def format_line_numbers(text: str) -> str:
    return ''.join(map(lambda x: f"{x[0]+1:>6}: {x[1]}\n", enumerate(text.split('\n'))))

class ConfigText(str):
    def with_line_numbers(self, show_line_numbers=True):
        return format_line_numbers(self) if show_line_numbers else self

def fconfig(obj, sort_items=True, indent_level=2):
    """
    Recursively pretty-format a configuration
    """
    def indent_block(block):
        indent = ' ' * indent_level
        s = ''.join(map(lambda s: indent + s + '\n', block.split('\n')))
        return s[:-1]
    if isinstance(obj, ConfigText):
        return obj.with_line_numbers()
    elif isinstance(obj, LoadConfigOutput) or isinstance(obj, MaterializedOutput):
        return str(obj)
    elif isinstance(obj, str):
        return f"'{obj}'"
    elif isinstance(obj, Set):
        return fconfig(tuple(obj), sort_items, indent_level)
    elif isinstance(obj, Mapping):
        s = ''
        items = obj.items()
        if sort_items:
            items = dict(sorted(items)).items()
        for key, value in items:
            fmt_value = fconfig(value, sort_items, indent_level)
            if '\n' in fmt_value or len(fmt_value) > 80:
                s += f"{key}:\n" + indent_block(fmt_value) + '\n'
            else:
                s += f"{key}: " + fmt_value + '\n'
        return s[:-1]
    elif isinstance(obj, Sequence):
        s = ''
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
            s += '- ' + fconfig(value, sort_items, indent_level) + '\n'
        return s[:-1]
    elif isinstance(obj, Latent):
        s = f"Latent '{obj.constructor}'"
        if len(obj.args): 
            s += '\n' + indent_block(fconfig(obj.args, sort_items, indent_level))
        if len(obj.kwargs): 
            s += '\n' + indent_block(fconfig(obj.kwargs, sort_items, indent_level))
        return s
    else:
        return pformat(obj)

def pconfig(obj, /, *args, **kwargs):
    """
    Print a config
    """
    print(fconfig(obj, *args, **kwargs))

class LoadConfigOutput(NamedTuple):
    config: Any
    pp_config: ConfigText | NoneType

    def __str__(self):
        return fconfig(self._asdict())

class MaterializedOutput(NamedTuple):
    config: Any
    pp_config: ConfigText | NoneType
    whitelist: Container
    pp_whitelist: ConfigText | NoneType

    def __str__(self):
        return fconfig(self._asdict())

def __add_exception_notes(error: Exception, *args):
    """
    Add a note to an exception.

    Python 3.11 supports adding additional details to exceptions via notes.
    Unfortunately, our present target is Python 3.10...

    We will try to add a note to an exception anyway via adding it to something
    in the exception which will be printed with the exception.

    This is most definitely a hack, but it's the 'least-bad-thing,' as the line
    numbers reported by Yaml will likely correspond to a pre-processed Yaml file,
    making the line-numbers relatively useless, unless you have access to the 
    preprocessed data, with line-numbers.

    Ideally, at some later point, when we upgrade to a newer version of Python, this
    should be replaced with the native method of adding notes.
    """
    note = ''
    for arg in args:
        note += '\n\n' + str(arg)

    # Some Yaml exceptions have a 'note' attribute, which will be printed with
    # the exception. If it has this, use it.
    if hasattr(error, 'note'):
        if isinstance(error, str):
            error.note += note
        else:
            error.note = note
        return error
    # Try appending the note to the first str in the exception's args.
    else:
        # Not all exception have a string in the first arg. For example, "yaml.MarkedYAMLError"
        # We try to be generic here by find the first string in args, but it's impossible to 
        # know if there could be side effects -- probably not, but committed to Python 3.10 at present,
        # so "add_not() is not an option, so do the least-bad-thing.
        error_args = list(error.args)
        for i, arg in enumerate(error_args):
            if isinstance(arg, str):
                error_args[i] += '\n' + note
                error.args = tuple(error_args)
                return error
    # Fallback to generic exception, which at least should chain them.
    raise Exception(note)

def _os_path_join(*args):
    """
    Calling os.path.join() on jinja's Undefined types results in a difficult to
    diagnose exepction, as it does not show what is undefined or provide context.

    This is just a wrapper for better diagnostics on undefined paths.
    """
    for arg in args:
        if isinstance(arg, jinja2.Undefined):
            raise arg._fail_with_undefined_error("Undefined path passed to path_join")
    return os.path.join(*args)

class CustomLoader(jinja2.FileSystemLoader):
    """
    Custom Jinja2 loader which can split a file template into multiple named sub-templates

    ```
    Main template
    ##--- my_template_label ---
    Sub template, named 'my_template_label'
    ```

    As Jinja2 does not allow extending imported templates, this makes it easier to extend multiple
    parent templates in the same file. Witout this, each must be in a seperate file.

    This just makes things a little easier to work with.
    """
    split_on = re.compile(r'\n#\s*-{3,}\s*([\w./]+)\s*-{3,}\n')
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.templates = {}

    def get_source(self, environment, template_name):
        if (template_info := self.templates.get(template_name)) is None:
            source, filename, uptodate = super().get_source(environment, template_name)
            main_template = next(iter := self.split_templates(source))
            for sub in iter:
                self.templates[sub[0]] = (sub[1], uptodate)
            return main_template[1], filename, uptodate
        return template_info[0], template_name, template_info[1]

    def split_templates(self, template):
        prev = 0
        name = None
        for match in self.split_on.finditer(template):
            yield (name, template[prev:match.start()])
            name = match.group(1)
            prev = match.end()
        yield(name, template[prev:])

class LineStatementProcessor(Extension):
    newline_re = re.compile(r'(\n|\r\n|\r)')
    full_line_comment = re.compile(r'\s*##(.*)')
    line_comment = re.compile(r'(.*)\s+#{2,}.*')
    line_statement = re.compile(r'\s*--\s(.*)')
    lstrip_line_statement = re.compile(r'\s*<<\s(.*)')
    rstrip_line_statement = re.compile(r'\s*>>\s(.*)')
    line_print = re.compile(r'\s*==\s(.*)')
    rstrip_line_print = re.compile(r'\s*=>\s(.*)')

    # Jinja comments add a blank line to the output. This makes it very difficult to
    # format things the way you would like /and/ use Jinja comments. The preprocessor
    # strips Jinja comments, which would otherwise add empty lines, from the input
    # to keep things tidy.
    #
    # If anything goes wrong, this can make things very difficult to debug, as Jinja
    # will report the wrong line numbers.
    # 
    # The work-around is thins flag. When set to True, line-comments are not removed,
    # thus preserving line-numbers. Once things are working, you can turn it off, as
    # to preserve formatting.
    preserve_line_numbers: bool = False

    # The preprocessor converts the syntactic-sugrar coated line-statements into
    # regular Jinja statements. Asside from stipping comments, as mentioned above,
    # this /should/ have little impact on the outcome, but it can be useful to 
    # see the translated input for diagnostics. Setting this to True results in
    # uber-verbose output, where every pre-processed input template is dumped to
    # stdout for analysis.
    pp_verbose: bool = False

    def preprocess(self, source, name, filename=None):
        source = "\n".join(self.pp_generate(source))
        if LineStatementProcessor.pp_verbose:
            print(f"{' '+name+' ':-^80}")
            print(format_line_numbers(source))
        return source

    def pp_generate(self, source):
        for line in self.newline_re.split(source)[::2]:
            # Completely delete full comment lines
            if (match := self.full_line_comment.fullmatch(line)) is not None:
                if LineStatementProcessor.preserve_line_numbers:
                    line = r'{# ' + match[1] + r' #}'
                else:
                    continue
            # Delete training comments to end-of-line
            elif (match := self.line_comment.fullmatch(line)) is not None:
                line = match[1]
            # Convert line-stantement to regular syntax line-statements
            if (match := self.line_statement.fullmatch(line)) is not None:
                line = r'{% ' + match[1] + r' %}'
            # Left trim
            elif (match := self.lstrip_line_statement.fullmatch(line)) is not None:
                line = r'{%- ' + match[1] + r' %}'
            # Right trim
            elif (match := self.rstrip_line_statement.fullmatch(line)) is not None:
                line = r'{% ' + match[1] + r' -%}'
            # Line print
            elif (match := self.line_print.fullmatch(line)) is not None:
                line = r'{{ ' + match[1] + r' }}'
            elif (match := self.rstrip_line_print.fullmatch(line)) is not None:
                line = r'{{ ' + match[1] + r' -}}'
            yield line

version_info = { 
    "python": sys.version
} | {
        lib: version(lib) for lib in (
            "torch",
            "transformers",
            "accelerate",
        )
}

def preprocess_config(
    config:  os.PathLike | str, *,
    search_path: str | List[str] = '.',
    load_method: LoadMethod = DEFAULT_LOAD_METHOD,
    preserve_line_numbers=False,
    pp_verbose=False,
    **kwargs,
) -> str:
    """
    Preprocess a configuration file with Jinja2
    """
    load_method = LoadMethod(load_method)

    # If given a file path, implicilty prepend the file's directory to the search path.
    if load_method != LoadMethod.FROM_STRING:
        config_dir = os.path.dirname(config)
        assert config_dir is not None
        if isinstance(search_path, str):
            search_path = [config_dir, search_path]
        else:
            search_path = [config_dir, *search_path]
    
    environment = SandboxedEnvironment(
        loader = CustomLoader(searchpath=search_path),
        extensions = [ LineStatementProcessor ],
        auto_reload=True,
        undefined=jinja2.StrictUndefined,
        trim_blocks=True,
    )
    
    environment.globals["now"] = datetime.datetime.now().strftime(TIME_FORMAT)
    environment.globals["utcnow"] = datetime.datetime.utcnow().strftime(TIME_FORMAT)
    environment.globals["time_ns"] = str(time.time_ns())
    environment.globals["path_join"] = _os_path_join
    environment.globals["versions"] = version

    # Set class-level debug flags
    # There's not easy to to pass arguments...
    LineStatementProcessor.preserve_line_numbers = preserve_line_numbers
    LineStatementProcessor.pp_verbose = pp_verbose
    
    match load_method:
        case LoadMethod.FROM_STRING:
            jinja_template = environment.from_string(config)
        case LoadMethod.FROM_FILE_SEARCH:
            jinja_template = environment.get_template(os.path.basename(config))
        case LoadMethod.FROM_FILE:
            with open(config, 'r') as f:
                config_text = f.read()
            jinja_template = environment.from_string(config_text)
        case _:
            raise InvalidLoadMethod
    
    # Some configs expect the caller to set these values. We set them here to
    # allow them to run, but the callers should set these, if the configs expect them.
    default_args = dict(
        script_args='N/A',
        world_size=1,
        rank=0,
        local_rank=0,
        hostname=platform.node(),
        versions=version_info,
    )
    return ConfigText(jinja_template.render(**(default_args | kwargs)))

def __load(config, *, preprocess, search_path, load_method, pp_kwargs) -> LoadConfigOutput:
    pp_config = config
    if preprocess:
        # Let Jinja handle it
        config = pp_config = preprocess_config(
            config, search_path=search_path, load_method=load_method, **pp_kwargs)
    else:
        match load_method:
            case LoadMethod.FROM_STRING:
                config = ConfigText(pp_config)
            case LoadMethod.FROM_FILE:
                with open(pp_config, 'r') as file:
                    config = ConfigText(file.read())
            case LoadMethod.FROM_FILE_SEARCH:
                # TODO: Implement search path
                raise NotImplementedError("FROM_FILE_SEARCH has not been implemeted for this function yet. ")
            case _:
                raise InvalidLoadMethod()
    return LoadConfigOutput(config, pp_config)

def load_config(
    config: os.PathLike | str, *,
    preprocess: bool = True,
    search_path: str | List[str] = '.',
    load_method: LoadMethod = DEFAULT_LOAD_METHOD,
    **kwargs,
) -> LoadConfigOutput:
    """
    Load Jinja/Yaml configuration

    Input may be text (str) or a file-path to process.
    Optionally preprocess the input with Jinja2, then process with Yaml.

    See 'preprocess_config()' for preprocessing details.
    
    The Yaml SafeLoader is modified to have to additional tag constructors:
        !object:<import-spec>
            Construct Latent object from import-spec. The object is not
            immediatly instantiated until Latent.materialize(object, whitelist) is called
            on it, where 'whitelist' specifies which constructors are allowed.
        !getitem: [ object, key ]
            Equivalent to Python 'obect[key],' but does is latent, like !object and
            is not resolved until materialized.

    Args:
        config: A configuration string or file-path
        preprocess: Preprocess the config. with Jinja2
        search_path: List of search paths to find file.
        load_method:
            from_string: 'config' is a string
            from_file: 'config' is a path
    Returns:
        
    """
    load_method = LoadMethod(load_method)
    load_out = __load(
        config, preprocess=preprocess, search_path=search_path,
        load_method=load_method, pp_kwargs=kwargs)
    yaml.SafeLoader.add_multi_constructor("!callable", callable_constructor)
    yaml.SafeLoader.add_constructor("!tuple", tuple_constructor)
    try:
        loaded_config = load_depth_first(load_out.config, Loader=yaml.SafeLoader)
    # The line numbers in the Yaml error will only make sense with the preproessed data
    # Be helpful and prepend the context to the exception.
    except Exception as error:
        raise __add_exception_notes(error, load_out)
        
    # Filter hidden keys when dictionary type
    if isinstance(loaded_config, dict):
        loaded_config = dict(filter(lambda item: not item[0].startswith('.'), loaded_config.items()))
    return LoadConfigOutput(loaded_config, load_out.pp_config)

def load_whitelist_as_set(
    config: os.PathLike | str, *,
    preprocess: bool = True,
    search_path: str | List[str] = '.',
    load_method: LoadMethod = DEFAULT_LOAD_METHOD
) -> Set[str]:
    """
    Load a whitelist configuration from a file or string

    This is essentially just load_config, but it normalizes the paths in the whitelist
    and converts the list to a set, to improve search performance.
    """
    load_method = LoadMethod(load_method)
    load_out = load_config(
        config, preprocess=preprocess, search_path=search_path, load_method=load_method)
    # Convert paths to normalized form
    if not isinstance(load_out.config, Sequence):
        raise WhitelistError("The whitelist must resolve to a Sequence\n" + str(load_out))
    try:
        whitelist = set(map(normalize_import_spec, load_out.config))
    except Exception as error:
        raise __add_exception_notes(error, load_out)
    return LoadConfigOutput(whitelist, load_out.pp_config)

def enumerate_whitelist_exceptions(config: Any, whitelist: Container = set()):
    """
    Print all import-specs not matching the whitelist

    This is useful for determining which imports a configuration uses and/or which are 
    missing from the whitelist.

    Without specifying the whitelist, this tests against the empty-set, which shows
    all unique import-specs.
    """
    assert isinstance(whitelist, Container), "The whitelist must be a 'Container'; use load_whitelist_as_set()"
    assert not isinstance(config, str), "The input config should be a parsed config; use load_config()"
    print(fconfig(Latent.validate_whitelist(config, whitelist)))

def materialize_config(
    config: Any,
    whitelist: Container | os.PathLike | str = None,
    preprocess: bool = True,
    search_path: str | List[str] = '.',
    load_method: LoadMethod=DEFAULT_LOAD_METHOD,
    pp_kwargs: Dict[str, Any] = {},
    kwargs: Dict[str, Any] = {},
) -> MaterializedOutput:
    """
    Materialize the Latent objects in the configuration

    This can take an instantiated, but Latent, configuration; a preprocessed configuration string,
    or a path to a configuraiton file.
    """
    load_method = LoadMethod(load_method)
    if isinstance(whitelist, str):
        whitelist_out = load_whitelist_as_set(
            whitelist, preprocess=preprocess, search_path=search_path, load_method=load_method)
    else:
        whitelist_out = LoadConfigOutput(whitelist, None)

    if isinstance(config, str) or isinstance(config, os.PathLike):
        config_out = load_config(
            config, preprocess=preprocess, search_path=search_path, load_method=load_method, **pp_kwargs)
    else:
        config_out = LoadConfigOutput(config, None)
    
    if whitelist_out.config is not None:
        invalid_set = Latent.validate_whitelist(config_out.config, whitelist_out.config)
        if len(invalid_set):
            raise WhitelistError(
                'The following import-specs were not found in the whitelist:\n\n'
                + pformat(invalid_set) + "\n\n"
                + str(MaterializedOutput(
                    None, config_out.pp_config, whitelist_out.config, whitelist_out.pp_config))
            )
    try:
        kwargs['pp_config'] = lambda: config_out.pp_config
        materialized_config = Latent.materialize(config_out.config, **kwargs)
    except Exception as error:
        raise __add_exception_notes(error, MaterializedOutput(
            config, config_out.pp_config, whitelist_out.config, whitelist_out.pp_config))
    return MaterializedOutput(materialized_config, config_out.pp_config, whitelist_out.config, whitelist_out.pp_config)