import getpass
import logging
import os
import re
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import yaml
from jinja2 import FileSystemLoader, StrictUndefined, Undefined
from jinja2.ext import Extension
from jinja2.sandbox import SandboxedEnvironment
from platformdirs import (
    site_config_dir,
    site_data_dir,
    user_cache_dir,
    user_config_dir,
    user_data_dir,
)

from .utils import format_line_numbers

logger = logging.getLogger(__name__)


def forgather_config_dir():
    return user_config_dir("forgather", getpass.getuser())


def forgather_home_dir():
    return str(Path.home() / ".forgather")


_HARDWARE_YAML = "hardware.yaml"
_GPU_FLOPS_YAML = Path(__file__).with_name("gpu_flops.yaml")

# Loaded lazily by _get_gpu_flops_table().
_gpu_flops_table: list[tuple[list[str], float]] | None = None


def _get_gpu_flops_table() -> list[tuple[list[str], float]]:
    """Load and cache the GPU FLOPS reference table from gpu_flops.yaml."""
    global _gpu_flops_table
    if _gpu_flops_table is None:
        entries = yaml.safe_load(_GPU_FLOPS_YAML.read_text())
        _gpu_flops_table = [
            (entry["match"], entry["tflops"] * 1e12) for entry in entries
        ]
    return _gpu_flops_table


def _match_gpu_flops(device_name: str) -> float | None:
    """Match a CUDA device name against the reference table."""
    for patterns, flops in _get_gpu_flops_table():
        if all(p in device_name for p in patterns):
            return flops
    return None


def _detect_gpu_flops() -> tuple[str | None, float | None]:
    """Detect the current GPU and return (device_name, peak_flops)."""
    try:
        import torch

        if not torch.cuda.is_available():
            return None, None
        device_name = torch.cuda.get_device_name(0)
        flops = _match_gpu_flops(device_name)
        return device_name, flops
    except (ImportError, RuntimeError):
        return None, None


def get_peak_hardware_flops() -> float | None:
    """
    Return the peak BF16 FLOP/s (FP32 accumulation) for a single GPU.

    Resolution order:

    1. Read ``~/.forgather/hardware.yaml`` -- if it contains a
       ``peak_hardware_flops`` value, return it immediately.
    2. Detect the current GPU via ``torch.cuda.get_device_name(0)`` and
       look it up in the built-in reference table.
    3. If found, write the result to ``~/.forgather/hardware.yaml`` so
       subsequent calls (and other sessions) skip detection.

    Returns ``None`` when no GPU is available or the GPU is not in the
    reference table. In that case the user can create
    ``~/.forgather/hardware.yaml`` manually -- see
    ``docs/trainers/training-performance-metrics.md`` for values.
    """
    home_dir = Path(forgather_home_dir())
    hardware_file = home_dir / _HARDWARE_YAML

    # 1. Check for a cached / manually specified value.
    if hardware_file.is_file():
        try:
            data = yaml.safe_load(hardware_file.read_text())
            if isinstance(data, dict) and "peak_hardware_flops" in data:
                value = data["peak_hardware_flops"]
                if value is not None:
                    return float(value)
        except (yaml.YAMLError, ValueError, OSError):
            pass

    # 2. Auto-detect from current GPU.
    device_name, flops = _detect_gpu_flops()

    # 3. Cache the result for future sessions.
    if flops is not None:
        try:
            home_dir.mkdir(parents=True, exist_ok=True)
            hardware_file.write_text(
                f"# Forgather hardware settings\n"
                f"# Auto-detected from: {device_name}\n"
                f"# Edit this value to override, or delete the file to re-detect.\n"
                f"# See docs/trainers/training-performance-metrics.md for reference values.\n"
                f"peak_hardware_flops: {flops}\n"
            )
        except OSError as exc:
            logger.debug("Could not write %s: %s", hardware_file, exc)

    return flops


def split_templates(template, name=None):
    split_on = re.compile(r"\n#\s*-{3,}\s*([\w./]+)\s*-{3,}\n")
    prev = 0
    for match in split_on.finditer(template):
        yield (name, template[prev : match.start()])
        name = match.group(1)
        prev = match.end()
    yield (name, template[prev:])


def open_block(block_name, options):
    s = ""
    if options == "!":
        s += r"{% filter trim %}"
    s += r"{% block " + block_name + r" %}"
    return s


def close_block(block_name, options):
    s = r"{% endblock " + block_name + r" +%}"
    if "!" in options:
        s += r"{% endfilter %}"
    return s


def preprocess_toml_blocks(source):
    """
    Preprocess TOML-style block syntax into Jinja2 blocks.

    Converts:
        [block_name]
        content

        [another_block]
        more content

    To:
        {% block block_name %}
        content
        {% endblock block_name %}
        {% block another_block %}
        more content
        {% endblock another_block %}
    """
    lines = source.split("\n")
    result_lines = []
    block_stack = []  # Stack of (block_name, indentation_level) tuples
    block_pattern = re.compile(r"^(\s*)\[(\w+)([!])*\]\s*$")

    def get_indentation_level(line):
        """Get the number of leading whitespace characters."""
        return len(line) - len(line.lstrip())

    for line in lines:
        block_match = block_pattern.match(line)

        if block_match:
            leading_whitespace = block_match.group(1)
            block_name = block_match.group(2)
            block_options = block_match.group(3)
            if block_options is None:
                block_options = ""
            current_indent = len(leading_whitespace)

            # Close blocks that are at the same level or deeper than the current block
            while block_stack and block_stack[-1][1] >= current_indent:
                closed_block_name, _, close_options = block_stack.pop()
                if len(result_lines[-1]):
                    result_lines[-1] += r"{{'\n'}}"
                result_lines[-1] += close_block(closed_block_name, close_options)

            result_lines.append(open_block(block_name, block_options))
            block_stack.append((block_name, current_indent, block_options))
        else:
            # Regular line - just add it
            result_lines.append(line)

    # Close all remaining blocks in reverse order
    while block_stack:
        closed_block_name, _, close_options = block_stack.pop()
        if len(result_lines[-1]):
            result_lines[-1] += r"{{'\n'}}"
        result_lines[-1] += close_block(closed_block_name, close_options)

    return "\n".join(result_lines)


def preprocess(source):
    def pp_generate(source):
        newline_re = re.compile(r"(\n|\r\n|\r)")
        full_line_comment = re.compile(r"\s*##(.*)")
        line_comment = re.compile(r"(.*)\s+#{2,}.*")
        line_statement = re.compile(r"\s*(--|<<|>>|==|=>)\s(.*)")

        for line in newline_re.split(source)[::2]:
            # Completely delete full comment lines
            if (re_match := full_line_comment.fullmatch(line)) is not None:
                if LineStatementProcessor.preserve_line_numbers:
                    line = r"{# " + re_match[1] + r" #}"
                else:
                    continue
            # Delete training comments to end-of-line
            elif (re_match := line_comment.fullmatch(line)) is not None:
                line = re_match[1]

            if (re_match := line_statement.fullmatch(line)) is not None:
                match re_match[1]:
                    case "--":
                        line = r"{% " + re_match[2] + r" %}"
                    case "<<":
                        line = r"{%- " + re_match[2] + r" %}"
                    case ">>":
                        line = r"{% " + re_match[2] + r" -%}"
                    case "==":
                        line = r"{{ " + re_match[2] + r" }}"
                    case "=>":
                        line = r"{{ " + re_match[2] + r"|trim('\n')}}"
                    case _:
                        pass
            yield line

    source = "\n".join(pp_generate(source))
    source = preprocess_toml_blocks(source)
    return source


class PPLoader(FileSystemLoader):
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.templates = {}

    def get_source(self, environment, template):
        if (template_info := self.templates.get(template)) is None:
            source, filename, uptodate = super().get_source(environment, template)
            main_template = next(iter := split_templates(source))
            for sub in iter:
                self.templates[sub[0]] = (sub[1], filename, uptodate)
            return main_template[1], filename, uptodate
        return template_info

    def add_template(self, name, data):
        self.templates[name] = (data, "", lambda: False)


def _raise_on_undefined(*args):
    for arg in args:
        if isinstance(arg, Undefined):
            raise arg._fail_with_undefined_error("Undefined argument")


def _os_path_join(*args):
    """
    Calling os.path.join() on jinja's Undefined types results in a difficult to
    diagnose exepction, as it does not show what is undefined or provide context.

    This is just a wrapper for better diagnostics on undefined paths.
    """
    _raise_on_undefined(*args)
    return os.path.join(*args)


def _os_path_normpath(*args):
    _raise_on_undefined(*args)
    return os.path.normpath(*args)


class LineStatementProcessor(Extension):
    # Jinja comments add a blank line to the output. This makes it very difficult to
    # format things the way you would like /and/ use Jinja comments. The preprocessor
    # strips Jinja comments, which would otherwise add empty lines, from the input
    # to keep things tidy.
    #
    # If anything goes wrong, this can make things very difficult to debug, as Jinja
    # will report the wrong line numbers.
    #
    # The work-around is this flag. When set to True, line-comments are not removed,
    # thus preserving line-numbers. Once things are working, you can turn it off, as
    # to preserve formatting.
    preserve_line_numbers: bool = True

    # The preprocessor converts the syntactic-sugrar coated line-statements into
    # regular Jinja statements. Asside from stipping comments, as mentioned above,
    # this /should/ have little impact on the outcome, but it can be useful to
    # see the translated input for diagnostics. Setting this to True results in
    # uber-verbose output, where every pre-processed input template is dumped to
    # stdout for analysis.
    pp_verbose: bool = False

    # When set to a list (typically by capture_pp()), every preprocessed
    # template (name, source) is appended to it in the order they are loaded.
    # This is the structured / programmatic counterpart to pp_verbose, used by
    # the webui debug panel to display the preprocessed form of each template
    # consumed by a config.
    pp_capture: list | None = None

    def preprocess(self, source, name, filename=None):
        source = preprocess(source)
        if LineStatementProcessor.pp_verbose:
            label = name or ""
            print(f"{' '+label+' ':-^80}")
            print(format_line_numbers(source))
        if LineStatementProcessor.pp_capture is not None:
            LineStatementProcessor.pp_capture.append((name, source))
        return source


@contextmanager
def debug_pp(debug):
    """
    Context manager, which enabled pp debug and restores it to the previous state on exit
    """
    preserve_line_numbers = LineStatementProcessor.preserve_line_numbers
    pp_verbose = LineStatementProcessor.pp_verbose
    try:
        LineStatementProcessor.preserve_line_numbers = debug
        LineStatementProcessor.pp_verbose = debug
        yield
    finally:
        LineStatementProcessor.preserve_line_numbers = preserve_line_numbers
        LineStatementProcessor.pp_verbose = pp_verbose


@contextmanager
def capture_pp():
    """
    Context manager that collects every preprocessed template into a list of
    (name, preprocessed_source) tuples in load order.

    Yields the list itself, so the caller can read it after the with-block exits:

        with capture_pp() as trace:
            env.preprocess("configs/foo.yaml")
        # trace -> [(name, src), ...]

    Also enables preserve_line_numbers for the duration so line numbers in the
    collected sources align with the originals (matching pp_debug behaviour).
    """
    prev_capture = LineStatementProcessor.pp_capture
    prev_preserve = LineStatementProcessor.preserve_line_numbers
    trace: list[tuple[str, str]] = []
    LineStatementProcessor.pp_capture = trace
    LineStatementProcessor.preserve_line_numbers = True
    try:
        yield trace
    finally:
        LineStatementProcessor.pp_capture = prev_capture
        LineStatementProcessor.preserve_line_numbers = prev_preserve


class ConfigDiagnostic(Exception):
    """Base class for structured config-pipeline errors.

    Subclasses share the same field shape so the webui can render them
    uniformly: a header (``template`` + ``lineno``), a message body, and an
    optional line-numbered source excerpt around the offending line.

    Attributes
    ----------
    kind : str
        Discriminator the frontend uses to label / style the error
        (``preprocess_error``, ``yaml_error``, ``code_error``). Defined
        per-subclass as a ``ClassVar``.
    template_name : str | None
        The template / config path where the error originated.
    lineno : int | None
        1-based line number when known.
    message : str
        Short error message (no traceback).
    source_context : str | None
        Line-numbered excerpt of the source around ``lineno``, ready to drop
        into a ``<pre>`` block.
    original : BaseException
        The underlying exception, also kept as ``__cause__``.
    """

    kind: str = "config_error"

    def __init__(
        self,
        message: str,
        *,
        template_name: str | None = None,
        lineno: int | None = None,
        source_context: str | None = None,
        original: BaseException | None = None,
    ):
        super().__init__(message)
        self.template_name = template_name
        self.lineno = lineno
        self.message = message
        self.source_context = source_context
        self.original = original

    def __str__(self) -> str:
        loc = self.template_name or "<config>"
        if self.lineno is not None:
            loc = f"{loc}:{self.lineno}"
        if self.source_context:
            return f"{loc}: {self.message}\n{self.source_context}"
        return f"{loc}: {self.message}"


class PreprocessError(ConfigDiagnostic):
    """Jinja2 ``TemplateError`` raised during config preprocessing."""

    kind = "preprocess_error"


def toyaml(obj, default_value=Undefined):
    """
    Convert Python data structures to YAML-compatible string representation.

    Handles the following Python types:
    - dict: Converted to YAML mapping
    - list: Converted to YAML sequence
    - str: Converted to YAML string (quoted if necessary)
    - None: Converted to YAML null
    - int: Converted to YAML integer
    - float: Converted to YAML float with explicit type tag
    - bool: Converted to YAML boolean

    This filter ensures that Python values injected into Jinja2 templates
    will render as valid YAML that can be parsed correctly.

    Examples:
        {{ my_dict | to_yaml }}
        {{ my_list | to_yaml }}
        {{ my_value | to_yaml }}

    Parameters
    ----------
    obj : any
        Python object to convert to YAML string
    default_value : any, optional
        An optional default, if the variable Jinja is undefined. This will raise if a default
        is not provided and the variable is undefined.

    Returns
    -------
    str
        YAML-compatible string representation
    """
    if isinstance(obj, Undefined):
        if default_value is Undefined:
            return obj.fail(
                "A required variable was undefined and no default was provided."
            )
        obj = default_value

    # Short circuit, if simple type -- it's faster
    if isinstance(obj, int | bool):
        return str(obj)
    elif obj is None:
        return "null"

    # Use PyYAML's safe dump with sensible defaults for inline rendering
    # default_flow_style=True produces inline JSON-like style which is more compact
    # and easier to embed in templates
    yaml_str = yaml.safe_dump(
        obj,
        default_flow_style=True,
        allow_unicode=True,
        width=float("inf"),  # Don't wrap lines
    ).strip()

    # Remove the trailing newline and '...' document end marker if present
    if yaml_str.endswith("\n..."):
        yaml_str = yaml_str[:-4]
    elif yaml_str.endswith("..."):
        yaml_str = yaml_str[:-3]

    return yaml_str


def utcnow():
    return datetime.now(timezone.utc)


def raise_helper(msg: str):
    raise ValueError(msg)


def assert_helper(condition: bool, msg: str):
    assert condition, msg


class PPEnvironment(SandboxedEnvironment):
    TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
    FILE_TIME_FORMAT = "%Y-%m-%dT%H-%M-%S"

    default_globals = {
        "isotime": lambda: datetime.now().isoformat(timespec="seconds"),
        "utcisotime": lambda: utcnow().isoformat(timespec="seconds"),
        # An ISO 8601-like time, suitable for use in file names
        "filetime": lambda: datetime.now().strftime(PPEnvironment.FILE_TIME_FORMAT),
        "utcfiletime": lambda: utcnow().strftime(PPEnvironment.FILE_TIME_FORMAT),
        "now": datetime.now,
        "utcnow": utcnow,
        "joinpath": _os_path_join,
        "normpath": _os_path_normpath,
        "abspath": os.path.abspath,
        "relpath": os.path.relpath,
        "dirname": os.path.dirname,
        "basename": os.path.basename,
        "splitext": os.path.splitext,
        "getenv": os.environ.get,
        "raise": raise_helper,
        "assert": assert_helper,
        "repr": repr,
        # Given a module file path, return the module name
        "modname_from_path": lambda path: os.path.splitext(os.path.basename(path))[0],
        "user_home_dir": lambda: os.path.expanduser("~"),
        "getcwd": os.getcwd,
        "forgather_config_dir": forgather_config_dir,
        "forgather_home_dir": forgather_home_dir,
        "get_peak_hardware_flops": get_peak_hardware_flops,
        # https://pypi.org/project/platformdirs/
        "user_data_dir": user_data_dir,
        "user_cache_dir": user_cache_dir,
        "user_config_dir": user_config_dir,
        "site_data_dir": site_data_dir,
        "site_config_dir": site_config_dir,
    }

    default_filters = {
        # Convert Python data structures to YAML-compatible strings
        "toyaml": toyaml,
    }

    def __init__(
        self,
        *args,
        loader=None,
        searchpath=None,
        extensions=None,
        auto_reload=True,
        trim_blocks=True,
        undefined=StrictUndefined,
        **kwargs,
    ):
        if extensions is None:
            extensions = []
        assert (
            loader is not None or searchpath is not None
        ), "Either a loader or searchpath must be specified"
        if loader is None:
            loader = PPLoader(searchpath)

        extensions.insert(0, LineStatementProcessor)
        super().__init__(
            *args,
            loader=loader,
            extensions=extensions,
            auto_reload=auto_reload,
            trim_blocks=trim_blocks,
            undefined=undefined,
            **kwargs,
        )

        self.globals |= self.default_globals
        self.filters |= self.default_filters
