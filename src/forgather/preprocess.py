import os
import datetime
import time
import re
import getpass

from jinja2 import FileSystemLoader, StrictUndefined, Undefined
from jinja2.ext import Extension
from jinja2.sandbox import SandboxedEnvironment
from platformdirs import (
    user_data_dir,
    user_cache_dir,
    user_config_dir,
    site_data_dir,
    site_config_dir,
)

from .utils import format_line_numbers


def forgather_config_dir():
    return user_config_dir("forgather", getpass.getuser())


def split_templates(template, name=None):
    split_on = re.compile(r"\n#\s*-{3,}\s*([\w./]+)\s*-{3,}\n")
    prev = 0
    for match in split_on.finditer(template):
        yield (name, template[prev : match.start()])
        name = match.group(1)
        prev = match.end()
    yield (name, template[prev:])


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
                        line = r"{{ " + re_match[2] + r" -}}"
                    case _:
                        pass
            yield line

    return "\n".join(pp_generate(source))


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

    def get_source(self, environment, template_name):
        if (template_info := self.templates.get(template_name)) is None:
            source, filename, uptodate = super().get_source(environment, template_name)
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
    # The work-around is thins flag. When set to True, line-comments are not removed,
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

    def preprocess(self, source, name, filename=None):
        source = preprocess(source)
        if LineStatementProcessor.pp_verbose:
            print(f"{' '+name+' ':-^80}")
            print(format_line_numbers(source))
        return source


class PPEnvironment(SandboxedEnvironment):
    TIME_FORMAT = "%Y-%m-%dT%H:%M:%S"
    FILE_TIME_FORMAT = "%Y-%m-%dT%H-%M-%S"

    default_globals = {
        "isotime": lambda: datetime.datetime.now().isoformat(timespec="seconds"),
        "utcisotime": lambda: datetime.datetime.utcnow().isoformat(timespec="seconds"),
        # An ISO 8601-like time, suitable for use in file names
        "filetime": lambda: datetime.datetime.now().strftime(
            PPEnvironment.FILE_TIME_FORMAT
        ),
        "utcfiletime": lambda: datetime.datetime.utcnow().strftime(
            PPEnvironment.FILE_TIME_FORMAT
        ),
        "now": datetime.datetime.now,
        "utcnow": datetime.datetime.utcnow,
        "joinpath": _os_path_join,
        "normpath": _os_path_normpath,
        "abspath": os.path.abspath,
        "relpath": os.path.relpath,
        "dirname": os.path.dirname,
        "basename": os.path.basename,
        "splitext": os.path.splitext,
        "repr": repr,
        # Given a module file path, return the module name
        "modname_from_path": lambda path: os.path.splitext(os.path.basename(path))[0],
        "user_home_dir": lambda: os.path.expanduser("~"),
        "getcwd": os.getcwd,
        "forgather_config_dir": forgather_config_dir,
        # https://pypi.org/project/platformdirs/
        "user_data_dir": user_data_dir,
        "user_cache_dir": user_cache_dir,
        "user_config_dir": user_config_dir,
        "site_data_dir": site_data_dir,
        "site_config_dir": site_config_dir,
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
