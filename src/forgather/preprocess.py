import os
import datetime
import time
import re

from jinja2 import FileSystemLoader, StrictUndefined, Undefined
from jinja2.ext import Extension
from jinja2.sandbox import SandboxedEnvironment

from .utils import format_line_numbers


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

    split_on = re.compile(r"\n#\s*-{3,}\s*([\w./]+)\s*-{3,}\n")

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.templates = {}

    def get_source(self, environment, template_name):
        if (template_info := self.templates.get(template_name)) is None:
            source, filename, uptodate = super().get_source(environment, template_name)
            main_template = next(iter := self.split_templates(source))
            for sub in iter:
                self.templates[sub[0]] = (sub[1], filename, uptodate)
            return main_template[1], filename, uptodate
        return template_info

    def split_templates(self, template):
        prev = 0
        name = None
        for match in self.split_on.finditer(template):
            yield (name, template[prev : match.start()])
            name = match.group(1)
            prev = match.end()
        yield (name, template[prev:])


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
    newline_re = re.compile(r"(\n|\r\n|\r)")
    full_line_comment = re.compile(r"\s*##(.*)")
    line_comment = re.compile(r"(.*)\s+#{2,}.*")

    line_statement = re.compile(r"\s*(--|<<|>>|==|=>)\s(.*)")
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
        source = "\n".join(self.pp_generate(source))
        if LineStatementProcessor.pp_verbose:
            print(f"{' '+name+' ':-^80}")
            print(format_line_numbers(source))
        return source

    def pp_generate(self, source):
        for line in self.newline_re.split(source)[::2]:
            # Completely delete full comment lines
            if (re_match := self.full_line_comment.fullmatch(line)) is not None:
                if LineStatementProcessor.preserve_line_numbers:
                    line = r"{# " + r" #}"
                else:
                    continue
            # Delete training comments to end-of-line
            elif (re_match := self.line_comment.fullmatch(line)) is not None:
                line = re_match[1]

            if (re_match := self.line_statement.fullmatch(line)) is not None:
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


class PPEnvironment(SandboxedEnvironment):
    TIME_FORMAT = "%Y-%m-%d %H:%M:%S"

    def __init__(
        self,
        *args,
        loader=None,
        searchpath=None,
        extensions=[],
        auto_reload=True,
        trim_blocks=True,
        undefined=StrictUndefined,
        **kwargs,
    ):
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
        self.globals["now"] = lambda: datetime.datetime.now().strftime(self.TIME_FORMAT)
        self.globals["utcnow"] = lambda: datetime.datetime.utcnow().strftime(
            self.TIME_FORMAT
        )
        self.globals["time_ns"] = lambda: str(time.time_ns())
        self.globals["path_join"] = _os_path_join
        self.globals["normpath"] = _os_path_normpath
