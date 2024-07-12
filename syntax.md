# Configuration Syntax

We use [Jinja2](https://jinja.palletsprojects.com/) for preprocessing and [YAML](https://pyyaml.org/wiki/PyYAMLDocumentation) for the actual configuration. I'll spare going into details, as these are well covered in the links, but it may be helpful to point out a few non-standard and non-obvious things.

## Jinja

Jinja is running in a sandboxed environment. This limits what functions and data may be accessed.

We have enabled line-statement and line comments.

```jinja2
## This is a line-comment. The next line is a line-statement.
-- set foo = 'bar'

## Line comments are shorthand for...
{# ...regular comments and line-statements are short for... #}
{%- macro foobar() %}
```
While both of these are regular Jinja features, there is not a standard prefix for either. The prefix is set when the environment is created.

Line-comments don't show up in the pre-processed configuration, while regular Yaml comments will show up.  

We have injected a number of symbols into the environment.
```
now : datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    local time
utcnow : datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    UTC time
time_ns : str(time.time_ns())
    Timestamp, integer nanoseconds
path_join(...) : os.path.join(...)
    Join path names in an os independent way.

# Defaults for the following have been set, but can be overridden by the training-script.
world_size : 1
    The number of concurrent proccesses used in distributed training.
rank: 0
    The global multiprocess rank. See torch.distributed
local_rank:
    The local multiprocess rank. See torch.distributed
script_args : 'N/A'
    The args passed to the configuration script.
hostname: platform.node()
    The hostname of the machine.
```

*White-space control*

If you need wish to strip leading and/or trailing whitespace surrounding a Jinja statement, add the '-' symbol to the start and end tokens.
```jinja2
## Strip both left and right sides
{{- foo + bar -}}
{#- strip only left #}
## Strip only right-side.
{% if foo > 12 -%}
```

*Namespaces*

By default, 'included' templates inherit the namespace of the caller, while 'imported' templates do not.

Despite what you may assume, 'include' is not quite the same as directly substituting text. If root template 'A' includes templates 'B' and 'C', the namespace of A is visible to both B and C and vice-versa. What is not obvious is that B and C will not have access to each others namespaces.

```jinja2
## Content of A.jinja
-- include 'B.jinja'
-- include 'C.jinja'
## Contents of B.jinja
-- set FOO = 1
## Contents of C.jinja
{{ FOO }}
```

This will not work, as 'FOO' will not be visible in C. To work around this, you can declare a namespace in A, which will be visible to both B and C. Any changes by one will be visible to the other.

```jinja2
## Content of A.jinja
-- set experiment = namespace()
-- include 'B.jinja'
-- include 'C.jinja'
## Contents of B.jinja
-- set experiment.FOO = 1
## Contents of C.jinja
{{ experiment.FOO }}
```
This will work.

Another namespace oddity is that you can 'set' a variable on a namespace, but you can't directly define a macro in a namespace.

```jinja2
## Contents of A.jinja
-- set experiment = namespace()
-- set experiment.foobar = "foobar"
-- include 'B'jinja'
{{ experimment.some_macro() }}
## Contents of B.jinja
-- macro experimment.some_macro()
{{ experiment.foobar }}
-- endmacro
```
This will not work, as you can't declare a macro in a namespace... but you can assign one!

```jinja2
## Contents of A.jinja
-- set experiment = namespace()
-- set experiment.foobar = "foobar"
-- include 'B'jinja'
{{ experimment.some_macro() }}
## Contents of B.jinja
-- macro B__some_macro()
{{ experiment.foobar }}
-- endmacro
-- set experimment.some_macro = B__some_macro
```
Seems like a bug, but at least it can be worked around.

The preprocessor accepts a list of search directories for template files, with the directories being searched in the order they are defined. The directory of your configuration file is implicilty added to the head of this list, thus you can specify file paths to the configuration file as 'includes' or 'imports.'

## YAML

We are using the PyYAML, with all of its warts. This follows the Yaml 1.1 specification... mostly.

The loader is the 'yaml.SafeLoader,' which prohibits constructing arbitrary Python objects.

There is one custom tag present: '!callable'

!callable constructs 'Latent' objects. That is to say, a Laent holds the definition for a Python Callable, but does not immediatly load any module code or construct anything.

The Latent objects must be explicilty 'materialized,' at which point the safety of the types are checked, the symbols are resolved, and only then, is anything actually constructed.

- !callable must always be followed by either a list or a mapping.
- Passing either an empty list [] or empty mapping {} invokes the callable without args.
- If passed a list, all of the args are passed as positional arguments.
- If passed a mapping, all of the the args are passed as key-word arguments.
- Exception: If you need to pass both key-word and positional arguments, the presence of the key 'kwargs' and 'args' allows for both to be passed.


```yaml
- !callable:datetime:now [] # No arguments
- !callable:datetime:now {} # No arguments
- !callable:torch:tensor [ 1, 2, 3 ] # Only positional arguments
- !callable:torch:tensor # Alternate syntax for only positional
    - 1
    - 2
    - 3
- !callable:foo:Point { x: 1, y: -5 } # Only keywords
- !callable:foo:Point # Alternate syntax for only keywords
    x: 1
    y: -5
- !callable:torch:tensor { args: [4, 5, 6], kwargs: { requires_grad: True }} # Both positional and keyword
- !callable:torch:tensor # Alternate
    args:
        - 4
        - 5
        - 6
    kwargs:
        requires_grad: True
```

The value after '!callable:' is the import-spec:

```bnf
import-spec ::=  <module-name-or-path> ':' <symbol-name>
symbol-name ::= <symbol> [ '.' <symbol-name> ]
module-name-or-path ::= <module-path> | <module-name>
module-path ::= [ <os-path> ] <module-name> '.py'
```
You can get a locally defined symbol, for testing, like this:

```python
config_def = """
!callable:__main__:squared [2]
"""

def squared(x):
    return x**2

materialize_config(config_def, load_method='from_string').config
> 4
```

One feature in Yaml, which you may not be familiar with, are 'anchors' and 'aliases.' An anchor defines a symbolic reference which may be used again later in the definition.
```yaml
point: &my_anchor
    x: 1
    y: -5

line:
    start: *my_anchor
    end: *my_anchor 
```
Note that using an alias does not create a copy, it refers to the same instance!

We also make use of Yaml's esoteric 'merge' operator, '<<:'
```yaml
defauts: &defaults
    x: 1
    y: 2

point:
    <<: *defaults
    x: 5
    z: 10
# The above results in:
point:
    x: 5
    y: 2
    z: 10
```

Yaml does not allow you abstract anchor definitions; an anchor must refer to an actual object in the graph.

This is rather annoying when you just want to define something which is only for later use. To address this, any keys at the root level which start with '.' will be pruned before the configuration is returned. By convention, I give name all of these keys '.define,' as Yaml has no issues with using the same key more than once, but anything starting with '.' will do.

```yaml
.define: &x !callable:torch.tensor [ 2 ]
.define: &y !callable:torch.tensor [ 3 ]
sum: !callable:torch.add [ *x, *y ]
```
After loading, only the key for 'sum' will be in the dictionary.

*File Extensions*

You may use whater extension you like. All of the pre-defined templates end in '.yaml' This produces the best syntax highlighting compromise between Jinja2 and YAML syntax.