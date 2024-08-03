# Forgather Syntax Reference

Forgather defines a domain-specific language for the dynamic construciton of Python objects using a combination of Jinja2, YAML, and a few extensions.

This guide will focus on the extensions to these languages. For details on YAML and Jinja2, see:

- [Jinja2 Template Designer Documentation](https://jinja.palletsprojects.com/en/3.1.x/templates/)
- [YAML 1.1](https://yaml.org/spec/1.1/)
- [PyYAML Documentation](https://pyyaml.org/wiki/PyYAMLDocumentation)

## Jinja2 Extensions
---
### The Preprocessor

There is a custom Jinja2 preprocessor which implemnts an extended version of Jinja2's [Line Statements](https://jinja.palletsprojects.com/en/3.1.x/templates/#line-statements). These are implemented via regex substition, where the match is converted to normal Jinja syntax.


- \#\# : Line Comment
- \-\- : Line Statement
- << : Line Statement w/ left-trim
- \>> : Line Statement w/ right-trim
- == : Print Command
- '=>' : Print Command w/ right-trim

Example Input:

```jinja2
## If 'do_loop' is True, then output a list of numbers.
-- if do_loop:
    -- for i in range(how_many): ## Loop 'how_many' times.
        == '- ' + i|string
    -- endfor
<< endif
```

Is translated to:

```jinja2
{# If 'do_loop' is True, then output a list of numbers. #}
{% if do_loop: %}
{% for i in range(how_many): %}
{{ '- ' + i|string }}
{% endfor %}
{%- endif %}
```

Output, when passed: do_loop=True, how_many=3
```yaml
- 0
- 1
- 2

```


Normal Jinja2 syntax works just fine too. I just find that the normal syntax is visually difficult to parse (without syntax-highlighting) and is awkward to type.

More Formally

```python
line_comment = r'(.*)\s+#{2,}.*'
line_statement = r'\s*(--|<<|>>|==|=>)\s(.*)'

Substitutions:
{
    '--': r"{% " + re_match[2] + r" %}
    '<<': r"{%- " + re_match[2] + r" %}"
    '>>': r"{% " + re_match[2] + r" -%}"
    '==': r"{{ " + re_match[2] + r" }}"
    '=>': r"{{ " + re_match[2] + r" -}}"
}
```

---
### Jinja2 Globals

A number of globals have been introduced to the Jinja2 environment to assist with pre-processing.

- isotime() : Returns ISO formatted local-time, with 1-second resolution ("%Y-%m-%dT%H:%M:%S")
- utcisotime() : As with isotime(), but UTC time.
- filetime(): Generates a local-time string suitable to be concatenated with a file-name. ("%Y-%m-%dT%H-%M-%S")
- utcfiletime() : As filetime(), but in UTC time.
- now() : Get datetime.datetime.now()
- utcnow() : Get datetime.datetime.utcnow()
- joinpath(*names) : Join a list of file-path segments via os.path.join()
- normpath(path) : Normalize a file path; os.path.normpath()
- abspath(path) : Convert path to absolute path; os.path.abspath()
- relpath(path) : Convert a path to a relative path; os.path.relpath()
- repr(obj) : Get Python representation of object; repr()
- modname_from_path(module_name) : Given a module file path, return the module name
- user_home_dir() : Return absolute path of user's home directory  
- getcwd() : Get the current working directory
- forgather_config_dir() : Get the platform-specific config directory for Forgather.

The following functions from https://pypi.org/project/platformdirs/
- user_data_dir()
- user_cache_dir()
- user_config_dir()
- site_data_dir()
- site_config_dir()

---
### Custom File Loader

A custom loader, derived from the FileSystemLoader, is defined. This loader has a syntax for splitting a single loaded template file into multiple sub-templates.

The primary use-case for this syntax is [template inheritance](https://jinja.palletsprojects.com/en/3.1.x/templates/#template-inheritance), which disallows multiple-inheritance. If you inherit from a template and include a template which is derived from another, Jinja2 does not allow you to direclty override blocks from the included template. You can get around this by creating another template, which overrides the desired blocks, and is included by the top-level template.

Normally, this would require creating another template file, but who needs that!? That's much more difficult to work with.

```jinja2
## This is the main template
-- extends 'base_template.jinja'

## Override block 'foo' from 'base_template.jinja'
-- block foo
    -- include 'foo.bar' ## Include the sub-template
-- endblock


##--------------------- foo.bar ---------------------
## This is a sub-template named 'foo.bar'
-- extends 'some_other_base_template.jinja'

## Override block 'bar' from 'some_other_base_template.jinja'
-- block bar
    ## ... stuff
-- endblock
```

More formally, the syntax for splitting a document is:

```python
split_on = r"\n#\s*-{3,}\s*([\w./]+)\s*-{3,}\n"
```

Note: You can't split a template defined via a Python string, as this bypasses the Loader; only file templates may be split like this.

---
## YAML

### Dot-Name Elision
YAML does not have a way of defining an object, without also constructing it. This can be inconvienient, as it may not be known ahead of time where the first use of an object will be and YAML requires that the defition occur at this point.

To work around this, if the root-node is a mapping, we delete all keys containing strings starting with a dot. Once the object has been defined, YAML does not care if we delete the original definition/instance. My convention is to use ".define", but any name, starting with a dot, will work.

By convention, the primary output object of such a mapping is named "main"

```yaml
# Define points
.define: &pt1 { x: 0, y: 0 }
.define: &pt2 { x: 5, y: 0 }
.define: &pt3 { x: 0, y: 5 }

main:
    # A list of lines, each defined by a pair of points.
    - [ *pt1, *pt2 ]
    - [ *pt2, *pt3 ]
    - [ *pt3, *pt1 ]
```

Constructed graph...

```python
Latent.materialize(graph)

{'main': [[{'x': 0, 'y': 0}, {'x': 5, 'y': 0}],
          [{'x': 5, 'y': 0}, {'x': 0, 'y': 5}],
          [{'x': 0, 'y': 5}, {'x': 0, 'y': 0}]]}
```

While not apparent from the representation, the points in the lines are not copies, they are all references to the original three points from the definition. There are only three point objects present in the graph!

---
### YAML Types

Of the standard YAML 1.1 types, only those which can be implicilty (without specifying the tag) are supported

YAML 1.1 Tag : Python Type / Examples
- !!null : None
    - null
- !!bool : bool
    - True
    - False
- !!int : int
    - 2
    - -6
- !!float : float
    - 2.0
    - 1.2e-4
- !!str : str
    - "Hello"
    -  world
- !!seq : list
    - \[ 1, 2, 3 \] 
- !!map : dict
    - { x: 1, y: 12 }

The following standard types are presently unsupported:
- !!binary
- !!timestamp
- !!omap, !!pairs
- !!set -- TODO: Implement me!

---
Complex types are instead supported through Forgather specific tags:

#### !tuple

Construct a Python tuple from a YAML sequence

```yaml
!tuple [ 1, 2, 3 ]
```

---
#### !var

This declares a variable, which can be substituted when the graph is constructed.

```yaml
point:
    x: !var "x" # Define a variable named 'x'
    y: !var # Define a variable named 'y' with a default value of 16
        name: y
        default: 16
```

---
#### !singleton

This is a callable object with only a single instance; any aliases refers to the same object instance.

```yaml
# Construct three random ints, all having the same value.
- &random_int !singleton:random:randrange [ 1000 ]
- *random_int
- *random_int
```

Constructed...
```python
Latent.materialize(graph)

[247, 247, 247]
```

The "SingletonNode" will generally be your 'go-to' for constructing objects, as the symantics mirror what is expected for YAML anchors and aliases.

However, there are a few exceptions...

---
#### !factory

This is a callable object which instantiates a new instance everywhere it appears in the graph.

```yaml
# Construct three random ints, all (probably) having different values.
- &random_int !factory:random:randrange [ 1000 ]
- *random_int
- *random_int
```

Constructed...
```python
Latent.materialize(graph)

[99, 366, 116]
```

---
#### !lambda

This returns the entire sub-graph as a callable. This can be used when a callable needs to be passed as an argument.

```yaml
# Compute powers-of-two from a list, returning a list.
!singleton:list
    - !singleton:map
        # The generated object is equivalent to: "lambda arg0: pow(arg0, 2)"
        - !lambda:pow [ !var "arg0", 2 ]
        - [ 1, 2, 3, 4 ]
```

Constructed...
```python
Latent.materialize(graph)

[1, 4, 9, 16]
```

Note that any positional arguments are implicity converted to the variables named \[ 'arg0', 'arg1', 'arg2', ... \]

---
### CallableNodes


SingletonNode, FactoryNode, and FactoryNode are all instances of the abstract-base-class "CallableNode." A CallableNode can call any Python function, including class constructors. As Python differentiates between positional args and kwargs, making use of both requires the following syntax:

```yaml
!singleton:random:sample
    args:
        - ['red', 'blue']
        - 5
    kwargs:
        counts: [4, 2]
```

Generally speaking, you can omit the explict 'args' and 'kwargs' names, as long as the syntax is unambigous.

```yaml
- !singleton:torch:tensor
    - 2
    - 2
- !singleton:random.binomialvariate { n: 1, p: 0.5 }
```

---
#### CallableNode Tag Syntax

The part of the YAML tag after the first ':' provides the information required to locate and import the requested Callable.

In the simplest case, a [built-in](https://docs.python.org/3/library/functions.html) Python callable just needs to specify the name of the built-in.

```yaml
!singleton:tuple [ 1, 2, 3 ]
```

When the Callable is defined in a module, a second ':' is used to seperate the module name from the name within the module.

```yaml
# See: https://docs.python.org/3/library/operator.html
!singleton:operator:mod [ 365, 7 ]
```

You can also dynamically import a name from a file.

```yaml
# See: https://docs.python.org/3/library/operator.html
!singleton:/path/to/my/pymodule.py:MyClass [ "foo", "bar" ]
```

When using a file-import which itself has relative imports, you will need to specify which directories to search for relative imports:

```yaml
# See: https://docs.python.org/3/library/operator.html
!singleton:/path/to/my/pymodule.py:MyClass 
    args: [ "foo", "bar" ]
    kwargs:
        submodule_searchpath:
            - "/path/to/my/"
            - "/path/to/shared/modules/"
```

By specifying multiple locations, the import system treats all of the directories in the list as a union, thus you can perform a relative import from any of these directories.